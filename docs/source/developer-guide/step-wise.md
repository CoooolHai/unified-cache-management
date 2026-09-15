# Request-Level Async Load：PR #1246 与 HLA/HMA 扩展

本文说明 PR #1246（提交 `e80ff8b`）如何为 `UCMDirectConnector` 实现请求级异步 KV
Cache 加载，并说明后续提如何把同一套请求状态协议扩展到 HLA 和
HMA/FAWA Connector。文中会明确区分“原始 PR”与“后续扩展”，避免混淆

文档分为三部分：

- **展示版**：适合设计评审或汇报，突出模块边界、两阶段协议和并发收益。
- **实现版**：按代码调用顺序展开，用于开发、调试和故障定位。
- **Connector 扩展版**：说明 HLA 与 HMA/FAWA 的多 group 映射、Task 聚合和失败恢复。

## 核心结论

PR #1246 实现的是 `request-level async load`：

> 请求 A 等待外部 KV 时，暂时退出运行队列；Worker 可以继续执行请求 B、C。A 的
> KV 加载完成后，Scheduler 再恢复 A。


---

# 异步 load 概述：

## 模块总览

```{mermaid}
flowchart LR
    Client["Client"]

    subgraph Control["控制面：Scheduler Process"]
        Scheduler["vLLM Scheduler<br/>请求状态与 HBM block 分配"]
        SchedulerConnector["Scheduler Connector<br/>外存查询与 Load Plan"]
    end

    subgraph Execution["执行面：Worker Process"]
        Worker["Model Worker<br/>执行其他可运行请求"]
        WorkerConnector["Worker Connector<br/>提交并轮询 Load Task"]
    end

    Store["UCM Store<br/>DRAM / SSD / Remote Storage"]

    Client -->|"请求"| Scheduler
    Scheduler -->|"查询外存命中"| SchedulerConnector
    SchedulerConnector -->|"external tokens + async 标志"| Scheduler
    Scheduler -->|"Connector metadata"| Worker
    Worker -->|"start_load_kv"| WorkerConnector
    WorkerConnector -->|"submit / check / wait"| Store
    WorkerConnector -.->|"finished_recving"| Scheduler
    Worker -->|"推理结果"| Client
```

### 模块职责

| 模块 | 主要职责 | 不负责什么 |
| --- | --- | --- |
| Scheduler | 请求状态机、HBM block 分配、请求恢复 | 不直接执行外存 IO |
| Scheduler Connector | 查询外部命中、生成 UCM block 到 HBM block 的加载计划 | 不持有 Worker Store Task |
| Worker Connector | 提交 Store Task、非阻塞轮询、报告完成或失败 | 不决定请求何时恢复运行 |
| UCM Store | 执行外存到 HBM 的数据传输 | 不管理 vLLM 请求状态 |



### 特点：

- **异步发生在哪里：** `submit_load()` 返回 Task 后不立即 `wait_load()`；Task 留到后续
  step 轮询。
- **谁决定请求暂停和恢复：** Scheduler；Connector 只返回 `load_async` 和
  `finished_recving`。
- **HBM block 什么时候分配：** 提交 load 前分配，否则 Worker 不知道外存数据的目标地址。



## 同步与异步对比


```{mermaid}
flowchart LR
    subgraph Phase1["阶段一：Discover & Dispatch"]
        A["查询 HBM 与外存"] --> B["返回 external_hit_tokens"]
        B --> C["分配目标 HBM blocks"]
        C --> D["建立 UCM → HBM 映射"]
        D --> E["提交异步 Store Task"]
    end

    subgraph Waiting["等待窗口"]
        F["A: WAITING_FOR_REMOTE_KVS"]
        G["B / C: 正常 forward"]
    end

    subgraph Phase2["阶段二：Complete & Resume"]
        H["轮询 Task"] --> I["finished_recving"]
        I --> J["注册已加载 blocks"]
        J --> K["恢复请求 A"]
    end

    E --> F
    E --> G
    F --> H
    G -.-> H
```

```{mermaid}
flowchart TB
    subgraph Sync["同步模式"]
        S1["请求 A 命中外存"] --> S2["submit_load(A)"]
        S2 --> S3["wait_load(A)"]
        S3 --> S4["forward(A, B, C)"]
    end

    subgraph Async["异步模式"]
        A1["请求 A 命中外存"] --> A2["submit_load(A)"]
        A2 --> A3["A 暂停等待"]
        A2 --> A4["forward(B, C)"]
        A3 --> A5["poll load(A)"]
        A4 -.-> A5
        A5 --> A6["恢复并 forward(A)"]
    end
```

### 特点

- 减少一个慢外存请求对同机其他请求造成的 head-of-line blocking。
- 同时存在其他 runnable requests，且外存 load 延迟明显高于
  Scheduler step 间隔时,收益最大。只有 A 一个请求时，A 仍需等待自己的 KV 完成。

## 状态转移

```{mermaid}
stateDiagram-v2
    [*] --> WAITING

    WAITING --> RUNNING: 无异步外存加载
    WAITING --> WAITING_FOR_REMOTE_KVS: 外存命中且 load_async=true

    WAITING_FOR_REMOTE_KVS --> WAITING_FOR_REMOTE_KVS: Task 未完成
    WAITING_FOR_REMOTE_KVS --> WAITING: Task 成功
    WAITING_FOR_REMOTE_KVS --> WAITING: Task 失败，转入重算恢复

    WAITING --> RUNNING: 再次获得调度预算
    RUNNING --> FINISHED: 推理结束
    FINISHED --> [*]
```


# 具体实现

## 阶段一：查询 HBM 与外部缓存

入口：

```python
get_num_new_matched_tokens(
    request,
    num_computed_tokens,
) -> tuple[int, bool]
```

```{mermaid}
flowchart TD
    G["计算 external_hit_blocks"]
    G --> H["external_hit_tokens = blocks × block_size"]

    H --> I{"external_hit_tokens > 0?"}
    I -->|"否"| J["返回 0, false"]
    I -->|"是"| K{"use_request_async_load?"}
    K -->|"否"| L["返回 tokens, false"]
    K -->|"是"| M["保存 RequestMeta"]
    M --> N["返回 tokens, true"]
```
如果第二个返回值为true，则进入了异步load， Scheduler 应先分配目标 blocks，再让请求
  进入 `WAITING_FOR_REMOTE_KVS`，而不是立即 forward。



## 阶段二：分配 HBM 并建立物理映射

入口：

```python
update_state_after_alloc(request, blocks, num_external_tokens)
```

```{mermaid}
flowchart TD
   F["调用 update_state_after_alloc"]

    F --> G["读取第一个 KV group 的 block IDs"]
    G --> H["external_start = hbm_hit_block_num"]
    H --> I["external_end = start + external blocks"]
    I --> J["切片 UCM block IDs"]
    I --> K["切片 vLLM block IDs"]

    J --> L["校验 block 数量"]
    K --> L
    L --> M["构造 load_async=true 的 RequestDispatchMeta"]
    M --> N["保存到 pending async dispatches"]
    N --> O["记录 request_id，防止重复 load"]
```


## 阶段三：下发 load

异步请求进入 `WAITING_FOR_REMOTE_KVS` 后，不属于本轮 scheduled requests，但它的 load
计划仍必须发送给 Worker。

```{mermaid}
flowchart TD
    A["A 进入 WAITING_FOR_REMOTE_KVS"] --> B["A 不参加本轮 forward"]

    A --> H["update_state_after_alloc 保存 pending dispatch"]
    H --> I["build_connector_meta"]
    I --> J["合并 pending async dispatches"]
    J --> K["Worker 收到 A 的 load plan"]
```


## 阶段四：Worker 提交 Task，不阻塞 forward

入口：

```python
start_load_kv()
```

```{mermaid}
flowchart TD
    A["Worker Connector 收到 metadata"] --> B["读取 load block IDs"]
    B --> C["解析目标 HBM 地址"]
    C --> D["submit_load()"]
    D --> E["获得 Store Task"]

    E --> F{"load_async?"}
    F -->|"否"| G["wait_load(Task)"]
    G --> H["完成后进入 forward"]

    F -->|"是"| I{"已有相同 request pending task?"}
    I -->|"是"| J["忽略重复 metadata"]
    I -->|"否"| K["保存 PendingLoadTask"]
    K --> L["start_load_kv 立即返回"]
    L --> M["Worker forward 其他请求"]
```

### 关键问题与答案

- **有没有创建额外 Python 线程？** 没有。异步能力来自 Store 的 Task API；Connector 只是
  延迟等待。
- **为什么同步和异步任务共用 `start_load_kv()`？** `RequestDispatchMeta.load_async` 是每个
  request 的属性，同一 metadata 中理论上可以同时包含两类任务。
- **为什么需要 Worker 侧重复检查？** Scheduler metadata 可能因重试或版本行为重复到达，
  Worker 必须保证 Store Task 至多提交一次。
- **异步任务的 load bytes 是否在原始 PR 中完整统计？** 没有。原始同步统计路径不等待
  async task，完成轮询路径也没有等价的完整 timing/bytes 观测，这是后续可观测性工作。

## 阶段五：轮询完成并恢复请求

```{mermaid}
sequenceDiagram
    autonumber

    participant Scheduler
    participant Worker
    participant Connector
    participant Store

    loop 每个执行 step 结束
        Worker->>Connector: get_finished()
        Connector->>Store: check(task)

        alt Task 未完成
            Store-->>Connector: false
            Connector-->>Worker: finished_recving=None
            Worker-->>Scheduler: 请求继续等待
        else Task 已完成
            Store-->>Connector: true
            Connector->>Store: wait_load(task)
            Store-->>Connector: 成功或延迟错误
            Connector-->>Worker: finished_recving={request_id}
            Worker-->>Scheduler: KVConnectorOutput
        end
    end

    Scheduler->>Scheduler: cache_blocks(request)
    Scheduler->>Scheduler: 更新 num_computed_tokens
    Scheduler->>Scheduler: WAITING_FOR_REMOTE_KVS → WAITING
    Scheduler->>Worker: 调度剩余 token
```


## 阶段六：失败恢复

```{mermaid}
flowchart TD
    A["异步 Store Task"] --> B{"submit / check / wait 异常?"}

    B -->|"否"| C["正常完成"]
    C --> D["finished_recving 加入 request_id"]

    B -->|"是"| E["记录错误"]
    E --> F["mark_failed(request_id)"]
    F --> G["记录 invalid HBM block IDs"]
    G --> H["仍加入 finished_recving"]

    D --> I["Scheduler 退出远端等待状态"]
    H --> I
    I --> J{"存在 invalid blocks?"}

    J -->|"否"| K["登记全部 loaded prefix"]
    J -->|"是"| L["截断到第一个失败 block"]
    L --> M{"还有有效前缀?"}
    M -->|"是"| N["保留有效前缀"]
    M -->|"否"| O["释放已分配 blocks"]
    N --> P["失败部分重新计算"]
    O --> P
```

### 关键点

- **为什么失败也必须返回 `finished_recving`？** 这里的“finished”表示远端传输生命周期结束，
  不表示成功。若不返回，请求会永久卡在 `WAITING_FOR_REMOTE_KVS`。
- **如何避免使用损坏的 KV？** Worker 报告 invalid block IDs，Scheduler 将 computed tokens
  截断到第一个失败 block。
- **是否整个请求都必须重算？** 不一定；第一个失败 block 之前的有效前缀可以保留。

## 去重与状态归属

```{mermaid}
flowchart LR
    subgraph SchedulerSide["Scheduler Connector"]
        A["pending_async_load_dispatches<br/>待下发的一次性计划"]
        B["async_load_req_ids<br/>防止重复生成计划"]
    end

    subgraph WorkerSide["Worker Connector"]
        C["pending_load_tasks<br/>正在执行的 Store Task"]
        D["finished_async_load_req_ids<br/>需要立即上报完成的请求"]
    end

    A -->|"metadata"| C
    B -.->|"need_load=false"| A
    C -->|"check 完成"| D
    D -->|"finished_recving"| Scheduler["vLLM Scheduler"]
```

### 关键点


- **哪些状态是一次性的？** `_pending_async_load_dispatches` 在 metadata 构造后清空。
- **哪些状态跨多个 step？** `_async_load_req_ids` 和 `_pending_load_tasks`。
- **请求结束时需要清理什么？** `RequestMeta`、未下发 dispatch 和 Scheduler 侧的 async
  request 标记，避免 request ID 复用或状态泄漏。

## 原始 PR 的支持边界

```{mermaid}
flowchart TD
    A["use_request_async_load=true"] --> B{"具体类型是否<br/>UCMDirectConnector?"}
    B -->|"否"| C["关闭 request async load"]
    B -->|"是"| D{"满足原始约束?"}

    D --> E["单 KV cache group"]
    D --> F["PP = 1"]
    D --> G["无 CP"]
    D --> H["非 layerwise"]

    E --> I["启用"]
    F --> I
    G --> I
    H --> I
```

原始开关实现为严格类型判断：

```python
self.use_request_async_load = (
    request_async_configured
    and type(self) is UCMDirectConnector
)
```

因此继承 `UCMDirectConnector` 的 HLA、HMA、Blend、LayerWise Connector 不会自动获得支持。
PR 中对 HLA 和 Blend 的改动只是把 `RequestDispatchMeta` 子类构造改为关键字参数，以兼容
新增的 keyword-only `load_async` 字段，不代表这些 Connector 已支持异步加载。

·

---

# 后续扩展——HLA 与 HMA/FAWA

本部分描述提交 `5adbd37c00c74ea10c35471f0ee6a4943ac3cdcd` 中的修改。它没有改变
vLLM 的请求级异步协议：请求依旧经历“查询命中、分配 HBM、异步加载、等待完成、恢复调度”。
真正变化的是 Connector 内部如何把一个请求转换成多 group 的加载计划，以及如何判断这些
加载是否全部完成。

## 扩展范围与命名

```{mermaid}
flowchart TB
    Entry["UCMConnector<br/>按 KV cache layout 选择实现"]

    Entry --> Direct["UCMDirectConnector<br/>普通单 group KV"]
    Entry --> HLA["UCMHybridLinearAttentionConnector<br/>Full Attention + Mamba/KDA"]
    Entry --> HMA["UCMFAWAConnector<br/>HMA: Full Attention + Window Attention"]

    Direct --> Common["请求级异步协议<br/>WAITING_FOR_REMOTE_KVS / finished_recving"]
    HLA --> Common
    HMA --> Common

    HLA -.-> HLALW["HLA LayerWise<br/>仍不支持 request async"]
    Direct -.-> DirectLW["Direct LayerWise<br/>仍不支持 request async"]
```

HLA 与 HMA/FAWA 是两个独立选择的 Connector，不是一个 Connector 对另一个 Connector 的
rewrite：

- `UCMHybridLinearAttentionConnector` 处理 Full Attention 与线性状态缓存混合的多 group
  layout，例如 Full Attention + Mamba align/KDA。
- `UCMFAWAConnector` 是 HMA 场景在当前代码中的具体实现，处理 Full Attention（FA）与
  Window Attention（WA）混合的多 group layout。
- 两者都继承 `UCMDirectConnector` 以复用公共基础设施，但各自重写查询、block 映射、metadata
  和 Worker load 路径。



## 面向展示的扩展模块图

```{mermaid}
flowchart LR
    Config["use_request_async_load"] --> Capability{"Connector 显式声明支持?"}
    Capability -->|否| Sync["保留同步 load"]
    Capability -->|是| Lookup["查询外存命中"]

    Lookup --> Alloc["Scheduler 分配全部 KV groups"]
    Alloc --> Mapping{"Connector 专属映射"}

    Mapping -->|Direct| D["group 0<br/>UCM block ↔ HBM block"]
    Mapping -->|HLA| H["各 group 映射<br/>展平为一个 load plan"]
    Mapping -->|HMA/FAWA| F["各 group 映射<br/>拆成 FA 与 WA plans"]

    D --> One["一个 Store Task"]
    H --> One
    F --> Two["FA Task + WA Task"]

    One --> Done["Task 完成 → finished_recving"]
    Two --> Barrier["请求级完成屏障"]
    Barrier --> Done
```



### 关键点

- **HLA 能继续使用一个 pending task** HLA 先把各 group 的 UCM/HBM block 对按
  确定顺序展平，再向同一个 Store 提交一个 Task。
- HMA中：FA 与 WA 使用不同数据形状、地址提取规则和 backing
  store，一个请求会产生两个独立 Task。会有些区别




## HLA：多 group 展平后复用单 Task 生命周期

### Scheduler 侧映射

```{mermaid}
sequenceDiagram
    participant S as vLLM Scheduler
    participant H as HLA Scheduler Connector
    participant M as HLARequestMeta
    participant W as Worker Connector

    S->>H: get_num_new_matched_tokens()
    H->>M: 保存每个 group 的 UCM block IDs
    H-->>S: external_hit_tokens, load_async=true
    S->>S: 为所有 KV groups 分配 HBM blocks
    S->>H: update_state_after_alloc(blocks)
    H->>M: 保存 group_vllm_block_ids
    H->>H: _generate_hla_dispatch_meta(new_tokens=0)
    Note over H: Full Attention blocks 在前<br/>Mamba align state blocks 在后<br/>跳过 null block_id=0
    H->>H: 设置 load_async=true 并缓存 pending dispatch
    H-->>W: 下一份 connector metadata 携带展平 load plan
```

`HLARequestMeta` 新增两份逐 group 状态：

- `group_ucm_block_ids`：查询阶段得到的每个 group 的外存 key。
- `group_vllm_block_ids`：分配阶段得到的每个 group 的目标 HBM block。

`update_state_after_alloc()` 必须接收并校验全部 group，随后调用
`_generate_hla_dispatch_meta(..., new_tokens=0, need_load=True)` 生成纯加载 metadata。
`incoming_block_ids_are_full=True` 表示传入的是本次完整分配结果，而不是需要追加的增量，避免
后续恢复时重复追加 block ID。

### Worker 侧执行

```{mermaid}
flowchart TD
    A["HLARequestDispatchMeta"] --> B["按 full-attention count<br/>做 MLA/KDA rank scope"]
    B --> C["展平后的 UCM IDs + vLLM IDs"]
    C --> D["HybridLinearAttentionLayout<br/>提取各 block 目标地址"]
    D --> E["submit_load()"]
    E --> F{"load_async?"}
    F -->|否| G["本 step wait_load()"]
    F -->|是| H["PendingLoadTask[request_id]"]
    H --> I["后续 get_finished()"]
    I --> J{"check_load()"}
    J -->|未完成| I
    J -->|完成| K["wait_load() 收尾/暴露延迟错误"]
    K --> L["finished_recving"]
```

HLA 复用 DirectConnector 的 `_pending_load_tasks` 和 `_poll_pending_load_tasks()`；它的特殊性
在提交前已经被 HLA metadata 与 layout 消化。`load_full_attn_count` 保留展平列表中的分界，
用于 MLA/KDA 在不同 TP rank 上选择共享 key 或 rank-scoped key。

HLA 在提交 load 前仍执行一次 `device.synchronize()`，确保前一 step 的 Mamba state copy 不会
晚于 Store DMA 写入并覆盖新加载的数据。这个同步点是 HLA 内部的数据依赖保护，不会把异步
请求重新变成全局同步等待 Store Task。

### 关键问题与答案

- **为什么 `new_tokens=0`？** 此次 metadata 只负责加载已经命中的 prefix，不应推进
  `token_processed` 或生成 dump plan。
- **为什么必须保存所有 group 的分配结果？** HLA 的一个逻辑 prefix 同时依赖 Full
  Attention KV 与状态 group；只使用 group 0 会得到不完整状态。
- **HLA 何时算加载完成？** 展平计划对应的单个 Store Task 完成时。
- **恢复调度会不会再次 load？** `_async_load_req_ids` 阻止恢复路径重新生成 load，pending
  dispatch 也只下发一次。

## HMA/FAWA：一个请求拆成 FA 与 WA 两个 Task

### Scheduler 侧计划

```{mermaid}
flowchart TD
    A["外存命中 canonical hash blocks"] --> B["Scheduler 分配全部 KV groups"]
    B --> C["FAWARequestMeta.vllm_block_ids<br/>保存各 group 完整 block rows"]
    C --> D["_generate_dispatch_meta(new_tokens=0)"]
    D --> E["FA load rows<br/>每个外存命中 boundary"]
    D --> F["WA load rows<br/>只取最后命中 boundary 的 window tail"]
    E --> G["FAWARequestDispatchMeta"]
    F --> G
    G --> H["load_async=true<br/>进入 pending dispatch"]
```

FAWA 的 hash block 是统一的 canonical 边界，但各 KV group 的物理 block size 与窗口 tail
形状可以不同。`_slice_group_block_ids()` 负责把 canonical boundary 映射到每个 group 的
物理 block：

- FA group 为每个外存命中 boundary 选择对应的 block row。
- WA group 加载时只恢复最后一个匹配 boundary 所需的完整 window tail。
- 地址提取由 `_extract_fa_ptr()` 与 `_extract_wa_ptr()` 分别完成，支持 block 内 token offset
  以及多 block tail 展平。

同 HLA 一样，`update_state_after_alloc()` 先保存完整的 `vllm_block_ids`，再给
`_generate_dispatch_meta()` 传入逐 group 空列表。这样 generator 只切片已有分配，不会把同一批
block ID 追加两次。

### Worker 侧双任务提交

```{mermaid}
sequenceDiagram
    participant W as UCMFAWAConnector
    participant FA as FA Store
    participant WA as WA Store
    participant P as FAWAPendingLoad
    participant S as vLLM Scheduler

    W->>FA: submit_load(all external-hit keys, FA ptrs)
    FA-->>W: FA Task
    W->>WA: submit_load(last key, WA tail ptrs)
    WA-->>W: WA Task
    W->>P: 聚合 FA Task + WA Task + affected HBM IDs
    Note over W,P: start_load_kv 返回，不调用同步 wait
    loop 后续 get_finished
        W->>FA: check_load(FA Task)
        W->>WA: check_load(WA Task)
    end
    P-->>W: 所有子任务终态
    W-->>S: finished_recving(request_id)
```

`FAWAPendingLoad` 是新增的请求级聚合对象：

| 字段 | 作用 |
| --- | --- |
| `tasks` | 尚未完成的 FA/WA `FAWALoadTask` |
| `vllm_block_ids` | 任一子任务失败时需要失效的 HBM block 并集 |
| `failed` | 请求是否已有任一子任务失败 |
| `successful_bytes` | 已成功子任务的加载字节数累计 |



轮询时，已经完成的子任务会被 `wait_load()` 收尾并从 `pending.tasks` 移除；未完成的保留到下个
step。只有 `pending.tasks` 为空，request ID 才进入 `finished_recving`。

失败不会让请求立即越过屏障：

1. 标记 `pending.failed=true`。
2. 把该请求加载计划涉及的 HBM block ID 并集记录为 invalid。
3. 通过 Worker metadata 标记请求 load failed。
4. 继续排空其他已提交 Task。
5. 所有 Task 到达终态后仍报告 `finished_recving`，让 vLLM 离开
   `WAITING_FOR_REMOTE_KVS` 并进入重算/失败处理，而不是永久卡住。

指标只在整个请求没有失败时，把 FA 和 WA 的 `key_count * file_size[label]` 累加到
`load_bytes_total`。这样不会把部分成功的失败请求误计为完整 cache load 收益。

## vLLM Scheduler 的多 group 失败恢复适配

原始 PR 的 scheduler patch 使用单 group 解包：

```python
(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
```

HLA/FAWA 返回多个 group 后，这段代码会直接触发 tuple unpack 错误。扩展将两个相关位置改为：

```python
req_block_groups = self.kv_cache_manager.get_block_ids(req_id)
req_block_ids = req_block_groups[0] if req_block_groups else []
```

```{mermaid}
flowchart LR
    Failure["Worker 上报 load failed<br/>flat invalid block IDs"] --> Groups["Scheduler 取得<br/>block_groups"]
    Groups --> Anchor["选择 group 0<br/>full-attention / anchor group"]
    Anchor --> Match["检查外存命中范围内<br/>是否包含 invalid block"]
    Match --> Recover["撤销外存命中登记<br/>重算或走失败路径"]
```

这里是兼容现有“flat invalid block set”协议的最小修改：group 0 被当作请求恢复的 anchor。
它解决多 group tuple unpack 崩溃，但并没有把 Scheduler 的 invalid-block 协议升级为逐 group
结构；这是后续若要做到更精确局部恢复时需要继续演进的点。



## 三种 Connector 的实现对比

| 维度 | Direct | HLA | HMA/FAWA |
| --- | --- | --- | --- |
| KV group | 单 group | 多 group：Full Attention + Mamba/KDA | 多 group：FA + WA |
| Scheduler 保存的目标 block | group 0 | 每个 group | 每个 group |
| load plan | 一一映射 | 各 group 按语义选择后展平 | FA rows 与最后一个 WA tail |
| Worker Store Task | 1 | 1 | 通常 2：FA + WA |
| pending 状态 | `PendingLoadTask` | 复用 `PendingLoadTask` | `FAWAPendingLoad` 聚合多个子任务 |
| 完成条件 | 单 Task 终态 | 单 Task 终态 | 所有 FA/WA Task 终态 |
| rank/layout 特殊处理 | 普通 TP key scope | MLA/KDA scope、Mamba copy 顺序 | 两个 store、不同 ptr extraction |
| 失败影响 | 当前 load blocks | 展平后的 load blocks | 请求涉及的多 group block 并集 |
| LayerWise request async | 不支持 | 不支持 | 当前无对应 LayerWise 实现 |



