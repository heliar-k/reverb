# 修正计划:本地 TrajectoryWriter 解绑单一 table

> 本文档承接 `docs/numpy-embed-design.md` 的 §3.2/§3.4/§3.5,记录对"本地 writer
> 绑定单一 table"这一历史实现选择的纠正。起因见下文"问题溯源"。

## 问题溯源

设计文档 §3.2 把"本地 `TrajectoryWriter` 绑定单一 table"当作类结构折中的一部分,
§3.4(`LocalClient.sample` 默认 `emit_timesteps=False`)和 §3.5(`LocalClient` 无
`insert`/`writer`)被列为两个平行的"不得不做的变更"。

代码核实后发现这三者其实是**同一个根因的三面**:

1. 本地 writer 构造函数收 `shared_ptr<Table> table_`,worker `RunLocalWorker`
   硬编码 `table_->InsertOrAssignAsync(...)`。
2. `CreateItem(table, ...)` 的 `table` 参数在本地路径**完全不用于路由**——
   只写进 `item.set_table()` 字段,该字段本地路径从不读取;signature 校验默认
   跳过(`options.flat_signature_map = nullopt`)。
3. 绑定单一 table → `LocalClient.trajectory_writer(table=...)` 必须显式传 table
   → 无法对齐 gRPC `Client.trajectory_writer(num_keep_alive_refs)` 的无 table 签名
   → 砍掉 `insert`/`writer`(§3.5)→ 只剩多步 trajectory 写入 → 翻默认值(§3.4)。

**判断**:当时选择绑定单一 table 是偷懒。`InProcessClient` 本就持有
`flat_hash_map<string, shared_ptr<Table>> tables_`,把 writer 改成持 map、worker
按 `item.table()` 查表分发,是顺着现有结构就能做的事。省了几行 C++,付了三层
API 债。这个修正把债还掉。

## 修正目标

**交付边界(已定)**:本次修正是一个原子交付单元——`TrajectoryWriter` 解绑与
`Writer` 本地化(D3-a)**一起上,不拆分**。理由:简化用户认知(一次性对齐),
避免中间态(只解绑不补 Writer 时,`LocalClient` 有 `trajectory_writer` 但无
`insert`/`writer`,仍不对称)。blast radius 合并可接受。

- C++ 层:本地 `TrajectoryWriter` 持 `tables_` map 而非单一 `table_`,worker 按
  `item.table()` 分发。`CreateItem` 指向未知表时报错(原静默)。
- Python 层:`LocalClient.trajectory_writer`/`structured_writer` 去掉 `table` 参数,
  签名对齐 gRPC `Client`。补 `insert`/`writer`,语义对齐。
- 消解 §3.4:`_default_emit_timesteps` 统一回 `True`。
- 保留 §3.5 中唯一真实的物理约束:无 `__reduce__`(pickle),因为 `LocalClient`
  持进程内指针不可跨进程序列化。
- 更新设计文档 §3.2/§3.4/§3.5,如实反映修正后的状态。

## 不改动的部分

- `is_local_` 分支结构(§3.2 的类结构折中保留):仍是同一个 `TrajectoryWriter`
  类用 `is_local_` 分两条 worker 路径,复用全部 chunker/column/history 逻辑。
- chunker / column / flush / EndEpisode 逻辑:完全复用,已有测试覆盖。
- gRPC 路径:不动。

## 决策点

### D1: `InProcessClient::NewTrajectoryWriter` 去掉 `table` 参数(已定)

核实 gRPC `Client::NewTrajectoryWriter` 签名(`reverb/cc/client.h:133`):

```cpp
absl::Status NewTrajectoryWriter(const TrajectoryWriter::Options& options,
                                 std::unique_ptr<TrajectoryWriter>* writer);
```

**不收 table 参数**。gRPC 路径的表名校验延迟到 `CreateItem` 时,由
`ItemAndRefs::Validate` 用 `flat_signature_map`(从 `ServerInfo` 缓存里拿的
全部表 signature)查表做。

因此决策:**去掉 `table` 参数,完全对齐 gRPC 签名**。理由:

- 保留 `table` 作 early 校验会引入新的签名不对称(gRPC 不收、内嵌收),
  陷入"看起来对齐、实际契约不同"的陷阱——正是本次修正要消灭的东西。
- 对齐比 early 校验更重要;gRPC 的延迟校验本就是 proven 设计,内嵌沿用即可。

落地:

- `InProcessClient::NewTrajectoryWriter(const Options& options, ...)` 不收
  table,writer 构造时持完整 `tables_` map。
- `RunLocalWorker` 按 `item.table()` 查 `tables_` 分发,找不到报 `kNotFound`。
- signature 校验对齐 gRPC(已核实可行):`Table::signature()` 返回
  `absl::optional<SignatureProto>`(`table.h:584`),本地构造 options 时用现成的
  `internal::FlatSignatureFromSignatureProto`(`signature.h:65`,gRPC 侧同样走的
  成熟函数)把各 Table 的 signature 转成 `FlatSignatureMap` 塞进 options,~8 行:

  ```cpp
  internal::FlatSignatureMap signatures;
  for (const auto& [name, table] : tables_) {
    if (table->signature().has_value()) {
      REVERB_RETURN_IF_ERROR(internal::FlatSignatureFromSignatureProto(
          table->signature().value(), &(signatures[name])));
    }
    // signature 为 nullopt 时,signatures[name] 保持默认空(等价于无 signature,跳过校验)
  }
  options.flat_signature_map = std::move(signatures);
  ```

  `ItemAndRefs::Validate` 走 gRPC 同一路径,无边界 bug 风险。
- `NewStructuredWriter` 同步去掉 `table` 参数(内部调 `NewTrajectoryWriter`)。

此决策联动 Python 侧:`LocalClient.trajectory_writer`/`structured_writer`
去掉 `table` 参数(已在 P2 落实)。

### D2: backpressure 多表化——writer 级,对齐 gRPC(已定)

核实 gRPC 路径的 backpressure 是 **writer 级单一通道**(`reverb/cc/trajectory_writer.cc`
`RunStreamWorker` + `WriteIfNotEmpty`):

- 所有表的数据共用一个 gRPC `InsertStream`,`write_inflight_` 是 writer 级单一
  bool,`WriteIfNotEmpty` 在 write 在飞期间 `mu_.Await` 阻塞。
- 不管 item 要写哪个表,只要 stream 上有一个 write 未完成,整个 writer 就停。
  A 表的写未完成,B 表的写也得等。

因此决策:**writer 级 backpressure,任意表满则 writer 停**(原 D2 选项 (i))。
理由:与 gRPC 语义完全同构——gRPC 是 stream 级阻塞,本地是
`local_can_insert_more_` 单一 flag,两者语义等价。改动最小(沿用单一 flag,
只是 `InsertOrAssignAsync` 的表从固定变为按 item 查),单表场景行为不变。

代价:A 满 B 没满时 writer 也停,但这是可接受的(与 gRPC 一致,且一个 writer
交替写多表本就是 `insert` 多表场景,停顿不会放大)。

### D3: Python `LocalClient.insert` / `writer` —— 给 `Writer` 类加本地路径(已定,选 a)

核实 `insert` 与 `writer` 的绑定关系及 gRPC `Client` 用法:

- `Client.insert`(`client.py:460`)**硬依赖** `writer`:`insert` 实现就是
  `with self.writer(max_sequence_length=1) as w: w.append(data); w.create_item(...)`。
  `insert` 是 gRPC `Client` 高频主力 API(tests 26 处用法),内部依赖 `writer`,
  两者都是活 API。
- `Writer` 类**没有真正废弃**:仍在 `__init__.py:44` 正式导出,docstring 是
  "will eventually be deprecated"(将来某天),非已废弃。本 fork `__init__.py:24-28`
  还主动恢复了它(去掉 `NotImplementedError`)。
- `Writer` 类(`writer.h:43`,650 行)是**独立实现**,非 `TrajectoryWriter` 的封装:
  自带 `buffer_`/`chunks_`/`pending_items_`/gRPC `stream_`/
  `item_confirmation_worker_` 线程,不用 chunker/column 抽象,是更早期的同步实现。

**决策:选 (a)**——给 `Writer` 类加本地路径,实现 `LocalClient.writer`/`insert`,
与 gRPC `Client` 完全对齐。理由:用户指定走 (a),追求本地/gRPC API 严格镜像。

#### (a) 的实施:gRPC 链路调研与复用分析

**调研充分性说明**:本节是 `Writer` 本地化的设计基线,需先摸清 gRPC 链路全貌与
可复用点,再定实施步骤。避免凭"650 行独立类"的表像误判工作量。

##### gRPC 链路全貌

```
Python Client.writer()  →  C++ Writer (Append/CreateItem/Finish/WritePendingData)
                              ↓ stream_->Write(InsertStreamRequest)
                           gRPC 网络栈
                              ↓
                           ReverbServiceImpl::InsertStream  (server 侧 reactor)
                              ↓ ProcessIncomingRequest
                              ↓   SaveChunks          ← chunk 存入 reactor 的 chunks_ map
                              ↓   GetItemWithChunks   ← chunk 关联到 item
                              ↓   table->InsertOrAssignAsync(item, &can_insert, callback)
                              ↓
                           Table (真正存数据的地方)
```

Python `Writer` 是 C++ `Writer` 类的薄封装(pybind.cc:345 绑定 `Append`/
`CreateItem`/`Flush`/`Close` 直接调 C++ 方法)。`#3~#7` 是 `Writer` 类的 C++ 方法,
不是 Python 代码。

##### server 侧"远端处理"只有三件事,且**全不依赖 gRPC**

`ProcessIncomingRequest`(`reverb_service_impl.cc:210`)的全部工作:

1. **`SaveChunks`**——把请求里的 `ChunkData` 存进 reactor 的 `chunks_` map
   (`ChunkStore::Key → shared_ptr<Chunk>`),纯内存操作。
2. **`GetItemWithChunks`**——按 item 的 `flat_trajectory` 引用的 chunk key,从
   `chunks_` 找出对应 `shared_ptr<Chunk>`,构造 `Table::Item(item, chunks)`,
   纯内存操作。
3. **`table->InsertOrAssignAsync`**——`table = server_->TableByName(item.table())`,
   按表名查表,调 `InsertOrAssignAsync` 插入,`insert_completed_` callback 在插入
   完成后触发。`Table::InsertOrAssignAsync` 是 `Table` 现成方法,本地路径本来就在用。

gRPC 只是把 `InsertStreamRequest` 从客户端搬到 server,server 拆包后调 `Table`。
三件事没有一件依赖 gRPC。

##### 关键发现:本地 `Writer` 是"把 reactor 的 ProcessIncomingRequest 平移进 Writer"

C++ `Writer` 本就持有与 reactor 一一对应的成员:

| reactor 侧 | Writer 已有成员 |
| --- | --- |
| `chunks_` (reactor 成员) | `chunks_` (Writer 已有,`std::list<ChunkData>`) |
| `server_->TableByName(table_name)` | `tables_[table_name]` (本地化后新增) |
| `insert_completed_` callback | 本地确认 callback(递减 `num_items_in_flight_`) |
| `InsertStreamResponse.keys` 回写 | (不需要,直接 callback 递减) |

因此 `Writer` 本地路径几乎是把 reactor 的 `ProcessIncomingRequest` 平移进
`Writer::WritePendingData`——变量名都一一对应。

##### 实际工作量修正

原误判:"650 行独立类从零写本地化"。实际:

- **绝大部分原样复用**:`Append`/`AppendSequence`/`CreateItem`(signature 校验)/
  `Finish`(成 chunk)/`Flush`/`Close`/`ConfirmItems` 同步骨架不动。
- **只改 `WritePendingData`**:把 `stream_->Write(request)` 替换为内联的
  `SaveChunks` + `GetItemWithChunks` + `tables_[name]->InsertOrAssignAsync`。
  其中 `SaveChunks`/`GetItemWithChunks` 可从 reactor 抽成共享函数复用,或直接
  内联(Writer 的 `chunks_` 已持 `ChunkData`,`GetItemWithChunks` 逻辑极简)。
- **`stream_` 相关路径跳过**:懒创建/`WritesDone`/`stream_->Finish`/
  `ItemConfirmationWorker` 用 `is_local_` 分支跳过,确认靠 callback。
- **`Table::InsertOrAssignAsync` 现成可用**:server 侧和 `TrajectoryWriter` 本地
  路径都在用,`Writer` 本地路径直接用。

##### 确认机制:保留 `Writer` 同步骨架,callback 递减计数(方案 I,对齐原有形式)

**原则:尽量对齐 `Writer` 原有的同步形式**,不借本地化之名重写为异步。

`Writer` 是同步模型:调用线程直接 `Append`→`Finish`→`WritePendingData`,
`ConfirmItems(limit)` 在调用线程同步阻塞等 `num_items_in_flight_` 降下来。这与
`TrajectoryWriter` 的异步模型(write_queue + worker 线程)根本不同,不能套用
`TrajectoryWriter` 的 `local_can_insert_more_` flag 模式。

本地化方案(I)——保留同步骨架,只换信号源:

- **callback 递减计数**:每 insert 建 per-item `InsertCallback`,捕获 `this`,在
  表 worker 线程触发时持 `mu_` 递减 `num_items_in_flight_` 并 signal
  (`ItemConfirmationWorker` 原本用的 condvar/通知机制)。
- **`ConfirmItems(limit)`/`num_items_in_flight_` 不动**:gRPC 同步等待骨架原样复用,
  唤醒源从"gRPC Read 响应"换成"callback signal"。
- **`pending_items_` 结构调整**:原是 `std::list<PrioritizedItem>`,item 不持 callback。
  本地路径需挂 callback——或让 `pending_items_` 改持
  `shared_ptr<InsertCallback>`,或旁挂一个 callback 容器(实现时定)。
- **不需要 `weak_ptr<Writer>`**:`Writer` 是同步模型,`Close()` 会 `ConfirmItems(0)`
  等所有 in-flight 确认完才返回,析构时无遗留 callback,`this` 生命周期覆盖所有
  callback 触发。这比 `TrajectoryWriter` 异步情况简单(异步靠 `Close` drain 队列)。

##### 本地化的主要风险点(已识别)

`Writer` 本地化的风险不在"写本地插入逻辑"(那部分是平移 reactor),而在"给同步模型
引入跨线程 callback":

- callback 在表 worker 线程触发,持 `mu_` 递减 `num_items_in_flight_`;调用线程
  `ConfirmItems` 持 `mu_` 等。两个线程通过 `mu_` + condvar 协调——结构上和 gRPC
  的"worker 线程 Read 响应递减 / 调用线程等"同构,但要确保 signal 路径对齐。
- `Close()` 必须先 `ConfirmItems(0)` drain 再释放 callback,保证析构竞态不发生。
- per-item callback 的存放位置(`pending_items_` 改结构 vs 旁挂容器)需实现时定,
  影响 `WritePendingData` 的 pop 逻辑。

此风险点由 P0 的 `LocalWriterConfirmItemsEquivalence` 测试锁定。

##### `Writer` 本地化是「直接持 map」还是「先绑表再解绑」?

吸取本次 `TrajectoryWriter` 的教训(先绑单一 table 逼出三层 API 债),`Writer` 本地化
**一步到位持 `tables_` map**,不重走「先绑表再解绑」的弯路。`CreateItem` 的 `table`
参数在本地路径按 `item.table()` 路由(与 `TrajectoryWriter` 解绑后一致),不绑死。

##### writer 持 map 的生命周期方式:持拷贝(方案 P,已定)

两个类(`TrajectoryWriter`/`Writer`)解绑后都要持 `tables_` map。`tables_` 归
`InProcessClient` 所有(直接成员,非 shared)。两种持有方式:

- (P) writer 持 map **拷贝**:复制 `flat_hash_map<string, shared_ptr<Table>>`,
  每个 `shared_ptr<Table>` 引用计数 +1。
- (Q) 改 `InProcessClient::tables_` 为 `shared_ptr<map>`:零拷贝,但改成员类型,
  ripple 到所有用 `tables_` 的地方,且 `LoadLatest` 逻辑要重新验证。

**决策:P,持拷贝**。理由:

1. **checkpoint 恢复语义正确**:`LoadLatest` 原地改写 `Table` 对象内容
   (`InitializeFromCheckpoint` 重建 sampler/remover/rate_limiter,
   `InsertCheckpointItem` 灌回 item),**不替换 `shared_ptr<Table>` 指向**。
   writer 持的 shared_ptr 副本与 client 的指向同一 Table 对象,`LoadLatest` 后
   writer 自动看到恢复状态。
2. **`LoadLatest` 前提保证安全**:`LoadLatest` 要求各 table 为空
   (`REVERB_CHECK(data_.empty())`),即仅新建 server 时调用一次。有 writer 在写的
   表不空,不会与 `LoadLatest` 竞态。此约束是已有的,非 P 引入。
3. **性能可忽略**:writer 创建是 per-context(per-episode),非 per-step 热路径。
   map 拷贝 = 几个 `shared_ptr` 原子递增(典型 1-4 个表,十几纳秒),相对 writer
   构造本身的微秒/毫秒开销(make_unique、起 worker 线程、构造 chunker)是噪声。
   `RunLocalWorker` 查的是 writer 已持有的 map 副本,不是每次拷贝。
4. **ripple 最小**:不改 `InProcessClient` 成员类型,`LoadLatest`/`Load`/`GetTable`
   逻辑全不动。

##### `Writer` 与 `TrajectoryWriter` 本地化的结构性差异(保留供参考)

- `TrajectoryWriter` 是**异步**(write_queue + worker 线程),本地化用
  `local_can_insert_more_` flag + worker 里 `table_->InsertOrAssignAsync`。
- `Writer` 是**同步**(`Append`→`Finish`→`WritePendingData`→`stream_->Write` 直接调),
  无 worker 线程,backpressure 靠 `ConfirmItems(limit)` 同步阻塞等
  (`num_items_in_flight_` 递减)。本地化复用这个同步骨架,信号源换 callback。

#### 联动影响

- `InProcessClient` 需新增 `NewWriter` 方法(当前没有),构造本地 `Writer`。
- `LocalClient.writer` 直接复用,`LocalClient.insert` 上提到 `_BaseClient`(方案 α,
  靠 `self.writer` 鸭子类型分派),`LocalClient` 与 `Client` 共享单一实现,签名
  与 gRPC 完全一致。
- P1 扩展到 `Writer` 类,P0 补 `Writer` 本地路径测试。

## 工作量评估(基于 D3 调研修正)

原计划凭"`Writer` 是 650 行独立类"的表像估为"从零写一套本地化",高估。D3 调研
发现 server 侧 `ProcessIncomingRequest` 的三件事(`SaveChunks`/`GetItemWithChunks`/
`InsertOrAssignAsync`)全不依赖 gRPC,且 `Writer` 本就持有与 reactor 一一对应的成员。
修正后:

| 子任务 | 原估 | 修正后 | 理由 |
| --- | --- | --- | --- |
| `TrajectoryWriter` 解绑 | ~30 行 | ~30 行(不变) | 持 map + worker 查表分发 |
| `Writer` 本地化 | ~200+ 行(从零) | ~40-60 行 | `Append`/`CreateItem`/`Finish`/`Flush`/`Close`/`ConfirmItems` 原样复用,只改 `WritePendingData` |
| 确认机制(方案 I) | 需新设计 | ~20 行 | 保留同步骨架,callback 递减 `num_items_in_flight_`,`Close` drain 保安全。风险点见上 |
| `InProcessClient::NewWriter` | 新增 | ~10 行 | 构造本地 `Writer` 传 `tables_` |
| Python `LocalClient.writer`/`insert` | 新增 | ~20 行 | `writer` 调 `NewWriter`,`insert` 复用 gRPC 实现 |
| C++ 测试(P0) | 6 个 | 7 个 | TrajectoryWriter 3 + Writer 3 + StructuredWriter 多表 1 |
| Python 测试调整(P2) | 既有 | 既有 | 调用点去 `table=` + 新增 insert/writer 可用性 |

**总工作量**:中等。核心是两个 C++ 类的同构改动(持 map + 路由 + callback 确认),
不是两个独立大改造。`Writer` 本地化与 `TrajectoryWriter` 解绑是同构模式,但 P1
实施串行推进(A→B→C),不可并行(理由见 P1 串行约束)。

## 实施阶段

### P0 — C++ 测试先行(红)

三组测试同构:都验证"持多表 map + 按 item.table() 路由 + callback 确认"。
**先写,预期失败**(C++ 还没改)。

**TDD 红态说明(C++ 编译期红,R1)**:C++ 是静态编译,新测试调用的签名
(持 `{A,B}` map 的构造函数、去 table 的 `NewStructuredWriter`、本地 `Writer`
构造)P1 才存在。所以 P0 新增的测试 **target 会编译失败**,这是预期的红态,不是
bug。各 `cc_test` 是独立 target(`in_process_client_test`/`trajectory_writer_test`/
`writer_test` 各自独立),编译失败只阻塞该 target,不拖垮其他 target。**P0 完成时
`bazel test //reverb/cc/...` 会有数个 target ERROR,这是 TDD 红态,非回归**——P1
落地后转绿。不要误判为构建破坏。

**`TrajectoryWriter` 组**(`reverb/cc/trajectory_writer_test.cc`,新增 3 个
`LocalRoundTrip*`):

- [ ] `LocalRoundTripDispatchesToMultipleTables`:单 writer 持 `{A, B}` 两表,
      `CreateItem("a", ...)` 落 A、`CreateItem("b", ...)` 落 B,验证 `A.size()==1`、
      `B.size()==1`,采样返回各自数据。
- [ ] `LocalRoundTripRejectsCreateItemForUnknownTable`:writer 持 `{A, B}`,
      `CreateItem("c", ...)` 返回 `kNotFound`(`RunLocalWorker` 查 `tables_` 未命中)。
- [ ] `LocalRoundTripBackpressureAcrossTables`:小 `max_enqueued_inserts` 的 A 表,
      连续写 A 直至 `can_insert_more=false`,验证 writer 整体阻塞(writer 级
      backpressure,对齐 gRPC)。用 `table_test.cc` 的
      `InsertOrAssignAsync` + `can_insert_more` 用法参照。

**`Writer` 组**(`reverb/cc/writer_test.cc`,新增 3 个 `LocalWriter*`):

- [ ] `LocalWriterRoundTrip`:本地 `Writer` append → create_item → close,
      验证 item 落入指定 table 且可采样。
- [ ] `LocalWriterDispatchesToMultipleTables`:本地 `Writer` 持 `{A, B}` 两表,
      `create_item("a", ...)` / `create_item("b", ...)` 各落各表。
- [ ] `LocalWriterConfirmItemsEquivalence`:验证本地 `Writer` 的同步确认机制
      (`InsertCallback` 递减 `num_items_in_flight_`)与 gRPC `ConfirmItems` 语义一致——
      小 `max_in_flight_items`,连续 create_item 触发背压阻塞,close 后全部确认。

**`StructuredWriter` 组**(`reverb/cc/in_process_client_test.cc`,新增 1 个,
用真实 `Table` 端到端,非 `FakeWriter`):

- [ ] `StructuredWriterDispatchesToMultipleTablesViaConfigs`:通过
      `InProcessClient::NewStructuredWriter`(去 table 后的签名)创建 writer,
      传入两个 config(config A 的 `table` 字段 = "a",config B 的 = "b"),
      append 步骤后,验证 A 表与 B 表各收到对应 item(`A.size()`/`B.size()` 正确,
      采样数据匹配)。覆盖 config.table() → `StructuredWriter::ApplyConfigs` →
      `writer_->CreateItem(c.config.table(), ...)` → 本地 `TrajectoryWriter` 路由 →
      `tables_` 分发的完整链路。注意:现有 `structured_writer_test.cc` 用
      `FakeWriter`(忽略 table 参数)无法测此路径,故放 `in_process_client_test.cc`
      走真实 `Table`。P0 写时按去 table 后的签名预期,编译期即红(TDD)。

### P1 — C++ 层改动(绿)

两个类同构改动:持 `tables_` map + 按 `item.table()` 路由 + 用 `InsertOrAssignAsync`

- callback 确认。**串行推进 A→B→C**,不可并行(理由见下)。

**串行约束**:

- `NewStructuredWriter` 内部调 `NewTrajectoryWriter`(`in_process_client.cc:87`),
  A 改 `NewTrajectoryWriter` 签名(去 table),`NewStructuredWriter` 必须同步改——
  原子的一改俱改。
- B 的本地 `Writer` 持 `tables_` map 参照 A 落地的方案 P(持拷贝),A 先走能填坑。
- `in_process_client.cc` 是 A/B 共享写点,并行必 merge 冲突。
- C(pybind)依赖 A+B 都完。

**步骤 A:`TrajectoryWriter` 解绑(D1/D2)**

- [ ] `reverb/cc/trajectory_writer.h`:`table_` → `tables_`(map),构造函数签名
      改为收 `flat_hash_map<string, shared_ptr<Table>>`(持拷贝,方案 P)。
      `local_can_insert_more_` 维持 writer 级单一 bool(D2:对齐 gRPC writer 级
      backpressure)。
- [ ] `reverb/cc/trajectory_writer.cc`:`RunLocalWorker` 里
      `table_->InsertOrAssignAsync` 改为按 `item_and_refs->item.table()` 查
      `tables_` 分发,找不到返回 `kNotFound`。
- [ ] `reverb/cc/in_process_client.cc`:`NewTrajectoryWriter` 按决策点 D1 去 `table`
      参数,构造 writer 时传入 `tables_` 拷贝 + 按 D1 填 `flat_signature_map`
      (用 `FlatSignatureFromSignatureProto`,见 D1 代码)。
      **同步改 `NewStructuredWriter`**:去 `table` 参数,内部调
      `NewTrajectoryWriter(options, ...)`(不传 table)。

**步骤 B:`Writer` 本地化(D3 选 a,基于 server 侧复用,参照 A 的 map 模式)**

- [ ] `reverb/cc/writer.h`:加本地构造函数(收 `tables_` map 拷贝,方案 P,
      一步到位不绑表),加 `is_local_` 标志。`num_items_in_flight_` 复用于本地确认计数。
- [ ] `reverb/cc/writer.cc`:`WritePendingData` 里 `stream_->Write(request)` 分支为
      本地路径——按 `item.table()` 查 `tables_` 调 `InsertOrAssignAsync`,
      `InsertCallback` 递减 `num_items_in_flight_` 复刻 `ConfirmItems` 语义。
      `SaveChunks`/`GetItemWithChunks` 逻辑直接内联(Writer 的 `chunks_` 已持
      `ChunkData`,逻辑极简)或从 reactor 抽共享函数。`Finish`/`Close` 本地路径
      跳过 stream 收尾。
      **实现要点(chunk 类型转换 + 去重)**:`Writer::chunks_` 是
      `std::list<ChunkData>`(值类型),而 reactor/`Table::Item` 用
      `shared_ptr<ChunkStore::Chunk>`。本地化内联 `SaveChunks` 时要从 `ChunkData`
      构造 `shared_ptr<ChunkStore::Chunk>`(`std::make_shared<ChunkStore::Chunk>`,
      同 reactor),并保留 `streamed_chunk_keys_` 去重逻辑(避免同一 chunk 给表
      插多次)。多表 `insert` 时多个 item 引用同一 chunk,需共享同一 `shared_ptr<Chunk>`
      不复制——此点与 gRPC 路径语义一致。
      **实现要点(`ConfirmItems` 防死锁)**:`absl::Mutex::Await` 等待时释放锁,
      `InsertCallback` 能拿写锁递减,不死锁。但本地化无 `ItemConfirmationWorker`
      线程,gRPC 的 `done` 条件里的 `!item_confirmation_worker_running_` 不适用。
      本地 `done` 条件改为 `num_items_in_flight_ <= limit || closed_`(加 `closed_`
      逃生,防表 worker 出错致 callback 永不触发时永久阻塞,对齐 `TrajectoryWriter`
      的 `!closed_ && !stream_ok_` 等待模式)。`Close()` 本地路径:`ConfirmItems(0)`
      drain → 释放 callback,无遗留 in-flight,析构安全。
- [ ] `reverb/cc/in_process_client.h`/`.cc`:新增 `NewWriter` 方法,
      构造本地 `Writer`(传 `tables_` 拷贝)。

**步骤 C:pybind 绑定(依赖 A+B 都完)**

- [ ] `reverb/pybind.cc`:`InProcessClient` 绑定 `NewWriter`/`new_writer`(snake +
      PascalCase 双名,对齐 §3.3),返回 `unique_ptr<Writer>`(对齐 gRPC
      `Client.NewWriter` 绑定模式)。`new_trajectory_writer`/`NewTrajectoryWriter`
      按 D1 去掉 `table` 参数绑定。
      **后端无关确认**:`Writer` 类的方法绑定(`Append`/`CreateItem`/`Flush`/
      `Close`,`pybind.cc:345`)是类级绑定,与 gRPC/本地后端无关——`is_local_`
      分支在 C++ 方法内部。本地 `Writer` 实例自动复用现有绑定,无需新绑方法,
      `type_caster<TensorBuffer>` 在本地路径无坑。

**验证:**

- [ ] 跑 P0 全部测试(含 StructuredWriter 多表)+ 现有 `LocalRoundTrip*` +
      `writer_test.cc` + `in_process_client_test.cc`,全绿。
- [ ] `bazel test //reverb/cc/...` 全量验证。

### P2 — Python 层对齐

- [ ] `reverb/client.py`:`LocalClient.trajectory_writer`/`structured_writer` 去掉
      `table` 参数,签名对齐 gRPC `Client`。
- [ ] `reverb/client.py`:`insert` 上提到 `_BaseClient`(方案 α,靠
      `self.writer` 鸭子类型分派),`LocalClient` 与 `Client` 共享单一实现,签名不漂移。
- [ ] `reverb/client.py`:补 `LocalClient.writer`(调 `InProcessClient.NewWriter`)
      与 `LocalClient.insert`(因上提而自动获得,但需确认 `LocalClient.writer` 就位后
      `self.writer` 调用链通)。
- [ ] `reverb/client.py`:`_default_emit_timesteps` 统一回 `True`。保留
      `__reduce__` 不实现(pickle 约束)。
- [ ] 更新 `reverb/tests/in_process_test.py`:
      `test_sample_emit_timesteps_defaults_differ` 改为验证两边默认一致(或删除)。
      所有 `local.trajectory_writer(table='t', ...)` 调用点去掉 `table=`,
      `local.structured_writer(table='t', ...)` 同理。
- [ ] `reverb/tests/in_process_test.py` 新增:`LocalClient.insert` / `writer` 可用性
      测试,验证与 gRPC `Client` 行为一致。
- [ ] **双客户端 parity 回归与补全**(验证"严格镜像"核心承诺):
      - 更新 `_insert_both`(`in_process_test.py:683`):现状是 gRPC 用 `insert`、
        Local 用 `_insert_local`(trajectory_writer 绕路,注释明说"LocalClient
        has no `insert`")。修正后两边都用 `insert`,回归成真 API 对齐。
      - 新增 `writer` parity 测试:对同一 server,`grpc.writer(...)` 与
        `local.writer(...)` 各 append→create_item→close,断言 `sample` 返回数据一致。
        (现有 parity 系列只覆盖 `sample`/`mutate_priorities`/`reset`/`server_info`,
        `writer` 完全无双客户端对齐覆盖,而它是 D3-a 新补本地路径,最需对齐验证。)
      - 新增 `insert` parity 测试:两边 `insert` 同数据,断言 `sample` 一致。
        (由 `_insert_both` 回归间接覆盖,但建议显式一个。)
- [ ] `bazel test //reverb/tests/...` 全量验证。

### P3 — 文档更新

- [ ] `docs/numpy-embed-design.md` §3.2:改述"本地 writer 绑定单一 table"为
      已纠正的实现选择,补 backpressure 多表语义说明。
- [ ] §3.4:整节删除或改述为"`_default_emit_timesteps` 两端统一 `True`"。
- [ ] §3.5:缩减为仅"无 pickle",补 `insert`/`writer` 已对齐的说明。
- [ ] §1.3 决策汇总表 A1 行:更新"内嵌只做 TrajectoryWriter + StructuredWriter"——
      现内嵌也提供 `Writer`/`insert`(本地化),A1 的"砍掉省复杂度"理由不再成立。
      标注 A1 的 Writer 部分被 [ADR-0001](adr/0001-embedded-writer-local-path.md) 推翻
      (`StreamingTrajectoryWriter` 部分保留)。
- [ ] §1.3 决策汇总表:补 D1/D2/D3 决策行。
- [ ] 新增 `docs/adr/0001-embedded-writer-local-path.md`(已写):记录 A1→D3 的决策推翻,
      含背景/触发事实/决策/后果/status。P3 阶段随实施落地确认其 accepted 状态。

## 验收标准

- P0 的 7 个新 C++ 测试(3 个 TrajectoryWriter + 3 个 Writer + 1 个
  StructuredWriter 多表)+ 现有全部 C++ 测试绿。
- `bazel test //reverb/cc/...` 和 `bazel test //reverb/tests/...` 全绿。
- `LocalClient.trajectory_writer`/`structured_writer` 签名与 gRPC `Client` 一致
  (无 `table` 参数)。
- `LocalClient.insert`/`writer` 可用且与 gRPC `Client` 语义一致(多表 `insert` 成立,
  `writer` 流式写入本地路径与 gRPC 行为对齐)。
  **可证伪验证**:P2 的双客户端 parity 测试(`_insert_both` 回归 + 新增 `writer`/
  `insert` parity)全绿——同一 server 用 gRPC `Client` 与 `LocalClient` 各跑
  `insert`/`writer` + `sample`,断言返回数据/行为一致。这是"严格镜像"核心承诺的
  端到端锁定,非仅签名对齐。
- `_default_emit_timesteps` 两端统一为 `True`。
- 设计文档 §1.3(A1)/§3.2/§3.4/§3.5 如实反映修正后状态。
