# Tickets: numpy↔字节流零拷贝（Tier 1+2，2026-07-26 完成）

> 归档自原 `tickets.md` 活动板。本组 ticket 全部结案。

调研结论(见上一 session 报告):numpy↔`TensorBuffer` 两边缘各有一次强制
memcpy(`tensor_proxy.cc` FromNdArray/ToNdArray),torch↔numpy 已零拷贝。
本期按分层方案的 Tier 1+2 落地,TDD 推进,三 slice 串行。

## Slice 1:采样侧 ToNdArray 零拷贝

**What to build**:`TensorBuffer` 存储从 `std::string bytes_` 改为
`shared_ptr<void> owner_ + string_view bytes_`(公共构造签名不变,链路零改动);
`ToNdArray` 用 `PyArray_SimpleNewFromData` + capsule 持 `owner_` 副本,
返回数组独立于源 TensorBuffer 存活。维度操作(InsertBatchDim 等)顺带共享
宿主,消掉原有的隐式 string 拷贝。

- [x] C++ `ToNdArraySharesStorage`(指针相等)/`ToNdArrayOutlivesSource`
  /`FinishMimicSharedStorage`(Writer::Finish 对象图回归)
- [x] Python seam(in_process_test):采样数组 `owndata == False`;
  del client+gc 后值不变
- [x] 现有 RoundTrip 全套 + in_process 44 例绿(值语义不变)

## Slice 2:写入侧 FromNdArray 零拷贝(opt-in)

**What to build**:`FromNdArray(zero_copy=true)` 数值 dtype 走视图:
`Py_INCREF` 持 numpy + 裸指针;任意线程析构将 PyObject* 入
DeferredFreeQueue,主线程在 FromNdArray/ToNdArray 入口(持 GIL)统一
DECREF。开启方式:环境变量 `REVERB_ZERO_COPY_APPEND=1`(type_caster 首次
Append 读一次)。**快照语义丢失**(append 后原地复用 buffer 会写脏数据),
故 opt-in 且开启时源数组置 read-only,让原地改写立刻 `ValueError` 而非静默
写脏(best-effort:torch 侧或其他 view 写入绕过该 flag,文档已注明)。

- [x] C++ 四例:指针相等+guard 置位/默认快照语义钉住/源释放后 buffer 存活
  /无 GIL 析构入队、主线程 drain 后 refcount 回落
- [x] Python seam(新 target `zero_copy_write_test`,bazel env 属性置 1):
  read-only 置位、append 后原地改写抛 ValueError、roundtrip 值一致
- [x] 默认路径行为不变(全部存量回归绿)

## Slice 3:SHM 插入直写 pool

**What to build**:`RunShmWorker` 的 `SerializeToString`+memcpy 两跳改为
`ByteSizeLong` 预算→ALLOCATE→`SerializeToArray(pool.At(offset))` 一跳;
序列化失败(理论不可达)走统一 alloc_failed 释放路径不泄漏 offset。
无行为变化,不新写测试:shm_insert/shm_sample/shm_test 回归即保护网。

- [x] 三个 SHM 测试目标绿

> **实现说明(2026-07-26)**:诊断插曲——stash/pop + 被杀的构建留下不一致
> .so 导致一次假 segfault(ABI 混合),全量重编后消失;教训:ABI 敏感改动
> 的诊断前先全量重编。torch 侧零拷贝随本期自动成立:采样 `from_numpy`
> 本即零拷贝(view 链 torch→numpy→capsule→owner);写入 CPU tensor 经
> `.numpy()` 视图接上 zero_copy 路径,无需额外改动;CUDA 的 D2H 物理不可免。
> 遗留:Concat/`set_tensor_content` 的物化拷贝属 wire format 强制(Tier 3,
> 协议层,待 profile);snappy 输出 buffer 算法固有;DeferredFreeQueue 进程
> 退出时残留随进程回收(ponytail 标注)。
>
> **code-review(双轴,2026-07-26)**:Standards 0 硬违规,唯一行动项
> RunShmWorker 失败模板重复→已提取 `fail_alloc` lambda(7 处收敛);文档级
> 重复以 tensor_proxy.h 类注释为权威版。Spec 验收全覆盖,两处修复:
> SubSlice 数值路径视图化(去物化,Slice 1 意图补齐);ToNdArray 空数组
> 分配失败回退 `py::none()`(修本期引入的小回归)。int 截断注记同 proto
> API 上限,不处理。修复后 14 个受影响测试目标全绿。
