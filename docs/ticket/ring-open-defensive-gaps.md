# Tickets: Ring::Open/Create 防御性校验缺口（2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review，LOW #7（1/2 票存活，另一质疑方
> 独立复核了三项子声明）。可达性限于段损坏/版本错配/并发启动，故定级 LOW。

## Open 只校验 magic；EEXIST 回退可误删活段；Read 不校验 body_len

**What to build**：补齐段几何校验，消除并发启动误删窗口。

**现状**（ring.cc）：

1. **Open 校验不足**（:150-161）：仅验证 `hdr.magic == kRingMagic`，不验证版本、
   `capacity` 是否为 2 的幂、`slot_size` 下限。`Create` 侧有校验（:90-94 拒绝非 2 幂
   capacity 与过小 slot_size），但 `Open` 信任段内几何 —— 旧版本或损坏段会按错误
   几何 mmap，行为未定义。
2. **EEXIST unlink-and-retry**（:98-104）：`Create` 遇 `shm_open(O_CREAT|O_EXCL)`
   EEXIST 时 `shm_unlink` 后重试。真实场景是**崩溃重启竞态**：server 崩溃后旧段残留，
   重启的 server（同 socket_path）重建客户端段时 EEXIST → unlink —— 而崩溃瞬间客户端
   尚未察觉 EOF、仍在映射并写旧段（客户端 mmap 后已关 fd，无法用 nlink 判断死活）→
   客户端写入落进孤儿 inode，新 server 永远看不到。窗口自愈（客户端很快察觉 EOF 重连），
   但期间数据静默丢失。
3. **Read 不校验 body_len**（Ring::Read 读多槽消息时）：不验证 `body_len <= slot_size`
   （或 `body_cap`），损坏段可让消费端越界读取。

**修复选项**：

1. Open 增验：版本字段（`RingHeader` 加 version）、`capacity` 2 的幂、`slot_size` 下限，
   不满足返回 `InternalError` 而非裸用。
2. EEXIST 路径：段内容无法区分死活（见上），「校验后 unlink」不可行。选项：文档化接受
   （窗口自愈）或段名加 server epoch（重启递增，彻底消除碰撞，改动大，列入远期）。
3. Read 对 `body_len` 做 `<= capacity` 校验，超限按协议错误处理。

**Blocked by**：None — 低优先级加固。

- [x] `RingHeader` 版本字段 + Open 几何校验（2 的幂 / slot_size 下限）
- [x] EEXIST 路径不再盲目 unlink（校验后打开或确认归属）
- [x] Read 校验 `body_len`

## 修复记录（2026-07-31，已完成）

三项一起做完（红测先行，均确定性复现，无时序竞态）：

1. **Open 校验**（ring.cc）：`version != kRingVersion` 报错；`capacity` 非 2 幂 / `slot_size
   <= sizeof(SlotHeader)` 报错。注：`RingHeader.version` 字段早已存在（Create 侧已写入），
   本票只补 Open 侧校验，无 header 布局变更。
2. **EEXIST 根治 = 段名加 server epoch**：`ShmServer::Create` 生成
   `name_token = <socket_path>_<pid>_<boot_nanos>`，pool 与四条 ring 名全部折进该 token
   （`MakeShmNames`/`MakePoolShmName` 签名不变）。客户端段名本就从 Welcome 下发
   （shm_client.cc 不自己算名），故**协议无破坏**——只有服务端命名变化。碰撞消失后
   `Ring::Create`/`ShmBytePool::Create` 里的 unlink-and-retry 实际成为死代码（保留作防御）。
   权衡：崩溃残留段不再被重启清理（tmpfs 少量泄漏，重启自清）；刻意不加启动扫描清理——
   两活 server 同 socket 的误配置场景下扫描会误删对方活段，违背本票初衷。
3. **Read 校验 `body_len > SlotBodyCap()`**（覆盖首槽与 continuation 槽，一处 guard）。

**复现教训**：

- 红测全部免时序：Open/Read 用手搓坏 header/slot（`PokeSegment`：shm_open+mmap 第三映射
  写字段）；body_len 红测在旧代码下直接 **SIGSEGV**（段外读），证明越界读风险真实。
- epoch 红测用「同 socket_path 起第二个 server + 同进程同 PID 客户端」：旧代码下第二个
  server 的 Create 把活 pool/ring unlink 到 nlink=0（fstat 观测），段名逐字节相等；修复后
  nlink 保持 1 且段名相异。测试在 `shm_crash_test.cc`
  （`SecondServerSameSocketDoesNotClobberLiveSegments`）。
- 手算段名的既有断言改为从连接读名（`NamesFromConn`）；fork 子进程场景父进程读不到子
  连接，改用 /dev/shm 按 `_<pid>` 后缀计数（`CountShmSegmentsWithSuffix`）。
- `docs/spec/numpy-shm-spec.md`、`docs/design/numpy-shm-design.md` 中的段名公式早已落后于
  scan #12（socket token），本次 epoch 后更旧；以 `bootstrap.h` 注释为准，未同步改文档。
