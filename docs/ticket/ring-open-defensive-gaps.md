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

- [ ] `RingHeader` 版本字段 + Open 几何校验（2 的幂 / slot_size 下限）
- [ ] EEXIST 路径不再盲目 unlink（校验后打开或确认归属）
- [ ] Read 校验 `body_len`
