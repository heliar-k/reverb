#include <complex>

#include "reverb/cc/support/tensor_proxy.h"

#include <gtest/gtest.h>
#include "numpy/arrayobject.h"
#include "pybind11/embed.h"
#include "pybind11/complex.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
using deepmind::reverb::DataType;
using deepmind::reverb::TensorBuffer;
using ::reverb::tensor::TensorProto;

namespace {

// 进程级单例 interpreter。numpy C-API 的全局状态(import_array、类型表)
// 在 Py_Finalize 后失效,因此整个测试进程只初始化/销毁 Python 一次。
// ponytail: 不用每测试 scoped_interpreter,那会在 Finalize 后让 numpy 崩。
struct PyEnv : public ::testing::Environment {
  void SetUp() override {
    static py::scoped_interpreter guard;
    if (_import_array() < 0) PyErr_Clear();
  }
};

::testing::Environment* const kPyEnv =
    ::testing::AddGlobalTestEnvironment(new PyEnv);

py::object MakeArray(const std::vector<float>& v, const std::string& dtype) {
  py::module np = py::module::import("numpy");
  py::list lst;
  for (float x : v) lst.append(x);
  return np.attr("array")(lst, py::arg("dtype") = dtype.c_str());
}

py::object MakeArrayInt(const std::vector<int32_t>& v, const std::string& dt) {
  py::module np = py::module::import("numpy");
  py::list lst;
  for (int32_t x : v) lst.append(x);
  return np.attr("array")(lst, py::arg("dtype") = dt.c_str());
}

// 复数数组:把 std::complex 列表喂给 np.array(dtype=...)。依赖 pybind11 的
// type_caster<std::complex<T>>(pybind11/complex.h)。
py::object MakeArrayComplex64(const std::vector<std::complex<float>>& v) {
  py::module np = py::module::import("numpy");
  py::list lst;
  for (const auto& z : v) lst.append(py::cast(z));
  return np.attr("array")(lst, py::arg("dtype") = "complex64");
}

py::object MakeArrayComplex128(const std::vector<std::complex<double>>& v) {
  py::module np = py::module::import("numpy");
  py::list lst;
  for (const auto& z : v) lst.append(py::cast(z));
  return np.attr("array")(lst, py::arg("dtype") = "complex128");
}

// 从 numpy array 对象按 C-order 读一个标量(给定元素字节偏移)。
template <typename T>
T ReadScalar(py::object arr, int64_t flat_index) {
  PyArrayObject* a = reinterpret_cast<PyArrayObject*>(arr.ptr());
  return *reinterpret_cast<const T*>(
      static_cast<const char*>(PyArray_DATA(a)) +
      flat_index * PyArray_ITEMSIZE(a));
}
}  // namespace

TEST(TensorBuffer, RoundTripFloat32) {
  py::object arr = MakeArray({1.0f, 2.0f, 3.0f}, "float32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  EXPECT_EQ(buf.dtype(), DataType::Float32);
  EXPECT_EQ(buf.shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(buf.NumElements(), 3);
  EXPECT_EQ(buf.TotalBytes(), 12);
  py::object out = buf.ToNdArray();
  EXPECT_EQ(ReadScalar<float>(out, 0), 1.0f);
  EXPECT_EQ(ReadScalar<float>(out, 1), 2.0f);
  EXPECT_EQ(ReadScalar<float>(out, 2), 3.0f);
}

TEST(TensorBuffer, ToNdArraySharesStorage) {
  // 零拷贝契约:数值数组的 ToNdArray 直接视图 TensorBuffer 的字节存储,
  // 不做 memcpy;数组必须自带存储所有权(base 非空)以存活于源对象之后。
  py::object arr = MakeArray({1.0f, 2.0f, 3.0f}, "float32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  py::object out = buf.ToNdArray();
  PyArrayObject* out_arr = reinterpret_cast<PyArrayObject*>(out.ptr());
  EXPECT_EQ(PyArray_DATA(out_arr), buf.bytes().data());
  EXPECT_NE(PyArray_BASE(out_arr), nullptr);
}

TEST(TensorBuffer, ToNdArrayOutlivesSource) {
  py::object out;
  {
    py::object arr = MakeArray({4.0f, 5.0f}, "float32");
    auto buf = TensorBuffer::FromNdArray(arr).value();
    out = buf.ToNdArray();
  }  // buf 在此析构;out 必须仍持有有效字节。
  EXPECT_EQ(ReadScalar<float>(out, 0), 4.0f);
  EXPECT_EQ(ReadScalar<float>(out, 1), 5.0f);
}

TEST(TensorBuffer, FromNdArrayZeroCopySharesBuffer) {
  // 零拷贝契约(opt-in):buffer 直接视图 numpy 存储,不做 memcpy;
  // 源数组被置 read-only——append 后原地改写立刻报错而非静默写脏。
  py::object arr = MakeArray({1.0f, 2.0f, 3.0f}, "float32");
  auto buf = TensorBuffer::FromNdArray(arr, /*zero_copy=*/true).value();
  PyArrayObject* a = reinterpret_cast<PyArrayObject*>(arr.ptr());
  EXPECT_EQ(buf.bytes().data(), PyArray_DATA(a));
  EXPECT_EQ(PyArray_FLAGS(a) & NPY_ARRAY_WRITEABLE, 0);
  EXPECT_EQ(buf.TotalBytes(), 12);
}

TEST(TensorBuffer, DefaultCopySemanticsSnapshot) {
  // 默认路径保持 append 时刻快照:改写源数组不影响已取字节。
  py::object arr = MakeArray({1.0f}, "float32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  PyArrayObject* a = reinterpret_cast<PyArrayObject*>(arr.ptr());
  ASSERT_NE(PyArray_FLAGS(a) & NPY_ARRAY_WRITEABLE, 0);
  static_cast<float*>(PyArray_DATA(a))[0] = 9.0f;
  EXPECT_EQ(reinterpret_cast<const float*>(buf.bytes().data())[0], 1.0f);
}

TEST(TensorBuffer, ZeroCopyHoldsArrayAlive) {
  TensorBuffer buf;
  {
    py::object arr = MakeArray({3.0f}, "float32");
    buf = TensorBuffer::FromNdArray(arr, /*zero_copy=*/true).value();
  }  // py::object 释放;buffer 必须自持引用保活底层存储。
  EXPECT_EQ(reinterpret_cast<const float*>(buf.bytes().data())[0], 3.0f);
}

TEST(TensorBuffer, ZeroCopyDestroyWithoutGILDefersFree) {
  py::object arr = MakeArray({1.0f}, "float32");
  PyObject* raw = arr.ptr();
  Py_ssize_t before = Py_REFCNT(raw);
  {
    auto buf = TensorBuffer::FromNdArray(arr, /*zero_copy=*/true).value();
    EXPECT_EQ(Py_REFCNT(raw), before + 1);
    // 模拟 worker 线程析构:无 GIL → 引用入 DeferredFreeQueue 而非就地
    // Py_DECREF。
    py::gil_scoped_release release;
    auto gone = std::move(buf);
  }
  // 尚未 drain:引用仍挂起。
  EXPECT_EQ(Py_REFCNT(raw), before + 1);
  // 主线程在 FromNdArray 入口顺带 drain。
  auto dummy = TensorBuffer::FromNdArray(MakeArray({0.0f}, "float32")).value();
  EXPECT_EQ(Py_REFCNT(raw), before);
}

TEST(TensorBuffer, FinishMimicSharedStorage) {
  // 复刻 Writer::Finish 的对象图:InsertBatchDim 共享 owner 后,
  // 按 Finish 的析构顺序销毁(concat 产物独立于源存活)。
  py::object arr = MakeArray({1.0, 2.0}, "float64");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  std::vector<TensorBuffer> buffer_col;
  buffer_col.push_back(std::move(buf));
  std::vector<TensorBuffer> tensors;
  tensors.push_back(buffer_col[0].InsertBatchDim());
  auto concat = TensorBuffer::Concat(tensors).value();
  tensors.clear();
  buffer_col.clear();
  EXPECT_EQ(concat.TotalBytes(), 16);
  py::object out = concat.ToNdArray();
  EXPECT_EQ(ReadScalar<double>(out, 1), 2.0);
}

TEST(TensorBuffer, RoundTripInt64) {
  py::object arr = MakeArrayInt({10, 20, 30}, "int64");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  EXPECT_EQ(buf.dtype(), DataType::Int64);
  EXPECT_EQ(buf.TotalBytes(), 24);
  py::object out = buf.ToNdArray();
  EXPECT_EQ(ReadScalar<int64_t>(out, 0), 10);
  EXPECT_EQ(ReadScalar<int64_t>(out, 2), 30);
}

TEST(TensorBuffer, Concat) {
  py::object a = MakeArray({1.0f, 2.0f}, "float32");
  py::object b = MakeArray({3.0f, 4.0f}, "float32");
  auto ba = TensorBuffer::FromNdArray(a).value();
  auto bb = TensorBuffer::FromNdArray(b).value();
  auto merged = TensorBuffer::Concat({ba, bb}).value();
  EXPECT_EQ(merged.shape(), std::vector<int64_t>({4}));
  EXPECT_EQ(merged.TotalBytes(), 16);
  py::object out = merged.ToNdArray();
  EXPECT_EQ(ReadScalar<float>(out, 0), 1.0f);
  EXPECT_EQ(ReadScalar<float>(out, 3), 4.0f);
}

TEST(TensorBuffer, SubSlice) {
  py::module np = py::module::import("numpy");
  // [[1,2],[3,4],[5,6]]
  py::list rows;
  for (int i = 0; i < 3; ++i) {
    py::list r;
    r.append(static_cast<float>(i * 2 + 1));
    r.append(static_cast<float>(i * 2 + 2));
    rows.append(r);
  }
  py::object arr = np.attr("array")(rows, py::arg("dtype") = "float32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  ASSERT_EQ(buf.shape(), std::vector<int64_t>({3, 2}));
  auto row1 = buf.SubSlice(1);
  EXPECT_EQ(row1.shape(), std::vector<int64_t>({2}));
  py::object out = row1.ToNdArray();
  EXPECT_EQ(ReadScalar<float>(out, 0), 3.0f);
  EXPECT_EQ(ReadScalar<float>(out, 1), 4.0f);
}

TEST(TensorBuffer, SerializeRoundTrip) {
  py::object arr = MakeArrayInt({10, 20, 30}, "int32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  TensorProto proto;
  ASSERT_TRUE(buf.SerializeToProto(&proto).ok());
  EXPECT_EQ(proto.tensor_content().size(), 12);
  auto back = TensorBuffer::DeserializeFromProto(proto).value();
  EXPECT_EQ(back.dtype(), DataType::Int32);
  EXPECT_EQ(back.shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(back.bytes(), buf.bytes());
}

TEST(TensorBuffer, DeserializeRejectsTruncatedContent) {
  // A corrupt/peer-crafted TensorProto whose tensor_content is shorter than
  // shape*dtype implies must be rejected — otherwise ToNdArray memcpys
  // NumElements*itemsize bytes out of the undersized string (heap over-read).
  py::object arr = MakeArrayInt({10, 20, 30}, "int32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  TensorProto proto;
  ASSERT_TRUE(buf.SerializeToProto(&proto).ok());
  proto.set_tensor_content(std::string(4, '\0'));  // shape says 12 bytes
  auto bad = TensorBuffer::DeserializeFromProto(proto);
  EXPECT_FALSE(bad.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(bad.status())) << bad.status();

  // Oversized content is equally suspect (length-prefixed framing elsewhere
  // relies on exact sizes).
  proto.set_tensor_content(std::string(16, '\0'));
  EXPECT_FALSE(TensorBuffer::DeserializeFromProto(proto).ok());
}

TEST(TensorBuffer, InsertAndRemoveBatchDim) {
  py::object arr = MakeArray({1.0f, 2.0f, 3.0f}, "float32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  auto batched = buf.InsertBatchDim();
  EXPECT_EQ(batched.shape(), std::vector<int64_t>({1, 3}));
  EXPECT_EQ(batched.TotalBytes(), 12);
  auto unbatched = batched.RemoveBatchDim();
  EXPECT_EQ(unbatched.shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(unbatched.bytes(), buf.bytes());
}

TEST(TensorBuffer, CopyReshaped) {
  py::object arr = MakeArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, "float32");
  auto buf = TensorBuffer::FromNdArray(arr).value();
  auto reshaped = buf.CopyReshaped({2, 3});
  EXPECT_EQ(reshaped.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(reshaped.NumElements(), 6);
  EXPECT_EQ(reshaped.bytes(), buf.bytes());
}

// 复现标量列 shape 语义:0-d 标量经 InsertBatchDim(应保持 0-d)→
// N 个堆叠 Concat 成 [N] → SubSlice 取行应回 0-d。TF 期标量跨步堆叠为 [N]。
TEST(TensorBuffer, ScalarColumnShapeSemantics) {
  py::module np = py::module::import("numpy");
  // 3 个 0-d 标量,模拟跨 3 个时间步 append 同一标量列。
  // 走真实路径:Python int -> FromNdArray(asarray) -> 0-d。
  std::vector<TensorBuffer> batched;
  for (int i = 0; i < 3; ++i) {
    py::object arr = np.attr("asarray")(py::int_(i));
    auto buf = TensorBuffer::FromNdArray(arr).value();
    ASSERT_EQ(buf.shape(), std::vector<int64_t>({})) << "0-d 标量";
    // InsertBatchDim 对标量不应插入 batch 维(TF 语义:标量保持 0-d)。
    batched.push_back(buf.InsertBatchDim());
    EXPECT_EQ(batched.back().shape(), std::vector<int64_t>({}))
        << "标量 InsertBatchDim 应保持 0-d";
  }
  // Concat 把 N 个 0-d 堆叠成 [N]。
  auto merged = TensorBuffer::Concat(batched).value();
  EXPECT_EQ(merged.shape(), std::vector<int64_t>({3}))
      << "N 个 0-d 标量 Concat 应为 [N]";
  // SubSlice 取一行应回到 0-d 标量。
  auto row = merged.SubSlice(1);
  EXPECT_EQ(row.shape(), std::vector<int64_t>({}))
      << "[N] SubSlice 取行应回 0-d";
  py::object out = row.ToNdArray();
  PyArrayObject* a = reinterpret_cast<PyArrayObject*>(out.ptr());
  EXPECT_EQ(PyArray_NDIM(a), 0) << "ToNdArray 应为 0-d";
  EXPECT_EQ(ReadScalar<int64_t>(out, 0), 1);
}

TEST(TensorBuffer, ZeroDimFromScalar) {
  // numpy 把标量 coerce 成 0-d array,TensorBuffer 应接受。
  py::module np = py::module::import("numpy");
  py::object arr = np.attr("ascontiguousarray")(py::int_(42));
  auto status = TensorBuffer::FromNdArray(arr);
  EXPECT_TRUE(status.ok());
}

// RED: complex64 itemsize 当前为 4(np.complex64 = 2×float32 = 8 字节)。
// TotalBytes 会返回 8(应 16);ToNdArray 的 memcpy 只拷 8 字节,虚部丢失。
TEST(TensorBuffer, RoundTripComplex64) {
  py::object arr =
      MakeArrayComplex64({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto buf = TensorBuffer::FromNdArray(arr).value();
  EXPECT_EQ(buf.dtype(), DataType::Complex64);
  EXPECT_EQ(buf.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(buf.NumElements(), 2);
  // 2 元素 × 8 字节 = 16。当前错返回 8。
  EXPECT_EQ(buf.TotalBytes(), 16);
  py::object out = buf.ToNdArray();
  EXPECT_EQ(ReadScalar<std::complex<float>>(out, 0),
            std::complex<float>(1.0f, 2.0f));
  EXPECT_EQ(ReadScalar<std::complex<float>>(out, 1),
            std::complex<float>(3.0f, 4.0f));
}

// RED: complex128 itemsize 当前为 8(np.complex128 = 2×float64 = 16 字节)。
TEST(TensorBuffer, RoundTripComplex128) {
  py::object arr =
      MakeArrayComplex128({{1.0, 2.0}, {3.0, 4.0}});
  auto buf = TensorBuffer::FromNdArray(arr).value();
  EXPECT_EQ(buf.dtype(), DataType::Complex128);
  // 2 元素 × 16 字节 = 32。当前错返回 16。
  EXPECT_EQ(buf.TotalBytes(), 32);
  py::object out = buf.ToNdArray();
  EXPECT_EQ(ReadScalar<std::complex<double>>(out, 0),
            std::complex<double>(1.0, 2.0));
  EXPECT_EQ(ReadScalar<std::complex<double>>(out, 1),
            std::complex<double>(3.0, 4.0));
}

// 覆盖 Concat 标量堆叠路径(最危险的 under-allocation 点):
// 0-d 标量 N 个 Concat 成 [N],bytes.resize(N*itemsize)。itemsize 修复后应
// 正确容纳全部复数。此为覆盖增补(修复已在),非 red-green。
TEST(TensorBuffer, ConcatComplex64) {
  py::module np = py::module::import("numpy");
  // 2 个 0-d 复数标量,Concat 成 [2]。
  std::vector<TensorBuffer> buffers;
  buffers.push_back(
      TensorBuffer::FromNdArray(np.attr("array")(
          py::cast(std::complex<float>(1.0f, 2.0f)),
          py::arg("dtype") = "complex64")).value());
  buffers.push_back(
      TensorBuffer::FromNdArray(np.attr("array")(
          py::cast(std::complex<float>(3.0f, 4.0f)),
          py::arg("dtype") = "complex64")).value());
  auto merged = TensorBuffer::Concat(buffers).value();
  EXPECT_EQ(merged.shape(), std::vector<int64_t>({2}));
  // 2 元素 × 8 字节 = 16。
  EXPECT_EQ(merged.TotalBytes(), 16);
  py::object out = merged.ToNdArray();
  EXPECT_EQ(ReadScalar<std::complex<float>>(out, 0),
            std::complex<float>(1.0f, 2.0f));
  EXPECT_EQ(ReadScalar<std::complex<float>>(out, 1),
            std::complex<float>(3.0f, 4.0f));
}
