#include "reverb/cc/support/tensor_proxy.h"

#include <gtest/gtest.h>
#include "numpy/arrayobject.h"
#include "pybind11/embed.h"
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

TEST(TensorBuffer, ZeroDimFromScalar) {
  // numpy 把标量 coerce 成 0-d array,TensorBuffer 应接受。
  py::module np = py::module::import("numpy");
  py::object arr = np.attr("ascontiguousarray")(py::int_(42));
  auto status = TensorBuffer::FromNdArray(arr);
  EXPECT_TRUE(status.ok());
}
