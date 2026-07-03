#ifndef REVERB_CC_SUPPORT_TENSOR_PROXY_H_
#define REVERB_CC_SUPPORT_TENSOR_PROXY_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {

namespace py = pybind11;

// C++ 侧 dtype 枚举,对齐 numpy(不对齐 TF)
enum class DataType : uint8_t {
  Invalid = 0,
  Float32, Float64,
  Int8, Int16, Int32, Int64,
  Uint8, Uint16, Uint32, Uint64,
  Bool,
  Complex64, Complex128,
  String,
};

struct TensorSpec {
  DataType dtype;
  std::vector<int64_t> shape;
};

// ponytail: 当前拷贝 bytes(std::string)语义,worker 线程零 GIL。
// 若 profile 显示写入 memcpy 成瓶颈,改零拷贝裸指针 + DeferredFreeQueue:
//   - FromNdArray 时 Py_INCREF 持 PyObject*,拿 PyArray_DATA 裸指针
//   - worker 线程读裸指针,不持 GIL
//   - 释放时入 DeferredFreeQueue,主线程持 GIL 统一 Py_DECREF
class TensorBuffer {
 public:
  TensorBuffer() = default;
  TensorBuffer(TensorSpec spec, std::string bytes);

  // 主线程调用(持 GIL)
  static absl::StatusOr<TensorBuffer> FromNdArray(py::object ndarray);
  py::object ToNdArray() const;

  // worker 线程安全(零 GIL),只读
  DataType dtype() const { return spec_.dtype; }
  const std::vector<int64_t>& shape() const { return spec_.shape; }
  absl::string_view bytes() const { return bytes_; }
  int64_t NumElements() const;
  int64_t TotalBytes() const;  // = NumElements * sizeof(dtype)
  bool IsAligned() const { return true; }  // FromNdArray 强制 C-contiguous

  // 维度操作(返回新对象,bytes 不变)
  TensorBuffer InsertBatchDim() const;   // shape 前插 1
  TensorBuffer RemoveBatchDim() const;   // shape 去 [0]
  TensorBuffer CopyReshaped(const std::vector<int64_t>& shape) const;

  // 拼接/切片
  static absl::StatusOr<TensorBuffer> Concat(
      const std::vector<TensorBuffer>& buffers);  // 沿第 0 维拼接
  TensorBuffer SubSlice(int64_t offset) const;     // 取第 offset 行,去 batch 维

  // 序列化
  absl::Status SerializeToProto(::reverb::tensor::TensorProto* out) const;
  static absl::StatusOr<TensorBuffer> DeserializeFromProto(
      const ::reverb::tensor::TensorProto& proto);

 private:
  TensorSpec spec_;
  std::string bytes_;  // ponytail: 升级路径见类顶部注释
};

// DataType 与 proto 枚举、numpy 类型号互转
const char* DataTypeName(DataType dt);
::reverb::tensor::DataType DataTypeToProto(DataType dt);
absl::StatusOr<DataType> DataTypeFromProto(::reverb::tensor::DataType dt);

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_TENSOR_PROXY_H_
