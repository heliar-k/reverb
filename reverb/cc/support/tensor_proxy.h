#ifndef REVERB_CC_SUPPORT_TENSOR_PROXY_H_
#define REVERB_CC_SUPPORT_TENSOR_PROXY_H_

#include <cstdint>
#include <memory>
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

// 存储模型:owner_ 持字节宿主(shared_ptr,任意线程可析构),bytes_ 是
// 指向宿主的 view。维度操作(InsertBatchDim 等)共享宿主零拷贝;ToNdArray
// 产出 view 该宿主的 numpy 数组(capsule 持 owner_ 副本)。
//
// 写入侧默认拷贝(append 时刻快照语义,worker 线程零 GIL)。
// FromNdArray(zero_copy=true) 时视图 numpy 存储:Py_INCREF 持宿主,
// 任意线程析构将 PyObject* 入 DeferredFreeQueue,主线程在下一次
// FromNdArray/ToNdArray 入口(持 GIL)统一 DECREF。零拷贝丢失快照语义
// (append 后原地复用 buffer 会写脏数据),故同时把源数组置 read-only,
// 让原地改写立刻报错。开启方式:环境变量 REVERB_ZERO_COPY_APPEND=1
// (type_caster 读取,见 pybind.cc)。
//
// ponytail: pybind11 把 pybind11 namespace 标为 visibility("hidden"),
// 参数含 py::object 的方法 (FromNdArray/ToNdArray) 可见性被降为 hidden,
// 导致 libreverb.so 的 version-script (*deepmind*reverb*) 无法导出它们,
// libpybind.so import 时 undefined symbol。显式 default 恢复导出。
class __attribute__((visibility("default"))) TensorBuffer {
 public:
  TensorBuffer() = default;
  TensorBuffer(TensorSpec spec, std::string bytes);

  // 主线程调用(持 GIL)。zero_copy=true 时数值数组走视图(见类注释)。
  static absl::StatusOr<TensorBuffer> FromNdArray(py::object ndarray,
                                                  bool zero_copy = false);
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
  // 共享存储构造:维度操作与 view 派生共享同一字节宿主。
  TensorBuffer(TensorSpec spec, std::shared_ptr<void> owner,
               absl::string_view bytes);

  TensorSpec spec_;
  std::shared_ptr<void> owner_;  // 字节宿主;空 = 无字节(默认构造)
  absl::string_view bytes_;      // 指向 owner_ 内容,worker 零 GIL 可读
};

// DataType 与 proto 枚举、numpy 类型号互转
const char* DataTypeName(DataType dt);
::reverb::tensor::DataType DataTypeToProto(DataType dt);
absl::StatusOr<DataType> DataTypeFromProto(::reverb::tensor::DataType dt);

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_TENSOR_PROXY_H_
