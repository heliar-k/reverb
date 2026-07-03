#include "reverb/cc/support/tensor_proxy.h"

#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "numpy/arrayobject.h"
#include "pybind11/pybind11.h"

namespace deepmind {
namespace reverb {

namespace py = pybind11;

namespace {

// numpy 的 import_array 必须在 interpreter 启动后调一次。懒加载,避免在
// 静态初始化期或无 interpreter 时崩溃。
void ImportNumpyOnce() {
  static bool imported = false;
  if (!imported) {
    if (_import_array() < 0) {
      // import_array 失败时已设置 Python 异常;此处吞掉,调用方 ensure()
      // 路径会自然走到错误返回。
      PyErr_Clear();
    }
    imported = true;
  }
}

// ponytail: 用 numpy C-API 而非 pybind11 的 py::array/py::dtype。
// pybind11 3.0.4 的 npy_api::lookup 会访问 ModuleSpec._initializing,
// 该属性 Python 3.10 没有(3.11+ 才有),嵌入 3.10 时必崩。直走 C-API 绕过。
DataType NpyTypeToDataType(int type_num) {
  switch (type_num) {
    case NPY_FLOAT32: return DataType::Float32;
    case NPY_FLOAT64: return DataType::Float64;
    case NPY_INT8: return DataType::Int8;
    case NPY_INT16: return DataType::Int16;
    case NPY_INT32: return DataType::Int32;
    case NPY_INT64: return DataType::Int64;
    case NPY_UINT8: return DataType::Uint8;
    case NPY_UINT16: return DataType::Uint16;
    case NPY_UINT32: return DataType::Uint32;
    case NPY_UINT64: return DataType::Uint64;
    case NPY_BOOL: return DataType::Bool;
    case NPY_COMPLEX64: return DataType::Complex64;
    case NPY_COMPLEX128: return DataType::Complex128;
    case NPY_STRING:
    case NPY_UNICODE:
    case NPY_OBJECT: return DataType::String;
    default: return DataType::Invalid;
  }
}

int DataTypeToNpy(DataType dt) {
  switch (dt) {
    case DataType::Float32: return NPY_FLOAT32;
    case DataType::Float64: return NPY_FLOAT64;
    case DataType::Int8: return NPY_INT8;
    case DataType::Int16: return NPY_INT16;
    case DataType::Int32: return NPY_INT32;
    case DataType::Int64: return NPY_INT64;
    case DataType::Uint8: return NPY_UINT8;
    case DataType::Uint16: return NPY_UINT16;
    case DataType::Uint32: return NPY_UINT32;
    case DataType::Uint64: return NPY_UINT64;
    case DataType::Bool: return NPY_BOOL;
    case DataType::Complex64: return NPY_COMPLEX64;
    case DataType::Complex128: return NPY_COMPLEX128;
    case DataType::String: return NPY_OBJECT;
    case DataType::Invalid: return NPY_NOTYPE;
  }
  return NPY_NOTYPE;
}

int DataTypeItemsize(DataType dt) {
  switch (dt) {
    case DataType::Float32:
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Complex64:
      return 4;
    case DataType::Float64:
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Complex128:
      return 8;
    case DataType::Int16:
    case DataType::Uint16:
      return 2;
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Bool:
      return 1;
    case DataType::String:
      return 0;  // 变长
    case DataType::Invalid:
      return 0;
  }
  return 0;
}

// string bytes 编码:每个元素 [4 字节 little-endian 长度][内容]。
// 仅 String dtype 使用。读时按 NumElements 解。
void EncodeString(std::string* out, const std::string& item) {
  uint32_t len = static_cast<uint32_t>(item.size());
  char header[4] = {
      static_cast<char>(len & 0xff),
      static_cast<char>((len >> 8) & 0xff),
      static_cast<char>((len >> 16) & 0xff),
      static_cast<char>((len >> 24) & 0xff),
  };
  out->append(header, 4);
  out->append(item);
}

std::string DecodeString(absl::string_view src, size_t* pos) {
  uint32_t len = static_cast<uint8_t>(src[*pos]) |
                 (static_cast<uint8_t>(src[*pos + 1]) << 8) |
                 (static_cast<uint8_t>(src[*pos + 2]) << 16) |
                 (static_cast<uint8_t>(src[*pos + 3]) << 24);
  *pos += 4;
  std::string s(src.substr(*pos, len));
  *pos += len;
  return s;
}

}  // namespace

TensorBuffer::TensorBuffer(TensorSpec spec, std::string bytes)
    : spec_(std::move(spec)), bytes_(std::move(bytes)) {}

const char* DataTypeName(DataType dt) {
  switch (dt) {
    case DataType::Float32: return "Float32";
    case DataType::Float64: return "Float64";
    case DataType::Int8: return "Int8";
    case DataType::Int16: return "Int16";
    case DataType::Int32: return "Int32";
    case DataType::Int64: return "Int64";
    case DataType::Uint8: return "Uint8";
    case DataType::Uint16: return "Uint16";
    case DataType::Uint32: return "Uint32";
    case DataType::Uint64: return "Uint64";
    case DataType::Bool: return "Bool";
    case DataType::Complex64: return "Complex64";
    case DataType::Complex128: return "Complex128";
    case DataType::String: return "String";
    case DataType::Invalid: return "Invalid";
  }
  return "Invalid";
}

::reverb::tensor::DataType DataTypeToProto(DataType dt) {
  switch (dt) {
    case DataType::Float32: return ::reverb::tensor::DT_FLOAT32;
    case DataType::Float64: return ::reverb::tensor::DT_FLOAT64;
    case DataType::Int8: return ::reverb::tensor::DT_INT8;
    case DataType::Int16: return ::reverb::tensor::DT_INT16;
    case DataType::Int32: return ::reverb::tensor::DT_INT32;
    case DataType::Int64: return ::reverb::tensor::DT_INT64;
    case DataType::Uint8: return ::reverb::tensor::DT_UINT8;
    case DataType::Uint16: return ::reverb::tensor::DT_UINT16;
    case DataType::Uint32: return ::reverb::tensor::DT_UINT32;
    case DataType::Uint64: return ::reverb::tensor::DT_UINT64;
    case DataType::Bool: return ::reverb::tensor::DT_BOOL;
    case DataType::Complex64: return ::reverb::tensor::DT_COMPLEX64;
    case DataType::Complex128: return ::reverb::tensor::DT_COMPLEX128;
    case DataType::String: return ::reverb::tensor::DT_STRING;
    case DataType::Invalid: return ::reverb::tensor::DT_INVALID;
  }
  return ::reverb::tensor::DT_INVALID;
}

absl::StatusOr<DataType> DataTypeFromProto(::reverb::tensor::DataType dt) {
  switch (dt) {
    case ::reverb::tensor::DT_FLOAT32: return DataType::Float32;
    case ::reverb::tensor::DT_FLOAT64: return DataType::Float64;
    case ::reverb::tensor::DT_INT8: return DataType::Int8;
    case ::reverb::tensor::DT_INT16: return DataType::Int16;
    case ::reverb::tensor::DT_INT32: return DataType::Int32;
    case ::reverb::tensor::DT_INT64: return DataType::Int64;
    case ::reverb::tensor::DT_UINT8: return DataType::Uint8;
    case ::reverb::tensor::DT_UINT16: return DataType::Uint16;
    case ::reverb::tensor::DT_UINT32: return DataType::Uint32;
    case ::reverb::tensor::DT_UINT64: return DataType::Uint64;
    case ::reverb::tensor::DT_BOOL: return DataType::Bool;
    case ::reverb::tensor::DT_COMPLEX64: return DataType::Complex64;
    case ::reverb::tensor::DT_COMPLEX128: return DataType::Complex128;
    case ::reverb::tensor::DT_STRING: return DataType::String;
    case ::reverb::tensor::DT_INVALID:
      return absl::InvalidArgumentError("DT_INVALID in proto");
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unknown proto DataType: ", static_cast<int>(dt)));
}

int64_t TensorBuffer::NumElements() const {
  int64_t n = 1;
  for (int64_t d : spec_.shape) n *= d;
  return n;
}

int64_t TensorBuffer::TotalBytes() const {
  if (spec_.dtype == DataType::String) {
    return static_cast<int64_t>(bytes_.size());
  }
  return NumElements() * DataTypeItemsize(spec_.dtype);
}

absl::StatusOr<TensorBuffer> TensorBuffer::FromNdArray(py::object ndarray) {
  ImportNumpyOnce();
  // 强制 C-contiguous + 拷贝(不依赖 pybind11 的 py::array)。
  // 用 numpy.ascontiguousarray(pure Python 调用,不触发 pybind11 npy_api)。
  py::module np = py::module::import("numpy");
  py::object as_obj = np.attr("ascontiguousarray")(ndarray);
  PyArrayObject* contig = reinterpret_cast<PyArrayObject*>(as_obj.ptr());
  if (!PyArray_Check(contig)) {
    return absl::InvalidArgumentError(
        "FromNdArray: input is not a numpy array");
  }
  py::object holder = std::move(as_obj);  // 持有生命周期

  int type_num = PyArray_TYPE(contig);
  TensorSpec spec;
  spec.dtype = NpyTypeToDataType(type_num);
  if (spec.dtype == DataType::Invalid) {
    return absl::InvalidArgumentError(absl::StrCat(
        "FromNdArray: unsupported numpy type_num=", type_num));
  }

  int ndim = PyArray_NDIM(contig);
  npy_intp* dims = PyArray_DIMS(contig);
  spec.shape.assign(dims, dims + ndim);

  std::string bytes;
  if (spec.dtype == DataType::String) {
    // 遍历每个元素,按 [len][bytes] 编码。持 GIL。
    NpyIter* it = NpyIter_New(
        contig, NPY_ITER_READONLY | NPY_ITER_C_INDEX,
        NPY_CORDER, NPY_NO_CASTING, nullptr);
    if (!it) {
      PyErr_Clear();
      return absl::InternalError("FromNdArray: NpyIter_New failed");
    }
    NpyIter_IterNextFunc* next = NpyIter_GetIterNext(it, nullptr);
    char** dataptr = NpyIter_GetDataPtrArray(it);
    while (next(it)) {
      PyObject* item = PyArray_GETITEM(contig, *dataptr);
      if (!item) { PyErr_Clear(); NpyIter_Deallocate(it);
        return absl::InternalError("FromNdArray: PyArray_GETITEM failed"); }
      std::string s = py::str(item).cast<std::string>();
      Py_DECREF(item);
      EncodeString(&bytes, s);
    }
    NpyIter_Deallocate(it);
  } else {
    size_t nbytes = static_cast<size_t>(PyArray_NBYTES(contig));
    bytes.resize(nbytes);
    std::memcpy(bytes.data(), PyArray_DATA(contig), nbytes);
  }

  return TensorBuffer(std::move(spec), std::move(bytes));
}

py::object TensorBuffer::ToNdArray() const {
  ImportNumpyOnce();
  int npy_type = DataTypeToNpy(spec_.dtype);

  if (spec_.dtype == DataType::String) {
    // 解码每个元素,构造 list 再 np.array。
    py::module np = py::module::import("numpy");
    py::list lst;
    size_t pos = 0;
    while (pos < bytes_.size()) {
      lst.append(py::str(DecodeString(bytes_, &pos)));
    }
    return np.attr("array")(lst);
  }

  std::vector<npy_intp> dims(spec_.shape.begin(), spec_.shape.end());
  PyArrayObject* out = reinterpret_cast<PyArrayObject*>(
      PyArray_New(&PyArray_Type, static_cast<int>(dims.size()),
                  dims.empty() ? nullptr : dims.data(), npy_type,
                  nullptr, nullptr, 0, NPY_ARRAY_C_CONTIGUOUS, nullptr));
  if (!out) {
    PyErr_Clear();
    // 回退:返回 None(不应发生在数值类型)。
    return py::none();
  }
  size_t nbytes = static_cast<size_t>(NumElements()) *
                  DataTypeItemsize(spec_.dtype);
  if (nbytes > 0) {
    std::memcpy(PyArray_DATA(out), bytes_.data(), nbytes);
  }
  return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject*>(out));
}

TensorBuffer TensorBuffer::InsertBatchDim() const {
  TensorSpec spec = spec_;
  spec.shape.insert(spec.shape.begin(), 1);
  return TensorBuffer(std::move(spec), bytes_);
}

TensorBuffer TensorBuffer::RemoveBatchDim() const {
  TensorSpec spec = spec_;
  if (!spec.shape.empty()) spec.shape.erase(spec.shape.begin());
  return TensorBuffer(std::move(spec), bytes_);
}

TensorBuffer TensorBuffer::CopyReshaped(
    const std::vector<int64_t>& shape) const {
  TensorSpec spec = spec_;
  spec.shape = shape;
  // ponytail: NumElements 一致性由调用方保证;此处不校验以省一次乘法,
  // 若需防御可在上层加。
  return TensorBuffer(std::move(spec), bytes_);
}

absl::StatusOr<TensorBuffer> TensorBuffer::Concat(
    const std::vector<TensorBuffer>& buffers) {
  if (buffers.empty()) {
    return absl::InvalidArgumentError("Concat: no buffers");
  }
  DataType dt = buffers[0].dtype();
  for (const auto& b : buffers) {
    if (b.dtype() != dt) {
      return absl::InvalidArgumentError(
          "Concat: dtype mismatch");
    }
    if (b.shape().size() != buffers[0].shape().size()) {
      return absl::InvalidArgumentError("Concat: rank mismatch");
    }
    for (size_t i = 1; i < b.shape().size(); ++i) {
      if (b.shape()[i] != buffers[0].shape()[i]) {
        return absl::InvalidArgumentError(
            "Concat: non-batch dim mismatch");
      }
    }
  }

  TensorSpec spec;
  spec.dtype = dt;
  spec.shape = buffers[0].shape();
  if (spec.shape.empty()) {
    return absl::InvalidArgumentError(
        "Concat: 0-d tensor cannot concat on dim 0");
  }
  spec.shape[0] = 0;
  for (const auto& b : buffers) spec.shape[0] += b.shape()[0];

  // String: 顺序拼接编码即可(各 buffer 的编码独立)。
  std::string bytes;
  if (dt == DataType::String) {
    for (const auto& b : buffers) bytes.append(b.bytes());
    return TensorBuffer(std::move(spec), std::move(bytes));
  }

  int64_t row_size = buffers[0].TotalBytes() / buffers[0].shape()[0];
  bytes.resize(static_cast<size_t>(spec.shape[0] * row_size));
  char* dst = bytes.data();
  for (const auto& b : buffers) {
    int64_t n = b.shape()[0] * row_size;
    std::memcpy(dst, b.bytes().data(), static_cast<size_t>(n));
    dst += n;
  }
  return TensorBuffer(std::move(spec), std::move(bytes));
}

TensorBuffer TensorBuffer::SubSlice(int64_t offset) const {
  if (spec_.shape.empty()) {
    return TensorBuffer(spec_, bytes_);  // ponytail: 0-d 无 batch 维,原样返回
  }
  if (offset < 0 || offset >= spec_.shape[0]) {
    // ponytail: 越界返回空 buffer,调用方应自行校验。生产路径可改 StatusOr。
    return TensorBuffer(spec_, std::string());
  }
  int64_t row_size = (spec_.dtype == DataType::String)
                         ? 0
                         : (TotalBytes() / spec_.shape[0]);
  TensorSpec spec = spec_;
  spec.shape.erase(spec.shape.begin());

  std::string bytes;
  if (spec_.dtype == DataType::String) {
    // 跳过 offset 个元素,取 1 个元素编码。
    int64_t skip = offset;
    size_t pos = 0;
    int64_t idx = 0;
    while (pos < bytes_.size() && idx < skip) {
      DecodeString(bytes_, &pos);
      ++idx;
    }
    if (pos < bytes_.size()) {
      std::string s = DecodeString(bytes_, &pos);
      EncodeString(&bytes, s);
    }
  } else {
    size_t start = static_cast<size_t>(offset * row_size);
    bytes.assign(bytes_.data() + start, static_cast<size_t>(row_size));
  }
  return TensorBuffer(std::move(spec), std::move(bytes));
}

absl::Status TensorBuffer::SerializeToProto(
    ::reverb::tensor::TensorProto* out) const {
  out->set_dtype(DataTypeToProto(spec_.dtype));
  auto* shape = out->mutable_shape();
  shape->clear_dim();
  for (int64_t d : spec_.shape) shape->add_dim(d);
  if (spec_.dtype == DataType::String) {
    out->clear_tensor_content();
    out->clear_string_val();
    size_t pos = 0;
    while (pos < bytes_.size()) {
      *out->add_string_val() = DecodeString(bytes_, &pos);
    }
  } else {
    out->clear_string_val();
    out->set_tensor_content(bytes_);
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> TensorBuffer::DeserializeFromProto(
    const ::reverb::tensor::TensorProto& proto) {
  absl::StatusOr<DataType> dt = DataTypeFromProto(proto.dtype());
  if (!dt.ok()) return dt.status();
  TensorSpec spec;
  spec.dtype = *dt;
  spec.shape.reserve(proto.shape().dim_size());
  for (int64_t d : proto.shape().dim()) spec.shape.push_back(d);

  std::string bytes;
  if (spec.dtype == DataType::String) {
    for (const std::string& s : proto.string_val()) {
      EncodeString(&bytes, s);
    }
  } else {
    bytes = proto.tensor_content();
  }
  return TensorBuffer(std::move(spec), std::move(bytes));
}

}  // namespace reverb
}  // namespace deepmind
