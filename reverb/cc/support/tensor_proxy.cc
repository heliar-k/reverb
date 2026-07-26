#include "reverb/cc/support/tensor_proxy.h"

#include <cstring>
#include <mutex>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "numpy/arrayobject.h"
#include "pybind11/pybind11.h"

namespace deepmind {
namespace reverb {

namespace py = pybind11;

namespace {

// ponytail: 用 numpy C-API 而非 pybind11 的 py::array/py::dtype。
// pybind11 3.0.4 的 npy_api::lookup 会访问 ModuleSpec._initializing,
// 该属性 Python 3.10 没有(3.11+ 才有),嵌入 3.10 时必崩。直走 C-API 绕过。

// ponytail: 收敛 6 个 DataType switch 到一张映射表。加新 dtype 只改表一行。
// proto 枚举(::reverb::tensor::DT_*)由 protobuf 生成,是 constexpr;NPY_* 是整数宏。
struct DataTypeMapping {
  DataType dt;
  int npy_type;                   // NPY_FLOAT32 等(NPY_NOTYPE for Invalid)
  ::reverb::tensor::DataType proto_type;
  const char* name;
  int itemsize;
};
constexpr DataTypeMapping kMappings[] = {
    {DataType::Float32,    NPY_FLOAT32,    ::reverb::tensor::DT_FLOAT32,    "Float32",    4},
    {DataType::Float64,    NPY_FLOAT64,    ::reverb::tensor::DT_FLOAT64,    "Float64",    8},
    {DataType::Int8,       NPY_INT8,       ::reverb::tensor::DT_INT8,       "Int8",       1},
    {DataType::Int16,      NPY_INT16,      ::reverb::tensor::DT_INT16,      "Int16",      2},
    {DataType::Int32,      NPY_INT32,      ::reverb::tensor::DT_INT32,      "Int32",      4},
    {DataType::Int64,      NPY_INT64,      ::reverb::tensor::DT_INT64,      "Int64",      8},
    {DataType::Uint8,      NPY_UINT8,      ::reverb::tensor::DT_UINT8,      "Uint8",      1},
    {DataType::Uint16,     NPY_UINT16,     ::reverb::tensor::DT_UINT16,     "Uint16",     2},
    {DataType::Uint32,     NPY_UINT32,     ::reverb::tensor::DT_UINT32,     "Uint32",     4},
    {DataType::Uint64,     NPY_UINT64,     ::reverb::tensor::DT_UINT64,     "Uint64",     8},
    {DataType::Bool,       NPY_BOOL,       ::reverb::tensor::DT_BOOL,       "Bool",       1},
    {DataType::Complex64,  NPY_COMPLEX64,  ::reverb::tensor::DT_COMPLEX64,  "Complex64",  8},
    {DataType::Complex128, NPY_COMPLEX128, ::reverb::tensor::DT_COMPLEX128, "Complex128", 16},
    {DataType::String,     NPY_OBJECT,     ::reverb::tensor::DT_STRING,     "String",     0},
    {DataType::Invalid,    NPY_NOTYPE,     ::reverb::tensor::DT_INVALID,    "Invalid",    0},
};

DataType NpyTypeToDataType(int type_num) {
  // numpy 的 NPY_STRING/UNICODE 都映射到 String(bytes/str 变长)。
  if (type_num == NPY_STRING || type_num == NPY_UNICODE ||
      type_num == NPY_OBJECT) {
    return DataType::String;
  }
  for (const auto& m : kMappings) {
    if (m.npy_type == type_num) return m.dt;
  }
  return DataType::Invalid;
}

int DataTypeToNpy(DataType dt) {
  for (const auto& m : kMappings) {
    if (m.dt == dt) return m.npy_type;
  }
  return NPY_NOTYPE;
}

int DataTypeItemsize(DataType dt) {
  for (const auto& m : kMappings) {
    if (m.dt == dt) return m.itemsize;
  }
  return 0;
}

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

// 零拷贝视图持有的 PyObject* 不能在任意线程就地 Py_DECREF(需 GIL),
// 统一入队,由主线程在 FromNdArray/ToNdArray 入口(必持 GIL,见头注释)
// 顺带 drain。
// ponytail: 无独立 drain 线程/atexit 钩子;进程退出时队列残留随进程回收。
std::mutex g_deferred_free_mu;
std::vector<PyObject*> g_deferred_free;

void DeferredFree(PyObject* obj) {
  std::lock_guard<std::mutex> l(g_deferred_free_mu);
  g_deferred_free.push_back(obj);
}

void DrainDeferredFree() {
  std::vector<PyObject*> batch;
  {
    std::lock_guard<std::mutex> l(g_deferred_free_mu);
    if (g_deferred_free.empty()) return;
    batch.swap(g_deferred_free);
  }
  for (PyObject* o : batch) Py_DECREF(o);
}

}  // namespace

TensorBuffer::TensorBuffer(TensorSpec spec, std::string bytes)
    : spec_(std::move(spec)) {
  auto s = std::make_shared<std::string>(std::move(bytes));
  bytes_ = absl::string_view(*s);
  owner_ = std::move(s);
}

TensorBuffer::TensorBuffer(TensorSpec spec, std::shared_ptr<void> owner,
                           absl::string_view bytes)
    : spec_(std::move(spec)), owner_(std::move(owner)), bytes_(bytes) {}

const char* DataTypeName(DataType dt) {
  for (const auto& m : kMappings) {
    if (m.dt == dt) return m.name;
  }
  return "Invalid";
}

::reverb::tensor::DataType DataTypeToProto(DataType dt) {
  for (const auto& m : kMappings) {
    if (m.dt == dt) return m.proto_type;
  }
  return ::reverb::tensor::DT_INVALID;
}

absl::StatusOr<DataType> DataTypeFromProto(::reverb::tensor::DataType dt) {
  if (dt == ::reverb::tensor::DT_INVALID) {
    return absl::InvalidArgumentError("DT_INVALID in proto");
  }
  for (const auto& m : kMappings) {
    if (m.proto_type == dt) return m.dt;
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

absl::StatusOr<TensorBuffer> TensorBuffer::FromNdArray(py::object ndarray,
                                                       bool zero_copy) {
  ImportNumpyOnce();
  DrainDeferredFree();
  // 保留输入维度:np.asarray 不提升 Python 标量(0-d 保持 0-d),
  // 而 np.ascontiguousarray 会把标量提升为 [1]——那是标量列 shape
  // 退化为 [N,1] 的根因。仅在非 0-d 且非 C-contiguous 时转连续。
  // ponytail: 纯 Python 调用,绕过 pybind11 npy_api(见文件顶部注释)。
  py::module np = py::module::import("numpy");
  py::object as_obj = np.attr("asarray")(ndarray);
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(as_obj.ptr());
  if (!PyArray_Check(arr)) {
    return absl::InvalidArgumentError(
        "FromNdArray: input is not a numpy array");
  }
  // 0-d 必然 C-contiguous;非 0-d 不连续时升为 C-contiguous(拷贝)。
  if (PyArray_NDIM(arr) > 0 && !PyArray_ISCARRAY_RO(arr)) {
    as_obj = np.attr("ascontiguousarray")(as_obj);
    arr = reinterpret_cast<PyArrayObject*>(as_obj.ptr());
  }
  PyArrayObject* contig = arr;
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
    // NpyIter 初始已停在元素 0:必须 do-while(先处理后推进),
    // while(next) 会跳过首元素,单元素数组则编出空 bytes。
    if (PyArray_SIZE(contig) == 0) {
      NpyIter_Deallocate(it);
      return TensorBuffer(std::move(spec), std::move(bytes));
    }
    do {
      PyObject* item = PyArray_GETITEM(contig, *dataptr);
      if (!item) { PyErr_Clear(); NpyIter_Deallocate(it);
        return absl::InternalError("FromNdArray: PyArray_GETITEM failed"); }
      // S-dtype 元素是 PyBytes(np.bytes_ 继承 bytes),必须逐字节拷贝;
      // py::str(item) 会存成 repr "b'abc'" —— 静默损坏。U-dtype 走
      // py::str 得到 UTF-8。
      std::string s;
      if (PyBytes_Check(item)) {
        s.assign(PyBytes_AS_STRING(item), PyBytes_GET_SIZE(item));
      } else {
        s = py::str(item).cast<std::string>();
      }
      Py_DECREF(item);
      EncodeString(&bytes, s);
    } while (next(it));
    NpyIter_Deallocate(it);
  } else if (zero_copy) {
    // 视图 numpy 存储:holder 的引用转由 TensorBuffer 持有,任意线程析构时
    // PyObject* 入 DeferredFreeQueue(见上)。read-only guard:append 后原地
    // 改写立刻报错而非静默写脏(零拷贝丢失快照语义,见头文件类注释)。
    PyArray_CLEARFLAGS(contig, NPY_ARRAY_WRITEABLE);
    PyObject* raw = holder.ptr();
    Py_INCREF(raw);
    std::shared_ptr<void> owner(static_cast<void*>(raw), [](void* p) {
      DeferredFree(static_cast<PyObject*>(p));
    });
    size_t nbytes = static_cast<size_t>(PyArray_NBYTES(contig));
    absl::string_view view(static_cast<const char*>(PyArray_DATA(contig)),
                           nbytes);
    return TensorBuffer(std::move(spec), std::move(owner), view);
  } else {
    size_t nbytes = static_cast<size_t>(PyArray_NBYTES(contig));
    bytes.resize(nbytes);
    std::memcpy(bytes.data(), PyArray_DATA(contig), nbytes);
  }

  return TensorBuffer(std::move(spec), std::move(bytes));
}

py::object TensorBuffer::ToNdArray() const {
  ImportNumpyOnce();
  DrainDeferredFree();
  int npy_type = DataTypeToNpy(spec_.dtype);

  if (spec_.dtype == DataType::String) {
    // 解码每个元素,构造 list 再 np.array。
    py::module np = py::module::import("numpy");
    py::list lst;
    size_t pos = 0;
    while (pos < bytes_.size()) {
      // 解码为 bytes:线格式不区分 S/U,取 TF 语义 string==bytes,
      // 与写侧 PyBytes 逐字节拷贝自洽(S 数组 dtype 往返一致)。
      // ponytail: U-dtype 数组读回变成 S(dtype 变化,值为 UTF-8 字节);
      // 升级路径 = 线格式加 dtype 标记。
      lst.append(py::bytes(DecodeString(bytes_, &pos)));
    }
    // reshape 回 spec_.shape:否则 [T,2] 字符串列塌成 [T*2],
    // 与数值路径(ToNdArray 用 spec_.shape 建数组)不一致。
    // (无 pybind11/stl.h,手动构 tuple)
    py::tuple dims(spec_.shape.size());
    for (size_t i = 0; i < spec_.shape.size(); ++i) dims[i] = spec_.shape[i];
    return np.attr("array")(lst).attr("reshape")(dims);
  }

  std::vector<npy_intp> dims(spec_.shape.begin(), spec_.shape.end());
  size_t nbytes = static_cast<size_t>(NumElements()) *
                  DataTypeItemsize(spec_.dtype);
  if (nbytes == 0) {
    // 空数组无字节可视图,走分配路径(data 不会被读)。
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(
        static_cast<int>(dims.size()),
        dims.empty() ? nullptr : dims.data(), npy_type));
    if (!out) {
      PyErr_Clear();
      return py::none();
    }
    return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject*>(out));
  }
  // 零拷贝:视图 TensorBuffer 的字节存储,capsule 持 owner_ 副本使数组
  // 独立于源对象存活。const_cast:数组可写,与旧 SimpleNew 路径行为一致。
  // ponytail: SimpleNewFromData 与 SimpleNew 同语义,strides 按 C-order
  // 计算;勿换成 PyArray_New + C_CONTIGUOUS(ndim>=2 误产 F-order strides)。
  PyArrayObject* out = reinterpret_cast<PyArrayObject*>(
      PyArray_SimpleNewFromData(static_cast<int>(dims.size()),
                                dims.empty() ? nullptr : dims.data(),
                                npy_type,
                                const_cast<char*>(bytes_.data())));
  if (!out) {
    PyErr_Clear();
    return py::none();
  }
  auto* holder = new std::shared_ptr<void>(owner_);
  py::capsule cap(holder, [](void* p) {
    delete static_cast<std::shared_ptr<void>*>(p);
  });
  if (PyArray_SetBaseObject(out, cap.ptr()) < 0) {
    PyErr_Clear();
    delete holder;
    Py_DECREF(out);
    return py::none();
  }
  cap.release();  // SetBaseObject 已窃取引用
  return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject*>(out));
}

TensorBuffer TensorBuffer::InsertBatchDim() const {
  TensorSpec spec = spec_;
  // ponytail: 标量(0-d)不插 batch 维——TF 期标量跨 N 步堆叠为 [N]
  // 而非 [N,1]。非标量照常插 [1,...]。这把标量路径与 [1] 路径分流,
  // 避免采样产出 [N,1]。
  if (!spec.shape.empty()) {
    spec.shape.insert(spec.shape.begin(), 1);
  }
  return TensorBuffer(std::move(spec), owner_, bytes_);
}

TensorBuffer TensorBuffer::RemoveBatchDim() const {
  TensorSpec spec = spec_;
  if (!spec.shape.empty()) spec.shape.erase(spec.shape.begin());
  return TensorBuffer(std::move(spec), owner_, bytes_);
}

TensorBuffer TensorBuffer::CopyReshaped(
    const std::vector<int64_t>& shape) const {
  TensorSpec spec = spec_;
  spec.shape = shape;
  // ponytail: NumElements 一致性由调用方保证;此处不校验以省一次乘法,
  // 若需防御可在上层加。
  return TensorBuffer(std::move(spec), owner_, bytes_);
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
    // 0-d 标量堆叠:N 个 {} 拼成 [N](TF 期标量跨步语义)。
    // ponytail: 每个标量 1 个元素,顺序拼接即 [N]。String 同理。
    spec.shape = {static_cast<int64_t>(buffers.size())};
    std::string bytes;
    if (dt == DataType::String) {
      for (const auto& b : buffers) bytes.append(b.bytes());
    } else {
      int itemsize = DataTypeItemsize(dt);
      bytes.resize(buffers.size() * itemsize);
      char* dst = bytes.data();
      for (const auto& b : buffers) {
        std::memcpy(dst, b.bytes().data(), itemsize);
        dst += itemsize;
      }
    }
    return TensorBuffer(std::move(spec), std::move(bytes));
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
    // ponytail: 0-d 无 batch 维,原样返回(共享存储)。
    return TensorBuffer(spec_, owner_, bytes_);
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

  if (spec_.dtype == DataType::String) {
    std::string bytes;
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
    return TensorBuffer(std::move(spec), std::move(bytes));
  } else {
    // 视图共享宿主:去 batch 维只是偏移+shape 变化,无需物化拷贝。
    size_t start = static_cast<size_t>(offset * row_size);
    return TensorBuffer(std::move(spec), owner_,
                        bytes_.substr(start, static_cast<size_t>(row_size)));
  }
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
    // Validate length BEFORE anyone reads it: ToNdArray memcpys
    // NumElements*itemsize bytes out of bytes_ — a truncated/corrupt proto
    // must fail here, not over-read the heap (old TF Tensor::FromProto
    // CHECK-failed; we return a status).
    int64_t numel = 1;
    for (int64_t d : spec.shape) {
      if (d < 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "DeserializeFromProto: negative dim ", d));
      }
      numel *= d;
    }
    size_t expected =
        static_cast<size_t>(numel) * DataTypeItemsize(spec.dtype);
    if (bytes.size() != expected) {
      return absl::InvalidArgumentError(absl::StrCat(
          "DeserializeFromProto: tensor_content is ", bytes.size(),
          " bytes, shape*dtype implies ", expected));
    }
  }
  return TensorBuffer(std::move(spec), std::move(bytes));
}

}  // namespace reverb
}  // namespace deepmind
