// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "numpy/arrayobject.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/client.h"
#include "reverb/cc/in_process_client.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/platform/checkpointing.h"
#include "reverb/cc/platform/checkpointing_utils.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/server.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/selectors/lifo.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"
#include "reverb/cc/trajectory_writer.h"
#include "reverb/cc/writer.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace {

// Lazily fetches and caches reverb.errors.DeadlineExceededError so C++ statuses
// with kDeadlineExceeded surface as the project's own exception type.
// Lazy (not module-init): reverb.errors may not be imported when libpybind
// inits; this is an error path so the import cost is irrelevant.
// Thread-safe via C++11 magic statics; called with the GIL held.
PyObject* DeadlineExceededPyExc() {
  static PyObject* cls = []() -> PyObject* {
    pybind11::object obj = pybind11::module::import("reverb.errors")
                               .attr("DeadlineExceededError");
    return obj.inc_ref().ptr();
  }();
  return cls;
}

// Converts non OK statuses to Python exceptions and throws. Does nothing for
// OK statuses.
inline void MaybeRaiseFromStatus(const absl::Status& status) {
  if (status.ok()) return;

  switch (status.code()) {
#define CODE_TO_PY_EXC(CODE, PY_EXC)                         \
  case CODE:                                                 \
    PyErr_SetString(PY_EXC, std::string(status.message()).data()); \
    break;

    CODE_TO_PY_EXC(absl::StatusCode::kInvalidArgument, PyExc_ValueError)
    CODE_TO_PY_EXC(absl::StatusCode::kResourceExhausted, PyExc_IndexError)
    CODE_TO_PY_EXC(absl::StatusCode::kDeadlineExceeded, DeadlineExceededPyExc())
    CODE_TO_PY_EXC(absl::StatusCode::kNotFound, PyExc_FileNotFoundError)
    CODE_TO_PY_EXC(absl::StatusCode::kUnimplemented, PyExc_NotImplementedError)
    CODE_TO_PY_EXC(absl::StatusCode::kInternal, PyExc_RuntimeError)

#undef CODE_TO_PY_EXC

    default:
      PyErr_SetString(PyExc_RuntimeError, std::string(status.message()).data());
  }

  throw pybind11::error_already_set();
}

// Maps a Reverb `DataType` (numpy-aligned) to a numpy dtype string understood
// by `py::dtype`. Replaces the historical TF-backed `conversions` target
// (now deleted) which depended on `tensorflow::Tensor`.
// ponytail: 如果新增 DataType,这里同步加一行。
const char* DataTypeToNumpyString(::deepmind::reverb::DataType dt) {
  using ::deepmind::reverb::DataType;
  switch (dt) {
    case DataType::Float32: return "float32";
    case DataType::Float64: return "float64";
    case DataType::Int8: return "int8";
    case DataType::Int16: return "int16";
    case DataType::Int32: return "int32";
    case DataType::Int64: return "int64";
    case DataType::Uint8: return "uint8";
    case DataType::Uint16: return "uint16";
    case DataType::Uint32: return "uint32";
    case DataType::Uint64: return "uint64";
    case DataType::Bool: return "bool";
    case DataType::Complex64: return "complex64";
    case DataType::Complex128: return "complex128";
    case DataType::String: return "object";
    case DataType::Invalid: return "object";
  }
  return "object";
}

// This wrapper exists for the sole purpose of allowing the weak_ptr to be
// handled in Python. Pybind supports shared_ptr and unique_ptr out of the box
// and although it is possible to implement our own `SmartPointer, using a
// minimal wrapper class like WeakCellRef is much simpler when the weak_ptr
// is only required for one class (in Python).
class WeakCellRef {
 public:
  explicit WeakCellRef(std::weak_ptr<::deepmind::reverb::CellRef> ref)
      : ref_(std::move(ref)) {}

  std::weak_ptr<::deepmind::reverb::CellRef> ref() const { return ref_; }

  bool expired() const { return ref_.expired(); }

 private:
  std::weak_ptr<::deepmind::reverb::CellRef> ref_;
};

}  // namespace

namespace pybind11 {
namespace detail {

// Convert between absl::optional and python.
//
// pybind11 supports std::optional, and absl::optional is meant to be a
// drop-in replacement for std::optional, so we can just use the built in
// implementation.
#ifndef ABSL_USES_STD_OPTIONAL
template <typename T>
struct type_caster<absl::optional<T>>
    : public optional_caster<absl::optional<T>> {};

template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

// Automatic conversion between Python numpy arrays and Reverb `TensorBuffer`.
// Python passes an ndarray -> `FromNdArray` builds a `TensorBuffer`. C++ returns
// a `TensorBuffer` -> `ToNdArray` produces an ndarray. Replaces the old
// `type_caster<tensorflow::Tensor>` so the higher-level bindings
// (`Sampler`, `TrajectoryWriter`, `WeakCellRef`) need no per-call glue.
template <>
struct type_caster<::deepmind::reverb::TensorBuffer> {
 public:
  PYBIND11_TYPE_CASTER(::deepmind::reverb::TensorBuffer,
                       _("reverb.TensorBuffer"));

  bool load(handle src, bool) {
    auto buf = ::deepmind::reverb::TensorBuffer::FromNdArray(
        pybind11::reinterpret_borrow<pybind11::object>(src));
    if (!buf.ok()) {
      REVERB_LOG(REVERB_ERROR)
          << "TensorBuffer can't be extracted from the source ndarray: "
          << buf.status().ToString();
      PyErr_Clear();
      return false;
    }
    value = std::move(*buf);
    return true;
  }

  static handle cast(const ::deepmind::reverb::TensorBuffer& src,
                     return_value_policy, handle) {
    return src.ToNdArray().release().ptr();
  }
};

// Raise an exception if a given status is not OK, otherwise return None.
template <>
struct type_caster<absl::Status> {
 public:
  PYBIND11_TYPE_CASTER(absl::Status, _("Status"));
  static handle cast(absl::Status status, return_value_policy, handle) {
    MaybeRaiseFromStatus(status);
    return none().inc_ref();
  }
};

}  // namespace detail
}  // namespace pybind11

// LINT.IfChange
namespace deepmind {
namespace reverb {
namespace {

namespace py = pybind11;

PYBIND11_MODULE(libpybind, m) {
  // numpy C-API import; must run once after the interpreter is up.
  if (_import_array() < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    throw py::error_already_set();
  }

  py::class_<ItemSelector, std::shared_ptr<ItemSelector>>(m, "ItemSelector")
      .def("__repr__", &ItemSelector::DebugString,
           py::call_guard<py::gil_scoped_release>());

  py::class_<PrioritizedSelector, ItemSelector,
             std::shared_ptr<PrioritizedSelector>>(m, "PrioritizedSelector")
      .def(py::init<double>(), py::arg("priority_exponent"));

  py::class_<FifoSelector, ItemSelector, std::shared_ptr<FifoSelector>>(
      m, "FifoSelector")
      .def(py::init());

  py::class_<LifoSelector, ItemSelector, std::shared_ptr<LifoSelector>>(
      m, "LifoSelector")
      .def(py::init());

  py::class_<UniformSelector, ItemSelector, std::shared_ptr<UniformSelector>>(
      m, "UniformSelector")
      .def(py::init());

  py::class_<HeapSelector, ItemSelector, std::shared_ptr<HeapSelector>>(
      m, "HeapSelector")
      .def(py::init<bool>(), py::arg("min_heap"));

  m.def(
      "selector_from_proto",
      [](const std::string& options_str) {
        KeyDistributionOptions options;
        deepmind::reverb::ItemSelector* result = nullptr;
        if (!options.ParseFromString(options_str)) {
          MaybeRaiseFromStatus(absl::InvalidArgumentError(absl::StrCat(
              "Unable to deserialize KeyDistributionOptions from serialized "
              "proto bytes: '", options_str, "'")));
        } else {
          result = MakeSelector(options).release();
        }
        return result;
      });

  py::class_<TableExtension, std::shared_ptr<TableExtension>>(m,
                                                              "TableExtension")
      .def("__repr__", &TableExtension::DebugString,
           py::call_guard<py::gil_scoped_release>());

  py::class_<RateLimiter, std::shared_ptr<RateLimiter>>(m, "RateLimiter")
      .def(py::init<double, int, double, double>(),
           py::arg("samples_per_insert"), py::arg("min_size_to_sample"),
           py::arg("min_diff"), py::arg("max_diff"))
      .def("__repr__", &RateLimiter::DebugString,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Table, std::shared_ptr<Table>>(m, "Table")
      .def(py::init(
               [](const std::string& name,
                  const std::shared_ptr<ItemSelector>& sampler,
                  const std::shared_ptr<ItemSelector>& remover, int max_size,
                  int max_times_sampled,
                  const std::shared_ptr<RateLimiter>& rate_limiter,
                  const std::vector<std::shared_ptr<TableExtension>>&
                      extensions,
                  const std::optional<std::string>& serialized_signature =
                      std::nullopt) -> Table* {
                 absl::optional<::reverb::tensor::SignatureProto> signature =
                     absl::nullopt;
                 if (serialized_signature) {
                   signature.emplace();
                   if (!signature->ParseFromString(*serialized_signature)) {
                     MaybeRaiseFromStatus(
                         absl::InvalidArgumentError(absl::StrCat(
                             "Unable to deserialize SignatureProto from "
                             "serialized proto bytes: '",
                             *serialized_signature, "'")));
                     return nullptr;
                   }
                 }
                 return new Table(name, sampler, remover, max_size,
                                  max_times_sampled, rate_limiter, extensions,
                                  std::move(signature));
               }),
           py::arg("name"), py::arg("sampler"), py::arg("remover"),
           py::arg("max_size"), py::arg("max_times_sampled"),
           py::arg("rate_limiter"), py::arg("extensions"), py::arg("signature"))
      .def("name", &Table::name)
      .def("can_sample", &Table::CanSample,
           py::call_guard<py::gil_scoped_release>())
      .def("can_insert", &Table::CanInsert,
           py::call_guard<py::gil_scoped_release>())
      .def("info",
           [](Table* table) -> py::bytes {
             // Return a serialized TableInfo proto bytes string.
             std::string table_info;
             {
               py::gil_scoped_release g;
               table_info = table->info().SerializeAsString();
             }
             return py::bytes(table_info);
           })
      .def("__repr__", &Table::DebugString,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Sampler>(m, "Sampler")
      .def("GetNextTrajectory",
           [](Sampler *sampler) {
             absl::Status status;
             std::shared_ptr<const SampleInfo> info;
             std::vector<TensorBuffer> data;

             {
               py::gil_scoped_release g;
               status = sampler->GetNextTrajectory(&data, &info);
             }

             MaybeRaiseFromStatus(status);
             return Sampler::WithInfoTensors(*info, std::move(data));
           })
      .def_property_readonly_static("NUM_INFO_TENSORS", [](py::object) {
        return Sampler::kNumInfoTensors;
      });

  // gRPC-backed `Writer` (see reverb/cc/writer.h). The data interface is now
  // numpy-backed: `Append`/`AppendSequence` take flattened `TensorBuffer`
  // vectors (auto-converted from Python ndarrays by `type_caster<TensorBuffer>`).
  py::class_<Writer>(m, "Writer")
      .def("Append",
           [](Writer *writer, std::vector<TensorBuffer> data) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = writer->Append(std::move(data));
             }
             MaybeRaiseFromStatus(status);
           })
      .def("AppendSequence",
           [](Writer *writer, std::vector<TensorBuffer> sequence) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = writer->AppendSequence(std::move(sequence));
             }
             MaybeRaiseFromStatus(status);
           })
      .def("CreateItem", &Writer::CreateItem,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "Flush",
          [](Writer *writer) {
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = writer->Flush();
            }
            MaybeRaiseFromStatus(status);
          })
      .def("Close", &Writer::Close, py::arg("retry_on_unavailable") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", &Writer::DebugString,
           py::call_guard<py::gil_scoped_release>());

  // gRPC-backed `Client` (see reverb/cc/client.h). Connects to a Reverb
  // `Server` over gRPC. The numpy-only in-process mode uses `InProcessClient`.
  py::class_<Client>(m, "Client")
      .def(py::init<std::string>(), py::arg("server_name"))
      .def(
          "NewWriter",
          [](Client* client, int chunk_length, int max_timesteps,
             bool delta_encoded, int max_in_flight_items) {
            std::unique_ptr<Writer> writer;
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = client->NewWriter(
                  chunk_length, max_timesteps, delta_encoded,
                  max_in_flight_items, &writer);
            }
            MaybeRaiseFromStatus(status);
            return writer;
          },
          py::arg("chunk_length"), py::arg("max_timesteps"),
          py::arg("delta_encoded") = false, py::arg("max_in_flight_items"))
      .def("NewSampler",
           [](Client* client, const std::string& table, int64_t max_samples,
              size_t buffer_size) {
             std::unique_ptr<Sampler> sampler;
             Sampler::Options options;
             options.max_samples = max_samples;
             options.max_in_flight_samples_per_worker = buffer_size;
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->NewSamplerWithoutSignatureCheck(table, options,
                                                                &sampler);
             }
             MaybeRaiseFromStatus(status);
             return sampler;
           })
      .def("NewTrajectoryWriter",
           [](Client* client, std::shared_ptr<ChunkerOptions> chunker_options,
              bool validate_items) {
             std::unique_ptr<TrajectoryWriter> writer;
             TrajectoryWriter::Options options;
             options.chunker_options = std::move(chunker_options);
             absl::Status status;
             if (validate_items) {
               py::gil_scoped_release g;
               status = client->NewTrajectoryWriter(
                   options, absl::InfiniteDuration(), &writer);
             } else {
               status = client->NewTrajectoryWriter(options, &writer);
             }
             MaybeRaiseFromStatus(status);
             return writer.release();
           })
      .def("NewStructuredWriter",
           [](Client* client, std::vector<std::string> serialized_configs)
               -> StructuredWriter* {
             std::vector<StructuredWriterConfig> configs;
             for (const auto &serialised_config : serialized_configs) {
               configs.emplace_back();
               if (!configs.back().ParseFromString(
                       std::string(serialised_config))) {
                 MaybeRaiseFromStatus(absl::InvalidArgumentError(absl::StrCat(
                     "Unable to deserialize StructuredWriterConfig from "
                     "serialized proto bytes: '",
                     std::string(serialised_config), "'")));
                 return nullptr;
               }
             }
             std::unique_ptr<StructuredWriter> writer;
             absl::Status status;
             {
               py::gil_scoped_release g;
               status =
                   client->NewStructuredWriter(std::move(configs), &writer);
             }
             if (!status.ok()) {
               MaybeRaiseFromStatus(status);
               return nullptr;
             }
             return writer.release();
           })
      .def(
          "MutatePriorities",
          [](Client* client, const std::string& table,
             const std::vector<std::pair<uint64_t, double>>& updates,
             const std::vector<uint64_t>& deletes) {
            std::vector<KeyWithPriority> update_protos;
            for (const auto &update : updates) {
              update_protos.emplace_back();
              update_protos.back().set_key(update.first);
              update_protos.back().set_priority(update.second);
            }
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = client->MutatePriorities(table, update_protos, deletes);
            }
            MaybeRaiseFromStatus(status);
          },
          py::arg("table"), py::arg("updates"), py::arg("deletes"))
      .def("Reset", [](Client* client, const std::string& table) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->Reset(table);
             }
             MaybeRaiseFromStatus(status);
           },
           py::arg("table"))
      .def("ServerInfo",
           [](Client* client, int timeout_sec) {
             auto timeout = timeout_sec > 0 ? absl::Seconds(timeout_sec)
                                            : absl::InfiniteDuration();
             struct Client::ServerInfo info;
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->ServerInfo(timeout, &info);
             }
             MaybeRaiseFromStatus(status);
             std::vector<py::bytes> serialized_table_info;
             serialized_table_info.reserve(info.table_info.size());
             for (const auto &table_info : info.table_info) {
               serialized_table_info.push_back(
                   py::bytes(table_info.SerializeAsString()));
             }
             return serialized_table_info;
           })
      .def("Checkpoint", [](Client* client) {
        std::string path;
        absl::Status status;
        {
          py::gil_scoped_release g;
          status = client->Checkpoint(&path);
        }
        MaybeRaiseFromStatus(status);
        return path;
      });

  py::class_<Checkpointer, std::shared_ptr<Checkpointer>>(m, "Checkpointer")
      .def("__repr__", &Checkpointer::DebugString,
           py::call_guard<py::gil_scoped_release>());

  m.def(
      "create_default_checkpointer",
      [](const std::string& name, const std::string& group,
         std::optional<std::string> fallback_checkpoint_path) {
        auto checkpointer = CreateDefaultCheckpointer(
            name, group, std::move(fallback_checkpoint_path));
        return std::shared_ptr<Checkpointer>(checkpointer.release());
      },
      py::call_guard<py::gil_scoped_release>());

  py::class_<Server, std::shared_ptr<Server>>(m, "Server")
      .def(
          py::init([](std::vector<std::shared_ptr<Table>> priority_tables,
                      int port,
                      std::shared_ptr<Checkpointer> checkpointer = nullptr) {
            std::unique_ptr<Server> server;
            MaybeRaiseFromStatus(StartServer(std::move(priority_tables), port,
                                             std::move(checkpointer), &server));
            return server.release();
          }),
          py::arg("priority_tables"), py::arg("port"),
          py::arg("checkpointer") = nullptr)
      .def("Stop", &Server::Stop, py::call_guard<py::gil_scoped_release>())
      .def("Wait", &Server::Wait, py::call_guard<py::gil_scoped_release>())
      .def("__repr__", &Server::DebugString,
           py::call_guard<py::gil_scoped_release>());

  py::class_<WeakCellRef, std::shared_ptr<WeakCellRef>>(m, "WeakCellRef")
      .def_property_readonly("expired", &WeakCellRef::expired)
      .def("numpy",
           [](WeakCellRef* ref) -> TensorBuffer {
             TensorBuffer buffer;

             auto sp = ref->ref().lock();
             if (!sp) {
               MaybeRaiseFromStatus(absl::FailedPreconditionError(
                   "Cannot access data from expired WeakCellRef"));
               return buffer;
             }

             absl::Status status;
             {
               py::gil_scoped_release g;
               status = sp->GetData(&buffer);
             }
             MaybeRaiseFromStatus(status);

             return buffer;
           })
      .def_property_readonly(
          "shape",
          [](WeakCellRef* ref) -> std::vector<std::optional<int>> {
            std::vector<std::optional<int>> out_shape;

            auto sp = ref->ref().lock();
            if (!sp) {
              MaybeRaiseFromStatus(absl::FailedPreconditionError(
                  "Cannot access data from expired WeakCellRef"));
              return out_shape;
            }

            absl::Status status;
            {
              py::gil_scoped_release g;
              internal::TensorSpec spec;
              status = sp->GetSpec(&spec);
              out_shape.reserve(spec.shape.size());
              for (auto dim : spec.shape) {
                // Replace -1 with absl::nullopt because the Python API uses
                // None instead of -1 to represent unknown dimensions.
                out_shape.push_back(dim == -1 ? std::nullopt
                                              : std::make_optional(
                                                    static_cast<int>(dim)));
              }
            }
            MaybeRaiseFromStatus(status);

            return out_shape;
          })
      .def_property_readonly(
          "dtype", [](WeakCellRef* ref) -> py::dtype {
            auto sp = ref->ref().lock();
            if (!sp) {
              MaybeRaiseFromStatus(absl::FailedPreconditionError(
                  "Cannot access data from expired WeakCellRef"));
              return py::dtype();
            }

            absl::Status status;
            internal::TensorSpec spec;
            {
              py::gil_scoped_release g;
              status = sp->GetSpec(&spec);
            }
            MaybeRaiseFromStatus(status);
            return py::dtype(DataTypeToNumpyString(spec.dtype));
          });

  py::class_<ChunkerOptions, std::shared_ptr<ChunkerOptions>>(m,
                                                              "ChunkerOptions");

  py::class_<ConstantChunkerOptions, ChunkerOptions,
             std::shared_ptr<ConstantChunkerOptions>>(m,
                                                      "ConstantChunkerOptions")
      .def(py::init<int, int>(), py::arg("max_chunk_length"),
           py::arg("num_keep_alive_refs"))
      .def("__eq__", [](ConstantChunkerOptions *self,
                        std::shared_ptr<ConstantChunkerOptions> other) {
        return self->GetMaxChunkLength() == other->GetMaxChunkLength() &&
               self->GetNumKeepAliveRefs() == other->GetNumKeepAliveRefs();
      });

  py::class_<AutoTunedChunkerOptions, ChunkerOptions,
             std::shared_ptr<AutoTunedChunkerOptions>>(
      m, "AutoTunedChunkerOptions")
      .def(py::init<int, double>(), py::arg("num_keep_alive_refs"),
           py::arg("throughput_weight"))
      .def("__eq__", [](AutoTunedChunkerOptions *self,
                        std::shared_ptr<AutoTunedChunkerOptions> other) {
        return self->GetNumKeepAliveRefs() == other->GetNumKeepAliveRefs();
      });

  py::class_<TrajectoryWriter, std::shared_ptr<TrajectoryWriter>>(
      m, "TrajectoryWriter")
      .def(
          "Append",
          [](TrajectoryWriter* writer,
             std::vector<std::optional<TensorBuffer>> data) {
            std::vector<std::optional<std::weak_ptr<CellRef>>> refs;
            MaybeRaiseFromStatus(writer->Append(std::move(data), &refs));

            std::vector<absl::optional<std::shared_ptr<WeakCellRef>>> weak_refs(
                refs.size());
            for (int i = 0; i < refs.size(); i++) {
              if (refs[i].has_value()) {
                weak_refs[i] =
                    std::make_shared<WeakCellRef>(std::move(refs[i].value()));
              } else {
                weak_refs[i] = std::nullopt;
              }
            }

            return weak_refs;
          })
      .def(
          "AppendPartial",
          [](TrajectoryWriter* writer,
             std::vector<std::optional<TensorBuffer>> data) {
            std::vector<std::optional<std::weak_ptr<CellRef>>> refs;
            MaybeRaiseFromStatus(writer->AppendPartial(std::move(data), &refs));

            std::vector<absl::optional<std::shared_ptr<WeakCellRef>>> weak_refs(
                refs.size());
            for (int i = 0; i < refs.size(); i++) {
              if (refs[i].has_value()) {
                weak_refs[i] =
                    std::make_shared<WeakCellRef>(std::move(refs[i].value()));
              } else {
                weak_refs[i] = std::nullopt;
              }
            }

            return weak_refs;
          })
      .def(
          "CreateItem",
          [](TrajectoryWriter* writer, const std::string& table,
             double priority,
             std::vector<std::vector<std::shared_ptr<WeakCellRef>>>
                 py_trajectory,
             std::vector<bool> squeeze_column) {
            if (py_trajectory.size() != squeeze_column.size()) {
              MaybeRaiseFromStatus(absl::InternalError(
                  "Length of py_trajectory and squeeze_column did not match."));
              return;
            }

            std::vector<TrajectoryColumn> trajectory;
            trajectory.reserve(py_trajectory.size());
            for (int i = 0; i < py_trajectory.size(); i++) {
              auto &py_column = py_trajectory[i];
              std::vector<std::weak_ptr<CellRef>> column;
              column.reserve(py_column.size());
              for (auto &weak_ref : py_column) {
                column.push_back(weak_ref->ref());
              }
              trajectory.push_back(
                  TrajectoryColumn(std::move(column), squeeze_column[i]));
            }
            MaybeRaiseFromStatus(
                writer->CreateItem(table, priority, trajectory));
          })
      .def("Flush",
           [](TrajectoryWriter* writer, int ignore_last_num_items,
              int timeout_ms) {
             absl::Status status;
             auto timeout = timeout_ms > 0 ? absl::Milliseconds(timeout_ms)
                                           : absl::InfiniteDuration();
             {
               py::gil_scoped_release g;
               status = writer->Flush(ignore_last_num_items, timeout);
             }
             MaybeRaiseFromStatus(status);
           })
      .def("EndEpisode",
           [](TrajectoryWriter* writer, bool clear_buffers,
              std::optional<int> timeout_ms) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = writer->EndEpisode(
                   clear_buffers, timeout_ms.has_value()
                                      ? absl::Milliseconds(timeout_ms.value())
                                      : absl::InfiniteDuration());
             }
             MaybeRaiseFromStatus(status);
           })
      .def("Close", &TrajectoryWriter::Close,
           py::call_guard<py::gil_scoped_release>())
      .def("ConfigureChunker", &TrajectoryWriter::ConfigureChunker,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("max_num_keep_alive_refs",
                             &TrajectoryWriter::max_num_keep_alive_refs)
      .def_property_readonly("episode_steps", &TrajectoryWriter::episode_steps,
                             py::call_guard<py::gil_scoped_release>());

  // `StructuredWriter`: wraps a `TrajectoryWriter` and applies
  // `StructuredWriterConfig` patterns. Created via `Client` (gRPC) or
  // `InProcessClient` (in-process).
  py::class_<StructuredWriter, std::shared_ptr<StructuredWriter>>(
      m, "StructuredWriter")
      .def(
          "Append",
          [](StructuredWriter* writer,
             std::vector<std::optional<TensorBuffer>> data) {
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = writer->Append(std::move(data));
            }
            MaybeRaiseFromStatus(status);
          })
      .def(
          "AppendPartial",
          [](StructuredWriter* writer,
             std::vector<std::optional<TensorBuffer>> data) {
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = writer->AppendPartial(std::move(data));
            }
            MaybeRaiseFromStatus(status);
          })
      .def("Flush",
           [](StructuredWriter* writer, int ignore_last_num_items,
              int timeout_ms) {
             absl::Status status;
             auto timeout = timeout_ms > 0 ? absl::Milliseconds(timeout_ms)
                                           : absl::InfiniteDuration();
             {
               py::gil_scoped_release g;
               status = writer->Flush(ignore_last_num_items, timeout);
             }
             MaybeRaiseFromStatus(status);
           })
      .def("EndEpisode",
           [](StructuredWriter* writer, bool clear_buffers,
              std::optional<int> timeout_ms) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = writer->EndEpisode(
                   clear_buffers, timeout_ms.has_value()
                                      ? absl::Milliseconds(timeout_ms.value())
                                      : absl::InfiniteDuration());
             }
             MaybeRaiseFromStatus(status);
           })
      .def_property_readonly("step_is_open", &StructuredWriter::step_is_open);

  // InProcessClient: zero-gRPC client that holds Tables directly. Used by the
  // Python `Server(in_process=True)` path. The gRPC-backed `Client`/`Writer`
  // bindings are defined above.
  py::class_<InProcessClient, std::shared_ptr<InProcessClient>>(
      m, "InProcessClient")
      .def(py::init<std::vector<std::shared_ptr<Table>>,
                    std::shared_ptr<Checkpointer>>(),
           py::arg("tables"), py::arg("checkpointer") = nullptr)
      .def(
          "new_trajectory_writer",
          [](InProcessClient* client, const std::string& table,
             std::shared_ptr<ChunkerOptions> chunker_options)
              -> TrajectoryWriter* {
            TrajectoryWriter::Options options;
            options.chunker_options = std::move(chunker_options);
            std::unique_ptr<TrajectoryWriter> writer;
            absl::Status status;
            {
              py::gil_scoped_release g;
              status =
                  client->NewTrajectoryWriter(table, options, &writer);
            }
            MaybeRaiseFromStatus(status);
            return writer.release();
          },
          py::arg("table"), py::arg("chunker_options"))
      .def(
          "new_structured_writer",
          [](InProcessClient* client, const std::string& table,
             std::vector<StructuredWriterConfig> configs)
              -> StructuredWriter* {
            std::unique_ptr<StructuredWriter> writer;
            absl::Status status;
            {
              py::gil_scoped_release g;
              status =
                  client->NewStructuredWriter(table, std::move(configs), &writer);
            }
            MaybeRaiseFromStatus(status);
            return writer.release();
          },
          py::arg("table"), py::arg("configs"))
      .def(
          "new_sampler",
          [](InProcessClient* client, const std::string& table,
             int64_t max_samples, size_t buffer_size,
             int64_t rate_limiter_timeout_ms) -> Sampler* {
            Sampler::Options options;
            options.max_samples = max_samples;
            options.max_in_flight_samples_per_worker = buffer_size;
            options.rate_limiter_timeout =
                Int64MillisToNonnegativeDuration(rate_limiter_timeout_ms);
            std::unique_ptr<Sampler> sampler;
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = client->NewSampler(table, options, &sampler);
            }
            MaybeRaiseFromStatus(status);
            return sampler.release();
          },
          py::arg("table"), py::arg("max_samples") = 1,
          py::arg("buffer_size") = 1,
          // -1 (or any negative) means wait forever (InfiniteDuration).
          py::arg("rate_limiter_timeout_ms") = -1)
      .def(
          "mutate_priorities",
          [](InProcessClient* client, const std::string& table,
             const std::vector<std::pair<uint64_t, double>>& updates,
             const std::vector<uint64_t>& deletes) {
            std::vector<KeyWithPriority> update_protos;
            for (const auto &update : updates) {
              update_protos.emplace_back();
              update_protos.back().set_key(update.first);
              update_protos.back().set_priority(update.second);
            }
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = client->MutatePriorities(table, update_protos, deletes);
            }
            MaybeRaiseFromStatus(status);
          },
          py::arg("table"), py::arg("updates"), py::arg("deletes"))
      .def("reset",
           [](InProcessClient* client, const std::string& table) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->Reset(table);
             }
             MaybeRaiseFromStatus(status);
           },
           py::arg("table"))
      .def("checkpoint",
           [](InProcessClient* client) {
             std::string path;
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->Checkpoint(&path);
             }
             MaybeRaiseFromStatus(status);
             return path;
           })
      .def("load_latest",
           [](InProcessClient* client) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->LoadLatest();
             }
             MaybeRaiseFromStatus(status);
           })
      .def("load",
           [](InProcessClient* client, const std::string& path) {
             absl::Status status;
             {
               py::gil_scoped_release g;
               status = client->Load(path);
             }
             MaybeRaiseFromStatus(status);
           },
           py::arg("path"))
      .def(
          "server_info",
          [](InProcessClient* client) {
            std::vector<TableInfo> table_info;
            absl::Status status;
            {
              py::gil_scoped_release g;
              status = client->ServerInfo(&table_info);
            }
            MaybeRaiseFromStatus(status);

            // Return a list of serialized TableInfo proto bytes strings.
            std::vector<py::bytes> serialized_table_info;
            serialized_table_info.reserve(table_info.size());
            for (const auto &info : table_info) {
              serialized_table_info.push_back(
                  py::bytes(info.SerializeAsString()));
            }
            return serialized_table_info;
          });
}  // NOLINT(readability/fn_size)

}  // namespace
}  // namespace reverb
}  // namespace deepmind
// LINT.ThenChange(pybind.pyi)
