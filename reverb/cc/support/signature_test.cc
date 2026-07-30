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

#include "reverb/cc/support/signature.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"
// ponytail: 本地内联 proto 匹配工具,避免拉入 //reverb/cc/testing:proto_test_util
// (该 cc_library 的 .cc 仍依赖 TF CompressTensorAsProto,Task 4 后未迁移)。
// 升级路径: Task 7 迁完测试工具后改回 #include proto_test_util.h。
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

using ::reverb::tensor::SignatureProto;

template <typename T>
T CreateProto(const std::string& textual_proto) {
  T proto;
  REVERB_CHECK(google::protobuf::TextFormat::ParseFromString(textual_proto,
                                                              &proto));
  return proto;
}

// Compares two protos by full equality via MessageDifferencer.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const std::string& expected)
      : expected_proto_str_(expected) {}
  template <typename Message>
  bool MatchAndExplain(const Message& actual_proto,
                      ::testing::MatchResultListener* listener) const {
    Message expected_proto = CreateProto<Message>(expected_proto_str_);
    google::protobuf::util::MessageDifferencer differencer;
    std::string differences;
    differencer.ReportDifferencesToString(&differences);
    if (!differencer.Compare(expected_proto, actual_proto)) {
      *listener << "the protos are different:\n" << differences;
      return false;
    }
    return true;
  }
  void DescribeTo(::std::ostream* os) const { *os << expected_proto_str_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_proto_str_;
  }
 private:
  const std::string expected_proto_str_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const std::string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Builds a TensorSpec leaf. An empty `shape` represents a scalar (rank 0).
SignatureProto MakeLeaf(const std::string& name,
                        ::reverb::tensor::DataType dtype =
                            ::reverb::tensor::DT_FLOAT32,
                        const std::vector<int64_t>& shape = {}) {
  SignatureProto value;
  SignatureProto::TensorSpec* tensor_spec = value.mutable_tensor_spec();
  tensor_spec->set_name(name);
  tensor_spec->set_dtype(dtype);
  for (int64_t d : shape) tensor_spec->mutable_shape()->add_dim(d);
  return value;
}

TEST(FlatSignatureFromSignatureProtoTest, TensorSpec) {
  SignatureProto value = MakeLeaf("leaf");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(dtypes_and_shapes.has_value());
  EXPECT_EQ(dtypes_and_shapes.value().size(), 1);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "leaf");
  EXPECT_EQ(dtypes_and_shapes.value()[0].dtype, DataType::Float32);
  EXPECT_TRUE(dtypes_and_shapes.value()[0].shape.empty());
}

TEST(FlatSignatureFromSignatureProtoTest, BoundedTensorSpec) {
  SignatureProto value;
  SignatureProto::BoundedTensorSpec* bounded_tensor_spec =
      value.mutable_bounded_tensor_spec();
  bounded_tensor_spec->set_name("leaf");
  bounded_tensor_spec->set_dtype(::reverb::tensor::DT_INT32);
  bounded_tensor_spec->mutable_shape()->add_dim(8);

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(dtypes_and_shapes.has_value());
  EXPECT_EQ(dtypes_and_shapes.value().size(), 1);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "leaf");
  EXPECT_EQ(dtypes_and_shapes.value()[0].dtype, DataType::Int32);
  ASSERT_EQ(dtypes_and_shapes.value()[0].shape.size(), 1);
  EXPECT_EQ(dtypes_and_shapes.value()[0].shape[0], 8);
}

TEST(FlatSignatureFromSignatureProtoTest, ListNaming) {
  SignatureProto value;
  *value.mutable_list_value()->add_values() = MakeLeaf("one");
  *value.mutable_list_value()->add_values() = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  ASSERT_TRUE(dtypes_and_shapes.has_value());
  ASSERT_EQ(dtypes_and_shapes.value().size(), 2);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "1/two");
}

TEST(FlatSignatureFromSignatureProtoTest, TupleNaming) {
  SignatureProto value;
  *value.mutable_tuple_value()->add_values() = MakeLeaf("one");
  *value.mutable_tuple_value()->add_values() = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  ASSERT_TRUE(dtypes_and_shapes.has_value());
  ASSERT_EQ(dtypes_and_shapes.value().size(), 2);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "1/two");
}

TEST(FlatSignatureFromSignatureProtoTest, DictNaming) {
  SignatureProto value;
  (*value.mutable_dict_value()->mutable_values())["a"] = MakeLeaf("one");
  (*value.mutable_dict_value()->mutable_values())["b"] = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  ASSERT_TRUE(dtypes_and_shapes.has_value());
  ASSERT_EQ(dtypes_and_shapes.value().size(), 2);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "b/two");
}

TEST(FlatSignatureFromSignatureProtoTest, NamedTupleNaming) {
  SignatureProto value;
  value.mutable_named_tuple_value()->set_name("namedtuple");
  value.mutable_named_tuple_value()->add_keys("a");
  *value.mutable_named_tuple_value()->add_values() = MakeLeaf("one");
  value.mutable_named_tuple_value()->add_keys("b");
  *value.mutable_named_tuple_value()->add_values() = MakeLeaf("two");
  value.mutable_named_tuple_value()->add_keys("c");
  *value.mutable_named_tuple_value()->add_values() = MakeLeaf("three");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  ASSERT_TRUE(dtypes_and_shapes.has_value());
  ASSERT_EQ(dtypes_and_shapes.value().size(), 3);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "b/two");
  EXPECT_EQ(dtypes_and_shapes.value()[2].name, "c/three");
}

TEST(FlatSignatureFromSignatureProtoTest, NestedNaming) {
  SignatureProto value;
  value.mutable_named_tuple_value()->set_name("namedtuple");
  value.mutable_named_tuple_value()->add_keys("a");
  *value.mutable_named_tuple_value()->add_values()->mutable_list_value()
       ->add_values() = MakeLeaf("one");
  *value.mutable_named_tuple_value()->mutable_values(0)
       ->mutable_list_value()
       ->add_values() = MakeLeaf("two");
  value.mutable_named_tuple_value()->add_keys("b");
  *value.mutable_named_tuple_value()->add_values() = MakeLeaf("three");
  value.mutable_named_tuple_value()->add_keys("c");
  *value.mutable_named_tuple_value()->add_values() = MakeLeaf("four");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  ASSERT_TRUE(dtypes_and_shapes.has_value());
  ASSERT_EQ(dtypes_and_shapes.value().size(), 4);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "a/1/two");
  EXPECT_EQ(dtypes_and_shapes.value()[2].name, "b/three");
  EXPECT_EQ(dtypes_and_shapes.value()[3].name, "c/four");
}

TEST(FlatSignatureFromSignatureProtoTest, EmptyLeaf) {
  SignatureProto value;
  value.mutable_named_tuple_value()->set_name("namedtuple");
  value.mutable_named_tuple_value()->add_keys("a");
  *value.mutable_named_tuple_value()->add_values()->mutable_list_value()
       ->add_values() = MakeLeaf("one");
  *value.mutable_named_tuple_value()->mutable_values(0)
       ->mutable_list_value()
       ->add_values() = MakeLeaf("");
  value.mutable_named_tuple_value()->add_keys("b");
  *value.mutable_named_tuple_value()->add_values() = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromSignatureProto(value, &dtypes_and_shapes);
  ASSERT_TRUE(dtypes_and_shapes.has_value());
  ASSERT_EQ(dtypes_and_shapes.value().size(), 3);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "a/1");
  EXPECT_EQ(dtypes_and_shapes.value()[2].name, "b/two");
}

TEST(AddBatchDim, EmptyStructure) {
  SignatureProto value;
  REVERB_EXPECT_OK(AddBatchDim(&value, 10));
  EXPECT_THAT(value, EqualsProto(""));
}

TEST(AddBatchDim, NestedStructure) {
  auto value = CreateProto<SignatureProto>(R"pb(
    dict_value {
      values {
        key: "a"
        value {
          list_value {
            values {
              tensor_spec {
                name: "spec_1"
                shape { dim: 5 }
                dtype: DT_FLOAT32
              }
            }
            values {
              bounded_tensor_spec {
                name: "bounded_spec_1"
                shape {}
                dtype: DT_INT32
              }
            }
          }
        }
      }
      values {
        key: "b"
        value {
          tuple_value {
            values {
              tensor_spec {
                name: "spec_2"
                shape { dim: 1 }
                dtype: DT_FLOAT64
              }
            }
            values {
              named_tuple_value {
                name: "named_tuple"
                keys: "first"
                values {
                  tensor_spec {
                    name: "spec_3"
                    shape {}
                    dtype: DT_BOOL
                  }
                }
              }
            }
          }
        }
      }
    }
  )pb");
  REVERB_EXPECT_OK(AddBatchDim(&value, 10));
  EXPECT_THAT(value, EqualsProto(R"pb(
    dict_value {
      values {
        key: "a"
        value {
          list_value {
            values {
              tensor_spec {
                name: "spec_1"
                shape { dim: 10 dim: 5 }
                dtype: DT_FLOAT32
              }
            }
            values {
              bounded_tensor_spec {
                name: "bounded_spec_1"
                shape { dim: 10 }
                dtype: DT_INT32
              }
            }
          }
        }
      }
      values {
        key: "b"
        value {
          tuple_value {
            values {
              tensor_spec {
                name: "spec_2"
                shape { dim: 10 dim: 1 }
                dtype: DT_FLOAT64
              }
            }
            values {
              named_tuple_value {
                name: "named_tuple"
                keys: "first"
                values {
                  tensor_spec {
                    name: "spec_3"
                    shape { dim: 10 }
                    dtype: DT_BOOL
                  }
                }
              }
            }
          }
        }
      }
    }
  )pb"));
}

}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
