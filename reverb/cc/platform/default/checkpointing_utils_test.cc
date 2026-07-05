// Copyright 2024 DeepMind Technologies Limited.
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

#include "reverb/cc/platform/checkpointing_utils.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/lifo.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(MakeSelectorTest, Fifo) {
  KeyDistributionOptions options;
  options.set_fifo(true);
  auto selector = MakeSelector(options);
  ASSERT_NE(selector, nullptr);
  EXPECT_NE(dynamic_cast<FifoSelector*>(selector.get()), nullptr);
}

TEST(MakeSelectorTest, Lifo) {
  KeyDistributionOptions options;
  options.set_lifo(true);
  auto selector = MakeSelector(options);
  ASSERT_NE(selector, nullptr);
  EXPECT_NE(dynamic_cast<LifoSelector*>(selector.get()), nullptr);
}

TEST(MakeSelectorTest, Uniform) {
  KeyDistributionOptions options;
  options.set_uniform(true);
  auto selector = MakeSelector(options);
  ASSERT_NE(selector, nullptr);
  EXPECT_NE(dynamic_cast<UniformSelector*>(selector.get()), nullptr);
}

TEST(MakeSelectorTest, Prioritized) {
  KeyDistributionOptions options;
  options.mutable_prioritized()->set_priority_exponent(0.5);
  auto selector = MakeSelector(options);
  ASSERT_NE(selector, nullptr);
  EXPECT_NE(dynamic_cast<PrioritizedSelector*>(selector.get()), nullptr);
}

TEST(MakeSelectorTest, HeapMin) {
  KeyDistributionOptions options;
  options.mutable_heap()->set_min_heap(true);
  auto selector = MakeSelector(options);
  ASSERT_NE(selector, nullptr);
  EXPECT_NE(dynamic_cast<HeapSelector*>(selector.get()), nullptr);
}

TEST(MakeSelectorTest, HeapMax) {
  KeyDistributionOptions options;
  options.mutable_heap()->set_min_heap(false);
  auto selector = MakeSelector(options);
  ASSERT_NE(selector, nullptr);
  EXPECT_NE(dynamic_cast<HeapSelector*>(selector.get()), nullptr);
}

#if GTEST_HAS_DEATH_TEST
TEST(MakeSelectorDeathTest, NotSetIsFatal) {
  KeyDistributionOptions options;  // distribution not set
  EXPECT_DEATH(MakeSelector(options), "Selector not set");
}
#endif

}  // namespace
}  // namespace reverb
}  // namespace deepmind
