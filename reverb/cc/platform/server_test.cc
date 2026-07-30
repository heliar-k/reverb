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

#include "reverb/cc/platform/server.h"

#include <memory>

#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "reverb/cc/platform/net.h"
#include "reverb/cc/platform/status_matchers.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(ServerTest, StartServer) {
  int port = internal::PickUnusedPortOrDie();
  Server server(port);
  REVERB_EXPECT_OK(server.Initialize(/*tables=*/{},
                                     /*checkpointer=*/nullptr));
}

TEST(ServerTest, ErrorOnUnavailablePort) {
  // We expect that port==-1 to always be unavailable.
  Server server(-1);
  auto status = server.Initialize(/*tables=*/{}, /*checkpointer=*/nullptr);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Failed to BuildAndStart gRPC server"));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
