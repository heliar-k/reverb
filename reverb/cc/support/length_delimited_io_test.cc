#include "reverb/cc/support/length_delimited_io.h"

#include <gtest/gtest.h>
#include <fstream>
#include <vector>

#include "absl/status/status.h"
#include "reverb/cc/schema.pb.h"

namespace deepmind {
namespace reverb {

TEST(LengthDelimitedIO, WriteAndReadBackSingleMessage) {
  std::string path = "/tmp/test_ld_io_single.pb";
  PrioritizedItem item;
  item.set_key(42);
  item.set_table("test_table");
  item.set_priority(0.5);

  ASSERT_TRUE(WriteLengthDelimitedToFile(path, item).ok());

  PrioritizedItem restored;
  ASSERT_TRUE(ReadLengthDelimitedFromFile(path, &restored).ok());
  EXPECT_EQ(restored.key(), 42u);
  EXPECT_EQ(restored.table(), "test_table");
  EXPECT_DOUBLE_EQ(restored.priority(), 0.5);
}

TEST(LengthDelimitedIO, WriteAndReadBackMultipleMessages) {
  std::string path = "/tmp/test_ld_io_multi.pb";
  std::vector<uint64_t> keys = {1, 100, 999};
  {
    std::ofstream ofs(path, std::ios::binary);
    for (uint64_t k : keys) {
      PrioritizedItem item;
      item.set_key(k);
      ASSERT_TRUE(WriteLengthDelimited(ofs, item).ok());
    }
  }
  std::ifstream ifs(path, std::ios::binary);
  int i = 0;
  while (true) {
    PrioritizedItem item;
    absl::Status s = ReadLengthDelimited(ifs, &item);
    if (absl::IsNotFound(s)) break;
    ASSERT_TRUE(s.ok()) << s.message();
    EXPECT_EQ(item.key(), keys[i]);
    i++;
  }
  EXPECT_EQ(i, 3);
}

TEST(LengthDelimitedIO, ReadEmptyFileReturnsNotFound) {
  std::string path = "/tmp/test_ld_io_empty.pb";
  { std::ofstream ofs(path, std::ios::binary); }  // 创建空文件
  PrioritizedItem item;
  absl::Status s = ReadLengthDelimitedFromFile(path, &item);
  EXPECT_TRUE(absl::IsNotFound(s)) << s.message();
}

}  // namespace reverb
}  // namespace deepmind
