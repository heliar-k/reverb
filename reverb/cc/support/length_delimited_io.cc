#include "reverb/cc/support/length_delimited_io.h"

#include <fstream>
#include <string>

namespace deepmind {
namespace reverb {

// ponytail: 手写 varint + 直接 istream::read,绕开 protobuf CodedInputStream 的
// 缓冲(小文件会被一次性读光并扰动 eof/peek,导致逐条重建流时 EOF 误判)。
// 升级路径:超大消息可换回 CodedInputStream 的 PushLimit 流式解析省一次拷贝。

absl::Status WriteLengthDelimited(std::ostream& os,
                                  const google::protobuf::Message& msg) {
  std::string data;
  if (!msg.SerializeToString(&data)) {
    return absl::InternalError("SerializeToString failed");
  }
  uint32_t size = data.size();
  while (size >= 0x80) {
    os.put(static_cast<char>(size | 0x80));
    size >>= 7;
  }
  os.put(static_cast<char>(size));
  os.write(data.data(), static_cast<std::streamsize>(data.size()));
  return os ? absl::OkStatus() : absl::InternalError("stream write failed");
}

absl::Status ReadLengthDelimited(std::istream& is, google::protobuf::Message* msg) {
  uint32_t size = 0;
  int shift = 0;
  int c;
  while ((c = is.get()) != std::char_traits<char>::eof()) {
    size |= (static_cast<uint32_t>(c & 0x7F) << shift);
    if (!(c & 0x80)) break;
    shift += 7;
    if (shift >= 32) return absl::InternalError("length varint too long");
  }
  // 读到的第一个字节就是 EOF => 干净的流尾,无更多消息。
  if (shift == 0 && is.eof()) {
    return absl::NotFoundError("EOF");
  }
  std::string buf(size, '\0');
  is.read(buf.data(), static_cast<std::streamsize>(size));
  if (is.gcount() != static_cast<std::streamsize>(size)) {
    return absl::InternalError("short read");
  }
  return msg->ParseFromString(buf)
             ? absl::OkStatus()
             : absl::InternalError("ParseFromString failed");
}

absl::Status WriteLengthDelimitedToFile(const std::string& path,
                                        const google::protobuf::Message& msg) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) {
    return absl::InternalError("Failed to open file for writing: " + path);
  }
  return WriteLengthDelimited(ofs, msg);
}

absl::Status ReadLengthDelimitedFromFile(const std::string& path,
                                         google::protobuf::Message* msg) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    return absl::InternalError("Failed to open file for reading: " + path);
  }
  return ReadLengthDelimited(ifs, msg);
}

}  // namespace reverb
}  // namespace deepmind
