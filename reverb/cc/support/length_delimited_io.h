#ifndef REVERB_CC_SUPPORT_LENGTH_DELIMITED_IO_H_
#define REVERB_CC_SUPPORT_LENGTH_DELIMITED_IO_H_

#include <istream>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "google/protobuf/message.h"

namespace deepmind {
namespace reverb {

// 写一条 protobuf 消息到流:varint 长度前缀 + 消息字节。
// 流式,可连续写多条。
absl::Status WriteLengthDelimited(std::ostream& os,
                                  const google::protobuf::Message& msg);

// 从流读一条 protobuf 消息(与 WriteLengthDelimited 配对)。
// 读到 EOF 且无数据时返回 absl::NotFoundError(用于循环结束判断)。
absl::Status ReadLengthDelimited(std::istream& is, google::protobuf::Message* msg);

// 便捷:单条消息写文件 / 读文件。
absl::Status WriteLengthDelimitedToFile(const std::string& path,
                                        const google::protobuf::Message& msg);
absl::Status ReadLengthDelimitedFromFile(const std::string& path,
                                         google::protobuf::Message* msg);

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_LENGTH_DELIMITED_IO_H_
