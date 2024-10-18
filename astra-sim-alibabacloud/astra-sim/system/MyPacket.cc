/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "MyPacket.hh"
namespace AstraSim {
MyPacket::MyPacket(int preferred_vnet, int preferred_src, int preferred_dest) {
  this->preferred_vnet = preferred_vnet;
  this->preferred_src = preferred_src;
  this->preferred_dest = preferred_dest;
  this->msg_size = 0;
}
MyPacket::MyPacket(
    uint64_t msg_size,
    int preferred_vnet,
    int preferred_src,
    int preferred_dest) {
  this->preferred_vnet = preferred_vnet;
  this->preferred_src = preferred_src;
  this->preferred_dest = preferred_dest;
  this->msg_size = msg_size;
}
MyPacket::MyPacket(
    int preferred_vnet,
    int preferred_src,
    int preferred_dest,
    uint64_t msg_size,
    int channel_id,
    int flow_id) {
  this->preferred_vnet = preferred_vnet;
  this->preferred_src = preferred_src;
  this->preferred_dest = preferred_dest;
  this->msg_size = msg_size;
  this->channel_id = channel_id;
  this->flow_id = flow_id;
}
void MyPacket::set_flow_id(int flow_id) {
  this->flow_id = flow_id;
}
void MyPacket::set_notifier(Callable* c) {
  notifier = c;
}
void MyPacket::call(EventType event, CallData* data) {
  cycles_needed = 0;
  if (notifier != nullptr)
    notifier->call(EventType::General, nullptr);
}
} // namespace AstraSim
