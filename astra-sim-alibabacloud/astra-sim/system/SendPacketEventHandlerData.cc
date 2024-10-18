/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "SendPacketEventHandlerData.hh"
namespace AstraSim {
SendPacketEventHandlerData::SendPacketEventHandlerData(
    Sys* node,
    int senderNodeId,
    int receiverNodeId,
    int tag)
    : BasicEventHandlerData(node, EventType::PacketSent) {
  this->senderNodeId=senderNodeId;
  this->receiverNodeId = receiverNodeId;
  this->tag = tag;
  this->child_flow_id = -2;
};

SendPacketEventHandlerData::SendPacketEventHandlerData(
    BaseStream* owner,
    int senderNodeId,
    int receiverNodeId,
    int tag,
    EventType event)
    : BasicEventHandlerData(owner->owner, event) {
  this->owner = owner;
  this->senderNodeId = senderNodeId;
  this->receiverNodeId = receiverNodeId;
  this->tag = tag;
  this->channel_id = -1;
  this->child_flow_id = -1;
}

} // namespace AstraSim
