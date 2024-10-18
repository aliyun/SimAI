/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "RecvPacketEventHadndlerData.hh"
namespace AstraSim {
RecvPacketEventHadndlerData::RecvPacketEventHadndlerData(
    BaseStream* owner,
    int nodeId,
    EventType event,
    int vnet,
    int stream_num)
    : BasicEventHandlerData(owner->owner, event) {
  this->owner = owner;
  this->vnet = vnet;
  this->stream_num = stream_num;
  this->message_end = true;
  ready_time = Sys::boostedTick();
  flow_id = -2;
  child_flow_id = -1;
}
RecvPacketEventHadndlerData::RecvPacketEventHadndlerData(
    BaseStream* owner,
    EventType event,
    AstraSim::ncclFlowTag _flowTag)
    : BasicEventHandlerData(owner->owner, event) {
  this->flowTag = _flowTag;
}

} // namespace AstraSim
