/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PacketBundle.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "PhyMultiThread.hh"
namespace AstraSim {
PacketBundle::PacketBundle(
    Sys* generator,
    BaseStream* stream,
    std::list<MyPacket*> locked_packets,
    bool needs_processing,
    bool send_back,
    uint64_t size,
    MemBus::Transmition transmition) {
  this->generator = generator;
  this->locked_packets = locked_packets;
  this->needs_processing = needs_processing;
  this->send_back = send_back;
  this->size = size;
  this->stream = stream;
  this->transmition = transmition;
  creation_time = Sys::boostedTick();
  this->channel_id = -1;
}
PacketBundle::PacketBundle(
    Sys* generator,
    BaseStream* stream,
    std::list<MyPacket*> locked_packets,
    bool needs_processing,
    bool send_back,
    uint64_t size,
    MemBus::Transmition transmition,
    int channel_id,
    int flow_id) {
  this->generator = generator;
  this->locked_packets = locked_packets;
  this->needs_processing = needs_processing;
  this->send_back = send_back;
  this->size = size;
  this->stream = stream;
  this->transmition = transmition;
  this->channel_id = channel_id;
  this->flow_id = flow_id;
  creation_time = Sys::boostedTick();
}
PacketBundle::PacketBundle(
    Sys* generator,
    BaseStream* stream,
    bool needs_processing,
    bool send_back,
    uint64_t size,
    MemBus::Transmition transmition) {
  this->generator = generator;
  this->needs_processing = needs_processing;
  this->send_back = send_back;
  this->size = size;
  this->stream = stream;
  this->transmition = transmition;
  creation_time = Sys::boostedTick();
  this->channel_id = -1;
}
void PacketBundle::send_to_MA() {
  generator->memBus->send_from_NPU_to_MA(
      transmition, size, needs_processing, send_back, this);
}
void PacketBundle::send_to_NPU() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  generator->memBus->send_from_MA_to_NPU(
      transmition, size, needs_processing, send_back, this);
  NcclLog->writeLog(NcclLogLevel::DEBUG,"send_to_NPU done");
}
void PacketBundle::call(EventType event, CallData* data) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"packet bundle call");
  if (needs_processing == true) {
    needs_processing = false;
    this->delay = generator->mem_write(size) + generator->mem_read(size) +
        generator->mem_read(size);
    generator->try_register_event(
        this, EventType::CommProcessingFinished, data, this->delay);
    return;
  }
  Tick current = Sys::boostedTick();
  #ifndef PHY_MTP
  for (auto& packet : locked_packets) {
    packet->ready_time = current;
  }
  #endif
  BasicEventHandlerData* ehd = new BasicEventHandlerData(channel_id, flow_id);
  if(channel_id == -1) stream->call(EventType::General, data);
  else stream->call(EventType::NCCL_General, ehd);
  delete this;
}
} // namespace AstraSim
