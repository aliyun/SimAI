/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "StreamBaseline.hh"
#include "MockNcclLog.h"
#include "astra-sim/system/collective/Algorithm.hh"
namespace AstraSim {
StreamBaseline::StreamBaseline(
    Sys* owner,
    DataSet* dataset,
    int stream_num,
    std::list<CollectivePhase> phases_to_go,
    int priority)
    : BaseStream(stream_num, owner, phases_to_go) {
  this->owner = owner;
  this->stream_num = stream_num;
  this->phases_to_go = phases_to_go;
  this->dataset = dataset;
  this->priority = priority;
  steps_finished = 0;
  initial_data_size = phases_to_go.front().initial_data_size;
}
void StreamBaseline::init() {
  initialized = true;
  last_init = Sys::boostedTick();
  if (!my_current_phase.enabled) {
    return;
  }
  my_current_phase.algorithm->run(EventType::StreamInit, nullptr);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"StreamBaseline::algorithm->run finished");
  if (steps_finished == 1) {
    queuing_delay.push_back(last_phase_change - creation_time);
  }
  queuing_delay.push_back(Sys::boostedTick() - last_phase_change);
  total_packets_sent = 1;
}
void StreamBaseline::call(EventType event, CallData* data) {
  if (event == EventType::WaitForVnetTurn) {
    owner->proceed_to_next_vnet_baseline(this);
    return;
  } else if(event == EventType::NCCL_General) {
    
    BasicEventHandlerData* behd = (BasicEventHandlerData*) data;
    int channel_id = behd->channel_id;
    my_current_phase.algorithm->run(EventType::General, data);
  } else {
    // std::cout<<"general event called in stream"<<std::endl;
    SharedBusStat* sharedBusStat = (SharedBusStat*)data;
    update_bus_stats(BusType::Both, sharedBusStat);
    my_current_phase.algorithm->run(EventType::General, data);
    if (data != nullptr) {
      delete sharedBusStat;
    }
  }
}
void StreamBaseline::consume(RecvPacketEventHadndlerData* message) {
  net_message_latency.back() +=
      Sys::boostedTick() - message->ready_time; 
  net_message_counter++;
  my_current_phase.algorithm->run(EventType::PacketReceived, message);
}
void StreamBaseline::sendcallback(SendPacketEventHandlerData* messages){
  if(my_current_phase.algorithm!=nullptr)
    my_current_phase.algorithm->run(EventType::PacketSentFinshed,messages);
}
} // namespace AstraSim
