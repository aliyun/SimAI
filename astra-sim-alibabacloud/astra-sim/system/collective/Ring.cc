/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Ring.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
namespace AstraSim {
Ring::Ring(
    ComType type,
    int id,
    int layer_num,
    RingTopology* ring_topology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode)
    : Algorithm(layer_num) {
  this->comType = type;
  this->id = id;
  this->logicalTopology = ring_topology;
  this->data_size = data_size;
  this->direction = direction;
  this->nodes_in_ring = ring_topology->get_nodes_in_ring();
  this->current_receiver = ring_topology->get_receiver_node(id, direction);
  this->current_sender = ring_topology->get_sender_node(id, direction);
  this->parallel_reduce = 1;    // each ring has xx channels
  this->injection_policy = injection_policy;
  this->total_packets_sent = 0;
  this->total_packets_received = 0;
  this->free_packets = 0;
  this->zero_latency_packets = 0;
  this->non_zero_latency_packets = 0;
  this->toggle = false;
  this->name = Name::Ring;
  this->enabled = true;
  if (boost_mode) {
    this->enabled = ring_topology->is_enabled();
  }
  if (ring_topology->dimension == RingTopology::Dimension::Local) {
    transmition = MemBus::Transmition::Fast;
  } else {
    transmition = MemBus::Transmition::Usual;
  }
  switch (type) {
    case ComType::All_Reduce:
      stream_count = 2 * (nodes_in_ring - 1);
      break;
    case ComType::All_to_All:
      this->stream_count = ((nodes_in_ring - 1) * nodes_in_ring) / 2;
      switch (injection_policy) {
        case InjectionPolicy::Aggressive:
          this->parallel_reduce = nodes_in_ring - 1;
          break;
        case InjectionPolicy::Normal:
          this->parallel_reduce = 1;
          break;
        default:
          this->parallel_reduce = 1;
          break;
      }
      break;
    default:
      stream_count = nodes_in_ring - 1;
  }
  if (type == ComType::All_to_All || type == ComType::All_Gather) {
    max_count = 0;
  } else {
    max_count = nodes_in_ring - 1;
  }
  remained_packets_per_message = 1;
  remained_packets_per_max_count = 1;
  switch (type) {
    case ComType::All_Reduce:
      this->final_data_size = data_size;
      this->msg_size = data_size / nodes_in_ring;
      break;
    case ComType::All_Gather:
      this->final_data_size = data_size * nodes_in_ring;
      this->msg_size = data_size;
      break;
    case ComType::Reduce_Scatter:
      this->final_data_size = data_size / nodes_in_ring;
      this->msg_size = data_size / nodes_in_ring;
      break;
    case ComType::All_to_All:
      this->final_data_size = data_size;
      this->msg_size = data_size / nodes_in_ring;
      break;
    default:;
  }
}
int Ring::get_non_zero_latency_packets() {
  return (nodes_in_ring - 1) * parallel_reduce * 1;
}
void Ring::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;
    ready();
    iteratable();
  } else if (event == EventType::PacketReceived) {
    total_packets_received++;
    insert_packet(nullptr);
  } else if (event == EventType::StreamInit) {
    for (int i = 0; i < parallel_reduce; i++) {
      insert_packet(nullptr);
    }
  }
}
void Ring::release_packets() {
  for (auto packet : locked_packets) {
    packet->set_notifier(this);
  }
  if (NPU_to_MA == true) {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         msg_size,
         transmition))
        ->send_to_MA();
  } else {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         msg_size,
         transmition))
        ->send_to_NPU();
  }
  locked_packets.clear();
}
void Ring::process_stream_count() {
  if (remained_packets_per_message > 0) {
    remained_packets_per_message--;
  }
  if (id == 0) {
  }
  if (remained_packets_per_message == 0 && stream_count > 0) {
    stream_count--;
    if (stream_count > 0) {
      remained_packets_per_message = 1;
    }
  }
  if (remained_packets_per_message == 0 && stream_count == 0 &&
      stream->state != StreamState::Dead) {
    stream->changeState(StreamState::Zombie);
  }
}
void Ring::process_max_count() {
  if (remained_packets_per_max_count > 0)
    remained_packets_per_max_count--;
  if (remained_packets_per_max_count == 0) {
    max_count--;
    release_packets();
    remained_packets_per_max_count = 1;
  }
}
void Ring::reduce() {
  process_stream_count();
  packets.pop_front();
  free_packets--;
  total_packets_sent++;
  // not_delivered++;
}
bool Ring::iteratable() {
  if (stream_count == 0 &&
      free_packets == (parallel_reduce * 1)) { // && not_delivered==0
      
    exit();
    return false;
  }
  return true;
}
void Ring::insert_packet(Callable* sender) {
  if (!enabled) {
    return;
  }
  if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
    zero_latency_packets = parallel_reduce * 1;
    non_zero_latency_packets =
        get_non_zero_latency_packets(); //(nodes_in_ring-1)*parallel_reduce*1;
    toggle = !toggle;
  }
  if (zero_latency_packets > 0) {
    packets.push_back(MyPacket(
        stream->current_queue_id,
        current_sender,
        current_receiver)); // vnet Must be changed for alltoall topology
    packets.back().sender = sender;
    locked_packets.push_back(&packets.back());
    processed = false;
    send_back = false;
    NPU_to_MA = true;
    process_max_count();
    zero_latency_packets--;
    return;
  } else if (non_zero_latency_packets > 0) {
    // if(id == 0) std::cout << "non_zero_latency_packets > 0" << std::endl;
    packets.push_back(MyPacket(
        stream->current_queue_id,
        current_sender,
        current_receiver)); // vnet Must be changed for alltoall topology
    packets.back().sender = sender;
    locked_packets.push_back(&packets.back());
    if (comType == ComType::Reduce_Scatter ||
        (comType == ComType::All_Reduce && toggle)) {
      processed = true;
    } else {
      processed = false;
    }
    if (non_zero_latency_packets <= parallel_reduce * 1) {
      send_back = false;
    } else {
      send_back = true;
    }
    NPU_to_MA = false;
    std::cout << "id: " << id << " non-zero latency packets at tick: " << Sys::boostedTick() << std::endl;
    process_max_count();
    non_zero_latency_packets--;
    return;
  }
  Sys::sys_panic("should not inject nothing!");

}
bool Ring::ready() {
  if (stream->state == StreamState::Created ||
      stream->state == StreamState::Ready) {
    stream->changeState(StreamState::Executing);
  }
  if (!enabled || packets.size() == 0 || stream_count == 0 ||
      free_packets == 0) {
    return false;
  }
  MyPacket packet = packets.front();
  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = packet.preferred_dest;
  snd_req.tag = stream->stream_num;
  snd_req.reqType = UINT8;
  snd_req.vnet = this->stream->current_queue_id;
  snd_req.layerNum = layer_num;
  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      msg_size,
      UINT8,
      packet.preferred_dest,
      stream->stream_num,
      &snd_req,
      &Sys::handleEvent,
      nullptr); // stream_num+(packet.preferred_dest*50)
  sim_request rcv_req;
  rcv_req.vnet = this->stream->current_queue_id;
  rcv_req.layerNum = layer_num;
  RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
      stream,
      stream->owner->id,
      EventType::PacketReceived,
      packet.preferred_vnet,
      packet.stream_num);
  stream->owner->front_end_sim_recv(
      0,
      Sys::dummy_data,
      msg_size,
      UINT8,
      packet.preferred_src,
      stream->stream_num,
      &rcv_req,
      &Sys::handleEvent,
      ehd); // stream_num+(owner->id*50)
  reduce();
  return true;
}
void Ring::exit() {
  if (packets.size() != 0) {
    packets.clear();
  }
  if (locked_packets.size() != 0) {
    locked_packets.clear();
  }
  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
  return;
}
} // namespace AstraSim
