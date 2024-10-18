/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "AllToAll.hh"
namespace AstraSim {
AllToAll::AllToAll(
    ComType type,
    int window,
    int id,
    int layer_num,
    RingTopology* allToAllTopology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode)
    : Ring(
          type,
          id,
          layer_num,
          allToAllTopology,
          data_size,
          direction,
          injection_policy,
          boost_mode) {
  this->name = Name::AllToAll;
  this->enabled = true;
  this->middle_point = nodes_in_ring - 1;
  if (boost_mode) {
    this->enabled = allToAllTopology->is_enabled();
  }
  if (window == -1) {
    parallel_reduce = nodes_in_ring - 1;
  } else {
    parallel_reduce = (int)std::min(window, nodes_in_ring - 1);
  }
  if (type == ComType::All_to_All) {
    this->stream_count = nodes_in_ring - 1;
  }
}
int AllToAll::get_non_zero_latency_packets() {
  if (((RingTopology*)logicalTopology)->dimension !=
      RingTopology::Dimension::Local) {
    return parallel_reduce * 1;
  } else {
    return (nodes_in_ring - 1) * parallel_reduce * 1;
  }
}
void AllToAll::process_max_count() {
  if (remained_packets_per_max_count > 0)
    remained_packets_per_max_count--;
  if (remained_packets_per_max_count == 0) {
    max_count--;
    release_packets();
    remained_packets_per_max_count = 1;

    current_receiver = ((RingTopology*)logicalTopology)
                           ->get_receiver_node(current_receiver, direction);
    if (current_receiver == id) {
      current_receiver = ((RingTopology*)logicalTopology)
                             ->get_receiver_node(current_receiver, direction);
    }
    current_sender = ((RingTopology*)logicalTopology)
                         ->get_sender_node(current_sender, direction);
    if (current_sender == id) {
      current_sender = ((RingTopology*)logicalTopology)
                           ->get_sender_node(current_sender, direction);
    }
  }
}
void AllToAll::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;
    if (comType == ComType::All_Reduce && stream_count <= middle_point) {
      if (total_packets_received < middle_point) {
        return;
      }
      for (int i = 0; i < parallel_reduce; i++) {
        ready();
      }
      iteratable();
    } else {
      ready();
      iteratable();
    }
  } else if (event == EventType::PacketReceived) {
    total_packets_received++;
    insert_packet(nullptr);
  } else if (event == EventType::StreamInit) {
    for (int i = 0; i < parallel_reduce; i++) {
      insert_packet(nullptr);
    }
  }
}
} // namespace AstraSim
