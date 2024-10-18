/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "DoubleBinaryTreeAllReduce.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
namespace AstraSim {
DoubleBinaryTreeAllReduce::DoubleBinaryTreeAllReduce(
    int id,
    int layer_num,
    BinaryTree* tree,
    uint64_t data_size,
    bool boost_mode)
    : Algorithm(layer_num) {
  this->id = id;
  this->logicalTopology = tree;
  this->data_size = data_size;
  this->state = State::Begin;
  this->reductions = 0;
  this->parent = tree->get_parent_id(id);
  this->left_child = tree->get_left_child_id(id);
  this->right_child = tree->get_right_child_id(id);
  this->type = tree->get_node_type(id);
  this->final_data_size = data_size;
  this->comType = ComType::All_Reduce;
  this->name = Name::DoubleBinaryTree;
  this->enabled = true;
  if (boost_mode) {
    this->enabled = tree->is_enabled(id);
  }
}
void DoubleBinaryTreeAllReduce::run(EventType event, CallData* data) {
  if (state == State::Begin && type == BinaryTree::Type::Leaf) { // leaf.1
    (new PacketBundle(
         stream->owner,
         stream,
         false,
         false,
         data_size,
         MemBus::Transmition::Usual))
        ->send_to_MA();
    state = State::SendingDataToParent;
    return;
  } else if (
      state == State::SendingDataToParent &&
      type == BinaryTree::Type::Leaf) { // leaf.3
    sim_request snd_req;
    snd_req.srcRank = stream->owner->id;
    snd_req.dstRank = parent;
    snd_req.tag = stream->stream_num;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    snd_req.layerNum = layer_num;
    stream->owner->front_end_sim_send(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        parent,
        stream->stream_num,
        &snd_req,
        &Sys::handleEvent,
        nullptr);
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;
    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        stream->current_queue_id,
        stream->stream_num);
    stream->owner->front_end_sim_recv(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        parent,
        stream->stream_num,
        &rcv_req,
        &Sys::handleEvent,
        ehd);
    state = State::WaitingDataFromParent;
    return;
  } else if (
      state == State::WaitingDataFromParent &&
      type == BinaryTree::Type::Leaf) { // leaf.4
    (new PacketBundle(
         stream->owner,
         stream,
         false,
         false,
         data_size,
         MemBus::Transmition::Usual))
        ->send_to_NPU();
    state = State::End;
    return;
  } else if (state == State::End && type == BinaryTree::Type::Leaf) { // leaf.5
    exit();
    return;
  }

  else if (
      state == State::Begin &&
      type == BinaryTree::Type::Intermediate) { // int.1
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;
    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        stream->current_queue_id,
        stream->stream_num);
    stream->owner->front_end_sim_recv(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        left_child,
        stream->stream_num,
        &rcv_req,
        &Sys::handleEvent,
        ehd);
    sim_request rcv_req2;
    rcv_req2.vnet = this->stream->current_queue_id;
    rcv_req2.layerNum = layer_num;
    RecvPacketEventHadndlerData* ehd2 = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        stream->current_queue_id,
        stream->stream_num);
    stream->owner->front_end_sim_recv(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        right_child,
        stream->stream_num,
        &rcv_req2,
        &Sys::handleEvent,
        ehd2);
    state = State::WaitingForTwoChildData;
    return;
  } else if (
      state == State::WaitingForTwoChildData &&
      type == BinaryTree::Type::Intermediate &&
      event == EventType::PacketReceived) { // int.2
    (new PacketBundle(
         stream->owner,
         stream,
         true,
         false,
         data_size,
         MemBus::Transmition::Usual))
        ->send_to_NPU();
    state = State::WaitingForOneChildData;
    return;
  } else if (
      state == State::WaitingForOneChildData &&
      type == BinaryTree::Type::Intermediate &&
      event == EventType::PacketReceived) { // int.3
    (new PacketBundle(
         stream->owner,
         stream,
         true,
         true,
         data_size,
         MemBus::Transmition::Usual))
        ->send_to_NPU();
    state = State::SendingDataToParent;
    return;
  } else if (
      reductions < 1 && type == BinaryTree::Type::Intermediate &&
      event == EventType::General) { // int.4
    reductions++;
    return;
  } else if (
      state == State::SendingDataToParent &&
      type == BinaryTree::Type::Intermediate) { // int.5
    sim_request snd_req;
    snd_req.srcRank = stream->owner->id;
    snd_req.dstRank = parent;
    snd_req.tag = stream->stream_num;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    snd_req.layerNum = layer_num;
    stream->owner->front_end_sim_send(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        parent,
        stream->stream_num,
        &snd_req,
        &Sys::handleEvent,
        nullptr);
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;
    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        stream->current_queue_id,
        stream->stream_num);
    stream->owner->front_end_sim_recv(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        parent,
        stream->stream_num,
        &rcv_req,
        &Sys::handleEvent,
        ehd);
    state = State::WaitingDataFromParent;
  } else if (
      state == State::WaitingDataFromParent &&
      type == BinaryTree::Type::Intermediate &&
      event == EventType::PacketReceived) { // int.6
    (new PacketBundle(
         stream->owner,
         stream,
         true,
         true,
         data_size,
         MemBus::Transmition::Usual))
        ->send_to_NPU();
    state = State::SendingDataToChilds;
    return;
  } else if (
      state == State::SendingDataToChilds &&
      type == BinaryTree::Type::Intermediate) {
    sim_request snd_req;
    snd_req.srcRank = stream->owner->id;
    snd_req.dstRank = left_child;
    snd_req.tag = stream->stream_num;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    snd_req.layerNum = layer_num;
    stream->owner->front_end_sim_send(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        left_child,
        stream->stream_num,
        &snd_req,
        &Sys::handleEvent,
        nullptr);
    sim_request snd_req2;
    snd_req2.srcRank = stream->owner->id;
    snd_req2.dstRank = left_child;
    snd_req2.tag = stream->stream_num;
    snd_req2.reqType = UINT8;
    snd_req2.vnet = this->stream->current_queue_id;
    snd_req2.layerNum = layer_num;
    stream->owner->front_end_sim_send(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        right_child,
        stream->stream_num,
        &snd_req2,
        &Sys::handleEvent,
        nullptr);
    exit();
    return;
  }

  else if (state == State::Begin && type == BinaryTree::Type::Root) { // root.1
    int only_child_id = left_child >= 0 ? left_child : right_child;
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;
    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        stream->current_queue_id,
        stream->stream_num);
    stream->owner->front_end_sim_recv(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        only_child_id,
        stream->stream_num,
        &rcv_req,
        &Sys::handleEvent,
        ehd);
    state = State::WaitingForOneChildData;
  } else if (
      state == State::WaitingForOneChildData &&
      type == BinaryTree::Type::Root) { // root.2
    (new PacketBundle(
         stream->owner,
         stream,
         true,
         true,
         data_size,
         MemBus::Transmition::Usual))
        ->send_to_NPU();
    state = State::SendingDataToChilds;
    return;
  } else if (
      state == State::SendingDataToChilds &&
      type == BinaryTree::Type::Root) { // root.2
    int only_child_id = left_child >= 0 ? left_child : right_child;
    sim_request snd_req;
    snd_req.srcRank = stream->owner->id;
    snd_req.dstRank = only_child_id;
    snd_req.tag = stream->stream_num;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    snd_req.layerNum = layer_num;
    stream->owner->front_end_sim_send(
        0,
        Sys::dummy_data,
        data_size,
        UINT8,
        only_child_id,
        stream->stream_num,
        &snd_req,
        &Sys::handleEvent,
        nullptr);
    exit();
    return;
  }
}
} // namespace AstraSim
