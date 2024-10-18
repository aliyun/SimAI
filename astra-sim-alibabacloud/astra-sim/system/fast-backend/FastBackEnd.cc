/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include "FastBackEnd.hh"
namespace AstraSim {
WrapperData::WrapperData(
    WrapperData::Type type,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  this->type = type;
  this->msg_handler = msg_handler;
  this->fun_arg = fun_arg;
}

WrapperRelayData::WrapperRelayData(
    FastBackEnd* fast_backend,
    WrapperData::Type type,
    double creation_time,
    int partner_node,
    uint64_t comm_size,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg)
    : WrapperData(type, msg_handler, fun_arg) {
  this->creation_time = creation_time;
  this->partner_node = partner_node;
  this->comm_size = comm_size;
  this->fast_backend = fast_backend;
  this->type = type;
  this->msg_handler = msg_handler;
  this->fun_arg = fun_arg;
}

void InflightPairsMap::insert(
    int src,
    int dest,
    int tag,
    int communicationSize,
    WrapperData::Type simulationType) {
  auto insertionResult = inflightPairs.emplace(
      std::make_tuple(src, dest, tag, communicationSize), simulationType);

  assert(insertionResult.second);
}

std::pair<bool, WrapperData::Type> InflightPairsMap::pop(
    int src,
    int dest,
    int tag,
    int communicationSize) {
  auto searchResult =
      inflightPairs.find(std::make_tuple(src, dest, tag, communicationSize));

  if (searchResult == inflightPairs.end()) {
    return std::make_pair(false, WrapperData::Type::Undefined);
  }

  auto simulationType = searchResult->second;
  inflightPairs.erase(searchResult);

  return std::make_pair(true, simulationType);
}

void InflightPairsMap::print() {
  for (const auto& pair : inflightPairs) {
    std::cout << "src: " << std::get<0>(pair.first)
              << ", dest: " << std::get<1>(pair.first)
              << ", tag: " << std::get<2>(pair.first)
              << ", communicationSize: " << std::get<3>(pair.first)
              << std::endl;
  }
}

void DynamicLatencyTable::insertLatencyData(
    std::pair<int, int> nodesPair,
    int communicationSize,
    double latency) {
  auto latencyTable = latencyTables.find(nodesPair);

  if (latencyTable == latencyTables.end()) {
    insertNewTableForNode(nodesPair);

    latencyTable = latencyTables.find(nodesPair);
  }

  auto latencyInsertionResult =
      latencyTable->second.emplace(communicationSize, latency);

  if (latencyInsertionResult.second) {
    latencyDataCountTable[nodesPair]++;
  }
}

std::pair<bool, double> DynamicLatencyTable::lookupLatency(
    std::pair<int, int> nodesPair,
    int communicationSize) {
  auto latencyTable = latencyTables.find(nodesPair);
  if (latencyTable == latencyTables.end()) {
    return std::make_pair(false, -1);
  }

  auto latency = latencyTable->second.find(communicationSize);
  if (latency == latencyTable->second.end()) {
    return std::make_pair(false, -1);
  }

  return std::make_pair(true, latency->second);
}

double DynamicLatencyTable::predictLatency(
    std::pair<int, int> nodesPair,
    int communicationSize) {
  assert(canPredictLatency(nodesPair));
  assert(!lookupLatency(nodesPair, communicationSize).first);

  auto latencyTable = latencyTables[nodesPair];

  auto smallerPoint = latencyTable.upper_bound(communicationSize);
  auto largerPoint = latencyTable.lower_bound(communicationSize);

  if (smallerPoint == latencyTable.begin()) {
    smallerPoint = latencyTable.begin();
    largerPoint = std::next(smallerPoint);
  } else if (largerPoint == latencyTable.end()) {
    largerPoint = std::prev(latencyTable.end());
    smallerPoint = std::prev(largerPoint);
  } else {
    smallerPoint = std::prev(smallerPoint);
  }

  auto x1 = smallerPoint->first;
  auto y1 = smallerPoint->second;
  auto x2 = largerPoint->first;
  auto y2 = largerPoint->second;

  auto slope = (double)(y2 - y1) / (x2 - x1);
  auto predictedLatency = (int)(slope * (communicationSize - x1) + y1);

  assert(predictedLatency > 0);

  return predictedLatency;
}

bool DynamicLatencyTable::canPredictLatency(std::pair<int, int> nodesPair) {
  auto latencyDataCount = latencyDataCountTable.find(nodesPair);

  if (latencyDataCount == latencyDataCountTable.end()) {
    return false;
  }

  return latencyDataCount->second >= 2;
};

void DynamicLatencyTable::print() {
  for (const auto& latencyTable : latencyTables) {
    std::cout << "src: " << latencyTable.first.first
              << ", dest: " << latencyTable.first.second
              << ", datapoints: " << latencyDataCountTable[latencyTable.first]
              << std::endl;
    for (const auto& latencyPair : latencyTable.second) {
      std::cout << "\t- commSize: " << latencyPair.first
                << " - latency: " << latencyPair.second << std::endl;
    }
  }
}

void DynamicLatencyTable::insertNewTableForNode(std::pair<int, int> nodesPair) {
  auto latencyTable = latencyTables.find(nodesPair);
  auto latencyDataCount = latencyDataCountTable.find(nodesPair);
  assert(
      (latencyTable == latencyTables.end()) &&
      (latencyDataCount == latencyDataCountTable.end()));

  auto latencyTableInsertionResult =
      latencyTables.emplace(nodesPair, std::map<int, double>());
  auto latencyDataCountInsertionResult =
      latencyDataCountTable.emplace(nodesPair, 0);
  assert(
      latencyTableInsertionResult.second &&
      latencyDataCountInsertionResult.second);
}

InflightPairsMap FastBackEnd::inflightPairsMap;
DynamicLatencyTable FastBackEnd::dynamicLatencyTable;

void FastBackEnd::handleEvent(void* arg) {
  WrapperData* wrapperData = (WrapperData*)arg;
  if (wrapperData->type == WrapperData::Type::FastSendRecv) {
    (*(wrapperData->msg_handler))(wrapperData->fun_arg);
    delete wrapperData;
  } else if (
      ((WrapperRelayData*)wrapperData)->type ==
      WrapperData::Type::
          DetailedRecv) { // should be replaced by
                          // type==WrapperData::Type::DetailedRecv
    WrapperRelayData* wrapperRelayData = (WrapperRelayData*)arg;
    timespec_t current_time =
        wrapperRelayData->fast_backend->wrapped_backend->sim_get_time();
    wrapperRelayData->fast_backend->update_table_recv(
        wrapperRelayData->creation_time,
        current_time.time_val,
        wrapperRelayData->partner_node,
        wrapperRelayData->comm_size);

    (*(wrapperRelayData->msg_handler))(wrapperRelayData->fun_arg);
    delete wrapperRelayData;
  } else if (
      ((WrapperRelayData*)wrapperData)->type ==
      WrapperData::Type::
          DetailedSend) { // should be replaced by
                          // type==WrapperData::Type::DetailedSend
    WrapperRelayData* wrapperRelayData = (WrapperRelayData*)arg;
    timespec_t current_time =
        wrapperRelayData->fast_backend->wrapped_backend->sim_get_time();
    wrapperRelayData->fast_backend->update_table_send(
        wrapperRelayData->creation_time,
        current_time.time_val,
        wrapperRelayData->partner_node,
        wrapperRelayData->comm_size);
    (*(wrapperRelayData->msg_handler))(wrapperRelayData->fun_arg);
    delete wrapperRelayData;
  } else {
    std::cerr << "Event type undefined!" << std::endl;
  }
}

FastBackEnd::FastBackEnd(int rank, AstraNetworkAPI* wrapped_backend)
    : AstraNetworkAPI(rank) {
  this->wrapped_backend = wrapped_backend;
}

double FastBackEnd::sim_time_resolution() {
  return wrapped_backend->sim_time_resolution();
}

int FastBackEnd::sim_finish() {
  wrapped_backend->sim_finish();
  delete this;
  return 1;
}

// int FastBackEnd::sim_comm_get_rank(sim_comm comm, int *size) {
//    return wrapped_backend->sim_comm_get_rank(comm,size);
//}

// int FastBackEnd::sim_comm_set_rank(sim_comm comm, int rank) {
//    return wrapped_backend->sim_comm_set_rank(comm,rank);
//}

int FastBackEnd::sim_comm_size(sim_comm comm, int* size) {
  return wrapped_backend->sim_comm_size(comm, size);
}

timespec_t FastBackEnd::sim_get_time() {
  return wrapped_backend->sim_get_time();
}

int FastBackEnd::sim_init(AstraMemoryAPI* MEM) {
  return wrapped_backend->sim_init(MEM);
}

void FastBackEnd::sim_schedule(
    timespec_t delta,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
  wrapped_backend->sim_schedule(delta, fun_ptr, fun_arg);
  return;
}

void FastBackEnd::update_table_send(
    double start,
    double finished,
    int dst,
    uint64_t comm_size) {
  auto latency = finished - start;
  auto src = sim_comm_get_rank();
  dynamicLatencyTable.insertLatencyData(
      std::make_pair(src, dst), comm_size, latency);
}

void FastBackEnd::update_table_recv(
    double start,
    double finished,
    int src,
    uint64_t comm_size) {
  auto latency = finished - start;
  auto dst = sim_comm_get_rank();
  dynamicLatencyTable.insertLatencyData(
      std::make_pair(src, dst), comm_size, latency);
}

int FastBackEnd::relay_recv_request(
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  timespec_t current_time = wrapped_backend->sim_get_time();
  WrapperRelayData* wrapperData = new WrapperRelayData(
      this,
      WrapperData::Type::DetailedRecv,
      current_time.time_val,
      src,
      count,
      msg_handler,
      fun_arg);
  return wrapped_backend->sim_recv(
      buffer,
      count,
      type,
      src,
      tag,
      request,
      &FastBackEnd::handleEvent,
      wrapperData);
}

int FastBackEnd::relay_send_request(
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  timespec_t current_time = wrapped_backend->sim_get_time();
  WrapperRelayData* wrapperData = new WrapperRelayData(
      this,
      WrapperData::Type::DetailedSend,
      current_time.time_val,
      dst,
      count,
      msg_handler,
      fun_arg);
  return wrapped_backend->sim_send(
      buffer,
      count,
      type,
      dst,
      tag,
      request,
      &FastBackEnd::handleEvent,
      wrapperData);
}

int FastBackEnd::fast_send_recv_request(
    double delay,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  timespec_t delta;
  delta.time_res = time_type_e::NS;
  delta.time_val = delay;
  WrapperData* wrapperData =
      new WrapperData(WrapperData::Type::FastSendRecv, msg_handler, fun_arg);
  wrapped_backend->sim_schedule(
      delta, &FastBackEnd::handleEvent, (void*)wrapperData);
  return 1;
}

int FastBackEnd::sim_send(
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  auto src = sim_comm_get_rank();
  auto srcDestPair = std::make_pair(src, dst);

  auto inflightPair = inflightPairsMap.pop(src, dst, tag, count);

  if (inflightPair.first) {
    // inflight pair exists. Use that method.
    switch (inflightPair.second) {
      case WrapperData::Type::FastSendRecv: {
        // Use fast send. either lookup latency or predicted one is used.
        auto lookupResult =
            dynamicLatencyTable.lookupLatency(srcDestPair, count);
        if (lookupResult.first) {
          // lookup result exists
          return fast_send_recv_request(
              lookupResult.second, msg_handler, fun_arg);
        }

        // lookup doesn't exist - use prediction
        auto predictedLatency =
            dynamicLatencyTable.predictLatency(srcDestPair, count);
        return fast_send_recv_request(predictedLatency, msg_handler, fun_arg);
      }

      case WrapperData::Type::DetailedRecv: {
        // relay send request to ns3
        return relay_send_request(
            buffer, count, type, dst, tag, request, msg_handler, fun_arg);
      }

      default: {
        // should not fall here
        std::cerr << "sim_send inflight pair error" << std::endl;
        exit(-1);
      }
    }
  }

  // inflight pair doesn't exist -- initiate one
  auto lookupResult = dynamicLatencyTable.lookupLatency(srcDestPair, count);

  if (lookupResult.first) {
    // lookup result exists; use this latency for fast simulation
    inflightPairsMap.insert(
        src, dst, tag, count, WrapperData::Type::FastSendRecv);
    return fast_send_recv_request(lookupResult.second, msg_handler, fun_arg);
  }

  // lookup doesn't exist
  if (dynamicLatencyTable.canPredictLatency(srcDestPair)) {
    // prediction available
    // for 10%, don't use prediction and relay the request.
    if ((rand() % 100) < 10) {
      inflightPairsMap.insert(
          src, dst, tag, count, WrapperData::Type::DetailedSend);
      return relay_send_request(
          buffer, count, type, dst, tag, request, msg_handler, fun_arg);
    }

    // for 90% cases, use prediction result.
    inflightPairsMap.insert(
        src, dst, tag, count, WrapperData::Type::FastSendRecv);
    auto predictedLatency =
        dynamicLatencyTable.predictLatency(srcDestPair, count);
    return fast_send_recv_request(predictedLatency, msg_handler, fun_arg);
  }

  // prediction not available -- should relay to ns3
  inflightPairsMap.insert(
      src, dst, tag, count, WrapperData::Type::DetailedSend);
  return relay_send_request(
      buffer, count, type, dst, tag, request, msg_handler, fun_arg);
}

int FastBackEnd::sim_recv(
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  auto dst = sim_comm_get_rank();
  auto srcDestPair = std::make_pair(src, dst);

  auto inflightPair = inflightPairsMap.pop(src, dst, tag, count);

  if (inflightPair.first) {
    // inflight pair exists. Use that method.
    switch (inflightPair.second) {
      case WrapperData::Type::FastSendRecv: {
        // Use fast recv. either lookup latency or predicted one is used.
        auto lookupResult =
            dynamicLatencyTable.lookupLatency(srcDestPair, count);
        if (lookupResult.first) {
          // lookup result exists. use this latency
          return fast_send_recv_request(
              lookupResult.second, msg_handler, fun_arg);
        }

        // lookup doesn't exist. use prediction.
        auto predictedLatency =
            dynamicLatencyTable.predictLatency(srcDestPair, count);
        return fast_send_recv_request(predictedLatency, msg_handler, fun_arg);
      }

      case WrapperData::Type::DetailedSend: {
        // relay to ns3
        return relay_recv_request(
            buffer, count, type, src, tag, request, msg_handler, fun_arg);
      }

      default: {
        // should not fall here
        std::cerr << "sim_recv inflight pair error" << std::endl;
        exit(-1);
      }
    }
  }

  // inflight pair doesn't exist -- initiate one
  auto lookupResult = dynamicLatencyTable.lookupLatency(srcDestPair, count);

  if (lookupResult.first) {
    // lookup result exists; use fast simulation
    inflightPairsMap.insert(
        src, dst, tag, count, WrapperData::Type::FastSendRecv);
    return fast_send_recv_request(lookupResult.second, msg_handler, fun_arg);
  }

  // lookup doesn't exist
  if (dynamicLatencyTable.canPredictLatency(srcDestPair)) {
    // prediction available
    // for 10%, don't use prediction and relay the request.
    if ((rand() % 100) < 10) {
      inflightPairsMap.insert(
          src, dst, tag, count, WrapperData::Type::DetailedRecv);
      return relay_recv_request(
          buffer, count, type, src, tag, request, msg_handler, fun_arg);
    }

    // for 90% cases, use prediction result.
    inflightPairsMap.insert(
        src, dst, tag, count, WrapperData::Type::FastSendRecv);
    auto predictedLatency =
        dynamicLatencyTable.predictLatency(srcDestPair, count);
    return fast_send_recv_request(predictedLatency, msg_handler, fun_arg);
  }

  // cannot predict -- relay to ns3
  inflightPairsMap.insert(
      src, dst, tag, count, WrapperData::Type::DetailedRecv);
  return relay_recv_request(
      buffer, count, type, src, tag, request, msg_handler, fun_arg);
}
} // namespace AstraSim
