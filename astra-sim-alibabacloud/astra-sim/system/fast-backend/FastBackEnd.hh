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

#ifndef __FASTBACKEND_HH__
#define __FASTBACKEND_HH__

#include <cassert>
#include <iostream>
#include <map>
#include <tuple>

#include "astra-sim/system/AstraNetworkAPI.hh"
namespace AstraSim {
class FastBackEnd;

class WrapperData : public MetaData {
 public:
  enum class Type { FastSendRecv, DetailedSend, DetailedRecv, Undefined };
  Type type;
  void (*msg_handler)(void* fun_arg);
  void* fun_arg;
  WrapperData(Type type, void (*msg_handler)(void* fun_arg), void* fun_arg);
};

class WrapperRelayData : public WrapperData {
 public:
  int partner_node;
  uint64_t comm_size;
  double creation_time;
  FastBackEnd* fast_backend;
  WrapperRelayData(
      FastBackEnd* fast_backend,
      Type type,
      double creation_time,
      int partner_node,
      uint64_t comm_size,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
};

class InflightPairsMap {
 public:
  InflightPairsMap() = default;

  /**
   * Insert a pair to the inflight map.
   * @param src
   * @param dest
   * @param tag
   * @param communicationSize
   * @param simulationType
   */
  void insert(
      int src,
      int dest,
      int tag,
      int communicationSize,
      WrapperData::Type simulationType);

  /**
   * Remove a pair from the inflight map.
   * @param src
   * @param dest
   * @param tag
   * @param communicationSize
   * @return <true, WrapperData::Type> if matching pair exists, return <false,
   * Undefined> if not.
   */
  std::pair<bool, WrapperData::Type> pop(
      int src,
      int dest,
      int tag,
      int communicationSize);

  /**
   * Print out all residing pairs in the inflight map.
   */
  void print();

 private:
  // key: <src, dest, tag, communicationSize>
  std::map<std::tuple<int, int, int, int>, WrapperData::Type> inflightPairs;
};

class DynamicLatencyTable {
 public:
  DynamicLatencyTable() = default;

  /**
   * Insert a new latency data to the (src, dest) table.
   *
   * @param nodesPair (src, dest) pair
   * @param communicationSize communication size in bytes
   * @param latency
   */
  void insertLatencyData(
      std::pair<int, int> nodesPair,
      int communicationSize,
      double latency);

  /**
   * lookup whether there exists a exact matching latency value in the table.
   *
   * @param nodesPair (src, dest) pair for lookup
   * @param communicationSize communication size (in bytes) for lookup
   * @return <true, latency> is exists, <false, -1> if not.
   */
  std::pair<bool, double> lookupLatency(
      std::pair<int, int> nodesPair,
      int communicationSize);

  /**
   * Linearly interpolate/extrapolate two nearest points and predict latency.
   *
   * @param nodesPair (src, dest) pair for prediction
   * @param communicationSize communication size (in bytes) for prediction.
   * @return predicted traffic latency
   */
  double predictLatency(std::pair<int, int> nodesPair, int communicationSize);

  /**
   * Returns a boolean whether there's enough number of data points available
   * for prediction.
   * @param nodesPair (src, dest) pair for test
   * @return true if there are at least 2 data points, false if not.
   */
  bool canPredictLatency(std::pair<int, int> nodesPair);

  /**
   * Print latency data table.
   */
  void print();

 private:
  std::map<std::pair<int, int>, std::map<int, double>> latencyTables;
  std::map<std::pair<int, int>, int> latencyDataCountTable;

  /**
   * Insert new table for a new (src, dest) pair.
   * @param nodesPair (src, dest) pair
   */
  void insertNewTableForNode(std::pair<int, int> nodesPair);
};

class FastBackEnd : public AstraNetworkAPI {
 public:
  AstraNetworkAPI* wrapped_backend;
  static void handleEvent(void* arg);
  int sim_comm_size(sim_comm comm, int* size);
  // int sim_comm_get_rank();
  // int sim_comm_set_rank(sim_comm comm, int rank);
  int sim_finish();
  double sim_time_resolution();
  int sim_init(AstraMemoryAPI* MEM);
  timespec_t sim_get_time();
  void sim_schedule(
      timespec_t delta,
      void (*fun_ptr)(void* fun_arg),
      void* fun_arg);
  int sim_send(
      void* buffer,
      uint64_t count,
      int type,
      int dst,
      int tag,
      sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
  int sim_recv(
      void* buffer,
      uint64_t count,
      int type,
      int src,
      int tag,
      sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
  FastBackEnd(int rank, AstraNetworkAPI* wrapped_backend);

  void update_table_recv(
      double start,
      double finished,
      int src,
      uint64_t comm_size);
  void update_table_send(
      double start,
      double finished,
      int dst,
      uint64_t comm_size);
  int relay_send_request(
      void* buffer,
      uint64_t count,
      int type,
      int dst,
      int tag,
      sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
  int relay_recv_request(
      void* buffer,
      uint64_t count,
      int type,
      int src,
      int tag,
      sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
  int fast_send_recv_request(
      double delay,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);

 private:
  static InflightPairsMap inflightPairsMap;
  static DynamicLatencyTable dynamicLatencyTable;
};
} // namespace AstraSim

#endif
