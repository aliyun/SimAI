/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMON_HH__
#define __COMMON_HH__
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include "AstraNetworkAPI.hh"

enum class GPUType { A100, A800, H100, H800, NONE, H20};

namespace AstraSim {
#define CLOCK_PERIOD 1
#define FREQ (1000.0 / CLOCK_PERIOD)
#define GBps 1.0 / (1024 * 1024 * 1024)
typedef unsigned long long Tick;
enum class ComType {
  None,
  Reduce_Scatter,
  All_Gather,
  All_Reduce,
  All_to_All,
  All_Reduce_All_to_All,
  All_Reduce_NVLS
};
enum class CollectiveOptimization { Baseline, LocalBWAware };
enum class CollectiveImplementationType {
  Ring, 
  OneRing,
  Direct, 
  OneDirect,
  AllToAll,
  DoubleBinaryTreeLocalAllToAll,
  LocalRingNodeA2AGlobalDBT,
  HierarchicalRing,
  DoubleBinaryTree,
  HalvingDoubling,  
  OneHalvingDoubling,
  NcclFlowModel,
  NcclTreeFlowModel,
};
enum class CollectiveBarrier { Blocking, Non_Blocking };
enum class SchedulingPolicy { LIFO, FIFO, HIGHEST, None };
enum class IntraDimensionScheduling {
  FIFO,
  RG,
  SmallestFirst,
  LessRemainingPhaseFirst
};
enum class InterDimensionScheduling {
  Ascending,
  OnlineGreedy,
  RoundRobin,
  OfflineGreedy,
  OfflineGreedyFlex
};
enum class InjectionPolicy {
  Infinite,
  Aggressive,
  SemiAggressive,
  ExtraAggressive,
  Normal
};
enum class PacketRouting { Hardware, Software };
enum class BusType { Both, Shared, Mem };
enum class StreamState {
  Created,
  Transferring,
  Ready,
  Executing,
  Zombie,
  Dead
};
enum class EventType {
  NONE,
  RendezvousSend,
  RendezvousRecv,
  CallEvents,
  PacketReceived,
  PacketSent,
  PacketSentFinshed,
  WaitForVnetTurn,
  NCCL_General,
  General,
  TX_DMA,
  RX_DMA,
  Wight_Grad_Comm_Finished,
  Input_Grad_Comm_Finished,
  Fwd_Comm_Finished,
  Wight_Grad_Comm_Finished_After_Delay,
  Input_Grad_Comm_Finished_After_Delay,
  Fwd_Comm_Finished_After_Delay,
  Workload_Wait,
  Reduction_Ready,
  Rec_Finished,
  Send_Finished,
  Processing_Finished,
  Delivered,
  NPU_to_MA,
  MA_to_NPU,
  Read_Port_Free,
  Write_Port_Free,
  Apply_Boost,
  Stream_Transfer_Started,
  Stream_Ready,
  Consider_Process,
  Consider_Retire,
  Consider_Send_Back,
  StreamInit,
  StreamsFinishedIncrease,
  CommProcessingFinished,
  NotInitialized
};
class CloneInterface {
 public:
  virtual CloneInterface* clone() const = 0;
  virtual ~CloneInterface() = default;
};
class CollectiveImplementation : public CloneInterface {
 public:
  CollectiveImplementationType type;
  CollectiveImplementation(CollectiveImplementationType type) {
    this->type = type;
  };
  virtual CloneInterface* clone() const {
    return new CollectiveImplementation(*this);
  }
};
class DirectCollectiveImplementation : public CollectiveImplementation {
 public:
  int direct_collective_window;
  CloneInterface* clone() const {
    return new DirectCollectiveImplementation(*this);
  };
  DirectCollectiveImplementation(
      CollectiveImplementationType type,
      int direct_collective_window)
      : CollectiveImplementation(type) {
    this->direct_collective_window = direct_collective_window;
  }
};
} // namespace AstraSim
#endif
