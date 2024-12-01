/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Sys.hh"
#include "BaseStream.hh"
#include "DataSet.hh"
#include "MemBus.hh"
#include "QueueLevels.hh"
#include "SimRecvCaller.hh"
#include "SimSendCaller.hh"
#include "StreamBaseline.hh"
#include "Common.hh"
#include "RendezvousRecvData.hh"
#include "RendezvousSendData.hh"
#include "astra-sim/system/collective/AllToAll.hh"
#include "astra-sim/system/collective/DoubleBinaryTreeAllReduce.hh"
#include "astra-sim/system/collective/HalvingDoubling.hh"
#include "astra-sim/system/collective/Ring.hh"
#include "astra-sim/system/collective/NcclTreeFlowModel.hh"
#include "astra-sim/system/scheduling/OfflineGreedy.hh"
#include "astra-sim/system/topology/BasicLogicalTopology.hh"
#include "astra-sim/system/topology/DoubleBinaryTreeTopology.hh"
#include "astra-sim/system/topology/GeneralComplexTopology.hh"
#include "astra-sim/system/topology/LocalRingGlobalBinaryTree.hh"
#include "astra-sim/system/topology/LocalRingNodeA2AGlobalDBT.hh"
#include "astra-sim/system/topology/Torus3D.hh"
#include "astra-sim/system/MockNcclLog.h"

#include <algorithm>
#include <cmath>
#include <numeric>

MockNccl::MockNcclGroup* GlobalGroup = nullptr;

namespace AstraSim {
std::atomic<bool> Sys::g_sys_inCriticalSection(false);
Tick Sys::offset = 0;
uint8_t* Sys::dummy_data = new uint8_t[2];
std::vector<Sys*> Sys::all_generators;

Sys::~Sys() {
  end_sim_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::minutes>(
      end_sim_time - start_sim_time);
  if (id == 0) {
    auto timenow =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "*****" << std::endl
              << "Time to exit: " << ctime(&timenow)
              << "all-reduce Collective implementation: "
              << inp_all_reduce_implementation << std::endl
              << "reduce-scatter Collective implementation: "
              << inp_reduce_scatter_implementation << std::endl
              << "all-gather Collective implementation: "
              << inp_all_gather_implementation << std::endl
              << "all-to-all Collective implementation: "
              << inp_all_to_all_implementation << std::endl
              << "Collective optimization: " << inp_collective_optimization
              << std::endl
              << "Total sim duration: " << duration.count() / 60 << ":"
              << duration.count() % 60 << " hours" << std::endl
              << "Total streams injected: " << streams_injected << std::endl
              << "Total streams finished: " << streams_finished << std::endl
              << "Percentage of finished streams: "
              << (((double)streams_finished) / streams_injected) * 100 << " %"
              << std::endl
              << "*****" << std::endl;
  }
  #ifndef PHY_MTP
  all_generators[id + npu_offset] = nullptr;
  for (auto lt : logical_topologies) {
    delete lt.second;
  }
  logical_topologies.clear();
  for (auto ci : all_reduce_implementation_per_dimension) {
    delete ci;
  }
  for (auto ci : reduce_scatter_implementation_per_dimension) {
    delete ci;
  }
  for (auto ci : all_gather_implementation_per_dimension) {
    delete ci;
  }
  for (auto ci : all_to_all_implementation_per_dimension) {
    delete ci;
  }
  if (scheduler_unit != nullptr)
    delete scheduler_unit;
  if (vLevels != nullptr)
    delete vLevels;
  if (memBus != nullptr)
    delete memBus;
  if (workload != nullptr)
    delete workload;
  if (offline_greedy != nullptr)
    delete offline_greedy;
  bool shouldExit = true;
  
  for(int i = 0; i < num_gpus; ++ i) {
    auto& a = all_generators[i];
    if (a != nullptr) {
      shouldExit = false;
      break;
    }
  }

  if (shouldExit) {
    exitSimLoop("Exiting");
  }
  #else
    exitSimLoop("Exiting");
  #endif
}

Sys::Sys(
    AstraNetworkAPI* NI,
    AstraMemoryAPI* MEM,
    int id,
    int npu_offset,
    int num_passes,
    std::vector<int> physical_dims,
    std::vector<int> queues_per_dim,
    std::string my_sys,
    std::string my_workload,
    float comm_scale,
    float compute_scale,
    float injection_scale,
    int total_stat_rows,
    int stat_row,
    std::string path,
    std::string run_name,
    bool seprate_log,
    bool rendezvous_enabled,
    GPUType _gpu_type,
    std::vector<int>_all_gpus,
    std::vector<int>_NVSwitchs,
    int _ngpus_per_node) {
  scheduler_unit = nullptr;
  vLevels = nullptr;
  memBus = nullptr;
  workload = nullptr;
  offline_greedy = nullptr;
  this->initialized = false;
  this->intra_dimension_scheduling = IntraDimensionScheduling::FIFO;
  this->inter_dimension_scheduling = InterDimensionScheduling::Ascending;
  round_robin_inter_dimension_scheduler = 0;
  this->last_scheduled_collective = 0;
  this->dim_to_break = -1;

  start_sim_time = std::chrono::high_resolution_clock::now();
  this->NI = NI;
  this->MEM = MEM;
  this->id = id;
  this->npu_offset=npu_offset;
  this->method = "baseline";
  this->finished_workloads = 0;
  this->streams_finished = 0;
  this->streams_injected = 0;
  this->first_phase_streams = 0;
  this->total_running_streams = 0;
  this->priority_counter = 0;
  this->comm_scale = comm_scale;
  this->compute_scale = compute_scale;
  this->injection_scale = injection_scale;
  this->inp_model_shared_bus = 0;
  this->inp_boost_mode = 0;
  this->num_channels = 1;
  this->processing_latency = 10;
  this->communication_delay = 10;
  this->local_reduction_delay = 1;
  this->active_chunks_per_dimension = 1;
  this->seprate_log = seprate_log;
  this->rendezvous_enabled = rendezvous_enabled;
  this->NVSwitchs = _NVSwitchs;
  this->all_gpus = _all_gpus;
  this->gpu_type = _gpu_type;
  this->ngpus_per_node = _ngpus_per_node;
  if ((id + npu_offset + 1) > all_generators.size()) {
    all_generators.resize(id + npu_offset + 1);
  }
  all_generators[id+npu_offset] = this;

  inp_scheduling_policy = "LIFO";
  communication_delay = 10 * injection_scale;
  active_chunks_per_dimension = 1;
  preferred_dataset_splits = 1;
  inp_boost_mode = 0;
  inp_all_reduce_implementation = "ncclRingFlowModel";
  inp_all_gather_implementation = "ncclRingFlowModel";
  inp_reduce_scatter_implementation = "ncclRingFlowModel";
  inp_all_to_all_implementation = "direct";
  inp_collective_optimization = "baseline";
  bool result = post_process_inputs();

  if (result == false) {
    sys_panic(
        "Unable to initialize the system layer because the file can not be openned");
  }

  this->pending_events = 0;

  int total_disabled = 0;
  this->physical_dims = physical_dims;
  this->queues_per_dim = queues_per_dim;
  int element = 0;
  all_queues = 0;
  total_nodes = 1;
  for (int current_dim = 0; current_dim < queues_per_dim.size();
       current_dim++) {
    all_queues += queues_per_dim[current_dim];
    bool enabled = !boost_mode;
    if (id % total_nodes == 0 &&
        id < total_nodes * physical_dims[current_dim]) {
      enabled = true;
    }
    if (!enabled) {
      total_disabled += queues_per_dim[current_dim];
    }
    if (physical_dims[current_dim] >= 1) {
      total_nodes *= physical_dims[current_dim];
    }
    for (int j = 0; j < queues_per_dim[current_dim]; j++) {
      std::list<BaseStream*> temp;
      active_Streams[element] = temp;
      std::list<int> pri;
      stream_priorities[element] = pri;
      element++;
    }
  }
  if (all_queues == total_disabled) {
    NI->enabled = false;
    std::cout << "Node " << id << " has been totally disabled" << std::endl;
  }
  concurrent_streams =
      (int)std::ceil(((double)active_chunks_per_dimension) / queues_per_dim[0]);
  active_first_phase = 100000000;
  if (id == 0) {
    std::cout
        << "The final active chunks per dimension 1 after allocating to queues is: "
        << concurrent_streams * queues_per_dim[0] << std::endl;
  }
  max_running = 100000000;
  scheduler_unit = new SchedulerUnit(
      this,
      queues_per_dim,
      max_running,
      active_first_phase,
      concurrent_streams);
  vLevels = new QueueLevels(queues_per_dim, 0, NI->get_backend_type());

  logical_topologies["AllReduce"] = new GeneralComplexTopology(
      id, physical_dims, all_reduce_implementation_per_dimension);
  logical_topologies["ReduceScatter"] = new GeneralComplexTopology(
      id, physical_dims, reduce_scatter_implementation_per_dimension);
  logical_topologies["AllGather"] = new GeneralComplexTopology(
      id, physical_dims, all_gather_implementation_per_dimension);
  logical_topologies["AllToAll"] = new GeneralComplexTopology(
      id, physical_dims, all_to_all_implementation_per_dimension);
  stream_counter = 0;
  if (id == 0) {
    std::atexit(exiting);
    std::cout << "total nodes: " << total_nodes << std::endl;
  }
  NI->sim_init(MEM);
  memBus = new MemBus(
      "NPU",
      "MA",
      this,
      inp_L,
      inp_o,
      inp_g,
      inp_G,
      model_shared_bus,
      communication_delay,
      true);
  workload = new Workload(
      run_name,
      this,
      my_workload,
      num_passes,
      total_stat_rows,
      stat_row,
      path,
      this->seprate_log);
  if (workload->initialized == false) {
    sys_panic(
        "Unable to initialize the workload layer because it can not open the workload file");
    return;
  }
  #if defined(NS3_MTP) || defined(NS3_MPI) || defined(PHY_MTP)
  result = mock_nccl_grobal_group_init();
  if(result == false) {
    sys_panic(
        "Unable to initialize the system grobal group because the file can not be openned");
  }
  result = mock_nccl_comms_init();
  if (result == false) {
    sys_panic(
        "Unable to initialize the system mockncclComm because the file can not be openned");
  }
  #endif
  if (inter_dimension_scheduling == InterDimensionScheduling::OfflineGreedy ||
      inter_dimension_scheduling ==
          InterDimensionScheduling::OfflineGreedyFlex) {
    offline_greedy = new OfflineGreedy(this);
  }
  this->initialized = true;
}

int Sys::break_dimension(int model_parallel_npu_group) {
  if (model_parallel_npu_group == 1) {
    return -1;
  }
  int dimension_to_break = 0;
  int all_npus = 1;
  for (; dimension_to_break < physical_dims.size(); dimension_to_break++) {
    if (all_npus * physical_dims[dimension_to_break] <
        model_parallel_npu_group) {
      all_npus *= physical_dims[dimension_to_break];
    } else if (
        all_npus * physical_dims[dimension_to_break] >
        model_parallel_npu_group) {
      for (auto lt : logical_topologies) {
        delete lt.second;
      }
      logical_topologies.clear();

      delete scheduler_unit;
      delete vLevels;
      std::vector<int>::iterator levelIterator = queues_per_dim.begin();
      std::advance(levelIterator, dimension_to_break);
      queues_per_dim.insert(levelIterator, queues_per_dim[dimension_to_break]);
      scheduler_unit = new SchedulerUnit(
          this,
          queues_per_dim,
          max_running,
          active_first_phase,
          concurrent_streams);
      vLevels = new QueueLevels(queues_per_dim, 0, NI->get_backend_type());

      int first_subdim = model_parallel_npu_group / all_npus;
      int second_subdim = physical_dims[dimension_to_break] / first_subdim;
      std::vector<int> logical_dims;

      for (int dim = 0; dim < physical_dims.size(); dim++) {
        if (dim != dimension_to_break) {
          logical_dims.push_back(physical_dims[dim]);
        } else {
          logical_dims.push_back(first_subdim);
          logical_dims.push_back(second_subdim);
        }
      }

      std::vector<CollectiveImplementation*>::iterator it =
          all_reduce_implementation_per_dimension.begin();
      if (all_reduce_implementation_per_dimension.size() > dimension_to_break) {
        std::advance(it, dimension_to_break);
      } else {
        std::advance(it, all_reduce_implementation_per_dimension.size());
      }
      CollectiveImplementation* replicate =
          (CollectiveImplementation*)(*it)->clone();
      all_reduce_implementation_per_dimension.insert(it, replicate);

      it = reduce_scatter_implementation_per_dimension.begin();
      if (reduce_scatter_implementation_per_dimension.size() >
          dimension_to_break) {
        std::advance(it, dimension_to_break);
      } else {
        std::advance(it, reduce_scatter_implementation_per_dimension.size());
      }
      replicate = (CollectiveImplementation*)(*it)->clone();
      reduce_scatter_implementation_per_dimension.insert(it, replicate);

      it = all_gather_implementation_per_dimension.begin();
      if (all_gather_implementation_per_dimension.size() > dimension_to_break) {
        std::advance(it, dimension_to_break);
      } else {
        std::advance(it, all_gather_implementation_per_dimension.size());
      }
      replicate = (CollectiveImplementation*)(*it)->clone();
      all_gather_implementation_per_dimension.insert(it, replicate);

      it = all_to_all_implementation_per_dimension.begin();
      if (all_to_all_implementation_per_dimension.size() > dimension_to_break) {
        std::advance(it, dimension_to_break);
      } else {
        std::advance(it, all_to_all_implementation_per_dimension.size());
      }
      replicate = (CollectiveImplementation*)(*it)->clone();
      all_to_all_implementation_per_dimension.insert(it, replicate);
      logical_topologies["AllReduce"] = new GeneralComplexTopology(
          id, logical_dims, all_reduce_implementation_per_dimension);
      logical_topologies["ReduceScatter"] = new GeneralComplexTopology(
          id, logical_dims, reduce_scatter_implementation_per_dimension);
      logical_topologies["AllGather"] = new GeneralComplexTopology(
          id, logical_dims, all_gather_implementation_per_dimension);
      logical_topologies["AllToAll"] = new GeneralComplexTopology(
          id, logical_dims, all_to_all_implementation_per_dimension);
      this->logical_broken_dims = logical_dims;
      this->dim_to_break = dimension_to_break;
      

      return dimension_to_break;
    } else if (
        all_npus * physical_dims[dimension_to_break] ==
        model_parallel_npu_group) {
      return dimension_to_break;
    }
  }
  return -1;
}
int Sys::get_layer_numbers(std::string workload_input) {
  return Workload::get_layer_numbers(workload_input);
}
int Sys::get_priority(SchedulingPolicy pref_scheduling) {
  if (pref_scheduling == SchedulingPolicy::None) {
    if (scheduling_policy == SchedulingPolicy::LIFO) {
      return priority_counter++;
    } else {
      return priority_counter--;
    }
  } else if (pref_scheduling == SchedulingPolicy::HIGHEST) {
    return 100000000;
  } else {
    if (scheduling_policy == SchedulingPolicy::LIFO) {
      return priority_counter++;
    } else {
      return priority_counter--;
    }
  }
}
int Sys::rendezvous_sim_send(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  RendezvousSendData* rsd = new RendezvousSendData(
      id, this, buffer, count, type, dst, tag, *request, msg_handler, fun_arg);
  sim_request newReq = *request;
  uint64_t rendevouz_size = 8192;
  newReq.dstRank = request->srcRank;
  newReq.srcRank = request->dstRank;
  newReq.reqCount = rendevouz_size;
  int newTag = tag + 500000000;
  newReq.tag = newTag;
  sim_recv(
      delay,
      buffer,
      rendevouz_size,
      type,
      dst,
      newTag,
      &newReq,
      &Sys::handleEvent,
      rsd);
  return 1;
}
int Sys::sim_send(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  if (delay == 0 && fun_arg == nullptr) {
    Sys::sysCriticalSection cs;
      
    SendPacketEventHandlerData* fun_arg_tmp =
        new SendPacketEventHandlerData(this, id+npu_offset, dst, tag);
    fun_arg = (void*)fun_arg_tmp;
    if (is_there_pending_sends.find(std::make_pair(dst, tag)) == is_there_pending_sends.end() ||
    is_there_pending_sends[std::make_pair(dst, tag)] == false) {
      is_there_pending_sends[std::make_pair(dst, tag)] = true;
      cs.ExitSection();
    } else {
      if (pending_sends.find(std::make_pair(dst, tag)) ==
          pending_sends.end()) {
        std::list<SimSendCaller*> tmp;
        pending_sends[std::make_pair(dst, tag)] = tmp;
      }
      pending_sends[std::make_pair(dst, tag)].push_back(
          new SimSendCaller(
              this,
              buffer,
              count,
              type,
              dst,
              tag,
              *request,
              msg_handler,
              fun_arg));
      
      cs.ExitSection();
      return 1;
    }
  }

  if (delay == 0) {
    NI->sim_send(buffer, count, type, dst, tag, request, msg_handler, fun_arg);
  } else {
    try_register_event(
        new SimSendCaller(
            this,
            buffer,
            count,
            type,
            dst,
            tag,
            *request,
            msg_handler,
            fun_arg),
        EventType::General,
        nullptr,
        delay);
  }
  return 1;
}
int Sys::front_end_sim_send(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  if (rendezvous_enabled) {
    return rendezvous_sim_send(
        delay, buffer, count, type, dst, tag, request, msg_handler, fun_arg);
  } else {
    return sim_send(
        delay, buffer, count, type, dst, tag, request, msg_handler, fun_arg);
  }
}
int Sys::rendezvous_sim_recv(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  RendezvousRecvData* rrd = new RendezvousRecvData(
      id, this, buffer, count, type, src, tag, *request, msg_handler, fun_arg);
  sim_request newReq = *request;
  uint64_t rendevouz_size = 8192;
  newReq.dstRank = request->srcRank;
  newReq.srcRank = request->dstRank;
  newReq.reqCount = rendevouz_size;
  int newTag = tag + 500000000;
  newReq.tag = newTag;
  sim_send(
      delay,
      buffer,
      rendevouz_size,
      type,
      src,
      newTag,
      &newReq,
      &Sys::handleEvent,
      rrd);
  return 1;
}
int Sys::sim_recv(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  if (delay == 0) {
    NI->sim_recv(buffer, count, type, src, tag, request, msg_handler, fun_arg);
  } else {
    try_register_event(
        new SimRecvCaller(
            this,
            buffer,
            count,
            type,
            src,
            tag,
            *request,
            msg_handler,
            fun_arg),
        EventType::General,
        nullptr,
        delay);
  }
  return 1;
}
int Sys::front_end_sim_recv(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  if (rendezvous_enabled) {
    return rendezvous_sim_recv(
        delay, buffer, count, type, src, tag, request, msg_handler, fun_arg);
  } else {
    return sim_recv(
        delay, buffer, count, type, src, tag, request, msg_handler, fun_arg);
  }
}
Tick Sys::mem_read(uint64_t bytes) {
  if (MEM == nullptr) {
    return 10;
  }
  uint64_t delay_ns = MEM->npu_mem_read(bytes);
  Tick delay_cycles = delay_ns / CLOCK_PERIOD;
  return delay_cycles;
}
Tick Sys::mem_write(uint64_t bytes) {
  if (MEM == nullptr) {
    return 10;
  }
  uint64_t delay_ns = MEM->npu_mem_write(bytes);
  Tick delay_cycles = delay_ns / CLOCK_PERIOD;
  return delay_cycles;
}
std::string Sys::trim(
    const std::string& str,
    const std::string& whitespace = " \t") {
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos)
    return ""; 

  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;

  return str.substr(strBegin, strRange);
}
std::vector<CollectiveImplementation*> Sys::
    generate_collective_implementation_from_input(std::string input) {
  std::vector<std::string> inputs_per_dimension = split_string(input, "_");
  std::vector<CollectiveImplementation*> result;
  for (std::string dimension_input : inputs_per_dimension) {
    if (dimension_input == "ring") {
      result.push_back(
          new CollectiveImplementation(CollectiveImplementationType::Ring));
    } else if (dimension_input == "oneRing") {
      result.push_back(
          new CollectiveImplementation(CollectiveImplementationType::OneRing));
    } else if (dimension_input == "doubleBinaryTree") {
      result.push_back(new CollectiveImplementation(
          CollectiveImplementationType::DoubleBinaryTree));
    } else if (dimension_input.rfind("direct", 0) == 0) {
      int window = -1;
      if (dimension_input != "direct") {
        window = std::stoi(dimension_input.substr(6, 5));
      }
      result.push_back(new DirectCollectiveImplementation(
          CollectiveImplementationType::Direct, window));
    } else if (dimension_input.rfind("oneDirect", 0) == 0) {
      int window = -1;
      if (dimension_input != "oneDirect") {
        window = std::stoi(dimension_input.substr(9, 5));
      }
      result.push_back(new DirectCollectiveImplementation(
          CollectiveImplementationType::OneDirect, window));
    } else if (dimension_input == "halvingDoubling") {
      result.push_back(new CollectiveImplementation(
          CollectiveImplementationType::HalvingDoubling));
    } else if (dimension_input == "oneHalvingDoubling") {
      result.push_back(new CollectiveImplementation(
          CollectiveImplementationType::OneHalvingDoubling));
    } else if(dimension_input == "ncclRingFlowModel") {
      result.push_back(new CollectiveImplementation(
          CollectiveImplementationType::NcclRingFlowModel));
    } else if(dimension_input == "ncclRingTreeModel") {
      result.push_back(new CollectiveImplementation(
          CollectiveImplementationType::NcclTreeFlowModel));
    } else {
      sys_panic(
          "Cannot interpret collective implementations. Please check the collective implementations in the sys"
          "input file");
    }
  }
  return result;
}
bool Sys::parse_var(std::string var, std::string value) {
  var = trim(var);
  value = trim(value);
  if (id == 0) {
    std::cout << "Var is: " << var << " ,val is: " << value << std::endl;
  }
  if (var == "scheduling-policy:") {
    inp_scheduling_policy = value;
  } else if (var == "all-reduce-implementation:") {
    std::stringstream mval(value);
    mval >> inp_all_reduce_implementation;
  } else if (var == "reduce-scatter-implementation:") {
    std::stringstream mval(value);
    mval >> inp_reduce_scatter_implementation;
  } else if (var == "all-gather-implementation:") {
    std::stringstream mval(value);
    mval >> inp_all_gather_implementation;
  } else if (var == "all-to-all-implementation:") {
    std::stringstream mval(value);
    mval >> inp_all_to_all_implementation;
  } else if (var == "collective-optimization:") {
    std::stringstream mval(value);
    mval >> inp_collective_optimization;
  } else if (var == "endpoint-delay:") {
    std::stringstream mval(value);
    mval >> communication_delay;
    communication_delay = communication_delay * injection_scale;
  } else if (var == "local-reduction-delay:") {
    std::stringstream mval(value);
    mval >> local_reduction_delay;
  } else if (var == "active-chunks-per-dimension:") {
    std::stringstream mval(value);
    mval >> active_chunks_per_dimension;
  } else if (var == "L:") {
    std::stringstream mval(value);
    mval >> inp_L;
  } else if (var == "o:") {
    std::stringstream mval(value);
    mval >> inp_o;
  } else if (var == "g:") {
    std::stringstream mval(value);
    mval >> inp_g;
  } else if (var == "G:") {
    std::stringstream mval(value);
    mval >> inp_G;
  } else if (var == "model-shared-bus:") {
    std::stringstream mval(value);
    mval >> inp_model_shared_bus;
  } else if (var == "preferred-dataset-splits:") {
    std::stringstream mval(value);
    mval >> preferred_dataset_splits;
  } else if (var == "boost-mode:") {
    std::stringstream mval(value);
    mval >> inp_boost_mode;
  } else if (var == "intra-dimension-scheduling:") {
    std::stringstream mval(value);
    std::string tmp;
    mval >> tmp;
    if (tmp == "FIFO") {
      intra_dimension_scheduling = IntraDimensionScheduling::FIFO;
    } else if (tmp == "RG") {
      intra_dimension_scheduling = IntraDimensionScheduling::RG;
    } else if (tmp == "smallestFirst") {
      intra_dimension_scheduling = IntraDimensionScheduling::SmallestFirst;
    } else if (tmp == "lessRemainingPhaseFirst") {
      intra_dimension_scheduling =
          IntraDimensionScheduling::LessRemainingPhaseFirst;
    } else {
      sys_panic(
          "unknown value for intra-dimension-scheduling  in sys input file");
    }
  } else if (var == "inter-dimension-scheduling:") {
    std::stringstream mval(value);
    std::string tmp;
    mval >> tmp;
    if (tmp == "ascending") {
      inter_dimension_scheduling = InterDimensionScheduling::Ascending;
    } else if (tmp == "offlineGreedy") {
      inter_dimension_scheduling = InterDimensionScheduling::OfflineGreedy;
    } else if (tmp == "offlineGreedyFlex") {
      inter_dimension_scheduling = InterDimensionScheduling::OfflineGreedyFlex;
    } else if (tmp == "roundRobin") {
      inter_dimension_scheduling = InterDimensionScheduling::RoundRobin;
    } else {
      sys_panic(
          "unknown value for inter-dimension-scheduling  in sys input file");
    }
  } else if (var == "seprate-log:") {
    std::stringstream mval(value);
    int int_to_bool;
    mval >> int_to_bool;
    if (int_to_bool == 0) {
      this->seprate_log = false;
    } else {
      this->seprate_log = true;
    }
  } else if (var != "") {
    std::cerr
        << "######### Exiting because " << var
        << " is an unknown variable. Check your system input file. #########"
        << std::endl;
    exit(1);
  }
  return true;
}
bool Sys::post_process_inputs() {
  all_reduce_implementation_per_dimension =
      generate_collective_implementation_from_input(
          inp_all_reduce_implementation);
  if (all_reduce_implementation_per_dimension.size() == 0) {
    sys_panic("unknown value for all-reduce-implementation in sys input file");
  }
  reduce_scatter_implementation_per_dimension =
      generate_collective_implementation_from_input(
          inp_reduce_scatter_implementation);
  if (reduce_scatter_implementation_per_dimension.size() == 0) {
    sys_panic(
        "unknown value for reduce-scatter-implementation in sys input file");
  }
  all_gather_implementation_per_dimension =
      generate_collective_implementation_from_input(
          inp_all_gather_implementation);
  if (all_gather_implementation_per_dimension.size() == 0) {
    sys_panic("unknown value for all-gather-implementation in sys input file");
  }
  all_to_all_implementation_per_dimension =
      generate_collective_implementation_from_input(
          inp_all_to_all_implementation);
  if (all_to_all_implementation_per_dimension.size() == 0) {
    sys_panic("unknown value for all-to-all-implementation in sys input file");
  }
  if (inp_collective_optimization == "baseline") {
    collectiveOptimization = CollectiveOptimization::Baseline;
  } else if (inp_collective_optimization == "localBWAware") {
    collectiveOptimization = CollectiveOptimization::LocalBWAware;
  } else {
    sys_panic("unknown value for collective optimization in sys input file");
  }

  if (inp_boost_mode == 1) {
    boost_mode = true;
  } else {
    boost_mode = false;
  }
  if (inp_scheduling_policy == "LIFO") {
    this->scheduling_policy = SchedulingPolicy::LIFO;
  } else if (inp_scheduling_policy == "FIFO") {
    this->scheduling_policy = SchedulingPolicy::FIFO;
  } else {
    sys_panic("unknown value for scheduling policy in sys input file");
  }
  if (inp_model_shared_bus == 1) {
    model_shared_bus = true;
  } else {
    model_shared_bus = false;
  }
  return true;
}
bool Sys::initialize_sys(std::string name) {
  std::ifstream inFile;
  inFile.open(name);
  if (!inFile) {
    if (id == 0) {
      std::cerr << "Unable to open file: " << name << std::endl;
      std::cerr << "############ Exiting because unable to open the system "
                   "input file ############"
                << std::endl;
      std::cerr << "This error is fatal. Please check your path and filename."
                << std::endl;
    }
    exit(1);
  } else {
    if (id == 0) {
      std::cout << "Success in opening system file" << std::endl;
    }
  }
  std::string var;
  std::string value;
  while (inFile.peek() != EOF) {
    var = "";
    inFile >> var;
    if (inFile.peek() != EOF) {
      inFile >> value;
    }
    bool result = parse_var(var, value);
    if (result == false) {
      inFile.close();
      return result;
    }
  }
  inFile.close();
  return post_process_inputs();
}
Sys::SchedulerUnit::SchedulerUnit(
    Sys* sys,
    std::vector<int> queues,
    int max_running_streams,
    int ready_list_threshold,
    int queue_threshold) {
  this->sys = sys;
  this->ready_list_threshold = ready_list_threshold;
  this->queue_threshold = queue_threshold;
  this->max_running_streams = max_running_streams;

  this->latency_per_dimension.resize(queues.size(), 0);
  this->total_chunks_per_dimension.resize(queues.size(), 0);
  this->total_active_chunks_per_dimension.resize(queues.size(), 0);

  int base = 0;
  int dimension = 0;
  for (auto q : queues) {
    for (int i = 0; i < q; i++) {
      this->running_streams[base] = 0;
      std::list<BaseStream*>::iterator it;
      this->stream_pointer[base] = it;
      this->queue_id_to_dimension[base] = dimension;
      base++;
    }
    dimension++;
    UsageTracker u(2);
    usage.push_back(u);
  }
}
void Sys::SchedulerUnit::notify_stream_added_into_ready_list() {
  if (this->sys->first_phase_streams < ready_list_threshold &&
      this->sys->total_running_streams < max_running_streams) {
    int max = ready_list_threshold - sys->first_phase_streams;
    if (max > max_running_streams - this->sys->total_running_streams) {
      max = max_running_streams - this->sys->total_running_streams;
    }
    sys->schedule(max);
  }
  return;
}
void Sys::SchedulerUnit::notify_stream_added(int vnet) {
  if (sys->id == 0 &&
      ++total_active_chunks_per_dimension[queue_id_to_dimension[vnet]] == 1) {
    usage[queue_id_to_dimension[vnet]].increase_usage();
  }
  stream_pointer[vnet] = sys->active_Streams[vnet].begin();
  std::advance(stream_pointer[vnet], running_streams[vnet]);
  while (stream_pointer[vnet] != sys->active_Streams[vnet].end() &&
         running_streams[vnet] < queue_threshold) {
    (*stream_pointer[vnet])->init();
    running_streams[vnet]++;
    std::advance(stream_pointer[vnet], 1);
  }
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"Sys::SchedulerUnit::notify_stream_added finished");
}
void Sys::SchedulerUnit::notify_stream_removed(int vnet, Tick running_time) {
  if (sys->id == 0 &&
      --total_active_chunks_per_dimension[queue_id_to_dimension[vnet]] == 0) {
    usage[queue_id_to_dimension[vnet]].decrease_usage();
  }
  running_streams[vnet]--;

  int dimension = this->queue_id_to_dimension[vnet];
  latency_per_dimension[dimension] += running_time;
  total_chunks_per_dimension[dimension]++;

  if (this->sys->first_phase_streams < ready_list_threshold &&
      this->sys->total_running_streams < max_running_streams) {
    int max = ready_list_threshold - sys->first_phase_streams;
    if (max > max_running_streams - this->sys->total_running_streams) {
      max = max_running_streams - this->sys->total_running_streams;
    }
    sys->schedule(max);
  }
  stream_pointer[vnet] = sys->active_Streams[vnet].begin();
  std::advance(stream_pointer[vnet], running_streams[vnet]);
  while (stream_pointer[vnet] != sys->active_Streams[vnet].end() &&
         running_streams[vnet] < queue_threshold) {
    (*stream_pointer[vnet])->init();
    running_streams[vnet]++;
    std::advance(stream_pointer[vnet], 1);
  }
}
std::vector<double> Sys::SchedulerUnit::get_average_latency_per_dimension() {
  std::vector<double> result;
  result.resize(latency_per_dimension.size(), -1);
  for (int i = 0; i < result.size(); i++) {
    result[i] = latency_per_dimension[i] / total_chunks_per_dimension[i];
  }
  return result;
}
int Sys::nextPowerOf2(int n) {
  int count = 0;
  if (n && !(n & (n - 1)))
    return n;
  while (n != 0) {
    n >>= 1;
    count += 1;
  }
  return 1 << count;
}
void Sys::sys_panic(std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}
void Sys::iterate() {
  call_events();
}
std::vector<std::string> Sys::split_string(std::string str, std::string sep) {
  char* cstr = const_cast<char*>(str.c_str());
  char* current;
  std::vector<std::string> arr;
  current = strtok(cstr, sep.c_str());
  while (current != nullptr) {
    arr.push_back(current);
    current = strtok(nullptr, sep.c_str());
  }
  return arr;
}
uint64_t Sys::determine_chunk_size(uint64_t size, ComType type) {
  uint64_t chunk_size = size / preferred_dataset_splits;
  return chunk_size;
}
DataSet* Sys::generate_all_reduce(
    uint64_t size,
    std::vector<bool> involved_dimensions,
    SchedulingPolicy pref_scheduling,
    int layer,
    EventType event,
    Callable* layer_ptr) {
  return generate_collective(
      size,
      layer,
      logical_topologies["AllReduce"],
      all_reduce_implementation_per_dimension,
      involved_dimensions,
      ComType::All_Reduce,
      pref_scheduling,
      event,
      layer_ptr);
}
DataSet* Sys::generate_all_gather(
    uint64_t size,
    std::vector<bool> involved_dimensions,
    SchedulingPolicy pref_scheduling,
    int layer,
    EventType event,
    Callable* layer_ptr) {
  return generate_collective(
      size,
      layer,
      logical_topologies["AllGather"],
      all_gather_implementation_per_dimension,
      involved_dimensions,
      ComType::All_Gather,
      pref_scheduling,
      event,
      layer_ptr);
}
DataSet* Sys::generate_reduce_scatter(
    uint64_t size,
    std::vector<bool> involved_dimensions,
    SchedulingPolicy pref_scheduling,
    int layer,
    EventType event,
    Callable* layer_ptr) {
  return generate_collective(
      size,
      layer,
      logical_topologies["ReduceScatter"],
      reduce_scatter_implementation_per_dimension,
      involved_dimensions,
      ComType::Reduce_Scatter,
      pref_scheduling,
      event,
      layer_ptr);
}
DataSet* Sys::generate_all_to_all(
    uint64_t size,
    std::vector<bool> involved_dimensions,
    SchedulingPolicy pref_scheduling,
    int layer,
    EventType event,
    Callable* layer_ptr) {
  return generate_collective(
      size,
      layer,
      logical_topologies["AllToAll"],
      all_to_all_implementation_per_dimension,
      involved_dimensions,
      ComType::All_to_All,
      pref_scheduling,
      event,
      layer_ptr);
}
CollectivePhase Sys::generate_collective_phase(
    ComType collective_type,
    int layer_num,
    BasicLogicalTopology* topology,
    uint64_t data_size,
    int queue_id,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    CollectiveImplementation* collective_implementation,
    bool boost_mode) {
        MockNcclLog* NcclLog = MockNcclLog::getInstance();

        if (collective_implementation->type == CollectiveImplementationType::Ring ||
              collective_implementation->type ==
                  CollectiveImplementationType::OneRing) {
            CollectivePhase vn(
                this,
                queue_id,
                new Ring(
                    collective_type,
                    id,
                    layer_num,
                    (RingTopology*)topology,
                    data_size,
                    direction,
                    injection_policy,
                    boost_mode));
                  return vn;
          } else if (
              collective_implementation->type == CollectiveImplementationType::Direct ||
              collective_implementation->type ==
                  CollectiveImplementationType::OneDirect) {
            CollectivePhase vn(
                this,
                queue_id,
                new AllToAll(
                    collective_type,
                    ((DirectCollectiveImplementation*)collective_implementation)
                        ->direct_collective_window,
                    id,
                    layer_num,
                    (RingTopology*)topology,
                    data_size,
                    direction,
                    InjectionPolicy::Normal,
                    boost_mode));
                return vn;
          } else if (
              collective_implementation->type ==
              CollectiveImplementationType::DoubleBinaryTree) {
            CollectivePhase vn(
                this,
                queue_id,
                new DoubleBinaryTreeAllReduce(
                    id, layer_num, (BinaryTree*)topology, data_size, boost_mode));
                return vn;
          } else if (
              collective_implementation->type ==
                  CollectiveImplementationType::HalvingDoubling ||
              collective_implementation->type ==
                  CollectiveImplementationType::OneHalvingDoubling) {
            CollectivePhase vn(
                this,
                queue_id,
                new HalvingDoubling(
                    collective_type,
                    id,
                    layer_num,
                    (RingTopology*)topology,
                    data_size,
                    boost_mode));
                    return vn;
          } else if(collective_implementation->type == CollectiveImplementationType::NcclRingFlowModel) {
              ParallelStrategy  comm_ps;
              if (workload->current_state == Workload::LoopState::Forward_Pass){
                comm_ps = ParallelStrategy::TP;
              }
              else if(workload->current_state == Workload::LoopState::Input_Gradient){
                comm_ps = ParallelStrategy::TP;
              }
              else if(workload->current_state == Workload::LoopState::Weight_Gradient){
                comm_ps = ParallelStrategy::DP;
              }
              MockNccl::ncclInfo *nccl_info;
              std::shared_ptr<void> ptr_FlowModels;
              {
                Sys::sysCriticalSection cs;
                nccl_info = get_nccl_Info(comm_ps,data_size,collective_type);
                ptr_FlowModels = generate_flow_model(comm_ps, data_size, collective_type); 
                cs.ExitSection();
              }
              
              if(nccl_info->algorithm == NCCL_ALGO_RING) {
                std::shared_ptr<MockNccl::FlowModels> RingFlowModels = std::static_pointer_cast<MockNccl::FlowModels>(ptr_FlowModels);
                std::map<int,std::map<int,std::vector<int>>> channels;
                {
                  Sys::sysCriticalSection cs;
                  channels = mock_nccl_comms[comm_ps]->get_rings();
                  cs.ExitSection();
                }
                NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d generate FlowModels",id);
                if(RingFlowModels != nullptr){
                  NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d NcclMock generate  %d channel and flow model count:  %d",id,channels.size(),RingFlowModels->size());
                  for (auto flow : *RingFlowModels) {
                    int prev;
                    int parent_flow_id;
                    int child_flow_id;
                    if (flow.second.prev.size() == 0) {
                      prev = -1;
                    } else {
                      prev = flow.second.prev[0];
                    }
                    if (flow.second.child_flow_id.size() == 0) {
                      child_flow_id = -1;
                    } else {
                      child_flow_id = flow.second.child_flow_id[0];
                    }
                    if (flow.second.parent_flow_id.size() == 0) {
                      parent_flow_id = -1;
                    } else {
                      parent_flow_id = flow.second.parent_flow_id[0];
                    }
                    NcclLog->writeLog(NcclLogLevel::DEBUG," %d,  %d,  %d to  %d current_flow_id %d prev rank:  %d parent_flow_id:  %d child_flow_id:  %d chunk_id:  %d flow_size: %lu chunk_count:  %d ",flow.first.first,flow.first.second,flow.second.src,flow.second.dest,flow.second.flow_id,prev,parent_flow_id,child_flow_id,flow.second.chunk_id,flow.second.flow_size,flow.second.chunk_count);
                  }
                }
                CollectivePhase vn(
                    this,
                    queue_id,
                    new NcclTreeFlowModel(
                        collective_type,
                        id,
                        layer_num,
                        (RingTopology*)topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        RingFlowModels,
                        channels.size()));
                return vn;
              } else if(nccl_info->algorithm == NCCL_ALGO_TREE) {
                std::shared_ptr<MockNccl::FlowModels> TreeFlowModels;
                MockNccl::TreeChannels treechannels;
                {
                  Sys::sysCriticalSection cs;
                  TreeFlowModels = std::static_pointer_cast<MockNccl::FlowModels>(ptr_FlowModels);
                  treechannels = mock_nccl_comms[comm_ps]->get_treechannels();
                  cs.ExitSection();
                }
                CollectivePhase vn(
                    this,
                    queue_id,
                    new NcclTreeFlowModel(
                        collective_type,
                        id,
                        layer_num,
                        (RingTopology*)topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        TreeFlowModels,
                        treechannels.size()));
                return vn;

              } else if(nccl_info->algorithm == NCCL_ALGO_NVLS) {
                collective_type = ComType::All_Reduce_NVLS;
                std::shared_ptr<MockNccl::FlowModels> RingFlowModels = std::static_pointer_cast<MockNccl::FlowModels>(ptr_FlowModels);
                MockNccl::TreeChannels treechannels;
                {
                  Sys::sysCriticalSection cs;
                  treechannels = mock_nccl_comms[comm_ps]->get_treechannels();
                  cs.ExitSection();
                }
                NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d generate FlowModels",id);
                if(RingFlowModels != nullptr){
                  NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d NcclMock generate  %d channel and flow model count:  %d",id,treechannels.size(),RingFlowModels->size());
                  for (auto flow : *RingFlowModels) {
                    int prev;
                    int parent_flow_id;
                    int child_flow_id;
                    if (flow.second.prev.size() == 0) {
                      prev = -1;
                    } else {
                      prev = flow.second.prev[0];
                    }
                    if (flow.second.child_flow_id.size() == 0) {
                      child_flow_id = -1;
                    } else {
                      child_flow_id = flow.second.child_flow_id[0];
                    }
                    if (flow.second.parent_flow_id.size() == 0) {
                      parent_flow_id = -1;
                    } else {
                      parent_flow_id = flow.second.parent_flow_id[0];
                    }
                    NcclLog->writeLog(NcclLogLevel::DEBUG," %d,  %d,  %d to  %d current_flow_id %d prev rank:  %d parent_flow_id:  %d child_flow_id:  %d chunk_id:  %d flow_size: %lu chunk_count:  %d ",flow.first.first,flow.first.second,flow.second.src,flow.second.dest,flow.second.flow_id,prev,parent_flow_id,child_flow_id,flow.second.chunk_id,flow.second.flow_size,flow.second.chunk_count);
                  }
                }
                CollectivePhase vn(
                    this,
                    queue_id,
                    new NcclTreeFlowModel(
                        collective_type,
                        id,
                        layer_num,
                        (RingTopology*)topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        RingFlowModels,
                        treechannels.size()));
                return vn;
              } 

          } else {
            std::cerr
                << "Error: No known collective implementation for collective phase"
                << std::endl;
            exit(1);
          }
}

std::map<std::pair<int,int>, MockNccl::SingleFlow> Sys:: generate_net_test_flow_model(uint64_t data_size, int nums) {
  std::map<std::pair<int,int>, MockNccl::SingleFlow> result;
  MockNccl::SingleFlow tmp;
  for (int i = 0; i < nums; i++) {
    tmp.flow_id = i;
    tmp.src = 0;
    tmp.dest = 1;
    tmp.flow_size = data_size;
    tmp.parent_flow_id = {};
    tmp.child_flow_id = {};
    tmp.channel_id = 0;
    result[make_pair(0, i)] = tmp;
  }
  return result;
}

std::map<std::pair<int,int>, MockNccl::SingleFlow> Sys::generate_nvl_test_flow_model(uint64_t data_size, int nums) {
  std::map<std::pair<int,int>, MockNccl::SingleFlow> result;
  MockNccl::SingleFlow tmp;
  for (int i = 0; i < nums; i++) {
    tmp.flow_id = i;
    tmp.src = 0;
    tmp.dest = 1;
    tmp.flow_size = data_size;
    tmp.parent_flow_id = {};
    tmp.child_flow_id =  {};
    tmp.channel_id = 0;
    result[make_pair(0, i)] = tmp;
  }
  return result;
}

std::map<int, std::vector<int>> Sys::gen_local_ring(std::vector<int> gpus) {
  std::map<int, std::vector<int>> localrings;
  int n = gpus.size();
  for (int i = 0; i < n; ++i) {
    std::vector<int> vec;
    for (int j = 0; j < n; ++j) {
      vec.push_back(gpus[(i + j) % n]);
    }
    localrings[i] = vec;
  }
  return localrings;
}

bool Sys::mock_nccl_grobal_group_init(){
  if (GlobalGroup != nullptr)
    return true;
  else {
    int total_nodes = this->total_nodes;
    int TP_size = workload->model_parallel_npu_group == 0
        ? total_nodes
        : workload->model_parallel_npu_group;
    GlobalGroup =
        new MockNccl::MockNcclGroup(all_gpus, TP_size, ngpus_per_node, NVSwitchs,gpu_type);
    
    std::map<int,std::vector<int>>TP_localring;
    std::map<int,std::vector<int>>DP_localring;

    if (TP_size <= ngpus_per_node) {
      std::vector<int> TP_gpus;
      for (int i = 0; i < TP_size; i++) {
        TP_gpus.push_back(i);
      }
      TP_localring = gen_local_ring(TP_gpus);
    } else {
      std::vector<int> TP_gpus;
      for (int i = 0; i < ngpus_per_node; i++) {
        TP_gpus.push_back(i);
      }
      TP_localring = gen_local_ring(TP_gpus);
    }

    if (TP_size <= ngpus_per_node) {
      std::vector<int> DP_gpus;
      for (int i = 0; i < ngpus_per_node / TP_size; i++) {
        DP_gpus.push_back(i * TP_size);
      }
      DP_localring = gen_local_ring(DP_gpus);
    } else {
      std::vector<int> DP_gpus;
      DP_gpus.push_back(0);
      DP_localring = gen_local_ring(DP_gpus);
    }
    GlobalGroup->setlocalrings(TP_localring, MockNccl::GroupType::TP);
    GlobalGroup->setlocalrings(DP_localring, MockNccl::GroupType::DP);
    return true;
  }
}

bool Sys::mock_nccl_comms_init(){
    int TP_size = workload->model_parallel_npu_group == 0
       ? total_nodes
       : workload->model_parallel_npu_group;
    int DP_size = total_nodes / TP_size;
    MockNccl::MockNcclComm* pComm;
    if (TP_size > 1) {
      pComm = new MockNccl::MockNcclComm(id,MockNccl::GroupType::TP,GlobalGroup);
      mock_nccl_comms[TP] = pComm;
    }
    if(DP_size > 1) {
      pComm = new MockNccl::MockNcclComm(id,MockNccl::GroupType::DP,GlobalGroup);
      mock_nccl_comms[DP] = pComm;
    }
    return true;
}

struct MockNccl::ncclInfo* Sys::get_nccl_Info(ParallelStrategy comm_ps, uint64_t data_size, ComType collective_type) {
    return mock_nccl_comms[comm_ps]->get_algo_proto_info(data_size, collective_type );
}

std::shared_ptr<void> Sys::generate_flow_model(ParallelStrategy comm_ps, uint64_t data_size, ComType collective_type) {
    std::string flow_model_name;
    if (comm_ps == ParallelStrategy::TP) {
        flow_model_name = "TP";
    } else if (comm_ps == ParallelStrategy::DP) {
        flow_model_name = "DP";
    }
    MockNccl::MockNcclComm* pComm = mock_nccl_comms[comm_ps];
    MockNccl::State current_state;
    switch (this->workload->current_state) {
      case Workload::LoopState::Forward_Pass:
        current_state = MockNccl::State::Forward_Pass;
        break;
      case Workload::LoopState::Input_Gradient:
        current_state = MockNccl::State::Input_Gradient;
        break;
      case Workload::LoopState::Weight_Gradient:
        current_state = MockNccl::State::Weight_Gradient;
        break;
    }
    return  pComm->get_flow_model(data_size,collective_type,this->workload->index,current_state);
}

DataSet* Sys::generate_collective(
    uint64_t size,
    int layer_num,
    LogicalTopology* topology,
    std::vector<CollectiveImplementation*> implementation_per_dimension,
    std::vector<bool> dimensions_involved,
    ComType collective_type,
    SchedulingPolicy pref_scheduling,
    EventType event,
    Callable* layer_ptr ) {
  uint64_t chunk_size = determine_chunk_size(size, collective_type);
  if(id == 0) std::cout << "chunk size is: " << chunk_size << " , size is: " << size << " , layer_num is: " << layer_num << " , node: " << id << std::endl;
  uint64_t recommended_chunk_size = chunk_size;
  int streams = ceil(((double)size) / chunk_size);
  int64_t tmp;
  DataSet* dataset = new DataSet(streams);
  #ifdef PHY_MTP
  if (event != EventType::NONE && layer_ptr != nullptr) {
    dataset->set_notifier(layer_ptr,event);
  }
  #endif
  int pri = get_priority(pref_scheduling);
  int count = 0;
  if (id == 0 &&
      (inter_dimension_scheduling == InterDimensionScheduling::OfflineGreedy ||
       inter_dimension_scheduling ==
           InterDimensionScheduling::OfflineGreedyFlex)) {
    if (last_scheduled_collective != Sys::boostedTick()) {
      offline_greedy->reset_loads();
      last_scheduled_collective = Sys::boostedTick();
    }
  }

  while (size > 0) {
    count++;
    chunk_size=std::min(chunk_size,size); 
    std::vector<int> dim_mapper(topology->get_num_of_dimensions());
    std::iota(std::begin(dim_mapper), std::end(dim_mapper), 0);
    if (collective_type == ComType::All_Gather) {
      std::reverse(dim_mapper.begin(), dim_mapper.end());
    }
    if (inter_dimension_scheduling == InterDimensionScheduling::RoundRobin) {
      std::rotate(
          dim_mapper.begin(),
          dim_mapper.begin() + round_robin_inter_dimension_scheduler,
          dim_mapper.end());
      round_robin_inter_dimension_scheduler++;
      if (round_robin_inter_dimension_scheduler ==
          topology->get_num_of_dimensions()) {
        round_robin_inter_dimension_scheduler = 0;
      }
    } else if (
        collective_type != ComType::All_to_All &&
        (inter_dimension_scheduling ==
             InterDimensionScheduling::OfflineGreedy ||
         inter_dimension_scheduling ==
             InterDimensionScheduling::OfflineGreedyFlex)) {
      uint64_t prev_size = size;
      dim_mapper = offline_greedy->get_chunk_scheduling(
          stream_counter,
          size,
          recommended_chunk_size,
          dimensions_involved,
          inter_dimension_scheduling,
          collective_type);
      chunk_size = prev_size - size;
    }

    if (collective_type == ComType::All_to_All ||
        (inter_dimension_scheduling !=
             InterDimensionScheduling::OfflineGreedy &&
         inter_dimension_scheduling !=
             InterDimensionScheduling::OfflineGreedyFlex)) {
      size -= chunk_size;
    }
    tmp = chunk_size;
    std::list<CollectivePhase> vect;
    CollectivePhase phase;

    if (collective_type != ComType::All_Reduce ||
        collectiveOptimization == CollectiveOptimization::Baseline) {
      for (int dim = 0; dim < topology->get_num_of_dimensions(); dim++) {
        if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1 ||
            !dimensions_involved[dim_mapper[dim]]) {
          continue;
        }
        std::pair<int, RingTopology::Direction> queue =
            vLevels->get_next_queue_at_level(dim_mapper[dim]);
        phase = generate_collective_phase(
            collective_type,
            layer_num,
            topology->get_basic_topology_at_dimension(
                dim_mapper[dim], collective_type),
            tmp,
            queue.first,
            queue.second,
            InjectionPolicy::Normal,
            implementation_per_dimension[dim_mapper[dim]],
            boost_mode);
        vect.push_back(phase);
        tmp = phase.final_data_size;
      }
    } else if (
        inter_dimension_scheduling == InterDimensionScheduling::OfflineGreedy ||
        inter_dimension_scheduling ==
            InterDimensionScheduling::OfflineGreedyFlex ||
        inter_dimension_scheduling == InterDimensionScheduling::OnlineGreedy) {
      int dim = 0;
      for (dim = 0; dim < topology->get_num_of_dimensions(); dim++) {
        if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1 ||
            !dimensions_involved[dim_mapper[dim]]) {
          continue;
        }
        std::pair<int, RingTopology::Direction> queue =
            vLevels->get_next_queue_at_level(dim_mapper[dim]);
        phase = generate_collective_phase(
            ComType::Reduce_Scatter,
            layer_num,
            topology->get_basic_topology_at_dimension(
                dim_mapper[dim], ComType::Reduce_Scatter),
            tmp,
            queue.first,
            queue.second,
            InjectionPolicy::Normal,
            implementation_per_dimension[dim_mapper[dim]],
            boost_mode);
        vect.push_back(phase);
        tmp = phase.final_data_size;
      }
      dim--;
      for (; dim >= 0; dim--) {
        if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1 ||
            !dimensions_involved[dim_mapper[dim]]) {
          continue;
        }
        std::pair<int, RingTopology::Direction> queue =
            vLevels->get_next_queue_at_level(dim_mapper[dim]);
        phase = generate_collective_phase(
            ComType::All_Gather,
            layer_num,
            topology->get_basic_topology_at_dimension(
                dim_mapper[dim], ComType::All_Gather),
            tmp,
            queue.first,
            queue.second,
            InjectionPolicy::Normal,
            implementation_per_dimension[dim_mapper[dim]],
            boost_mode);
        vect.push_back(phase);
        tmp = phase.final_data_size;
      }
    } else {
      int dim = 0;
      int last_active_dim = 0;
      for (dim = 0; dim < topology->get_num_of_dimensions(); dim++) {
        if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) != 1 &&
            dimensions_involved[dim_mapper[dim]]) {
          last_active_dim = dim;
        }
      }
      for (dim = 0; dim < last_active_dim; dim++) {
        if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1 ||
            !dimensions_involved[dim_mapper[dim]]) {
          continue;
        }
        std::pair<int, RingTopology::Direction> queue =
            vLevels->get_next_queue_at_level(dim_mapper[dim]);
        phase = generate_collective_phase(
            ComType::Reduce_Scatter,
            layer_num,
            topology->get_basic_topology_at_dimension(
                dim_mapper[dim], ComType::Reduce_Scatter),
            tmp,
            queue.first,
            queue.second,
            InjectionPolicy::Normal,
            implementation_per_dimension[dim_mapper[dim]],
            boost_mode);
        vect.push_back(phase);
        tmp = phase.final_data_size;
      }
      while (dim > 0 &&
             (dimensions_involved[dim_mapper[dim]] == false ||
              topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1)) {
        dim--;
      }
      if (dimensions_involved[dim_mapper[dim]] &&
          topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) > 1) {
        std::pair<int, RingTopology::Direction> queue =
            vLevels->get_next_queue_at_level(dim_mapper[dim]);
        phase = generate_collective_phase(
            ComType::All_Reduce,
            layer_num,
            topology->get_basic_topology_at_dimension(
                dim_mapper[dim], ComType::All_Reduce),
            tmp,
            queue.first,
            queue.second,
            InjectionPolicy::Normal,
            implementation_per_dimension[dim_mapper[dim]],
            boost_mode);
        vect.push_back(phase);
        tmp = phase.final_data_size;
      }
      dim--;
      for (; dim >= 0; dim--) {
        if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1 ||
            !dimensions_involved[dim_mapper[dim]]) {
          continue;
        }
        std::pair<int, RingTopology::Direction> queue =
            vLevels->get_next_queue_at_level(dim_mapper[dim]);
        phase = generate_collective_phase(
            ComType::All_Gather,
            layer_num,
            topology->get_basic_topology_at_dimension(
                dim_mapper[dim], ComType::All_Gather),
            tmp,
            queue.first,
            queue.second,
            InjectionPolicy::Normal,
            implementation_per_dimension[dim_mapper[dim]],
            boost_mode);
        vect.push_back(phase);
        tmp = phase.final_data_size;
      }
    }
    if (vect.size() > 0) {
      StreamBaseline* newStream =
          new StreamBaseline(this, dataset, stream_counter++, vect, pri);
      newStream->current_queue_id = -1;
      #ifdef PHY_MTP
      insert_into_running_list(newStream);
      #endif
      insert_into_ready_list(newStream);
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
      NcclLog->writeLog(NcclLogLevel::DEBUG,"Sys::generate_collective finished");
    } else {
      dataset->active = false;
      break;
    }
  }
  if (dataset->active) {
    streams_injected += count;
    dataset->total_streams = count;
  }
  return dataset;
}
void Sys::call_events() {
  if(event_queue.find(Sys::boostedTick())==event_queue.end()){
    goto FINISH_CHECK;
  }
  for (auto& callable : event_queue[Sys::boostedTick()]) {
    try {
      pending_events--;
      (std::get<0>(callable))
          ->call(std::get<1>(callable), std::get<2>(callable));
    } catch (...) {
      std::cerr << "warning! a callable is removed before call" << std::endl;
    }
  }
  {
  Sys::sysCriticalSection cs;
  if (event_queue[Sys::boostedTick()].size() > 0) {
    event_queue[Sys::boostedTick()].clear();
  }
  event_queue.erase(Sys::boostedTick());
  cs.ExitSection();
  }
  FINISH_CHECK: if ((finished_workloads == 1 && event_queue.size() == 0 && pending_sends.size() == 0) ||
      initialized == false) {
    delete this;
  }

}
void Sys::exitSimLoop(std::string msg) {
  if(id == 0 ){
  std::cout << msg << std::endl;
  }
  NI->sim_finish();
  return;
}
Tick Sys::boostedTick() {
  Sys* ts = all_generators[0];
  if (ts == nullptr) {
    for (int i = 1; i < all_generators.size(); i++) {
      if (all_generators[i] != nullptr) {
        ts = all_generators[i];
        break;
      }
    }
  }
  timespec_t tmp = ts->NI->sim_get_time();
  Tick tick = tmp.time_val / CLOCK_PERIOD;
  return tick + offset;
}
void Sys::proceed_to_next_vnet_baseline(StreamBaseline* stream) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase1, stream->current_queue_id %d stream->phases_to_go.size %d",stream->current_queue_id,stream->phases_to_go.size());
  int previous_vnet = stream->current_queue_id;
  if (stream->steps_finished == 1) {
    first_phase_streams--;
  }
  if (stream->steps_finished != 0) {
    stream->net_message_latency.back() /= stream->net_message_counter;
  }
  if (stream->my_current_phase.algorithm != nullptr) {
    delete stream->my_current_phase.algorithm;
  }
  if (stream->phases_to_go.size() == 0) {
    stream->take_bus_stats_average();
    stream->dataset->notify_stream_finished((StreamStat*)stream);
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase2");
  if (stream->current_queue_id >= 0 && stream->my_current_phase.enabled) {
    std::list<BaseStream*>& target =
        active_Streams.at(stream->my_current_phase.queue_id);
    for (std::list<BaseStream*>::iterator it = target.begin();
         it != target.end();
         ++it) {
      if (((StreamBaseline*)(*it))->stream_num == stream->stream_num) {
        target.erase(it);
        break;
      }
    }
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase2-1");
  if (stream->phases_to_go.size() == 0) {
    total_running_streams--;
    if (previous_vnet >= 0) {
      NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase2-1");
      scheduler_unit->notify_stream_removed(
          previous_vnet, Sys::boostedTick() - stream->last_init);
    }
    #ifdef PHY_MTP
    running_list.pop_front();
    #endif
    NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: delete stream");
    delete stream;
    return;
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase3");
  stream->steps_finished++;
  stream->current_queue_id = stream->phases_to_go.front().queue_id;  
  stream->current_com_type = stream->phases_to_go.front().comm_type;

  CollectivePhase vi = stream->phases_to_go.front();
  stream->my_current_phase = vi;
  stream->phases_to_go.pop_front();
  stream->test = 0;
  stream->test2 = 0;
  stream->initialized = false;
  stream->last_phase_change = Sys::boostedTick();
  stream->total_packets_sent = 0;

  stream->net_message_latency.push_back(0);
  stream->net_message_counter = 0;
  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase1, stream->current_queue_id %d stream->phases_to_go.size %d",stream->current_queue_id,stream->phases_to_go.size());

  if (stream->my_current_phase.enabled) {
    insert_stream(&active_Streams[stream->current_queue_id], stream);
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: phase4");

  stream->state = StreamState::Ready;

  if (previous_vnet >= 0) {
    scheduler_unit->notify_stream_removed(
        previous_vnet, Sys::boostedTick() - stream->last_init);
  }
  #ifdef PHY_MTP
  ready_list.pop_front();
  first_phase_streams++;
  total_running_streams++;
  #endif

  scheduler_unit->notify_stream_added(stream->current_queue_id);

  NcclLog->writeLog(NcclLogLevel::DEBUG,"proceed_to_next_vnet_baseline :: exit");
}
void Sys::exiting() {}
void Sys::insert_stream(std::list<BaseStream*>* queue, BaseStream* baseStream) {
  std::list<BaseStream*>::iterator it = queue->begin();
  if (intra_dimension_scheduling == IntraDimensionScheduling::FIFO ||
      baseStream->current_queue_id < 0 ||
      baseStream->current_com_type == ComType::All_to_All ||
      baseStream->current_com_type == ComType::All_Reduce) {
    while (it != queue->end()) {
      if ((*it)->initialized == true) {
        std::advance(it, 1);
        continue;
      } else if ((*it)->priority >= baseStream->priority) {
        std::advance(it, 1);
        continue;
      } else {
        break;
      }
    }
  } else if (intra_dimension_scheduling == IntraDimensionScheduling::RG) {
    ComType one_to_last = ComType::None;
    ComType last = ComType::None;
    while (it != queue->end()) {
      one_to_last = last;
      last = (*it)->current_com_type;
      if ((*it)->initialized == true) {
        std::advance(it, 1);
        if (it != queue->end() && (*it)->initialized == false) {
          one_to_last = last;
          last = (*it)->current_com_type;
          std::advance(it, 1);
        }
        continue;
      } else if ((*it)->priority > baseStream->priority) {
        std::advance(it, 1);
        continue;
      } else if (
          (last == ComType::Reduce_Scatter &&
           one_to_last == ComType::All_Gather) ||
          (last == ComType::All_Gather &&
           one_to_last == ComType::Reduce_Scatter)) {
        std::advance(it, 1);
        continue;
      } else {
        break;
      }
    }
  } else if (
      intra_dimension_scheduling == IntraDimensionScheduling::SmallestFirst) {
    while (it != queue->end()) {
      if ((*it)->initialized == true) {
        std::advance(it, 1);
        continue;
      } else if (
          (*it)->my_current_phase.initial_data_size <
          baseStream->my_current_phase.initial_data_size) {
        std::advance(it, 1);
        continue;
      } else {
        break;
      }
    }
  } else if (
      intra_dimension_scheduling ==
      IntraDimensionScheduling::LessRemainingPhaseFirst) {
    while (it != queue->end()) {
      if ((*it)->initialized == true) {
        std::advance(it, 1);
        continue;
      } else if ((*it)->phases_to_go.size() < baseStream->phases_to_go.size()) {
        std::advance(it, 1);
        continue;
      } else {
        break;
      }
    }
  }
  queue->insert(it, baseStream);
}
void Sys::register_for_finished_stream(Callable* callable) {
  registered_for_finished_stream_event.push_back(callable);
}
void Sys::increase_finished_streams(int amount) {
  streams_finished += amount;
  for (auto c : registered_for_finished_stream_event) {
    c->call(EventType::StreamsFinishedIncrease, nullptr);
  }
}

void Sys::register_phases(
    BaseStream* stream,
    std::list<CollectivePhase> phases_to_go) {
  for (auto& vnet : phases_to_go) {
    stream_priorities[vnet.queue_id].push_back(stream->stream_num);
  }
}

void Sys::zero_latecy_register_event(
      Callable* callable,
      EventType event,
      CallData* callData,
      int cycles){
  Tick mycycles = 0;
  bool should_schedule = false;
  {
    #ifdef NS3_MTP
    Sys::sysCriticalSection cs;
    #endif
    #ifdef PHY_MTP
    Sys::sysCriticalSection cs;
    #endif
    if (event_queue.find(Sys::boostedTick() + mycycles) == event_queue.end()) {
      std::list<std::tuple<Callable*, EventType, CallData*>> tmp;
      event_queue[Sys::boostedTick() + mycycles] = tmp;
      should_schedule = true;
    }
    event_queue[Sys::boostedTick() + mycycles].push_back(
        std::make_tuple(callable, event, callData));
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
    #ifdef PHY_MTP
    cs.ExitSection();
    #endif
  }
  pending_events++;
  if (should_schedule) {
    timespec_t tmp = generate_time(mycycles);
    BasicEventHandlerData* data =
        new BasicEventHandlerData(this, EventType::CallEvents);
    this->handleEvent(data);
  }
}

void Sys::register_event(
    Callable* callable,
    EventType event,
    CallData* callData,
    int cycles) {
  Tick mycycles = cycles;
  try_register_event(callable, event, callData, mycycles);
  return;
}
void Sys::call(EventType type, CallData* data) {
  if (id == 0 && type == EventType::General) {
    increase_finished_streams(1);
  }
}
void Sys::try_register_event(
    Callable* callable,
    EventType event,
    CallData* callData,
    Tick& cycles) {
  bool should_schedule = false;
  {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(
        NcclLogLevel::DEBUG, "try_register_event EventType %d ", event);
    #ifdef NS3_MTP
        Sys::sysCriticalSection cs;
    #endif
    if (event_queue.find(Sys::boostedTick() + cycles) == event_queue.end()) {
      std::list<std::tuple<Callable*, EventType, CallData*>> tmp;
      event_queue[Sys::boostedTick() + cycles] = tmp;
      should_schedule = true;
    }
    event_queue[Sys::boostedTick() + cycles].push_back(
        std::make_tuple(callable, event, callData));
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
  if (should_schedule) {
    timespec_t tmp = generate_time(cycles);
    BasicEventHandlerData* data =
        new BasicEventHandlerData(this, EventType::CallEvents);
    NI->sim_schedule(tmp, &Sys::handleEvent, data);
  }
  cycles = 0;
  pending_events++;
  return;
}
#ifdef PHY_MTP
void Sys::insert_into_running_list(StreamBaseline* stream) {
  running_list.push_back(stream);
}
#endif

void Sys::insert_into_ready_list(BaseStream* stream) {
  insert_stream(&ready_list, stream);
  scheduler_unit->notify_stream_added_into_ready_list();
}
void Sys::schedule(int num) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  int ready_list_size = ready_list.size();
  int counter = std::min(num, ready_list_size);
  NcclLog->writeLog(NcclLogLevel::DEBUG,"Sys.cc::schedule num %d ready_list_size %d",num,ready_list_size);
  while (counter > 0) {
    int top_vn = ready_list.front()->phases_to_go.front().queue_id;
    int total_waiting_streams = ready_list.size();
    int total_phases = ready_list.front()->phases_to_go.size();

    proceed_to_next_vnet_baseline((StreamBaseline*)ready_list.front());

    #ifndef PHY_MTP
    if (ready_list.front()->current_queue_id == -1) {
      Sys::sys_panic(
          "should not happen! " 
          );
    }
    ready_list.pop_front();
    first_phase_streams++;
    total_running_streams++;
    #endif 
    counter--;
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"Sys::shedule finished");
}
void Sys::handleEvent(void* arg) {
  if (arg == nullptr) { 
    return;
  }
  BasicEventHandlerData* ehd = (BasicEventHandlerData*)arg;
  Sys* node = ehd->node;
  EventType event = ehd->event;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();


  if (event == EventType::CallEvents) {
    NcclLog->writeLog(NcclLogLevel::DEBUG," Sys::handleEvent EventType::CallEvents");
    node->iterate();
    delete ehd;
  } else if (event == EventType::RendezvousSend) {
    RendezvousSendData* rsd = (RendezvousSendData*)ehd;
    rsd->send->call(EventType::General, nullptr);
    delete rsd;
  } else if (event == EventType::RendezvousRecv) {
    RendezvousRecvData* rrd = (RendezvousRecvData*)ehd;
    rrd->recv->call(EventType::General, nullptr);
    delete rrd;
  } else if (event == EventType::PacketReceived) {
    RecvPacketEventHadndlerData* rcehd = (RecvPacketEventHadndlerData*)ehd;
    StreamBaseline* owner = static_cast<StreamBaseline*>(rcehd->owner);
    owner->consume(rcehd);
    delete rcehd;
  } else if (event == EventType::PacketSent) {
    SendPacketEventHandlerData* sendhd = (SendPacketEventHandlerData*)ehd;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"packet sent, sender id:  %d, node id:  %d",sendhd->senderNodeId,node->id);
    #ifdef NS3_MTP
    Sys::sysCriticalSection cs;
    #endif
    #ifdef PHY_MTP
    Sys::sysCriticalSection cs;
    #endif
    if(all_generators[sendhd->senderNodeId]== nullptr){
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      #ifdef PHY_MTP
      cs.ExitSection();
      #endif
      goto SEND_HANDLER_END;
    }
    
    if (node->pending_sends.find(
            std::make_pair(sendhd->receiverNodeId, sendhd->tag)) ==
            node->pending_sends.end() ||
            node->pending_sends[std::make_pair(sendhd->receiverNodeId, sendhd->tag)]
                .size() == 0) {

      node->is_there_pending_sends[std::make_pair(
          sendhd->receiverNodeId, sendhd->tag)] = false;
      
      if(node->event_queue.find(Sys::boostedTick())==node->event_queue.end())
        if ((node->finished_workloads == 1 && node->event_queue.size() == 0 && node->pending_sends.size() == 0) ||
      node->initialized == false) {
        delete node;
      }
      #ifdef NS3_MTP  
      cs.ExitSection();
      #endif
      #ifdef PHY_MTP
      cs.ExitSection();
      #endif
    } else {

      SimSendCaller* simSendCaller =
          node->pending_sends[std::make_pair(sendhd->receiverNodeId, sendhd->tag)]
                           .front();
      node->pending_sends[std::make_pair(sendhd->receiverNodeId, sendhd->tag)]
          .pop_front();
      if(node->pending_sends[std::make_pair(sendhd->receiverNodeId, sendhd->tag)].size() == 0)
        node->pending_sends.erase(std::make_pair(sendhd->receiverNodeId, sendhd->tag));

      #ifdef NS3_MTP  
      cs.ExitSection();
      #endif
      #ifdef PHY_MTP  
      cs.ExitSection();
      #endif
      simSendCaller->call(EventType::General, nullptr);
    }
    SEND_HANDLER_END: delete sendhd;
  }else if(event==EventType::PacketSentFinshed){
    AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) arg;
    if(ehd->owner!=nullptr)
      ehd->owner->sendcallback(ehd);
  }
}

AstraSim::timespec_t Sys::generate_time(int cycles) {
  timespec_t tmp = NI->sim_get_time();
  double addition = cycles * ((double)CLOCK_PERIOD);
  tmp.time_val = addition;
  return tmp;
}
} // namespace AstraSim
