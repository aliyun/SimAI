/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include "astra-sim/system/Callable.hh"

namespace AstraSim {
class Workload;
class Sys;
class Callable;
class Layer;
class CSVWriter;
} // namespace AstraSim

#include "astra-sim/system/AstraSimDataAPI.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {
enum class ParallelismPolicy {
  MicroBenchmark,
  Data,
  Transformer,
  TransformerFwdInBckwd,
  DLRM,
  DLRMEnhanced,
  Model,
  HybridDataModel,
  HybridModelData,
  HybridCustomized,
  DistributedInference,
  All,
  None
};

class Workload : Callable {
 public:
  enum class LoopState {
    Forward_Pass,
    Weight_Gradient,
    Input_Gradient,
    Wait_For_Sim_Finish,
    Forward_In_BackPass
  };
  ~Workload();
  Layer** layers;
  int SIZE;
  Sys* generator;
  std::string run_type;
  Tick counter;
  int index;
  LoopState current_state;
  bool delay_loaded;
  bool seprate_log;
  bool checkpoint_initiated;
  bool collective_issued;
  bool initialized;
  int TOTAL_PASS;
  int DLRM_LAST_BOTTOM_LAYER;
  int pass_counter;
  int pending_collectives;
  int model_parallel_npu_group;   // TP Size
  int expert_parallel_npu_group;  //Ep Size
  int pipeline_model_parallelism; //PP Size
  int GA;                         //Ga_Size
  int all_gpus;
  int vpp;
  ParallelismPolicy parallelismPolicy;
  Tick waiting_for_comm;
  Workload(
      std::string run_name,
      Sys* generator,
      std::string name,
      int TOTAL_PASS,
      int total_rows,
      int stat_row,
      std::string path,
      bool seprate_log);
  ParallelismPolicy decode_parallelsim(std::string parallelism);
  void call(EventType event, CallData* data);
  void iterate_micro_benchmark();
  void iterate_data_parallel();
  void iterate_hybrid_parallel_Transformer();
  void iterate_hybrid_parallel_Transformer_fwd_in_bckwd();
  void iterate_hybrid_parallel_DLRM();
  void iterate_hybrid_parallel_data_model();
  void iterate_hybrid_parallel_model_data();
  void iterate_hybrid_parallel_customized();
  void iterate_model_parallel();
  void iterate_distributed_inference();
  bool initialize_workload(std::string name);
  void initialize_stat_files();
  std::map<std::string, std::vector<bool>> decode_involved_dimensions(
      ParallelismPolicy policy,
      int model_parallel_npu_group);
  void fire();
  void report();
  void check_for_sim_end();
  static int get_layer_numbers(std::string workload_input);
  CSVWriter* detailed;
  CSVWriter* end_to_end;
  CSVWriter* dimension_utilization;
  std::string path;
  std::string run_name;
  int stat_row;
  int total_rows;
  bool registered_for_finished_streams;
};
} // namespace AstraSim
#endif
