/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Workload.hh"
#include "CSVWriter.hh"
#include "Layer.hh"
#include "astra-sim/system/MockNcclLog.h"

namespace AstraSim {
Workload::~Workload() {
  if (end_to_end != nullptr) {
    delete end_to_end;
  }
  if (detailed != nullptr) {
    delete detailed;
  }
  if (dimension_utilization != nullptr) {
    delete dimension_utilization;
  }
  for (int i = 0; i < SIZE; i++) {
    delete layers[i];
  }
  if (layers != nullptr) {
    delete[] layers;
  }
}
Workload::Workload(
    std::string run_name,
    Sys* generator,
    std::string name,
    int TOTAL_PASS,
    int total_rows,
    int stat_row,
    std::string path,
    bool seprate_log) {
  this->initialized = false;
  this->layers = nullptr;
  this->SIZE = 0;
  this->counter = 0;
  this->delay_loaded = false;
  this->checkpoint_initiated = false;
  this->collective_issued = false;
  this->current_state = LoopState::Forward_Pass;
  this->generator = generator;
  this->TOTAL_PASS = TOTAL_PASS;
  this->pass_counter = 0;
  this->index = 0;
  this->waiting_for_comm = 0;
  end_to_end = nullptr;
  detailed = nullptr;
  dimension_utilization = nullptr;
  this->path = path;
  this->stat_row = stat_row;
  this->seprate_log = seprate_log;
  this->initialized = initialize_workload(name);
  if (this->initialized == false) {
    return;
  }
  this->total_rows = total_rows;
  this->run_name = run_name;
  this->registered_for_finished_streams = false;
  #ifndef PHY_MTP
  if (generator->id == 0 && seprate_log) {
    std::cout << "stat path: " << path << " ,total rows: " << total_rows
              << " ,stat row: " << stat_row << std::endl;
    detailed = new CSVWriter(path, "detailed_"+std::to_string(generator->total_nodes)+".csv");
    end_to_end = new CSVWriter(path, "EndToEnd_"+std::to_string(generator->total_nodes)+".csv");
    dimension_utilization =
        new CSVWriter(path, run_name + "_dimension_utilization_"+std::to_string(generator->npu_offset)+".csv");
    if (stat_row == 0) {
      initialize_stat_files();
    }
  }
  #endif
}
void Workload::initialize_stat_files() {
  detailed->initialize_csv(SIZE * total_rows + 20, 50);
  end_to_end->initialize_csv(SIZE * total_rows + 20, 50);
}
void Workload::call(EventType event, CallData* data) {
  if (counter > 0) {
    if(generator->id == 0) std::cout << "counter > 0" << std::endl;
    generator->try_register_event(
        this, EventType::Workload_Wait, NULL, counter);
    return;
  }
  if (parallelismPolicy == ParallelismPolicy::Data) {
    iterate_data_parallel();
  } else if (parallelismPolicy == ParallelismPolicy::Transformer) {
    iterate_hybrid_parallel_Transformer();
  } else if (
      parallelismPolicy == ParallelismPolicy::DLRM ||
      parallelismPolicy == ParallelismPolicy::DLRMEnhanced) {
    iterate_hybrid_parallel_DLRM();
  } else if (parallelismPolicy == ParallelismPolicy::MicroBenchmark) {
    iterate_micro_benchmark();
  } else if (parallelismPolicy == ParallelismPolicy::Model) {
    iterate_model_parallel();
  } else if (parallelismPolicy == ParallelismPolicy::HybridDataModel) {
    iterate_hybrid_parallel_data_model();
  } else if (parallelismPolicy == ParallelismPolicy::HybridModelData) {
    iterate_hybrid_parallel_model_data();
  } else if (parallelismPolicy == ParallelismPolicy::DistributedInference) {
    iterate_distributed_inference();
  } else if (parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd) {
    iterate_hybrid_parallel_Transformer_fwd_in_bckwd();
  } else if (parallelismPolicy == ParallelismPolicy::HybridCustomized) {
    iterate_hybrid_parallel_customized();
  } else {
    Sys::sys_panic("No known parallelism!");
  }
}
void Workload::report() {
  double total_compute = 0;
  double total_exposed = 0;
  #ifdef ANALYTI
  double pre_bubble_time = 0;
  double DP_comm = 0;
  double Expose_TP_comm = 0;
  #endif
  std::vector<double> total_fwd_time = {0, 0, 0};
  std::vector<double> total_wg_time = {0, 0, 0};
  std::vector<double> total_ig_time = {0, 0, 0};
  AstraSimDataAPI astraSimDataAPI;
  astraSimDataAPI.run_name = run_name;
  astraSimDataAPI.workload_finished_time = ((double)Sys::boostedTick()) / FREQ;
  std::cout<<"workload stats for the job scheduled at NPU offset: "
            <<generator->npu_offset<<std::endl;
  for (int i = 0; i < SIZE; i++) {
    #ifdef ANALYTI
    astraSimDataAPI.layers_stats.push_back(layers[i]->report(
        run_name,
        i,
        total_rows,
        stat_row,
        detailed,
        end_to_end,
        total_compute,
        total_exposed,
        pre_bubble_time,
        DP_comm,
        Expose_TP_comm,
        this->seprate_log));
    #else
    astraSimDataAPI.layers_stats.push_back(layers[i]->report(
        run_name,
        i,
        total_rows,
        stat_row,
        detailed,
        end_to_end,
        total_compute,
        total_exposed,
        this->seprate_log,
        total_fwd_time,
        total_wg_time,
        total_ig_time));
    #endif
  }
  astraSimDataAPI.total_compute = total_compute;
  astraSimDataAPI.total_exposed_comm = total_exposed;
  astraSimDataAPI.avg_chunk_latency_per_logical_dimension =
      generator->scheduler_unit->get_average_latency_per_dimension();
  for (auto& latency :
       astraSimDataAPI.avg_chunk_latency_per_logical_dimension) {
    latency /= FREQ;
  }
  std::cout << "*************************" << std::endl;
  std::cout << "all passes finished at time: " << Sys::boostedTick()
            << ", id of first layer: " << layers[0]->id << std::endl;
  generator->NI->pass_front_end_report(astraSimDataAPI);

  if (this->seprate_log) {
    std::list<std::list<std::pair<uint64_t, double>>> dims;
    for (int i = 0; i < generator->scheduler_unit->usage.size(); i++) {
      dims.push_back(
          generator->scheduler_unit->usage[i].report_percentage(10000));
    }
    dimension_utilization->finalize_csv(dims);
  }
}
void Workload::check_for_sim_end() {
  if (pass_counter == TOTAL_PASS) {
    current_state = LoopState::Wait_For_Sim_Finish;
    if (generator->streams_finished != generator->streams_injected &&
        registered_for_finished_streams == false) {
      generator->register_for_finished_stream(this);
      registered_for_finished_streams = true;
      layers[0]->is_weight_grad_comm_finished_blocking();
      return;
    }
    if (generator->streams_finished == generator->streams_injected) {
      #ifndef PHY_MTP
      if (generator->id == 0) {
        report();
      }
      #endif
      generator->workload_finished();
      return;
    }
  }
  return;
}
void Workload::iterate_micro_benchmark() {
  assert(index >= 0);
  assert(index < SIZE);
  if (current_state != LoopState::Wait_For_Sim_Finish) {
    for (pass_counter = 0; pass_counter < TOTAL_PASS; pass_counter++) {
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
    }
  }
  check_for_sim_end();
}
void Workload::iterate_data_parallel() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    index++;
    delay_loaded = false;
    if (index >= SIZE) {
      current_state = LoopState::Weight_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    delay_loaded = false;
    layers[index]->issue_weight_grad_comm(
        SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
    if (index == 0) {
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    delay_loaded = false;
    index--;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_hybrid_parallel_customized() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_hybrid_parallel_data_model() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_hybrid_parallel_model_data() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_distributed_inference() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      index = 0;
      pass_counter++;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_model_parallel() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      std::vector<bool> involved_dimensions{true, true, true};
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      std::vector<bool> involved_dimensions{true, true, true};
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_hybrid_parallel_Transformer() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Blocking);
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_hybrid_parallel_Transformer_fwd_in_bckwd() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG,"workload::call fwd_pass register_event EventType::General ");
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (layers[index]->needs_fwd_in_bckwd_initiation && !checkpoint_initiated) {
      int tmp = index;
      while (!layers[index--]->is_checkpoint)
        ;
      index++;
      current_state = LoopState::Forward_In_BackPass;
      checkpoint_initiated = true;
      generator->register_event(this, EventType::General, NULL, 1);
      if (generator->id == 0) {
        std::cout << "***** info, initiating fwd_in_bkwd starting from layer:"
                  << index << " to layer: " << tmp
                  << " ,at time: " << Sys::boostedTick() << std::endl;
      }
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Blocking);
      return;
    }
    checkpoint_initiated = false;
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Forward_In_BackPass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (layers[index]->needs_fwd_in_bckwd_initiation) {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
void Workload::iterate_hybrid_parallel_DLRM() {
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued &&
        layers[index]->fwd_pass_comm_type == ComType::All_to_All) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::HIGHEST, CollectiveBarrier::Non_Blocking);

    } else if (index == DLRM_LAST_BOTTOM_LAYER) {
      if (!layers[0]->is_fwd_pass_comm_finished_blocking()) {
        return;
      }
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Weight_Gradient;
      index--;
    }
    if (generator->id == 0) {
      std::cout << "*************************layer changed to: " << index
                << std::endl;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
    }
    if (parallelismPolicy == ParallelismPolicy::DLRM &&
        !layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    if (index == 0) {
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    delay_loaded = false;
    collective_issued = false;
    generator->register_event(this, EventType::General, NULL, 1);
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (index == DLRM_LAST_BOTTOM_LAYER + 1) {
      layers[0]->issue_input_grad_comm(
          SchedulingPolicy::HIGHEST, CollectiveBarrier::Non_Blocking);
    }
    index--;
    if (generator->id == 0) {
      std::cout << "*************************layer changed to: " << index
                << " in ig" << std::endl;
    }
    current_state = LoopState::Weight_Gradient;
    collective_issued = false;
    delay_loaded = false;
    generator->register_event(this, EventType::General, NULL, 1);
  }
}
int Workload::get_layer_numbers(std::string workload_input) {
  std::ifstream inFile;
  inFile.open("workload_inputs/" + workload_input);
  if (!inFile) {
    std::cerr << "Unable to open file: " << workload_input << std::endl;
    std::cerr << "This error is fatal. Please check your path and filename."
              << std::endl;
    exit(1);
  } else {
    std::cout << "Success in opening workload file" << std::endl;
  }
  std::string dummyLine;
  std::getline(inFile, dummyLine);
  int layers;
  inFile >> layers;
  inFile.close();
  return layers;
}
ParallelismPolicy Workload::decode_parallelsim(std::string parallelism) {
  if (parallelism == "DATA")
    return ParallelismPolicy::Data;
  else if (parallelism == "HYBRID_TRANSFORMER")
    return ParallelismPolicy::Transformer;
  else if (parallelism == "HYBRID_TRANSFORMER_FWD_IN_BCKWD")
    return ParallelismPolicy::TransformerFwdInBckwd;
  else if (parallelism == "HYBRID_DLRM")
    return ParallelismPolicy::DLRM;
  else if (parallelism == "HYBRID_DLRM_ENHANCED")
    return ParallelismPolicy ::DLRMEnhanced;
  else if (parallelism == "MODEL")
    return ParallelismPolicy::Model;
  else if (parallelism == "HYBRID_DATA_MODEL")
    return ParallelismPolicy::HybridDataModel;
  else if (parallelism == "HYBRID_MODEL_DATA")
    return ParallelismPolicy::HybridModelData;
  else if (parallelism == "HYBRID_CUSTOMIZED")
    return ParallelismPolicy::HybridCustomized;
  else if (parallelism == "MICRO")
    return ParallelismPolicy::MicroBenchmark;
  else if (parallelism == "DISTRIBUTED_INFERENCE")
    return ParallelismPolicy::DistributedInference;
  else
    return ParallelismPolicy::None;
}
std::map<std::string, std::vector<bool>> Workload::decode_involved_dimensions(
    ParallelismPolicy policy,
    int model_parallel_npu_group) {
  std::map<std::string, std::vector<bool>> result;
  std::vector<bool> none{
      false, false, false, false, false, false, false, false, false, false};
  std::vector<bool> all{
      true, true, true, true, true, true, true, true, true, true};
  if (policy == ParallelismPolicy::All) {
    result["fwd"] = all;
    result["ig"] = all;
    result["wg"] = all;
  } else if (
      policy == ParallelismPolicy::Data || policy == ParallelismPolicy::DLRM ||
      policy == ParallelismPolicy::DLRMEnhanced ||
      policy == ParallelismPolicy::MicroBenchmark) {
    result["fwd"] = none;
    result["ig"] = none;
    result["wg"] = all;
  } else if (
      policy == ParallelismPolicy::Model ||
      policy == ParallelismPolicy::DistributedInference) {
    result["fwd"] = all;
    result["ig"] = all;
    result["wg"] = none;
  } else if (policy == ParallelismPolicy::HybridModelData) {
    std::vector<bool> data{
        true, false, false, false, false, false, false, false, false, false};
    std::vector<bool> model{
        false, true, true, true, true, true, true, true, true, true};
    result["fwd"] = model;
    result["ig"] = model;
    result["wg"] = data;
  } else if (policy == ParallelismPolicy::HybridDataModel) {
    std::vector<bool> model{
        true, false, false, false, false, false, false, false, false, false};
    std::vector<bool> data{
        false, true, true, true, true, true, true, true, true, true};
    result["fwd"] = model;
    result["ig"] = model;
    result["wg"] = data;
  } else if (
      policy == ParallelismPolicy::TransformerFwdInBckwd ||
      policy == ParallelismPolicy::Transformer) {
    int model_parallel_boundary =
        generator->break_dimension(model_parallel_npu_group);
    std::vector<bool> model;
    std::vector<bool> data;
    for (int i = 0; i <= model_parallel_boundary; i++) {
      model.push_back(true);
      data.push_back(false);
    }
    for (int i = model_parallel_boundary + 1; i < 10; i++) {
      model.push_back(false);
      data.push_back(true);
    }
    result["fwd"] = model;
    result["ig"] = model;
    result["wg"] = data;
  }
  return result;
}
bool Workload::initialize_workload(std::string name) {
  std::map<int, bool> chekpoints;
  std::map<int, bool> need_checkpoint_initiation;
  std::ifstream inFile;
  inFile.open(name);
  if (!inFile) {
    std::cerr << "Unable to open file: " << name << std::endl;
    std::cerr << "######### Exiting because unable to open the workload input "
                 "file #########"
              << std::endl;
    std::cerr << "This error is fatal. Please check your path and filename."
              << std::endl;
    exit(1);
  } else {
    if (generator->id == 0) {
      std::cout << "Success in opening workload file" << std::endl;
    }
  }
  std::string type;
  int lines;
  inFile >> type;
  parallelismPolicy = decode_parallelsim(type);
  if (parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd ||
      parallelismPolicy == ParallelismPolicy::Transformer) {
    std::string tmp;
    int i;
    inFile >> tmp;
    inFile >> model_parallel_npu_group;
    if (generator->id == 0) {
      std::cout << tmp << " is: " << model_parallel_npu_group << std::endl;
    }
    inFile >> tmp;
    inFile >> expert_parallel_npu_group;
    inFile >> tmp;
    inFile >> pipeline_model_parallelism;
    inFile >> tmp;
    inFile >> GA;
    inFile >> tmp;
    inFile >> all_gpus;
    if (parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd) {
      inFile >> tmp;
      inFile >> i;
      if (generator->id == 0) {
        std::cout << "checkpoints layers are: ";
      }
      while (i-- > 0) {
        int layer;
        inFile >> layer;
        chekpoints[layer] = true;
        if (generator->id == 0) {
          std::cout << layer << ", ";
        }
      }
      if (generator->id == 0) {
        std::cout << std::endl;
        std::cout << "layers initiating fwd_in_bckwd are: ";
      }
      inFile >> tmp;
      inFile >> i;
      while (i-- > 0) {
        int layer;
        inFile >> layer;
        need_checkpoint_initiation[layer] = true;
        if (generator->id == 0) {
          std::cout << layer << ", ";
        }
      }
      if (generator->id == 0) {
        std::cout << std::endl;
      }
    }
  } else if (
      parallelismPolicy == ParallelismPolicy::DLRM ||
      parallelismPolicy == ParallelismPolicy::DLRMEnhanced) {
    inFile >> DLRM_LAST_BOTTOM_LAYER;
    if (generator->id == 0) {
      std::cout
          << "****************** info: DLRM workload last bottom layer is: "
          << DLRM_LAST_BOTTOM_LAYER << std::endl;
    }
  } else if (parallelismPolicy == ParallelismPolicy::None) {
    #ifndef PHY_MTP
    std::cerr << "######### Exiting because unable to decode the workload "
                 "parallelization strategy #########"
              << std::endl;
    inFile.close();
    exit(1);
    #else
    parallelismPolicy = ParallelismPolicy::TransformerFwdInBckwd;
    #endif
  }
  std::map<std::string, std::vector<bool>> general_involved_dimensions =
      decode_involved_dimensions(parallelismPolicy, model_parallel_npu_group);
  inFile >> lines;
  run_type = type;
  SIZE = lines;
  layers = new Layer*[SIZE];
  for (int i = 0; i < lines; i++) {
    std::string id;
    inFile >> id;
    int depen;
    inFile >> depen;

    Tick fp_compute_time;
    inFile >> fp_compute_time;
    std::string fp_comm_type_s;
    inFile >> fp_comm_type_s;
    uint64_t fp_comm_size;
    inFile >> fp_comm_size;

    Tick ig_compute_time;
    inFile >> ig_compute_time;
    std::string ig_comm_type_s;
    inFile >> ig_comm_type_s;
    uint64_t ig_comm_size;
    inFile >> ig_comm_size;

    Tick wg_compute_time;
    inFile >> wg_compute_time;
    std::string wg_comm_type_s;
    inFile >> wg_comm_type_s;
    uint64_t wg_comm_size;
    inFile >> wg_comm_size;
    Tick wg_update_time;
    inFile >> wg_update_time;

    ParallelismPolicy specific_policy = ParallelismPolicy::None;
    std::map<std::string, std::vector<bool>> selected_involved_dimensions;
    ComType fp_type = ComType::None;
    ComType ig_type = ComType::None;
    ComType wg_type = ComType::None;
    MockNccl::GroupType fp_group_type = MockNccl::GroupType::NONE;
    MockNccl::GroupType ig_group_type = MockNccl::GroupType::NONE;
    MockNccl::GroupType wg_group_type = MockNccl::GroupType::NONE;
    #ifdef ANALYTI
    if (wg_comm_type_s.substr(0,9) == "ALLREDUCE") {
      wg_type = ComType::All_Reduce;
      if(wg_comm_type_s == "ALLREDUCE"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLREDUCE_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLREDUCE_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,8) == "ALLTOALL") {
      wg_type = ComType::All_to_All;
      if(wg_comm_type_s == "ALLTOALL"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLTOALL_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLTOALL_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,17) == "ALLREDUCEALLTOALL") {
      wg_type = ComType::All_Reduce_All_to_All;
      if(wg_comm_type_s == "ALLREDUCEALLTOALL"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLREDUCEALLTOALL_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLREDUCEALLTOALL_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,9) == "ALLGATHER") {
      wg_type = ComType::All_Gatehr;
      if(wg_comm_type_s == "ALLGATHER"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLGATHER_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLGATHER_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,13) == "REDUCESCATTER") {
      wg_type = ComType::Reduce_Scatter;
      if(wg_comm_type_s == "REDUCESCATTER"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "REDUCESCATTER_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "REDUCESCATTER_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    }

    // generate flow model

    if (ig_comm_type_s.substr(0,9) == "ALLREDUCE") {
      ig_type = ComType::All_Reduce;
      if(ig_comm_type_s == "ALLREDUCE"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLREDUCE_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLREDUCE_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,8) == "ALLTOALL") {
      ig_type = ComType::All_to_All;
      if(ig_comm_type_s == "ALLTOALL"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLTOALL_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLTOALL_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,17) == "ALLREDUCEALLTOALL") {
      ig_type = ComType::All_Reduce_All_to_All;
      if(ig_comm_type_s == "ALLREDUCEALLTOALL"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLREDUCEALLTOALL_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLREDUCEALLTOALL_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,9) == "ALLGATHER") {
      ig_type = ComType::All_Gatehr;
      if(ig_comm_type_s == "ALLGATHER"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLGATHER_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLGATHER_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,13) == "REDUCESCATTER") {
      ig_type = ComType::Reduce_Scatter;
      if(ig_comm_type_s == "REDUCESCATTER"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "REDUCESCATTER_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "REDUCESCATTER_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    }

    if (fp_comm_type_s.substr(0,9) == "ALLREDUCE") {
      fp_type = ComType::All_Reduce;
      if(fp_comm_type_s == "ALLREDUCE"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLREDUCE_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLREDUCE_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,8) == "ALLTOALL") {
      fp_type = ComType::All_to_All;
      if(fp_comm_type_s == "ALLTOALL"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLTOALL_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLTOALL_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,17) == "ALLREDUCEALLTOALL") {
      fp_type = ComType::All_Reduce_All_to_All;
      if(fp_comm_type_s == "ALLREDUCEALLTOALL"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLREDUCEALLTOALL_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLREDUCEALLTOALL_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,9) == "ALLGATHER") {
      fp_type = ComType::All_Gatehr;
      if(fp_comm_type_s == "ALLGATHER"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLGATHER_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLGATHER_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,13) == "REDUCESCATTER") {
      fp_type = ComType::Reduce_Scatter;
      if(fp_comm_type_s == "REDUCESCATTER"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "REDUCESCATTER_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "REDUCESCATTER_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    }
    #else
    if (wg_comm_type_s == "ALLREDUCE") {
      wg_type = ComType::All_Reduce;
    } else if (wg_comm_type_s == "ALLTOALL") {
      wg_type = ComType::All_to_All;
    } else if (wg_comm_type_s == "ALLREDUCEALLTOALL") {
      wg_type = ComType::All_Reduce_All_to_All;
    } else if (wg_comm_type_s == "ALLGATHER") {
      wg_type = ComType::All_Gatehr;
    } else if (wg_comm_type_s == "REDUCESCATTER") {
      wg_type = ComType::Reduce_Scatter;
    }

    if (ig_comm_type_s == "ALLREDUCE") {
      ig_type = ComType::All_Reduce;
    } else if (ig_comm_type_s == "ALLTOALL") {
      ig_type = ComType::All_to_All;
    } else if (ig_comm_type_s == "ALLREDUCEALLTOALL") {
      ig_type = ComType::All_Reduce_All_to_All;
    } else if (ig_comm_type_s == "ALLGATHER") {
      ig_type = ComType::All_Gatehr;
    } else if (ig_comm_type_s == "REDUCESCATTER") {
      ig_type = ComType::Reduce_Scatter;
    }

    if (fp_comm_type_s == "ALLREDUCE") {
      fp_type = ComType::All_Reduce;
    } else if (fp_comm_type_s == "ALLTOALL") {
      fp_type = ComType::All_to_All;
    } else if (fp_comm_type_s == "ALLREDUCEALLTOALL") {
      fp_type = ComType::All_Reduce_All_to_All;
    } else if (fp_comm_type_s == "ALLGATHER") {
      fp_type = ComType::All_Gatehr;
    } else if (fp_comm_type_s == "REDUCESCATTER") {
      fp_type = ComType::Reduce_Scatter;
    }

    #endif
    if (generator->id == 0) {
      std::cout << "id: " << id << " , depen: " << depen
                << " , wg_comp_time: " << wg_compute_time << std::endl;
    }
    if (parallelismPolicy == ParallelismPolicy::HybridCustomized) {
      std::string specific_parallelsim;
      inFile >> specific_parallelsim;
      specific_policy = decode_parallelsim(specific_parallelsim);
    }
    if ((parallelismPolicy == ParallelismPolicy::DLRM ||
         parallelismPolicy == ParallelismPolicy::DLRMEnhanced) &&
        i == 0) {
      specific_policy = ParallelismPolicy::All;
    }
    if (specific_policy != ParallelismPolicy::None) {
      selected_involved_dimensions =
          decode_involved_dimensions(specific_policy, model_parallel_npu_group);
    } else {
      selected_involved_dimensions = general_involved_dimensions;
    }
    Layer* l = new Layer(
        id,
        i,
        generator,
        this,
        fp_compute_time * generator->compute_scale,
        fp_type,
        fp_group_type,
        fp_comm_size * generator->comm_scale,
        selected_involved_dimensions["fwd"],
        ig_compute_time * generator->compute_scale,
        ig_type,
        ig_group_type,
        ig_comm_size * generator->comm_scale,
        selected_involved_dimensions["ig"],
        wg_compute_time * generator->compute_scale,
        wg_type,
        wg_group_type,
        wg_comm_size * generator->comm_scale,
        selected_involved_dimensions["wg"],
        wg_update_time,
        specific_policy);
    if (chekpoints.find(i) != chekpoints.end()) {
      l->is_checkpoint = true;
    }
    if (need_checkpoint_initiation.find(i) !=
        need_checkpoint_initiation.end()) {
      l->needs_fwd_in_bckwd_initiation = true;
    }
    layers[i] = l;
  }
  if (generator->id == 0) {
    std::cout << "type: " << type << " ,num passes: " << TOTAL_PASS
              << " ,lines: " << lines
              << " compute scale: " << generator->compute_scale
              << " ,comm scale: " << generator->comm_scale << std::endl;
  }
  inFile.close();
  return true;
}
void Workload::fire() {
  call(EventType::General, NULL);
}
} // namespace AstraSim
