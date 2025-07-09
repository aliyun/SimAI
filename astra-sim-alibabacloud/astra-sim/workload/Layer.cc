/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Layer.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/AstraParamParse.hh"
// #ifdef ANALYTI
#include "astra-sim/system/calbusbw.h"
// #endif

#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
using namespace ns3;
#endif




namespace AstraSim {
Layer::Layer(
    std::string id,
    int layer_num,
    Sys* generator,
    Workload* workload,
    Tick fwd_pass_compute_time,
    ComType fwd_pass_comm_type,
    MockNccl::GroupType fwd_pass_group_type,
    uint64_t fwd_pass_comm_size,
    std::vector<bool> fwd_pass_comm_involved_dimensions,
    Tick input_grad_compute_time,
    ComType input_grad_comm_type,
    MockNccl::GroupType input_grad_group_type,
    uint64_t input_grad_comm_size,
    std::vector<bool> input_grad_comm_involved_dimensions,
    Tick weight_grad_compute_time,
    ComType weight_grad_comm_type,
    MockNccl::GroupType weight_grad_group_type,
    uint64_t weight_grad_comm_size,
    std::vector<bool> weight_grad_comm_involved_dimensions,
    Tick weight_grad_update_time,
    ParallelismPolicy specific_policy) {
  this->id = id;
  this->layer_num = layer_num;
  this->generator = generator;
  this->workload = workload;
  this->fwd_pass_compute_time = fwd_pass_compute_time;
  this->fwd_pass_comm_type = fwd_pass_comm_type;
  this->fwd_pass_group_type = fwd_pass_group_type;
  this->fwd_pass_comm_size = fwd_pass_comm_size;
  this->fwd_pass_comm_involved_dimensions = fwd_pass_comm_involved_dimensions;
  this->input_grad_compute_time = input_grad_compute_time;
  this->input_grad_comm_type = input_grad_comm_type;
  this->input_grad_group_type = input_grad_group_type;
  this->input_grad_comm_size = input_grad_comm_size;
  this->input_grad_comm_involved_dimensions =
      input_grad_comm_involved_dimensions;
  this->weight_grad_compute_time = weight_grad_compute_time;
  this->weight_grad_comm_type = weight_grad_comm_type;
  this->weight_grad_group_type = weight_grad_group_type;
  this->weight_grad_comm_size = weight_grad_comm_size;
  this->weight_grad_comm_involved_dimensions =
      weight_grad_comm_involved_dimensions;
  this->collective_counter = 0;

  this->weight_grad_update_time = weight_grad_update_time;
  this->fwd_update_time = weight_grad_update_time;
  this->input_grad_update_time = weight_grad_update_time;

  this->total_forward_pass_compute = 0;
  this->total_input_grad_compute = 0;
  this->total_weight_grad_compute = 0;
  this->total_weight_grad_comm = 0;
  this->total_input_grad_comm = 0;
  this->total_fwd_comm = 0;
  this->total_waiting_for_wg_comm = 0;
  this->total_waiting_for_ig_comm = 0;
  this->total_waiting_for_fwd_comm = 0;
  this->last_fwd_finished = 0;
  this->last_ig_finished = 0;
  this->last_wg_finished = 0;
  this->needs_fwd_in_bckwd_initiation = false;
  this->is_checkpoint = false;
  this->specific_parallellism = specific_policy;
  assert(generator != NULL);
}

void Layer::call(EventType event, CallData* mdata) {
  if (event == EventType::Wight_Grad_Comm_Finished) {
    last_wg_finished = Sys::boostedTick();
    generator->register_event(
        this,
        EventType::Wight_Grad_Comm_Finished_After_Delay,
        mdata,
        weight_grad_update_time);
    return;
  } else if (event == EventType::Input_Grad_Comm_Finished) {
    last_ig_finished = Sys::boostedTick();
    generator->register_event(
        this,
        EventType::Input_Grad_Comm_Finished_After_Delay,
        mdata,
        input_grad_update_time);
    return;
  } else if (event == EventType::Fwd_Comm_Finished) {
    last_fwd_finished = Sys::boostedTick();
    generator->register_event(
        this, EventType::Fwd_Comm_Finished_After_Delay, mdata, fwd_update_time);
    return;
  }
  int data = ((IntData*)mdata)->data;
  IntData* intData = ((IntData*)mdata);
  if (event == EventType::Wight_Grad_Comm_Finished_After_Delay) {
    #ifndef PHY_MTP
    if (generator->id == 0) {
      std::cout << "***** info: weight gradient collective for layer: " << id
                << " is finished************" << std::endl;
    }
    weight_grad_datasets[data]->finish_tick += weight_grad_update_time;
    total_weight_grad_comm += weight_grad_datasets[data]->finish_tick -
        weight_grad_datasets[data]->creation_tick;

    if (weight_grad_datasets.size() == 1 &&
        wg_barrier == CollectiveBarrier::Blocking) { 
      total_waiting_for_wg_comm += weight_grad_datasets[data]->finish_tick -
          weight_grad_datasets[data]->creation_tick;
      update_stream_stats(weight_grad_datasets[data]);
      int dataset_streams = weight_grad_datasets[data]->total_streams;
      delete weight_grad_datasets[data];
      weight_grad_datasets.erase(data);
      workload->call(EventType::General, NULL);
      generator->increase_finished_streams(dataset_streams);
      delete intData;
      return;
    } else if (started_waiting_for_weight_grad.size() > 0) {  
      total_waiting_for_wg_comm += weight_grad_datasets[data]->finish_tick -
          started_waiting_for_weight_grad.front();
      started_waiting_for_weight_grad.pop_front();
      update_stream_stats(weight_grad_datasets[data]);
      int dataset_streams = weight_grad_datasets[data]->total_streams;
      delete weight_grad_datasets[data];
      weight_grad_datasets.erase(data);
      workload->call(EventType::General, NULL);
      generator->increase_finished_streams(dataset_streams);
      delete intData;
      return;
    }
    update_stream_stats(weight_grad_datasets[data]);
    int dataset_streams = weight_grad_datasets[data]->total_streams;
    delete weight_grad_datasets[data];
    weight_grad_datasets.erase(data);
    generator->increase_finished_streams(dataset_streams);
    delete intData;
    #else
    workload->call(EventType::General, NULL);
    generator->increase_finished_streams(1);
    #endif
    return;
  } else if (event == EventType::Input_Grad_Comm_Finished_After_Delay) {
    #ifndef PHY_MTP
    if (generator->id == 0) {
      std::cout << "***** info: input gradient collective for layer: " << id
                << " is finished************" << std::endl;
    }
    input_grad_datasets[data]->finish_tick += input_grad_update_time;
    total_input_grad_comm += input_grad_datasets[data]->finish_tick -
        input_grad_datasets[data]->creation_tick;
    if (input_grad_datasets.size() == 1 &&
        ig_barrier == CollectiveBarrier::Blocking) {
      total_waiting_for_ig_comm += input_grad_datasets[data]->finish_tick -
          input_grad_datasets[data]->creation_tick;
      update_stream_stats(input_grad_datasets[data]);
      int dataset_streams = input_grad_datasets[data]->total_streams;
      delete input_grad_datasets[data];
      input_grad_datasets.erase(data);
      workload->call(EventType::General, NULL);
      generator->increase_finished_streams(dataset_streams);
      delete intData;
      return;
    } else if (started_waiting_for_input_grad.size() > 0) {
      total_waiting_for_ig_comm += input_grad_datasets[data]->finish_tick -
          started_waiting_for_input_grad.front();
      started_waiting_for_input_grad.pop_front();
      update_stream_stats(input_grad_datasets[data]);
      int dataset_streams = input_grad_datasets[data]->total_streams;
      delete input_grad_datasets[data];
      input_grad_datasets.erase(data);
      workload->call(EventType::General, NULL);
      generator->increase_finished_streams(dataset_streams);
      delete intData;
      return;
    }
    update_stream_stats(input_grad_datasets[data]);
    int dataset_streams = input_grad_datasets[data]->total_streams;
    delete input_grad_datasets[data];
    input_grad_datasets.erase(data);
    generator->increase_finished_streams(dataset_streams);
    delete intData;
    #else
    workload->call(EventType::General, NULL);
    generator->increase_finished_streams(1);
    #endif
    return;
  } else if (event == EventType::Fwd_Comm_Finished_After_Delay) {
    #ifndef PHY_MTP
    if (generator->id == 0) {
      std::cout << "***** info: fwd pass comm collective for layer: " << id
                << " is finished************" << std::endl;
    }
    fwd_pass_datasets[data]->finish_tick += fwd_update_time;
    total_fwd_comm += fwd_pass_datasets[data]->finish_tick -
        fwd_pass_datasets[data]->creation_tick;
    if (fwd_pass_datasets.size() == 1 &&
        fwd_barrier == CollectiveBarrier::Blocking) {
      total_waiting_for_fwd_comm += fwd_pass_datasets[data]->finish_tick -
          fwd_pass_datasets[data]->creation_tick;
      update_stream_stats(fwd_pass_datasets[data]);
      int dataset_streams = fwd_pass_datasets[data]->total_streams;
      delete fwd_pass_datasets[data];
      fwd_pass_datasets.erase(data);
      workload->call(EventType::General, NULL);
      generator->increase_finished_streams(dataset_streams);
      delete intData;
      return;
    } else if (started_waiting_for_fwd_pass.size() > 0) {
      total_waiting_for_fwd_comm += fwd_pass_datasets[data]->finish_tick -
          started_waiting_for_fwd_pass.front();
      started_waiting_for_fwd_pass.pop_front();
      update_stream_stats(fwd_pass_datasets[data]);
      int dataset_streams = fwd_pass_datasets[data]->total_streams;
      delete fwd_pass_datasets[data];
      fwd_pass_datasets.erase(data);
      workload->call(EventType::General, NULL);
      generator->increase_finished_streams(dataset_streams);
      delete intData;
      return;
    }
    update_stream_stats(fwd_pass_datasets[data]);
    int dataset_streams = fwd_pass_datasets[data]->total_streams;
    delete fwd_pass_datasets[data];
    fwd_pass_datasets.erase(data);
    generator->increase_finished_streams(dataset_streams);
    delete intData;
    #else
    workload->call(EventType::General, NULL);
    generator->increase_finished_streams(1);
    #endif
    return;
  }
}

Tick Layer::get_fwd_pass_compute() {
  total_forward_pass_compute += fwd_pass_compute_time;
  return fwd_pass_compute_time;
}
Tick Layer::get_input_grad_compute() {
  total_input_grad_compute += input_grad_compute_time;
  return input_grad_compute_time;
}
Tick Layer::get_weight_grad_compute() {
  total_weight_grad_compute += weight_grad_compute_time;
  return weight_grad_compute_time;
}
void Layer::increment_waiting_for_wg() {
  total_waiting_for_wg_comm++;
}
void Layer::increment_waiting_for_ig() {
  total_waiting_for_ig_comm++;
}
void Layer::increment_waiting_for_fwd() {
  total_waiting_for_fwd_comm++;
}
bool Layer::is_fwd_pass_comm_finished() {
  if (fwd_pass_datasets.size() ==
      0) { 
    return true;
  }
  return false;
}
bool Layer::is_fwd_pass_comm_finished_blocking() {
  if (fwd_pass_datasets.size() == 0) {
    return true;
  }
  if (started_waiting_for_fwd_pass.size() == 0) {
    started_waiting_for_fwd_pass.push_back(Sys::boostedTick());
  }
  return false;
}
bool Layer::is_input_grad_comm_finished() {
  if (input_grad_datasets.size() ==
      0) { 
    return true;
  }
  return false;
}
bool Layer::is_input_grad_comm_finished_blocking() {
  if (input_grad_datasets.size() ==
      0) { 
    return true;
  }
  if (started_waiting_for_input_grad.size() == 0) {
    started_waiting_for_input_grad.push_back(Sys::boostedTick());
  }
  return false;
}
bool Layer::is_weight_grad_comm_finished() {
  if (weight_grad_datasets.size() ==
      0) { 
    return true;
  }
  return false;
}
bool Layer::is_weight_grad_comm_finished_blocking() {
  if (weight_grad_datasets.size() == 0) {
    return true;
  }
  if (started_waiting_for_weight_grad.size() == 0) {
    this->started_waiting_for_weight_grad.push_back(Sys::boostedTick());
  }
  return false;
}
void Layer::print_involved_dimensions(std::vector<bool>& involved_dimensions) {
  std::cout << " involved dimensions: ";
  for (int i = 0; i < involved_dimensions.size(); i++) {
    if (involved_dimensions[i] == true) {
      std::cout << " 1,";
    } else {
      std::cout << " 0,";
    }
  }
  std::cout << std::endl;
}
LayerData Layer::report(
    std::string run_name,
    int layer_num,
    int total_rows,
    int stat_row,
    CSVWriter* detailed,
    CSVWriter* EndToEnd,
    double& total_compute,
    double& total_exposed,
    bool seprate_log,
    vector<double>& total_fwd_time,
    vector<double>& total_wg_time,
    vector<double>& total_ig_time,
    double& pre_bubble_time,
    double& DP_comm,
    double& DP_EP_comm,
    double& Expose_TP_comm,
    double& Expose_EP_comm) {
  LayerData layerData;
  take_stream_stats_average();
  int TP_size = workload->model_parallel_npu_group;
  int PP_size = workload->pipeline_model_parallelism;
  int DP_size = workload->all_gpus / (TP_size * PP_size);
  int EP_size = workload->expert_parallel_npu_group;
  int vpp = workload->vpp;
  uint32_t pp_commsize = workload->pp_commsize;
  int GA = workload->GA;
  UserParam* param = UserParam::getInstance();
  int input_grad_group_size =
      input_grad_group_type == MockNccl::GroupType::EP ? EP_size : TP_size;
  int fwd_pass_group_size =
      fwd_pass_group_type == MockNccl::GroupType::EP ? EP_size : TP_size;
  int weight_grad_group_size =
      weight_grad_group_type == MockNccl::GroupType::DP_EP ? DP_size / EP_size
                                                           : DP_size;
  if (id != "embedding_layer"){
      pre_bubble_time += ((total_waiting_for_fwd_comm + total_forward_pass_compute + total_weight_grad_compute + total_input_grad_compute + total_waiting_for_ig_comm) / FREQ);
    }
  if(weight_grad_group_type == MockNccl::GroupType::DP_EP){
    DP_EP_comm += (total_waiting_for_wg_comm / FREQ);
  }
  else{
    DP_comm += (total_waiting_for_wg_comm / FREQ);
  }
  if(fwd_pass_group_type == MockNccl::GroupType::EP){
    Expose_EP_comm += ((total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / FREQ);
  }
  else{
    Expose_TP_comm += ((total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / FREQ);
  }
  total_compute += (total_forward_pass_compute / FREQ);
  total_compute += (total_weight_grad_compute / FREQ);
  total_compute += (total_input_grad_compute / FREQ);
  total_exposed += (total_waiting_for_fwd_comm / FREQ);
  total_exposed += (total_waiting_for_wg_comm / FREQ);
  total_exposed += (total_waiting_for_ig_comm / FREQ);
  layerData.layer_name = id;
  layerData.total_forward_pass_compute = total_forward_pass_compute / FREQ;
  layerData.total_weight_grad_compute = total_weight_grad_compute / FREQ;
  layerData.total_input_grad_compute = total_input_grad_compute / FREQ;
  layerData.total_waiting_for_fwd_comm = total_waiting_for_fwd_comm / FREQ;
  layerData.total_waiting_for_wg_comm = total_waiting_for_wg_comm / FREQ;
  layerData.total_waiting_for_ig_comm = total_waiting_for_ig_comm / FREQ;
  layerData.total_fwd_comm = total_fwd_comm / FREQ;
  layerData.total_weight_grad_comm = total_weight_grad_comm / FREQ;
  layerData.total_input_grad_comm = total_input_grad_comm / FREQ;
  total_fwd_time[0] +=total_forward_pass_compute / FREQ;
  total_fwd_time[1] +=total_waiting_for_fwd_comm / FREQ;
  total_fwd_time[2] +=total_fwd_comm / FREQ;
  total_wg_time[0] +=total_weight_grad_compute / FREQ;
  total_wg_time[1] +=total_waiting_for_wg_comm / FREQ;
  total_wg_time[2] +=total_weight_grad_comm / FREQ;
  total_ig_time[0] +=total_input_grad_compute / FREQ;
  total_ig_time[1] +=total_waiting_for_ig_comm / FREQ;
  total_ig_time[2] +=total_input_grad_comm / FREQ;
  int i = 0;
  for (auto& qd : queuing_delay) {
    layerData.avg_queuing_delay.push_back(std::make_pair(i, qd / FREQ));
  }
  i = 1;
  for (auto& ml : net_message_latency) {
    layerData.avg_network_message_dealy.push_back(std::make_pair(i, ml / FREQ));
  }
  if (seprate_log)
  {
    std::string data;
    std::pair<float, float> total_bw;
    std::cout << "*******************" << std::endl;
    std::cout << "Layer id: " << id << std::endl;
    std::cout << "Total collectives issued for this layer: "
              << collective_counter << std::endl;
    std::cout << "*************************  Workload stats  "
                 "************************* "
              << id << std::endl;
    if(stat_row == 0 && layer_num == 0) {
      data = "layer_name,"+run_name+",fwd compute,wg compute,ig compute,fwd exposed comm,wg exposed comm,ig exposed comm,fwd total comm,algbw,busbw,wg total comm,algbw,busbw,ig total comm,algbw,busbw,workload finished at";
      EndToEnd->write_line(data);
    }
    data = "";
    if(stat_row == 0){
      data += id;
    }
    data = data + "," + run_name;

    std::cout << "id: " << id << " ,Total cycles spent on fwd pass compute: "
              << total_forward_pass_compute << std::endl;
    data = data + "," + std::to_string(total_forward_pass_compute/FREQ);

    std::cout << "id: " << id << " ,Total cycles spent on weight grad compute: "
              << total_weight_grad_compute << std::endl;
    data = data + "," + to_string(total_weight_grad_compute/FREQ);

    std::cout << "id: " << id << " ,Total cycles spent on input grad compute: "
              << total_input_grad_compute << std::endl;
    data = data + "," + to_string(total_input_grad_compute/FREQ);

    std::cout << "id: " << id
              << " ,Total cycles spent idle waiting for fwd finish: "
              << total_waiting_for_fwd_comm << std::endl;
    data = data + "," + to_string(total_waiting_for_fwd_comm/FREQ);

    std::cout << "id: " << id
              << " ,Total cycles spent idle waiting for weight grad finish: "
              << total_waiting_for_wg_comm << std::endl;
    data = data + "," + to_string(total_waiting_for_wg_comm / FREQ);

    std::cout << "id: " << id
              << " ,Total cycles spent idle waiting for input grad finish: "
              << total_waiting_for_ig_comm << std::endl;
    data = data + "," + to_string(total_waiting_for_ig_comm / FREQ);

    std::cout << "id: " << id
              << " ,Total cycles spent on fwd pass comm: " << total_fwd_comm
              << std::endl;
 
    total_bw = compute_busbw(fwd_pass_comm_type, fwd_pass_group_size, fwd_pass_comm_size, total_fwd_comm);
    data = data + "," + to_string(total_fwd_comm / FREQ);
    data = data + "," + to_string(total_bw.first);
    data = data + "," + to_string(total_bw.second);

    std::cout << "id: " << id << " ,Total cycles spent on weight grad comm: "
              << total_weight_grad_comm << std::endl;

    total_bw = compute_busbw(weight_grad_comm_type,weight_grad_group_size,weight_grad_comm_size,total_weight_grad_comm);
    data = data + "," + to_string(total_weight_grad_comm / FREQ);
    data = data + "," + to_string(total_bw.first);
    data = data + "," + to_string(total_bw.second);

    std::cout << "id: " << id << " ,Total cycles spent on input grad comm: "
              << total_input_grad_comm << std::endl;
    
    total_bw = compute_busbw(input_grad_comm_type,input_grad_group_size,input_grad_comm_size,total_input_grad_comm);
    data = data + "," + to_string(total_input_grad_comm / FREQ);
    data = data + "," + to_string(total_bw.first);
    data = data + "," + to_string(total_bw.second);
    data = data + "," + to_string(((double)Sys::boostedTick()) / FREQ);
    EndToEnd->write_line(data);

    data = "layer_name,"+run_name+",fwd compute,wg compute,ig compute,fwd exposed comm,wg exposed comm,ig exposed comm,fwd total comm,algbw,busbw,wg total comm,algbw,busbw,ig total comm,algbw,busbw,workload finished at";
    if (layer_num == workload->SIZE - 1) {
      total_exposed = (((double)Sys::boostedTick()) / FREQ) - total_compute;
      data = "SUM," + run_name + "," + to_string(total_fwd_time[0]) + "," + to_string(total_wg_time[0]) + "," + to_string(total_ig_time[0]) + "," + to_string(total_fwd_time[1]) + "," + to_string(total_wg_time[1]) + "," + to_string(total_ig_time[1]) + "," + to_string(total_fwd_time[2]) + ",NONE,NONE," + to_string(total_wg_time[2]) + ",NONE,NONE," + to_string(total_ig_time[2]) + ",NONE,NONE";
      EndToEnd->write_line(data);
      double total_time = total_compute + total_exposed;
      data = "total exposed comm," + to_string(total_exposed) + ",total comp," + to_string(total_compute) + ",total time," + to_string(total_time);
      EndToEnd->write_line(data);

      Tick Expose_PP_time = (2 * vpp * GA * (pp_commsize * GBps / (param->net_work_param.pp_overlap_ratio) * 1e9) / FREQ );
      Expose_PP_time *= (1-param->net_work_param.pp_overlap_ratio) ;
      //pp bubble time
      pre_bubble_time *= static_cast<double>(PP_size - 1) / (GA * vpp);
      auto format_value = [](double value) {
        std::ostringstream stream;
       if (std::isfinite(value)) {
           stream << std::fixed << std::setprecision(0) << value;
       } else {
           stream << "NaN or Inf";
       }
        return stream.str();
      };
      auto format_percentage = [&](double value) {
        double percentage = (value / total_time) * 100;
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << percentage;
        return stream.str() + "%";
        };
      std::string keys = "File name, Expose DP comm, Expose DP_EP comm, Expose TP comm, Expose_EP_comm, Expose_PP_comm, bubble time, total comp, total exposed comm, Total time";
      std::string values = run_name + ", " +
                          format_value(DP_comm) + " (" + format_percentage(DP_comm) + "), " +
                          format_value(DP_EP_comm) + " (" + format_percentage(DP_EP_comm) + "), " +
                          format_value(Expose_TP_comm) + " (" + format_percentage(Expose_TP_comm) + "), " +
                          format_value(Expose_EP_comm) + " (" + format_percentage(Expose_EP_comm) + "), " +
                          format_value(Expose_PP_time) + " (" + format_percentage(Expose_PP_time) + "), " +
                          format_value(pre_bubble_time) + " (" + format_percentage(pre_bubble_time) + "), " +
                          format_value(total_compute) + " (" + format_percentage(total_compute) + "), " +
                          format_value(total_exposed) + " (" + format_percentage(total_exposed) + "), " +
                          format_value(total_time);
      data = keys + "\n" + values;
      EndToEnd->write_res(data);
    }
  }
  return layerData;
}
std::string getFileName(const std::string& path) {
    size_t pos = path.find_last_of("/"); 
    if (pos != std::string::npos) {

        return path.substr(pos + 1, path.length() - pos - 1);
    }
    return path; 
}
LayerData Layer::report(
    std::string run_name,
    int layer_num,
    int total_rows,
    int stat_row,
    CSVWriter* detailed,
    CSVWriter* EndToEnd,
    double& total_compute,
    double& total_exposed,
    double& pre_bubble_time,
    double& DP_comm,
    double& DP_EP_comm,
    double& Expose_TP_comm,
    double& Expose_EP_comm,
    bool seprate_log) {
  LayerData layerData;
  take_stream_stats_average();
  int TP_size = workload->model_parallel_npu_group;
  int PP_size = workload->pipeline_model_parallelism;
  int vpp = workload->vpp;
  uint32_t pp_commsize = workload->pp_commsize;
  int DP_size = generator->all_gpus[0] / (TP_size * PP_size);
  int GA = workload->GA;
  int EP_size = workload->expert_parallel_npu_group;
  int fwd_pass_group_size ;
  int weight_grad_group_size ;
  int input_grad_group_size ;
  UserParam* param = UserParam::getInstance();
  input_grad_group_size =
        input_grad_group_type == MockNccl::GroupType::EP ? EP_size : TP_size;
    fwd_pass_group_size =
        fwd_pass_group_type == MockNccl::GroupType::EP ? EP_size : TP_size;
    weight_grad_group_size =
        weight_grad_group_type == MockNccl::GroupType::DP_EP ? DP_size / EP_size
                                                             : DP_size;
  if(param->mode == ModeType::ANALYTICAL){
    
    total_fwd_comm = compute_time(fwd_pass_comm_type,TP_size,fwd_pass_group_size,fwd_pass_comm_size,fwd_pass_group_type,generator->all_gpus[0],EP_size);
    total_weight_grad_comm = compute_time(weight_grad_comm_type,TP_size,weight_grad_group_size,weight_grad_comm_size,weight_grad_group_type,generator->all_gpus[0],EP_size);
    total_input_grad_comm = compute_time(input_grad_comm_type,TP_size,input_grad_group_size,input_grad_comm_size,input_grad_group_type,generator->all_gpus[0],EP_size);
    total_waiting_for_fwd_comm = total_fwd_comm; //tp forward
    total_waiting_for_ig_comm = total_input_grad_comm;  //tp backward
    total_waiting_for_wg_comm = total_weight_grad_comm;
    

  }
  if (id != "embedding_layer"){
      pre_bubble_time += ((total_waiting_for_fwd_comm + total_forward_pass_compute + total_weight_grad_compute + total_input_grad_compute + total_waiting_for_ig_comm) / FREQ);
    }
  if(weight_grad_group_type == MockNccl::GroupType::DP_EP){
    total_waiting_for_wg_comm *= (1-param->net_work_param.dp_overlap_ratio);
    DP_EP_comm += (total_waiting_for_wg_comm / FREQ);
  }
  else{
    total_waiting_for_wg_comm *= (1-param->net_work_param.dp_overlap_ratio);
    DP_comm += (total_waiting_for_wg_comm / FREQ);
  }
  if(fwd_pass_group_type == MockNccl::GroupType::EP){
    total_waiting_for_fwd_comm *= (1-param->net_work_param.ep_overlap_ratio);
    total_waiting_for_ig_comm *= (1-param->net_work_param.ep_overlap_ratio);
    Expose_EP_comm += ((total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / FREQ);
  }
  else{
    total_waiting_for_fwd_comm *= (1-param->net_work_param.tp_overlap_ratio);
    total_waiting_for_ig_comm *= (1-param->net_work_param.tp_overlap_ratio);
    Expose_TP_comm += ((total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / FREQ);
  }

  total_compute += (total_forward_pass_compute / FREQ);
  total_compute += (total_weight_grad_compute / FREQ);
  total_compute += (total_input_grad_compute / FREQ);
  total_exposed += (total_waiting_for_fwd_comm / FREQ);
  total_exposed += (total_waiting_for_wg_comm / FREQ);
  total_exposed += (total_waiting_for_ig_comm / FREQ);
  layerData.layer_name = id;
  layerData.total_forward_pass_compute = total_forward_pass_compute / FREQ;
  layerData.total_weight_grad_compute = total_weight_grad_compute / FREQ;
  layerData.total_input_grad_compute = total_input_grad_compute / FREQ;
  layerData.total_waiting_for_fwd_comm = total_waiting_for_fwd_comm / FREQ;
  layerData.total_waiting_for_wg_comm = total_waiting_for_wg_comm / FREQ;
  layerData.total_waiting_for_ig_comm = total_waiting_for_ig_comm / FREQ;
  layerData.total_fwd_comm = total_fwd_comm / FREQ;
  layerData.total_weight_grad_comm = total_weight_grad_comm / FREQ;
  layerData.total_input_grad_comm = total_input_grad_comm / FREQ;
  int i = 0;
  for (auto& qd : queuing_delay) {
    layerData.avg_queuing_delay.push_back(std::make_pair(i, qd / FREQ));
  }
  i = 1;
  for (auto& ml : net_message_latency) {
    layerData.avg_network_message_dealy.push_back(std::make_pair(i, ml / FREQ));
   }
  #ifdef NS3_MPI
  if (seprate_log)
  #else
  if (seprate_log) 
  #endif
  {
    std::string data;
    std::pair<float, float> total_bw;
    std::cout << "*******************" << std::endl;
    std::cout << "Layer id: " << id << std::endl;
    std::cout << "Total collectives issued for this layer: " << collective_counter << std::endl;
    std::cout << "*************************  Workload stats  ************************* " << id << std::endl;

    if (stat_row == 0 && layer_num == 0) {
        data = "layer_name," + run_name + ",fwd compute,wg compute,ig compute,fwd exposed comm,wg exposed comm,ig exposed comm,fwd total comm,algbw,busbw,wg total comm,algbw,busbw,ig total comm,algbw,busbw";
        EndToEnd->write_line(data);
    }
    data = "";
    if (stat_row == 0) {
        data += id;
    }
    data = data + "," + run_name;

    auto format_value = [](double value) {
        std::ostringstream stream;
       if (std::isfinite(value)) {
           stream << std::fixed << std::setprecision(0) << value;
       } else {
           stream << "NaN or Inf";
       }
        return stream.str();
    };
    auto format_value_bs = [](double value) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << value;
        return stream.str();
    };

    std::cout << "id: " << id << " ,Total cycles spent on fwd pass compute: "
              << format_value(total_forward_pass_compute / FREQ ) << std::endl;
    data = data + "," + format_value(total_forward_pass_compute / FREQ );

    std::cout << "id: " << id << " ,Total cycles spent on weight grad compute: "
              << format_value(total_weight_grad_compute / FREQ ) << std::endl;
    data = data + "," + format_value(total_weight_grad_compute / FREQ );

    std::cout << "id: " << id << " ,Total cycles spent on input grad compute: "
              << format_value(total_input_grad_compute / FREQ ) << std::endl;
    data = data + "," + format_value(total_input_grad_compute / FREQ );

    std::cout << "id: " << id
              << " ,Total cycles spent idle waiting for fwd finish: "
              << format_value(total_waiting_for_fwd_comm / FREQ ) << std::endl;
    data = data + "," + format_value(total_waiting_for_fwd_comm / FREQ );

    std::cout << "id: " << id
              << " ,Total cycles spent idle waiting for weight grad finish: "
              << format_value(total_waiting_for_wg_comm / FREQ ) << std::endl;
    data = data + "," + format_value(total_waiting_for_wg_comm / FREQ );

    std::cout << "id: " << id
              << " ,Total cycles spent idle waiting for input grad finish: "
              << format_value(total_waiting_for_ig_comm / FREQ ) << std::endl;
    data = data + "," + format_value(total_waiting_for_ig_comm / FREQ );

    std::cout << "id: " << id
              << " ,Total cycles spent on fwd pass comm: " << format_value(total_fwd_comm / FREQ ) << std::endl;
    total_bw = compute_busbw(fwd_pass_comm_type, fwd_pass_group_size, fwd_pass_comm_size, total_fwd_comm);
    data = data + "," + format_value(total_fwd_comm / FREQ );
    data = data + "," + format_value_bs(total_bw.first);
    data = data + "," + format_value_bs(total_bw.second);

    std::cout << "id: " << id << " ,Total cycles spent on weight grad comm: "
              << format_value(total_weight_grad_comm / FREQ ) << std::endl;
    total_bw = compute_busbw(weight_grad_comm_type, weight_grad_group_size, weight_grad_comm_size, total_weight_grad_comm);
    data = data + "," + format_value(total_weight_grad_comm / FREQ );
    data = data + "," + format_value_bs(total_bw.first);
    data = data + "," + format_value_bs(total_bw.second);

    std::cout << "id: " << id << " ,Total cycles spent on input grad comm: "
              << format_value(total_input_grad_comm / FREQ ) << std::endl;
    total_bw = compute_busbw(input_grad_comm_type, input_grad_group_size, input_grad_comm_size, total_input_grad_comm);
    data = data + "," + format_value(total_input_grad_comm / FREQ );
    data = data + "," + format_value_bs(total_bw.first);
    data = data + "," + format_value_bs(total_bw.second);

    // data = data + "," + format_value(((double)Sys::boostedTick()) / FREQ );
    EndToEnd->write_line(data);

    if (layer_num == workload->SIZE - 1) {
        if (param->mode != ModeType::ANALYTICAL) {
            total_exposed = (((double)Sys::boostedTick()) / FREQ ) - total_compute;
        }
        //pp commtime
        Tick Expose_PP_time = (2 * vpp * GA * (pp_commsize * GBps / (param->net_work_param.pp_overlap_ratio) * 1e9) / FREQ );
        Expose_PP_time *= (1-param->net_work_param.pp_overlap_ratio) ;
        //pp bubble time
        pre_bubble_time *= static_cast<double>(PP_size - 1) / (GA * vpp);
        //total time
        double total_time = total_compute + total_exposed + pre_bubble_time + Expose_PP_time;
        auto format_percentage = [&](double value) {
        double percentage = (value / total_time) * 100;
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << percentage;
        return stream.str() + "%";
        };
      std::string file_name = param->res;
      size_t last_slash_pos = param->res.find_last_of('/');
      std::string result;
      if (last_slash_pos != std::string::npos) {
          file_name = param->res.substr(last_slash_pos + 1); // 取 '/' 后面的部分
      }
      std::string keys = "File name, Expose DP comm, Expose DP_EP comm, Expose TP comm, Expose_EP_comm, Expose_PP_comm, bubble time, total comp, total exposed comm, Total time";
      std::string values = file_name + ", " +
                          format_value(DP_comm) + " (" + format_percentage(DP_comm) + "), " +
                          format_value(DP_EP_comm) + " (" + format_percentage(DP_EP_comm) + "), " +
                          format_value(Expose_TP_comm) + " (" + format_percentage(Expose_TP_comm) + "), " +
                          format_value(Expose_EP_comm) + " (" + format_percentage(Expose_EP_comm) + "), " +
                          format_value(Expose_PP_time) + " (" + format_percentage(Expose_PP_time) + "), " +
                          format_value(pre_bubble_time) + " (" + format_percentage(pre_bubble_time) + "), " +
                          format_value(total_compute) + " (" + format_percentage(total_compute) + "), " +
                          format_value(total_exposed) + " (" + format_percentage(total_exposed) + "), " +
                          format_value(total_time);

      data = keys + "\n" + values;
      EndToEnd->write_res(data);
    if(param->net_work_param.visual){
      std::string chart_path = EndToEnd->path;
      std::ofstream htmlFile(chart_path + "chart.html");
      std::string file_name = getFileName(chart_path); 
      htmlFile << "<!DOCTYPE html>\n";
      htmlFile << "<html>\n<head>\n";
      htmlFile << "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n";
      htmlFile << "<style>\n";
      htmlFile << "body { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh; margin: 0; padding-top: 10%; }\n";
      htmlFile << "canvas { width: 50%; max-width: 400px; height: auto; }\n"; 
      htmlFile << "h2 { margin: 5px 0; }\n"; 
      htmlFile << "</style>\n";
      htmlFile << "</head>\n<body>\n";
      htmlFile << "<canvas id=\"myPieChart\"></canvas>\n";
      htmlFile << "<h2>Total Time: " << to_string(total_time) << " ns</h2>\n"; 
      htmlFile << "<h2>model: " << file_name << " </h2>\n"; 
      htmlFile << "<script>\n";
      htmlFile << "var ctx = document.getElementById('myPieChart').getContext('2d');\n";
      htmlFile << "var myPieChart = new Chart(ctx, {\n";
      htmlFile << "    type: 'pie',\n";
      htmlFile << "    data: {\n";
      htmlFile << "        labels: ['Expose DP comm', 'Expose DP_EP comm','Expose TP comm', 'Expose_EP_comm','Total compute', 'PP Bubble time', 'Expose PP comm'],\n";
      htmlFile << "        datasets: [{\n";
      htmlFile << "            data: [" 
              << DP_comm << ", " 
              << DP_EP_comm << ", "
              << Expose_TP_comm << ", " 
              << Expose_EP_comm << ", " 
              << total_compute << ", " 
              << pre_bubble_time << ", " 
              << Expose_PP_time << "],\n";
      htmlFile << "            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40','#FF5733'],\n";
      htmlFile << "        }]\n";
      htmlFile << "    },\n";
      htmlFile << "    options: {\n";
      htmlFile << "        responsive: true,\n";
      htmlFile << "        maintainAspectRatio: true,\n";
      htmlFile << "        plugins: {\n";
      htmlFile << "            tooltip: {\n";
      htmlFile << "                callbacks: {\n";
      htmlFile << "                    label: function(context) {\n";
      htmlFile << "                        var label = context.label || '';\n";
      htmlFile << "                        if (label) {\n";
      htmlFile << "                            label += ': ';\n";
      htmlFile << "                        }\n";
      htmlFile << "                        if (context.parsed !== null) {\n";
      htmlFile << "                            label += context.parsed + ' ns';\n";
      htmlFile << "                        }\n";
      htmlFile << "                        return label;\n";
      htmlFile << "                    }\n";
      htmlFile << "                }\n";
      htmlFile << "            }\n";
      htmlFile << "        }\n";
      htmlFile << "    }\n";
      htmlFile << "});\n";
      htmlFile << "</script>\n";
      htmlFile << "</body>\n</html>";

      htmlFile.close();
      std::cout << "HTML file created" << std::endl;
    }

      
    }
  } 

  return layerData;
}
static std::pair<int, int> binarySearch(const std::vector<long>& arr, long target) {
    int low = 0;
    int high = arr.size() - 1;
    int leftIndex = -1, rightIndex = -1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] < target) {
            low = mid + 1;
            leftIndex = mid; 
        } else if (arr[mid] > target) {
            high = mid - 1;
            rightIndex = mid; 
        } else {
            leftIndex = mid; 
            rightIndex = mid;
            break;
        }
    }
    return std::make_pair(leftIndex, rightIndex);
}

char* comtype_to_coll(ComType comtype) {
    switch (comtype) {
        case ComType::None:
            return "none";
        case ComType::Reduce_Scatter:
            return "reducescatter";
        case ComType::All_Gather:
            return "allgather";
        case ComType::All_Reduce:
            return "allreduce";
        case ComType::All_to_All:
            return "alltoall";
        case ComType::All_Reduce_All_to_All:
            return "all_reduce_all_to_all";
        case ComType::All_Reduce_NVLS:
            return "all_reduce_nvls";
        default:
            return "unknown";
    }
}
float Layer::cal_ratio(
    uint64_t data_size,
    int nranks,
    int tp_size,
    uint32_t gpus_per_server,
    MockNccl::GroupType group_type,
    char* coll_type,
    bool is_nvlink){
    UserParam* param = UserParam::getInstance();
    auto nic_ratio_data = generator->nic_ratio_data;
    auto nvlink_ratio_data = generator->nvlink_ratio_data;
    auto ata_ratio_data = generator->ata_ratio_data;
    if ((strcmp(coll_type, "allgather") == 0 || strcmp(coll_type, "reducescatter") == 0 ) && group_type == MockNccl::GroupType::TP){
        auto data = is_nvlink ? nvlink_ratio_data : nic_ratio_data;
        int _temp_nnode = (tp_size < gpus_per_server) ? 1 : tp_size / gpus_per_server ;
        return getValue(data_size, _temp_nnode, data);
    } else if (strcmp(coll_type, "alltoall") == 0 && group_type == MockNccl::GroupType::EP){
        auto data = ata_ratio_data;
        if(tp_size * nranks <= gpus_per_server){
            return getValue(data_size, 1, data);
        }else if(tp_size >= gpus_per_server){    //multi
            return getValue(data_size, 9, data);
        } else {
            int _temp_nnode = (tp_size * nranks) / gpus_per_server;
            return getValue(data_size, _temp_nnode, data);
        }
    } else if (strcmp(coll_type, "alltoall") == 0 && group_type == MockNccl::GroupType::TP){
        auto data = ata_ratio_data;
        if (tp_size <= gpus_per_server){
            return getValue(data_size, 1, data);
        } else {
            int _temp_nnode = tp_size / gpus_per_server;
            return getValue(data_size, _temp_nnode, data);
        }
    }
    else if(group_type == MockNccl::GroupType::DP || group_type == MockNccl::GroupType::DP_EP){
        return 1; 
    }else{
        return 1;
    }
}
Tick Layer::compute_time(
    ComType comtype,
    int tp_size,
    int nranks,
    uint64_t data_size,
    MockNccl::GroupType group_type,
    int all_gpus,
    int ep_size) {
  UserParam* param = UserParam::getInstance();
  Tick comp_time = 0;
  if (comtype == ComType::None) {
    return 0;
  }


    int n_ranks;
    int nnics;
    uint32_t  gpus_per_server = param->net_work_param.gpus_per_server;
    GPUType gpu_type = param->net_work_param.gpu_type;
    float nvlink_bw = param->net_work_param.nvlink_bw;
    float bw_per_nic = param->net_work_param.bw_per_nic;
    uint32_t nics_per_server = param->net_work_param.nics_per_server;
    char* nic_type =  param->net_work_param.nic_type;
    char* coll_type = comtype_to_coll(comtype);
    float bw_ratio = 1.0;
    BusBwResult result;

    if (1 < data_size && data_size < 1048576){
      if(nranks == 2) comp_time = 10000;
      if(nranks == 4) comp_time = 12000;
      if(nranks == 8) comp_time = 15000;
      if(nranks == 16) comp_time = 66000;
      if(nranks == 32) comp_time = 135000;
      if(nranks == 64) comp_time = 200000;
      if(nranks == 128) comp_time = 320000;
      return comp_time;
    }
  if (group_type == MockNccl::GroupType::TP ){
      //TP_comm_inside
      if(tp_size <= gpus_per_server){
      result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,nics_per_server,1,coll_type,tp_size,nic_type);
      }else{
        int _node_count = tp_size / gpus_per_server;
        result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,nics_per_server,_node_count,coll_type,gpus_per_server,nic_type);
      }
    }else if (group_type == MockNccl::GroupType::EP && nranks > 1)
    {
     if(tp_size * nranks <= gpus_per_server){
      uint32_t _temp_gpus_per_server = gpus_per_server / tp_size;
      result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,nics_per_server,1,coll_type,_temp_gpus_per_server,nic_type);

     }else{
      int _node_count = (tp_size * nranks) / gpus_per_server;
      uint32_t _temp_gpus_per_server = (gpus_per_server / tp_size > 1) ? (gpus_per_server / tp_size) : 1;
      float _temp_nics_per_server = (tp_size > gpus_per_server) ? (nics_per_server / gpus_per_server) : (nics_per_server / tp_size);
      result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,_temp_nics_per_server,_node_count,coll_type,_temp_gpus_per_server,nic_type);
     }
    }else if(group_type == MockNccl::GroupType::DP && nranks > 1){
      if(tp_size <= gpus_per_server){
        uint32_t _temp_gpus_per_server = gpus_per_server / tp_size;
        float _temp_nics_per_server = nics_per_server / tp_size;
        result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,_temp_nics_per_server,nranks,coll_type,_temp_gpus_per_server,nic_type);
      }else{
        float _temp_nics_per_server = nics_per_server / gpus_per_server;
        result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,_temp_nics_per_server,nranks,coll_type,1,nic_type);
      }
    }else if(group_type == MockNccl::GroupType::DP_EP && nranks > 1){
      if(tp_size * ep_size <= gpus_per_server){
        float _temp_nics_per_server = nics_per_server / (tp_size * ep_size);
        uint32_t _temp_gpus_per_server = gpus_per_server / (tp_size * ep_size);
        result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,_temp_nics_per_server,nranks,coll_type,_temp_gpus_per_server,nic_type);
       
      }else{
        float _temp_nics_per_server = nics_per_server / gpus_per_server;
        result = cal_busbw(gpu_type,nvlink_bw,bw_per_nic,_temp_nics_per_server,nranks,coll_type,1,nic_type);
      }
    }else{
      
      comp_time = 0;
      return comp_time;
    }
    
    bw_ratio = cal_ratio(data_size,nranks,tp_size,gpus_per_server,group_type,coll_type,result.is_nvlink);
    cout<<"Communication Type: "<<coll_type<<"Communication Group: "<<group_type<<"Group Size: "<< nranks<<"Data Size: "<<data_size<<"Ratio: "<<bw_ratio<<"Bottleneck is nvlink: "<<result.is_nvlink<<endl;
    if(comtype == ComType::All_Reduce){
      comp_time = data_size * GBps / (bw_ratio * result.busbw) * 1e9 * 2 * 
            (nranks - 1) / (nranks / 1.0);
            
    } else {
      comp_time = data_size * GBps / (bw_ratio * result.busbw) * 1e9  * 
            (nranks - 1) / (nranks / 1.0);
             
    }
    
  return comp_time;
}

std::pair<float,float> Layer::compute_busbw(ComType comtype, int nranks, uint64_t data_size,Tick total_comm){
  float algbw = data_size / (total_comm / FREQ) * 1000000 * GBps;
  float busbw = 0.0;
  if (comtype == ComType::All_Reduce) {
    busbw = algbw * 2 * (nranks - 1) / (nranks / 1.0);
  } else if (
      comtype == ComType::All_Gather || comtype == ComType::Reduce_Scatter ||
      comtype == ComType::All_to_All) {
    busbw = algbw * (nranks - 1) / (nranks / 1.0);
  } else {
    busbw = 0.0;
  }

  return std::make_pair(algbw,busbw);
}
void Layer::issue_forward_pass_comm(
    SchedulingPolicy pref_scheduling,
    CollectiveBarrier barrier) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  #ifdef ANALYTI
    fwd_barrier = barrier;
    if (generator->id == 0){
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "forward pass for layer %s is analytical ",
          id.c_str());
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "forward pass for layer-id %d is analytical ",
          layer_num);
    }
    if (barrier == CollectiveBarrier::Blocking) {
      workload->call(EventType::General, NULL);
    }
    return;
  #endif
  DataSet* fp = NULL;
  fwd_barrier = barrier;
  collective_counter++;
  if (fwd_pass_comm_type == ComType::All_Reduce) {
    #ifdef PHY_MTP
    fp = generator->generate_all_reduce(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Fwd_Comm_Finished,
        this);
    #else
    fp = generator->generate_all_reduce(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!fp->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no forward pass collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete fp;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-reduce forward pass collective issued for layer: "
                << id << ",";
      print_involved_dimensions(fwd_pass_comm_involved_dimensions);
    }
  } else if (fwd_pass_comm_type == ComType::All_to_All) {
    #ifdef PHY_MTP
    fp = generator->generate_all_to_all(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Fwd_Comm_Finished,
        this);
    #else
    fp = generator->generate_all_to_all(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!fp->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no forward pass collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete fp;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-to-all forward pass collective issued for layer: "
                << id << ",";
      print_involved_dimensions(fwd_pass_comm_involved_dimensions);
    }
  } else if (fwd_pass_comm_type == ComType::All_Gather) {
    #ifdef PHY_MTP
    fp = generator->generate_all_gather(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Fwd_Comm_Finished,
        this);
    #else
    fp = generator->generate_all_gather(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!fp->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no forward pass collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete fp;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-gather forward pass collective issued for layer: "
                << id << ",";
      print_involved_dimensions(fwd_pass_comm_involved_dimensions);
    }
  } else if (fwd_pass_comm_type == ComType::Reduce_Scatter) {
    #ifdef PHY_MTP
    fp = generator->generate_reduce_scatter(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Fwd_Comm_Finished,
        this);
    #else
    fp = generator->generate_reduce_scatter(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!fp->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no forward pass collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete fp;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout
          << "info: reduce-scatter forward pass collective issued for layer: "
          << id << ",";
      print_involved_dimensions(fwd_pass_comm_involved_dimensions);
    }
  } else if (fwd_pass_comm_type == ComType::None) {
    collective_counter--;
    if (generator->id == 0) {
      std::cout << "info: no forward pass collective for layer: " << id
                << std::endl;
    }
    if (barrier == CollectiveBarrier::Blocking) {
      workload->call(EventType::General, NULL);
    }
    return;
  } else {
    Sys::sys_panic("no known collective operation! ");
  }
  #ifndef PHY_MTP
  fwd_pass_datasets[fp->my_id] = fp;
  fp->set_notifier(this, EventType::Fwd_Comm_Finished);
  #endif
  NcclLog->writeLog(NcclLogLevel::DEBUG,"Fwd_Comm_Finished set_notifier success");
}
void Layer::issue_input_grad_comm(
    SchedulingPolicy pref_scheduling,
    CollectiveBarrier barrier) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();  
  #ifdef ANALYTI
  ig_barrier = barrier;
  if (generator->id == 0){
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "input grad collective for layer %s is analytical ",
        id.c_str());
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "input grad collective for layer-id %d is analytical ",
        layer_num);
  }
    
  if (barrier == CollectiveBarrier::Blocking) {
    workload->call(EventType::General, NULL);
  }
  return;
  #endif
  DataSet* ig = NULL;
  ig_barrier = barrier;
  collective_counter++;
  if (input_grad_comm_type == ComType::All_Reduce) {
    #ifdef PHY_MTP
        ig = generator->generate_all_reduce(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Input_Grad_Comm_Finished,
        this);
    #else
    ig = generator->generate_all_reduce(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!ig->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no input grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete ig;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-reduce input grad collective issued for layer: "
                << id << ",";
      print_involved_dimensions(input_grad_comm_involved_dimensions);
    }
  } else if (input_grad_comm_type == ComType::All_to_All) {
    #ifdef PHY_MTP
    ig = generator->generate_all_to_all(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Input_Grad_Comm_Finished,
        this);
    #else
    ig = generator->generate_all_to_all(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!ig->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no input grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete ig;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-to-all input grad collective issued for layer: "
                << id << ",";
      print_involved_dimensions(input_grad_comm_involved_dimensions);
    }
  } else if (input_grad_comm_type == ComType::All_Gather) {
    #ifdef PHY_MTP
    ig = generator->generate_all_gather(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Input_Grad_Comm_Finished,
        this);
    #else
    ig = generator->generate_all_gather(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!ig->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no input grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete ig;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-gather input grad collective issued for layer: "
                << id << ",";
      print_involved_dimensions(input_grad_comm_involved_dimensions);
    }
  } else if (input_grad_comm_type == ComType::Reduce_Scatter) {
    #ifdef PHY_MTP
    ig = generator->generate_reduce_scatter(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Input_Grad_Comm_Finished,
        this);
    #else
    ig = generator->generate_reduce_scatter(
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!ig->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no input grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete ig;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout
          << "info: reduce-scatter input grad collective issued for layer: "
          << id << ",";
      print_involved_dimensions(input_grad_comm_involved_dimensions);
    }
  } else if (input_grad_comm_type == ComType::None) {
    collective_counter--;
    if (generator->id == 0) {
      std::cout << "info: no input grad collective for layer: " << id
                << std::endl;
    }
    if (barrier == CollectiveBarrier::Blocking) {
      workload->call(EventType::General, NULL);
    }
    return;
  } else {
    std::cout << "no known collective operation! for layer: " << id
              << std::endl;
    Sys::sys_panic("no known collective operation! ");
  }
  #ifndef PHY_MTP
  input_grad_datasets[ig->my_id] = ig;
  ig->set_notifier(this, EventType::Input_Grad_Comm_Finished);
  #endif
}
void Layer::issue_weight_grad_comm(
    SchedulingPolicy pref_scheduling,
    CollectiveBarrier barrier) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  #ifdef ANALYTI
  wg_barrier = barrier;
  if (generator->id == 0){
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "weight grad collective for layer %s is analytical ",
        id.c_str());
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "weight grad collective for layer-id %d is analytical ",
        layer_num);
  }
    
  if (barrier == CollectiveBarrier::Blocking) {
    workload->call(EventType::General, NULL);
  }
  return;
  #endif
  DataSet* wg = NULL;
  wg_barrier = barrier;
  collective_counter++;
  if (weight_grad_comm_type == ComType::All_Reduce) {
    #ifdef PHY_MTP
    wg = generator->generate_all_reduce(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Wight_Grad_Comm_Finished,
        this);
    #else
    wg = generator->generate_all_reduce(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!wg->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no weight grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete wg;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-reduce weight grad collective issued for layer: "
                << id << " with size: " << weight_grad_comm_size << ",";
      print_involved_dimensions(weight_grad_comm_involved_dimensions);
    }
  } else if (weight_grad_comm_type == ComType::All_to_All) {
    #ifdef PHY_MTP
    wg = generator->generate_all_to_all(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Wight_Grad_Comm_Finished,
        this);
    #else
    wg = generator->generate_all_to_all(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!wg->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no weight grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete wg;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-to-all weight grad collective issued for layer: "
                << id << " with size: " << weight_grad_comm_size << ",";
      print_involved_dimensions(weight_grad_comm_involved_dimensions);
    }
  } else if (weight_grad_comm_type == ComType::All_Gather) {
    if(generator->id == 0) std::cout << "Layer issue wg all gather at tick: " << Sys::boostedTick() << std::endl;
    #ifdef PHY_MTP
    wg = generator->generate_all_gather(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Wight_Grad_Comm_Finished,
        this);
    #else
    wg = generator->generate_all_gather(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!wg->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no weight grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete wg;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout << "info: all-gather weight grad collective issued for layer: "
                << id << ",";
      print_involved_dimensions(weight_grad_comm_involved_dimensions);
    }
  } else if (weight_grad_comm_type == ComType::Reduce_Scatter) {
    #ifdef PHY_MTP
    wg = generator->generate_reduce_scatter(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Wight_Grad_Comm_Finished,
        this);
    #else
    wg = generator->generate_reduce_scatter(
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
    #endif
    if (!wg->active) {
      if (generator->id == 0) {
        std::cout
            << "info: all dims disabled, no weight grad collective for layer: "
            << id << std::endl;
      }
      collective_counter--;
      delete wg;
      if (barrier == CollectiveBarrier::Blocking) {
        workload->call(EventType::General, NULL);
      }
      return;
    }
    if (generator->id == 0) {
      std::cout
          << "info: reduce-scatter weight grad collective issued for layer: "
          << id << ",";
      print_involved_dimensions(weight_grad_comm_involved_dimensions);
    }
  } else if (weight_grad_comm_type == ComType::None) {
    collective_counter--;
    if (generator->id == 0) {
      std::cout << "info: no weight grad collective for layer: " << id
                << std::endl;
    }
    if (barrier == CollectiveBarrier::Blocking) {
      workload->call(EventType::General, NULL);
    }
    return;
  } else {
    Sys::sys_panic("no known collective operation! ");
  }
  #ifndef PHY_MTP
  weight_grad_datasets[wg->my_id] = wg;
  wg->set_notifier(this, EventType::Wight_Grad_Comm_Finished);
  #endif
}
} // namespace AstraSim
