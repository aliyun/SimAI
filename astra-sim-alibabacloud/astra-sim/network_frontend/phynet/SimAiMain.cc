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

#include<unistd.h>
#include<string>
#include<iostream>
#include"SimAiPhyNetwork.h"
#include"PhySimAi.h"
#include"SimAiEntry.h"

#include "astra-sim/system/AstraComputeAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/BootStrapnet.hh"
#include "astra-sim/system/PhyMultiThread.hh"
#include "astra-sim/system/Common.hh"
#ifdef PHY_RDMA
#include "astra-sim/system/SimAiFlowModelRdma.hh"
#endif
#define RESULT_PATH "/etc/astra-sim/results/ncclFlowModel_"

using namespace std;

extern int local_rank;
extern AstraSim::Sys* global_sys;
extern FlowPhyRdma flow_rdma;

struct user_param {
  int thread;
  int gpus;
  string workload;
  int comm_scale;
  GPUType gpu_type;
  int nvswitch_num;
  int gpus_per_server;
  int gid_index;
  user_param() {
    thread = 1;
    gpus = 8;
    workload = "microAllReduce.txt";
    comm_scale = 1;
    gpu_type = GPUType::A100;
    nvswitch_num = 1;
    gpus_per_server = 8 ;
    gid_index = 0;
  };
  ~user_param(){};
  user_param(int _thread, int _gpus, string _workload, int _comm_scale = 1)
      : thread(_thread),
        gpus(_gpus),
        workload(_workload),
        comm_scale(_comm_scale){};
};

static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
  int opt;
  static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"workloads", required_argument, 0, 'w'},
        {"gpus", required_argument, 0, 'g'},
        {"comm_scale", required_argument, 0, 's'},
        {"gid_index", required_argument, 0, 'i'},
        {0, 0, 0, 0}};
  while ((opt = getopt(argc,argv,"ht:w:g:s:i:"))!=-1){
    switch (opt)
    {
    case 'h':
      /* code */
      std::cout<<"-w    workloads default microAllReduce.txt "<<std::endl;
      std::cout<<"-g    number of gpus,default 1"<<std::endl;
      std::cout<<"-s    comm_scale default 1"<<std::endl;
      std::cout<<"-i    rdma gid_indxe default 0" <<std::endl;
      break;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'g':
      user_param->gpus = stoi(optarg);
      if(user_param->gpus <= 8){
        user_param->gpus =8;
      }
      break;
    case 's':
      user_param->comm_scale = stof(optarg);
      break;
    case 'i':
      user_param->gid_index = stoi(optarg);
      break;
    default:
      break;
    }
  }
  return 0 ;
}

int main(int argc,char *argv[]){
  BootStrapNet(argc,argv);
  pid_t pid = getpid();
  MockNcclLog::set_log_name("SimAi_"+to_string(local_rank)+".log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG," Local rank %d PID %d ",local_rank,pid);
  struct user_param user_param;
  if(user_param_prase(argc,argv,&user_param)){
    return -1;
  }
  #ifdef PHY_RDMA
  flow_rdma = FlowPhyRdma(user_param.gid_index);
  flow_rdma.ibv_init();
  #endif
  set_simai_network_callback();
  std::vector<int> physical_dims = {user_param.gpus};
  std::vector<int>NVswitchs;  
  std::vector<int> queues_per_dim={1};
  std::map<int, int> node2nvswitch;
  for(int i = 0; i < user_param.gpus; ++ i) {
    node2nvswitch[i] = user_param.gpus + i / user_param.gpus_per_server;
  }
  for(int i = user_param.gpus; i < user_param.gpus + user_param.nvswitch_num; ++ i){
  node2nvswitch[i] = i;
  NVswitchs.push_back(i);
  } 
  physical_dims[0] += user_param.nvswitch_num;

  SimAiPhyNetWork* phy_network = new SimAiPhyNetWork(local_rank);
  global_sys = new AstraSim::Sys(
    phy_network,
    nullptr,
    local_rank,
    0,
    1,
    physical_dims,
    queues_per_dim,
    "",
    user_param.workload,
    user_param.comm_scale,
    1,
    1,
    1,
    0,
    RESULT_PATH,
    "phynet_test",
    true,
    false,
    user_param.gpu_type,
    {user_param.gpus},
    NVswitchs,
    user_param.gpus_per_server
  );
  global_sys->nvswitch_id = node2nvswitch[local_rank];
  global_sys->num_gpus = user_param.gpus;
  global_sys->workload->fire();
  PhyNetSim::Run();
  PhyNetSim::Stop();
  notify_all_thread_finished();
  PhyNetSim::Destory();
  MPI_Finalize();
  return 0;
};