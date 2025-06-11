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

#ifndef __ASTRAPARAMPARSE_HH__
#define __ASTRAPARAMPARSE_HH__

#include <iostream>
#include<sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <cstdarg>
#include <vector>
#include <cstdint>
#include "Common.hh"
#define BUSBW_PATH ""
using namespace std;

enum class ModeType { NONE, ASTRA_SIM, MOCKNCCL, ANALYTICAL };

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <regex>


struct NetWorkParam{
  uint32_t node_num;
  uint32_t switch_num;
  uint32_t link_num;
  uint32_t trace_num;
  uint32_t nvswitch_num;
  uint32_t gpus_per_server;
  uint32_t nics_per_server;
  float nvlink_bw = -1.0;
  float bw_per_nic = -1.0;
  char* nic_type = "cx7";
  bool visual = 0;
  float dp_overlap_ratio = 0;
  float tp_overlap_ratio = 0;
  float ep_overlap_ratio = 0;
  float pp_overlap_ratio = 1;
  GPUType gpu_type;
  std::vector<int>NVswitchs;
  std::vector<std::vector<int>>all_gpus;
};

class UserParam {
private:
  static UserParam* instance;
  static std::mutex mtx;

  UserParam() {
    thread = 1;
    gpus = {};
    workload = {};
    comm_scale = 1;
    mode = ModeType::MOCKNCCL;
  }

public:
  int thread;
  std::vector<int> gpus;
  std::string workload;
  std::string res = "None";
  std::string res_folder = "None";
  int comm_scale;
  ModeType mode;
  NetWorkParam net_work_param;



  static UserParam* getInstance(){
    std::lock_guard<std::mutex> lock(mtx);
    if(instance == nullptr){
      instance = new UserParam();
    }
    return instance;
  }

int parse(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "-w,     --workload          Workloads, default none" << std::endl;
            std::cout << "-g,     --gpus              Number of GPUs, default 1" << std::endl;
            std::cout << "-g_p_s, --gpus-per-server   GPUs per server" << std::endl;
            std::cout << "-r,     --result            Output results path" << std::endl;
            std::cout << "-nv, --nvlink     Nvlink" << std::endl;
            std::cout << "-nic, --nic_busbw     NIC busbw" << std::endl;
            std::cout << "-n_p_s, --bus-bandwidth     Bus bandwidth file" << std::endl;
            std::cout << "-nic_t, --nic_type     NIC type(cx7,bf3),choose when disable nic " << std::endl;
            std::cout << "-g_type, --gpu_type     GPU type(A100,H100),choose when disable nvlink " << std::endl;
            std::cout << "-v, --visual    Enable visual output" << std::endl;
            std::cout << "-dp_o, --dp_overlap    dp overlap ratio(Default 0)" << std::endl;
            std::cout << "-ep_o, --ep_overlap    ep overlap ratio(Default 0)" << std::endl;
            std::cout << "-tp_o, --tp_overlap    tp overlap ratio(Default 0)" << std::endl;
            std::cout << "-pp_o, --pp_overlap    pp overlap ratio(Default 1)" << std::endl;
            return 1;
        } else if (arg == "-w" || arg == "--workload") {
            if (++i < argc) this->workload = argv[i];
        } else if (arg == "-g" || arg == "--gpus") {
            if (++i < argc) this->gpus.push_back(std::stoi(argv[i]));
        } else if (arg == "-r" || arg == "--result") {
            if (++i < argc) this->res = argv[i];
        } else if (arg == "-r_f" || arg == "--result_folder") {
            if (++i < argc) this->res_folder = argv[i];
        } else if (arg == "-g_p_s" || arg == "--gpus-per-server") {
            if (++i < argc) this->net_work_param.gpus_per_server = std::stoi(argv[i]);
        } else if (arg == "-nv" || arg == "--nvlink") {
            if (++i < argc) this->net_work_param.nvlink_bw = std::stof(argv[i]);
        } else if (arg == "-nic"|| arg == "--nic_busbw") {
            if (++i < argc) this->net_work_param.bw_per_nic = std::stof(argv[i]);
        } else if (arg == "-n_p_s" || arg == "--nic_per_server") {
            if (++i < argc) this->net_work_param.nics_per_server = std::stoi(argv[i]);
        } else if (arg == "-nic_t" || arg == "--nic_type") {
            if (++i < argc) this->net_work_param.nic_type = argv[i];
        } else if (arg == "-g_type" || arg == "--gpu_type") {
            if (++i < argc) {
                std::string gpu_type = argv[i];
                if (gpu_type == "A100" || gpu_type == "a100") this->net_work_param.gpu_type = GPUType::A100;
                else if (gpu_type == "A800" || gpu_type == "a800" ) this->net_work_param.gpu_type = GPUType::A800;
                else if (gpu_type == "H100" || gpu_type == "h100") this->net_work_param.gpu_type = GPUType::H100;
                else if (gpu_type == "H800" || gpu_type == "h800") this->net_work_param.gpu_type = GPUType::H800;
                else if (gpu_type == "H20" || gpu_type == "h20") this->net_work_param.gpu_type = GPUType::H20;
                else this->net_work_param.gpu_type = GPUType::NONE;
            }
        }else if (arg == "-v" || arg == "--visual") {
            if (++i < argc) this->net_work_param.visual = std::stoi(argv[i]);
        }else if (arg == "--dp_overlap" || arg == "-dp_o") {
            if (++i < argc) this->net_work_param.dp_overlap_ratio = std::stof(argv[i]);
        }else if (arg == "--tp_overlap" || arg == "-tp_o") {
            if (++i < argc) this->net_work_param.tp_overlap_ratio = std::stof(argv[i]);
        }else if (arg == "--ep_overlap" || arg == "-ep_o") {
            if (++i < argc) this->net_work_param.ep_overlap_ratio = std::stof(argv[i]);
        }else if (arg == "--pp_overlap" || arg == "-pp_o") {
            if (++i < argc) this->net_work_param.pp_overlap_ratio = std::stof(argv[i]);
        }
        else {
            return 1; 
        }
    }

    if (!this->gpus.empty()) {
        this->net_work_param.nvswitch_num = this->gpus[0] / this->net_work_param.gpus_per_server;
        this->net_work_param.switch_num = 120 + this->net_work_param.gpus_per_server;
        this->net_work_param.node_num = this->net_work_param.nvswitch_num + this->net_work_param.switch_num + this->gpus[0];
    }

    if (this->res == "None" ){
        std::string full_path = this->workload;
        std::string model_info = full_path;
        size_t last_slash_pos = full_path.find_last_of('/');
        if (last_slash_pos != std::string::npos) {
            model_info = full_path.substr(last_slash_pos + 1); 
        }
        std::string model_name; 
        int world_size = 0, tp = 0, pp = 0, ep = 0, gbs = 0, mbs = 0, seq = 0;

        
        size_t world_size_pos = model_info.find("world_size");
        if (world_size_pos != std::string::npos) {
            model_name = model_info.substr(0, world_size_pos - 1); 
        }

        
        std::regex param_regex(R"((world_size|tp|pp|ep|gbs|mbs|seq)(\d+))");
        std::smatch matches;

        std::string params = model_info; 
        while (std::regex_search(params, matches, param_regex)) {
            std::string param_name = matches[1].str();
            int param_value = std::stoi(matches[2].str());

            if (param_name == "world_size") {
                world_size = param_value;
            } else if (param_name == "tp") {
                tp = param_value;
            } else if (param_name == "pp") {
                pp = param_value;
            } else if (param_name == "ep") {
                ep = param_value;
            } else if (param_name == "gbs") {
                gbs = param_value;
            } else if (param_name == "mbs") {
                mbs = param_value;
            } else if (param_name == "seq") {
                seq = param_value;
            }

            
            params = matches.suffix().str();
        }

        
        int dp = world_size / (tp * pp); 
        double ga = static_cast<double>(gbs) / (dp * mbs); 

        std::ostringstream result;
        result << model_name << '-' 
            << "tp" << tp << '-'
            << "pp" << pp << '-'
            << "dp" << dp << '-'
            << "ga" << static_cast<int>(ga) << '-'
            << "ep" << ep << '-'
            << "NVL" << this->net_work_param.gpus_per_server << '-'
            << std::fixed << std::setprecision(1) << (this->net_work_param.bw_per_nic * 8) << "G" << '-'
            << "DP" << this->net_work_param.dp_overlap_ratio << '-' ;
        
        this->res = result.str();
        
        
    }
    if (this->res_folder != "None"){
        if (this->res_folder.back() != '/'){
            this->res = this->res_folder + '/' + this->res;
        }
        else{
            this->res = this->res_folder + this->res;
        }

    }
    return 0;
}
  ~UserParam(){}
};

#endif // __ASTRAPARAMPARSE_HH__