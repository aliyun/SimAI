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
#include <regex>
enum class ModeType { NONE, ASTRA_SIM, MOCKNCCL, ANALYTICAL };

struct NetWorkParam{
  uint32_t node_num;
  uint32_t switch_num;
  uint32_t link_num;
  uint32_t trace_num;
  uint32_t nvswitch_num;
  uint32_t gpus_per_server;
  uint32_t nics_per_server;
  uint32_t nvlink_bw;
  uint32_t nic_bw;
  GPUType gpu_type;
  float tp_ar = -1.0f; 
  float tp_ag = -1.0f; 
  float tp_rs = -1.0f; 
  float tp_ata = -1.0f; 
  float dp_ar = -1.0f; 
  float dp_ag = -1.0f;
  float dp_rs = -1.0f;
  float dp_ata = -1.0f;
  float ep_ar = -1.0f;
  float ep_ag = -1.0f; 
  float ep_rs = -1.0f; 
  float ep_ata = -1.0f; 
  float dp_overlap_ratio = 0;
  float tp_overlap_ratio = 0;
  float ep_overlap_ratio = 0;
  float pp_overlap_ratio = 0;
  std::vector<int> NVswitchs;
  std::vector<std::vector<int>> all_gpus;
  int visual = 0;
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
    string workload;
    string res = "None";
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

    void parseYaml(NetWorkParam& params, const std::string& filename) {
        std::ifstream file(BUSBW_PATH + filename);
        if (!file) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            exit(-1);
        }
        std::string line;
        std::string currentSection;
        std::getline(file, line);
        while (std::getline(file, line)) {
            // Remove whitespace

            line.erase(0, line.find_first_not_of(' '));
            line.erase(line.find_last_not_of(' ') + 1);

            if (line.empty() || line[0] == '#') continue;

            if (line.back() == ':') {
                currentSection = line.substr(0, line.size() - 1);
            } else {
                std::istringstream ss(line);
                std::string key, valueStr;
                if (std::getline(ss, key, ':') && ss >> valueStr) {
                    key.erase(key.find_last_not_of(' ') + 1);

                    // Remove part after comma
                    auto commaPos = key.find(',');
                    if (commaPos != std::string::npos) {
                        key = key.substr(0, commaPos);
                    }

                    if (valueStr != "null") {
                        float value = std::stof(valueStr);
                        
                        if (currentSection == "TP") {
                            if (key == "allreduce") params.tp_ar = value;
                            else if (key == "allgather") params.tp_ag = value;
                            else if (key == "reducescatter") params.tp_rs = value;
                            else if (key == "alltoall") params.tp_ata = value;
                        } else if (currentSection == "DP") {
                            if (key == "allreduce") params.dp_ar = value;
                            else if (key == "allgather") params.dp_ag = value;
                            else if (key == "reducescatter") params.dp_rs = value;
                            else if (key == "alltoall") params.dp_ata = value;
                        } else if (currentSection == "EP") {
                            if (key == "allreduce") params.ep_ar = value;
                            else if (key == "allgather") params.ep_ag = value;
                            else if (key == "reducescatter") params.ep_rs = value;
                            else if (key == "alltoall") params.ep_ata = value;
                        }
                    }
                }
            }
        }
    }

    void printHelp() const {
        std::cout << " ____  _              _    ___        _                _       _   _           _ \n"
                << "/ ___|(_)_ __ ___    / \\  |_ _|      / \\   _ __   __ _| |_   _| |_(_) ___ __ _| |\n"
                << "\\___ \\| | '_ ' _ \\  / _ \\  | |_____ / _ \\ | '_ \\ / _' | | | | | __| |/ __/ _' | |\n"
                << " ___) | | | | | | |/ ___ \\ | |_____/ ___ \\| | | | (_| | | |_| | |_| | (_| (_| | |\n"
                << "|____/|_|_| |_| |_/_/   \\_\\___|   /_/   \\_\\_| |_|\\__,_|_|\\__, |\\__|_|\\___\\__,_|_|\n"
                << "                                                           |___/                   \n";
        std::cout << "-w,       --workload            Workloads, must set" << std::endl;
        std::cout << "-g,       --gpus                Number of GPUs, default 1" << std::endl;
        std::cout << "-g_p_s,   --gpus-per-server     GPUs per server" << std::endl;
        std::cout << "-r,       --result              Output results path, default: ./results/" << std::endl;
        std::cout << "-busbw,   --bus-bandwidth       Bus bandwidth file, must set" << std::endl;
        std::cout << "-v,       --visual              Enable visual output (Default disable)" << std::endl;
        std::cout << "-dp_o,    --dp-overlap-ratio    DP overlap ratio [float: 0.0-1.0] (Default: 0.0)" << std::endl;
        std::cout << "-ep_o,    --ep-overlap-ratio    EP overlap ratio [float: 0.0-1.0] (Default: 0.0)" << std::endl;
        std::cout << "-tp_o,    --tp-overlap-ratio    TP overlap ratio [float: 0.0-1.0] (Default: 0.0)" << std::endl;
        std::cout << "-pp_o,    --pp-overlap-ratio    PP overlap ratio [float: 0.0-1.0] (Default: 0.0)" << std::endl;
    }

    int printError(const std::string& arg) const {
        std::cerr << "Error: Missing value for argument '" << arg << "'." << std::endl;
        return 1;
    }

    int printUnknownOption(const std::string& arg) const {
        std::cerr << "Error: Unknown option '" << arg << "'." << std::endl;
        return 1;
    }

    int parseArg(int argc, char *argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                printHelp();
                return 1;
            } else if (arg == "-w" || arg == "--workload") {
                if (++i < argc) this->workload = argv[i];
                else return printError(arg);
            } else if (arg == "-g" || arg == "--gpus") {
                if (++i < argc) this->gpus.push_back(std::stoi(argv[i]));
                else return printError(arg);
            } else if (arg == "-r" || arg == "--result") {
                if (++i < argc) this->res = argv[i];
                else return printError(arg);
            } else if (arg == "-g_p_s" || arg == "--gpus-per-server") {
                if (++i < argc) this->net_work_param.gpus_per_server = std::stoi(argv[i]);
                else return printError(arg);
            } else if (arg == "-busbw" || arg == "--bus-bandwidth") {
                if (++i < argc) parseYaml(this->net_work_param,argv[i]);
                else return printError(arg);
            } else if (arg == "--dp-overlap-ratio" || arg == "-dp_o") {
                if (++i < argc) this->net_work_param.dp_overlap_ratio = std::stof(argv[i]);
                else return printError(arg);
            } else if (arg == "--tp-overlap-ratio" || arg == "-tp_o") {
                if (++i < argc) this->net_work_param.tp_overlap_ratio = std::stof(argv[i]);
                else return printError(arg);
            } else if (arg == "--ep-overlap-ratio" || arg == "-ep_o") {
                if (++i < argc) this->net_work_param.ep_overlap_ratio = std::stof(argv[i]);
                else return printError(arg);
            } else if (arg == "--pp-overlap-ratio" || arg == "-pp_o") {
                if (++i < argc) this->net_work_param.pp_overlap_ratio = std::stof(argv[i]);
                else return printError(arg);
            } else if (arg == "-v" || arg == "--visual") {
                if (++i < argc) this->net_work_param.visual = std::stoi(argv[i]);
                else return printError(arg);
            }
            else {
                return printUnknownOption(arg);
            }
        }

        if (!this->gpus.empty()) {
            this->net_work_param.nvswitch_num = this->gpus[0] / this->net_work_param.gpus_per_server;
            this->net_work_param.switch_num = 120 + this->net_work_param.gpus_per_server;
            this->net_work_param.node_num = this->net_work_param.nvswitch_num + this->net_work_param.switch_num + this->gpus[0];
        }
        if (this->res == "None" || this->res.back() == '/'){
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
                << "DP" << this->net_work_param.dp_overlap_ratio << '-' ;
            if(this->res.back() == '/') {
                this->res = this->res + result.str();
            }
            else{
                this->res = result.str();
            }
            
        }
        return 0;
    }
    ~UserParam(){}
};

#endif // __ASTRAPARAMPARSE_HH__
