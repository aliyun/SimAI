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

#ifndef __ASTRASIMNETWORK_HH__
#define __ASTRASIMNETWORK_HH__
#include <iostream>
using namespace std;
class ASTRASimNetwork: AstraSim::AstraNetworkAPI{
    public:
        ASTRASimNetwork(int rank):AstraNetworkAPI(rank){
        }
        ~ASTRASimNetwork(){}
        int sim_comm_size(AstraSim::sim_comm comm, int* size){
            return 0;
        }
        int sim_finish(){
            return 0;
        }
        double sim_time_resolution(){
            return 0;
        }
        int sim_init(AstraSim::AstraMemoryAPI* MEM){
            return 0;
        }
        AstraSim::timespec_t sim_get_time(){
            AstraSim::timespec_t timeSpec;
            timeSpec.time_val = 0.0;
            return timeSpec;
        }
        virtual void sim_schedule(
            AstraSim::timespec_t delta,
            void (*fun_ptr)(void* fun_arg),
            void* fun_arg){
                return;
            }
        virtual int sim_send(
            void* buffer,
            uint64_t count,
            int type,
            int dst,
            int tag,
            AstraSim::sim_request* request,
            void (*msg_handler)(void* fun_arg),
            void* fun_arg){
                system("./waf --run  scratch/myTCPMultiple");
                return 0;
            }
        virtual int sim_recv(
            void* buffer,
            uint64_t count,
            int type,
            int src,
            int tag,
            AstraSim::sim_request* request,
            void (*msg_handler)(void* fun_arg),
            void* fun_arg){
                return 0;
            }

};
#endif
