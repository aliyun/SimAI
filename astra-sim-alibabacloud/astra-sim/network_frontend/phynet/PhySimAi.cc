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

#include <unistd.h>
#include"PhySimAi.h"
#include"astra-sim/system/MockNcclLog.h"
using namespace std;

queue<struct CallTask> PhyNetSim::call_list = {};
int PhyNetSim::tick = 0;

void PhyNetSim::Run() {
    while (!call_list.empty())
    {
        CallTask calltask = call_list.front();
        while (true) {
          if (calltask.time != tick) {
            tick++;
          } else {
            break;
          }
        }
        call_list.pop();
        MockNcclLog* NcclLog = MockNcclLog::getInstance();
        NcclLog->writeLog(
            NcclLogLevel::DEBUG, "PhyNetSim::Run calltask begin tick %d",tick);
        calltask.fun_ptr(calltask.fun_arg);
        NcclLog->writeLog(
            NcclLogLevel::DEBUG, "PhyNetSim::Run calltask end tick %d",tick);
    }
}

void PhyNetSim::Schedule(
    int delay,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    int time = tick + delay;
    CallTask calltask = CallTask(time,fun_ptr,fun_arg);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(
        NcclLogLevel::DEBUG, "PhyNetSim::Schedule calltask ");
    call_list.push(calltask);
}

void PhyNetSim::Stop(){
    return;
}

void PhyNetSim::Destory(){
    return;
}

double PhyNetSim::Now(){
    return tick;
}