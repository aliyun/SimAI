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
#include"AnaSim.h"
using namespace std;

priority_queue<struct CallTask, vector<struct CallTask>, CallTaskCompare> AnaSim::call_list = {};
uint64_t AnaSim::tick = 0;
uint64_t AnaSim::seq_counter = 0;                     // [patch @sharding_simai]
void AnaSim::Run() {
    while (!call_list.empty())
    {
        CallTask calltask = call_list.top();          // [patch] 取「時間最小」的事件（原本 FIFO front）
        call_list.pop();
        if (calltask.time > tick) tick = calltask.time;  // [patch] 直接跳到事件時間，不再 tick++ 一格格空轉
        calltask.fun_ptr(calltask.fun_arg);
    }
}

void AnaSim::Schedule(
  
    uint64_t delay,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    uint64_t time = tick + delay;
    CallTask calltask = CallTask(time, seq_counter++, fun_ptr, fun_arg);   // [patch @sharding_simai]
    // std::cout << "before push all_list: " << call_list.size() << std::endl;
    call_list.push(calltask);
    // std::cout << "after push of call_list: " << call_list.size() << std::endl;
}

void AnaSim::Stop(){
    return;
}

void AnaSim::Destroy(){
    return;
}

double AnaSim::Now(){
    return tick;
}