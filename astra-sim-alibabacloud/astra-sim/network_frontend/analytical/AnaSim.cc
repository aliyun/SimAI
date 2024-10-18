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

queue<struct CallTask> AnaSim::call_list = {};
uint64_t AnaSim::tick = 0;
void AnaSim::Run() {
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
        // std::cout << "after pop call_list: " << call_list.size() << std::endl;
        calltask.fun_ptr(calltask.fun_arg);
        
        // sleep(calltask.delay);
    }
}

void AnaSim::Schedule(
  
    uint64_t delay,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    uint64_t time = tick + delay;
    CallTask calltask = CallTask(time,fun_ptr,fun_arg);
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