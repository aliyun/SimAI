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

#ifndef __PHYSIMAI_HH__
#define __PHYSIMAI_HH__

#include<iostream>
#include<queue>
#include<list>

using namespace std;

struct CallTask {
  int time;
  void (*fun_ptr)(void* fun_arg);
  void* fun_arg;
  CallTask(int _time, void (*_fun_ptr)(void* _fun_arg), void* _fun_arg)
      : time(_time), fun_ptr(_fun_ptr), fun_arg(_fun_arg) {};
  ~CallTask(){}
};

class PhyNetSim {
 private:
  static queue<struct CallTask> call_list;
  static int tick;

 public:
  static double Now();
  static void Run(void);
  static void Schedule(
      int delay,
      void (*fun_ptr)(void* fun_arg),
      void* fun_arg);
  static void Stop();
  static void Destory();
};
#endif