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

#ifndef __ANASIM_HH__
#define __ANASIM_HH__
#include<iostream>
#include<queue>
#include<vector>
#include<list>
#include<cstdint>

using namespace std;

struct CallTask {
  uint64_t time;
  uint64_t seq;                          // [patch @sharding_simai] 插入序：同時間事件保留 FIFO tie-break
  void (*fun_ptr)(void* fun_arg);
  void* fun_arg;
  CallTask(uint64_t _time, uint64_t _seq, void (*_fun_ptr)(void* _fun_arg), void* _fun_arg)
      : time(_time), seq(_seq), fun_ptr(_fun_ptr), fun_arg(_fun_arg) {};
  ~CallTask(){}
};

// [patch @sharding_simai] min-heap（先 time 再 seq）。修 AnaSim::Run() 的 tick++ 空轉：
//   原本 call_list 是 FIFO、Run() 用 tick++ 一格格爬到事件時間 → 事件亂序時 tick 追不上、
//   爬到 2^64 才停 → tp=2 這種大延遲/亂序 config 卡幾小時。改成照時間排序處理即可。
struct CallTaskCompare {
  bool operator()(const CallTask& a, const CallTask& b) const {
    return a.time != b.time ? a.time > b.time : a.seq > b.seq;
  }
};

class AnaSim {
 private:
  static priority_queue<struct CallTask, vector<struct CallTask>, CallTaskCompare> call_list;
  static uint64_t tick;
  static uint64_t seq_counter;           // [patch @sharding_simai] 遞增插入序

 public:
  static double Now();
  static void Run(void);
  static void Schedule(
      uint64_t delay,
      void (*fun_ptr)(void* fun_arg),
      void* fun_arg);
  static void Stop();
  static void Destroy();
};
#endif
