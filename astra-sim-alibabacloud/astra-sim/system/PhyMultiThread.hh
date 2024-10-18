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

#ifndef __PHYMULTITHREAD_HH__
#define __PHYMULTITHREAD_HH__
#include <mutex>
#include <thread>
#include<condition_variable>
#include<atomic>

#include"MockNcclLog.h"
#include"AstraNetworkAPI.hh"
#include"SimAiPhyCommon.hh"
#ifdef PHY_RDMA
#include"SimAiFlowModelRdma.hh"
#endif 

enum WORK_TYPE{SENDFINISHED,RECEIVEFINISHED};

void set_send_finished_callback(void (*msg_handler)(AstraSim::ncclFlowTag flowTag));

void set_receive_finished_callback(void (*msg_handler)(AstraSim::ncclFlowTag flowTag));

bool create_polling_cqe_thread(void * cq_ptr,int lcore_id = 0);

void notify_all_thread_finished();

class PhyMtpInterface{
public:
     class explicitCriticalSection
  {
  public:
    inline explicitCriticalSection ()
    {
      while (g_e_inCriticalSection.exchange (true, std::memory_order_acquire))
        ;
    }

    inline void ExitSection() 
    {
      g_e_inCriticalSection.store (false, std::memory_order_release);
    }

    inline ~explicitCriticalSection ()
    {
      g_e_inCriticalSection.store (false, std::memory_order_release);
    }
  };
  static std::atomic<bool> g_e_inCriticalSection;
};

#endif