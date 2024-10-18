/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "MemMovRequest.hh"
#include "LogGP.hh"
#include "Sys.hh"
namespace AstraSim {
int MemMovRequest::id = 0;
MemMovRequest::MemMovRequest(
    int request_num,
    Sys* generator,
    LogGP* loggp,
    int size,
    int latency,
    Callable* callable,
    bool processed,
    bool send_back)
    : SharedBusStat(BusType::Mem, 0, 0, 0, 0) {
  this->size = size;
  this->latency = latency;
  this->callable = callable;
  this->processed = processed;
  this->send_back = send_back;
  this->my_id = id++;
  this->generator = generator;
  this->loggp = loggp;
  this->total_transfer_queue_time = 0;
  this->total_transfer_time = 0;
  this->total_processing_queue_time = 0;
  this->total_processing_time = 0;
  this->request_num = request_num;
  this->start_time = Sys::boostedTick();
  this->mem_bus_finished = true;
}
void MemMovRequest::call(EventType event, CallData* data) {
  update_bus_stats(BusType::Mem, (SharedBusStat*)data);
  total_mem_bus_transfer_delay +=
      ((SharedBusStat*)data)->total_shared_bus_transfer_delay;
  total_mem_bus_processing_delay +=
      ((SharedBusStat*)data)->total_shared_bus_processing_delay;
  total_mem_bus_processing_queue_delay +=
      ((SharedBusStat*)data)->total_shared_bus_processing_queue_delay;
  total_mem_bus_transfer_queue_delay +=
      ((SharedBusStat*)data)->total_shared_bus_transfer_queue_delay;
  mem_request_counter = 1;
  mem_bus_finished = true;
  loggp->talking_it = pointer;
  loggp->call(callEvent, data);
}
void MemMovRequest::wait_wait_for_mem_bus(
    std::list<MemMovRequest>::iterator pointer) {
  mem_bus_finished = false;
  this->pointer = pointer;
}
} // namespace AstraSim
