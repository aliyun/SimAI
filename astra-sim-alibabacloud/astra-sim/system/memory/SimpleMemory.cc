/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "SimpleMemory.hh"
namespace AstraSim {
SimpleMemory::SimpleMemory(
    AstraNetworkAPI* NI,
    double access_latency,
    double npu_access_bw_GB,
    double nic_access_bw_GB) {
  this->last_read_request_serviced = 0;
  this->last_write_request_serviced = 0;
  this->nic_read_request_count = 0;
  this->nic_write_request_count = 0;
  this->npu_read_request_count = 0;
  this->npu_write_request_count = 0;
  this->NI = NI;
  this->access_latency = access_latency;
  this->npu_access_bw_GB = npu_access_bw_GB;
  this->nic_access_bw_GB = nic_access_bw_GB;
}
void SimpleMemory::set_network_api(AstraNetworkAPI* astraNetworkApi) {
  this->NI = astraNetworkApi;
}
uint64_t SimpleMemory::npu_mem_read(uint64_t size) {
  npu_read_request_count++;
  double delay = (size / npu_access_bw_GB);
  return delay;
}
uint64_t SimpleMemory::SimpleMemory::npu_mem_write(uint64_t size) {
  npu_write_request_count++;
  double delay = (size / npu_access_bw_GB);
  return delay;
}
uint64_t SimpleMemory::nic_mem_read(uint64_t size) {
  nic_read_request_count++;
  timespec_t time = NI->sim_get_time();
  double time_ns = time.time_val;
  double delay = (size / nic_access_bw_GB);
  double offset = 0;
  if (time_ns + access_latency < last_read_request_serviced) {
    offset = (last_read_request_serviced + delay) - time_ns;
    last_read_request_serviced += delay;
  } else {
    offset = (time_ns + access_latency + delay) - time_ns;
    last_read_request_serviced = time_ns + access_latency + delay;
  }
  return (uint64_t)(offset);
}
uint64_t SimpleMemory::nic_mem_write(uint64_t size) {
  nic_write_request_count++;
  timespec_t time = NI->sim_get_time();
  double time_ns = time.time_val;
  double delay = (size / nic_access_bw_GB);
  double offset = 0;
  if (time_ns + access_latency < last_write_request_serviced) {
    offset = (last_write_request_serviced + delay) - time_ns;
    last_write_request_serviced += delay;
  } else {
    offset = (time_ns + access_latency + delay) - time_ns;
    last_write_request_serviced = time_ns + access_latency + delay;
  }
  return (uint64_t)(offset);
}
uint64_t SimpleMemory::mem_read(uint64_t size) {
  return nic_mem_read(size);
}
uint64_t SimpleMemory::mem_write(uint64_t size) {
  return nic_mem_write(size);
}
} // namespace AstraSim
