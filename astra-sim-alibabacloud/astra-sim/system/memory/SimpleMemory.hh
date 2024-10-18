/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SIMPLEMEMORY_HH__
#define __SIMPLEMEMORY_HH__
#include "astra-sim/system/AstraMemoryAPI.hh"
#include "astra-sim/system/AstraNetworkAPI.hh"
namespace AstraSim {
class SimpleMemory : public AstraMemoryAPI {
 public:
  double last_read_request_serviced;
  double last_write_request_serviced;
  double npu_access_bw_GB;
  double nic_access_bw_GB;
  double access_latency;
  uint64_t nic_read_request_count;
  uint64_t nic_write_request_count;
  uint64_t npu_read_request_count;
  uint64_t npu_write_request_count;
  AstraNetworkAPI* NI;
  SimpleMemory(
      AstraNetworkAPI* NI,
      double access_latency,
      double npu_access_bw_GB,
      double nic_access_bw_GB);
  void set_network_api(AstraNetworkAPI* astraNetworkApi);
  uint64_t npu_mem_read(uint64_t size);
  uint64_t npu_mem_write(uint64_t size);
  uint64_t nic_mem_read(uint64_t size);
  uint64_t nic_mem_write(uint64_t size);
  uint64_t mem_read(uint64_t size);
  uint64_t mem_write(uint64_t size);
};
} // namespace AstraSim
#endif
