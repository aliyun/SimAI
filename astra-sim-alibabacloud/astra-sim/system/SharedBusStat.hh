/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SHAREDBUSSTAT_HH__
#define __SHAREDBUSSTAT_HH__

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>
#include "CallData.hh"

namespace AstraSim {
class SharedBusStat : public CallData {
 public:
  double total_shared_bus_transfer_queue_delay;
  double total_shared_bus_transfer_delay;
  double total_shared_bus_processing_queue_delay;
  double total_shared_bus_processing_delay;

  double total_mem_bus_transfer_queue_delay;
  double total_mem_bus_transfer_delay;
  double total_mem_bus_processing_queue_delay;
  double total_mem_bus_processing_delay;
  int mem_request_counter;
  int shared_request_counter;
  SharedBusStat(
      BusType busType,
      double total_bus_transfer_queue_delay,
      double total_bus_transfer_delay,
      double total_bus_processing_queue_delay,
      double total_bus_processing_delay) {
    if (busType == BusType::Shared) {
      this->total_shared_bus_transfer_queue_delay =
          total_bus_transfer_queue_delay;
      this->total_shared_bus_transfer_delay = total_bus_transfer_delay;
      this->total_shared_bus_processing_queue_delay =
          total_bus_processing_queue_delay;
      this->total_shared_bus_processing_delay = total_bus_processing_delay;

      this->total_mem_bus_transfer_queue_delay = 0;
      this->total_mem_bus_transfer_delay = 0;
      this->total_mem_bus_processing_queue_delay = 0;
      this->total_mem_bus_processing_delay = 0;
    } else {
      this->total_shared_bus_transfer_queue_delay = 0;
      this->total_shared_bus_transfer_delay = 0;
      this->total_shared_bus_processing_queue_delay = 0;
      this->total_shared_bus_processing_delay = 0;

      this->total_mem_bus_transfer_queue_delay = total_bus_transfer_queue_delay;
      this->total_mem_bus_transfer_delay = total_bus_transfer_delay;
      this->total_mem_bus_processing_queue_delay =
          total_bus_processing_queue_delay;
      this->total_mem_bus_processing_delay = total_bus_processing_delay;
    }
    shared_request_counter = 0;
    mem_request_counter = 0;
  }
  void update_bus_stats(BusType busType, SharedBusStat* sharedBusStat) {
    if (busType == BusType::Shared) {
      total_shared_bus_transfer_queue_delay +=
          sharedBusStat->total_shared_bus_transfer_queue_delay;
      total_shared_bus_transfer_delay +=
          sharedBusStat->total_shared_bus_transfer_delay;
      total_shared_bus_processing_queue_delay +=
          sharedBusStat->total_shared_bus_processing_queue_delay;
      total_shared_bus_processing_delay +=
          sharedBusStat->total_shared_bus_processing_delay;
      shared_request_counter++;
    } else if (busType == BusType::Mem) {
      total_mem_bus_transfer_queue_delay +=
          sharedBusStat->total_mem_bus_transfer_queue_delay;
      total_mem_bus_transfer_delay +=
          sharedBusStat->total_mem_bus_transfer_delay;
      total_mem_bus_processing_queue_delay +=
          sharedBusStat->total_mem_bus_processing_queue_delay;
      total_mem_bus_processing_delay +=
          sharedBusStat->total_mem_bus_processing_delay;
      mem_request_counter++;
    } else {
      total_shared_bus_transfer_queue_delay +=
          sharedBusStat->total_shared_bus_transfer_queue_delay;
      total_shared_bus_transfer_delay +=
          sharedBusStat->total_shared_bus_transfer_delay;
      total_shared_bus_processing_queue_delay +=
          sharedBusStat->total_shared_bus_processing_queue_delay;
      total_shared_bus_processing_delay +=
          sharedBusStat->total_shared_bus_processing_delay;
      total_mem_bus_transfer_queue_delay +=
          sharedBusStat->total_mem_bus_transfer_queue_delay;
      total_mem_bus_transfer_delay +=
          sharedBusStat->total_mem_bus_transfer_delay;
      total_mem_bus_processing_queue_delay +=
          sharedBusStat->total_mem_bus_processing_queue_delay;
      total_mem_bus_processing_delay +=
          sharedBusStat->total_mem_bus_processing_delay;
      shared_request_counter++;
      mem_request_counter++;
    }
  }
  void update_bus_stats(BusType busType, SharedBusStat sharedBusStat) {
    if (busType == BusType::Shared) {
      total_shared_bus_transfer_queue_delay +=
          sharedBusStat.total_shared_bus_transfer_queue_delay;
      total_shared_bus_transfer_delay +=
          sharedBusStat.total_shared_bus_transfer_delay;
      total_shared_bus_processing_queue_delay +=
          sharedBusStat.total_shared_bus_processing_queue_delay;
      total_shared_bus_processing_delay +=
          sharedBusStat.total_shared_bus_processing_delay;
      shared_request_counter++;
    } else if (busType == BusType::Mem) {
      total_mem_bus_transfer_queue_delay +=
          sharedBusStat.total_mem_bus_transfer_queue_delay;
      total_mem_bus_transfer_delay +=
          sharedBusStat.total_mem_bus_transfer_delay;
      total_mem_bus_processing_queue_delay +=
          sharedBusStat.total_mem_bus_processing_queue_delay;
      total_mem_bus_processing_delay +=
          sharedBusStat.total_mem_bus_processing_delay;
      mem_request_counter++;
    } else {
      total_shared_bus_transfer_queue_delay +=
          sharedBusStat.total_shared_bus_transfer_queue_delay;
      total_shared_bus_transfer_delay +=
          sharedBusStat.total_shared_bus_transfer_delay;
      total_shared_bus_processing_queue_delay +=
          sharedBusStat.total_shared_bus_processing_queue_delay;
      total_shared_bus_processing_delay +=
          sharedBusStat.total_shared_bus_processing_delay;
      total_mem_bus_transfer_queue_delay +=
          sharedBusStat.total_mem_bus_transfer_queue_delay;
      total_mem_bus_transfer_delay +=
          sharedBusStat.total_mem_bus_transfer_delay;
      total_mem_bus_processing_queue_delay +=
          sharedBusStat.total_mem_bus_processing_queue_delay;
      total_mem_bus_processing_delay +=
          sharedBusStat.total_mem_bus_processing_delay;
      shared_request_counter++;
      mem_request_counter++;
    }
  }
  void take_bus_stats_average() {
    total_shared_bus_transfer_queue_delay /= shared_request_counter;
    total_shared_bus_transfer_delay /= shared_request_counter;
    total_shared_bus_processing_queue_delay /= shared_request_counter;
    total_shared_bus_processing_delay /= shared_request_counter;

    total_mem_bus_transfer_queue_delay /= mem_request_counter;
    total_mem_bus_transfer_delay /= mem_request_counter;
    total_mem_bus_processing_queue_delay /= mem_request_counter;
    total_mem_bus_processing_delay /= mem_request_counter;
  }
};
} // namespace AstraSim
#endif
