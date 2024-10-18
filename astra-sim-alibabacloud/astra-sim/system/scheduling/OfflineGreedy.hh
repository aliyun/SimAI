/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __OFFLINEGREEDY_HH__
#define __OFFLINEGREEDY_HH__
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/Sys.hh"
#include "vector"
namespace AstraSim {
class DimElapsedTime {
 public:
  int dim_num;
  double elapsed_time;
  DimElapsedTime(int dim_num);
  bool operator<(const DimElapsedTime& dimElapsedTime) const {
    return (elapsed_time < dimElapsedTime.elapsed_time);
  }
};
class OfflineGreedy {
 public:
  Sys* sys;
  std::vector<DimElapsedTime> dim_elapsed_time;
  std::vector<double> dim_BW;
  std::vector<int> dim_size;
  OfflineGreedy(Sys* sys);
  void reset_loads();
  std::vector<int> get_chunk_scheduling(
      long long chunk_id,
      uint64_t& remaining_data_size,
      uint64_t recommended_chunk_size,
      std::vector<bool>& dimensions_involved,
      InterDimensionScheduling inter_dim_scheduling,
      ComType comm_type);
  uint64_t get_chunk_size_from_elapsed_time(
      double elapsed_time,
      DimElapsedTime dim,
      ComType comm_type);
  static std::map<long long, std::vector<int>> chunk_schedule;
  static std::map<long long, int> schedule_consumer;
  static std::map<long long, uint64_t> global_chunk_size;
};
} // namespace AstraSim
#endif
