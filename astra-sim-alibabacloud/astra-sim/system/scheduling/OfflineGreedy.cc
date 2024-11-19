/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "OfflineGreedy.hh"
#include <math.h>
#include <numeric>
namespace AstraSim {

std::map<long long, std::vector<int>> OfflineGreedy::chunk_schedule;
std::map<long long, int> OfflineGreedy::schedule_consumer;
std::map<long long, uint64_t> OfflineGreedy::global_chunk_size;

DimElapsedTime::DimElapsedTime(int dim_num) {
  this->dim_num = dim_num;
  this->elapsed_time = 0;
}
OfflineGreedy::OfflineGreedy(Sys* sys) {
  this->sys = sys;
  if (sys->dim_to_break == -1) {
    this->dim_size = sys->physical_dims;
    this->dim_BW.resize(this->dim_size.size());
    for (int i = 0; i < this->dim_size.size(); i++) {
      this->dim_BW[i] = sys->NI->get_BW_at_dimension(i);
      this->dim_elapsed_time.push_back(DimElapsedTime(i));
    }
  } else {
    this->dim_size = sys->logical_broken_dims;
    this->dim_BW.resize(this->dim_size.size());
    for (int i = 0; i < this->dim_size.size(); i++) {
      if (i > sys->dim_to_break) {
        this->dim_BW[i] = sys->NI->get_BW_at_dimension(i - 1);
      } else {
        this->dim_BW[i] = sys->NI->get_BW_at_dimension(i);
      }
      this->dim_elapsed_time.push_back(DimElapsedTime(i));
    }
  }
  if (sys->id == 0) {
    std::cout << "Themis is configured with the following parameters: "
              << std::endl;
    std::cout << "Dim size: ";
    for (int i = 0; i < this->dim_size.size(); i++) {
      std::cout << this->dim_size[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "BW per dim: ";
    for (int i = 0; i < this->dim_BW.size(); i++) {
      std::cout << this->dim_BW[i] << ", ";
    }
    std::cout << std::endl << std::endl;
  }
}
uint64_t OfflineGreedy::get_chunk_size_from_elapsed_time(
    double elapsed_time,
    DimElapsedTime dim,
    ComType comm_type) {
  if (comm_type == ComType::Reduce_Scatter) {
    uint64_t result =
        ((elapsed_time * (dim_BW[dim.dim_num] / dim_BW[0])) /
         (((double)(dim_size[dim.dim_num] - 1)) / (dim_size[dim.dim_num]))) *
        1048576;
    return result;
  } else {
    uint64_t result = ((elapsed_time * (dim_BW[dim.dim_num] / dim_BW[0])) /
                       (((double)(dim_size[dim.dim_num] - 1)) / (1))) *
        1048576;
    return result;
  }
}
void OfflineGreedy::reset_loads() {
  int i = 0;
  for (auto& dim : dim_elapsed_time) {
    dim.elapsed_time = 0;
    dim.dim_num = i;
    i++;
  }
}
std::vector<int> OfflineGreedy::get_chunk_scheduling(
    long long chunk_id,
    uint64_t& remaining_data_size,
    uint64_t recommended_chunk_size,
    std::vector<bool>& dimensions_involved,
    InterDimensionScheduling inter_dim_scheduling,
    ComType comm_type) {
  if (chunk_schedule.find(chunk_id) != chunk_schedule.end()) {
    schedule_consumer[chunk_id]++;
    if (schedule_consumer[chunk_id] == sys->all_generators.size()) {
      std::vector<int> res = chunk_schedule[chunk_id];
      remaining_data_size -= global_chunk_size[chunk_id];
      chunk_schedule.erase(chunk_id);
      schedule_consumer.erase(chunk_id);
      global_chunk_size.erase(chunk_id);
      return res;
    }
    remaining_data_size -= global_chunk_size[chunk_id];
    return chunk_schedule[chunk_id];
  }
  if (sys->id != 0) {
    return sys->all_generators[0]->offline_greedy->get_chunk_scheduling(
        chunk_id,
        remaining_data_size,
        recommended_chunk_size,
        dimensions_involved,
        inter_dim_scheduling,
        comm_type);
  } else {
    if (comm_type == ComType::All_Reduce) {
      comm_type = ComType::Reduce_Scatter;
    }
    std::sort(dim_elapsed_time.begin(), dim_elapsed_time.end());
    if (comm_type == ComType::All_Gather) {
      std::reverse(dim_elapsed_time.begin(), dim_elapsed_time.end());
    }
    std::vector<int> result;
    uint64_t chunk_size =
        recommended_chunk_size; //*(dim_BW[dim_elapsed_time.front().dim_num]/dim_BW[0]);
    bool chunk_size_calculated = false;
    if (inter_dim_scheduling == InterDimensionScheduling::OfflineGreedy) {
      global_chunk_size[chunk_id] = std::min(remaining_data_size, chunk_size);
      remaining_data_size -= std::min(remaining_data_size, chunk_size);
    }
    int dim_elapsed_time_pointer = -1;
    for (auto& dim : dim_elapsed_time) {
      dim_elapsed_time_pointer++;
      if (!dimensions_involved[dim.dim_num] || dim_size[dim.dim_num] == 1) {
        result.push_back(dim.dim_num);
        continue;
      } else if (
          inter_dim_scheduling == InterDimensionScheduling::OfflineGreedyFlex &&
          !chunk_size_calculated) {
        chunk_size_calculated = true;
        if (comm_type == ComType::Reduce_Scatter) {
          double load_difference =
              fabs(dim_elapsed_time.back().elapsed_time - dim.elapsed_time);
          chunk_size = get_chunk_size_from_elapsed_time(
              load_difference, dim, ComType::Reduce_Scatter);
        } else {
          int lastIndex = dim_elapsed_time.size() - 1;
          while (!dimensions_involved[dim_elapsed_time[lastIndex].dim_num] ||
                 dim_size[dim_elapsed_time[lastIndex].dim_num] == 1) {
            lastIndex--;
          }
          double load_difference =
              fabs(dim_elapsed_time[lastIndex].elapsed_time - dim.elapsed_time);
          chunk_size = get_chunk_size_from_elapsed_time(
              load_difference,
              dim_elapsed_time[lastIndex],
              ComType::All_Gather);
          lastIndex--;
          while (dim_elapsed_time_pointer <= lastIndex) {
            if (dimensions_involved[dim_elapsed_time[lastIndex].dim_num] &&
                dim_size[dim_elapsed_time[lastIndex].dim_num] > 1) {
              chunk_size /= dim_size[dim_elapsed_time[lastIndex].dim_num];
            }
            lastIndex--;
          }
        }
        if (chunk_size < (recommended_chunk_size)) {
          result.resize(dim_elapsed_time.size());
          std::iota(std::begin(result), std::end(result), 0);
          global_chunk_size[chunk_id] =
              std::min(remaining_data_size, recommended_chunk_size);
          chunk_size = std::min(remaining_data_size, recommended_chunk_size);
          remaining_data_size -=
              std::min(remaining_data_size, recommended_chunk_size);
          chunk_schedule[chunk_id] = result;
          schedule_consumer[chunk_id] = 1;
          std::vector<DimElapsedTime> myReordered;
          myReordered.resize(dim_elapsed_time.size(), dim_elapsed_time[0]);
          for (int myDim = 0; myDim < dim_elapsed_time.size(); myDim++) {
            for (int searchDim = 0; searchDim < dim_elapsed_time.size();
                 searchDim++) {
              if (dim_elapsed_time[searchDim].dim_num == myDim) {
                myReordered[myDim] = dim_elapsed_time[searchDim];
                break;
              }
            }
          }
          dim_elapsed_time = myReordered;
          if (comm_type == ComType::All_Gather) {
            std::reverse(dim_elapsed_time.begin(), dim_elapsed_time.end());
          }
          for (int myDim = 0; myDim < dim_elapsed_time.size(); myDim++) {
            if (!dimensions_involved[myDim] || dim_size[myDim] == 1) {
              result.push_back(myDim);
              continue;
            }
            if (comm_type == ComType::Reduce_Scatter) {
              dim_elapsed_time[myDim].elapsed_time +=
                  ((((double)chunk_size) / 1048576) *
                   (((double)(dim_size[myDim] - 1)) / (dim_size[myDim]))) /
                  (dim_BW[myDim] / dim_BW[0]);
              chunk_size /= dim_size[myDim];
            } else {
              dim_elapsed_time[myDim].elapsed_time +=
                  ((((double)chunk_size) / 1048576) *
                   (((double)(dim_size[myDim] - 1)))) /
                  (dim_BW[myDim] / dim_BW[0]);
              chunk_size *= dim_size[myDim];
            }
          }
          return result;
        } else {
          global_chunk_size[chunk_id] =
              std::min(remaining_data_size, chunk_size);
          remaining_data_size -= std::min(remaining_data_size, chunk_size);
        }
      } else if (
          inter_dim_scheduling == InterDimensionScheduling::OfflineGreedy &&
          !chunk_size_calculated) {
        chunk_size_calculated = true;
        uint64_t diff_size = 0;
        if (comm_type == ComType::Reduce_Scatter) {
          double load_difference =
              fabs(dim_elapsed_time.back().elapsed_time - dim.elapsed_time);
          diff_size = get_chunk_size_from_elapsed_time(
              load_difference, dim, ComType::Reduce_Scatter);
        } else {
          int lastIndex = dim_elapsed_time.size() - 1;
          while (!dimensions_involved[dim_elapsed_time[lastIndex].dim_num] ||
                 dim_size[dim_elapsed_time[lastIndex].dim_num] == 1) {
            lastIndex--;
          }
          double load_difference =
              fabs(dim_elapsed_time[lastIndex].elapsed_time - dim.elapsed_time);
          diff_size = get_chunk_size_from_elapsed_time(
              load_difference,
              dim_elapsed_time[lastIndex],
              ComType::All_Gather);
          lastIndex--;
          while (dim_elapsed_time_pointer <= lastIndex) {
            if (dimensions_involved[dim_elapsed_time[lastIndex].dim_num] &&
                dim_size[dim_elapsed_time[lastIndex].dim_num] > 1) {
              diff_size /= dim_size[dim_elapsed_time[lastIndex].dim_num];
            }
            lastIndex--;
          }
        }
        if (diff_size < (recommended_chunk_size / 16)) {
          result.resize(dim_elapsed_time.size());
          std::iota(std::begin(result), std::end(result), 0);
          chunk_schedule[chunk_id] = result;
          schedule_consumer[chunk_id] = 1;
          std::vector<DimElapsedTime> myReordered;
          myReordered.resize(dim_elapsed_time.size(), dim_elapsed_time[0]);
          for (int myDim = 0; myDim < dim_elapsed_time.size(); myDim++) {
            for (int searchDim = 0; searchDim < dim_elapsed_time.size();
                 searchDim++) {
              if (dim_elapsed_time[searchDim].dim_num == myDim) {
                myReordered[myDim] = dim_elapsed_time[searchDim];
                break;
              }
            }
          }
          dim_elapsed_time = myReordered;
          if (comm_type == ComType::All_Gather) {
            std::reverse(dim_elapsed_time.begin(), dim_elapsed_time.end());
          }
          for (int myDim = 0; myDim < dim_elapsed_time.size(); myDim++) {
            if (!dimensions_involved[myDim] || dim_size[myDim] == 1) {
              // result.push_back(myDim);
              continue;
            }
            if (comm_type == ComType::Reduce_Scatter) {
              dim_elapsed_time[myDim].elapsed_time +=
                  ((((double)chunk_size) / 1048576) *
                   (((double)(dim_size[myDim] - 1)) / (dim_size[myDim]))) /
                  (dim_BW[myDim] / dim_BW[0]);
              chunk_size /= dim_size[myDim];
            } else {
              dim_elapsed_time[myDim].elapsed_time +=
                  ((((double)chunk_size) / 1048576) *
                   (((double)(dim_size[myDim] - 1)))) /
                  (dim_BW[myDim] / dim_BW[0]);
              chunk_size *= dim_size[myDim];
            }
          }
          return result;
        }
      }
      result.push_back(dim.dim_num);
      if (comm_type == ComType::Reduce_Scatter) {
        dim.elapsed_time += ((((double)chunk_size) / 1048576) *
                             (((double)(dim_size[dim.dim_num] - 1)) /
                              (dim_size[dim.dim_num]))) /
            (dim_BW[dim.dim_num] / dim_BW[0]);
        chunk_size /= dim_size[dim.dim_num];
      } else {
        dim.elapsed_time += ((((double)chunk_size) / 1048576) *
                             (((double)(dim_size[dim.dim_num] - 1)))) /
            (dim_BW[dim.dim_num] / dim_BW[0]);
        chunk_size *= dim_size[dim.dim_num];
      }
    }
    chunk_schedule[chunk_id] = result;
    schedule_consumer[chunk_id] = 1;
    return result;
  }
}
} // namespace AstraSim
