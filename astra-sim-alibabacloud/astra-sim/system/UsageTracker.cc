/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "UsageTracker.hh"
#include "Sys.hh"
namespace AstraSim {
UsageTracker::UsageTracker(int levels) {
  this->levels = levels;
  this->current_level = 0;
  this->last_tick = 0;
}
void UsageTracker::increase_usage() {
  if (current_level < levels - 1) {
    Usage u(current_level, last_tick, Sys::boostedTick());
    usage.push_back(u);
    current_level++;
    last_tick = Sys::boostedTick();
  }
}
void UsageTracker::decrease_usage() {
  if (current_level > 0) {
    Usage u(current_level, last_tick, Sys::boostedTick());
    usage.push_back(u);
    current_level--;
    last_tick = Sys::boostedTick();
  }
}
void UsageTracker::set_usage(int level) {
  if (current_level != level) {
    Usage u(current_level, last_tick, Sys::boostedTick());
    usage.push_back(u);
    current_level = level;
    last_tick = Sys::boostedTick();
  }
}
void UsageTracker::report(CSVWriter* writer, int offset) {
  uint64_t col = offset * 3;
  uint64_t row = 1;
  for (auto a : usage) {
    writer->write_cell(row, col, std::to_string(a.start));
    writer->write_cell(row++, col + 1, std::to_string(a.level));
  }
  // writer->close_file_cont();
  return;
}
std::list<std::pair<uint64_t, double>> UsageTracker::report_percentage(
    uint64_t cycles) {
  decrease_usage();
  increase_usage();
  Tick total_activity_possible = (this->levels - 1) * cycles;
  std::list<Usage>::iterator usage_pointer = this->usage.begin();
  Tick current_activity = 0;
  Tick period_start = 0;
  Tick period_end = cycles;
  std::list<std::pair<uint64_t, double>> result;
  while (usage_pointer != this->usage.end()) {
    Usage current_usage = *usage_pointer;
    uint64_t begin =
        std::max(static_cast<uint64_t>(period_start), current_usage.start);
    uint64_t end =
        std::min(static_cast<uint64_t>(period_end), current_usage.end);
    assert(begin <= end);
    current_activity += ((end - begin) * current_usage.level);
    if (current_usage.end >= period_end) {
      result.push_back(std::make_pair(
          (uint64_t)period_end,
          (((double)current_activity) / total_activity_possible) * 100));
      period_start += cycles;
      period_end += cycles;
      current_activity = 0;
    } else {
      std::advance(usage_pointer, 1);
    }
  }
  return result;
}
} // namespace AstraSim
