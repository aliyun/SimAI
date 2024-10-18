/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "QueueLevelHandler.hh"
namespace AstraSim {
QueueLevelHandler::QueueLevelHandler(
    int level,
    int start,
    int end,
    AstraNetworkAPI::BackendType backend) {
  for (int i = start; i <= end; i++) {
    queues.push_back(i);    // queues means that it has [start, end) queues 
  }
  allocator = 0;
  first_allocator = 0;
  last_allocator = queues.size() / 2;
  this->level = level;
  this->backend = backend;
}
std::pair<int, RingTopology::Direction> QueueLevelHandler::get_next_queue_id() {
  RingTopology::Direction dir;
  if ((backend != AstraNetworkAPI::BackendType::Garnet || level > 0) &&
      queues.size() > 1 && allocator >= (queues.size() / 2)) {
    dir = RingTopology::Direction::Anticlockwise;
  } else {
    dir = RingTopology::Direction::Clockwise;
  }
  if (queues.size() == 0) {
    return std::make_pair(-1, dir);
  }
  int tmp = queues[allocator++];
  if (allocator == queues.size()) {
    allocator = 0;
  }
  return std::make_pair(tmp, dir);
}
std::pair<int, RingTopology::Direction> QueueLevelHandler::
    get_next_queue_id_first() {
  RingTopology::Direction dir;
  dir = RingTopology::Direction::Clockwise;
  if (queues.size() == 0) {
    return std::make_pair(-1, dir);
  }
  int tmp = queues[first_allocator++];
  if (first_allocator == queues.size() / 2) {
    first_allocator = 0;
  }
  return std::make_pair(tmp, dir);
}
std::pair<int, RingTopology::Direction> QueueLevelHandler::
    get_next_queue_id_last() {
  RingTopology::Direction dir;
  dir = RingTopology::Direction::Anticlockwise;
  if (queues.size() == 0) {
    return std::make_pair(-1, dir);
  }
  int tmp = queues[last_allocator++];
  if (last_allocator == queues.size()) {
    last_allocator = queues.size() / 2;
  }
  return std::make_pair(tmp, dir);
}
} // namespace AstraSim
