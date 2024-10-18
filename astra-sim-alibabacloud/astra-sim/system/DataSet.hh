/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __DATASET_HH__
#define __DATASET_HH__

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
#include "Callable.hh"
#include "Common.hh"
#include "StreamStat.hh"

namespace AstraSim {
class DataSet : public Callable, public StreamStat {
 public:
  static int id_auto_increment;
  int my_id;
  int total_streams;
  int finished_streams;
  bool finished;
  bool active;
  Tick finish_tick;
  Tick creation_tick;
  Tick comm_start_tick;  
  Tick total_comm_time;
  std::pair<Callable*, EventType>* notifier;

  DataSet(int total_streams);
  void set_notifier(Callable* layer, EventType event);
  void notify_stream_finished(StreamStat* data);
  void call(EventType event, CallData* data);
  bool is_finished();
};
} // namespace AstraSim
#endif
