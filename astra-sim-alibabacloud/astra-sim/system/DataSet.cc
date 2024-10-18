/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "DataSet.hh"
#include "IntData.hh"
#include "Sys.hh"
#include "MockNcclLog.h"
namespace AstraSim {
int DataSet::id_auto_increment = 0;
DataSet::DataSet(int total_streams) {
  this->my_id = id_auto_increment++;
  this->total_streams = total_streams;
  this->finished_streams = 0;
  this->finished = false;
  this->finish_tick = 0;
  this->active = true;
  this->creation_tick = Sys::boostedTick();
  this->notifier = nullptr;
}
void DataSet::set_notifier(Callable* layer, EventType event) {
  notifier = new std::pair<Callable*, EventType>(layer, event);
}
void DataSet::notify_stream_finished(StreamStat* data) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"notify_stream_finished id: %d finished_streams: %d total streams: %d notify %p",my_id,finished_streams+1,total_streams,notifier);
  finished_streams++;
  if (data != nullptr) {
    update_stream_stats(data);
  }
  if (finished_streams == total_streams) {
    finished = true;
    finish_tick = Sys::boostedTick();
    if (notifier != nullptr) {
      NcclLog->writeLog(NcclLogLevel::DEBUG,"notify_stream_finished notifier != nullptr ");
      take_stream_stats_average();
      Callable* c = notifier->first;
      EventType ev = notifier->second;
      delete notifier;
      c->call(ev, new IntData(my_id));
    } else{
      NcclLog->writeLog(NcclLogLevel::ERROR,"notify_stream_finished notifier = nullptr ");
    }
  }
}
void DataSet::call(EventType event, CallData* data) {
  notify_stream_finished(((StreamStat*)data));
}
bool DataSet::is_finished() {
  return finished;
}
} // namespace AstraSim
