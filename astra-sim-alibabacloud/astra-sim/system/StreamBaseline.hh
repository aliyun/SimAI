/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __STREAMBASELINE_HH__
#define __STREAMBASELINE_HH__

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
#include "BaseStream.hh"
#include "CollectivePhase.hh"
#include "Common.hh"
#include "RecvPacketEventHadndlerData.hh"

namespace AstraSim {
class Sys;
class StreamBaseline : public BaseStream {
 public:
  StreamBaseline(
      Sys* owner,
      DataSet* dataset,
      int stream_num,
      std::list<CollectivePhase> phases_to_go,
      int priority);
  void init();
  void call(EventType event, CallData* data);
  void consume(RecvPacketEventHadndlerData* message); 
  void sendcallback(SendPacketEventHandlerData* messages);
  //~StreamBaseline()= default;
};
} // namespace AstraSim
#endif
