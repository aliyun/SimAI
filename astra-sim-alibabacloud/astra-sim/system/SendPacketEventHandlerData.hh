/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef ASTRA_SIM_SENDPACKETEVENTHANDLERDATA_H
#define ASTRA_SIM_SENDPACKETEVENTHANDLERDATA_H

class SendPacketEventHandlerData {};

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
#include "BasicEventHandlerData.hh"
#include "Common.hh"
#include "Sys.hh"

namespace AstraSim {
class SendPacketEventHandlerData : public BasicEventHandlerData,
                                   public MetaData {
 public:
  BaseStream* owner;
  int senderNodeId;
  int receiverNodeId;
  int tag;
  // flow model
  int child_flow_id;
  int channel_id;
  AstraSim::ncclFlowTag flowTag;
  SendPacketEventHandlerData(Sys *node, int senderNodeId, int receiverNodeId, int tag);
  SendPacketEventHandlerData(BaseStream* owner, int senderNodeId, int receiverNodeId,int tag,EventType event);
};
} // namespace AstraSim
#endif // ASTRA_SIM_SENDPACKETEVENTHANDLERDATA_H
