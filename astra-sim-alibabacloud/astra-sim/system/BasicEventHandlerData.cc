/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "BasicEventHandlerData.hh"
namespace AstraSim {
BasicEventHandlerData::BasicEventHandlerData(Sys* node, EventType event) {
  this->node = node;
  this->event = event;
  channel_id = -1;
  flow_id = -1;
}
BasicEventHandlerData::BasicEventHandlerData(int channel_id, int flow_id) {
  this->channel_id = channel_id;
  this->flow_id = flow_id;
}
} // namespace AstraSim
