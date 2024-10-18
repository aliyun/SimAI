/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Algorithm.hh"
namespace AstraSim {
Algorithm::Algorithm(int layer_num) {
  this->layer_num = layer_num;
  enabled = true;
}
void Algorithm::init(BaseStream* stream) {
  this->stream = stream;
  return;
}
void Algorithm::call(EventType event, CallData* data) {
  return;
}
void Algorithm::exit() {
  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
  return;
}
} // namespace AstraSim
