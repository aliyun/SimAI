/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "RendezvousSendData.hh"

namespace AstraSim {
RendezvousSendData::RendezvousSendData(
    int nodeId,
    Sys* generator,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg)
    : BasicEventHandlerData(generator, EventType::RendezvousSend) {
  this->send = new SimSendCaller(
      generator, buffer, count, type, dst, tag, request, msg_handler, fun_arg);
}
} // namespace AstraSim
