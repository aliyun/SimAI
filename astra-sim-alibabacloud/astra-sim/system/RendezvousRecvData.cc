/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "RendezvousRecvData.hh"
namespace AstraSim {
RendezvousRecvData::RendezvousRecvData(
    int nodeId,
    Sys* generator,
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg)
    : BasicEventHandlerData(generator, EventType::RendezvousRecv) {
  this->recv = new SimRecvCaller(
      generator, buffer, count, type, src, tag, request, msg_handler, fun_arg);
}
} // namespace AstraSim
