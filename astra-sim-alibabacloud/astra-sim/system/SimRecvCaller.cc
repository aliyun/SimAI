/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "SimRecvCaller.hh"
#include "Sys.hh"
namespace AstraSim {
SimRecvCaller::SimRecvCaller(
    Sys* generator,
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    sim_request request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  this->generator = generator;
  this->buffer = buffer;
  this->count = count;
  this->type = type;
  this->src = src;
  this->tag = tag;
  this->request = request;
  this->msg_handler = msg_handler;
  this->fun_arg = fun_arg;
}
void SimRecvCaller::call(EventType type, CallData* data) {
  generator->NI->sim_recv(
      this->buffer,
      this->count,
      this->type,
      this->src,
      this->tag,
      &this->request,
      this->msg_handler,
      this->fun_arg);
  delete this;
}
} // namespace AstraSim
