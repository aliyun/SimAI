/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "SimSendCaller.hh"
#include "Sys.hh"
namespace AstraSim {
SimSendCaller::SimSendCaller(
    Sys* generator,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  this->generator = generator;
  this->buffer = buffer;
  this->count = count;
  this->type = type;
  this->dst = dst;
  this->tag = tag;
  this->request = request;
  this->msg_handler = msg_handler;
  this->fun_arg = fun_arg;
}
void SimSendCaller::call(EventType type, CallData* data) {
  generator->NI->sim_send(
      this->buffer,
      this->count,
      this->type,
      this->dst,
      this->tag,
      &this->request,
      this->msg_handler,
      this->fun_arg);
  delete this;
}
} // namespace AstraSim
