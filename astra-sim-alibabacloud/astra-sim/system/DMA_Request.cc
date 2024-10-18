/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "DMA_Request.hh"
namespace AstraSim {
DMA_Request::DMA_Request(int id, int slots, int latency, int bytes) {
  this->slots = slots;
  this->latency = latency;
  this->id = id;
  this->executed = false;
  this->stream_owner = nullptr;
  this->bytes = bytes;
}
DMA_Request::DMA_Request(
    int id,
    int slots,
    int latency,
    int bytes,
    Callable* stream_owner) {
  this->slots = slots;
  this->latency = latency;
  this->id = id;
  this->executed = false;
  this->bytes = bytes;
  this->stream_owner = stream_owner;
}
} // namespace AstraSim
