/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "MemBus.hh"
#include "LogGP.hh"
#include "Sys.hh"
#include "astra-sim/system/MockNcclLog.h"
namespace AstraSim {
MemBus::~MemBus() {
  delete NPU_side;
  delete MA_side;
}
MemBus::MemBus(
    std::string side1,
    std::string side2,
    Sys* generator,
    Tick L,
    Tick o,
    Tick g,
    double G,
    bool model_shared_bus,
    int communication_delay,
    bool attach) {
  NPU_side = new LogGP(side1, generator, L, o, g, G, EventType::MA_to_NPU);
  MA_side = new LogGP(side2, generator, L, o, g, G, EventType::NPU_to_MA);
  NPU_side->partner = MA_side;
  MA_side->partner = NPU_side;
  this->generator = generator;
  this->model_shared_bus = model_shared_bus;
  this->communication_delay = communication_delay;
  if (attach) {
    NPU_side->attach_mem_bus(
        generator, L, o, g, 0.0038, model_shared_bus, communication_delay);
  }
}
void MemBus::send_from_NPU_to_MA(
    MemBus::Transmition transmition,
    uint64_t bytes,
    bool processed,
    bool send_back,
    Callable* callable) {
  if (model_shared_bus && transmition == Transmition::Usual) {
    NPU_side->request_read(bytes, processed, send_back, callable);
  } else {
    if (transmition == Transmition::Fast) {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
      generator->register_event(
          callable,
          EventType::NPU_to_MA,
          new SharedBusStat(BusType::Shared, 0, 10, 0, 0),
          10);
    } else {
      generator->register_event(
          callable,
          EventType::NPU_to_MA,
          new SharedBusStat(BusType::Shared, 0, communication_delay, 0, 0),
          communication_delay);
    }
  }
}
void MemBus::send_from_MA_to_NPU(
    MemBus::Transmition transmition,
    

    uint64_t bytes,
    bool processed,
    bool send_back,
    Callable* callable) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (model_shared_bus && transmition == Transmition::Usual) {
    MA_side->request_read(bytes, processed, send_back, callable);
  } else {
    if (transmition == Transmition::Fast) {
      generator->register_event(
          callable,
          EventType::MA_to_NPU,
          new SharedBusStat(BusType::Shared, 0, 10, 0, 0),
          10);
    } else {
      generator->register_event(
          callable,
          EventType::MA_to_NPU,
          new SharedBusStat(BusType::Shared, 0, communication_delay, 0, 0),
          communication_delay);
    }
  }
}
} // namespace AstraSim
