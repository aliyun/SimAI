/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "LogGP.hh"
#include "Sys.hh"
namespace AstraSim {
LogGP::~LogGP() {
  if (NPU_MEM != nullptr) {
    delete NPU_MEM;
  }
}
LogGP::LogGP(
    std::string name,
    Sys* generator,
    Tick L,
    Tick o,
    Tick g,
    double G,
    EventType trigger_event) {
  this->L = L;
  this->o = o;
  this->g = g;
  this->G = G;
  this->last_trans = 0;
  this->curState = State::Free;
  this->prevState = State::Free;
  this->generator = generator;
  this->processing_state = ProcState::Free;
  this->name = name;
  this->trigger_event = trigger_event;
  this->subsequent_reads = 0;
  this->THRESHOLD = 8;
  this->NPU_MEM = nullptr;
  request_num = 0;
  this->local_reduction_delay = generator->local_reduction_delay;

}
void LogGP::attach_mem_bus(
    Sys* generator,
    Tick L,
    Tick o,
    Tick g,
    double G,
    bool model_shared_bus,
    int communication_delay) {
  this->NPU_MEM = new MemBus(
      "NPU2",
      "MEM2",
      generator,
      L,
      o,
      g,
      G,
      model_shared_bus,
      communication_delay,
      false);
}
void LogGP::process_next_read() {
  Tick offset = 0;
  if (prevState == State::Sending) {
    assert(Sys::boostedTick() >= last_trans);
    if ((o + (Sys::boostedTick() - last_trans)) > g) {
      offset = o;
    } else {
      offset = g - (Sys::boostedTick() - last_trans);
    }
  } else {
    offset = o;
  }
  MemMovRequest tmp = sends.front();
  tmp.total_transfer_queue_time += Sys::boostedTick() - tmp.start_time;
  partner->switch_to_receiver(tmp, offset);
  sends.pop_front();
  curState = State::Sending;
  generator->register_event(
      this, EventType::Send_Finished, nullptr, offset + (G * (tmp.size - 1)));
}
void LogGP::request_read(
    int bytes,
    bool processed,
    bool send_back,
    Callable* callable) {
  MemMovRequest mr(
      request_num++, generator, this, bytes, 0, callable, processed, send_back);
  if (NPU_MEM != nullptr) {
    mr.callEvent = EventType::Consider_Send_Back;
    pre_send.push_back(mr);
    pre_send.back().wait_wait_for_mem_bus(--pre_send.end());
    NPU_MEM->send_from_MA_to_NPU(
        MemBus::Transmition::Usual, mr.size, false, false, &pre_send.back());
  } else {
    sends.push_back(mr);
    if (curState == State::Free) {
      if (subsequent_reads > THRESHOLD && partner->sends.size() > 0 &&
          partner->subsequent_reads <= THRESHOLD) {
        if (partner->curState == State::Free) {
          partner->call(EventType::General, nullptr);
        }
        return;
      }
      process_next_read();
    }
  }
}
void LogGP::switch_to_receiver(MemMovRequest mr, Tick offset) {
  mr.start_time = Sys::boostedTick();
  receives.push_back(mr);
  prevState = curState;
  curState = State::Receiving;
  generator->register_event(
      this,
      EventType::Rec_Finished,
      nullptr,
      offset + ((mr.size - 1) * G) + L + o);
  subsequent_reads = 0;
}
void LogGP::call(EventType event, CallData* data) {
  if (event == EventType::Send_Finished) {
    last_trans = Sys::boostedTick();
    prevState = curState;
    curState = State::Free;
    subsequent_reads++;
  } else if (event == EventType::Rec_Finished) {
    assert(receives.size() > 0);
    receives.front().total_transfer_time +=
        Sys::boostedTick() - receives.front().start_time;
    receives.front().start_time = Sys::boostedTick();
    last_trans = Sys::boostedTick();
    prevState = curState;
    if (receives.size() < 2) {
      curState = State::Free;
    }
    if (receives.front().processed == true) {
      if (NPU_MEM != nullptr) {
        receives.front().processed = false;
        receives.front().loggp = this;
        receives.front().callEvent = EventType::Consider_Process;
        pre_process.push_back(receives.front());
        receives.pop_front();
        pre_process.back().wait_wait_for_mem_bus(--pre_process.end());
        NPU_MEM->send_from_NPU_to_MA(
            MemBus::Transmition::Usual,
            pre_process.back().size,
            false,
            true,
            &pre_process.back());
      } else {
        receives.front().processed = false;
        processing.push_back(receives.front());
        receives.pop_front();
      }
      if (processing_state == ProcState::Free && processing.size() > 0) {
        processing.front().total_processing_queue_time +=
            Sys::boostedTick() - processing.front().start_time;
        processing.front().start_time = Sys::boostedTick();
        generator->register_event(
            this,
            EventType::Processing_Finished,
            nullptr,
            ((processing.front().size / 100) * local_reduction_delay) + 50);
        processing_state = ProcState::Processing;
      }
    } else if (receives.front().send_back == true) {
      if (NPU_MEM != nullptr) {
        receives.front().send_back = false;
        receives.front().callEvent = EventType::Consider_Send_Back;
        receives.front().loggp = this;
        pre_send.push_back(receives.front());
        receives.pop_front();
        pre_send.back().wait_wait_for_mem_bus(--pre_send.end());
        NPU_MEM->send_from_NPU_to_MA(
            MemBus::Transmition::Usual,
            pre_send.back().size,
            false,
            true,
            &pre_send.back());
      } else {
        receives.front().send_back = false;
        sends.push_back(receives.front());
        receives.pop_front();
      }
    } else {
      if (NPU_MEM != nullptr) {
        receives.front().callEvent = EventType::Consider_Retire;
        receives.front().loggp = this;
        retirements.push_back(receives.front());
        retirements.back().wait_wait_for_mem_bus(--retirements.end());
        NPU_MEM->send_from_NPU_to_MA(
            MemBus::Transmition::Usual,
            retirements.back().size,
            false,
            false,
            &retirements.back());
        receives.pop_front();
      } else {
        SharedBusStat* tmp = new SharedBusStat(
            BusType::Shared,
            receives.front().total_transfer_queue_time,
            receives.front().total_transfer_time,
            receives.front().total_processing_queue_time,
            receives.front().total_processing_time);
        tmp->update_bus_stats(BusType::Mem, receives.front());
        receives.front().callable->call(trigger_event, tmp);
        receives.pop_front();
      }
    }
  } else if (event == EventType::Processing_Finished) {
    assert(processing.size() > 0);
    processing.front().total_processing_time +=
        Sys::boostedTick() - processing.front().start_time;
    processing.front().start_time = Sys::boostedTick();
    processing_state = ProcState::Free;
    if (processing.front().send_back == true) {
      if (NPU_MEM != nullptr) {
        processing.front().send_back = false;
        processing.front().loggp = this;
        processing.front().callEvent = EventType::Consider_Send_Back;
        pre_send.push_back(processing.front());
        processing.pop_front();
        pre_send.back().wait_wait_for_mem_bus(--pre_send.end());
        NPU_MEM->send_from_NPU_to_MA(
            MemBus::Transmition::Usual,
            pre_send.back().size,
            false,
            true,
            &pre_send.back());
      } else {
        processing.front().send_back = false;
        sends.push_back(processing.front());
        processing.pop_front();
      }
    } else {
      if (NPU_MEM != nullptr) {
        processing.front().callEvent = EventType::Consider_Retire;
        processing.front().loggp = this;
        retirements.push_back(processing.front());
        retirements.back().wait_wait_for_mem_bus(--retirements.end());
        NPU_MEM->send_from_NPU_to_MA(
            MemBus::Transmition::Usual,
            retirements.back().size,
            false,
            false,
            &retirements.back());
        processing.pop_front();
      } else {
        SharedBusStat* tmp = new SharedBusStat(
            BusType::Shared,
            processing.front().total_transfer_queue_time,
            processing.front().total_transfer_time,
            processing.front().total_processing_queue_time,
            processing.front().total_processing_time);
        tmp->update_bus_stats(BusType::Mem, processing.front());
        processing.front().callable->call(trigger_event, tmp);
        processing.pop_front();
      }
    }
    if (processing.size() > 0) {
      processing.front().total_processing_queue_time +=
          Sys::boostedTick() - processing.front().start_time;
      processing.front().start_time = Sys::boostedTick();
      processing_state = ProcState::Processing;
      generator->register_event(
          this,
          EventType::Processing_Finished,
          nullptr,
          ((processing.front().size / 100) * local_reduction_delay) + 50);
    }
  } else if (event == EventType::Consider_Retire) {
    SharedBusStat* tmp = new SharedBusStat(
        BusType::Shared,
        retirements.front().total_transfer_queue_time,
        retirements.front().total_transfer_time,
        retirements.front().total_processing_queue_time,
        retirements.front().total_processing_time);
    MemMovRequest movRequest = *talking_it;
    tmp->update_bus_stats(BusType::Mem, movRequest);
    movRequest.callable->call(trigger_event, tmp);
    retirements.erase(talking_it);
    delete data;
  } else if (event == EventType::Consider_Process) {
    MemMovRequest movRequest = *talking_it;
    processing.push_back(movRequest);
    pre_process.erase(talking_it);
    if (processing_state == ProcState::Free && processing.size() > 0) {
      processing.front().total_processing_queue_time +=
          Sys::boostedTick() - processing.front().start_time;
      processing.front().start_time = Sys::boostedTick();
      generator->register_event(
          this,
          EventType::Processing_Finished,
          nullptr,
          ((processing.front().size / 100) * local_reduction_delay) + 50);
      processing_state = ProcState::Processing;
    }
    delete data;
  } else if (event == EventType::Consider_Send_Back) {
    assert(pre_send.size() > 0);
    MemMovRequest movRequest = *talking_it;
    sends.push_back(movRequest);
    pre_send.erase(talking_it);
    delete data;
  }
  if (curState == State::Free) {
    if (sends.size() > 0) {
      if (subsequent_reads > THRESHOLD && partner->sends.size() > 0 &&
          partner->subsequent_reads <= THRESHOLD) {
        if (partner->curState == State::Free) {
          partner->call(EventType::General, nullptr);
        }
        return;
      }
      process_next_read();
    }
  }
}
} // namespace AstraSim
