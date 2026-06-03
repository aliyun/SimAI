/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef WIFI_TX_TIMER_H
#define WIFI_TX_TIMER_H

#include "ns3/event-id.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/traced-callback.h"
#include <functional>
#include <unordered_map>

namespace ns3 {

class WifiMacQueueItem;
class WifiPsdu;
class WifiTxVector;
class Mac48Address;

typedef std::unordered_map <uint16_t /* staId */, Ptr<WifiPsdu> /* PSDU */> WifiPsduMap;

/**
 * \ingroup wifi
 *
 * This class is used to handle the timer that a station starts when transmitting
 * a frame that solicits a response. The timeout can be rescheduled (multiple times)
 * when the RXSTART.indication is received from the PHY.
 */
class WifiTxTimer
{
public:
  /**
   * \enum Reason
   * \brief The reason why the timer was started
   */
  enum Reason : uint8_t
    {
      NOT_RUNNING = 0,
      WAIT_CTS,
      WAIT_NORMAL_ACK,
      WAIT_BLOCK_ACK,
      WAIT_NORMAL_ACK_AFTER_DL_MU_PPDU,
      WAIT_BLOCK_ACKS_IN_TB_PPDU,
      WAIT_TB_PPDU_AFTER_BASIC_TF,
      WAIT_QOS_NULL_AFTER_BSRP_TF,
      WAIT_BLOCK_ACK_AFTER_TB_PPDU,
    };

  /** Default constructor */
  WifiTxTimer ();

  virtual ~WifiTxTimer ();

  /**
   * This method is called when a frame soliciting a response is transmitted.
   * This method starts a timer of the given duration and schedules a call to
   * the given method in case the timer expires.
   *
   * \tparam MEM \deduced Class method function signature type
   * \tparam OBJ \deduced Class type of the object
   * \tparam Args \deduced Type template parameter pack
   * \param reason the reason why the timer was started
   * \param delay the time to the expiration of the timer
   * \param mem_ptr Member method pointer to invoke
   * \param obj The object on which to invoke the member method
   * \param args The arguments to pass to the invoked method
   */
  template<typename MEM, typename OBJ, typename... Args>
  void Set (Reason reason, const Time &delay, MEM mem_ptr, OBJ obj, Args... args);

  /**
   * Reschedule the timer to time out the given amount of time from the moment
   * this function is called. Note that nothing is done if the timer is not running.
   *
   * \param delay the time to the expiration of the timer
   */
  void Reschedule (const Time &delay);

  /**
   * Get the reason why the timer was started. Call
   * this method only if the timer is running
   *
   * \return the reason why the timer was started
   */
  Reason GetReason (void) const;

  /**
   * Get a string associated with the given reason
   *
   * \param reason the given reason
   * \return a string associated with the given reason
   */
  std::string GetReasonString (Reason reason) const;

  /**
   * Return true if the timer is running
   *
   * \return true if the timer is running
   */
  bool IsRunning (void) const;

  /**
   * Cancel the timer.
   */
  void Cancel (void);

  /**
   * Get the remaining time until the timer will expire.
   *
   * \return the remaining time until the timer will expire.
   *         If the timer is not running, this method returns zero.
   */
  Time GetDelayLeft (void) const;

  /**
   * MPDU response timeout callback typedef
   */
  typedef Callback<void, uint8_t, Ptr<const WifiMacQueueItem>, const WifiTxVector&> MpduResponseTimeout;

  /**
   * PSDU response timeout callback typedef
   */
  typedef Callback<void, uint8_t, Ptr<const WifiPsdu>, const WifiTxVector&> PsduResponseTimeout;

  /**
   * PSDU map response timeout callback typedef
   */
  typedef Callback<void, uint8_t, WifiPsduMap*, const std::set<Mac48Address>*, std::size_t> PsduMapResponseTimeout;

  /**
   * Set the callback to invoke when the TX timer following the transmission of an MPDU expires.
   *
   * \param callback the callback to invoke when the TX timer following the transmission
   *                 of an MPDU expires
   */
  void SetMpduResponseTimeoutCallback (MpduResponseTimeout callback) const;

  /**
   * Set the callback to invoke when the TX timer following the transmission of a PSDU expires.
   *
   * \param callback the callback to invoke when the TX timer following the transmission
   *                 of a PSDU expires
   */
  void SetPsduResponseTimeoutCallback (PsduResponseTimeout callback) const;

  /**
   * Set the callback to invoke when the TX timer following the transmission of a PSDU map expires.
   *
   * \param callback the callback to invoke when the TX timer following the transmission
   *                 of a PSDU map expires
   */
  void SetPsduMapResponseTimeoutCallback (PsduMapResponseTimeout callback) const;

private:
  /**
   * This method is called when the timer expires. It invokes the callbacks
   * and the method set by the user.
   *
   * \tparam MEM \deduced Class method function signature type
   * \tparam OBJ \deduced Class type of the object
   * \tparam Args \deduced Type template parameter pack
   * \param mem_ptr Member method pointer to invoke
   * \param obj The object on which to invoke the member method
   * \param args The arguments to pass to the invoked method
   */
  template<typename MEM, typename OBJ, typename... Args>
  void Timeout (MEM mem_ptr, OBJ obj, Args... args);

  /**
   * Internal callback invoked when the timer expires.
   */
  void Expire (void);

  /**
   * This method is called when the timer expires to feed the MPDU response
   * timeout callback.
   *
   * \param item the MPDU followed by no response
   * \param txVector the TXVECTOR used to transmit the MPDU
   */
  void FeedTraceSource (Ptr<WifiMacQueueItem> item, WifiTxVector txVector);

  /**
   * This method is called when the timer expires to feed the PSDU response
   * timeout callback.
   *
   * \param psdu the PSDU followed by no response
   * \param txVector the TXVECTOR used to transmit the PSDU
   */
  void FeedTraceSource (Ptr<WifiPsdu> psdu, WifiTxVector txVector);

  /**
   * This method is called when the timer expires to feed the PSDU map response
   * timeout callback.
   *
   * \param psduMap the PSDU map for which not all responses were received
   * \param missingStations the set of stations that did not respond
   * \param nTotalStations the total number of expected responses
   */
  void FeedTraceSource (WifiPsduMap* psduMap, std::set<Mac48Address>* missingStations,
                        std::size_t nTotalStations);

  EventId m_timeoutEvent;         //!< the timeout event after a missing response
  Reason m_reason;                //!< the reason why the timer was started
  Ptr<EventImpl> m_impl;          /**< the timer implementation, which contains the bound
                                       callback function and arguments */
  Time m_end;                     //!< the absolute time when the timer will expire

  /// the MPDU response timeout callback
  mutable MpduResponseTimeout m_mpduResponseTimeoutCallback;
  /// the PSDU response timeout callback
  mutable PsduResponseTimeout m_psduResponseTimeoutCallback;
  /// the PSDU map response timeout callback
  mutable PsduMapResponseTimeout m_psduMapResponseTimeoutCallback;
};

} // namespace ns3


/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

namespace ns3 {

template<typename MEM, typename OBJ, typename... Args>
void
WifiTxTimer::Set (Reason reason, const Time &delay, MEM mem_ptr, OBJ obj, Args... args)
{
  typedef void (WifiTxTimer::*TimeoutType)(MEM, OBJ, Args...);

  m_timeoutEvent = Simulator::Schedule (delay, &WifiTxTimer::Expire, this);
  m_reason = reason;
  m_end = Simulator::Now () + delay;

  // create an event to invoke when the timer expires
  m_impl = Ptr<EventImpl> (MakeEvent<TimeoutType> (&WifiTxTimer::Timeout, this, mem_ptr, obj,
                                                   std::forward<Args> (args)... ), false);
}

template<typename MEM, typename OBJ, typename... Args>
void
WifiTxTimer::Timeout (MEM mem_ptr, OBJ obj, Args... args)
{
  FeedTraceSource (std::forward<Args> (args)...);

  // Invoke the method set by the user
  ((*obj).*mem_ptr)(std::forward<Args> (args)...);
}

} //namespace ns3

#endif /* WIFI_TX_TIMER_H */

