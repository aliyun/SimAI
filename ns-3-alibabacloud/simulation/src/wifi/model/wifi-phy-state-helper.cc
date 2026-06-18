/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include <algorithm>
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "wifi-phy-state-helper.h"
#include "wifi-tx-vector.h"
#include "wifi-phy-listener.h"
#include "wifi-psdu.h"
#include "wifi-phy.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiPhyStateHelper");

NS_OBJECT_ENSURE_REGISTERED (WifiPhyStateHelper);

TypeId
WifiPhyStateHelper::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiPhyStateHelper")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiPhyStateHelper> ()
    .AddTraceSource ("State",
                     "The state of the PHY layer",
                     MakeTraceSourceAccessor (&WifiPhyStateHelper::m_stateLogger),
                     "ns3::WifiPhyStateHelper::StateTracedCallback")
    .AddTraceSource ("RxOk",
                     "A packet has been received successfully.",
                     MakeTraceSourceAccessor (&WifiPhyStateHelper::m_rxOkTrace),
                     "ns3::WifiPhyStateHelper::RxOkTracedCallback")
    .AddTraceSource ("RxError",
                     "A packet has been received unsuccessfully.",
                     MakeTraceSourceAccessor (&WifiPhyStateHelper::m_rxErrorTrace),
                     "ns3::WifiPhyStateHelper::RxEndErrorTracedCallback")
    .AddTraceSource ("Tx", "Packet transmission is starting.",
                     MakeTraceSourceAccessor (&WifiPhyStateHelper::m_txTrace),
                     "ns3::WifiPhyStateHelper::TxTracedCallback")
  ;
  return tid;
}

WifiPhyStateHelper::WifiPhyStateHelper ()
  : m_sleeping (false),
    m_isOff (false),
    m_endTx (Seconds (0)),
    m_endRx (Seconds (0)),
    m_endCcaBusy (Seconds (0)),
    m_endSwitching (Seconds (0)),
    m_startTx (Seconds (0)),
    m_startRx (Seconds (0)),
    m_startCcaBusy (Seconds (0)),
    m_startSwitching (Seconds (0)),
    m_startSleep (Seconds (0)),
    m_previousStateChangeTime (Seconds (0))
{
  NS_LOG_FUNCTION (this);
}

void
WifiPhyStateHelper::SetReceiveOkCallback (RxOkCallback callback)
{
  m_rxOkCallback = callback;
}

void
WifiPhyStateHelper::SetReceiveErrorCallback (RxErrorCallback callback)
{
  m_rxErrorCallback = callback;
}

void
WifiPhyStateHelper::RegisterListener (WifiPhyListener *listener)
{
  m_listeners.push_back (listener);
}

void
WifiPhyStateHelper::UnregisterListener (WifiPhyListener *listener)
{
  auto it = find (m_listeners.begin (), m_listeners.end (), listener);
  if (it != m_listeners.end ())
    {
      m_listeners.erase (it);
    }
}

bool
WifiPhyStateHelper::IsStateIdle (void) const
{
  return (GetState () == WifiPhyState::IDLE);
}

bool
WifiPhyStateHelper::IsStateCcaBusy (void) const
{
  return (GetState () == WifiPhyState::CCA_BUSY);
}

bool
WifiPhyStateHelper::IsStateRx (void) const
{
  return (GetState () == WifiPhyState::RX);
}

bool
WifiPhyStateHelper::IsStateTx (void) const
{
  return (GetState () == WifiPhyState::TX);
}

bool
WifiPhyStateHelper::IsStateSwitching (void) const
{
  return (GetState () == WifiPhyState::SWITCHING);
}

bool
WifiPhyStateHelper::IsStateSleep (void) const
{
  return (GetState () == WifiPhyState::SLEEP);
}

bool
WifiPhyStateHelper::IsStateOff (void) const
{
  return (GetState () == WifiPhyState::OFF);
}

Time
WifiPhyStateHelper::GetDelayUntilIdle (void) const
{
  Time retval;

  switch (GetState ())
    {
    case WifiPhyState::RX:
      retval = m_endRx - Simulator::Now ();
      break;
    case WifiPhyState::TX:
      retval = m_endTx - Simulator::Now ();
      break;
    case WifiPhyState::CCA_BUSY:
      retval = m_endCcaBusy - Simulator::Now ();
      break;
    case WifiPhyState::SWITCHING:
      retval = m_endSwitching - Simulator::Now ();
      break;
    case WifiPhyState::IDLE:
    case WifiPhyState::SLEEP:
    case WifiPhyState::OFF:
      retval = Seconds (0);
      break;
    default:
      NS_FATAL_ERROR ("Invalid WifiPhy state.");
      retval = Seconds (0);
      break;
    }
  retval = Max (retval, Seconds (0));
  return retval;
}

Time
WifiPhyStateHelper::GetLastRxStartTime (void) const
{
  return m_startRx;
}

Time
WifiPhyStateHelper::GetLastRxEndTime (void) const
{
  return m_endRx;
}

WifiPhyState
WifiPhyStateHelper::GetState (void) const
{
  if (m_isOff)
    {
      return WifiPhyState::OFF;
    }
  if (m_sleeping)
    {
      return WifiPhyState::SLEEP;
    }
  else if (m_endTx > Simulator::Now ())
    {
      return WifiPhyState::TX;
    }
  else if (m_endRx > Simulator::Now ())
    {
      return WifiPhyState::RX;
    }
  else if (m_endSwitching > Simulator::Now ())
    {
      return WifiPhyState::SWITCHING;
    }
  else if (m_endCcaBusy > Simulator::Now ())
    {
      return WifiPhyState::CCA_BUSY;
    }
  else
    {
      return WifiPhyState::IDLE;
    }
}

void
WifiPhyStateHelper::NotifyTxStart (Time duration, double txPowerDbm)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyTxStart (duration, txPowerDbm);
    }
}

void
WifiPhyStateHelper::NotifyRxStart (Time duration)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyRxStart (duration);
    }
}

void
WifiPhyStateHelper::NotifyRxEndOk (void)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyRxEndOk ();
    }
}

void
WifiPhyStateHelper::NotifyRxEndError (void)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyRxEndError ();
    }
}

void
WifiPhyStateHelper::NotifyMaybeCcaBusyStart (Time duration)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyMaybeCcaBusyStart (duration);
    }
}

void
WifiPhyStateHelper::NotifySwitchingStart (Time duration)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifySwitchingStart (duration);
    }
}

void
WifiPhyStateHelper::NotifySleep (void)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifySleep ();
    }
}

void
WifiPhyStateHelper::NotifyOff (void)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyOff ();
    }
}

void
WifiPhyStateHelper::NotifyWakeup (void)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyWakeup ();
    }
}

void
WifiPhyStateHelper::NotifyOn (void)
{
  NS_LOG_FUNCTION (this);
  for (const auto& listener : m_listeners)
    {
      listener->NotifyOn ();
    }
}

void
WifiPhyStateHelper::LogPreviousIdleAndCcaBusyStates (void)
{
  NS_LOG_FUNCTION (this);
  Time now = Simulator::Now ();
  WifiPhyState state = GetState ();
  if (state == WifiPhyState::CCA_BUSY)
    {
      Time ccaStart = std::max ({m_endRx, m_endTx, m_startCcaBusy, m_endSwitching});
      m_stateLogger (ccaStart, now - ccaStart, WifiPhyState::CCA_BUSY);
    }
  else if (state == WifiPhyState::IDLE)
    {
      Time idleStart = std::max ({m_endCcaBusy, m_endRx, m_endTx, m_endSwitching});
      NS_ASSERT (idleStart <= now);
      if (m_endCcaBusy > m_endRx
          && m_endCcaBusy > m_endSwitching
          && m_endCcaBusy > m_endTx)
        {
          Time ccaBusyStart = std::max ({m_endTx, m_endRx, m_startCcaBusy, m_endSwitching});
          Time ccaBusyDuration = idleStart - ccaBusyStart;
          if (ccaBusyDuration.IsStrictlyPositive ())
            {
              m_stateLogger (ccaBusyStart, ccaBusyDuration, WifiPhyState::CCA_BUSY);
            }
        }
      Time idleDuration = now - idleStart;
      if (idleDuration.IsStrictlyPositive ())
        {
          m_stateLogger (idleStart, idleDuration, WifiPhyState::IDLE);
        }
    }
}

void
WifiPhyStateHelper::SwitchToTx (Time txDuration, WifiConstPsduMap psdus, double txPowerDbm, WifiTxVector txVector)
{
  NS_LOG_FUNCTION (this << txDuration << psdus << txPowerDbm << txVector);
  if (!m_txTrace.IsEmpty ())
    {
      for (auto const& psdu : psdus)
        {
          m_txTrace (psdu.second->GetPacket (), txVector.GetMode (psdu.first),
                     txVector.GetPreambleType (), txVector.GetTxPowerLevel ());
        }
    }
  Time now = Simulator::Now ();
  switch (GetState ())
    {
    case WifiPhyState::RX:
      /* The packet which is being received as well
       * as its endRx event are cancelled by the caller.
       */
      m_stateLogger (m_startRx, now - m_startRx, WifiPhyState::RX);
      m_endRx = now;
      break;
    case WifiPhyState::CCA_BUSY:
      [[fallthrough]];
    case WifiPhyState::IDLE:
      LogPreviousIdleAndCcaBusyStates ();
      break;
    default:
      NS_FATAL_ERROR ("Invalid WifiPhy state.");
      break;
    }
  m_stateLogger (now, txDuration, WifiPhyState::TX);
  m_previousStateChangeTime = now;
  m_endTx = now + txDuration;
  m_startTx = now;
  NotifyTxStart (txDuration, txPowerDbm);
}

void
WifiPhyStateHelper::SwitchToRx (Time rxDuration)
{
  NS_LOG_FUNCTION (this << rxDuration);
  NS_ASSERT (IsStateIdle () || IsStateCcaBusy ());
  Time now = Simulator::Now ();
  switch (GetState ())
    {
    case WifiPhyState::IDLE:
      [[fallthrough]];
    case WifiPhyState::CCA_BUSY:
      LogPreviousIdleAndCcaBusyStates ();
      break;
    default:
      NS_FATAL_ERROR ("Invalid WifiPhy state " << GetState ());
      break;
    }
  m_previousStateChangeTime = now;
  m_startRx = now;
  m_endRx = now + rxDuration;
  NotifyRxStart (rxDuration);
  NS_ASSERT (IsStateRx ());
}

void
WifiPhyStateHelper::SwitchToChannelSwitching (Time switchingDuration)
{
  NS_LOG_FUNCTION (this << switchingDuration);
  Time now = Simulator::Now ();
  switch (GetState ())
    {
    case WifiPhyState::RX:
      /* The packet which is being received as well
       * as its endRx event are cancelled by the caller.
       */
      m_stateLogger (m_startRx, now - m_startRx, WifiPhyState::RX);
      m_endRx = now;
      break;
    case WifiPhyState::CCA_BUSY:
      [[fallthrough]];
    case WifiPhyState::IDLE:
      LogPreviousIdleAndCcaBusyStates ();
      break;
    default:
      NS_FATAL_ERROR ("Invalid WifiPhy state.");
      break;
    }

  m_endCcaBusy = std::min (now, m_endCcaBusy);
  m_stateLogger (now, switchingDuration, WifiPhyState::SWITCHING);
  m_previousStateChangeTime = now;
  m_startSwitching = now;
  m_endSwitching = now + switchingDuration;
  NotifySwitchingStart (switchingDuration);
  NS_ASSERT (IsStateSwitching ());
}

void
WifiPhyStateHelper::ContinueRxNextMpdu (Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo, WifiTxVector txVector)
{
  NS_LOG_FUNCTION (this << *psdu << rxSignalInfo << txVector);
  std::vector<bool> statusPerMpdu;
  if (!m_rxOkCallback.IsNull ())
    {
      m_rxOkCallback (psdu, rxSignalInfo, txVector, statusPerMpdu);
    }
}

void
WifiPhyStateHelper::SwitchFromRxEndOk (Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo, WifiTxVector txVector, uint16_t staId, std::vector<bool> statusPerMpdu)
{
  NS_LOG_FUNCTION (this << *psdu << rxSignalInfo << txVector << staId << statusPerMpdu.size () <<
                   std::all_of(statusPerMpdu.begin(), statusPerMpdu.end(), [](bool v) { return v; })); //returns true if all true
  NS_ASSERT (statusPerMpdu.size () != 0);
  NS_ASSERT (Abs (m_endRx - Simulator::Now ()) < MicroSeconds (1)); //1us corresponds to the maximum propagation delay (delay spread)
  //TODO: a better fix would be to call the function once all HE TB PPDUs are received
  if (!m_rxOkTrace.IsEmpty ())
    {
      m_rxOkTrace (psdu->GetPacket (), rxSignalInfo.snr, txVector.GetMode (staId),
                   txVector.GetPreambleType ());
    }
  NotifyRxEndOk ();
  DoSwitchFromRx ();
  if (!m_rxOkCallback.IsNull ())
    {
      m_rxOkCallback (psdu, rxSignalInfo, txVector, statusPerMpdu);
    }
}

void
WifiPhyStateHelper::SwitchFromRxEndError (Ptr<WifiPsdu> psdu, double snr)
{
  NS_LOG_FUNCTION (this << *psdu << snr);
  NS_ASSERT (Abs (m_endRx - Simulator::Now ()) < MicroSeconds (1)); //1us corresponds to the maximum propagation delay (delay spread)
  //TODO: a better fix would be to call the function once all HE TB PPDUs are received
  if (!m_rxErrorTrace.IsEmpty ())
    {
      m_rxErrorTrace (psdu->GetPacket (), snr);
    }
  NotifyRxEndError ();
  DoSwitchFromRx ();
  if (!m_rxErrorCallback.IsNull ())
    {
      m_rxErrorCallback (psdu);
    }
}

void
WifiPhyStateHelper::DoSwitchFromRx (void)
{
  NS_LOG_FUNCTION (this);
  Time now = Simulator::Now ();
  m_stateLogger (m_startRx, now - m_startRx, WifiPhyState::RX);
  m_previousStateChangeTime = now;
  m_endRx = Simulator::Now ();
  NS_ASSERT (IsStateIdle () || IsStateCcaBusy ());
}

void
WifiPhyStateHelper::SwitchMaybeToCcaBusy (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  if (GetState () != WifiPhyState::RX)
    {
      NotifyMaybeCcaBusyStart (duration);
    }
  Time now = Simulator::Now ();
  switch (GetState ())
    {
    case WifiPhyState::IDLE:
      LogPreviousIdleAndCcaBusyStates ();
      break;
    case WifiPhyState::RX:
      return;
    default:
      break;
    }
  if (GetState () != WifiPhyState::CCA_BUSY)
    {
      m_startCcaBusy = now;
    }
  m_endCcaBusy = std::max (m_endCcaBusy, now + duration);
}

void
WifiPhyStateHelper::SwitchToSleep (void)
{
  NS_LOG_FUNCTION (this);
  Time now = Simulator::Now ();
  switch (GetState ())
    {
    case WifiPhyState::IDLE:
      [[fallthrough]];
    case WifiPhyState::CCA_BUSY:
      LogPreviousIdleAndCcaBusyStates ();
      break;
    default:
      NS_FATAL_ERROR ("Invalid WifiPhy state.");
      break;
    }
  m_previousStateChangeTime = now;
  m_sleeping = true;
  m_startSleep = now;
  NotifySleep ();
  NS_ASSERT (IsStateSleep ());
}

void
WifiPhyStateHelper::SwitchFromSleep (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (IsStateSleep ());
  Time now = Simulator::Now ();
  m_stateLogger (m_startSleep, now - m_startSleep, WifiPhyState::SLEEP);
  m_previousStateChangeTime = now;
  m_sleeping = false;
  NotifyWakeup ();
}

void
WifiPhyStateHelper::SwitchFromRxAbort (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (IsStateCcaBusy ()); //abort is called (with OBSS_PD_CCA_RESET reason) before RX is set by payload start
  NotifyRxEndOk ();
  DoSwitchFromRx ();
  m_endCcaBusy = Simulator::Now ();
  NotifyMaybeCcaBusyStart (Seconds (0));
  NS_ASSERT (IsStateIdle ());
}

void
WifiPhyStateHelper::SwitchToOff (void)
{
  NS_LOG_FUNCTION (this);
  Time now = Simulator::Now ();
  switch (GetState ())
    {
    case WifiPhyState::RX:
      /* The packet which is being received as well
       * as its endRx event are cancelled by the caller.
       */
      m_stateLogger (m_startRx, now - m_startRx, WifiPhyState::RX);
      m_endRx = now;
      break;
    case WifiPhyState::TX:
      /* The packet which is being transmitted as well
       * as its endTx event are cancelled by the caller.
       */
      m_stateLogger (m_startTx, now - m_startTx, WifiPhyState::TX);
      m_endTx = now;
      break;
    case WifiPhyState::IDLE:
      [[fallthrough]];
    case WifiPhyState::CCA_BUSY:
      LogPreviousIdleAndCcaBusyStates ();
      break;
    default:
      NS_FATAL_ERROR ("Invalid WifiPhy state.");
      break;
    }
  m_previousStateChangeTime = now;
  m_isOff = true;
  NotifyOff ();
  NS_ASSERT (IsStateOff ());
}

void
WifiPhyStateHelper::SwitchFromOff (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (IsStateOff ());
  Time now = Simulator::Now ();
  m_previousStateChangeTime = now;
  m_isOff = false;
  NotifyOn ();
}

} //namespace ns3
