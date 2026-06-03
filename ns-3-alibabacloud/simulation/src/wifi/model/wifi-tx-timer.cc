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

#include "ns3/log.h"
#include "wifi-tx-timer.h"
#include "wifi-mac-queue-item.h"
#include "wifi-tx-vector.h"
#include "wifi-psdu.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiTxTimer");

WifiTxTimer::WifiTxTimer ()
  : m_timeoutEvent (),
    m_reason (NOT_RUNNING),
    m_impl (nullptr),
    m_end (Seconds (0))
{
}

WifiTxTimer::~WifiTxTimer ()
{
  m_timeoutEvent.Cancel ();
  m_impl = 0;
}

void
WifiTxTimer::Reschedule (const Time& delay)
{
  NS_LOG_FUNCTION (this << delay);

  if (m_timeoutEvent.IsRunning ())
    {
      NS_LOG_DEBUG ("Rescheduling " << GetReasonString (m_reason) << " timeout in "
                    << delay.As (Time::US));
      Time end = Simulator::Now () + delay;
      // If timer expiration is postponed, we have to do nothing but updating
      // the timer expiration, because Expire() will reschedule itself to be
      // executed at the correct time. If timer expiration is moved up, we
      // have to reschedule Expire() (which would be executed too late otherwise)
      if (end < m_end)
        {
          // timer expiration is moved up
          m_timeoutEvent.Cancel ();
          m_timeoutEvent = Simulator::Schedule (delay, &WifiTxTimer::Expire, this);
        }
      m_end = end;
    }
}

void
WifiTxTimer::Expire (void)
{
  NS_LOG_FUNCTION (this);
  Time now = Simulator::Now ();

  if (m_end == now)
    {
      m_impl->Invoke ();
    }
  else
    {
      m_timeoutEvent = Simulator::Schedule (m_end - now, &WifiTxTimer::Expire, this);
    }
}

WifiTxTimer::Reason
WifiTxTimer::GetReason (void) const
{
  NS_ASSERT (IsRunning ());
  return m_reason;
}

std::string
WifiTxTimer::GetReasonString (Reason reason) const
{
#define FOO(x) \
case WAIT_ ## x: \
  return # x; \
  break;

  switch (reason)
    {
    case NOT_RUNNING:
      return "NOT_RUNNING";
      break;
    FOO (CTS);
    FOO (NORMAL_ACK);
    FOO (BLOCK_ACK);
    FOO (NORMAL_ACK_AFTER_DL_MU_PPDU);
    FOO (BLOCK_ACKS_IN_TB_PPDU);
    FOO (TB_PPDU_AFTER_BASIC_TF);
    FOO (QOS_NULL_AFTER_BSRP_TF);
    FOO (BLOCK_ACK_AFTER_TB_PPDU);
    default:
      NS_ABORT_MSG ("Unknown reason");
    }
#undef FOO
}

bool
WifiTxTimer::IsRunning (void) const
{
  return m_timeoutEvent.IsRunning ();
}

void
WifiTxTimer::Cancel (void)
{
  NS_LOG_FUNCTION (this << GetReasonString (m_reason));
  m_timeoutEvent.Cancel ();
  m_impl = 0;
}

Time
WifiTxTimer::GetDelayLeft (void) const
{
  return Simulator::GetDelayLeft (m_timeoutEvent);
}

void
WifiTxTimer::SetMpduResponseTimeoutCallback (MpduResponseTimeout callback) const
{
  m_mpduResponseTimeoutCallback = callback;
}

void
WifiTxTimer::FeedTraceSource (Ptr<WifiMacQueueItem> item, WifiTxVector txVector)
{
  if (!m_mpduResponseTimeoutCallback.IsNull ())
    {
      m_mpduResponseTimeoutCallback (m_reason, item, txVector);
    }
}

void
WifiTxTimer::SetPsduResponseTimeoutCallback (PsduResponseTimeout callback) const
{
  m_psduResponseTimeoutCallback = callback;
}

void
WifiTxTimer::FeedTraceSource (Ptr<WifiPsdu> psdu, WifiTxVector txVector)
{
  if (!m_psduResponseTimeoutCallback.IsNull ())
    {
      m_psduResponseTimeoutCallback (m_reason, psdu, txVector);
    }
}

void
WifiTxTimer::SetPsduMapResponseTimeoutCallback (PsduMapResponseTimeout callback) const
{
  m_psduMapResponseTimeoutCallback = callback;
}

void
WifiTxTimer::FeedTraceSource (WifiPsduMap* psduMap, std::set<Mac48Address>* missingStations,
                              std::size_t nTotalStations)
{
  if (!m_psduMapResponseTimeoutCallback.IsNull ())
    {
      m_psduMapResponseTimeoutCallback (m_reason, psduMap, missingStations, nTotalStations);
    }
}

} //namespace ns3
