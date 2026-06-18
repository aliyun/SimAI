/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 NITK Surathkal
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
 * Authors: Vivek Jain <jain.vivek.anand@gmail.com>
 *          Viyom Mittal <viyommittal@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

#include "tcp-bbr.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpBbr");
NS_OBJECT_ENSURE_REGISTERED (TcpBbr);

const double TcpBbr::PACING_GAIN_CYCLE [] = {5.0 / 4, 3.0 / 4, 1, 1, 1, 1, 1, 1};

TypeId
TcpBbr::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpBbr")
    .SetParent<TcpCongestionOps> ()
    .AddConstructor<TcpBbr> ()
    .SetGroupName ("Internet")
    .AddAttribute ("Stream",
                   "Random number stream (default is set to 4 to align with Linux results)",
                   UintegerValue (4),
                   MakeUintegerAccessor (&TcpBbr::SetStream),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("HighGain",
                   "Value of high gain",
                   DoubleValue (2.89),
                   MakeDoubleAccessor (&TcpBbr::m_highGain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("BwWindowLength",
                   "Length of bandwidth windowed filter",
                   UintegerValue (10),
                   MakeUintegerAccessor (&TcpBbr::m_bandwidthWindowLength),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("RttWindowLength",
                   "Length of RTT windowed filter",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&TcpBbr::m_rtPropFilterLen),
                   MakeTimeChecker ())
    .AddAttribute ("ProbeRttDuration",
                   "Time to be spent in PROBE_RTT phase",
                   TimeValue (MilliSeconds (200)),
                   MakeTimeAccessor (&TcpBbr::m_probeRttDuration),
                   MakeTimeChecker ())
    .AddAttribute ("ExtraAckedRttWindowLength",
                   "Window length of extra acked window",
                   UintegerValue (5),
                   MakeUintegerAccessor (&TcpBbr::m_extraAckedWinRttLength),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("AckEpochAckedResetThresh",
                   "Max allowed val for m_ackEpochAcked, after which sampling epoch is reset",
                   UintegerValue (1 << 12),
                   MakeUintegerAccessor (&TcpBbr::m_ackEpochAckedResetThresh),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

TcpBbr::TcpBbr ()
  : TcpCongestionOps ()
{
  NS_LOG_FUNCTION (this);
  m_uv = CreateObject<UniformRandomVariable> ();
}

TcpBbr::TcpBbr (const TcpBbr &sock)
  : TcpCongestionOps (sock),
    m_bandwidthWindowLength (sock.m_bandwidthWindowLength),
    m_pacingGain (sock.m_pacingGain),
    m_cWndGain (sock.m_cWndGain),
    m_highGain (sock.m_highGain),
    m_isPipeFilled (sock.m_isPipeFilled),
    m_minPipeCwnd (sock.m_minPipeCwnd),
    m_roundCount (sock.m_roundCount),
    m_roundStart (sock.m_roundStart),
    m_nextRoundDelivered (sock.m_nextRoundDelivered),
    m_probeRttDuration (sock.m_probeRttDuration),
    m_probeRtPropStamp (sock.m_probeRtPropStamp),
    m_probeRttDoneStamp (sock.m_probeRttDoneStamp),
    m_probeRttRoundDone (sock.m_probeRttRoundDone),
    m_packetConservation (sock.m_packetConservation),
    m_priorCwnd (sock.m_priorCwnd),
    m_idleRestart (sock.m_idleRestart),
    m_targetCWnd (sock.m_targetCWnd),
    m_fullBandwidth (sock.m_fullBandwidth),
    m_fullBandwidthCount (sock.m_fullBandwidthCount),
    m_rtProp (Time::Max ()),
    m_sendQuantum (sock.m_sendQuantum),
    m_cycleStamp (sock.m_cycleStamp),
    m_cycleIndex (sock.m_cycleIndex),
    m_rtPropExpired (sock.m_rtPropExpired),
    m_rtPropFilterLen (sock.m_rtPropFilterLen),
    m_rtPropStamp (sock.m_rtPropStamp),
    m_isInitialized (sock.m_isInitialized),
    m_uv (sock.m_uv),
    m_delivered (sock.m_delivered),
    m_appLimited (sock.m_appLimited),  
    m_txItemDelivered (sock.m_txItemDelivered),  
    m_extraAckedGain (sock.m_extraAckedGain),
    m_extraAckedWinRtt (sock.m_extraAckedWinRtt),
    m_extraAckedWinRttLength (sock.m_extraAckedWinRttLength),
    m_ackEpochAckedResetThresh (sock.m_ackEpochAckedResetThresh),
    m_extraAckedIdx (sock.m_extraAckedIdx),
    m_ackEpochTime (sock.m_ackEpochTime),
    m_ackEpochAcked (sock.m_ackEpochAcked),
    m_hasSeenRtt (sock.m_hasSeenRtt)
{
  NS_LOG_FUNCTION (this);
}

const char* const
TcpBbr::BbrModeName[BBR_PROBE_RTT + 1] =
{
  "BBR_STARTUP", "BBR_DRAIN", "BBR_PROBE_BW", "BBR_PROBE_RTT"
};

void
TcpBbr::SetStream (uint32_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_uv->SetStream (stream);
}

void
TcpBbr::InitRoundCounting ()
{
  NS_LOG_FUNCTION (this);
  m_nextRoundDelivered = 0;
  m_roundStart = false;
  m_roundCount = 0;
}

void
TcpBbr::InitFullPipe ()
{
  NS_LOG_FUNCTION (this);
  m_isPipeFilled = false;
  m_fullBandwidth = 0;
  m_fullBandwidthCount = 0;
}

void
TcpBbr::InitPacingRate (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);

  if (!tcb->m_pacing)
    {
      NS_LOG_WARN ("BBR must use pacing");
      tcb->m_pacing = true;
    }

  Time rtt;
  if (tcb->m_minRtt != Time::Max ())
    {
      rtt = MilliSeconds (std::max<long int> (tcb->m_minRtt.GetMilliSeconds (), 1));
      m_hasSeenRtt = true;
    }
  else
    {
      rtt = MilliSeconds (1);
    }
  
  DataRate nominalBandwidth (tcb->m_cWnd * 8 / rtt.GetSeconds ());
  tcb->m_pacingRate = DataRate (m_pacingGain * nominalBandwidth.GetBitRate ());
  m_maxBwFilter = MaxBandwidthFilter_t (m_bandwidthWindowLength,
                                        DataRate (tcb->m_cWnd * 8 / rtt.GetSeconds ()),
                                        0);
}

void
TcpBbr::EnterStartup ()
{
  NS_LOG_FUNCTION (this);
  SetBbrState (BbrMode_t::BBR_STARTUP);
  m_pacingGain = m_highGain;
  m_cWndGain = m_highGain;
}

void
TcpBbr::HandleRestartFromIdle (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  if (tcb->m_bytesInFlight.Get () == 0U && rs.m_isAppLimited)
    {
      m_idleRestart = true;
      if (m_state == BbrMode_t::BBR_PROBE_BW)
        {
          SetPacingRate (tcb, 1);
        }
    }
}

void
TcpBbr::SetPacingRate (Ptr<TcpSocketState> tcb, double gain)
{
  NS_LOG_FUNCTION (this << tcb << gain);
  DataRate rate (gain * m_maxBwFilter.GetBest ().GetBitRate ());
  rate = std::min (rate, tcb->m_maxPacingRate);
  
  if (!m_hasSeenRtt && tcb->m_minRtt != Time::Max ())
    {
      InitPacingRate (tcb);
    }

  if (m_isPipeFilled || rate > tcb->m_pacingRate)
    {
      tcb->m_pacingRate = rate;
    }
}

uint32_t
TcpBbr::InFlight (Ptr<TcpSocketState> tcb, double gain)
{
  NS_LOG_FUNCTION (this << tcb << gain);
  if (m_rtProp == Time::Max ())
    {
      return tcb->m_initialCWnd * tcb->m_segmentSize;
    }
  double quanta = 3 * m_sendQuantum;
  double estimatedBdp = m_maxBwFilter.GetBest () * m_rtProp / 8.0;

  if (m_state == BbrMode_t::BBR_PROBE_BW && m_cycleIndex == 0)
    {
      return (gain * estimatedBdp) + quanta + (2 * tcb->m_segmentSize);
    }
  return (gain * estimatedBdp) + quanta;
}

void
TcpBbr::AdvanceCyclePhase ()
{
  NS_LOG_FUNCTION (this);
  m_cycleStamp = Simulator::Now ();
  m_cycleIndex = (m_cycleIndex + 1) % GAIN_CYCLE_LENGTH;
  m_pacingGain = PACING_GAIN_CYCLE [m_cycleIndex];
}

bool
TcpBbr::IsNextCyclePhase (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  bool isFullLength = (Simulator::Now () - m_cycleStamp) > m_rtProp;
  if (m_pacingGain == 1)
    {
      return isFullLength;
    }
  else if (m_pacingGain > 1)
    {
      return isFullLength && (rs.m_bytesLoss > 0 || rs.m_priorInFlight >= InFlight (tcb, m_pacingGain));
    }
  else
    {
      return isFullLength || rs.m_priorInFlight <= InFlight (tcb, 1);
    }
}

void
TcpBbr::CheckCyclePhase (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  if (m_state == BbrMode_t::BBR_PROBE_BW && IsNextCyclePhase (tcb, rs))
    {
      AdvanceCyclePhase ();
    }
}

void
TcpBbr::CheckFullPipe (const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << rs);
  if (m_isPipeFilled || !m_roundStart || rs.m_isAppLimited)
    {
      return;
    }

  /* Check if Bottleneck bandwidth is still growing*/
  if (m_maxBwFilter.GetBest ().GetBitRate () >= m_fullBandwidth.GetBitRate () * 1.25)
    {
      m_fullBandwidth = m_maxBwFilter.GetBest ();
      m_fullBandwidthCount = 0;
      return;
    }

  m_fullBandwidthCount++;
  if (m_fullBandwidthCount >= 3)
    {
      NS_LOG_DEBUG ("Pipe filled");
      m_isPipeFilled = true;
    }
}

void
TcpBbr::EnterDrain ()
{
  NS_LOG_FUNCTION (this);
  SetBbrState (BbrMode_t::BBR_DRAIN);
  m_pacingGain = 1.0 / m_highGain;
  m_cWndGain = m_highGain;
}

void
TcpBbr::EnterProbeBW ()
{
  NS_LOG_FUNCTION (this);
  SetBbrState (BbrMode_t::BBR_PROBE_BW);
  m_pacingGain = 1;
  m_cWndGain = 2;
  m_cycleIndex = GAIN_CYCLE_LENGTH - 1 - (int) m_uv->GetValue (0, 6);
  AdvanceCyclePhase ();
}

void
TcpBbr::CheckDrain (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  if (m_state == BbrMode_t::BBR_STARTUP && m_isPipeFilled)
    {
      EnterDrain ();
      tcb->m_ssThresh = InFlight (tcb, 1);
    }

  if (m_state == BbrMode_t::BBR_DRAIN && tcb->m_bytesInFlight <= InFlight (tcb, 1))
    {
      EnterProbeBW ();
    }
}

void
TcpBbr::UpdateRTprop (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  m_rtPropExpired = Simulator::Now () > (m_rtPropStamp + m_rtPropFilterLen);
  if (tcb->m_lastRtt >= Seconds (0) && (tcb->m_lastRtt <= m_rtProp || m_rtPropExpired))
    {
      m_rtProp = tcb->m_lastRtt;
      m_rtPropStamp = Simulator::Now ();
    }
}

void
TcpBbr::EnterProbeRTT ()
{
  NS_LOG_FUNCTION (this);
  SetBbrState (BbrMode_t::BBR_PROBE_RTT);
  m_pacingGain = 1;
  m_cWndGain = 1;
}

void
TcpBbr::SaveCwnd (Ptr<const TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  if (tcb->m_congState != TcpSocketState::CA_RECOVERY && m_state != BbrMode_t::BBR_PROBE_RTT)
    {
      m_priorCwnd = tcb->m_cWnd;
    }
  else
    {
      m_priorCwnd = std::max (m_priorCwnd, tcb->m_cWnd.Get ());
    }
}

void
TcpBbr::RestoreCwnd (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  tcb->m_cWnd = std::max (m_priorCwnd, tcb->m_cWnd.Get ());
}

void
TcpBbr::ExitProbeRTT ()
{
  NS_LOG_FUNCTION (this);
  if (m_isPipeFilled)
    {
      EnterProbeBW ();
    }
  else
    {
      EnterStartup ();
    }
}

void
TcpBbr::HandleProbeRTT (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  m_appLimited = (m_delivered + tcb->m_bytesInFlight.Get ()) ? : 1;

  if (m_probeRttDoneStamp == Seconds (0) && tcb->m_bytesInFlight <= m_minPipeCwnd)
    {
      m_probeRttDoneStamp = Simulator::Now () + m_probeRttDuration;
      m_probeRttRoundDone = false;
      m_nextRoundDelivered = m_delivered;
    }
  else if (m_probeRttDoneStamp != Seconds (0))
    {
      if (m_roundStart)
        {
          m_probeRttRoundDone = true;
        }
      if (m_probeRttRoundDone && Simulator::Now () > m_probeRttDoneStamp)
        {
          m_rtPropStamp = Simulator::Now ();
          RestoreCwnd (tcb);
          ExitProbeRTT ();
        }
    }
}

void
TcpBbr::CheckProbeRTT (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb);
  if (m_state != BbrMode_t::BBR_PROBE_RTT && m_rtPropExpired && !m_idleRestart)
    {
      EnterProbeRTT ();
      SaveCwnd (tcb);
      m_probeRttDoneStamp = Seconds (0);
    }

  if (m_state == BbrMode_t::BBR_PROBE_RTT)
    {
      HandleProbeRTT (tcb);
    }

  if (rs.m_delivered)
    {
      m_idleRestart = false;
    }
}

void
TcpBbr::SetSendQuantum (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  m_sendQuantum = 1 * tcb->m_segmentSize;
}

void
TcpBbr::UpdateTargetCwnd (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  m_targetCWnd = InFlight (tcb, m_cWndGain) + AckAggregationCwnd ();
}

uint32_t
TcpBbr::AckAggregationCwnd ()
{
  uint32_t maxAggrBytes; // MaxBW * 0.1 secs
  uint32_t aggrCwndBytes = 0;

  if (m_extraAckedGain && m_isPipeFilled)
    {
      maxAggrBytes = m_maxBwFilter.GetBest ().GetBitRate () / (10 * 8);
      aggrCwndBytes = m_extraAckedGain * std::max (m_extraAcked[0], m_extraAcked[1]);
      aggrCwndBytes = std::min (aggrCwndBytes, maxAggrBytes);
    }
  return aggrCwndBytes;
}

void
TcpBbr::UpdateAckAggregation (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  uint32_t expectedAcked, extraAck;
  uint32_t epochProp;

  if (!m_extraAckedGain || rs.m_ackedSacked <= 0 || rs.m_delivered < 0)
    {
      return;
    }

  if (m_roundStart)
    {
      m_extraAckedWinRtt = std::min<uint32_t> (31, m_extraAckedWinRtt + 1);
      if (m_extraAckedWinRtt >= m_extraAckedWinRttLength)
        {
          m_extraAckedWinRtt = 0;
          m_extraAckedIdx = m_extraAckedIdx ? 0 : 1;
          m_extraAcked[m_extraAckedIdx] = 0;
        }
    }

  epochProp = Simulator::Now ().GetSeconds () - m_ackEpochTime.GetSeconds ();
  expectedAcked = m_maxBwFilter.GetBest ().GetBitRate () * epochProp / 8;

  if (m_ackEpochAcked <= expectedAcked ||
      (m_ackEpochAcked + rs.m_ackedSacked >= m_ackEpochAckedResetThresh))
    {
      m_ackEpochAcked = 0;
      m_ackEpochTime = Simulator::Now ();
      expectedAcked = 0;
    }

  m_ackEpochAcked = m_ackEpochAcked + rs.m_ackedSacked;
  extraAck = m_ackEpochAcked - expectedAcked;
  extraAck = std::min (extraAck, tcb->m_cWnd.Get ());

  if (extraAck > m_extraAcked[m_extraAckedIdx])
    {
      m_extraAcked[m_extraAckedIdx] = extraAck;
    }
}

bool
TcpBbr::ModulateCwndForRecovery (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  if (rs.m_bytesLoss > 0)
    {
      tcb->m_cWnd = std::max ((int) tcb->m_cWnd.Get () - (int) rs.m_bytesLoss, (int) tcb->m_segmentSize);
    }

  if (m_packetConservation)
    {
      tcb->m_cWnd = std::max (tcb->m_cWnd.Get (), tcb->m_bytesInFlight.Get () + rs.m_ackedSacked);
      return true;
    }
  return false;
}

void
TcpBbr::ModulateCwndForProbeRTT (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  if (m_state == BbrMode_t::BBR_PROBE_RTT)
    {
      tcb->m_cWnd = std::min (tcb->m_cWnd.Get (), m_minPipeCwnd);
    }
}

void
TcpBbr::SetCwnd (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);

  if (!rs.m_ackedSacked)
    {
      goto done;
    }

  if (tcb->m_congState == TcpSocketState::CA_RECOVERY)
    {
      if (ModulateCwndForRecovery (tcb, rs)) 
        {
          goto done;
        }
    }

    UpdateTargetCwnd (tcb);

  if (m_isPipeFilled)
    {
      tcb->m_cWnd = std::min (tcb->m_cWnd.Get () + (uint32_t) rs.m_ackedSacked, m_targetCWnd);
    }
  else if (tcb->m_cWnd < m_targetCWnd || m_delivered < tcb->m_initialCWnd * tcb->m_segmentSize)
    {
      tcb->m_cWnd = tcb->m_cWnd.Get () + rs.m_ackedSacked;
    }
  tcb->m_cWnd = std::max (tcb->m_cWnd.Get (), m_minPipeCwnd);
  
done:
  ModulateCwndForProbeRTT (tcb);
}

void
TcpBbr::UpdateRound (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  if (rs.m_priorDelivered >= m_nextRoundDelivered)
    {
      m_nextRoundDelivered = m_delivered;
      m_roundCount++;
      m_roundStart = true;
      m_packetConservation = false;
    }
  else
    {
      m_roundStart = false;
    }
}

void
TcpBbr::UpdateBtlBw (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);

  if (rs.m_deliveryRate == 0)
    {
      return;
    }

  UpdateRound (tcb, rs);

  if (rs.m_deliveryRate >= m_maxBwFilter.GetBest () || !rs.m_isAppLimited)
    {
      m_maxBwFilter.Update (rs.m_deliveryRate, m_roundCount);
    }
}

void
TcpBbr::UpdateModelAndState (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  UpdateBtlBw (tcb, rs);
  UpdateAckAggregation (tcb, rs);
  CheckCyclePhase (tcb, rs);
  CheckFullPipe (rs);
  CheckDrain (tcb);
  UpdateRTprop (tcb);
  CheckProbeRTT (tcb, rs);
}

void
TcpBbr::UpdateControlParameters (Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  SetPacingRate (tcb, m_pacingGain);
  SetSendQuantum (tcb);
  SetCwnd (tcb, rs);
}

void
TcpBbr::SetBbrState (BbrMode_t mode)
{
  NS_LOG_FUNCTION (this << mode);
  NS_LOG_DEBUG (Simulator::Now () << " Changing from " << BbrModeName[m_state] << " to " << BbrModeName[mode]);
  m_state = mode;
}

uint32_t
TcpBbr::GetBbrState ()
{
  NS_LOG_FUNCTION (this);
  return m_state;
}

double
TcpBbr::GetCwndGain ()
{
  NS_LOG_FUNCTION (this);
  return m_cWndGain;
}

double
TcpBbr::GetPacingGain ()
{
  NS_LOG_FUNCTION (this);
  return m_pacingGain;
}

std::string
TcpBbr::GetName () const
{
  return "TcpBbr";
}

bool
TcpBbr::HasCongControl () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

void
TcpBbr::CongControl (Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection &rc,
                     const TcpRateOps::TcpRateSample &rs)
{
  NS_LOG_FUNCTION (this << tcb << rs);
  m_delivered = rc.m_delivered;
  m_txItemDelivered = rc.m_txItemDelivered;
  UpdateModelAndState (tcb, rs);
  UpdateControlParameters (tcb, rs);
}

void
TcpBbr::CongestionStateSet (Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState)
{
  NS_LOG_FUNCTION (this << tcb << newState);
  if (newState == TcpSocketState::CA_OPEN && !m_isInitialized)
    {
      NS_LOG_DEBUG ("CongestionStateSet triggered to CA_OPEN :: " << newState);
      m_rtProp = tcb->m_lastRtt.Get () != Time::Max () ? tcb->m_lastRtt.Get () : Time::Max ();
      m_rtPropStamp = Simulator::Now ();
      m_priorCwnd = tcb->m_cWnd;
      tcb->m_ssThresh = tcb->m_initialSsThresh;
      m_targetCWnd = tcb->m_cWnd;
      m_minPipeCwnd = 4 * tcb->m_segmentSize;
      m_sendQuantum = 1 * tcb->m_segmentSize;
      
      InitRoundCounting ();
      InitFullPipe ();
      EnterStartup ();      
      InitPacingRate (tcb);
      m_ackEpochTime = Simulator::Now();
      m_extraAckedWinRtt = 0;
      m_extraAckedIdx = 0;
      m_ackEpochAcked = 0;
      m_extraAcked[0] = 0;
      m_extraAcked[1] = 0;
      m_isInitialized = true;
    }
  else if (newState == TcpSocketState::CA_LOSS)
    {
      NS_LOG_DEBUG ("CongestionStateSet triggered to CA_LOSS :: " << newState);
      SaveCwnd (tcb);      
      m_roundStart = true;
    }
  else if (newState == TcpSocketState::CA_RECOVERY)
    {
      NS_LOG_DEBUG ("CongestionStateSet triggered to CA_RECOVERY :: " << newState);
      SaveCwnd (tcb);
      tcb->m_cWnd = tcb->m_bytesInFlight.Get () + std::max (tcb->m_lastAckedSackedBytes, tcb->m_segmentSize);
      m_packetConservation = true;
    }
}

void
TcpBbr::CwndEvent (Ptr<TcpSocketState> tcb,
                   const TcpSocketState::TcpCAEvent_t event)
{
  NS_LOG_FUNCTION (this << tcb << event);
  if (event == TcpSocketState::CA_EVENT_COMPLETE_CWR)
    {
      NS_LOG_DEBUG ("CwndEvent triggered to CA_EVENT_COMPLETE_CWR :: " << event);
      m_packetConservation = false;
      RestoreCwnd (tcb);
    }
  else if (event == TcpSocketState::CA_EVENT_TX_START && m_appLimited )
    {
      NS_LOG_DEBUG ("CwndEvent triggered to CA_EVENT_TX_START :: " << event);
      m_idleRestart = true;
      m_ackEpochTime = Simulator::Now ();
      m_ackEpochAcked = 0;
      if (m_state == BbrMode_t::BBR_PROBE_BW)
        {
          SetPacingRate (tcb, 1);
        }
      else if (m_state == BbrMode_t::BBR_PROBE_RTT)
        {
          if (m_probeRttRoundDone && Simulator::Now () > m_probeRttDoneStamp)
            {
              m_rtPropStamp = Simulator::Now ();
              RestoreCwnd (tcb);
              ExitProbeRTT ();
            }
        }
    }
}

uint32_t
TcpBbr::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
  NS_LOG_FUNCTION (this << tcb << bytesInFlight);
  SaveCwnd (tcb);
  return tcb->m_ssThresh;
}

Ptr<TcpCongestionOps>
TcpBbr::Fork (void)
{
  return CopyObject<TcpBbr> (this);
}

} // namespace ns3
