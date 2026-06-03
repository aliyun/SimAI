/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 NITK Surathkal
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
 * Author: Ankit Deepak <adadeepak8@gmail.com>
 *
 */

#include "tcp-ledbat.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"
#include "ns3/simulator.h" // Now ()

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpLedbat");
NS_OBJECT_ENSURE_REGISTERED (TcpLedbat);

TypeId
TcpLedbat::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpLedbat")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpLedbat> ()
    .SetGroupName ("Internet")
    .AddAttribute ("TargetDelay",
                   "Targeted Queue Delay",
                   TimeValue (MilliSeconds (100)),
                   MakeTimeAccessor (&TcpLedbat::m_target),
                   MakeTimeChecker ())
    .AddAttribute ("BaseHistoryLen",
                   "Number of Base delay samples",
                   UintegerValue (10),
                   MakeUintegerAccessor (&TcpLedbat::m_baseHistoLen),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("NoiseFilterLen",
                   "Number of Current delay samples",
                   UintegerValue (4),
                   MakeUintegerAccessor (&TcpLedbat::m_noiseFilterLen),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Gain",
                   "Offset Gain",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&TcpLedbat::m_gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("SSParam",
                   "Possibility of Slow Start",
                   EnumValue (DO_SLOWSTART),
                   MakeEnumAccessor (&TcpLedbat::SetDoSs),
                   MakeEnumChecker (DO_SLOWSTART, "yes",
                                    DO_NOT_SLOWSTART, "no"))
    .AddAttribute ("MinCwnd",
                   "Minimum cWnd for Ledbat",
                   UintegerValue (2),
                   MakeUintegerAccessor (&TcpLedbat::m_minCwnd),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

void TcpLedbat::SetDoSs (SlowStartType doSS)
{
  NS_LOG_FUNCTION (this << doSS);
  m_doSs = doSS;
  if (m_doSs)
    {
      m_flag |= LEDBAT_CAN_SS;
    }
  else
    {
      m_flag &= ~LEDBAT_CAN_SS;
    }
}

TcpLedbat::TcpLedbat (void)
  : TcpNewReno ()
{
  NS_LOG_FUNCTION (this);
  m_target = MilliSeconds (100);
  m_gain = 1;
  m_doSs = DO_SLOWSTART;
  m_baseHistoLen = 10;
  m_noiseFilterLen = 4;
  InitCircBuf (m_baseHistory);
  InitCircBuf (m_noiseFilter);
  m_lastRollover = 0;
  m_sndCwndCnt = 0;
  m_flag = LEDBAT_CAN_SS;
  m_minCwnd = 2;
};

void TcpLedbat::InitCircBuf (struct OwdCircBuf &buffer)
{
  NS_LOG_FUNCTION (this);
  buffer.buffer.clear ();
  buffer.min = 0;
}

TcpLedbat::TcpLedbat (const TcpLedbat& sock)
  : TcpNewReno (sock)
{
  NS_LOG_FUNCTION (this);
  m_target = sock.m_target;
  m_gain = sock.m_gain;
  m_doSs = sock.m_doSs;
  m_baseHistoLen = sock.m_baseHistoLen;
  m_noiseFilterLen = sock.m_noiseFilterLen;
  m_baseHistory = sock.m_baseHistory;
  m_noiseFilter = sock.m_noiseFilter;
  m_lastRollover = sock.m_lastRollover;
  m_sndCwndCnt = sock.m_sndCwndCnt;
  m_flag = sock.m_flag;
  m_minCwnd = sock.m_minCwnd;
}

TcpLedbat::~TcpLedbat (void)
{
  NS_LOG_FUNCTION (this);
}

Ptr<TcpCongestionOps>
TcpLedbat::Fork (void)
{
  return CopyObject<TcpLedbat> (this);
}

std::string
TcpLedbat::GetName () const
{
  return "TcpLedbat";
}

uint32_t TcpLedbat::MinCircBuf (struct OwdCircBuf &b)
{
  NS_LOG_FUNCTION_NOARGS ();
  if (b.buffer.size () == 0)
    {
      return ~0U;
    }
  else
    {
      return b.buffer[b.min];
    }
}

uint32_t TcpLedbat::CurrentDelay (FilterFunction filter)
{
  NS_LOG_FUNCTION (this);
  return filter (m_noiseFilter);
}

uint32_t TcpLedbat::BaseDelay ()
{
  NS_LOG_FUNCTION (this);
  return MinCircBuf (m_baseHistory);
}

void TcpLedbat::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);
  if (tcb->m_cWnd.Get () <= tcb->m_segmentSize)
    {
      m_flag |= LEDBAT_CAN_SS;
    }
  if (m_doSs == DO_SLOWSTART && tcb->m_cWnd <= tcb->m_ssThresh && (m_flag & LEDBAT_CAN_SS))
    {
      SlowStart (tcb, segmentsAcked);
    }
  else
    {
      m_flag &= ~LEDBAT_CAN_SS;
      CongestionAvoidance (tcb, segmentsAcked);
    }
}

void TcpLedbat::CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);
  if ((m_flag & LEDBAT_VALID_OWD) == 0)
    {
      TcpNewReno::CongestionAvoidance (tcb, segmentsAcked); //letting it fall to TCP behaviour if no timestamps
      return;
    }
  int64_t queue_delay;
  double offset;
  uint32_t cwnd = (tcb->m_cWnd.Get ());
  uint32_t max_cwnd;
  uint64_t current_delay = CurrentDelay (&TcpLedbat::MinCircBuf);
  uint64_t base_delay = BaseDelay ();

  if (current_delay > base_delay)
    {
      queue_delay = static_cast<int64_t> (current_delay - base_delay);
      offset = m_target.GetMilliSeconds () - queue_delay;
    }
  else
    {
      queue_delay = static_cast<int64_t> (base_delay - current_delay);
      offset = m_target.GetMilliSeconds () + queue_delay;
    }
  offset *= m_gain;
  m_sndCwndCnt = static_cast<int32_t> (offset * segmentsAcked * tcb->m_segmentSize);
  double inc =  (m_sndCwndCnt * 1.0) / (m_target.GetMilliSeconds () * tcb->m_cWnd.Get ());
  cwnd += (inc * tcb->m_segmentSize);

  max_cwnd = static_cast<uint32_t>(tcb->m_highTxMark.Get () - tcb->m_lastAckedSeq) + segmentsAcked * tcb->m_segmentSize;
  cwnd = std::min (cwnd, max_cwnd);
  cwnd = std::max (cwnd, m_minCwnd * tcb->m_segmentSize);
  tcb->m_cWnd = cwnd;

  if (tcb->m_cWnd <= tcb->m_ssThresh)
    {
      tcb->m_ssThresh = tcb->m_cWnd - 1;
    }
}

void TcpLedbat::AddDelay (struct OwdCircBuf &cb, uint32_t owd, uint32_t maxlen)
{
  NS_LOG_FUNCTION (this << owd << maxlen << cb.buffer.size ());
  if (cb.buffer.size () == 0)
    {
      NS_LOG_LOGIC ("First Value for queue");
      cb.buffer.push_back (owd);
      cb.min = 0;
      return;
    }
  cb.buffer.push_back (owd);
  if (cb.buffer[cb.min] > owd)
    {
      cb.min = static_cast<uint32_t> (cb.buffer.size () - 1);
    }
  if (cb.buffer.size () >= maxlen)
    {
      NS_LOG_LOGIC ("Queue full" << maxlen);
      cb.buffer.erase (cb.buffer.begin ());
      cb.min = 0;
      NS_LOG_LOGIC ("Current min element" << cb.buffer[cb.min]);
      for (uint32_t i = 1; i < maxlen - 1; i++)
        {
          if (cb.buffer[i] < cb.buffer[cb.min])
            {
              cb.min = i;
            }
        }
    }
}

void TcpLedbat::UpdateBaseDelay (uint32_t owd)
{
  NS_LOG_FUNCTION (this << owd );
  if (m_baseHistory.buffer.size () == 0)
    {
      AddDelay (m_baseHistory, owd, m_baseHistoLen);
      return;
    }
  uint64_t timestamp = static_cast<uint64_t> (Simulator::Now ().GetSeconds ());

  if (timestamp - m_lastRollover > 60)
    {
      m_lastRollover = timestamp;
      AddDelay (m_baseHistory, owd, m_baseHistoLen);
    }
  else
    {
      uint32_t last = static_cast<uint32_t> (m_baseHistory.buffer.size () - 1);
      if (owd < m_baseHistory.buffer[last])
        {
          m_baseHistory.buffer[last] = owd;
          if (owd < m_baseHistory.buffer[m_baseHistory.min])
            {
              m_baseHistory.min = last;
            }
        }
    }
}

void TcpLedbat::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                           const Time& rtt)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked << rtt);
  if (tcb->m_rcvTimestampValue == 0 || tcb->m_rcvTimestampEchoReply == 0)
    {
      m_flag &= ~LEDBAT_VALID_OWD;
    }
  else
    {
      m_flag |= LEDBAT_VALID_OWD;
    }
  if (rtt.IsPositive ())
    {
      AddDelay (m_noiseFilter, tcb->m_rcvTimestampValue - tcb->m_rcvTimestampEchoReply, m_noiseFilterLen);
      UpdateBaseDelay (tcb->m_rcvTimestampValue - tcb->m_rcvTimestampEchoReply);
    }
}

} // namespace ns3
