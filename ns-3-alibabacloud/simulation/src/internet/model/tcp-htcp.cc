/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
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
 * by: Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-htcp.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpHtcp");

NS_OBJECT_ENSURE_REGISTERED (TcpHtcp);

TypeId TcpHtcp::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpHtcp")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpHtcp> ()
    .SetGroupName ("Internet")
    .AddAttribute ("DefaultBackoff",
                   "The default AIMD backoff factor",
                   DoubleValue (0.5),
                   MakeDoubleAccessor (&TcpHtcp::m_defaultBackoff),
                   MakeDoubleChecker<double> (0,1))
    .AddAttribute ("ThroughputRatio",
                   "Threshold value for updating beta",
                   DoubleValue (0.2),
                   MakeDoubleAccessor (&TcpHtcp::m_throughputRatio),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("DeltaL",
                   "Delta_L parameter in increase function",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&TcpHtcp::m_deltaL),
                   MakeTimeChecker ())
  ;
  return tid;
}

std::string TcpHtcp::GetName () const
{
  return "TcpHtcp";
}

TcpHtcp::TcpHtcp (void)
  : TcpNewReno (),
    m_alpha (0),
    m_beta (0),
    m_delta (0),
    m_lastCon (0),
    m_minRtt (Time::Max ()),
    m_maxRtt (Time::Min ()),
    m_throughput (0),
    m_lastThroughput (0),
    m_dataSent (0)
{
  NS_LOG_FUNCTION (this);
}

TcpHtcp::TcpHtcp (const TcpHtcp& sock)
  : TcpNewReno (sock),
    m_alpha (sock.m_alpha),
    m_beta (sock.m_beta),
    m_defaultBackoff (sock.m_defaultBackoff),
    m_throughputRatio (sock.m_throughputRatio),
    m_delta (sock.m_delta),
    m_deltaL (sock.m_deltaL),
    m_lastCon (sock.m_lastCon),
    m_minRtt (sock.m_minRtt),
    m_maxRtt (sock.m_maxRtt),
    m_throughput (sock.m_throughput),
    m_lastThroughput (sock.m_lastThroughput),
    m_dataSent (sock.m_dataSent)
{
  NS_LOG_FUNCTION (this);
}

TcpHtcp::~TcpHtcp (void)
{
  NS_LOG_FUNCTION (this);
}

Ptr<TcpCongestionOps> TcpHtcp::Fork (void)
{
  NS_LOG_FUNCTION (this);
  return CopyObject<TcpHtcp> (this);
}

void TcpHtcp::CongestionAvoidance (Ptr<TcpSocketState> tcb,
                                   uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);
  if (segmentsAcked > 0)
    {
      double adder = static_cast<double> (((tcb->m_segmentSize
                                            * tcb->m_segmentSize) + (tcb->m_cWnd * m_alpha)) / tcb->m_cWnd);
      adder = std::max (1.0, adder);
      tcb->m_cWnd += static_cast<uint32_t> (adder);
      NS_LOG_INFO ("In CongAvoid, updated to cwnd " << tcb->m_cWnd
                                                    << " ssthresh " << tcb->m_ssThresh);
    }
}

void TcpHtcp::UpdateAlpha (void)
{
  NS_LOG_FUNCTION (this);

  m_delta = (Simulator::Now () - m_lastCon);
  if (m_delta <= m_deltaL)
    {
      m_alpha = 1;
    }
  else
    {
      Time diff = m_delta - m_deltaL;
      double diffSec = diff.GetSeconds ();
      // alpha=1+10(Delta-Delta_L)+[0.5(Delta-Delta_L)]^2  (seconds)
      // from Leith and Shorten H-TCP paper
      m_alpha = (1 + 10 * diffSec + 0.25 * (diffSec * diffSec));
    }
  m_alpha = 2 * (1 - m_beta) * m_alpha;
  if (m_alpha < 1)
    {
      m_alpha = 1;
    }
  NS_LOG_DEBUG ("Updated m_alpha: " << m_alpha);
}

void TcpHtcp::UpdateBeta (void)
{
  NS_LOG_FUNCTION (this);

  // Default value for m_beta
  m_beta = m_defaultBackoff;

  if (m_throughput > m_lastThroughput && m_lastThroughput > 0)
    {
      uint32_t diff = m_throughput - m_lastThroughput;
      if (diff / m_lastThroughput <= m_throughputRatio)
        {
          m_beta = m_minRtt.GetDouble () / m_maxRtt.GetDouble ();
        }
    }
  NS_LOG_DEBUG ("Updated m_beta: " << m_beta);
}

uint32_t TcpHtcp::GetSsThresh (Ptr<const TcpSocketState> tcb,
                               uint32_t bytesInFlight)
{
  NS_LOG_FUNCTION (this << tcb << bytesInFlight);

  m_lastCon = Simulator::Now ();

  UpdateBeta ();
  UpdateAlpha ();

  uint32_t segWin = 2 * tcb->m_segmentSize;
  uint32_t bFlight = static_cast<uint32_t> (bytesInFlight * m_beta);
  uint32_t ssThresh = std::max (segWin, bFlight);
  m_minRtt = Time::Max ();
  m_maxRtt = Time::Min ();
  m_lastThroughput = m_throughput;
  m_throughput = 0;
  m_dataSent = 0;
  NS_LOG_DEBUG (this << " ssThresh: " << ssThresh << " m_beta: "  << m_beta);
  return ssThresh;
}

void TcpHtcp::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                         const Time &rtt)
{

  NS_LOG_FUNCTION (this << tcb << segmentsAcked << rtt);
  NS_LOG_DEBUG ("TcpSocketState: " << tcb->m_congState);
  if (tcb->m_congState == TcpSocketState::CA_OPEN)
    {
      m_dataSent += segmentsAcked * tcb->m_segmentSize;
    }

  m_throughput = static_cast<uint32_t> (m_dataSent
                                        / (Simulator::Now ().GetSeconds () - m_lastCon.GetSeconds ()));

  UpdateAlpha ();
  if (rtt < m_minRtt)
    {
      m_minRtt = rtt;
      NS_LOG_DEBUG ("Updated m_minRtt=" << m_minRtt);
    }
  if (rtt > m_maxRtt)
    {
      m_maxRtt = rtt;
      NS_LOG_DEBUG ("Updated m_maxRtt=" << m_maxRtt);
    }
}

} // namespace ns3
