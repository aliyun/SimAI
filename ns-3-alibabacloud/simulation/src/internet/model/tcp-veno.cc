/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
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
 * Author: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-veno.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpVeno");
NS_OBJECT_ENSURE_REGISTERED (TcpVeno);

TypeId
TcpVeno::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpVeno")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpVeno> ()
    .SetGroupName ("Internet")
    .AddAttribute ("Beta", "Threshold for congestion detection",
                   UintegerValue (3),
                   MakeUintegerAccessor (&TcpVeno::m_beta),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

TcpVeno::TcpVeno (void)
  : TcpNewReno (),
    m_baseRtt (Time::Max ()),
    m_minRtt (Time::Max ()),
    m_cntRtt (0),
    m_doingVenoNow (true),
    m_diff (0),
    m_inc (true),
    m_ackCnt (0),
    m_beta (6)
{
  NS_LOG_FUNCTION (this);
}

TcpVeno::TcpVeno (const TcpVeno& sock)
  : TcpNewReno (sock),
    m_baseRtt (sock.m_baseRtt),
    m_minRtt (sock.m_minRtt),
    m_cntRtt (sock.m_cntRtt),
    m_doingVenoNow (true),
    m_diff (0),
    m_inc (true),
    m_ackCnt (sock.m_ackCnt),
    m_beta (sock.m_beta)
{
  NS_LOG_FUNCTION (this);
}

TcpVeno::~TcpVeno (void)
{
  NS_LOG_FUNCTION (this);
}

Ptr<TcpCongestionOps>
TcpVeno::Fork (void)
{
  return CopyObject<TcpVeno> (this);
}

void
TcpVeno::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                    const Time& rtt)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked << rtt);

  if (rtt.IsZero ())
    {
      return;
    }

  m_minRtt = std::min (m_minRtt, rtt);
  NS_LOG_DEBUG ("Updated m_minRtt= " << m_minRtt);


  m_baseRtt = std::min (m_baseRtt, rtt);
  NS_LOG_DEBUG ("Updated m_baseRtt= " << m_baseRtt);

  // Update RTT counter
  m_cntRtt++;
  NS_LOG_DEBUG ("Updated m_cntRtt= " << m_cntRtt);
}

void
TcpVeno::EnableVeno ()
{
  NS_LOG_FUNCTION (this);

  m_doingVenoNow = true;
  m_minRtt = Time::Max ();
}

void
TcpVeno::DisableVeno ()
{
  NS_LOG_FUNCTION (this);

  m_doingVenoNow = false;
}

void
TcpVeno::CongestionStateSet (Ptr<TcpSocketState> tcb,
                             const TcpSocketState::TcpCongState_t newState)
{
  NS_LOG_FUNCTION (this << tcb << newState);
  if (newState == TcpSocketState::CA_OPEN)
    {
      EnableVeno ();
      NS_LOG_LOGIC ("Veno is now on.");
    }
  else
    {
      DisableVeno ();
      NS_LOG_LOGIC ("Veno is turned off.");
    }
}

void
TcpVeno::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  // Always calculate m_diff, even if we are not doing Veno now
  uint32_t targetCwnd;
  uint32_t segCwnd = tcb->GetCwndInSegments ();

  /*
   * Calculate the cwnd we should have. baseRtt is the minimum RTT
   * per-connection, minRtt is the minimum RTT in this window
   *
   * little trick:
   * desidered throughput is currentCwnd * baseRtt
   * target cwnd is throughput / minRtt
   */
  double tmp = m_baseRtt.GetSeconds () / m_minRtt.GetSeconds ();
  targetCwnd = static_cast<uint32_t> (segCwnd * tmp);
  NS_LOG_DEBUG ("Calculated targetCwnd = " << targetCwnd);
  NS_ASSERT (segCwnd >= targetCwnd); // implies baseRtt <= minRtt

  // Calculate the difference between actual and target cwnd
  m_diff = segCwnd - targetCwnd;
  NS_LOG_DEBUG ("Calculated m_diff = " << m_diff);

  if (!m_doingVenoNow)
    {
      // If Veno is not on, we follow NewReno algorithm
      NS_LOG_LOGIC ("Veno is not turned on, we follow NewReno algorithm.");
      TcpNewReno::IncreaseWindow (tcb, segmentsAcked);
      return;
    }

  // We do the Veno calculations only if we got enough RTT samples
  if (m_cntRtt <= 2)
    {    // We do not have enough RTT samples, so we should behave like NewReno
      NS_LOG_LOGIC ("We do not have enough RTT samples to perform Veno "
                    "calculations, we behave like NewReno.");
      TcpNewReno::IncreaseWindow (tcb, segmentsAcked);
    }
  else
    {
      NS_LOG_LOGIC ("We have enough RTT samples to perform Veno calculations.");

      if (tcb->m_cWnd < tcb->m_ssThresh)
        { // Slow start mode. Veno employs same slow start algorithm as NewReno's.
          NS_LOG_LOGIC ("We are in slow start, behave like NewReno.");
          TcpNewReno::SlowStart (tcb, segmentsAcked);
        }
      else
        { // Congestion avoidance mode
          NS_LOG_LOGIC ("We are in congestion avoidance, execute Veno additive "
                        "increase algo.");

          if (m_diff < m_beta)
            {
              // Available bandwidth is not fully utilized,
              // increase cwnd by 1 every RTT
              NS_LOG_LOGIC ("Available bandwidth not fully utilized, increase "
                            "cwnd by 1 every RTT");
              TcpNewReno::CongestionAvoidance (tcb, segmentsAcked);
            }
          else
            {
              // Available bandwidth is fully utilized,
              // increase cwnd by 1 every other RTT
              NS_LOG_LOGIC ("Available bandwidth fully utilized, increase cwnd "
                            "by 1 every other RTT");
              if (m_inc)
                {
                  TcpNewReno::CongestionAvoidance (tcb, segmentsAcked);
                  m_inc = false;
                }
              else
                {
                  m_inc = true;
                }
            }
        }
    }

  // Reset cntRtt & minRtt every RTT
  m_cntRtt = 0;
  m_minRtt = Time::Max ();
}

std::string
TcpVeno::GetName () const
{
  return "TcpVeno";
}

uint32_t
TcpVeno::GetSsThresh (Ptr<const TcpSocketState> tcb,
                      uint32_t bytesInFlight)
{
  NS_LOG_FUNCTION (this << tcb << bytesInFlight);

  if (m_diff < m_beta)
    {
      // random loss due to bit errors is most likely to have occurred,
      // we cut cwnd by 1/5
      NS_LOG_LOGIC ("Random loss is most likely to have occurred, "
                    "cwnd is reduced by 1/5");
      static double tmp = 4.0/5.0;
      return std::max (static_cast<uint32_t> (bytesInFlight * tmp),
                       2 * tcb->m_segmentSize);
    }
  else
    {
      // congestion-based loss is most likely to have occurred,
      // we reduce cwnd by 1/2 as in NewReno
      NS_LOG_LOGIC ("Congestive loss is most likely to have occurred, "
                    "cwnd is halved");
      return TcpNewReno::GetSsThresh (tcb, bytesInFlight);
    }
}

} // namespace ns3
