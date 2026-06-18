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
 * Authors: Charitha Sangaraju <charitha29193@gmail.com>
 *          Nandita G <gm.nandita@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "tcp-lp.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpLp");
NS_OBJECT_ENSURE_REGISTERED (TcpLp);

TypeId
TcpLp::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpLp")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpLp> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TcpLp::TcpLp (void)
  : TcpNewReno (),
    m_flag (0),
    m_sOwd (0),
    m_owdMin (0xffffffff),
    m_owdMax (0),
    m_owdMaxRsv (0),
    m_lastDrop (Time (0)),
    m_inference (Time (0))
{
  NS_LOG_FUNCTION (this);
}

TcpLp::TcpLp (const TcpLp& sock)
  : TcpNewReno (sock),
    m_flag (sock.m_flag),
    m_sOwd (sock.m_sOwd),
    m_owdMin (sock.m_owdMin),
    m_owdMax (sock.m_owdMax),
    m_owdMaxRsv (sock.m_owdMaxRsv),
    m_lastDrop (sock.m_lastDrop),
    m_inference (sock.m_inference)
{
  NS_LOG_FUNCTION (this);
}

TcpLp::~TcpLp (void)
{
  NS_LOG_FUNCTION (this);
}

Ptr<TcpCongestionOps>
TcpLp::Fork (void)
{
  return CopyObject<TcpLp> (this);
}

void
TcpLp::CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  if (!(m_flag & LP_WITHIN_INF))
    {
      TcpNewReno::CongestionAvoidance (tcb, segmentsAcked);
    }
}

uint32_t
TcpLp::OwdCalculator (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);

  int64_t owd = 0;

  owd = tcb->m_rcvTimestampValue - tcb->m_rcvTimestampEchoReply;

  if (owd < 0)
    {
      owd = -owd;
    }
  if (owd > 0)
    {
      m_flag |= LP_VALID_OWD;
    }
  else
    {
      m_flag &= ~LP_VALID_OWD;
    }
  return owd;
}

void
TcpLp::RttSample (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb );

  uint32_t mowd = OwdCalculator (tcb);

  if (!(m_flag & LP_VALID_OWD))
    {
      return;
    }

  /* record the next minimum owd */
  if (mowd < m_owdMin)
    {
      m_owdMin = mowd;
    }

  if (mowd > m_owdMax)
    {
      if (mowd > m_owdMaxRsv)
        {
          if (m_owdMaxRsv == 0)
            {
              m_owdMax = mowd;
            }
          else
            {
              m_owdMax = m_owdMaxRsv;
            }
          m_owdMaxRsv = mowd;
        }
      else
        {
          m_owdMax = mowd;
        }
    }

  /* Calculation for Smoothed Owd */
  if (m_sOwd != 0)
    {
      mowd -= m_sOwd >> 3;
      m_sOwd += mowd;           /* owd = 7/8 owd + 1/8 new owd */
    }
  else
    {
      m_sOwd = mowd << 3;       /* owd = 1/8 new owd */
    }
}

void
TcpLp::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                  const Time &rtt)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked << rtt);

  if (!rtt.IsZero ())
    {
      RttSample (tcb);
    }

  Time timestamp = Simulator::Now ();
  /* Calculation of inference time */
  if (timestamp.GetMilliSeconds () > tcb->m_rcvTimestampEchoReply)
    {
      m_inference = 3 * (timestamp - MilliSeconds (tcb->m_rcvTimestampEchoReply));
    }

  /* Test if within inference */
  if (!m_lastDrop.IsZero () && (timestamp - m_lastDrop < m_inference))
    {
      m_flag |= LP_WITHIN_INF;
    }
  else
    {
      m_flag &= ~LP_WITHIN_INF;
    }

  /* Test if within threshold */
  if (m_sOwd >> 3 <=
      m_owdMin + 15 * (m_owdMax - m_owdMin) / 100)
    {
      m_flag |= LP_WITHIN_THR;
    }
  else
    {
      m_flag &= ~LP_WITHIN_THR;
    }

  if (m_flag & LP_WITHIN_THR)
    {
      return;
    }

  m_owdMin = m_sOwd >> 3;
  m_owdMax = m_sOwd >> 2;
  m_owdMaxRsv = m_sOwd >> 2;

  /* happened within inference
   * drop congestion window to 1 */
  if (m_flag & LP_WITHIN_INF)
    {
      tcb->m_cWnd = 1U * tcb->m_segmentSize;
    }

  /* happened after inference
   * cut congestion window to half */
  else
    {
      tcb->m_cWnd = std::max (tcb->m_cWnd.Get () >> 1U, 1U * tcb->m_segmentSize);
    }

  /* record this time of reduction of cwnd */
  m_lastDrop = timestamp;
}

std::string
TcpLp::GetName () const
{
  return "TcpLp";
}
} // namespace ns3
