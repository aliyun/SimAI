/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 NITK Surathkal
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
 * Author: Apoorva Bharagava <apoorvabhargava13@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#ifndef TCPLINUXRENO_H
#define TCPLINUXRENO_H

#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-state.h"

namespace ns3 {

/**
 * \ingroup congestionOps
 *
 * \brief Reno congestion control algorithm
 * 
 * This class implement the Reno congestion control type
 * and it mimics the one implemented in the Linux kernel.
 */
class TcpLinuxReno : public TcpCongestionOps
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TcpLinuxReno ();

  /**
   * \brief Copy constructor.
   * \param sock object to copy.
   */
  TcpLinuxReno (const TcpLinuxReno& sock);

  ~TcpLinuxReno ();

  std::string GetName () const;

  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);
  virtual Ptr<TcpCongestionOps> Fork ();

protected:
  /**
   * Slow start phase handler
   * \param tcb Transmission Control Block of the connection
   * \param segmentsAcked count of segments acked
   * \return Number of segments acked minus the difference between the receiver and sender Cwnd
   */
  virtual uint32_t SlowStart (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  /**
   * Congestion avoidance phase handler
   * \param tcb Transmission Control Block of the connection
   * \param segmentsAcked count of segments acked
   */
  virtual void CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

private:
  uint32_t m_cWndCnt {0}; //!< Linear increase counter
};

} // namespace ns3

#endif // TCPLINUXRENO_H
