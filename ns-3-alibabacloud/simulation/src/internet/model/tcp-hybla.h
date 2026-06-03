/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
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
 */
#ifndef TCPHYBLA_H
#define TCPHYBLA_H

#include "tcp-congestion-ops.h"
#include "ns3/traced-value.h"

namespace ns3 {

class TcpSocketState;

/**
 * \ingroup congestionOps
 *
 * \brief Implementation of the TCP Hybla algorithm
 *
 * The key idea behind TCP Hybla is to obtain for long RTT connections the same
 * instantaneous transmission rate of a reference TCP connection with lower RTT.
 * With analytical steps, it is shown that this goal can be achieved by
 * modifying the time scale, in order for the throughput to be independent from
 * the RTT. This independence is obtained through the use of a coefficient rho.
 *
 * This coefficient is used to calculate both the slow start threshold
 * and the congestion window when in slow start and in congestion avoidance,
 * respectively.
 *
 * More information: http://dl.acm.org/citation.cfm?id=2756518
 */
class TcpHybla : public TcpNewReno
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Create an unbound tcp socket.
   */
  TcpHybla (void);

  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpHybla (const TcpHybla& sock);

  virtual ~TcpHybla (void) override;

  // Inherited
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                          const Time& rtt) override;
  virtual std::string GetName () const override;
  virtual Ptr<TcpCongestionOps> Fork () override;

protected:
  virtual uint32_t SlowStart (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
  virtual void CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;

private:
  TracedValue<double> m_rho;         //!< Rho parameter
  Time                m_rRtt;        //!< Reference RTT
  double              m_cWndCnt;     //!< cWnd integer-to-float counter

private:
  /**
   * \brief Recalculate algorithm parameters
   * \param tcb the socket state.
   */
  void RecalcParam (const Ptr<TcpSocketState> &tcb);
};

} // namespace ns3

#endif // TCPHYBLA_H
