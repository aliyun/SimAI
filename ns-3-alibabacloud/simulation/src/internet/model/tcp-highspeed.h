/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello, <natale.patriciello@gmail.com>
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

#ifndef TCPHIGHSPEED_H
#define TCPHIGHSPEED_H

#include "tcp-congestion-ops.h"

namespace ns3 {

class TcpSocketState;

/**
 * \ingroup congestionOps
 *
 * \brief An implementation of TCP HighSpeed
 *
 * TCP HighSpeed is designed for high-capacity channels or, in general, for
 * TCP connections with large congestion windows.
 * Conceptually, with respect to the standard TCP, HighSpeed makes the
 * cWnd grow faster during the probing phases and accelerates the
 * cWnd recovery from losses.
 * This behavior is executed only when the window grows beyond a
 * certain threshold, which allows TCP Highspeed to be friendly with standard
 * TCP in environments with heavy congestion, without introducing new dangers
 * of congestion collapse.
 * At the core of TCP HighSpeed there are two functions, a(w) and b(w), which respectively
 * specify the cWnd growth addendum and the cWnd reduction factor when
 * given an actual cWnd value w.
 *
 * More information: http://dl.acm.org/citation.cfm?id=2756518
 */
class TcpHighSpeed : public TcpNewReno
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
  TcpHighSpeed (void);

  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpHighSpeed (const TcpHighSpeed& sock);
  virtual ~TcpHighSpeed (void);

  virtual std::string GetName () const;

  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);

  virtual Ptr<TcpCongestionOps> Fork ();

  /**
   * \brief Lookup table for the coefficient a (from RFC 3649)
   *
   * \param w Window value (in packets)
   *
   * \return the coefficient a
   */
  static uint32_t  TableLookupA (uint32_t w);

  /**
   * \brief Lookup table for the coefficient b (from RFC 3649)
   *
   * \param w Window value (in packets)
   *
   * \return the coefficient b
   */
  static double    TableLookupB (uint32_t w);

protected:
  virtual void CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

private:
  uint32_t m_ackCnt; //!< Number of received ACK, corrected with the coefficient a
};

} // namespace ns3

#endif // TCPHIGHSPEED_H
