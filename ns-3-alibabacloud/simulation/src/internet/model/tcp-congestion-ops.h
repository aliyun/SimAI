/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
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
#ifndef TCPCONGESTIONOPS_H
#define TCPCONGESTIONOPS_H

#include "tcp-rate-ops.h"
#include "tcp-socket-state.h"

namespace ns3 {

/**
 * \ingroup tcp
 * \defgroup congestionOps Congestion Control Algorithms.
 *
 * The various congestion control algorithms, also known as "TCP flavors".
 */

/**
 * \ingroup congestionOps
 *
 * \brief Congestion control abstract class
 *
 * The design is inspired by what Linux v4.0 does (but it has been
 * in place for years). The congestion control is split from the main
 * socket code, and it is a pluggable component. An interface has been defined;
 * variables are maintained in the TcpSocketState class, while subclasses of
 * TcpCongestionOps operate over an instance of that class.
 *
 * Only three methods have been implemented right now; however, Linux has many others,
 * which can be added later in ns-3.
 *
 * \see IncreaseWindow
 * \see PktsAcked
 */
class TcpCongestionOps : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TcpCongestionOps ();

  /**
   * \brief Copy constructor.
   * \param other object to copy.
   */
  TcpCongestionOps (const TcpCongestionOps &other);

  virtual ~TcpCongestionOps ();

  /**
   * \brief Get the name of the congestion control algorithm
   *
   * \return A string identifying the name
   */
  virtual std::string GetName () const = 0;

  /**
   * \brief Set configuration required by congestion control algorithm
   *
   * \param tcb internal congestion state
   */
  virtual void Init ([[maybe_unused]] Ptr<TcpSocketState> tcb)
    {
    }

  /**
   * \brief Get the slow start threshold after a loss event
   *
   * Is guaranteed that the congestion control state (\p TcpAckState_t) is
   * changed BEFORE the invocation of this method.
   * The implementator should return the slow start threshold (and not change
   * it directly) because, in the future, the TCP implementation may require to
   * instantly recover from a loss event (e.g. when there is a network with an high
   * reordering factor).
   *
   * \param tcb internal congestion state
   * \param bytesInFlight total bytes in flight
   * \return Slow start threshold
   */
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight) = 0;

  /**
   * \brief Congestion avoidance algorithm implementation
   *
   * Mimic the function \pname{cong_avoid} in Linux. New segments have been ACKed,
   * and the congestion control duty is to update the window.
   *
   * The function is allowed to change directly cWnd and/or ssThresh.
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments acked
   */
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

  /**
   * \brief Timing information on received ACK
   *
   * The function is called every time an ACK is received (only one time
   * also for cumulative ACKs) and contains timing information. It is
   * optional (congestion controls need not implement it) and the default
   * implementation does nothing.
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments acked
   * \param rtt last rtt
   */
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                          const Time& rtt);

  /**
   * \brief Trigger events/calculations specific to a congestion state
   *
   * This function mimics the notification function \pname{set_state} in Linux.
   * The function does not change the congestion state in the tcb; it notifies
   * the congestion control algorithm that this state is about to be changed.
   * The tcb->m_congState variable must be separately set; for example:
   *
   * \code
   *   m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_RECOVERY);
   *   m_tcb->m_congState = TcpSocketState::CA_RECOVERY;
   * \endcode
   *
   * \param tcb internal congestion state
   * \param newState new congestion state to which the TCP is going to switch
   */
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState);

  /**
   * \brief Trigger events/calculations on occurrence of congestion window event
   *
   * This function mimics the function \pname{cwnd_event} in Linux.
   * The function is called in case of congestion window events.
   *
   * \param tcb internal congestion state
   * \param event the event which triggered this function
   */
  virtual void CwndEvent (Ptr<TcpSocketState> tcb,
                          const TcpSocketState::TcpCAEvent_t event);

  /**
   * \brief Returns true when Congestion Control Algorithm implements CongControl
   *
   * \return true if CC implements CongControl function
   *
   * This function is the equivalent in C++ of the C checks that are used
   * in the Linux kernel to see if an optional function has been defined.
   * Since CongControl is optional, not all congestion controls have it. But,
   * from the perspective of TcpSocketBase, the behavior is different if
   * CongControl is present. Therefore, this check should return true for any
   * congestion controls that implements the CongControl optional function.
   */
  virtual bool HasCongControl () const;

  /**
   * \brief Called when packets are delivered to update cwnd and pacing rate
   *
   * This function mimics the function cong_control in Linux. It is allowed to
   * change directly cWnd and pacing rate.
   *
   * \param tcb internal congestion state
   * \param rc Rate information for the connection
   * \param rs Rate sample (over a period of time) information
   */
  virtual void CongControl (Ptr<TcpSocketState> tcb,
                            const TcpRateOps::TcpRateConnection &rc,
                            const TcpRateOps::TcpRateSample &rs);

  // Present in Linux but not in ns-3 yet:
  /* call when ack arrives (optional) */
  //     void (*in_ack_event)(struct sock *sk, u32 flags);
  /* new value of cwnd after loss (optional) */
  //     u32  (*undo_cwnd)(struct sock *sk);
  /* hook for packet ack accounting (optional) */
  //     void (*pkts_acked)(struct sock *sk, u32 ext, int *attr, union tcp_cc_info *info);

  /**
   * \brief Copy the congestion control algorithm across sockets
   *
   * \return a pointer of the copied object
   */
  virtual Ptr<TcpCongestionOps> Fork () = 0;
};

/**
 * \brief The NewReno implementation
 *
 * New Reno introduces partial ACKs inside the well-established Reno algorithm.
 * This and other modifications are described in RFC 6582.
 *
 * \see IncreaseWindow
 */
class TcpNewReno : public TcpCongestionOps
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TcpNewReno ();

  /**
   * \brief Copy constructor.
   * \param sock object to copy.
   */
  TcpNewReno (const TcpNewReno& sock);

  ~TcpNewReno ();

  std::string GetName () const;

  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);
  virtual Ptr<TcpCongestionOps> Fork ();

protected:
  virtual uint32_t SlowStart (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  virtual void CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
};

} // namespace ns3

#endif // TCPCONGESTIONOPS_H
