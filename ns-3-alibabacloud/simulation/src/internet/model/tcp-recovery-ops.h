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
 * Author: Viyom Mittal <viyommittal@gmail.com>
 *         Vivek Jain <jain.vivek.anand@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */
#ifndef TCP_RECOVERY_OPS_H
#define TCP_RECOVERY_OPS_H

#include "ns3/object.h"

namespace ns3 {

class TcpSocketState;

/**
 * \ingroup tcp
 * \defgroup recoveryOps Recovery Algorithms.
 *
 * The various recovery algorithms used in recovery phase of TCP. The interface
 * is defined in class TcpRecoveryOps.
 */

/**
 * \ingroup recoveryOps
 *
 * \brief recovery abstract class
 *
 * The design is inspired by the TcpCongestionOps class in ns-3. The fast
 * recovery is split from the main socket code, and it is a pluggable
 * component. Subclasses of TcpRecoveryOps should modify TcpSocketState variables
 * upon three condition:
 *
 * - EnterRecovery (when the first loss is guessed)
 * - DoRecovery (each time a duplicate ACK or an ACK with SACK information is received)
 * - ExitRecovery (when the sequence transmitted when the socket entered the
 * Recovery phase is ACKed, therefore ending phase).
 *
 * Each condition is represented by a pure virtual method.
 *
 * \see TcpClassicRecovery
 * \see DoRecovery
 */
class TcpRecoveryOps : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Constructor
   */
  TcpRecoveryOps ();

  /**
   * \brief Copy constructor.
   * \param other object to copy.
   */
  TcpRecoveryOps (const TcpRecoveryOps &other);

  /**
   * \brief Deconstructor
   */
  virtual ~TcpRecoveryOps ();

  /**
   * \brief Get the name of the recovery algorithm
   *
   * \return A string identifying the name
   */
  virtual std::string GetName () const = 0;

  /**
   * \brief Performs variable initialization at the start of recovery
   *
   * The function is called when the TcpSocketState is changed to CA_RECOVERY.
   *
   * \param tcb internal congestion state
   * \param dupAckCount duplicate acknowldgement count
   * \param unAckDataCount total bytes of data unacknowledged
   * \param deliveredBytes bytes (S)ACKed in the last (S)ACK
   */
  virtual void EnterRecovery (Ptr<TcpSocketState> tcb, uint32_t dupAckCount,
                              uint32_t unAckDataCount, uint32_t deliveredBytes) = 0;

  /**
   * \brief Performs recovery based on the recovery algorithm
   *
   * The function is called on arrival of every ack when TcpSocketState
   * is set to CA_RECOVERY. It performs the necessary cwnd changes
   * as per the recovery algorithm.
   *
   * \param tcb internal congestion state
   * \param deliveredBytes bytes (S)ACKed in the last (S)ACK
   */
  virtual void DoRecovery (Ptr<TcpSocketState> tcb, uint32_t deliveredBytes) = 0;

  /**
   * \brief Performs cwnd adjustments at the end of recovery
   *
   * The function is called when the TcpSocketState is changed from CA_RECOVERY.
   *
   * \param tcb internal congestion state
   */
  virtual void ExitRecovery (Ptr<TcpSocketState> tcb) = 0;

  /**
   * \brief Keeps track of bytes sent during recovery phase
   *
   * The function is called whenever a data packet is sent during recovery phase
   * (optional).
   *
   * \param bytesSent bytes sent
   */
  virtual void UpdateBytesSent (uint32_t bytesSent);

  /**
   * \brief Copy the recovery algorithm across socket
   *
   * \return a pointer of the copied object
   */
  virtual Ptr<TcpRecoveryOps> Fork () = 0;
};

/**
 * \brief The Classic recovery implementation
 *
 * Classic recovery refers to the two well-established recovery algorithms,
 * namely, NewReno (RFC 6582) and SACK based recovery (RFC 6675).
 *
 * The idea of the algorithm is that when we enter recovery, we set the
 * congestion window value to the slow start threshold and maintain it
 * at such value until we are fully recovered (in other words, until
 * the highest sequence transmitted at time of detecting the loss is
 * ACKed by the receiver).
 *
 * \see DoRecovery
 */
class TcpClassicRecovery : public TcpRecoveryOps
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Constructor
   */
  TcpClassicRecovery ();

  /**
   * \brief Copy constructor.
   * \param recovery object to copy.
   */
  TcpClassicRecovery (const TcpClassicRecovery& recovery);

  /**
   * \brief Constructor
   */
  virtual ~TcpClassicRecovery () override;

  virtual std::string GetName () const override;

  virtual void EnterRecovery (Ptr<TcpSocketState> tcb, uint32_t dupAckCount,
                              uint32_t unAckDataCount, uint32_t deliveredBytes) override;

  virtual void DoRecovery (Ptr<TcpSocketState> tcb, uint32_t deliveredBytes) override;

  virtual void ExitRecovery (Ptr<TcpSocketState> tcb) override;

  virtual Ptr<TcpRecoveryOps> Fork () override;
};

} // namespace ns3

#endif /* TCP_RECOVERY_OPS_H */
