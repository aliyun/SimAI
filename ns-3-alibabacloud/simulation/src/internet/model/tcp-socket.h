/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 Georgia Tech Research Corporation
 *               2007 INRIA
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
 * Authors: George F. Riley<riley@ece.gatech.edu>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef TCP_SOCKET_H
#define TCP_SOCKET_H

#include "ns3/socket.h"
#include "ns3/traced-callback.h"
#include "ns3/callback.h"
#include "ns3/ptr.h"
#include "ns3/object.h"
#include "ns3/nstime.h"

namespace ns3 {

class Node;
class Packet;

/**
 * \ingroup socket
 * \ingroup tcp
 *
 * \brief (abstract) base class of all TcpSockets
 *
 * This class exists solely for hosting TcpSocket attributes that can
 * be reused across different implementations.
 */
class TcpSocket : public Socket
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
 
  TcpSocket (void);
  virtual ~TcpSocket (void);

  /**
   * \ingroup tcp
   * \brief Names of the 11 TCP states
   *
   */
  typedef enum {
    CLOSED = 0,   /**< Socket is finished                                     */
    LISTEN,       /**< Listening for a connection                             */
    SYN_SENT,     /**< Sent a connection request, waiting for ack             */
    SYN_RCVD,     /**< Received a connection request, sent ack,
                    *  waiting for final ack in three-way handshake.          */
    ESTABLISHED,  /**< Connection established                                 */
    CLOSE_WAIT,   /**< Remote side has shutdown and is waiting for
                    *  us to finish writing our data and to shutdown
                    *  (we have to close() to move on to LAST_ACK)            */
    LAST_ACK,     /**< Our side has shutdown after remote has
                    *  shutdown.  There may still be data in our
                    *  buffer that we have to finish sending                  */
    FIN_WAIT_1,   /**< Our side has shutdown, waiting to complete
                    *  transmission of remaining buffered data                */
    FIN_WAIT_2,   /**< All buffered data sent, waiting for remote to shutdown */
    CLOSING,      /**< Both sides have shutdown but we still have
                    *  data we have to finish sending                         */
    TIME_WAIT,    /**< Timeout to catch resent junk before entering
                    *  closed, can only be entered from FIN_WAIT2
                    *  or CLOSING.  Required because the other end
                    *  may not have gotten our last ACK causing it
                    *  to retransmit the data packet (which we ignore)        */
    LAST_STATE    /**< Last state, used only in debug messages                */
  } TcpStates_t;

  /**
   * \brief Literal names of TCP states for use in log messages
   */
  static const char* const TcpStateName[TcpSocket::LAST_STATE];

private:
  // Indirect the attribute setting and getting through private virtual methods

  /**
   * \brief Set the send buffer size.
   * \param size the buffer size (in bytes)
   */
  virtual void SetSndBufSize (uint32_t size) = 0;

  /**
   * \brief Get the send buffer size.
   * \returns the buffer size (in bytes)
   */
  virtual uint32_t GetSndBufSize (void) const = 0;

  /**
   * \brief Set the receive buffer size.
   * \param size the buffer size (in bytes)
   */
  virtual void SetRcvBufSize (uint32_t size) = 0;

  /**
   * \brief Get the receive buffer size.
   * \returns the buffer size (in bytes)
   */
  virtual uint32_t GetRcvBufSize (void) const = 0;

  /**
   * \brief Set the segment size.
   * \param size the segment size (in bytes)
   */
  virtual void SetSegSize (uint32_t size) = 0;

  /**
   * \brief Get the segment size.
   * \returns the segment size (in bytes)
   */
  virtual uint32_t GetSegSize (void) const = 0;

  /**
   * \brief Set the initial Slow Start Threshold.
   * \param threshold the Slow Start Threshold (in bytes)
   */
  virtual void SetInitialSSThresh (uint32_t threshold) = 0;

  /**
   * \brief Get the initial Slow Start Threshold.
   * \returns the Slow Start Threshold (in bytes)
   */
  virtual uint32_t GetInitialSSThresh (void) const = 0;

  /**
   * \brief Set the initial Congestion Window.
   * \param cwnd the initial congestion window (in segments)
   */
  virtual void SetInitialCwnd (uint32_t cwnd) = 0;

  /**
   * \brief Get the initial Congestion Window.
   * \returns the initial congestion window (in segments)
   */
  virtual uint32_t GetInitialCwnd (void) const = 0;

  /**
   * \brief Set the connection timeout.
   * \param timeout the connection timeout
   */
  virtual void SetConnTimeout (Time timeout) = 0;

  /**
   * \brief Get the connection timeout.
   * \returns the connection timeout
   */
  virtual Time GetConnTimeout (void) const = 0;

  /**
   * \brief Set the number of connection retries before giving up.
   * \param count the number of connection retries
   */
  virtual void SetSynRetries (uint32_t count) = 0;

  /**
   * \brief Get the number of connection retries before giving up.
   * \returns the number of connection retries
   */
  virtual uint32_t GetSynRetries (void) const = 0;

  /**
   * \brief Set the number of data transmission retries before giving up.
   * \param retries the number of data transmission retries
   */
  virtual void SetDataRetries (uint32_t retries) = 0;

  /**
   * \brief Get the number of data transmission retries before giving up.
   * \returns the number of data transmission retries
   */
  virtual uint32_t GetDataRetries (void) const = 0;

  /**
   * \brief Set the time to delay an ACK.
   * \param timeout the time to delay an ACK
   */
  virtual void SetDelAckTimeout (Time timeout) = 0;

  /**
   * \brief Get the time to delay an ACK.
   * \returns the time to delay an ACK
   */
  virtual Time GetDelAckTimeout (void) const = 0;

  /**
   * \brief Set the number of packet to fire an ACK before delay timeout.
   * \param count the umber of packet to fire an ACK before delay timeout
   */
  virtual void SetDelAckMaxCount (uint32_t count) = 0;

  /**
   * \brief Get the number of packet to fire an ACK before delay timeout.
   * \returns the number of packet to fire an ACK before delay timeout
   */
  virtual uint32_t GetDelAckMaxCount (void) const = 0;

  /**
   * \brief Enable/Disable Nagle's algorithm.
   * \param noDelay true to DISABLE Nagle's algorithm
   */
  virtual void SetTcpNoDelay (bool noDelay) = 0;

  /**
   * \brief Check if Nagle's algorithm is enabled or not.
   * \returns true if Nagle's algorithm is DISABLED
   */
  virtual bool GetTcpNoDelay (void) const = 0;

  /**
   * \brief Set the timeout for persistent connection
   *
   * When the timeout expires, send 1-byte data to probe for the window
   * size at the receiver when the local knowledge tells that the
   * receiver has zero window size
   *
   * \param timeout the persistent timeout
   */
  virtual void SetPersistTimeout (Time timeout) = 0;

  /**
   * \brief Get the timeout for persistent connection
   *
   * When the timeout expires, send 1-byte data to probe for the window
   * size at the receiver when the local knowledge tells that the
   * receiver has zero window size
   *
   * \returns the persistent timeout
   */
  virtual Time GetPersistTimeout (void) const = 0;

};

/**
 * \ingroup tcp
 * TracedValue Callback signature for TcpStates_t
 *
 * \param [in] oldValue original value of the traced variable
 * \param [in] newValue new value of the traced variable
 */
typedef void (* TcpStatesTracedValueCallback)(const TcpSocket::TcpStates_t oldValue,
                                              const TcpSocket::TcpStates_t newValue);

} // namespace ns3

#endif /* TCP_SOCKET_H */


