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

#ifndef NS3_SOCKET_H
#define NS3_SOCKET_H

#include "ns3/callback.h"
#include "ns3/ptr.h"
#include "ns3/tag.h"
#include "ns3/object.h"
#include "ns3/net-device.h"
#include "address.h"
#include <stdint.h>
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"

namespace ns3 {


class Node;
class Packet;

/**
 * \ingroup network
 * \defgroup socket Socket
 */

/**
 * \brief A low-level Socket API based loosely on the BSD Socket API.
 * \ingroup socket
 *
 * A few things to keep in mind about this type of socket:
 * - it uses ns-3 API constructs such as class ns3::Address instead of
 *   C-style structs
 * - in contrast to the original BSD socket API, this API is asynchronous:
 *   it does not contain blocking calls.  Sending and receiving operations
 *   must make use of the callbacks provided. 
 * - It also uses class ns3::Packet as a fancy byte buffer, allowing 
 *   data to be passed across the API using an ns-3 Packet instead of 
 *   a raw data pointer.
 * - Not all of the full POSIX sockets API is supported
 *
 * Other than that, it tries to stick to the BSD API to make it 
 * easier for those who know the BSD API to use this API.
 * More details are provided in the ns-3 tutorial.
 */
class Socket : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  Socket (void);
  virtual ~Socket (void);

  /**
   * \enum SocketErrno
   * \brief Enumeration of the possible errors returned by a socket.
   */
  enum SocketErrno {
    ERROR_NOTERROR,
    ERROR_ISCONN,
    ERROR_NOTCONN,
    ERROR_MSGSIZE,
    ERROR_AGAIN,
    ERROR_SHUTDOWN,
    ERROR_OPNOTSUPP,
    ERROR_AFNOSUPPORT,
    ERROR_INVAL,
    ERROR_BADF,
    ERROR_NOROUTETOHOST,
    ERROR_NODEV,
    ERROR_ADDRNOTAVAIL,
    ERROR_ADDRINUSE,
    SOCKET_ERRNO_LAST
  };

  /**
   * \enum SocketType
   * \brief Enumeration of the possible socket types.
   */
  enum SocketType {
    NS3_SOCK_STREAM,
    NS3_SOCK_SEQPACKET,
    NS3_SOCK_DGRAM,
    NS3_SOCK_RAW
  };

  /**
   * \enum SocketPriority
   * \brief Enumeration of the possible socket priorities.
   *
   * Names and corresponding values are derived from
   * the Linux TC_PRIO_* macros
   */
  enum SocketPriority {
    NS3_PRIO_BESTEFFORT = 0,
    NS3_PRIO_FILLER = 1,
    NS3_PRIO_BULK = 2,
    NS3_PRIO_INTERACTIVE_BULK = 4,
    NS3_PRIO_INTERACTIVE = 6,
    NS3_PRIO_CONTROL = 7
  };

  /**
   * \enum Ipv6MulticastFilterMode
   * \brief Enumeration of the possible filter of a socket.
   *
   * A socket can have filters on specific sources to include only
   * packets incoming from them, or to exclude packets incoming
   * from specific sources.
   * Moreover, inclusion and exclusion also works as a leave,
   * since "joining" a group without allowed sources is equivalent
   * to leaving it.
   */
  enum Ipv6MulticastFilterMode
  {
    INCLUDE=1,
    EXCLUDE
  };

  /**
   * This method wraps the creation of sockets that is performed
   * on a given node by a SocketFactory specified by TypeId.
   * 
   * \return A smart pointer to a newly created socket.
   * 
   * \param node The node on which to create the socket
   * \param tid The TypeId of a SocketFactory class to use
   */
  static Ptr<Socket> CreateSocket (Ptr<Node> node, TypeId tid);
  /**
   * \brief Get last error number.
   *
   * \return the errno associated to the last call which failed in this
   *         socket. Each socket's errno is initialized to zero
   *         when the socket is created.
   */
  virtual enum Socket::SocketErrno GetErrno (void) const = 0;
  /**
    * \return the socket type, analogous to getsockopt (SO_TYPE)
    */
  virtual enum Socket::SocketType GetSocketType (void) const = 0;
  /**
   * \brief Return the node this socket is associated with.
   * \returns the node
   */
  virtual Ptr<Node> GetNode (void) const = 0;
  /**
   * \brief Specify callbacks to allow the caller to determine if
   * the connection succeeds of fails.
   * \param connectionSucceeded this callback is invoked when the 
   *        connection request initiated by the user is successfully 
   *        completed. The callback is passed  back a pointer to 
   *        the same socket object.
   * \param connectionFailed this callback is invoked when the 
   *        connection request initiated by the user is unsuccessfully 
   *        completed. The callback is passed back a pointer to the 
   *        same socket object. 
   */
  void SetConnectCallback (Callback<void, Ptr<Socket> > connectionSucceeded,
                           Callback<void,  Ptr<Socket> > connectionFailed);
  /**
   * \brief Detect socket recv() events such as graceful shutdown or error.
   *
   * For connection-oriented sockets, the first callback is used to signal
   * that the remote side has gracefully shut down the connection, and the
   * second callback denotes an error corresponding to cases in which
   * a traditional recv() socket call might return -1 (error), such
   * as a connection reset.  For datagram sockets, these callbacks may
   * never be invoked.
   *
   * \param normalClose this callback is invoked when the
   *        peer closes the connection gracefully
   * \param errorClose this callback is invoked when the
   *        connection closes abnormally
   */
  void SetCloseCallbacks (Callback<void, Ptr<Socket> > normalClose,
                          Callback<void, Ptr<Socket> > errorClose);
  /**
   * \brief Accept connection requests from remote hosts
   * \param connectionRequest Callback for connection request from peer. 
   *        This user callback is passed a pointer to this socket, the 
   *        ip address and the port number of the connection originator. 
   *        This callback must return true to accept the incoming connection,
   *        false otherwise. If the connection is accepted, the 
   *        "newConnectionCreated" callback will be invoked later to 
   *        give access to the user to the socket created to match 
   *        this new connection. If the user does not explicitly 
   *        specify this callback, all incoming  connections will be refused.
   * \param newConnectionCreated Callback for new connection: when a new
   *        is accepted, it is created and the corresponding socket is passed
   *        back to the user through this callback. This user callback is 
   *        passed a pointer to the new socket, and the ip address and 
   *        port number of the connection originator.
   */
  void SetAcceptCallback (Callback<bool, Ptr<Socket>, 
                                   const Address &> connectionRequest,
                          Callback<void, Ptr<Socket>, 
                                   const Address&> newConnectionCreated);
  /**
   * \brief Notify application when a packet has been sent from transport 
   *        protocol (non-standard socket call)
   * \param dataSent Callback for the event that data is sent from the
   *        underlying transport protocol.  This callback is passed a
   *        pointer to the socket, and the number of bytes sent.
   */
  void SetDataSentCallback (Callback<void, Ptr<Socket>, 
                                     uint32_t> dataSent);
  /**
   * \brief Notify application when space in transmit buffer is added
   *
   *        This callback is intended to notify a 
   *        socket that would have been blocked in a blocking socket model
   *        that space is available in the transmit buffer and that it
   *        can call Send() again.
   *
   * \param sendCb Callback for the event that the socket transmit buffer
   *        fill level has decreased.  This callback is passed a pointer to
   *        the socket, and the number of bytes available for writing
   *        into the buffer (an absolute value).  If there is no transmit
   *        buffer limit, a maximum-sized integer is always returned.
   */
  void SetSendCallback (Callback<void, Ptr<Socket>, uint32_t> sendCb);
  /**
   * \brief Notify application when new data is available to be read.
   *
   *        This callback is intended to notify a socket that would
   *        have been blocked in a blocking socket model that data
   *        is available to be read.
   * \param receivedData Callback for the event that data is received
   *        from the underlying transport protocol.  This callback
   *        is passed a pointer to the socket.
   */
  void SetRecvCallback (Callback<void, Ptr<Socket> > receivedData);
  /** 
   * \brief Allocate a local endpoint for this socket.
   * \param address the address to try to allocate
   * \returns 0 on success, -1 on failure.
   */
  virtual int Bind (const Address &address) = 0;

  /** 
   * \brief Allocate a local IPv4 endpoint for this socket.
   *
   * \returns 0 on success, -1 on failure.
   */
  virtual int Bind () = 0;

  /** 
   * \brief Allocate a local IPv6 endpoint for this socket.
   *
   * \returns 0 on success, -1 on failure.
   */
  virtual int Bind6 () = 0;

  /**
   * \brief Close a socket.
   * \returns zero on success, -1 on failure.
   *
   * After the Close call, the socket is no longer valid, and cannot
   * safely be used for subsequent operations.
   */
  virtual int Close (void) = 0;

  /**
   * \returns zero on success, -1 on failure.
   *
   * Do not allow any further Send calls. This method is typically
   * implemented for Tcp sockets by a half close.
   */
  virtual int ShutdownSend (void) = 0;

  /**
   * \returns zero on success, -1 on failure.
   *
   * Do not allow any further Recv calls. This method is typically
   * implemented for Tcp sockets by a half close.
   */
  virtual int ShutdownRecv (void) = 0;

  /**
   * \brief Initiate a connection to a remote host
   * \param address Address of remote.
   * \returns 0 on success, -1 on error (in which case errno is set).
   */
  virtual int Connect (const Address &address) = 0;

  /**
   * \brief Listen for incoming connections.
   * \returns 0 on success, -1 on error (in which case errno is set).
   */
  virtual int Listen (void) = 0;

  /**
   * \brief Returns the number of bytes which can be sent in a single call
   * to Send. 
   * 
   * For datagram sockets, this returns the number of bytes that
   * can be passed atomically through the underlying protocol.
   *
   * For stream sockets, this returns the available space in bytes
   * left in the transmit buffer.
   *
   * \returns The number of bytes which can be sent in a single Send call.
   */
  virtual uint32_t GetTxAvailable (void) const = 0;
 
  /**
   * \brief Send data (or dummy data) to the remote host
   *
   * This function matches closely in semantics to the send() function
   * call in the standard C library (libc):
   *   ssize_t send (int s, const void *msg, size_t len, int flags);
   * except that the send I/O is asynchronous.  This is the
   * primary Send method at this low-level API and must be implemented 
   * by subclasses.
   * 
   * In a typical blocking sockets model, this call would block upon
   * lack of space to hold the message to be sent.  In ns-3 at this
   * API, the call returns immediately in such a case, but the callback
   * registered with SetSendCallback() is invoked when the socket
   * has space (when it conceptually unblocks); this is an asynchronous
   * I/O model for send().
   * 
   * This variant of Send() uses class ns3::Packet to encapsulate
   * data, rather than providing a raw pointer and length field.
   * This allows an ns-3 application to attach tags if desired (such
   * as a flow ID) and may allow the simulator to avoid some data
   * copies.  Despite the appearance of sending Packets on a stream
   * socket, just think of it as a fancy byte buffer with streaming
   * semantics.
   *
   * If either the message buffer within the Packet is too long to pass 
   * atomically through the underlying protocol (for datagram sockets), 
   * or the message buffer cannot entirely fit in the transmit buffer
   * (for stream sockets), -1 is returned and SocketErrno is set 
   * to ERROR_MSGSIZE.  If the packet does not fit, the caller can
   * split the Packet (based on information obtained from 
   * GetTxAvailable) and reattempt to send the data.
   *
   * The flags argument is formed by or'ing one or more of the values:
   *        MSG_OOB        process out-of-band data 
   *        MSG_DONTROUTE  bypass routing, use direct interface 
   * These flags are _unsupported_ as of ns-3.1.
   *
   * \param p ns3::Packet to send
   * \param flags Socket control flags
   * \returns the number of bytes accepted for transmission if no error
   *          occurs, and -1 otherwise.
   *
   * \see SetSendCallback
   */
  virtual int Send (Ptr<Packet> p, uint32_t flags) = 0;

  /**
   * \brief Send data to a specified peer.
   *
   * This method has similar semantics to Send () but subclasses may
   * want to provide checks on socket state, so the implementation is
   * pushed to subclasses.
   *
   * \param p packet to send
   * \param flags Socket control flags
   * \param toAddress IP Address of remote host
   * \returns -1 in case of error or the number of bytes copied in the 
   *          internal buffer and accepted for transmission.
   */
  virtual int SendTo (Ptr<Packet> p, uint32_t flags, 
                      const Address &toAddress) = 0;

  /**
   * Return number of bytes which can be returned from one or 
   * multiple calls to Recv.
   * Must be possible to call this method from the Recv callback.
   *
   * \returns the number of bytes which can be returned from one or
   *          multiple Recv calls.
   */
  virtual uint32_t GetRxAvailable (void) const = 0;

  /**
   * \brief Read data from the socket
   *
   * This function matches closely in semantics to the recv() function
   * call in the standard C library (libc):
   *   ssize_t recv (int s, void *buf, size_t len, int flags);
   * except that the receive I/O is asynchronous.  This is the
   * primary Recv method at this low-level API and must be implemented 
   * by subclasses.
   * 
   * This method is normally used only on a connected socket.
   * In a typical blocking sockets model, this call would block until
   * at least one byte is returned or the connection closes.
   * In ns-3 at this API, the call returns immediately in such a case
   * and returns 0 if nothing is available to be read.
   * However, an application can set a callback, ns3::SetRecvCallback,
   * to be notified of data being available to be read
   * (when it conceptually unblocks); this is an asynchronous
   * I/O model for recv().
   * 
   * This variant of Recv() uses class ns3::Packet to encapsulate
   * data, rather than providing a raw pointer and length field.
   * This allows an ns-3 application to attach tags if desired (such
   * as a flow ID) and may allow the simulator to avoid some data
   * copies.  Despite the appearance of receiving Packets on a stream
   * socket, just think of it as a fancy byte buffer with streaming
   * semantics.
   *
   * The semantics depend on the type of socket.  For a datagram socket,
   * each Recv() returns the data from at most one Send(), and order
   * is not necessarily preserved.  For a stream socket, the bytes
   * are delivered in order, and on-the-wire packet boundaries are
   * not preserved.
   * 
   * The flags argument is formed by or'ing one or more of the values:
   *        MSG_OOB             process out-of-band data
   *        MSG_PEEK            peek at incoming message
   * None of these flags are supported for now.
   *
   * Some variants of Recv() are supported as additional API,
   * including RecvFrom(), overloaded Recv() without arguments,
   * and variants that use raw character buffers.
   *
   * \param maxSize reader will accept packet up to maxSize
   * \param flags Socket control flags
   * \returns Ptr<Packet> of the next in-sequence packet.  Returns
   * 0 if the socket cannot return a next in-sequence packet conforming
   * to the maxSize and flags.
   *
   * \see SetRecvCallback
   */
  virtual Ptr<Packet> Recv (uint32_t maxSize, uint32_t flags) = 0;

  /**
   * \brief Read a single packet from the socket and retrieve the sender 
   * address.
   *
   * Calls Recv(maxSize, flags) with maxSize
   * implicitly set to maximum sized integer, and flags set to zero.
   *
   * This method has similar semantics to Recv () but subclasses may
   * want to provide checks on socket state, so the implementation is
   * pushed to subclasses.
   *
   * \param maxSize reader will accept packet up to maxSize
   * \param flags Socket control flags
   * \param fromAddress output parameter that will return the
   * address of the sender of the received packet, if any.  Remains
   * untouched if no packet is received.
   * \returns Ptr<Packet> of the next in-sequence packet.  Returns
   * 0 if the socket cannot return a next in-sequence packet.
   */
  virtual Ptr<Packet> RecvFrom (uint32_t maxSize, uint32_t flags,
                                Address &fromAddress) = 0;

  /////////////////////////////////////////////////////////////////////
  //   The remainder of these public methods are overloaded methods  //
  //   or variants of Send() and Recv(), and they are non-virtual    //
  /////////////////////////////////////////////////////////////////////
 
  /**
   * \brief Send data (or dummy data) to the remote host
   * 
   * Overloaded version of Send(..., flags) with flags set to zero.
   *
   * \param p ns3::Packet to send
   * \returns the number of bytes accepted for transmission if no error
   *          occurs, and -1 otherwise.
   */
  int Send (Ptr<Packet> p);

  /**
   * \brief Send data (or dummy data) to the remote host
   * 
   * This method is provided so as to have an API which is closer in 
   * appearance to that of real network or BSD sockets.
   *
   * \param buf A pointer to a raw byte buffer of some data to send.  If 
   * this buffer is 0, we send dummy data whose size is specified by the 
   * second parameter
   * \param size the number of bytes to copy from the buffer
   * \param flags Socket control flags
   * \returns the number of bytes accepted for transmission if no error
   *          occurs, and -1 otherwise.
   */
  int Send (const uint8_t* buf, uint32_t size, uint32_t flags);


  /**
   * \brief Send data to a specified peer.
   *
   * This method is provided so as to have an API which is closer in 
   * appearance to that of real network or BSD sockets.
   *
   * \param buf A pointer to a raw byte buffer of some data to send.
   * If this is 0, we send dummy data whose size is specified by the 
   * third parameter
   * \param size the number of bytes to copy from the buffer
   * \param flags Socket control flags
   * \param address IP Address of remote host
   * \returns -1 in case of error or the number of bytes copied in the 
   *          internal buffer and accepted for transmission.
   *
   */
  int SendTo (const uint8_t* buf, uint32_t size, uint32_t flags, 
              const Address &address); 

  /**
   * \brief Read a single packet from the socket
   *
   * Overloaded version of Recv(maxSize, flags) with maxSize
   * implicitly set to maximum sized integer, and flags set to zero.
   *
   * \returns Ptr<Packet> of the next in-sequence packet.  Returns
   * 0 if the socket cannot return a next in-sequence packet.
   */
  Ptr<Packet> Recv (void);

  /**
   * \brief Recv data (or dummy data) from the remote host
   *
   * This method is provided so as to have an API which is closer in 
   * appearance to that of real network or BSD sockets.
   * 
   * If the underlying packet was carring null (fake) data, this buffer
   * will be zeroed up to the length specified by the return value.
   *
   * \param buf A pointer to a raw byte buffer to write the data to. 
   * \param size Number of bytes (at most) to copy to buf
   * \param flags any flags to pass to the socket
   * \returns number of bytes copied into buf
   */
  int Recv (uint8_t* buf, uint32_t size, uint32_t flags);

  /**
   * \brief Read a single packet from the socket and retrieve the sender 
   * address.
   *
   * Calls RecvFrom (maxSize, flags, fromAddress) with maxSize
   * implicitly set to maximum sized integer, and flags set to zero.
   *
   * \param fromAddress output parameter that will return the
   * address of the sender of the received packet, if any.  Remains
   * untouched if no packet is received.
   * \returns Ptr<Packet> of the next in-sequence packet.  Returns
   * 0 if the socket cannot return a next in-sequence packet.
   */
  Ptr<Packet> RecvFrom (Address &fromAddress);

  /**
   * \brief Read a single packet from the socket and retrieve the sender
   * address.
   *
   * This method is provided so as to have an API which is closer in 
   * appearance to that of real network or BSD sockets.
   * 
   * \param buf A pointer to a raw byte buffer to write the data to. 
   * If the underlying packet was carring null (fake) data, this buffer
   * will be zeroed up to the length specified by the return value.
   * \param size Number of bytes (at most) to copy to buf
   * \param flags any flags to pass to the socket
   * \param fromAddress output parameter that will return the
   * address of the sender of the received packet, if any.  Remains
   * untouched if no packet is received.
   * \returns number of bytes copied into buf
   */
  int RecvFrom (uint8_t* buf, uint32_t size, uint32_t flags,
                Address &fromAddress);
  /**
   * \brief Get socket address.
   * \param address the address name this socket is associated with.
   * \returns 0 if success, -1 otherwise
   */
  virtual int GetSockName (Address &address) const = 0; 

  /**
   * \brief Get the peer address of a connected socket.
   * \param address the address this socket is connected to.
   * \returns 0 if success, -1 otherwise
   */
  virtual int GetPeerName (Address &address) const = 0;

  /**
   * \brief Bind a socket to specific device.
   *
   * This method corresponds to using setsockopt() SO_BINDTODEVICE
   * of real network or BSD sockets.   If set on a socket, this option will
   * force packets to leave the bound device regardless of the device that
   * IP routing would naturally choose.  In the receive direction, only
   * packets received from the bound interface will be delivered.
   *
   * This option has no particular relationship to binding sockets to
   * an address via Socket::Bind ().  It is possible to bind sockets to a 
   * specific IP address on the bound interface by calling both 
   * Socket::Bind (address) and Socket::BindToNetDevice (device), but it
   * is also possible to bind to mismatching device and address, even if
   * the socket can not receive any packets as a result.
   *
   * \param netdevice Pointer to NetDevice of desired interface
   */
  virtual void BindToNetDevice (Ptr<NetDevice> netdevice);

  /**
   * \brief Returns socket's bound NetDevice, if any.
   *
   * This method corresponds to using getsockopt() SO_BINDTODEVICE
   * of real network or BSD sockets.
   * 
   * 
   * \returns Pointer to interface.
   */
  Ptr<NetDevice> GetBoundNetDevice (); 


  /**
   * \brief Configure whether broadcast datagram transmissions are allowed
   *
   * This method corresponds to using setsockopt() SO_BROADCAST of
   * real network or BSD sockets.  If set on a socket, this option
   * will enable or disable packets to be transmitted to broadcast
   * destination addresses.
   *
   * \param allowBroadcast Whether broadcast is allowed
   * \return true if operation succeeds
   */
  virtual bool SetAllowBroadcast (bool allowBroadcast) = 0;

  /**
   * \brief Query whether broadcast datagram transmissions are allowed
   *
   * This method corresponds to using getsockopt() SO_BROADCAST of
   * real network or BSD sockets.
   *
   * \returns true if broadcast is allowed, false otherwise
   */
  virtual bool GetAllowBroadcast () const = 0;

  /**
   * \brief Enable/Disable receive packet information to socket.
   *
   * For IP_PKTINFO/IP6_PKTINFO. This method is only usable for 
   * Raw socket and Datagram Socket. Not supported for Stream socket.
   *
   * Method doesn't make distinction between IPv4 and IPv6. If it is enabled,
   * it is enabled for all types of sockets that supports packet information
   *
   * \param flag Enable/Disable receive information
   */
  void SetRecvPktInfo (bool flag);

  /**
   * \brief Get status indicating whether enable/disable packet information to socket
   *
   * \returns True if packet information should be sent to socket
   */
  bool IsRecvPktInfo () const;

  /**
   * \brief Manually set the socket priority
   *
   * This method corresponds to using setsockopt () SO_PRIORITY of
   * real network or BSD sockets. On Linux, the socket priority can be
   * set to a value in the range [0..6], unless the user process has the
   * CAP_NET_ADMIN capability (see the man page for socket). ns-3 allows
   * users to set the socket priority to any 8-bit non-negative value,
   * which is equivalent to assuming that the CAP_NET_ADMIN capability is set.
   *
   * \param priority The socket priority
   */
  void SetPriority (uint8_t priority);

  /**
   * \brief Query the priority value of this socket
   *
   * This method corresponds to using getsockopt () SO_PRIORITY of real network
   * or BSD sockets.
   *
   * \return The priority value
   */
  uint8_t GetPriority (void) const;

  /**
   * \brief Return the priority corresponding to a given TOS value
   *
   * This function is implemented after the Linux rt_tos2priority
   * function. The usage of the TOS byte has been originally defined by
   * RFC 1349 (http://www.ietf.org/rfc/rfc1349.txt):
   *
   *               0     1     2     3     4     5     6     7
   *           +-----+-----+-----+-----+-----+-----+-----+-----+
   *           |   PRECEDENCE    |          TOS          | MBZ |
   *           +-----+-----+-----+-----+-----+-----+-----+-----+
   *
   * where MBZ stands for 'must be zero'.
   *
   * The Linux rt_tos2priority function ignores the precedence bits and
   * maps each of the 16 values coded in bits 3-6 as follows:
   *
   * Bits 3-6 | Means                   | Linux Priority
   * ---------|-------------------------|----------------
   *     0    |  Normal Service         | Best Effort (0)
   *     1    |  Minimize Monetary Cost | Best Effort (0)
   *     2    |  Maximize Reliability   | Best Effort (0)
   *     3    |  mmc+mr                 | Best Effort (0)
   *     4    |  Maximize Throughput    | Bulk (2)
   *     5    |  mmc+mt                 | Bulk (2)
   *     6    |  mr+mt                  | Bulk (2)
   *     7    |  mmc+mr+mt              | Bulk (2)
   *     8    |  Minimize Delay         | Interactive (6)
   *     9    |  mmc+md                 | Interactive (6)
   *    10    |  mr+md                  | Interactive (6)
   *    11    |  mmc+mr+md              | Interactive (6)
   *    12    |  mt+md                  | Int. Bulk (4)
   *    13    |  mmc+mt+md              | Int. Bulk (4)
   *    14    |  mr+mt+md               | Int. Bulk (4)
   *    15    |  mmc+mr+mt+md           | Int. Bulk (4)
   *
   * RFC 2474 (http://www.ietf.org/rfc/rfc2474.txt) redefines the TOS byte:
   *
   *               0     1     2     3     4     5     6     7
   *           +-----+-----+-----+-----+-----+-----+-----+-----+
   *           |              DSCP                 |     CU    |
   *           +-----+-----+-----+-----+-----+-----+-----+-----+
   *
   * where DSCP is the Differentiated Services Code Point and CU stands for
   * 'currently unused' (actually, RFC 3168 proposes to use these two bits for
   * ECN purposes). The table above allows to determine how the Linux
   * rt_tos2priority function maps each DSCP value to a priority value. Such a
   * mapping is shown below.
   *
   * DSCP | Hex  | TOS (binary) | bits 3-6 | Linux Priority
   * -----|------|--------------|----------|----------------
   * EF   | 0x2E |   101110xx   |  12-13   |  Int. Bulk (4)
   * AF11 | 0x0A |   001010xx   |   4-5    |  Bulk (2)
   * AF21 | 0x12 |   010010xx   |   4-5    |  Bulk (2)
   * AF31 | 0x1A |   011010xx   |   4-5    |  Bulk (2)
   * AF41 | 0x22 |   100010xx   |   4-5    |  Bulk (2)
   * AF12 | 0x0C |   001100xx   |   8-9    |  Interactive (6)
   * AF22 | 0x14 |   010100xx   |   8-9    |  Interactive (6)
   * AF32 | 0x1C |   011100xx   |   8-9    |  Interactive (6)
   * AF42 | 0x24 |   100100xx   |   8-9    |  Interactive (6)
   * AF13 | 0x0E |   001110xx   |  12-13   |  Int. Bulk (4)
   * AF23 | 0x16 |   010110xx   |  12-13   |  Int. Bulk (4)
   * AF33 | 0x1E |   011110xx   |  12-13   |  Int. Bulk (4)
   * AF43 | 0x26 |   100110xx   |  12-13   |  Int. Bulk (4)
   * CS0  | 0x00 |   000000xx   |   0-1    |  Best Effort (0)
   * CS1  | 0x08 |   001000xx   |   0-1    |  Best Effort (0)
   * CS2  | 0x10 |   010000xx   |   0-1    |  Best Effort (0)
   * CS3  | 0x18 |   011000xx   |   0-1    |  Best Effort (0)
   * CS4  | 0x20 |   100000xx   |   0-1    |  Best Effort (0)
   * CS5  | 0x28 |   101000xx   |   0-1    |  Best Effort (0)
   * CS6  | 0x30 |   110000xx   |   0-1    |  Best Effort (0)
   * CS7  | 0x38 |   111000xx   |   0-1    |  Best Effort (0)
   *
   * \param ipTos the TOS value (in the range 0..255)
   * \return The priority value corresponding to the given TOS value
   */
  static uint8_t IpTos2Priority (uint8_t ipTos);

  /**
   * \brief Manually set IP Type of Service field
   * 
   * This method corresponds to using setsockopt () IP_TOS of
   * real network or BSD sockets. This option is for IPv4 only.
   * Setting the IP TOS also changes the socket priority as
   * stated in the man page.
   *
   * \param ipTos The desired TOS value for IP headers
   */
  void SetIpTos (uint8_t ipTos);

  /**
   * \brief Query the value of IP Type of Service of this socket
   *
   * This method corresponds to using getsockopt () IP_TOS of real network
   * or BSD sockets.
   *
   * \return The raw IP TOS value
   */
  uint8_t GetIpTos (void) const;

  /**
   * \brief Tells a socket to pass information about IP Type of Service up the stack
   *
   * This method corresponds to using setsockopt () IP_RECVTOS of real
   * network or BSD sockets. In our implementation, the socket simply 
   * adds a SocketIpTosTag tag to the packet before passing the
   * packet up the stack.
   *
   * \param ipv4RecvTos Whether the socket should add SocketIpv4TosTag tag
   * to the packet
   */
  void SetIpRecvTos (bool ipv4RecvTos);

  /**
   * \brief Ask if the socket is currently passing information about IP Type of Service up the stack
   *
   * This method corresponds to using getsockopt () IP_RECVTOS of real
   * network or BSD sockets.
   *
   * \return Whether the IP_RECVTOS is set
   */
  bool IsIpRecvTos (void) const;

  /**
   * \brief Manually set IPv6 Traffic Class field
   * 
   * This method corresponds to using setsockopt () IPV6_TCLASS of
   * real network or BSD sockets. This option is for IPv6 only.
   * Setting the IPV6_TCLASSS to -1 clears the option and let the socket
   * uses the default value.
   *
   * \param ipTclass The desired TCLASS value for IPv6 headers
   */
  void SetIpv6Tclass (int ipTclass);

  /**
   * \brief Query the value of IPv6 Traffic Class field of this socket
   *
   * This method corresponds to using getsockopt () IPV6_TCLASS of real network
   * or BSD sockets.
   *
   * \return The raw IPV6_TCLASS value
   */
  uint8_t GetIpv6Tclass (void) const;

  /**
   * \brief Tells a socket to pass information about IPv6 Traffic Class up the stack
   *
   * This method corresponds to using setsockopt () IPV6_RECVTCLASS of real
   * network or BSD sockets. In our implementation, the socket simply 
   * adds a SocketIpv6TclasssTag tag to the packet before passing the
   * packet up the stack.
   *
   * \param ipv6RecvTclass Whether the socket should add SocketIpv6TclassTag tag
   * to the packet
   */
  void SetIpv6RecvTclass (bool ipv6RecvTclass);

  /**
   * \brief Ask if the socket is currently passing information about IPv6 Traffic Class up the stack
   *
   * This method corresponds to using getsockopt () IPV6_RECVTCLASS of real
   * network or BSD sockets.
   *
   * \return Whether the IPV6_RECVTCLASS is set
   */
  bool IsIpv6RecvTclass (void) const;

  /**
   * \brief Manually set IP Time to Live field
   * 
   * This method corresponds to using setsockopt () IP_TTL of
   * real network or BSD sockets.
   *
   * \param ipTtl The desired TTL value for IP headers
   */
  virtual void SetIpTtl (uint8_t ipTtl);

  /**
   * \brief Query the value of IP Time to Live field of this socket
   *
   * This method corresponds to using getsockopt () IP_TTL of real network
   * or BSD sockets.
   *
   * \return The raw IP TTL value
   */
  virtual uint8_t GetIpTtl (void) const;

  /**
   * \brief Tells a socket to pass information about IP_TTL up the stack
   *
   * This method corresponds to using setsockopt () IP_RECVTTL of real
   * network or BSD sockets. In our implementation, the socket simply 
   * adds a SocketIpTtlTag tag to the packet before passing the
   * packet up the stack.
   *
   * \param ipv4RecvTtl Whether the socket should add SocketIpv4TtlTag tag
   * to the packet
   */
  void SetIpRecvTtl (bool ipv4RecvTtl);

  /**
   * \brief Ask if the socket is currently passing information about IP_TTL up the stack
   *
   * This method corresponds to using getsockopt () IP_RECVTTL of real
   * network or BSD sockets.
   *
   * \return Whether the IP_RECVTTL is set
   */
  bool IsIpRecvTtl (void) const;

  /**
   * \brief Manually set IPv6 Hop Limit
   * 
   * This method corresponds to using setsockopt () IPV6_HOPLIMIT of
   * real network or BSD sockets.
   *
   * \param ipHopLimit The desired Hop Limit value for IPv6 headers
   */
  virtual void SetIpv6HopLimit (uint8_t ipHopLimit);

  /**
   * \brief Query the value of IP Hop Limit field of this socket
   *
   * This method corresponds to using getsockopt () IPV6_HOPLIMIT of real network
   * or BSD sockets.
   *
   * \return The raw IPv6 Hop Limit value
   */
  virtual uint8_t GetIpv6HopLimit (void) const;

  /**
   * \brief Tells a socket to pass information about IPv6 Hop Limit up the stack
   *
   * This method corresponds to using setsockopt () IPV6_RECVHOPLIMIT of real
   * network or BSD sockets. In our implementation, the socket simply 
   * adds a SocketIpv6HopLimitTag tag to the packet before passing the
   * packet up the stack.
   *
   * \param ipv6RecvHopLimit Whether the socket should add SocketIpv6HopLimitTag tag
   * to the packet
   */
  void SetIpv6RecvHopLimit (bool ipv6RecvHopLimit);

  /**
   * \brief Ask if the socket is currently passing information about IPv6 Hop Limit up the stack
   *
   * This method corresponds to using getsockopt () IPV6_RECVHOPLIMIT of real
   * network or BSD sockets.
   *
   * \return Whether the IPV6_RECVHOPLIMIT is set
   */
  bool IsIpv6RecvHopLimit (void) const;
 
  /**
   * \brief Joins a IPv6 multicast group.
   *
   * Based on the filter mode and source addresses this can be interpreted as a
   * join, leave, or modification to source filtering on a multicast group.
   *
   * Mind that a socket can join only one multicast group. Any attempt to join another group will remove the old one.
   *
   *
   * \param address Requested multicast address.
   * \param filterMode Socket filtering mode (INCLUDE | EXCLUDE).
   * \param sourceAddresses All the source addresses on which socket is interested or not interested.
   */
  virtual void Ipv6JoinGroup (Ipv6Address address, Ipv6MulticastFilterMode filterMode, std::vector<Ipv6Address> sourceAddresses);

  /**
   * \brief Joins a IPv6 multicast group without filters.
   *
   * A socket can join only one multicast group. Any attempt to join another group will remove the old one.
   *
   * \param address Group address on which socket wants to join.
   */
  virtual void Ipv6JoinGroup (Ipv6Address address);

  /**
   * \brief Leaves IPv6 multicast group this socket is joined to.
   */
  virtual void Ipv6LeaveGroup (void);

protected:
  /**
   * \brief Notify through the callback (if set) that the connection has been
   *        established.
   */
  void NotifyConnectionSucceeded (void);

  /**
   * \brief Notify through the callback (if set) that the connection has not been
   *        established due to an error.
   */
  void NotifyConnectionFailed (void);

  /**
   * \brief Notify through the callback (if set) that the connection has been
   *        closed.
   */
  void NotifyNormalClose (void);

  /**
   * \brief Notify through the callback (if set) that the connection has been
   *        closed due to an error.
   */
  void NotifyErrorClose (void);

  /**
   * \brief Notify through the callback (if set) that an incoming connection
   *        is being requested by a remote host.
   *
   * This function returns true by default (i.e., accept all the incoming connections).
   * The callback (if set) might restrict this behaviour by returning zero for a
   * connection that should be refused.
   *
   * \param from the address the connection is incoming from
   * \returns true if the connection must be accepted, false otherwise.
   */
  bool NotifyConnectionRequest (const Address &from);

  /**
   * \brief Notify through the callback (if set) that a new connection has been
   *        created.
   * \param socket The socket receiving the new connection.
   * \param from The address of the node initiating the connection.
   */
  void NotifyNewConnectionCreated (Ptr<Socket> socket, const Address &from);

  /**
   * \brief Notify through the callback (if set) that some data have been sent.
   *
   * \param size number of sent bytes.
   */
  void NotifyDataSent (uint32_t size);

  /**
   * \brief Notify through the callback (if set) that some data have been sent.
   *
   * \param spaceAvailable the number of bytes available in the transmission buffer.
   */
  void NotifySend (uint32_t spaceAvailable);

  /**
   * \brief Notify through the callback (if set) that some data have been received.
   */
  void NotifyDataRecv (void);

  // inherited function, no doc necessary
  virtual void DoDispose (void);

  /**
   * \brief Checks if the socket has a specific IPv6 Tclass set
   *
   * \returns true if the socket has a IPv6 Tclass set, false otherwise.
   */
  bool IsManualIpv6Tclass (void) const;

  /**
   * \brief Checks if the socket has a specific IPv4 TTL set
   *
   * \returns true if the socket has a IPv4 TTL set, false otherwise.
   */
  bool IsManualIpTtl (void) const;

  /**
   * \brief Checks if the socket has a specific IPv6 Hop Limit set
   *
   * \returns true if the socket has a IPv6 Hop Limit set, false otherwise.
   */
  bool IsManualIpv6HopLimit (void) const;

  Ptr<NetDevice> m_boundnetdevice; //!< the device this socket is bound to (might be null).
  bool m_recvPktInfo; //!< if the socket should add packet info tags to the packet forwarded to L4.
  Ipv6Address m_ipv6MulticastGroupAddress; //!< IPv6 multicast group address.

private:
  Callback<void, Ptr<Socket> >                   m_connectionSucceeded;  //!< connection succeeded callback
  Callback<void, Ptr<Socket> >                   m_connectionFailed;     //!< connection failed callback
  Callback<void, Ptr<Socket> >                   m_normalClose;          //!< connection closed callback
  Callback<void, Ptr<Socket> >                   m_errorClose;           //!< connection closed due to errors callback
  Callback<bool, Ptr<Socket>, const Address &>   m_connectionRequest;    //!< connection request callback
  Callback<void, Ptr<Socket>, const Address&>    m_newConnectionCreated; //!< connection created callback
  Callback<void, Ptr<Socket>, uint32_t>          m_dataSent;             //!< data sent callback
  Callback<void, Ptr<Socket>, uint32_t >         m_sendCb;               //!< packet sent callback
  Callback<void, Ptr<Socket> >                   m_receivedData;         //!< data received callback

  uint8_t m_priority; //!< the socket priority

  //IPv4 options
  bool m_manualIpTtl; //!< socket has IPv4 TTL set
  bool m_ipRecvTos;   //!< socket forwards IPv4 TOS tag to L4
  bool m_ipRecvTtl;   //!< socket forwards IPv4 TTL tag to L4

  uint8_t m_ipTos; //!< the socket IPv4 TOS
  uint8_t m_ipTtl; //!< the socket IPv4 TTL

  //IPv6 options
  bool m_manualIpv6Tclass;    //!< socket has IPv6 Tclass set
  bool m_manualIpv6HopLimit;  //!< socket has IPv6 Hop Limit set
  bool m_ipv6RecvTclass;      //!< socket forwards IPv6 Tclass tag to L4
  bool m_ipv6RecvHopLimit;    //!< socket forwards IPv6 Hop Limit tag to L4

  uint8_t m_ipv6Tclass;     //!< the socket IPv6 Tclass
  uint8_t m_ipv6HopLimit;   //!< the socket IPv6 Hop Limit
};

/**
 * \brief This class implements a tag that carries the socket-specific
 * TTL of a packet to the IP layer
 */
class SocketIpTtlTag : public Tag
{
public:
  SocketIpTtlTag ();

  /**
   * \brief Set the tag's TTL
   *
   * \param ttl the TTL
   */
  void SetTtl (uint8_t ttl);

  /**
   * \brief Get the tag's TTL
   *
   * \returns the TTL
   */
  uint8_t GetTtl (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited function, no need to doc.
  virtual TypeId GetInstanceTypeId (void) const;

  // inherited function, no need to doc.
  virtual uint32_t GetSerializedSize (void) const;

  // inherited function, no need to doc.
  virtual void Serialize (TagBuffer i) const;

  // inherited function, no need to doc.
  virtual void Deserialize (TagBuffer i);

  // inherited function, no need to doc.
  virtual void Print (std::ostream &os) const;

private:
  uint8_t m_ttl; //!< the ttl carried by the tag
};

/**
 * \brief This class implements a tag that carries the socket-specific
 * HOPLIMIT of a packet to the IPv6 layer
 */
class SocketIpv6HopLimitTag : public Tag
{
public:
  SocketIpv6HopLimitTag ();

  /**
   * \brief Set the tag's Hop Limit
   *
   * \param hopLimit the Hop Limit
   */
  void SetHopLimit (uint8_t hopLimit);

  /**
   * \brief Get the tag's Hop Limit
   *
   * \returns the Hop Limit
   */
  uint8_t GetHopLimit (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited function, no need to doc.
  virtual TypeId GetInstanceTypeId (void) const;

  // inherited function, no need to doc.
  virtual uint32_t GetSerializedSize (void) const;

  // inherited function, no need to doc.
  virtual void Serialize (TagBuffer i) const;

  // inherited function, no need to doc.
  virtual void Deserialize (TagBuffer i);

  // inherited function, no need to doc.
  virtual void Print (std::ostream &os) const;

private:
  uint8_t m_hopLimit; //!< the Hop Limit carried by the tag
};

/**
 * \brief indicates whether packets should be sent out with
 * the DF (Don't Fragment) flag set.
 */
class SocketSetDontFragmentTag : public Tag
{
public:
  SocketSetDontFragmentTag ();

  /**
   * \brief Enables the DF (Don't Fragment) flag
   */
  void Enable (void);

  /**
   * \brief Disables the DF (Don't Fragment) flag
   */
  void Disable (void);

  /**
   * \brief Checks if the DF (Don't Fragment) flag is set
   *
   * \returns true if DF is set.
   */
  bool IsEnabled (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited function, no need to doc.
  virtual TypeId GetInstanceTypeId (void) const;

  // inherited function, no need to doc.
  virtual uint32_t GetSerializedSize (void) const;

  // inherited function, no need to doc.
  virtual void Serialize (TagBuffer i) const;

  // inherited function, no need to doc.
  virtual void Deserialize (TagBuffer i);

  // inherited function, no need to doc.
  virtual void Print (std::ostream &os) const;
private:
  bool m_dontFragment; //!< DF bit value for outgoing packets.
};

/**
 * \brief indicates whether the socket has IP_TOS set.
 * This tag is for IPv4 socket.
 */
class SocketIpTosTag : public Tag
{
public:
  SocketIpTosTag ();

  /**
   * \brief Set the tag's TOS
   *
   * \param tos the TOS
   */
  void SetTos (uint8_t tos);

  /**
   * \brief Get the tag's TOS
   *
   * \returns the TOS
   */
  uint8_t GetTos (void) const;
  
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited function, no need to doc.
  virtual TypeId GetInstanceTypeId (void) const;

  // inherited function, no need to doc.
  virtual uint32_t GetSerializedSize (void) const;

  // inherited function, no need to doc.
  virtual void Serialize (TagBuffer i) const;

  // inherited function, no need to doc.
  virtual void Deserialize (TagBuffer i);

  // inherited function, no need to doc.
  virtual void Print (std::ostream &os) const;
private:
  uint8_t m_ipTos;  //!< the TOS carried by the tag
};

/**
 * \brief indicates whether the socket has a priority set.
 */
class SocketPriorityTag : public Tag
{
public:
  SocketPriorityTag ();

  /**
   * \brief Set the tag's priority
   *
   * \param priority the priority
   */
  void SetPriority (uint8_t priority);

  /**
   * \brief Get the tag's priority
   *
   * \returns the priority
   */
  uint8_t GetPriority (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited function, no need to doc.
  virtual TypeId GetInstanceTypeId (void) const;

  // inherited function, no need to doc.
  virtual uint32_t GetSerializedSize (void) const;

  // inherited function, no need to doc.
  virtual void Serialize (TagBuffer i) const;

  // inherited function, no need to doc.
  virtual void Deserialize (TagBuffer i);

  // inherited function, no need to doc.
  virtual void Print (std::ostream &os) const;
private:
  uint8_t m_priority;  //!< the priority carried by the tag
};

/**
 * \brief indicates whether the socket has IPV6_TCLASS set.
 * This tag is for IPv6 socket.
 */
class SocketIpv6TclassTag : public Tag
{
public:
  SocketIpv6TclassTag ();

  /**
   * \brief Set the tag's Tclass
   *
   * \param tclass the Tclass
   */
  void SetTclass (uint8_t tclass);

  /**
   * \brief Get the tag's Tclass
   *
   * \returns the Tclass
   */
  uint8_t GetTclass (void) const;
  
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited function, no need to doc.
  virtual TypeId GetInstanceTypeId (void) const;

  // inherited function, no need to doc.
  virtual uint32_t GetSerializedSize (void) const;

  // inherited function, no need to doc.
  virtual void Serialize (TagBuffer i) const;

  // inherited function, no need to doc.
  virtual void Deserialize (TagBuffer i);

  // inherited function, no need to doc.
  virtual void Print (std::ostream &os) const;
private:
  uint8_t m_ipv6Tclass; //!< the Tclass carried by the tag
};

} // namespace ns3

#endif /* NS3_SOCKET_H */
