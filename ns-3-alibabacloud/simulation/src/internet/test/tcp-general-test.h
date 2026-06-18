/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
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
#ifndef TCPGENERALTEST_H
#define TCPGENERALTEST_H

#include "ns3/simple-net-device.h"
#include "ns3/error-model.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-recovery-ops.h"
#include "ns3/tcp-rate-ops.h"
#include "ns3/test.h"

namespace ns3 {

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Class for inserting callbacks special points of the flow of TCP sockets
 *
 * This subclass is born to extend TcpSocketBase, inserting callbacks in certain
 * points of the flow, to be used in testing to check certain values or flow
 * directions.
 *
 * There isn't the necessity to fill TcpSocketBase of TracedCallbacks; the rationale
 * is to maintain the base class as clean as possible.
 *
 * To be fair with testing, this class should NOT modify the behavior of TcpSocketBase.
 *
 * \see SetRcvAckCb
 * \see SetProcessedAckCb
 * \see SetAfterRetransmitCb
 * \see SetBeforeRetransmitCb
 */
class TcpSocketMsgBase : public ns3::TcpSocketBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TcpSocketMsgBase () : TcpSocketBase ()
  {
  }

  /**
   * \brief Constructor.
   * \param other The object to copy from.
   */
  TcpSocketMsgBase (const TcpSocketMsgBase &other) : TcpSocketBase (other)
  {
    m_rcvAckCb = other.m_rcvAckCb;
    m_processedAckCb = other.m_processedAckCb;
    m_beforeRetrCallback = other.m_beforeRetrCallback;
    m_afterRetrCallback = other.m_afterRetrCallback;
    m_forkCb = other.m_forkCb;
    m_updateRttCb = other.m_updateRttCb;
  }

  /// Callback for the ACK management.
  typedef Callback<void, Ptr<const Packet>, const TcpHeader&,
                   Ptr<const TcpSocketBase> > AckManagementCb;
  /// Callback for the packet retransmission management.
  typedef Callback<void, Ptr<const TcpSocketState>,
                   Ptr<const TcpSocketBase> > RetrCb;
  /// Callback for the RTT update management.
  typedef Callback<void, Ptr<const TcpSocketBase>, const SequenceNumber32&,
                   uint32_t, bool> UpdateRttCallback;

  /**
   * \brief Set the callback invoked when an ACK is received (at the beginning
   * of the processing)
   *
   * \param cb callback
   */
  void SetRcvAckCb (AckManagementCb cb);

  /**
   * \brief Set the callback invoked when an ACK is received and processed
   * (at the end of the processing)
   *
   * \param cb callback
   */
  void SetProcessedAckCb (AckManagementCb cb);

  /**
   * \brief Set the callback invoked after the processing of a retransmit timeout
   *
   * \param cb callback
   */
  void SetAfterRetransmitCb (RetrCb cb);

  /**
   * \brief Set the callback invoked before the processing of a retransmit timeout
   *
   * \param cb callback
   */
  void SetBeforeRetransmitCb (RetrCb cb);

  /**
   * \brief Set the callback invoked after the forking
   * \param cb callback
   */
  void SetForkCb (Callback<void, Ptr<TcpSocketMsgBase> > cb);

  /**
   * \brief Set the callback invoked when we update rtt history
   *
   * \param cb callback
   */
  void SetUpdateRttHistoryCb (UpdateRttCallback cb);

protected:
  virtual void ReceivedAck (Ptr<Packet> packet, const TcpHeader& tcpHeader);
  virtual void ReTxTimeout (void);
  virtual Ptr<TcpSocketBase> Fork (void);
  virtual void CompleteFork (Ptr<Packet> p, const TcpHeader& tcpHeader,
                             const Address& fromAddress, const Address& toAddress);
  virtual void UpdateRttHistory (const SequenceNumber32 &seq, uint32_t sz,
                                 bool isRetransmission);

private:
  AckManagementCb m_rcvAckCb;       //!< Receive ACK callback.
  AckManagementCb m_processedAckCb; //!< Processed ACK callback.
  RetrCb m_beforeRetrCallback;      //!< Before retransmission callback.
  RetrCb m_afterRetrCallback;       //!< After retransmission callback.
  Callback<void, Ptr<TcpSocketMsgBase> > m_forkCb;  //!< Fork callback.
  UpdateRttCallback m_updateRttCb;  //!< Update RTT callback.
};


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief A TCP socket which sends ACKs smaller than the segment received.
 *
 * Usually, a TCP socket which receives the sequence number "x" replies with
 * an ACK to "x+1". What happen if a malicious socket sends smaller ACKs
 * (e.g. two ACKs, one for "x/2", and the other for "x+1") ? A TCP implementation
 * should avoid to artificially increase the congestion window, thinking of
 * having ACKed 2 segments instead of 1.
 *
 * Set the number of bytes that should be acked in each ACK packet with
 * SetBytesToAck.
 *
 * \see TcpSlowStartAttackerTest
 */
class TcpSocketSmallAcks : public TcpSocketMsgBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TcpSocketSmallAcks ()
    : TcpSocketMsgBase (),
      m_bytesToAck (125),
      m_bytesLeftToBeAcked (0),
      m_lastAckedSeq (1)
  {
  }

  /**
   * \brief Constructor.
   * \param other The object to copy from.
   */
  TcpSocketSmallAcks (const TcpSocketSmallAcks &other)
    : TcpSocketMsgBase (other),
      m_bytesToAck (other.m_bytesToAck),
      m_bytesLeftToBeAcked (other.m_bytesLeftToBeAcked),
      m_lastAckedSeq (other.m_lastAckedSeq)
  {
  }

  /**
   * \brief Set the bytes to be ACKed.
   * \param bytes The number of bytes.
   */
  void SetBytesToAck (uint32_t bytes)
  {
    m_bytesToAck = bytes;
  }

protected:
  virtual void SendEmptyPacket (uint8_t flags);
  Ptr<TcpSocketBase> Fork (void);

  uint32_t m_bytesToAck;            //!< Number of bytes to be ACKed.
  uint32_t m_bytesLeftToBeAcked;    //!< Number of bytes to be ACKed left.
  SequenceNumber32 m_lastAckedSeq;  //!< Last sequence number ACKed.
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief General infrastructure for TCP testing
 *
 * The class provides a simple setup for a connection testing. Implement
 * or modify the virtual methods in order to install a specified
 * channel, a specified socket and a specified error model on this simulation.
 * Default values are a null error model, and as a channel a SimpleChannel with
 * the propagation delay set through the constructor.
 *
 * Check DoRun documentation for more information on the environment setup.
 *
 * Apart from setting up the environment for testing, subclassing permits also
 * to track and check what is happening inside the two connected sockets. Thanks
 * to TcpSocketMsgBase, there are many information provided to children:
 *
 * - Tracing of states inside the state machines (TCP and ACK ones, through
 *   functions CongStateTrace and TcpStateTrace)
 * - cWnd tracing (through CWndTrace)
 * - Socket closing: error state, or normal state (NormalClose and ErrorClose)
 * - Packet drop, inside queue or over the link (QueueDrop, PhyDrop)
 * - Ack received (RcvAck) and Ack processed (ProcessedAck). The first is used to
 *   signal that an ACK has been received; after the processing of it, the Second
 *   is called
 * - A packet is transmitted to IP layer or received from IP layer (Tx and Rx)
 * - The RTO expires (RTOExpired)
 *
 * The default version of such methods is empty; implement their behavior differently,
 * based on what you want to test. Default is empty to avoid the need to implement
 * useless pure virtual function.
 *
 * If you need values from TcpSocketBase, thanks to the friend relationship between
 * this class and TcpSocketBase itself you can get it. Remember that friendship
 * isn't passed by inheritance, so the way to go is to write a Getters (like
 * GetSegSize) and call it in the subclass.
 *
 * \see DoRun
 * \see TcpSocketMsgBase
 */
class TcpGeneralTest : public TestCase
{
public:
  /**
   * \brief TcpGeneralTest constructor
   *
   * Please use the method ConfigureEnvironment () to configure other
   * parameters than the test description.
   *
   * \param desc description of the test
   */
  TcpGeneralTest (const std::string &desc);
  ~TcpGeneralTest ();

  /**
   * \brief Used as parameter of methods, specifies on what node
   * the caller is interested (e.g. GetSegSize).
   */
  enum SocketWho
  {
    SENDER,  //!< Sender node
    RECEIVER //!< Receiver node
  };

protected:
  /**
   * \brief Create and return the channel installed between the two socket
   *
   * \return A SimpleChannel subclass
   */
  virtual Ptr<SimpleChannel> CreateChannel ();

  /**
   * \brief Create and return the error model to install in the sender node
   *
   * \return sender error model
   */
  virtual Ptr<ErrorModel> CreateSenderErrorModel ();

  /**
   * \brief Create and return the error model to install in the receiver node
   *
   * \return receiver error model
   */
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();

  /**
   * \brief Create and install the socket to install on the receiver
   * \param node receiver node pointer
   * \return the socket to be installed in the receiver
   */
  virtual Ptr<TcpSocketMsgBase> CreateReceiverSocket (Ptr<Node> node);

  /**
   * \brief Create and install the socket to install on the sender
   * \param node sender node pointer
   * \return the socket to be installed in the sender
   */
  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);

  /**
   * \brief Create a socket
   *
   * \param node associated node
   * \param socketType Type of the TCP socket
   * \param congControl congestion control
   * \return a pointer to the newer created socket
   */
  virtual Ptr<TcpSocketMsgBase> CreateSocket (Ptr<Node> node, TypeId socketType,
                                              TypeId congControl);

  /**
   * \brief Create a socket
   *
   * \param node associated node
   * \param socketType Type of the TCP socket
   * \param congControl congestion control
   * \param recoveryAlgorithm recovery algorithm
   * \return a pointer to the newer created socket
   */
  virtual Ptr<TcpSocketMsgBase> CreateSocket (Ptr<Node> node, TypeId socketType,
                                              TypeId congControl, TypeId recoveryAlgorithm);

  /**
   * \brief Get the pointer to a previously created sender socket
   * \return ptr to sender socket or 0
   */
  Ptr<TcpSocketMsgBase> GetSenderSocket ()
  {
    return m_senderSocket;
  }

  /**
   * \brief Get the pointer to a previously created receiver socket
   * \return ptr to receiver socket or 0
   */
  Ptr<TcpSocketMsgBase> GetReceiverSocket ()
  {
    return m_receiverSocket;
  }

  /**
   * \brief Execute the tcp test
   *
   * As environment, two socket are connected through a SimpleChannel. Each device
   * has an MTU of 1500 bytes, and the application starts to send packet at
   * 10s of simulated time, through SendPacket.
   *
   * If you need to change parameters of the environment, please inherit an
   * implement the method ConfigureEnvironment (); that will be called at the
   * beginning of this method. To configure Socket parameters (i.e. parameters
   * that should be applied after socket have been created) use ConfigureProperties.
   *
   * Please do not use any Config:: statements.
   *
   * \see ConfigureEnvironment
   */
  virtual void DoRun (void);

  /**
   * \brief Change the configuration of the environment
   */
  virtual void ConfigureEnvironment (void);

  /**
   * \brief Change the configuration of the socket properties
   */
  virtual void ConfigureProperties (void);

  /**
   * \brief Teardown the TCP test
   */
  virtual void DoTeardown (void);

  /**
   * \brief Scheduled at 0.0, SENDER starts the connection to RECEIVER
   */
  void DoConnect ();

  /**
   * \brief Packet received
   *
   * The method processes the packet (application-layer)
   * \param socket socket which has received the packet
   */
  virtual void ReceivePacket (Ptr<Socket> socket);

  /**
   * \brief Send packets to other endpoint
   *
   * \param socket Socket
   * \param pktSize size of the packet
   * \param pktCount number of packets to send
   * \param pktInterval interval between packet (application-level)
   */
  void SendPacket (Ptr<Socket> socket, uint32_t pktSize,
                   uint32_t pktCount, Time pktInterval);

  /**
   * \brief Get the segment size of the node specified
   *
   * \param who node to get the parameter from
   *
   * \return segment size of the specified node
   */
  uint32_t GetSegSize (SocketWho who);

  /**
   * \brief Get the highest tx mark of the node specified
   *
   * \param who node to get the parameter from
   *
   * \return mark of the specified node
   */
  SequenceNumber32 GetHighestTxMark (SocketWho who);

  /**
   * \brief Get the retransmission threshold
   * \param who node to get the parameter from
   * \return retransmission threshold
   */
  uint32_t GetReTxThreshold (SocketWho who);

  /**
   * \brief Get the initial slow start threshold
   * \param who node to get the parameter from
   * \return initial slow start threshold
   */
  uint32_t GetInitialSsThresh (SocketWho who);

  /**
   * \brief Get the initial congestion window
   * \param who node to get the parameter from
   * \return initial cwnd
   */
  uint32_t GetInitialCwnd (SocketWho who);

  /**
   * \brief Get the number of dupack received
   * \param who node to get the parameter from
   * \return number of dupack
   */
  uint32_t GetDupAckCount (SocketWho who);

  /**
   * \brief Get the number of delayed ack (if present)
   * \param who node to get the parameter from
   * \return number of ack we will wait before sending an ACK
   */
  uint32_t GetDelAckCount (SocketWho who);

  /**
   * \brief Get the timeout of delayed ack (if present)
   * \param who node to get the parameter from
   * \return time we will wait before sending an ACK
   */
  Time GetDelAckTimeout (SocketWho who);

  /**
   * \brief Get the retransmission time
   *
   * \param who node to get the parameter from
   * \return calculated RTO time
   */
  Time GetRto (SocketWho who);

  /**
   * \brief Get the minimum RTO attribute
   *
   * \param who node to get the parameter from
   * \return minimum RTO time
   */
  Time GetMinRto (SocketWho who);

  /**
   * \brief Get the retransmission time for the SYN segments
   *
   * \param who node to get the parameter from
   * \return SYN segments RTO time
   */
  Time GetConnTimeout (SocketWho who);

  /**
   * \brief Get the Rtt estimator of the socket
   *
   * \param who node to get the parameter from
   * \return Rtt estimator
   */
  Ptr<RttEstimator> GetRttEstimator (SocketWho who);

  /**
   * \brief Get the clock granularity attribute
   *
   * \param who node to get the parameter from
   * \return clock granularity
   */
  Time GetClockGranularity (SocketWho who);

  /**
   * \brief Get the state of the TCP state machine
   *
   * \param who socket where check the parameter
   * \return the state of the socket
   */
  TcpSocket::TcpStates_t GetTcpState (SocketWho who);

  /**
   * \brief Get the TCB from selected socket
   *
   * \param who socket where get the TCB
   * \return the transmission control block
   */
  Ptr<TcpSocketState> GetTcb (SocketWho who);

  /**
   * \brief Get the Rx buffer from selected socket
   *
   * \param who socket where get the TCB
   * \return the rx buffer
   */
  Ptr<TcpRxBuffer> GetRxBuffer (SocketWho who);

  /**
   * \brief Get the Tx buffer from selected socket
   *
   * \param who socket where get the TCB
   * \return the tx buffer
   */
  Ptr<TcpTxBuffer> GetTxBuffer (SocketWho who);

  /**
   * \brief Get the rWnd of the selected socket
   *
   * \param who socket where check the parameter
   * \return the received advertised window
   */
  uint32_t GetRWnd (SocketWho who);

  /**
   * \brief Get the persistent event of the selected socket
   *
   * \param who socket where check the parameter
   * \return the persistent event in the selected socket
   */
  EventId GetPersistentEvent (SocketWho who);

  /**
   * \brief Get the persistent timeout of the selected socket
   *
   * \param who socket where check the parameter
   * \return the persistent timeout in the selected socket
   */
  Time GetPersistentTimeout (SocketWho who);

  /**
   * \brief Forcefully set a defined size for rx buffer
   *
   * \param who socket to force
   * \param size size of the rx buffer
   */
  void SetRcvBufSize (SocketWho who, uint32_t size);

  /**
   * \brief Forcefully set the segment size
   *
   * \param who socket to force
   * \param segmentSize segmentSize
   */
  void SetSegmentSize (SocketWho who, uint32_t segmentSize);

  /**
   * \brief Forcefully set the initial cwnd
   *
   * \param who socket to force
   * \param initialCwnd size of the initial cwnd (segments)
   */
  void SetInitialCwnd (SocketWho who, uint32_t initialCwnd);

  /**
   * \brief Forcefully set the delayed acknowledgement count
   *
   * \param who socket to force
   * \param count value of delayed ACKs
   */
  void SetDelAckMaxCount (SocketWho who, uint32_t count);

  /**
   * \brief Forcefully set the ECN mode of use
   *
   * \param who socket to force
   * \param useEcn Value representing the mode of ECN usage requested
   */
  void SetUseEcn (SocketWho who, TcpSocketState::UseEcn_t useEcn);

  /**
   * \brief Enable or disable pacing in the TCP socket
   * 
   * \param who socket
   * \param pacing Boolean to enable or disable pacing
   */
  void SetPacingStatus (SocketWho who, bool pacing);

  /**
   * \brief Enable or disable pacing of the initial window
   * 
   * \param who socket
   * \param paceWindow Boolean to enable or disable pacing of initial window
   */
  void SetPaceInitialWindow (SocketWho who, bool paceWindow);

  /**
   * \brief Forcefully set the initial ssthresh
   *
   * \param who socket to force
   * \param initialSsThresh Initial slow start threshold (bytes)
   */
  void SetInitialSsThresh (SocketWho who, uint32_t initialSsThresh);

  /**
   * \brief Set app packet size
   *
   * The application will generate packet of this size.
   *
   * \param pktSize size of the packet
   */
  void SetAppPktSize (uint32_t pktSize) { m_pktSize = pktSize; }

  /**
   * \brief Set app packet count
   *
   * The application will generate this count of packets.
   *
   * \param pktCount count of packets to generate
   */
  void SetAppPktCount (uint32_t pktCount) { m_pktCount = pktCount; }

  /**
   * \brief Interval between app-generated packet
   *
   * \param pktInterval interval
   */
  void SetAppPktInterval (Time pktInterval) { m_interPacketInterval = pktInterval; }

  /**
   * \brief Propagation delay of the bottleneck link
   *
   * \param propDelay propagation delay
   */
  void SetPropagationDelay (Time propDelay) { m_propagationDelay = propDelay; }

  /**
   * \brief Set the initial time at which the application sends the first data packet
   *
   * \param startTime start time
   */
  void SetTransmitStart (Time startTime) { m_startTime = startTime; }

  /**
   * \brief Congestion control of the sender socket
   *
   * \param congControl typeid of the congestion control algorithm
   */
  void SetCongestionControl (TypeId congControl) { m_congControlTypeId = congControl; }

  /**
   * \brief recovery algorithm of the sender socket
   *
   * \param recovery typeid of the recovery algorithm
   */
  void SetRecoveryAlgorithm (TypeId recovery) { m_recoveryTypeId = recovery; }

  /**
   * \brief MTU of the bottleneck link
   *
   * \param mtu MTU
   */
  void SetMTU (uint32_t mtu) { m_mtu = mtu; }

  /**
   * \brief State on Ack state machine changes
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void CongStateTrace ([[maybe_unused]] const TcpSocketState::TcpCongState_t oldValue,
                               [[maybe_unused]] const TcpSocketState::TcpCongState_t newValue)
  {
  }

  /**
   * \brief Tracks the congestion window changes
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void CWndTrace ([[maybe_unused]] uint32_t oldValue,
                          [[maybe_unused]] uint32_t newValue)
  {
  }

  /**
   * \brief Tracks the inflated congestion window changes
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void CWndInflTrace ([[maybe_unused]] uint32_t oldValue,
                              [[maybe_unused]] uint32_t newValue)
  {
  }

  /**
   * \brief Rtt changes
   *
   * This applies only for sender socket.
   *
   * \param oldTime old value
   * \param newTime new value
   */
  virtual void RttTrace ([[maybe_unused]] Time oldTime,
                         [[maybe_unused]] Time newTime)
  {
  }

  /**
   *  \brief Slow start threshold changes
   *
   * This applies only for sender socket.
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void SsThreshTrace ([[maybe_unused]] uint32_t oldValue,
                              [[maybe_unused]] uint32_t newValue)
  {
  }

  /**
   * \brief Bytes in flight changes
   *
   * This applies only for sender socket.
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void BytesInFlightTrace ([[maybe_unused]] uint32_t oldValue,
                                   [[maybe_unused]] uint32_t newValue)
  {
  }

  /**
   * \brief RTO changes
   *
   * This applies only for sender socket.
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void RtoTrace ([[maybe_unused]] Time oldValue,
                         [[maybe_unused]] Time newValue)
  {
  }

  /**
   * \brief Next tx seq changes
   *
   * This applies only for sender socket.
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void NextTxSeqTrace ([[maybe_unused]] SequenceNumber32 oldValue,
                               [[maybe_unused]] SequenceNumber32 newValue)
  {
  }

  /**
   * \brief Highest tx seq changes
   *
   * This applies only for sender socket.
   *
   * \param oldValue old value
   * \param newValue new value
   */
  virtual void HighestTxSeqTrace ([[maybe_unused]] SequenceNumber32 oldValue,
                                  [[maybe_unused]] SequenceNumber32 newValue)
  {
  }

  /**
   * \brief Track the rate value of TcpRateLinux.
   * \param rate updated value of TcpRate.
   */
  virtual void RateUpdatedTrace ([[maybe_unused]] const TcpRateLinux::TcpRateConnection &rate)
  {
  }

  /**
   * \brief Track the rate sample value of TcpRateLinux.
   * \param sample updated value of TcpRateSample.
   */
  virtual void RateSampleUpdatedTrace ([[maybe_unused]] const TcpRateLinux::TcpRateSample &sample)
  {
  }

  /**
   * \brief Socket closed normally
   * \param who the socket closed (SENDER or RECEIVER)
   */
  virtual void NormalClose ([[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Socket closed with an error
   *
   * \param who the socket closed (SENDER or RECEIVER)
   */
  virtual void ErrorClose  ([[maybe_unused]] SocketWho who)
  {
    /** \todo indicate the error */
  }

  /**
   * \brief Drop on the queue
   * \param who where the drop occurred (SENDER or RECEIVER)
   */
  virtual void QueueDrop   ([[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Link drop
   * \param who where the drop occurred (SENDER or RECEIVER)
   */
  virtual void PhyDrop     ([[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Received ack
   *
   * Invoked when an ACK is received (no processing is done yet)
   *
   * \param tcb Transmission Control Block
   * \param h the header of segment
   * \param who the socket which has received the ACK (SENDER or RECEIVER)
   */
  virtual void RcvAck      ([[maybe_unused]] const Ptr<const TcpSocketState> tcb,
                            [[maybe_unused]] const TcpHeader& h,
                            [[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Processed ack
   *
   * Invoked after the processing of the ACK
   *
   * \param tcb Transmission Control Block
   * \param h the header of segment
   * \param who the socket which has processed the ACK (SENDER or RECEIVER)
   */
  virtual void ProcessedAck ([[maybe_unused]] const Ptr<const TcpSocketState> tcb,
                             [[maybe_unused]] const TcpHeader& h,
                             [[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Packet transmitted down to IP layer
   *
   * \param p packet
   * \param h header
   * \param who the socket which has received the packet (SENDER or RECEIVER)
   */
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);

  /**
   * \brief Packet received from IP layer
   *
   * \param p packet
   * \param h header
   * \param who  the socket which has received the packet (SENDER or RECEIVER)
   */
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);

  /**
   * \brief Rto has expired
   *
   * \param tcb Transmission control block
   * \param who where the RTO has expired (SENDER or RECEIVER)
   */
  virtual void AfterRTOExpired ([[maybe_unused]] const Ptr<const TcpSocketState> tcb,
                                [[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Rto has expired
   *
   * \param tcb Transmission control block
   * \param who where the RTO has expired (SENDER or RECEIVER)
   */
  virtual void BeforeRTOExpired ([[maybe_unused]] const Ptr<const TcpSocketState> tcb,
                                 [[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Updated the Rtt history
   * \param seq Sequence inserted
   * \param sz size
   * \param isRetransmission self-explanatory
   * \param who where the rtt history was updated
   */
  virtual void UpdatedRttHistory ([[maybe_unused]] const SequenceNumber32 & seq,
                                  [[maybe_unused]] uint32_t sz,
                                  [[maybe_unused]] bool isRetransmission,
                                  [[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Notifying application for sent data
   *
   * \param size the amount of bytes transmitted
   * \param who where the RTO has expired (SENDER or RECEIVER)
   */
  virtual void DataSent ([[maybe_unused]] uint32_t size,
                         [[maybe_unused]] SocketWho who)
  {
  }

  /**
   * \brief Performs the (eventual) final checks through test asserts
   */
  virtual void FinalChecks ()
  {
  }

  /**
   * \brief Get the channel Propagation Delay
   * \return propagation delay of the channel
   */
  Time GetPropagationDelay () const
  {
    return m_propagationDelay;
  }

  /**
   * \brief Get the data start time
   * \return start time of data packets
   */
  Time GetStartTime () const
  {
    return m_startTime;
  }

  /**
   * \brief Get the MTU of the environment
   * \return MTU of the environment
   */
  uint32_t GetMtu () const
  {
    return m_mtu;
  }

  /**
   * \brief Get the application packet size
   * \return application packet size
   */
  uint32_t GetPktSize () const
  {
    return m_pktSize;
  }

  /**
   * \brief Get the number of application packets
   * \return count of application packets to be generated
   */
  uint32_t GetPktCount () const
  {
    return m_pktCount;
  }

  /**
   * \brief Get the interval to wait for each packet sent down from application
   * to TCP
   * \return interval between packet
   */
  Time GetPktInterval () const
  {
    return m_interPacketInterval;
  }

  TypeId   m_congControlTypeId;      //!< Congestion control
  TypeId   m_recoveryTypeId;         //!< Recovery

private:
  // Member variables, accessible through getters
  // giving write access to subclass can be dangerous
  Time     m_propagationDelay;     //!< Propagation delay of the channel

  Time     m_startTime;            //!< Data transmission time
  uint32_t m_mtu;                  //!< MTU of the environment

  uint32_t m_pktSize;              //!< Size of the application packet
  uint32_t m_pktCount;             //!< Count of the application packet
  Time     m_interPacketInterval;  //!< Time between sending application packet down to tcp socket

  Ptr<TcpSocketMsgBase> m_senderSocket;   //!< Pointer to sender socket
  Ptr<TcpSocketMsgBase> m_receiverSocket; //!< Pointer to receiver socket

private:
  // De-multiplexing callbacks.

  /**
   * \brief Normal Close Callback.
   * \param socket The socket.
   */
  void NormalCloseCb  (Ptr<Socket> socket);
  /**
   * \brief Error Close Callback.
   * \param socket The socket.
   */
  void ErrorCloseCb   (Ptr<Socket> socket);
  /**
   * \brief Queue Drop Callback.
   * \param context The context.
   * \param p The packet.
   */
  void QueueDropCb    (std::string context, Ptr<const Packet> p);
  /**
   * \brief Drop at Phy layer Callback.
   * \param context The context.
   * \param p The packet.
   */
  void PhyDropCb      (std::string context, Ptr<const Packet> p);
  /**
   * \brief Receive ACK Callback.
   * \param p The packet.
   * \param h TCP header.
   * \param tcp The TCP socket.
   */
  void RcvAckCb       (Ptr<const Packet> p, const TcpHeader& h,
                       Ptr<const TcpSocketBase> tcp);
  /**
   * \brief ACK processed Callback.
   * \param p The packet.
   * \param h TCP header.
   * \param tcp The TCP socket.
   */
  void ProcessedAckCb (Ptr<const Packet> p, const TcpHeader& h,
                       Ptr<const TcpSocketBase> tcp);
  /**
   * \brief Tx packet Callback.
   * \param p The packet.
   * \param h TCP header.
   * \param tcp The TCP socket.
   */
  void TxPacketCb     (const Ptr<const Packet> p, const TcpHeader& h,
                       const Ptr<const TcpSocketBase> tcp);
  /**
   * \brief Rx packet Callback.
   * \param p The packet.
   * \param h TCP header.
   * \param tcp The TCP socket.
   */
  void RxPacketCb     (const Ptr<const Packet> p, const TcpHeader& h,
                       const Ptr<const TcpSocketBase> tcp);
  /**
   * \brief RTO expired Callback.
   * \param tcb Transmission control block.
   * \param tcp The TCP socket.
   */
  void RtoExpiredCb   (const Ptr<const TcpSocketState> tcb,
                       const Ptr<const TcpSocketBase> tcp);
  /**
   * \brief Update RTT with new data.
   * \param tcp The TCP socket.
   * \param seq The sequence number.
   * \param sz The segment size.
   * \param isRetransmission True if packet is a retransmission.
   */
  void UpdateRttHistoryCb (Ptr<const TcpSocketBase> tcp, const SequenceNumber32&seq,
                           uint32_t sz, bool isRetransmission);

  /**
   * \brief Invoked after a retransmit event.
   * \param tcb Transmission control block.
   * \param tcp The TCP socket.
   */
  void AfterRetransmitCb   (const Ptr<const TcpSocketState> tcb,
                            const Ptr<const TcpSocketBase> tcp);

  /**
   * \brief Invoked before a retransmit event.
   * \param tcb Transmission control block.
   * \param tcp The TCP socket.
   */
  void BeforeRetransmitCb   (const Ptr<const TcpSocketState> tcb,
                             const Ptr<const TcpSocketBase> tcp);

  /**
   * \brief Data sent Callback.
   * \param socket The socket.
   * \param size The data size.
   */
  void DataSentCb     (Ptr<Socket> socket, uint32_t size);
  /**
   * \brief Fork Callback.
   * \param tcp The TCP socket.
   */
  void ForkCb         (Ptr<TcpSocketMsgBase> tcp);
  /**
   * \brief Handle an accept connection.
   * \param socket The socket.
   * \param from The sender.
   */
  void HandleAccept (Ptr<Socket> socket, const Address& from);

  InetSocketAddress m_remoteAddr; //!< Remote peer address.
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Convenience function to retrieve the ACK state from a TCB
 *
 * \param tcb Transmission control block
 * \return the state of the ACK state machine
 */
static inline TcpSocketState::TcpCongState_t
GetCongStateFrom (Ptr<const TcpSocketState> tcb)
{
  return tcb->m_congState.Get ();
}

} // namespace ns3

#endif // TCPGENERALTEST_H

