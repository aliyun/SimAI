/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
 * Copyright (c) 2010 Adrian Sai-wah Tam
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
 * Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 */

#define NS_LOG_APPEND_CONTEXT \
  if (m_node) { std::clog << " [node " << m_node->GetId () << "] "; }

#include "ns3/abort.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/log.h"
#include "ns3/ipv4.h"
#include "ns3/ipv6.h"
#include "ns3/ipv4-interface-address.h"
#include "ns3/ipv4-route.h"
#include "ns3/ipv6-route.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/simulation-singleton.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/pointer.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/data-rate.h"
#include "ns3/object.h"
#include "tcp-socket-base.h"
#include "tcp-l4-protocol.h"
#include "ipv4-end-point.h"
#include "ipv6-end-point.h"
#include "ipv6-l3-protocol.h"
#include "tcp-tx-buffer.h"
#include "tcp-rx-buffer.h"
#include "rtt-estimator.h"
#include "tcp-header.h"
#include "tcp-option-winscale.h"
#include "tcp-option-ts.h"
#include "tcp-option-sack-permitted.h"
#include "tcp-option-sack.h"
#include "tcp-congestion-ops.h"
#include "tcp-recovery-ops.h"
#include "ns3/tcp-rate-ops.h"

#include <math.h>
#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpSocketBase");

NS_OBJECT_ENSURE_REGISTERED (TcpSocketBase);

TypeId
TcpSocketBase::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpSocketBase")
    .SetParent<TcpSocket> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpSocketBase> ()
//    .AddAttribute ("TcpState", "State in TCP state machine",
//                   TypeId::ATTR_GET,
//                   EnumValue (CLOSED),
//                   MakeEnumAccessor (&TcpSocketBase::m_state),
//                   MakeEnumChecker (CLOSED, "Closed"))
    .AddAttribute ("MaxSegLifetime",
                   "Maximum segment lifetime in seconds, use for TIME_WAIT state transition to CLOSED state",
                   DoubleValue (120), /* RFC793 says MSL=2 minutes*/
                   MakeDoubleAccessor (&TcpSocketBase::m_msl),
                   MakeDoubleChecker<double> (0))
    .AddAttribute ("MaxWindowSize", "Max size of advertised window",
                   UintegerValue (65535),
                   MakeUintegerAccessor (&TcpSocketBase::m_maxWinSize),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("IcmpCallback", "Callback invoked whenever an icmp error is received on this socket.",
                   CallbackValue (),
                   MakeCallbackAccessor (&TcpSocketBase::m_icmpCallback),
                   MakeCallbackChecker ())
    .AddAttribute ("IcmpCallback6", "Callback invoked whenever an icmpv6 error is received on this socket.",
                   CallbackValue (),
                   MakeCallbackAccessor (&TcpSocketBase::m_icmpCallback6),
                   MakeCallbackChecker ())
    .AddAttribute ("WindowScaling", "Enable or disable Window Scaling option",
                   BooleanValue (true),
                   MakeBooleanAccessor (&TcpSocketBase::m_winScalingEnabled),
                   MakeBooleanChecker ())
    .AddAttribute ("Sack", "Enable or disable Sack option",
                   BooleanValue (true),
                   MakeBooleanAccessor (&TcpSocketBase::m_sackEnabled),
                   MakeBooleanChecker ())
    .AddAttribute ("Timestamp", "Enable or disable Timestamp option",
                   BooleanValue (true),
                   MakeBooleanAccessor (&TcpSocketBase::m_timestampEnabled),
                   MakeBooleanChecker ())
    .AddAttribute ("MinRto",
                   "Minimum retransmit timeout value",
                   TimeValue (Seconds (1.0)), // RFC 6298 says min RTO=1 sec, but Linux uses 200ms.
                   // See http://www.postel.org/pipermail/end2end-interest/2004-November/004402.html
                   MakeTimeAccessor (&TcpSocketBase::SetMinRto,
                                     &TcpSocketBase::GetMinRto),
                   MakeTimeChecker ())
    .AddAttribute ("ClockGranularity",
                   "Clock Granularity used in RTO calculations",
                   TimeValue (MilliSeconds (1)), // RFC6298 suggest to use fine clock granularity
                   MakeTimeAccessor (&TcpSocketBase::SetClockGranularity,
                                     &TcpSocketBase::GetClockGranularity),
                   MakeTimeChecker ())
    .AddAttribute ("TxBuffer",
                   "TCP Tx buffer",
                   PointerValue (),
                   MakePointerAccessor (&TcpSocketBase::GetTxBuffer),
                   MakePointerChecker<TcpTxBuffer> ())
    .AddAttribute ("RxBuffer",
                   "TCP Rx buffer",
                   PointerValue (),
                   MakePointerAccessor (&TcpSocketBase::GetRxBuffer),
                   MakePointerChecker<TcpRxBuffer> ())
    .AddAttribute ("CongestionOps",
                   "Pointer to TcpCongestionOps object",
                   PointerValue (),
                   MakePointerAccessor (&TcpSocketBase::m_congestionControl),
                   MakePointerChecker<TcpCongestionOps> ())
    .AddAttribute ("ReTxThreshold", "Threshold for fast retransmit",
                   UintegerValue (3),
                   MakeUintegerAccessor (&TcpSocketBase::SetRetxThresh,
                                         &TcpSocketBase::GetRetxThresh),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("LimitedTransmit", "Enable limited transmit",
                   BooleanValue (true),
                   MakeBooleanAccessor (&TcpSocketBase::m_limitedTx),
                   MakeBooleanChecker ())
    .AddAttribute ("UseEcn", "Parameter to set ECN functionality",
                   EnumValue (TcpSocketState::Off),
                   MakeEnumAccessor (&TcpSocketBase::SetUseEcn),
                   MakeEnumChecker (TcpSocketState::Off, "Off",
                                    TcpSocketState::On, "On",
                                    TcpSocketState::AcceptOnly, "AcceptOnly"))
    .AddTraceSource ("RTO",
                     "Retransmission timeout",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_rto),
                     "ns3::TracedValueCallback::Time")
    .AddTraceSource ("RTT",
                     "Last RTT sample",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_lastRttTrace),
                     "ns3::TracedValueCallback::Time")
    .AddTraceSource ("NextTxSequence",
                     "Next sequence number to send (SND.NXT)",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_nextTxSequenceTrace),
                     "ns3::SequenceNumber32TracedValueCallback")
    .AddTraceSource ("HighestSequence",
                     "Highest sequence number ever sent in socket's life time",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_highTxMarkTrace),
                     "ns3::TracedValueCallback::SequenceNumber32")
    .AddTraceSource ("State",
                     "TCP state",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_state),
                     "ns3::TcpStatesTracedValueCallback")
    .AddTraceSource ("CongState",
                     "TCP Congestion machine state",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_congStateTrace),
                     "ns3::TcpSocketState::TcpCongStatesTracedValueCallback")
    .AddTraceSource ("EcnState",
                     "Trace ECN state change of socket",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_ecnStateTrace),
                     "ns3::TcpSocketState::EcnStatesTracedValueCallback")
    .AddTraceSource ("AdvWND",
                     "Advertised Window Size",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_advWnd),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("RWND",
                     "Remote side's flow control window",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_rWnd),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("BytesInFlight",
                     "Socket estimation of bytes in flight",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_bytesInFlightTrace),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("HighestRxSequence",
                     "Highest sequence number received from peer",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_highRxMark),
                     "ns3::TracedValueCallback::SequenceNumber32")
    .AddTraceSource ("HighestRxAck",
                     "Highest ack received from peer",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_highRxAckMark),
                     "ns3::TracedValueCallback::SequenceNumber32")
    .AddTraceSource ("PacingRate",
                     "The current TCP pacing rate",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_pacingRateTrace),
                     "ns3::TracedValueCallback::DataRate")
    .AddTraceSource ("CongestionWindow",
                     "The TCP connection's congestion window",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_cWndTrace),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("CongestionWindowInflated",
                     "The TCP connection's congestion window inflates as in older RFC",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_cWndInflTrace),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("SlowStartThreshold",
                     "TCP slow start threshold (bytes)",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_ssThTrace),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("Tx",
                     "Send tcp packet to IP protocol",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_txTrace),
                     "ns3::TcpSocketBase::TcpTxRxTracedCallback")
    .AddTraceSource ("Rx",
                     "Receive tcp packet from IP protocol",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_rxTrace),
                     "ns3::TcpSocketBase::TcpTxRxTracedCallback")
    .AddTraceSource ("EcnEchoSeq",
                     "Sequence of last received ECN Echo",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_ecnEchoSeq),
                     "ns3::SequenceNumber32TracedValueCallback")
    .AddTraceSource ("EcnCeSeq",
                     "Sequence of last received CE ",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_ecnCESeq),
                     "ns3::SequenceNumber32TracedValueCallback")
    .AddTraceSource ("EcnCwrSeq",
                     "Sequence of last received CWR",
                     MakeTraceSourceAccessor (&TcpSocketBase::m_ecnCWRSeq),
                     "ns3::SequenceNumber32TracedValueCallback")
  ;
  return tid;
}

TypeId
TcpSocketBase::GetInstanceTypeId () const
{
  return TcpSocketBase::GetTypeId ();
}

TcpSocketBase::TcpSocketBase (void)
  : TcpSocket ()
{
  NS_LOG_FUNCTION (this);
  m_txBuffer = CreateObject<TcpTxBuffer> ();
  m_txBuffer->SetRWndCallback (MakeCallback (&TcpSocketBase::GetRWnd, this));
  m_tcb      = CreateObject<TcpSocketState> ();
  m_rateOps  = CreateObject <TcpRateLinux> ();

  m_tcb->m_rxBuffer = CreateObject<TcpRxBuffer> ();

  m_tcb->m_pacingRate = m_tcb->m_maxPacingRate;
  m_pacingTimer.SetFunction (&TcpSocketBase::NotifyPacingPerformed, this);

  m_tcb->m_sendEmptyPacketCallback = MakeCallback (&TcpSocketBase::SendEmptyPacket, this);

  bool ok;

  ok = m_tcb->TraceConnectWithoutContext ("PacingRate",
                                          MakeCallback (&TcpSocketBase::UpdatePacingRateTrace, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("CongestionWindow",
                                          MakeCallback (&TcpSocketBase::UpdateCwnd, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("CongestionWindowInflated",
                                          MakeCallback (&TcpSocketBase::UpdateCwndInfl, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("SlowStartThreshold",
                                          MakeCallback (&TcpSocketBase::UpdateSsThresh, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("CongState",
                                          MakeCallback (&TcpSocketBase::UpdateCongState, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("EcnState",
                                          MakeCallback (&TcpSocketBase::UpdateEcnState, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("NextTxSequence",
                                          MakeCallback (&TcpSocketBase::UpdateNextTxSequence, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("HighestSequence",
                                          MakeCallback (&TcpSocketBase::UpdateHighTxMark, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("BytesInFlight",
                                          MakeCallback (&TcpSocketBase::UpdateBytesInFlight, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("RTT",
                                          MakeCallback (&TcpSocketBase::UpdateRtt, this));
  NS_ASSERT (ok == true);
}

TcpSocketBase::TcpSocketBase (const TcpSocketBase& sock)
  : TcpSocket (sock),
    //copy object::m_tid and socket::callbacks
    m_dupAckCount (sock.m_dupAckCount),
    m_delAckCount (0),
    m_delAckMaxCount (sock.m_delAckMaxCount),
    m_noDelay (sock.m_noDelay),
    m_synCount (sock.m_synCount),
    m_synRetries (sock.m_synRetries),
    m_dataRetrCount (sock.m_dataRetrCount),
    m_dataRetries (sock.m_dataRetries),
    m_rto (sock.m_rto),
    m_minRto (sock.m_minRto),
    m_clockGranularity (sock.m_clockGranularity),
    m_delAckTimeout (sock.m_delAckTimeout),
    m_persistTimeout (sock.m_persistTimeout),
    m_cnTimeout (sock.m_cnTimeout),
    m_endPoint (nullptr),
    m_endPoint6 (nullptr),
    m_node (sock.m_node),
    m_tcp (sock.m_tcp),
    m_state (sock.m_state),
    m_errno (sock.m_errno),
    m_closeNotified (sock.m_closeNotified),
    m_closeOnEmpty (sock.m_closeOnEmpty),
    m_shutdownSend (sock.m_shutdownSend),
    m_shutdownRecv (sock.m_shutdownRecv),
    m_connected (sock.m_connected),
    m_msl (sock.m_msl),
    m_maxWinSize (sock.m_maxWinSize),
    m_bytesAckedNotProcessed (sock.m_bytesAckedNotProcessed),
    m_rWnd (sock.m_rWnd),
    m_highRxMark (sock.m_highRxMark),
    m_highRxAckMark (sock.m_highRxAckMark),
    m_sackEnabled (sock.m_sackEnabled),
    m_winScalingEnabled (sock.m_winScalingEnabled),
    m_rcvWindShift (sock.m_rcvWindShift),
    m_sndWindShift (sock.m_sndWindShift),
    m_timestampEnabled (sock.m_timestampEnabled),
    m_timestampToEcho (sock.m_timestampToEcho),
    m_recover (sock.m_recover),
    m_recoverActive (sock.m_recoverActive),
    m_retxThresh (sock.m_retxThresh),
    m_limitedTx (sock.m_limitedTx),
    m_isFirstPartialAck (sock.m_isFirstPartialAck),
    m_txTrace (sock.m_txTrace),
    m_rxTrace (sock.m_rxTrace),
    m_pacingTimer (Timer::CANCEL_ON_DESTROY),
    m_ecnEchoSeq (sock.m_ecnEchoSeq),
    m_ecnCESeq (sock.m_ecnCESeq),
    m_ecnCWRSeq (sock.m_ecnCWRSeq)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC ("Invoked the copy constructor");
  // Copy the rtt estimator if it is set
  if (sock.m_rtt)
    {
      m_rtt = sock.m_rtt->Copy ();
    }
  // Reset all callbacks to null
  Callback<void, Ptr< Socket > > vPS = MakeNullCallback<void, Ptr<Socket> > ();
  Callback<void, Ptr<Socket>, const Address &> vPSA = MakeNullCallback<void, Ptr<Socket>, const Address &> ();
  Callback<void, Ptr<Socket>, uint32_t> vPSUI = MakeNullCallback<void, Ptr<Socket>, uint32_t> ();
  SetConnectCallback (vPS, vPS);
  SetDataSentCallback (vPSUI);
  SetSendCallback (vPSUI);
  SetRecvCallback (vPS);
  m_txBuffer = CopyObject (sock.m_txBuffer);
  m_txBuffer->SetRWndCallback (MakeCallback (&TcpSocketBase::GetRWnd, this));
  m_tcb = CopyObject (sock.m_tcb);
  m_tcb->m_rxBuffer = CopyObject (sock.m_tcb->m_rxBuffer);

  m_tcb->m_pacingRate = m_tcb->m_maxPacingRate;
  m_pacingTimer.SetFunction (&TcpSocketBase::NotifyPacingPerformed, this);

  if (sock.m_congestionControl)
    {
      m_congestionControl = sock.m_congestionControl->Fork ();
      m_congestionControl->Init (m_tcb);
    }

  if (sock.m_recoveryOps)
    {
      m_recoveryOps = sock.m_recoveryOps->Fork ();
    }

  m_rateOps = CreateObject <TcpRateLinux> ();
  if (m_tcb->m_sendEmptyPacketCallback.IsNull ())
    {
      m_tcb->m_sendEmptyPacketCallback = MakeCallback (&TcpSocketBase::SendEmptyPacket, this);
    }

  bool ok;

  ok = m_tcb->TraceConnectWithoutContext ("PacingRate",
                                          MakeCallback (&TcpSocketBase::UpdatePacingRateTrace, this));

  ok = m_tcb->TraceConnectWithoutContext ("CongestionWindow",
                                          MakeCallback (&TcpSocketBase::UpdateCwnd, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("CongestionWindowInflated",
                                          MakeCallback (&TcpSocketBase::UpdateCwndInfl, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("SlowStartThreshold",
                                          MakeCallback (&TcpSocketBase::UpdateSsThresh, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("CongState",
                                          MakeCallback (&TcpSocketBase::UpdateCongState, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("EcnState",
                                          MakeCallback (&TcpSocketBase::UpdateEcnState, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("NextTxSequence",
                                          MakeCallback (&TcpSocketBase::UpdateNextTxSequence, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("HighestSequence",
                                          MakeCallback (&TcpSocketBase::UpdateHighTxMark, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("BytesInFlight",
                                          MakeCallback (&TcpSocketBase::UpdateBytesInFlight, this));
  NS_ASSERT (ok == true);

  ok = m_tcb->TraceConnectWithoutContext ("RTT",
                                          MakeCallback (&TcpSocketBase::UpdateRtt, this));
  NS_ASSERT (ok == true);
}

TcpSocketBase::~TcpSocketBase (void)
{
  NS_LOG_FUNCTION (this);
  m_node = nullptr;
  if (m_endPoint != nullptr)
    {
      NS_ASSERT (m_tcp != nullptr);
      /*
       * Upon Bind, an Ipv4Endpoint is allocated and set to m_endPoint, and
       * DestroyCallback is set to TcpSocketBase::Destroy. If we called
       * m_tcp->DeAllocate, it will destroy its Ipv4EndpointDemux::DeAllocate,
       * which in turn destroys my m_endPoint, and in turn invokes
       * TcpSocketBase::Destroy to nullify m_node, m_endPoint, and m_tcp.
       */
      NS_ASSERT (m_endPoint != nullptr);
      m_tcp->DeAllocate (m_endPoint);
      NS_ASSERT (m_endPoint == nullptr);
    }
  if (m_endPoint6 != nullptr)
    {
      NS_ASSERT (m_tcp != nullptr);
      NS_ASSERT (m_endPoint6 != nullptr);
      m_tcp->DeAllocate (m_endPoint6);
      NS_ASSERT (m_endPoint6 == nullptr);
    }
  m_tcp = 0;
  CancelAllTimers ();
}

/* Associate a node with this TCP socket */
void
TcpSocketBase::SetNode (Ptr<Node> node)
{
  m_node = node;
}

/* Associate the L4 protocol (e.g. mux/demux) with this socket */
void
TcpSocketBase::SetTcp (Ptr<TcpL4Protocol> tcp)
{
  m_tcp = tcp;
}

/* Set an RTT estimator with this socket */
void
TcpSocketBase::SetRtt (Ptr<RttEstimator> rtt)
{
  m_rtt = rtt;
}

/* Inherit from Socket class: Returns error code */
enum Socket::SocketErrno
TcpSocketBase::GetErrno (void) const
{
  return m_errno;
}

/* Inherit from Socket class: Returns socket type, NS3_SOCK_STREAM */
enum Socket::SocketType
TcpSocketBase::GetSocketType (void) const
{
  return NS3_SOCK_STREAM;
}

/* Inherit from Socket class: Returns associated node */
Ptr<Node>
TcpSocketBase::GetNode (void) const
{
  return m_node;
}

/* Inherit from Socket class: Bind socket to an end-point in TcpL4Protocol */
int
TcpSocketBase::Bind (void)
{
  NS_LOG_FUNCTION (this);
  m_endPoint = m_tcp->Allocate ();
  if (0 == m_endPoint)
    {
      m_errno = ERROR_ADDRNOTAVAIL;
      return -1;
    }

  m_tcp->AddSocket (this);

  return SetupCallback ();
}

int
TcpSocketBase::Bind6 (void)
{
  NS_LOG_FUNCTION (this);
  m_endPoint6 = m_tcp->Allocate6 ();
  if (0 == m_endPoint6)
    {
      m_errno = ERROR_ADDRNOTAVAIL;
      return -1;
    }

  m_tcp->AddSocket (this);

  return SetupCallback ();
}

/* Inherit from Socket class: Bind socket (with specific address) to an end-point in TcpL4Protocol */
int
TcpSocketBase::Bind (const Address &address)
{
  NS_LOG_FUNCTION (this << address);
  if (InetSocketAddress::IsMatchingType (address))
    {
      InetSocketAddress transport = InetSocketAddress::ConvertFrom (address);
      Ipv4Address ipv4 = transport.GetIpv4 ();
      uint16_t port = transport.GetPort ();
      SetIpTos (transport.GetTos ());
      if (ipv4 == Ipv4Address::GetAny () && port == 0)
        {
          m_endPoint = m_tcp->Allocate ();
        }
      else if (ipv4 == Ipv4Address::GetAny () && port != 0)
        {
          m_endPoint = m_tcp->Allocate (GetBoundNetDevice (), port);
        }
      else if (ipv4 != Ipv4Address::GetAny () && port == 0)
        {
          m_endPoint = m_tcp->Allocate (ipv4);
        }
      else if (ipv4 != Ipv4Address::GetAny () && port != 0)
        {
          m_endPoint = m_tcp->Allocate (GetBoundNetDevice (), ipv4, port);
        }
      if (0 == m_endPoint)
        {
          m_errno = port ? ERROR_ADDRINUSE : ERROR_ADDRNOTAVAIL;
          return -1;
        }
    }
  else if (Inet6SocketAddress::IsMatchingType (address))
    {
      Inet6SocketAddress transport = Inet6SocketAddress::ConvertFrom (address);
      Ipv6Address ipv6 = transport.GetIpv6 ();
      uint16_t port = transport.GetPort ();
      if (ipv6 == Ipv6Address::GetAny () && port == 0)
        {
          m_endPoint6 = m_tcp->Allocate6 ();
        }
      else if (ipv6 == Ipv6Address::GetAny () && port != 0)
        {
          m_endPoint6 = m_tcp->Allocate6 (GetBoundNetDevice (), port);
        }
      else if (ipv6 != Ipv6Address::GetAny () && port == 0)
        {
          m_endPoint6 = m_tcp->Allocate6 (ipv6);
        }
      else if (ipv6 != Ipv6Address::GetAny () && port != 0)
        {
          m_endPoint6 = m_tcp->Allocate6 (GetBoundNetDevice (), ipv6, port);
        }
      if (0 == m_endPoint6)
        {
          m_errno = port ? ERROR_ADDRINUSE : ERROR_ADDRNOTAVAIL;
          return -1;
        }
    }
  else
    {
      m_errno = ERROR_INVAL;
      return -1;
    }

  m_tcp->AddSocket (this);

  NS_LOG_LOGIC ("TcpSocketBase " << this << " got an endpoint: " << m_endPoint);

  return SetupCallback ();
}

void
TcpSocketBase::SetInitialSSThresh (uint32_t threshold)
{
  NS_ABORT_MSG_UNLESS ( (m_state == CLOSED) || threshold == m_tcb->m_initialSsThresh,
                        "TcpSocketBase::SetSSThresh() cannot change initial ssThresh after connection started.");

  m_tcb->m_initialSsThresh = threshold;
}

uint32_t
TcpSocketBase::GetInitialSSThresh (void) const
{
  return m_tcb->m_initialSsThresh;
}

void
TcpSocketBase::SetInitialCwnd (uint32_t cwnd)
{
  NS_ABORT_MSG_UNLESS ( (m_state == CLOSED) || cwnd == m_tcb->m_initialCWnd,
                        "TcpSocketBase::SetInitialCwnd() cannot change initial cwnd after connection started.");

  m_tcb->m_initialCWnd = cwnd;
}

uint32_t
TcpSocketBase::GetInitialCwnd (void) const
{
  return m_tcb->m_initialCWnd;
}

/* Inherit from Socket class: Initiate connection to a remote address:port */
int
TcpSocketBase::Connect (const Address & address)
{
  NS_LOG_FUNCTION (this << address);

  // If haven't do so, Bind() this socket first
  if (InetSocketAddress::IsMatchingType (address))
    {
      if (m_endPoint == nullptr)
        {
          if (Bind () == -1)
            {
              NS_ASSERT (m_endPoint == nullptr);
              return -1; // Bind() failed
            }
          NS_ASSERT (m_endPoint != nullptr);
        }
      InetSocketAddress transport = InetSocketAddress::ConvertFrom (address);
      m_endPoint->SetPeer (transport.GetIpv4 (), transport.GetPort ());
      SetIpTos (transport.GetTos ());
      m_endPoint6 = nullptr;

      // Get the appropriate local address and port number from the routing protocol and set up endpoint
      if (SetupEndpoint () != 0)
        {
          NS_LOG_ERROR ("Route to destination does not exist ?!");
          return -1;
        }
    }
  else if (Inet6SocketAddress::IsMatchingType (address))
    {
      // If we are operating on a v4-mapped address, translate the address to
      // a v4 address and re-call this function
      Inet6SocketAddress transport = Inet6SocketAddress::ConvertFrom (address);
      Ipv6Address v6Addr = transport.GetIpv6 ();
      if (v6Addr.IsIpv4MappedAddress () == true)
        {
          Ipv4Address v4Addr = v6Addr.GetIpv4MappedAddress ();
          return Connect (InetSocketAddress (v4Addr, transport.GetPort ()));
        }

      if (m_endPoint6 == nullptr)
        {
          if (Bind6 () == -1)
            {
              NS_ASSERT (m_endPoint6 == nullptr);
              return -1; // Bind() failed
            }
          NS_ASSERT (m_endPoint6 != nullptr);
        }
      m_endPoint6->SetPeer (v6Addr, transport.GetPort ());
      m_endPoint = nullptr;

      // Get the appropriate local address and port number from the routing protocol and set up endpoint
      if (SetupEndpoint6 () != 0)
        {
          NS_LOG_ERROR ("Route to destination does not exist ?!");
          return -1;
        }
    }
  else
    {
      m_errno = ERROR_INVAL;
      return -1;
    }

  // Re-initialize parameters in case this socket is being reused after CLOSE
  m_rtt->Reset ();
  m_synCount = m_synRetries;
  m_dataRetrCount = m_dataRetries;

  // DoConnect() will do state-checking and send a SYN packet
  return DoConnect ();
}

/* Inherit from Socket class: Listen on the endpoint for an incoming connection */
int
TcpSocketBase::Listen (void)
{
  NS_LOG_FUNCTION (this);

  // Linux quits EINVAL if we're not in CLOSED state, so match what they do
  if (m_state != CLOSED)
    {
      m_errno = ERROR_INVAL;
      return -1;
    }
  // In other cases, set the state to LISTEN and done
  NS_LOG_DEBUG ("CLOSED -> LISTEN");
  m_state = LISTEN;
  return 0;
}

/* Inherit from Socket class: Kill this socket and signal the peer (if any) */
int
TcpSocketBase::Close (void)
{
  NS_LOG_FUNCTION (this);
  /// \internal
  /// First we check to see if there is any unread rx data.
  /// \bugid{426} claims we should send reset in this case.
  if (m_tcb->m_rxBuffer->Size () != 0)
    {
      NS_LOG_WARN ("Socket " << this << " << unread rx data during close.  Sending reset." <<
                   "This is probably due to a bad sink application; check its code");
      SendRST ();
      return 0;
    }

  if (m_txBuffer->SizeFromSequence (m_tcb->m_nextTxSequence) > 0)
    { // App close with pending data must wait until all data transmitted
      if (m_closeOnEmpty == false)
        {
          m_closeOnEmpty = true;
          NS_LOG_INFO ("Socket " << this << " deferring close, state " << TcpStateName[m_state]);
        }
      return 0;
    }
  return DoClose ();
}

/* Inherit from Socket class: Signal a termination of send */
int
TcpSocketBase::ShutdownSend (void)
{
  NS_LOG_FUNCTION (this);

  //this prevents data from being added to the buffer
  m_shutdownSend = true;
  m_closeOnEmpty = true;
  //if buffer is already empty, send a fin now
  //otherwise fin will go when buffer empties.
  if (m_txBuffer->Size () == 0)
    {
      if (m_state == ESTABLISHED || m_state == CLOSE_WAIT)
        {
          NS_LOG_INFO ("Empty tx buffer, send fin");
          SendEmptyPacket (TcpHeader::FIN);

          if (m_state == ESTABLISHED)
            { // On active close: I am the first one to send FIN
              NS_LOG_DEBUG ("ESTABLISHED -> FIN_WAIT_1");
              m_state = FIN_WAIT_1;
            }
          else
            { // On passive close: Peer sent me FIN already
              NS_LOG_DEBUG ("CLOSE_WAIT -> LAST_ACK");
              m_state = LAST_ACK;
            }
        }
    }

  return 0;
}

/* Inherit from Socket class: Signal a termination of receive */
int
TcpSocketBase::ShutdownRecv (void)
{
  NS_LOG_FUNCTION (this);
  m_shutdownRecv = true;
  return 0;
}

/* Inherit from Socket class: Send a packet. Parameter flags is not used.
    Packet has no TCP header. Invoked by upper-layer application */
int
TcpSocketBase::Send (Ptr<Packet> p, uint32_t flags)
{
  NS_LOG_FUNCTION (this << p);
  NS_ABORT_MSG_IF (flags, "use of flags is not supported in TcpSocketBase::Send()");
  if (m_state == ESTABLISHED || m_state == SYN_SENT || m_state == CLOSE_WAIT)
    {
      // Store the packet into Tx buffer
      if (!m_txBuffer->Add (p))
        { // TxBuffer overflow, send failed
          m_errno = ERROR_MSGSIZE;
          return -1;
        }
      if (m_shutdownSend)
        {
          m_errno = ERROR_SHUTDOWN;
          return -1;
        }

      m_rateOps->CalculateAppLimited(m_tcb->m_cWnd, m_tcb->m_bytesInFlight, m_tcb->m_segmentSize,
                                     m_txBuffer->TailSequence (), m_tcb->m_nextTxSequence,
                                     m_txBuffer->GetLost (), m_txBuffer->GetRetransmitsCount ());

      // Submit the data to lower layers
      NS_LOG_LOGIC ("txBufSize=" << m_txBuffer->Size () << " state " << TcpStateName[m_state]);
      if ((m_state == ESTABLISHED || m_state == CLOSE_WAIT) && AvailableWindow () > 0)
        { // Try to send the data out: Add a little step to allow the application
          // to fill the buffer
          if (!m_sendPendingDataEvent.IsRunning ())
            {
              m_sendPendingDataEvent = Simulator::Schedule (TimeStep (1),
                                                            &TcpSocketBase::SendPendingData,
                                                            this, m_connected);
            }
        }
      return p->GetSize ();
    }
  else
    { // Connection not established yet
      m_errno = ERROR_NOTCONN;
      return -1; // Send failure
    }
}

/* Inherit from Socket class: In TcpSocketBase, it is same as Send() call */
int
TcpSocketBase::SendTo (Ptr<Packet> p, uint32_t flags, [[maybe_unused]] const Address &address)
{
  return Send (p, flags); // SendTo() and Send() are the same
}

/* Inherit from Socket class: Return data to upper-layer application. Parameter flags
   is not used. Data is returned as a packet of size no larger than maxSize */
Ptr<Packet>
TcpSocketBase::Recv (uint32_t maxSize, uint32_t flags)
{
  NS_LOG_FUNCTION (this);
  NS_ABORT_MSG_IF (flags, "use of flags is not supported in TcpSocketBase::Recv()");
  if (m_tcb->m_rxBuffer->Size () == 0 && m_state == CLOSE_WAIT)
    {
      return Create<Packet> (); // Send EOF on connection close
    }
  Ptr<Packet> outPacket = m_tcb->m_rxBuffer->Extract (maxSize);
  return outPacket;
}

/* Inherit from Socket class: Recv and return the remote's address */
Ptr<Packet>
TcpSocketBase::RecvFrom (uint32_t maxSize, uint32_t flags, Address &fromAddress)
{
  NS_LOG_FUNCTION (this << maxSize << flags);
  Ptr<Packet> packet = Recv (maxSize, flags);
  // Null packet means no data to read, and an empty packet indicates EOF
  if (packet != nullptr && packet->GetSize () != 0)
    {
      if (m_endPoint != nullptr)
        {
          fromAddress = InetSocketAddress (m_endPoint->GetPeerAddress (), m_endPoint->GetPeerPort ());
        }
      else if (m_endPoint6 != nullptr)
        {
          fromAddress = Inet6SocketAddress (m_endPoint6->GetPeerAddress (), m_endPoint6->GetPeerPort ());
        }
      else
        {
          fromAddress = InetSocketAddress (Ipv4Address::GetZero (), 0);
        }
    }
  return packet;
}

/* Inherit from Socket class: Get the max number of bytes an app can send */
uint32_t
TcpSocketBase::GetTxAvailable (void) const
{
  NS_LOG_FUNCTION (this);
  return m_txBuffer->Available ();
}

/* Inherit from Socket class: Get the max number of bytes an app can read */
uint32_t
TcpSocketBase::GetRxAvailable (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tcb->m_rxBuffer->Available ();
}

/* Inherit from Socket class: Return local address:port */
int
TcpSocketBase::GetSockName (Address &address) const
{
  NS_LOG_FUNCTION (this);
  if (m_endPoint != nullptr)
    {
      address = InetSocketAddress (m_endPoint->GetLocalAddress (), m_endPoint->GetLocalPort ());
    }
  else if (m_endPoint6 != nullptr)
    {
      address = Inet6SocketAddress (m_endPoint6->GetLocalAddress (), m_endPoint6->GetLocalPort ());
    }
  else
    { // It is possible to call this method on a socket without a name
      // in which case, behavior is unspecified
      // Should this return an InetSocketAddress or an Inet6SocketAddress?
      address = InetSocketAddress (Ipv4Address::GetZero (), 0);
    }
  return 0;
}

int
TcpSocketBase::GetPeerName (Address &address) const
{
  NS_LOG_FUNCTION (this << address);

  if (!m_endPoint && !m_endPoint6)
    {
      m_errno = ERROR_NOTCONN;
      return -1;
    }

  if (m_endPoint)
    {
      address = InetSocketAddress (m_endPoint->GetPeerAddress (),
                                   m_endPoint->GetPeerPort ());
    }
  else if (m_endPoint6)
    {
      address = Inet6SocketAddress (m_endPoint6->GetPeerAddress (),
                                    m_endPoint6->GetPeerPort ());
    }
  else
    {
      NS_ASSERT (false);
    }

  return 0;
}

/* Inherit from Socket class: Bind this socket to the specified NetDevice */
void
TcpSocketBase::BindToNetDevice (Ptr<NetDevice> netdevice)
{
  NS_LOG_FUNCTION (netdevice);
  Socket::BindToNetDevice (netdevice); // Includes sanity check
  if (m_endPoint != nullptr)
    {
      m_endPoint->BindToNetDevice (netdevice);
    }

  if (m_endPoint6 != nullptr)
    {
      m_endPoint6->BindToNetDevice (netdevice);
    }

  return;
}

/* Clean up after Bind. Set up callback functions in the end-point. */
int
TcpSocketBase::SetupCallback (void)
{
  NS_LOG_FUNCTION (this);

  if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
      return -1;
    }
  if (m_endPoint != nullptr)
    {
      m_endPoint->SetRxCallback (MakeCallback (&TcpSocketBase::ForwardUp, Ptr<TcpSocketBase> (this)));
      m_endPoint->SetIcmpCallback (MakeCallback (&TcpSocketBase::ForwardIcmp, Ptr<TcpSocketBase> (this)));
      m_endPoint->SetDestroyCallback (MakeCallback (&TcpSocketBase::Destroy, Ptr<TcpSocketBase> (this)));
    }
  if (m_endPoint6 != nullptr)
    {
      m_endPoint6->SetRxCallback (MakeCallback (&TcpSocketBase::ForwardUp6, Ptr<TcpSocketBase> (this)));
      m_endPoint6->SetIcmpCallback (MakeCallback (&TcpSocketBase::ForwardIcmp6, Ptr<TcpSocketBase> (this)));
      m_endPoint6->SetDestroyCallback (MakeCallback (&TcpSocketBase::Destroy6, Ptr<TcpSocketBase> (this)));
    }

  return 0;
}

/* Perform the real connection tasks: Send SYN if allowed, RST if invalid */
int
TcpSocketBase::DoConnect (void)
{
  NS_LOG_FUNCTION (this);

  // A new connection is allowed only if this socket does not have a connection
  if (m_state == CLOSED || m_state == LISTEN || m_state == SYN_SENT || m_state == LAST_ACK || m_state == CLOSE_WAIT)
    { // send a SYN packet and change state into SYN_SENT
      // send a SYN packet with ECE and CWR flags set if sender is ECN capable
      if (m_tcb->m_useEcn == TcpSocketState::On)
        {
          SendEmptyPacket (TcpHeader::SYN | TcpHeader::ECE | TcpHeader::CWR);
        }
      else
        {
          SendEmptyPacket (TcpHeader::SYN);
        }
      NS_LOG_DEBUG (TcpStateName[m_state] << " -> SYN_SENT");
      m_state = SYN_SENT;
      m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;    // because sender is not yet aware about receiver's ECN capability
    }
  else if (m_state != TIME_WAIT)
    { // In states SYN_RCVD, ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2, and CLOSING, an connection
      // exists. We send RST, tear down everything, and close this socket.
      SendRST ();
      CloseAndNotify ();
    }
  return 0;
}

/* Do the action to close the socket. Usually send a packet with appropriate
    flags depended on the current m_state. */
int
TcpSocketBase::DoClose (void)
{
  NS_LOG_FUNCTION (this);
  switch (m_state)
    {
    case SYN_RCVD:
    case ESTABLISHED:
      // send FIN to close the peer
      SendEmptyPacket (TcpHeader::FIN);
      NS_LOG_DEBUG ("ESTABLISHED -> FIN_WAIT_1");
      m_state = FIN_WAIT_1;
      break;
    case CLOSE_WAIT:
      // send FIN+ACK to close the peer
      SendEmptyPacket (TcpHeader::FIN | TcpHeader::ACK);
      NS_LOG_DEBUG ("CLOSE_WAIT -> LAST_ACK");
      m_state = LAST_ACK;
      break;
    case SYN_SENT:
    case CLOSING:
      // Send RST if application closes in SYN_SENT and CLOSING
      SendRST ();
      CloseAndNotify ();
      break;
    case LISTEN:
      // In this state, move to CLOSED and tear down the end point
      CloseAndNotify ();
      break;
    case LAST_ACK:
    case CLOSED:
    case FIN_WAIT_1:
    case FIN_WAIT_2:
    case TIME_WAIT:
    default: /* mute compiler */
      // Do nothing in these five states
      break;
    }
  return 0;
}

/* Peacefully close the socket by notifying the upper layer and deallocate end point */
void
TcpSocketBase::CloseAndNotify (void)
{
  NS_LOG_FUNCTION (this);

  if (!m_closeNotified)
    {
      NotifyNormalClose ();
      m_closeNotified = true;
    }
  if (m_lastAckEvent.IsRunning ())
    {
      m_lastAckEvent.Cancel ();
    }
  NS_LOG_DEBUG (TcpStateName[m_state] << " -> CLOSED");
  m_state = CLOSED;
  DeallocateEndPoint ();
}


/* Tell if a sequence number range is out side the range that my rx buffer can
    accpet */
bool
TcpSocketBase::OutOfRange (SequenceNumber32 head, SequenceNumber32 tail) const
{
  if (m_state == LISTEN || m_state == SYN_SENT || m_state == SYN_RCVD)
    { // Rx buffer in these states are not initialized.
      return false;
    }
  if (m_state == LAST_ACK || m_state == CLOSING || m_state == CLOSE_WAIT)
    { // In LAST_ACK and CLOSING states, it only wait for an ACK and the
      // sequence number must equals to m_rxBuffer->NextRxSequence ()
      return (m_tcb->m_rxBuffer->NextRxSequence () != head);
    }

  // In all other cases, check if the sequence number is in range
  return (tail < m_tcb->m_rxBuffer->NextRxSequence () || m_tcb->m_rxBuffer->MaxRxSequence () <= head);
}

/* Function called by the L3 protocol when it received a packet to pass on to
    the TCP. This function is registered as the "RxCallback" function in
    SetupCallback(), which invoked by Bind(), and CompleteFork() */
void
TcpSocketBase::ForwardUp (Ptr<Packet> packet, Ipv4Header header, uint16_t port,
                          Ptr<Ipv4Interface> incomingInterface)
{
  NS_LOG_LOGIC ("Socket " << this << " forward up " <<
                m_endPoint->GetPeerAddress () <<
                ":" << m_endPoint->GetPeerPort () <<
                " to " << m_endPoint->GetLocalAddress () <<
                ":" << m_endPoint->GetLocalPort ());

  Address fromAddress = InetSocketAddress (header.GetSource (), port);
  Address toAddress = InetSocketAddress (header.GetDestination (),
                                         m_endPoint->GetLocalPort ());

  TcpHeader tcpHeader;
  uint32_t bytesRemoved = packet->PeekHeader (tcpHeader);

  if (!IsValidTcpSegment (tcpHeader.GetSequenceNumber (), bytesRemoved,
                          packet->GetSize () - bytesRemoved))
    {
      return;
    }

  if (header.GetEcn() == Ipv4Header::ECN_CE && m_ecnCESeq < tcpHeader.GetSequenceNumber ())
    {
      NS_LOG_INFO ("Received CE flag is valid");
      NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CE_RCVD");
      m_ecnCESeq = tcpHeader.GetSequenceNumber ();
      m_tcb->m_ecnState = TcpSocketState::ECN_CE_RCVD;
      m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_ECN_IS_CE);
    }
  else if (header.GetEcn() != Ipv4Header::ECN_NotECT && m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
    {
      m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_ECN_NO_CE);
    }

  DoForwardUp (packet, fromAddress, toAddress);
}

void
TcpSocketBase::ForwardUp6 (Ptr<Packet> packet, Ipv6Header header, uint16_t port,
                           Ptr<Ipv6Interface> incomingInterface)
{
  NS_LOG_LOGIC ("Socket " << this << " forward up " <<
                m_endPoint6->GetPeerAddress () <<
                ":" << m_endPoint6->GetPeerPort () <<
                " to " << m_endPoint6->GetLocalAddress () <<
                ":" << m_endPoint6->GetLocalPort ());

  Address fromAddress = Inet6SocketAddress (header.GetSource (), port);
  Address toAddress = Inet6SocketAddress (header.GetDestination (),
                                          m_endPoint6->GetLocalPort ());

  TcpHeader tcpHeader;
  uint32_t bytesRemoved = packet->PeekHeader (tcpHeader);

  if (!IsValidTcpSegment (tcpHeader.GetSequenceNumber (), bytesRemoved,
                          packet->GetSize () - bytesRemoved))
    {
      return;
    }

  if (header.GetEcn() == Ipv6Header::ECN_CE && m_ecnCESeq < tcpHeader.GetSequenceNumber ())
    {
      NS_LOG_INFO ("Received CE flag is valid");
      NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CE_RCVD");
      m_ecnCESeq = tcpHeader.GetSequenceNumber ();
      m_tcb->m_ecnState = TcpSocketState::ECN_CE_RCVD;
      m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_ECN_IS_CE);
    }
  else if (header.GetEcn() != Ipv6Header::ECN_NotECT)
    {
      m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_ECN_NO_CE);
    }

  DoForwardUp (packet, fromAddress, toAddress);
}

void
TcpSocketBase::ForwardIcmp (Ipv4Address icmpSource, uint8_t icmpTtl,
                            uint8_t icmpType, uint8_t icmpCode,
                            uint32_t icmpInfo)
{
  NS_LOG_FUNCTION (this << icmpSource << static_cast<uint32_t> (icmpTtl) <<
                   static_cast<uint32_t> (icmpType) <<
                   static_cast<uint32_t> (icmpCode) << icmpInfo);
  if (!m_icmpCallback.IsNull ())
    {
      m_icmpCallback (icmpSource, icmpTtl, icmpType, icmpCode, icmpInfo);
    }
}

void
TcpSocketBase::ForwardIcmp6 (Ipv6Address icmpSource, uint8_t icmpTtl,
                             uint8_t icmpType, uint8_t icmpCode,
                             uint32_t icmpInfo)
{
  NS_LOG_FUNCTION (this << icmpSource << static_cast<uint32_t> (icmpTtl) <<
                   static_cast<uint32_t> (icmpType) <<
                   static_cast<uint32_t> (icmpCode) << icmpInfo);
  if (!m_icmpCallback6.IsNull ())
    {
      m_icmpCallback6 (icmpSource, icmpTtl, icmpType, icmpCode, icmpInfo);
    }
}

bool
TcpSocketBase::IsValidTcpSegment (const SequenceNumber32 seq, const uint32_t tcpHeaderSize,
                                  const uint32_t tcpPayloadSize)
{
  if (tcpHeaderSize == 0 || tcpHeaderSize > 60)
    {
      NS_LOG_ERROR ("Bytes removed: " << tcpHeaderSize << " invalid");
      return false; // Discard invalid packet
    }
  else if (tcpPayloadSize > 0 && OutOfRange (seq, seq + tcpPayloadSize))
    {
      // Discard fully out of range data packets
      NS_LOG_WARN ("At state " << TcpStateName[m_state] <<
                   " received packet of seq [" << seq <<
                   ":" << seq + tcpPayloadSize <<
                   ") out of range [" << m_tcb->m_rxBuffer->NextRxSequence () << ":" <<
                   m_tcb->m_rxBuffer->MaxRxSequence () << ")");
      // Acknowledgement should be sent for all unacceptable packets (RFC793, p.69)
      SendEmptyPacket (TcpHeader::ACK);
      return false;
    }
  return true;
}

void
TcpSocketBase::DoForwardUp (Ptr<Packet> packet, const Address &fromAddress,
                            const Address &toAddress)
{
  // in case the packet still has a priority tag attached, remove it
  SocketPriorityTag priorityTag;
  packet->RemovePacketTag (priorityTag);

  // Peel off TCP header
  TcpHeader tcpHeader;
  packet->RemoveHeader (tcpHeader);
  SequenceNumber32 seq = tcpHeader.GetSequenceNumber ();

  if (m_state == ESTABLISHED && !(tcpHeader.GetFlags () & TcpHeader::RST))
    {
      // Check if the sender has responded to ECN echo by reducing the Congestion Window
      if (tcpHeader.GetFlags () & TcpHeader::CWR )
        {
          // Check if a packet with CE bit set is received. If there is no CE bit set, then change the state to ECN_IDLE to
          // stop sending ECN Echo messages. If there is CE bit set, the packet should continue sending ECN Echo messages
          //
          if (m_tcb->m_ecnState != TcpSocketState::ECN_CE_RCVD)
            {
              NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
              m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
            }
        }
    }

  m_rxTrace (packet, tcpHeader, this);

  if (tcpHeader.GetFlags () & TcpHeader::SYN)
    {
      /* The window field in a segment where the SYN bit is set (i.e., a <SYN>
       * or <SYN,ACK>) MUST NOT be scaled (from RFC 7323 page 9). But should be
       * saved anyway..
       */
      m_rWnd = tcpHeader.GetWindowSize ();

      if (tcpHeader.HasOption (TcpOption::WINSCALE) && m_winScalingEnabled)
        {
          ProcessOptionWScale (tcpHeader.GetOption (TcpOption::WINSCALE));
        }
      else
        {
          m_winScalingEnabled = false;
        }

      if (tcpHeader.HasOption (TcpOption::SACKPERMITTED) && m_sackEnabled)
        {
          ProcessOptionSackPermitted (tcpHeader.GetOption (TcpOption::SACKPERMITTED));
        }
      else
        {
          m_sackEnabled = false;
          m_txBuffer->SetSackEnabled (false);
        }

      // When receiving a <SYN> or <SYN-ACK> we should adapt TS to the other end
      if (tcpHeader.HasOption (TcpOption::TS) && m_timestampEnabled)
        {
          ProcessOptionTimestamp (tcpHeader.GetOption (TcpOption::TS),
                                  tcpHeader.GetSequenceNumber ());
        }
      else
        {
          m_timestampEnabled = false;
        }

      // Initialize cWnd and ssThresh
      m_tcb->m_cWnd = GetInitialCwnd () * GetSegSize ();
      m_tcb->m_cWndInfl = m_tcb->m_cWnd;
      m_tcb->m_ssThresh = GetInitialSSThresh ();

      if (tcpHeader.GetFlags () & TcpHeader::ACK)
        {
          EstimateRtt (tcpHeader);
          m_highRxAckMark = tcpHeader.GetAckNumber ();
        }
    }
  else if (tcpHeader.GetFlags () & TcpHeader::ACK)
    {
      NS_ASSERT (!(tcpHeader.GetFlags () & TcpHeader::SYN));
      if (m_timestampEnabled)
        {
          if (!tcpHeader.HasOption (TcpOption::TS))
            {
              // Ignoring segment without TS, RFC 7323
              NS_LOG_LOGIC ("At state " << TcpStateName[m_state] <<
                            " received packet of seq [" << seq <<
                            ":" << seq + packet->GetSize () <<
                            ") without TS option. Silently discard it");
              return;
            }
          else
            {
              ProcessOptionTimestamp (tcpHeader.GetOption (TcpOption::TS),
                                      tcpHeader.GetSequenceNumber ());
            }
        }

      EstimateRtt (tcpHeader);
      UpdateWindowSize (tcpHeader);
    }


  if (m_rWnd.Get () == 0 && m_persistEvent.IsExpired ())
    { // Zero window: Enter persist state to send 1 byte to probe
      NS_LOG_LOGIC (this << " Enter zerowindow persist state");
      NS_LOG_LOGIC (this << " Cancelled ReTxTimeout event which was set to expire at " <<
                    (Simulator::Now () + Simulator::GetDelayLeft (m_retxEvent)).GetSeconds ());
      m_retxEvent.Cancel ();
      NS_LOG_LOGIC ("Schedule persist timeout at time " <<
                    Simulator::Now ().GetSeconds () << " to expire at time " <<
                    (Simulator::Now () + m_persistTimeout).GetSeconds ());
      m_persistEvent = Simulator::Schedule (m_persistTimeout, &TcpSocketBase::PersistTimeout, this);
      NS_ASSERT (m_persistTimeout == Simulator::GetDelayLeft (m_persistEvent));
    }

  // TCP state machine code in different process functions
  // C.f.: tcp_rcv_state_process() in tcp_input.c in Linux kernel
  switch (m_state)
    {
    case ESTABLISHED:
      ProcessEstablished (packet, tcpHeader);
      break;
    case LISTEN:
      ProcessListen (packet, tcpHeader, fromAddress, toAddress);
      break;
    case TIME_WAIT:
      // Do nothing
      break;
    case CLOSED:
      // Send RST if the incoming packet is not a RST
      if ((tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG)) != TcpHeader::RST)
        { // Since m_endPoint is not configured yet, we cannot use SendRST here
          TcpHeader h;
          Ptr<Packet> p = Create<Packet> ();
          h.SetFlags (TcpHeader::RST);
          h.SetSequenceNumber (m_tcb->m_nextTxSequence);
          h.SetAckNumber (m_tcb->m_rxBuffer->NextRxSequence ());
          h.SetSourcePort (tcpHeader.GetDestinationPort ());
          h.SetDestinationPort (tcpHeader.GetSourcePort ());
          h.SetWindowSize (AdvertisedWindowSize ());
          AddOptions (h);
          m_txTrace (p, h, this);
          m_tcp->SendPacket (p, h, toAddress, fromAddress, m_boundnetdevice);
        }
      break;
    case SYN_SENT:
      ProcessSynSent (packet, tcpHeader);
      break;
    case SYN_RCVD:
      ProcessSynRcvd (packet, tcpHeader, fromAddress, toAddress);
      break;
    case FIN_WAIT_1:
    case FIN_WAIT_2:
    case CLOSE_WAIT:
      ProcessWait (packet, tcpHeader);
      break;
    case CLOSING:
      ProcessClosing (packet, tcpHeader);
      break;
    case LAST_ACK:
      ProcessLastAck (packet, tcpHeader);
      break;
    default: // mute compiler
      break;
    }

  if (m_rWnd.Get () != 0 && m_persistEvent.IsRunning ())
    { // persist probes end, the other end has increased the window
      NS_ASSERT (m_connected);
      NS_LOG_LOGIC (this << " Leaving zerowindow persist state");
      m_persistEvent.Cancel ();

      SendPendingData (m_connected);
    }
}

/* Received a packet upon ESTABLISHED state. This function is mimicking the
    role of tcp_rcv_established() in tcp_input.c in Linux kernel. */
void
TcpSocketBase::ProcessEstablished (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH, URG, CWR and ECE are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

  // Different flags are different events
  if (tcpflags == TcpHeader::ACK)
    {
      if (tcpHeader.GetAckNumber () < m_txBuffer->HeadSequence ())
        {
          // Case 1:  If the ACK is a duplicate (SEG.ACK < SND.UNA), it can be ignored.
          // Pag. 72 RFC 793
          NS_LOG_WARN ("Ignored ack of " << tcpHeader.GetAckNumber () <<
                       " SND.UNA = " << m_txBuffer->HeadSequence ());

          // TODO: RFC 5961 5.2 [Blind Data Injection Attack].[Mitigation]
        }
      else if (tcpHeader.GetAckNumber () > m_tcb->m_highTxMark)
        {
          // If the ACK acks something not yet sent (SEG.ACK > HighTxMark) then
          // send an ACK, drop the segment, and return.
          // Pag. 72 RFC 793
          NS_LOG_WARN ("Ignored ack of " << tcpHeader.GetAckNumber () <<
                       " HighTxMark = " << m_tcb->m_highTxMark);

          // Receiver sets ECE flags when it receives a packet with CE bit on or sender hasn’t responded to ECN echo sent by receiver
          if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
            {
              SendEmptyPacket (TcpHeader::ACK | TcpHeader::ECE);
              NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
              m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
            }
          else
            {
              SendEmptyPacket (TcpHeader::ACK);
            }
        }
      else
        {
          // SND.UNA < SEG.ACK =< HighTxMark
          // Pag. 72 RFC 793
          ReceivedAck (packet, tcpHeader);
        }
    }
  else if (tcpflags == TcpHeader::SYN)
    { // Received SYN, old NS-3 behaviour is to set state to SYN_RCVD and
      // respond with a SYN+ACK. But it is not a legal state transition as of
      // RFC793. Thus this is ignored.
    }
  else if (tcpflags == (TcpHeader::SYN | TcpHeader::ACK))
    { // No action for received SYN+ACK, it is probably a duplicated packet
    }
  else if (tcpflags == TcpHeader::FIN || tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
    { // Received FIN or FIN+ACK, bring down this socket nicely
      PeerClose (packet, tcpHeader);
    }
  else if (tcpflags == 0)
    { // No flags means there is only data
      ReceivedData (packet, tcpHeader);
      if (m_tcb->m_rxBuffer->Finished ())
        {
          PeerClose (packet, tcpHeader);
        }
    }
  else
    { // Received RST or the TCP flags is invalid, in either case, terminate this socket
      if (tcpflags != TcpHeader::RST)
        { // this must be an invalid flag, send reset
          NS_LOG_LOGIC ("Illegal flag " << TcpHeader::FlagsToString (tcpflags) << " received. Reset packet is sent.");
          SendRST ();
        }
      CloseAndNotify ();
    }
}

bool
TcpSocketBase::IsTcpOptionEnabled (uint8_t kind) const
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (kind));

  switch (kind)
    {
    case TcpOption::TS:
      return m_timestampEnabled;
    case TcpOption::WINSCALE:
      return m_winScalingEnabled;
    case TcpOption::SACKPERMITTED:
    case TcpOption::SACK:
      return m_sackEnabled;
    default:
      break;
    }
  return false;
}

void
TcpSocketBase::ReadOptions (const TcpHeader &tcpHeader, uint32_t *bytesSacked)
{
  NS_LOG_FUNCTION (this << tcpHeader);
  TcpHeader::TcpOptionList::const_iterator it;
  const TcpHeader::TcpOptionList options = tcpHeader.GetOptionList ();

  for (it = options.begin (); it != options.end (); ++it)
    {
      const Ptr<const TcpOption> option = (*it);

      // Check only for ACK options here
      switch (option->GetKind ())
        {
        case TcpOption::SACK:
          *bytesSacked = ProcessOptionSack (option);
          break;
        default:
          continue;
        }
    }
}

// Sender should reduce the Congestion Window as a response to receiver's
// ECN Echo notification only once per window
void
TcpSocketBase::EnterCwr (uint32_t currentDelivered)
{
  NS_LOG_FUNCTION (this << currentDelivered);
  m_tcb->m_ssThresh = m_congestionControl->GetSsThresh (m_tcb, BytesInFlight ());
  NS_LOG_DEBUG ("Reduce ssThresh to " << m_tcb->m_ssThresh);
  // Do not update m_cWnd, under assumption that recovery process will
  // gradually bring it down to m_ssThresh.  Update the 'inflated' value of
  // cWnd used for tracing, however.
  m_tcb->m_cWndInfl = m_tcb->m_ssThresh;
  NS_ASSERT (m_tcb->m_congState != TcpSocketState::CA_CWR);
  NS_LOG_DEBUG (TcpSocketState::TcpCongStateName[m_tcb->m_congState] << " -> CA_CWR");
  m_tcb->m_congState = TcpSocketState::CA_CWR;
  // CWR state will be exited when the ack exceeds the m_recover variable.
  // Do not set m_recoverActive (which applies to a loss-based recovery)
  // m_recover corresponds to Linux tp->high_seq
  m_recover = m_tcb->m_highTxMark;
  if (!m_congestionControl->HasCongControl ())
    {
      // If there is a recovery algorithm, invoke it.
      m_recoveryOps->EnterRecovery (m_tcb, m_dupAckCount, UnAckDataCount (), currentDelivered);
      NS_LOG_INFO ("Enter CWR recovery mode; set cwnd to " << m_tcb->m_cWnd
                    << ", ssthresh to " << m_tcb->m_ssThresh
                    << ", recover to " << m_recover);
    }
}

void
TcpSocketBase::EnterRecovery (uint32_t currentDelivered)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_tcb->m_congState != TcpSocketState::CA_RECOVERY);

  NS_LOG_DEBUG (TcpSocketState::TcpCongStateName[m_tcb->m_congState] <<
                " -> CA_RECOVERY");

  if (!m_sackEnabled)
    {
      // One segment has left the network, PLUS the head is lost
      m_txBuffer->AddRenoSack ();
      m_txBuffer->MarkHeadAsLost ();
    }
  else
    {
      if (!m_txBuffer->IsLost (m_txBuffer->HeadSequence ()))
        {
          // We received 3 dupacks, but the head is not marked as lost
          // (received less than 3 SACK block ahead).
          // Manually set it as lost.
          m_txBuffer->MarkHeadAsLost ();
        }
    }

  // RFC 6675, point (4):
  // (4) Invoke fast retransmit and enter loss recovery as follows:
  // (4.1) RecoveryPoint = HighData
  m_recover = m_tcb->m_highTxMark;
  m_recoverActive = true;

  m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_RECOVERY);
  m_tcb->m_congState = TcpSocketState::CA_RECOVERY;

  // (4.2) ssthresh = cwnd = (FlightSize / 2)
  // If SACK is not enabled, still consider the head as 'in flight' for
  // compatibility with old ns-3 versions
  uint32_t bytesInFlight = m_sackEnabled ? BytesInFlight () : BytesInFlight () + m_tcb->m_segmentSize;
  m_tcb->m_ssThresh = m_congestionControl->GetSsThresh (m_tcb, bytesInFlight);

  if (!m_congestionControl->HasCongControl ())
    {
      m_recoveryOps->EnterRecovery (m_tcb, m_dupAckCount, UnAckDataCount (), currentDelivered);
      NS_LOG_INFO (m_dupAckCount << " dupack. Enter fast recovery mode." <<
                  "Reset cwnd to " << m_tcb->m_cWnd << ", ssthresh to " <<
                   m_tcb->m_ssThresh << " at fast recovery seqnum " << m_recover <<
                   " calculated in flight: " << bytesInFlight);
    }


  // (4.3) Retransmit the first data segment presumed dropped
  DoRetransmit ();
  // (4.4) Run SetPipe ()
  // (4.5) Proceed to step (C)
  // these steps are done after the ProcessAck function (SendPendingData)
}

void
TcpSocketBase::DupAck (uint32_t currentDelivered)
{
  NS_LOG_FUNCTION (this);
  // NOTE: We do not count the DupAcks received in CA_LOSS, because we
  // don't know if they are generated by a spurious retransmission or because
  // of a real packet loss. With SACK, it is easy to know, but we do not consider
  // dupacks. Without SACK, there are some euristics in the RFC 6582, but
  // for now, we do not implement it, leading to ignoring the dupacks.
  if (m_tcb->m_congState == TcpSocketState::CA_LOSS)
    {
      return;
    }

  // RFC 6675, Section 5, 3rd paragraph:
  // If the incoming ACK is a duplicate acknowledgment per the definition
  // in Section 2 (regardless of its status as a cumulative
  // acknowledgment), and the TCP is not currently in loss recovery
  // the TCP MUST increase DupAcks by one ...
  if (m_tcb->m_congState != TcpSocketState::CA_RECOVERY)
    {
      ++m_dupAckCount;
    }

  if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
    {
      // From Open we go Disorder
      NS_ASSERT_MSG (m_dupAckCount == 1, "From OPEN->DISORDER but with " <<
                     m_dupAckCount << " dup ACKs");

      m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_DISORDER);
      m_tcb->m_congState = TcpSocketState::CA_DISORDER;

      NS_LOG_DEBUG ("CA_OPEN -> CA_DISORDER");
    }

  if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
    {
      if (!m_sackEnabled)
        {
          // If we are in recovery and we receive a dupack, one segment
          // has left the network. This is equivalent to a SACK of one block.
          m_txBuffer->AddRenoSack ();
        }
      if (!m_congestionControl->HasCongControl ())
        {
          m_recoveryOps->DoRecovery (m_tcb, currentDelivered);
          NS_LOG_INFO (m_dupAckCount << " Dupack received in fast recovery mode."
                       "Increase cwnd to " << m_tcb->m_cWnd);
        }
    }
  else if (m_tcb->m_congState == TcpSocketState::CA_DISORDER)
    {
      // m_dupackCount should not exceed its threshold in CA_DISORDER state
      // when m_recoverActive has not been set. When recovery point
      // have been set after timeout, the sender could enter into CA_DISORDER
      // after receiving new ACK smaller than m_recover. After that, m_dupackCount
      // can be equal and larger than m_retxThresh and we should avoid entering
      // CA_RECOVERY and reducing sending rate again.
      NS_ASSERT((m_dupAckCount <= m_retxThresh) || m_recoverActive);

      // RFC 6675, Section 5, continuing:
      // ... and take the following steps:
      // (1) If DupAcks >= DupThresh, go to step (4).
      //     Sequence number comparison (m_highRxAckMark >= m_recover) will take
      //     effect only when m_recover has been set. Hence, we can avoid to use
      //     m_recover in the last congestion event and fail to enter
      //     CA_RECOVERY when sequence number is advanced significantly since
      //     the last congestion event, which could be common for
      //     bandwidth-greedy application in high speed and reliable network
      //     (such as datacenter network) whose sending rate is constrainted by
      //     TCP socket buffer size at receiver side.
      if ((m_dupAckCount == m_retxThresh) && ((m_highRxAckMark >= m_recover) || (!m_recoverActive)))
        {
          EnterRecovery (currentDelivered);
          NS_ASSERT (m_tcb->m_congState == TcpSocketState::CA_RECOVERY);
        }
      // (2) If DupAcks < DupThresh but IsLost (HighACK + 1) returns true
      // (indicating at least three segments have arrived above the current
      // cumulative acknowledgment point, which is taken to indicate loss)
      // go to step (4).
      else if (m_txBuffer->IsLost (m_highRxAckMark + m_tcb->m_segmentSize))
        {
          EnterRecovery (currentDelivered);
          NS_ASSERT (m_tcb->m_congState == TcpSocketState::CA_RECOVERY);
        }
      else
        {
          // (3) The TCP MAY transmit previously unsent data segments as per
          // Limited Transmit [RFC5681] ...except that the number of octets
          // which may be sent is governed by pipe and cwnd as follows:
          //
          // (3.1) Set HighRxt to HighACK.
          // Not clear in RFC. We don't do this here, since we still have
          // to retransmit the segment.

          if (!m_sackEnabled && m_limitedTx)
            {
              m_txBuffer->AddRenoSack ();

              // In limited transmit, cwnd Infl is not updated.
            }
        }
    }
}

/* Process the newly received ACK */
void
TcpSocketBase::ReceivedAck (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  NS_ASSERT (0 != (tcpHeader.GetFlags () & TcpHeader::ACK));
  NS_ASSERT (m_tcb->m_segmentSize > 0);

  uint32_t previousLost = m_txBuffer->GetLost ();
  uint32_t priorInFlight = m_tcb->m_bytesInFlight.Get ();

  // RFC 6675, Section 5, 1st paragraph:
  // Upon the receipt of any ACK containing SACK information, the
  // scoreboard MUST be updated via the Update () routine (done in ReadOptions)
  uint32_t bytesSacked = 0;
  uint64_t previousDelivered = m_rateOps->GetConnectionRate ().m_delivered;
  ReadOptions (tcpHeader, &bytesSacked);

  SequenceNumber32 ackNumber = tcpHeader.GetAckNumber ();
  SequenceNumber32 oldHeadSequence = m_txBuffer->HeadSequence ();

  if (ackNumber < oldHeadSequence)
    {
      NS_LOG_DEBUG ("Possibly received a stale ACK (ack number < head sequence)");
      // If there is any data piggybacked, store it into m_rxBuffer
      if (packet->GetSize () > 0)
        {
          ReceivedData (packet, tcpHeader);
        }
      return;
    }
  if ((ackNumber > oldHeadSequence) && (ackNumber < m_recover)
                                    && (m_tcb->m_congState == TcpSocketState::CA_RECOVERY))
    {
      uint32_t segAcked = (ackNumber - oldHeadSequence)/m_tcb->m_segmentSize;
      for (uint32_t i = 0; i < segAcked; i++)
        {
          if (m_txBuffer->IsRetransmittedDataAcked (ackNumber - (i * m_tcb->m_segmentSize)))
            {
              m_tcb->m_isRetransDataAcked = true;
              NS_LOG_DEBUG ("Ack Number " << ackNumber <<
                            "is ACK of retransmitted packet.");
            }
        }
    }

  m_txBuffer->DiscardUpTo (ackNumber, MakeCallback (&TcpRateOps::SkbDelivered, m_rateOps));

  uint32_t currentDelivered = static_cast<uint32_t> (m_rateOps->GetConnectionRate ().m_delivered - previousDelivered);
  m_tcb->m_lastAckedSackedBytes = currentDelivered;

  if (m_tcb->m_congState == TcpSocketState::CA_CWR && (ackNumber > m_recover))
    {
      // Recovery is over after the window exceeds m_recover
      // (although it may be re-entered below if ECE is still set)
      NS_LOG_DEBUG (TcpSocketState::TcpCongStateName[m_tcb->m_congState] << " -> CA_OPEN");
      m_tcb->m_congState = TcpSocketState::CA_OPEN;
      if (!m_congestionControl->HasCongControl ())
        {
           m_tcb->m_cWnd = m_tcb->m_ssThresh.Get ();
           m_recoveryOps->ExitRecovery (m_tcb);
           m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
        }
    }

  if (ackNumber > oldHeadSequence && (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED) && (tcpHeader.GetFlags () & TcpHeader::ECE))
    {
      if (m_ecnEchoSeq < ackNumber)
        {
          NS_LOG_INFO ("Received ECN Echo is valid");
          m_ecnEchoSeq = ackNumber;
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_ECE_RCVD");
          m_tcb->m_ecnState = TcpSocketState::ECN_ECE_RCVD;
          if (m_tcb->m_congState != TcpSocketState::CA_CWR)
            {
              EnterCwr (currentDelivered);
            }
        }
    }
  else if (m_tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD && !(tcpHeader.GetFlags () & TcpHeader::ECE))
    {
      m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
    }

  // Update bytes in flight before processing the ACK for proper calculation of congestion window
  NS_LOG_INFO ("Update bytes in flight before processing the ACK.");
  BytesInFlight ();

  // RFC 6675 Section 5: 2nd, 3rd paragraph and point (A), (B) implementation
  // are inside the function ProcessAck
  ProcessAck (ackNumber, (bytesSacked > 0), currentDelivered, oldHeadSequence);
  m_tcb->m_isRetransDataAcked = false;

  if (m_congestionControl->HasCongControl ())
    {
      uint32_t currentLost = m_txBuffer->GetLost ();
      uint32_t lost = (currentLost > previousLost) ?
            currentLost - previousLost :
            previousLost - currentLost;
      auto rateSample = m_rateOps->GenerateSample (currentDelivered, lost,
                                              false, priorInFlight, m_tcb->m_minRtt);
      auto rateConn = m_rateOps->GetConnectionRate ();
      m_congestionControl->CongControl(m_tcb, rateConn, rateSample);
    }

  // If there is any data piggybacked, store it into m_rxBuffer
  if (packet->GetSize () > 0)
    {
      ReceivedData (packet, tcpHeader);
    }

  // RFC 6675, Section 5, point (C), try to send more data. NB: (C) is implemented
  // inside SendPendingData
  SendPendingData (m_connected);
}

void
TcpSocketBase::ProcessAck(const SequenceNumber32 &ackNumber, bool scoreboardUpdated,
                          uint32_t currentDelivered, const SequenceNumber32 &oldHeadSequence)
{
  NS_LOG_FUNCTION (this << ackNumber << scoreboardUpdated);
  // RFC 6675, Section 5, 2nd paragraph:
  // If the incoming ACK is a cumulative acknowledgment, the TCP MUST
  // reset DupAcks to zero.
  bool exitedFastRecovery = false;
  uint32_t oldDupAckCount = m_dupAckCount; // remember the old value
  m_tcb->m_lastAckedSeq = ackNumber; // Update lastAckedSeq
  uint32_t bytesAcked = 0;

  /* In RFC 5681 the definition of duplicate acknowledgment was strict:
   *
   * (a) the receiver of the ACK has outstanding data,
   * (b) the incoming acknowledgment carries no data,
   * (c) the SYN and FIN bits are both off,
   * (d) the acknowledgment number is equal to the greatest acknowledgment
   *     received on the given connection (TCP.UNA from [RFC793]),
   * (e) the advertised window in the incoming acknowledgment equals the
   *     advertised window in the last incoming acknowledgment.
   *
   * With RFC 6675, this definition has been reduced:
   *
   * (a) the ACK is carrying a SACK block that identifies previously
   *     unacknowledged and un-SACKed octets between HighACK (TCP.UNA) and
   *     HighData (m_highTxMark)
   */

  bool isDupack = m_sackEnabled ?
    scoreboardUpdated
    : ackNumber == oldHeadSequence &&
    ackNumber < m_tcb->m_highTxMark;

  NS_LOG_DEBUG ("ACK of " << ackNumber <<
                " SND.UNA=" << oldHeadSequence <<
                " SND.NXT=" << m_tcb->m_nextTxSequence <<
                " in state: " << TcpSocketState::TcpCongStateName[m_tcb->m_congState] <<
                " with m_recover: " << m_recover);

  // RFC 6675, Section 5, 3rd paragraph:
  // If the incoming ACK is a duplicate acknowledgment per the definition
  // in Section 2 (regardless of its status as a cumulative
  // acknowledgment), and the TCP is not currently in loss recovery
  if (isDupack)
    {
      // loss recovery check is done inside this function thanks to
      // the congestion state machine
      DupAck (currentDelivered);
    }

  if (ackNumber == oldHeadSequence
      && ackNumber == m_tcb->m_highTxMark)
    {
      // Dupack, but the ACK is precisely equal to the nextTxSequence
      return;
    }
  else if (ackNumber == oldHeadSequence
           && ackNumber > m_tcb->m_highTxMark)
    {
      // ACK of the FIN bit ... nextTxSequence is not updated since we
      // don't have anything to transmit
      NS_LOG_DEBUG ("Update nextTxSequence manually to " << ackNumber);
      m_tcb->m_nextTxSequence = ackNumber;
    }
  else if (ackNumber == oldHeadSequence)
    {
      // DupAck. Artificially call PktsAcked: after all, one segment has been ACKed.
      m_congestionControl->PktsAcked (m_tcb, 1, m_tcb->m_lastRtt);
    }
  else if (ackNumber > oldHeadSequence)
    {
      // Please remember that, with SACK, we can enter here even if we
      // received a dupack.
      bytesAcked = ackNumber - oldHeadSequence;
      uint32_t segsAcked  = bytesAcked / m_tcb->m_segmentSize;
      m_bytesAckedNotProcessed += bytesAcked % m_tcb->m_segmentSize;
      bytesAcked -= bytesAcked % m_tcb->m_segmentSize;

      if (m_bytesAckedNotProcessed >= m_tcb->m_segmentSize)
        {
          segsAcked += 1;
          bytesAcked += m_tcb->m_segmentSize;
          m_bytesAckedNotProcessed -= m_tcb->m_segmentSize;
        }

      // Dupack count is reset to eventually fast-retransmit after 3 dupacks.
      // Any SACK-ed segment will be cleaned up by DiscardUpTo.
      // In the case that we advanced SND.UNA, but the ack contains SACK blocks,
      // we do not reset. At the third one we will retransmit.
      // If we are already in recovery, this check is useless since dupAcks
      // are not considered in this phase. When from Recovery we go back
      // to open, then dupAckCount is reset anyway.
      if (!isDupack)
        {
          m_dupAckCount = 0;
        }

      // RFC 6675, Section 5, part (B)
      // (B) Upon receipt of an ACK that does not cover RecoveryPoint, the
      // following actions MUST be taken:
      //
      // (B.1) Use Update () to record the new SACK information conveyed
      //       by the incoming ACK.
      // (B.2) Use SetPipe () to re-calculate the number of octets still
      //       in the network.
      //
      // (B.1) is done at the beginning, while (B.2) is delayed to part (C) while
      // trying to transmit with SendPendingData. We are not allowed to exit
      // the CA_RECOVERY phase. Just process this partial ack (RFC 5681)
      if (ackNumber < m_recover && m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
        {
          if (!m_sackEnabled)
            {
              // Manually set the head as lost, it will be retransmitted.
              NS_LOG_INFO ("Partial ACK. Manually setting head as lost");
              m_txBuffer->MarkHeadAsLost ();
            }

          // Before retransmitting the packet perform DoRecovery and check if
          // there is available window
          if (!m_congestionControl->HasCongControl () && segsAcked >= 1)
            {
              m_recoveryOps->DoRecovery (m_tcb, currentDelivered);
            }

          // If the packet is already retransmitted do not retransmit it
          if (!m_txBuffer->IsRetransmittedDataAcked (ackNumber + m_tcb->m_segmentSize))
            {
              DoRetransmit (); // Assume the next seq is lost. Retransmit lost packet
              m_tcb->m_cWndInfl = SafeSubtraction (m_tcb->m_cWndInfl, bytesAcked);
            }

          // This partial ACK acknowledge the fact that one segment has been
          // previously lost and now successfully received. All others have
          // been processed when they come under the form of dupACKs
          m_congestionControl->PktsAcked (m_tcb, 1, m_tcb->m_lastRtt);
          NewAck (ackNumber, m_isFirstPartialAck);

          if (m_isFirstPartialAck)
            {
              NS_LOG_DEBUG ("Partial ACK of " << ackNumber <<
                            " and this is the first (RTO will be reset);"
                            " cwnd set to " << m_tcb->m_cWnd <<
                            " recover seq: " << m_recover <<
                            " dupAck count: " << m_dupAckCount);
              m_isFirstPartialAck = false;
            }
          else
            {
              NS_LOG_DEBUG ("Partial ACK of " << ackNumber <<
                            " and this is NOT the first (RTO will not be reset)"
                            " cwnd set to " << m_tcb->m_cWnd <<
                            " recover seq: " << m_recover <<
                            " dupAck count: " << m_dupAckCount);
            }
        }
      // From RFC 6675 section 5.1
      // In addition, a new recovery phase (as described in Section 5) MUST NOT
      // be initiated until HighACK is greater than or equal to the new value
      // of RecoveryPoint.
      else if (ackNumber < m_recover && m_tcb->m_congState == TcpSocketState::CA_LOSS)
        {
          m_congestionControl->PktsAcked (m_tcb, segsAcked, m_tcb->m_lastRtt);
          m_congestionControl->IncreaseWindow (m_tcb, segsAcked);

          NS_LOG_DEBUG (" Cong Control Called, cWnd=" << m_tcb->m_cWnd <<
                        " ssTh=" << m_tcb->m_ssThresh);
          if (!m_sackEnabled)
            {
              NS_ASSERT_MSG (m_txBuffer->GetSacked () == 0,
                             "Some segment got dup-acked in CA_LOSS state: " <<
                             m_txBuffer->GetSacked ());
            }
          NewAck (ackNumber, true);
        }
      else if (m_tcb->m_congState == TcpSocketState::CA_CWR)
        {
          m_congestionControl->PktsAcked (m_tcb, segsAcked, m_tcb->m_lastRtt);
          // TODO: need to check behavior if marking is compounded by loss
          // and/or packet reordering
          if (!m_congestionControl->HasCongControl () && segsAcked >= 1)
            {
              m_recoveryOps->DoRecovery (m_tcb, currentDelivered);
            }
          NewAck (ackNumber, true);
        }
      else
        {
          if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
            {
              m_congestionControl->PktsAcked (m_tcb, segsAcked, m_tcb->m_lastRtt);
            }
          else if (m_tcb->m_congState == TcpSocketState::CA_DISORDER)
            {
              if (segsAcked >= oldDupAckCount)
                {
                  m_congestionControl->PktsAcked (m_tcb, segsAcked - oldDupAckCount, m_tcb->m_lastRtt);
                }

              if (!isDupack)
                {
                  // The network reorder packets. Linux changes the counting lost
                  // packet algorithm from FACK to NewReno. We simply go back in Open.
                  m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_OPEN);
                  m_tcb->m_congState = TcpSocketState::CA_OPEN;
                  NS_LOG_DEBUG (segsAcked << " segments acked in CA_DISORDER, ack of " <<
                                ackNumber << " exiting CA_DISORDER -> CA_OPEN");
                }
              else
                {
                  NS_LOG_DEBUG (segsAcked << " segments acked in CA_DISORDER, ack of " <<
                                ackNumber << " but still in CA_DISORDER");
                }
            }
          // RFC 6675, Section 5:
          // Once a TCP is in the loss recovery phase, the following procedure
          // MUST be used for each arriving ACK:
          // (A) An incoming cumulative ACK for a sequence number greater than
          // RecoveryPoint signals the end of loss recovery, and the loss
          // recovery phase MUST be terminated.  Any information contained in
          // the scoreboard for sequence numbers greater than the new value of
          // HighACK SHOULD NOT be cleared when leaving the loss recovery
          // phase.
          else if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
            {
              m_isFirstPartialAck = true;

              // Recalculate the segs acked, that are from m_recover to ackNumber
              // (which are the ones we have not passed to PktsAcked and that
              // can increase cWnd)
              // TODO:  check consistency for dynamic segment size
              segsAcked = static_cast<uint32_t>(ackNumber - oldHeadSequence) / m_tcb->m_segmentSize;
              m_congestionControl->PktsAcked (m_tcb, segsAcked, m_tcb->m_lastRtt);
              m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
              m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_OPEN);
              m_tcb->m_congState = TcpSocketState::CA_OPEN;
              exitedFastRecovery = true;
              m_dupAckCount = 0; // From recovery to open, reset dupack

              NS_LOG_DEBUG (segsAcked << " segments acked in CA_RECOVER, ack of " <<
                            ackNumber << ", exiting CA_RECOVERY -> CA_OPEN");
            }
          else if (m_tcb->m_congState == TcpSocketState::CA_LOSS)
            {
              m_isFirstPartialAck = true;

              // Recalculate the segs acked, that are from m_recover to ackNumber
              // (which are the ones we have not passed to PktsAcked and that
              // can increase cWnd)
              segsAcked = (ackNumber - m_recover) / m_tcb->m_segmentSize;

              m_congestionControl->PktsAcked (m_tcb, segsAcked, m_tcb->m_lastRtt);

              m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_OPEN);
              m_tcb->m_congState = TcpSocketState::CA_OPEN;
              NS_LOG_DEBUG (segsAcked << " segments acked in CA_LOSS, ack of" <<
                            ackNumber << ", exiting CA_LOSS -> CA_OPEN");
            }

          if (ackNumber >= m_recover)
            {
              // All lost segments in the congestion event have been
              // retransmitted successfully. The recovery point (m_recover)
              // should be deactivated.
              m_recoverActive = false;
            }

          if (exitedFastRecovery)
            {
              NewAck (ackNumber, true);
              m_tcb->m_cWnd = m_tcb->m_ssThresh.Get ();
              m_recoveryOps->ExitRecovery (m_tcb);
              NS_LOG_DEBUG ("Leaving Fast Recovery; BytesInFlight() = " <<
                            BytesInFlight () << "; cWnd = " << m_tcb->m_cWnd);
            }
          if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
            {
              m_congestionControl->IncreaseWindow (m_tcb, segsAcked);

              m_tcb->m_cWndInfl = m_tcb->m_cWnd;

              NS_LOG_LOGIC ("Congestion control called: " <<
                            " cWnd: " << m_tcb->m_cWnd <<
                            " ssTh: " << m_tcb->m_ssThresh <<
                            " segsAcked: " << segsAcked);

              NewAck (ackNumber, true);
            }
        }
    }
  // Update the pacing rate, since m_congestionControl->IncreaseWindow() or
  // m_congestionControl->PktsAcked () may change m_tcb->m_cWnd
  // Make sure that control reaches the end of this function and there is no
  // return in between
  UpdatePacingRate ();
}

/* Received a packet upon LISTEN state. */
void
TcpSocketBase::ProcessListen (Ptr<Packet> packet, const TcpHeader& tcpHeader,
                              const Address& fromAddress, const Address& toAddress)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH, URG, CWR and ECE are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

  // Fork a socket if received a SYN. Do nothing otherwise.
  // C.f.: the LISTEN part in tcp_v4_do_rcv() in tcp_ipv4.c in Linux kernel
  if (tcpflags != TcpHeader::SYN)
    {
      return;
    }

  // Call socket's notify function to let the server app know we got a SYN
  // If the server app refuses the connection, do nothing
  if (!NotifyConnectionRequest (fromAddress))
    {
      return;
    }
  // Clone the socket, simulate fork
  Ptr<TcpSocketBase> newSock = Fork ();
  NS_LOG_LOGIC ("Cloned a TcpSocketBase " << newSock);
  Simulator::ScheduleNow (&TcpSocketBase::CompleteFork, newSock,
                          packet, tcpHeader, fromAddress, toAddress);
}

/* Received a packet upon SYN_SENT */
void
TcpSocketBase::ProcessSynSent (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH and URG are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG);

  if (tcpflags == 0)
    { // Bare data, accept it and move to ESTABLISHED state. This is not a normal behaviour. Remove this?
      NS_LOG_DEBUG ("SYN_SENT -> ESTABLISHED");
      m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_OPEN);
      m_tcb->m_congState = TcpSocketState::CA_OPEN;
      m_state = ESTABLISHED;
      m_connected = true;
      m_retxEvent.Cancel ();
      m_delAckCount = m_delAckMaxCount;
      ReceivedData (packet, tcpHeader);
      Simulator::ScheduleNow (&TcpSocketBase::ConnectionSucceeded, this);
    }
  else if (tcpflags & TcpHeader::ACK && !(tcpflags & TcpHeader::SYN))
    { // Ignore ACK in SYN_SENT
    }
  else if (tcpflags & TcpHeader::SYN && !(tcpflags & TcpHeader::ACK))
    { // Received SYN, move to SYN_RCVD state and respond with SYN+ACK
      NS_LOG_DEBUG ("SYN_SENT -> SYN_RCVD");
      m_state = SYN_RCVD;
      m_synCount = m_synRetries;
      m_tcb->m_rxBuffer->SetNextRxSequence (tcpHeader.GetSequenceNumber () + SequenceNumber32 (1));
      /* Check if we received an ECN SYN packet. Change the ECN state of receiver to ECN_IDLE if the traffic is ECN capable and
       * sender has sent ECN SYN packet
       */

      if (m_tcb->m_useEcn != TcpSocketState::Off && (tcpflags & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::CWR | TcpHeader::ECE))
        {
          NS_LOG_INFO ("Received ECN SYN packet");
          SendEmptyPacket (TcpHeader::SYN | TcpHeader::ACK | TcpHeader::ECE);
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
          m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
        }
      else
        {
          m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
          SendEmptyPacket (TcpHeader::SYN | TcpHeader::ACK);
        }
    }
  else if (tcpflags & (TcpHeader::SYN | TcpHeader::ACK)
           && m_tcb->m_nextTxSequence + SequenceNumber32 (1) == tcpHeader.GetAckNumber ())
    { // Handshake completed
      NS_LOG_DEBUG ("SYN_SENT -> ESTABLISHED");
      m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_OPEN);
      m_tcb->m_congState = TcpSocketState::CA_OPEN;
      m_state = ESTABLISHED;
      m_connected = true;
      m_retxEvent.Cancel ();
      m_tcb->m_rxBuffer->SetNextRxSequence (tcpHeader.GetSequenceNumber () + SequenceNumber32 (1));
      m_tcb->m_highTxMark = ++m_tcb->m_nextTxSequence;
      m_txBuffer->SetHeadSequence (m_tcb->m_nextTxSequence);
      // Before sending packets, update the pacing rate based on RTT measurement so far 
      UpdatePacingRate ();
      SendEmptyPacket (TcpHeader::ACK);

      /* Check if we received an ECN SYN-ACK packet. Change the ECN state of sender to ECN_IDLE if receiver has sent an ECN SYN-ACK
       * packet and the  traffic is ECN Capable
       */
      if (m_tcb->m_useEcn != TcpSocketState::Off && (tcpflags & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::ECE))
        {
          NS_LOG_INFO ("Received ECN SYN-ACK packet.");
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
          m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
        }
      else
        {
          m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
        }
      SendPendingData (m_connected);
      Simulator::ScheduleNow (&TcpSocketBase::ConnectionSucceeded, this);
      // Always respond to first data packet to speed up the connection.
      // Remove to get the behaviour of old NS-3 code.
      m_delAckCount = m_delAckMaxCount;
    }
  else
    { // Other in-sequence input
      if (!(tcpflags & TcpHeader::RST))
        { // When (1) rx of FIN+ACK; (2) rx of FIN; (3) rx of bad flags
          NS_LOG_LOGIC ("Illegal flag combination " << TcpHeader::FlagsToString (tcpHeader.GetFlags ()) <<
                        " received in SYN_SENT. Reset packet is sent.");
          SendRST ();
        }
      CloseAndNotify ();
    }
}

/* Received a packet upon SYN_RCVD */
void
TcpSocketBase::ProcessSynRcvd (Ptr<Packet> packet, const TcpHeader& tcpHeader,
                               const Address& fromAddress, 
                               [[maybe_unused]] const Address& toAddress)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH, URG, CWR and ECE are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

  if (tcpflags == 0
      || (tcpflags == TcpHeader::ACK
          && m_tcb->m_nextTxSequence + SequenceNumber32 (1) == tcpHeader.GetAckNumber ()))
    { // If it is bare data, accept it and move to ESTABLISHED state. This is
      // possibly due to ACK lost in 3WHS. If in-sequence ACK is received, the
      // handshake is completed nicely.
      NS_LOG_DEBUG ("SYN_RCVD -> ESTABLISHED");
      m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_OPEN);
      m_tcb->m_congState = TcpSocketState::CA_OPEN;
      m_state = ESTABLISHED;
      m_connected = true;
      m_retxEvent.Cancel ();
      m_tcb->m_highTxMark = ++m_tcb->m_nextTxSequence;
      m_txBuffer->SetHeadSequence (m_tcb->m_nextTxSequence);
      if (m_endPoint)
        {
          m_endPoint->SetPeer (InetSocketAddress::ConvertFrom (fromAddress).GetIpv4 (),
                               InetSocketAddress::ConvertFrom (fromAddress).GetPort ());
        }
      else if (m_endPoint6)
        {
          m_endPoint6->SetPeer (Inet6SocketAddress::ConvertFrom (fromAddress).GetIpv6 (),
                                Inet6SocketAddress::ConvertFrom (fromAddress).GetPort ());
        }
      // Always respond to first data packet to speed up the connection.
      // Remove to get the behaviour of old NS-3 code.
      m_delAckCount = m_delAckMaxCount;
      NotifyNewConnectionCreated (this, fromAddress);
      ReceivedAck (packet, tcpHeader);
      // Update the pacing rate based on RTT measurement so far
      UpdatePacingRate ();
      // As this connection is established, the socket is available to send data now
      if (GetTxAvailable () > 0)
        {
          NotifySend (GetTxAvailable ());
        }
    }
  else if (tcpflags == TcpHeader::SYN)
    { // Probably the peer lost my SYN+ACK
      m_tcb->m_rxBuffer->SetNextRxSequence (tcpHeader.GetSequenceNumber () + SequenceNumber32 (1));
      /* Check if we received an ECN SYN packet. Change the ECN state of receiver to ECN_IDLE if sender has sent an ECN SYN
       * packet and the  traffic is ECN Capable
       */
      if (m_tcb->m_useEcn != TcpSocketState::Off && (tcpHeader.GetFlags () & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::CWR | TcpHeader::ECE))
        {
          NS_LOG_INFO ("Received ECN SYN packet");
          SendEmptyPacket (TcpHeader::SYN | TcpHeader::ACK |TcpHeader::ECE);
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
          m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
       }
      else
        {
          m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
          SendEmptyPacket (TcpHeader::SYN | TcpHeader::ACK);
        }
    }
  else if (tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
    {
      if (tcpHeader.GetSequenceNumber () == m_tcb->m_rxBuffer->NextRxSequence ())
        { // In-sequence FIN before connection complete. Set up connection and close.
          m_connected = true;
          m_retxEvent.Cancel ();
          m_tcb->m_highTxMark = ++m_tcb->m_nextTxSequence;
          m_txBuffer->SetHeadSequence (m_tcb->m_nextTxSequence);
          if (m_endPoint)
            {
              m_endPoint->SetPeer (InetSocketAddress::ConvertFrom (fromAddress).GetIpv4 (),
                                   InetSocketAddress::ConvertFrom (fromAddress).GetPort ());
            }
          else if (m_endPoint6)
            {
              m_endPoint6->SetPeer (Inet6SocketAddress::ConvertFrom (fromAddress).GetIpv6 (),
                                    Inet6SocketAddress::ConvertFrom (fromAddress).GetPort ());
            }
          NotifyNewConnectionCreated (this, fromAddress);
          PeerClose (packet, tcpHeader);
        }
    }
  else
    { // Other in-sequence input
      if (tcpflags != TcpHeader::RST)
        { // When (1) rx of SYN+ACK; (2) rx of FIN; (3) rx of bad flags
          NS_LOG_LOGIC ("Illegal flag " << TcpHeader::FlagsToString (tcpflags) <<
                        " received. Reset packet is sent.");
          if (m_endPoint)
            {
              m_endPoint->SetPeer (InetSocketAddress::ConvertFrom (fromAddress).GetIpv4 (),
                                   InetSocketAddress::ConvertFrom (fromAddress).GetPort ());
            }
          else if (m_endPoint6)
            {
              m_endPoint6->SetPeer (Inet6SocketAddress::ConvertFrom (fromAddress).GetIpv6 (),
                                    Inet6SocketAddress::ConvertFrom (fromAddress).GetPort ());
            }
          SendRST ();
        }
      CloseAndNotify ();
    }
}

/* Received a packet upon CLOSE_WAIT, FIN_WAIT_1, or FIN_WAIT_2 states */
void
TcpSocketBase::ProcessWait (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH, URG, CWR and ECE are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

  if (packet->GetSize () > 0 && !(tcpflags & TcpHeader::ACK))
    { // Bare data, accept it
      ReceivedData (packet, tcpHeader);
    }
  else if (tcpflags == TcpHeader::ACK)
    { // Process the ACK, and if in FIN_WAIT_1, conditionally move to FIN_WAIT_2
      ReceivedAck (packet, tcpHeader);
      if (m_state == FIN_WAIT_1 && m_txBuffer->Size () == 0
          && tcpHeader.GetAckNumber () == m_tcb->m_highTxMark + SequenceNumber32 (1))
        { // This ACK corresponds to the FIN sent
          NS_LOG_DEBUG ("FIN_WAIT_1 -> FIN_WAIT_2");
          m_state = FIN_WAIT_2;
        }
    }
  else if (tcpflags == TcpHeader::FIN || tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
    { // Got FIN, respond with ACK and move to next state
      if (tcpflags & TcpHeader::ACK)
        { // Process the ACK first
          ReceivedAck (packet, tcpHeader);
        }
      m_tcb->m_rxBuffer->SetFinSequence (tcpHeader.GetSequenceNumber ());
    }
  else if (tcpflags == TcpHeader::SYN || tcpflags == (TcpHeader::SYN | TcpHeader::ACK))
    { // Duplicated SYN or SYN+ACK, possibly due to spurious retransmission
      return;
    }
  else
    { // This is a RST or bad flags
      if (tcpflags != TcpHeader::RST)
        {
          NS_LOG_LOGIC ("Illegal flag " << TcpHeader::FlagsToString (tcpflags) <<
                        " received. Reset packet is sent.");
          SendRST ();
        }
      CloseAndNotify ();
      return;
    }

  // Check if the close responder sent an in-sequence FIN, if so, respond ACK
  if ((m_state == FIN_WAIT_1 || m_state == FIN_WAIT_2) && m_tcb->m_rxBuffer->Finished ())
    {
      if (m_state == FIN_WAIT_1)
        {
          NS_LOG_DEBUG ("FIN_WAIT_1 -> CLOSING");
          m_state = CLOSING;
          if (m_txBuffer->Size () == 0
              && tcpHeader.GetAckNumber () == m_tcb->m_highTxMark + SequenceNumber32 (1))
            { // This ACK corresponds to the FIN sent
              TimeWait ();
            }
        }
      else if (m_state == FIN_WAIT_2)
        {
          TimeWait ();
        }
      SendEmptyPacket (TcpHeader::ACK);
      if (!m_shutdownRecv)
        {
          NotifyDataRecv ();
        }
    }
}

/* Received a packet upon CLOSING */
void
TcpSocketBase::ProcessClosing (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH and URG are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG);

  if (tcpflags == TcpHeader::ACK)
    {
      if (tcpHeader.GetSequenceNumber () == m_tcb->m_rxBuffer->NextRxSequence ())
        { // This ACK corresponds to the FIN sent
          TimeWait ();
        }
    }
  else
    { // CLOSING state means simultaneous close, i.e. no one is sending data to
      // anyone. If anything other than ACK is received, respond with a reset.
      if (tcpflags == TcpHeader::FIN || tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
        { // FIN from the peer as well. We can close immediately.
          SendEmptyPacket (TcpHeader::ACK);
        }
      else if (tcpflags != TcpHeader::RST)
        { // Receive of SYN or SYN+ACK or bad flags or pure data
          NS_LOG_LOGIC ("Illegal flag " << TcpHeader::FlagsToString (tcpflags) << " received. Reset packet is sent.");
          SendRST ();
        }
      CloseAndNotify ();
    }
}

/* Received a packet upon LAST_ACK */
void
TcpSocketBase::ProcessLastAck (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Extract the flags. PSH and URG are disregarded.
  uint8_t tcpflags = tcpHeader.GetFlags () & ~(TcpHeader::PSH | TcpHeader::URG);

  if (tcpflags == 0)
    {
      ReceivedData (packet, tcpHeader);
    }
  else if (tcpflags == TcpHeader::ACK)
    {
      if (tcpHeader.GetSequenceNumber () == m_tcb->m_rxBuffer->NextRxSequence ())
        { // This ACK corresponds to the FIN sent. This socket closed peacefully.
          CloseAndNotify ();
        }
    }
  else if (tcpflags == TcpHeader::FIN)
    { // Received FIN again, the peer probably lost the FIN+ACK
      SendEmptyPacket (TcpHeader::FIN | TcpHeader::ACK);
    }
  else if (tcpflags == (TcpHeader::FIN | TcpHeader::ACK) || tcpflags == TcpHeader::RST)
    {
      CloseAndNotify ();
    }
  else
    { // Received a SYN or SYN+ACK or bad flags
      NS_LOG_LOGIC ("Illegal flag " << TcpHeader::FlagsToString (tcpflags) << " received. Reset packet is sent.");
      SendRST ();
      CloseAndNotify ();
    }
}

/* Peer sent me a FIN. Remember its sequence in rx buffer. */
void
TcpSocketBase::PeerClose (Ptr<Packet> p, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);

  // Ignore all out of range packets
  if (tcpHeader.GetSequenceNumber () < m_tcb->m_rxBuffer->NextRxSequence ()
      || tcpHeader.GetSequenceNumber () > m_tcb->m_rxBuffer->MaxRxSequence ())
    {
      return;
    }
  // For any case, remember the FIN position in rx buffer first
  m_tcb->m_rxBuffer->SetFinSequence (tcpHeader.GetSequenceNumber () + SequenceNumber32 (p->GetSize ()));
  NS_LOG_LOGIC ("Accepted FIN at seq " << tcpHeader.GetSequenceNumber () + SequenceNumber32 (p->GetSize ()));
  // If there is any piggybacked data, process it
  if (p->GetSize ())
    {
      ReceivedData (p, tcpHeader);
    }
  // Return if FIN is out of sequence, otherwise move to CLOSE_WAIT state by DoPeerClose
  if (!m_tcb->m_rxBuffer->Finished ())
    {
      return;
    }

  // Simultaneous close: Application invoked Close() when we are processing this FIN packet
  if (m_state == FIN_WAIT_1)
    {
      NS_LOG_DEBUG ("FIN_WAIT_1 -> CLOSING");
      m_state = CLOSING;
      return;
    }

  DoPeerClose (); // Change state, respond with ACK
}

/* Received a in-sequence FIN. Close down this socket. */
void
TcpSocketBase::DoPeerClose (void)
{
  NS_ASSERT (m_state == ESTABLISHED || m_state == SYN_RCVD ||
             m_state == FIN_WAIT_1 || m_state == FIN_WAIT_2);

  // Move the state to CLOSE_WAIT
  NS_LOG_DEBUG (TcpStateName[m_state] << " -> CLOSE_WAIT");
  m_state = CLOSE_WAIT;

  if (!m_closeNotified)
    {
      // The normal behaviour for an application is that, when the peer sent a in-sequence
      // FIN, the app should prepare to close. The app has two choices at this point: either
      // respond with ShutdownSend() call to declare that it has nothing more to send and
      // the socket can be closed immediately; or remember the peer's close request, wait
      // until all its existing data are pushed into the TCP socket, then call Close()
      // explicitly.
      NS_LOG_LOGIC ("TCP " << this << " calling NotifyNormalClose");
      NotifyNormalClose ();
      m_closeNotified = true;
    }
  if (m_shutdownSend)
    { // The application declares that it would not sent any more, close this socket
      Close ();
    }
  else
    { // Need to ack, the application will close later
      SendEmptyPacket (TcpHeader::ACK);
    }
  if (m_state == LAST_ACK)
    {
      m_dataRetrCount = m_dataRetries; // prevent endless FINs
      NS_LOG_LOGIC ("TcpSocketBase " << this << " scheduling LATO1");
      Time lastRto = m_rtt->GetEstimate () + Max (m_clockGranularity, m_rtt->GetVariation () * 4);
      m_lastAckEvent = Simulator::Schedule (lastRto, &TcpSocketBase::LastAckTimeout, this);
    }
}

/* Kill this socket. This is a callback function configured to m_endpoint in
   SetupCallback(), invoked when the endpoint is destroyed. */
void
TcpSocketBase::Destroy (void)
{
  NS_LOG_FUNCTION (this);
  m_endPoint = nullptr;
  if (m_tcp != nullptr)
    {
      m_tcp->RemoveSocket (this);
    }
  NS_LOG_LOGIC (this << " Cancelled ReTxTimeout event which was set to expire at " <<
                (Simulator::Now () + Simulator::GetDelayLeft (m_retxEvent)).GetSeconds ());
  CancelAllTimers ();
}

/* Kill this socket. This is a callback function configured to m_endpoint in
   SetupCallback(), invoked when the endpoint is destroyed. */
void
TcpSocketBase::Destroy6 (void)
{
  NS_LOG_FUNCTION (this);
  m_endPoint6 = nullptr;
  if (m_tcp != nullptr)
    {
      m_tcp->RemoveSocket (this);
    }
  NS_LOG_LOGIC (this << " Cancelled ReTxTimeout event which was set to expire at " <<
                (Simulator::Now () + Simulator::GetDelayLeft (m_retxEvent)).GetSeconds ());
  CancelAllTimers ();
}

/* Send an empty packet with specified TCP flags */
void
TcpSocketBase::SendEmptyPacket (uint8_t flags)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (flags));

  if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
      NS_LOG_WARN ("Failed to send empty packet due to null endpoint");
      return;
    }

  Ptr<Packet> p = Create<Packet> ();
  TcpHeader header;
  SequenceNumber32 s = m_tcb->m_nextTxSequence;

  if (flags & TcpHeader::FIN)
    {
      flags |= TcpHeader::ACK;
    }
  else if (m_state == FIN_WAIT_1 || m_state == LAST_ACK || m_state == CLOSING)
    {
      ++s;
    }

  AddSocketTags (p);

  header.SetFlags (flags);
  header.SetSequenceNumber (s);
  header.SetAckNumber (m_tcb->m_rxBuffer->NextRxSequence ());
  if (m_endPoint != nullptr)
    {
      header.SetSourcePort (m_endPoint->GetLocalPort ());
      header.SetDestinationPort (m_endPoint->GetPeerPort ());
    }
  else
    {
      header.SetSourcePort (m_endPoint6->GetLocalPort ());
      header.SetDestinationPort (m_endPoint6->GetPeerPort ());
    }
  AddOptions (header);

  // RFC 6298, clause 2.4
  m_rto = Max (m_rtt->GetEstimate () + Max (m_clockGranularity, m_rtt->GetVariation () * 4), m_minRto);

  uint16_t windowSize = AdvertisedWindowSize ();
  bool hasSyn = flags & TcpHeader::SYN;
  bool hasFin = flags & TcpHeader::FIN;
  bool isAck = flags == TcpHeader::ACK;
  if (hasSyn)
    {
      if (m_winScalingEnabled)
        { // The window scaling option is set only on SYN packets
          AddOptionWScale (header);
        }

      if (m_sackEnabled)
        {
          AddOptionSackPermitted (header);
        }

      if (m_synCount == 0)
        { // No more connection retries, give up
          NS_LOG_LOGIC ("Connection failed.");
          m_rtt->Reset (); //According to recommendation -> RFC 6298
          NotifyConnectionFailed ();
          m_state = CLOSED;
          DeallocateEndPoint ();
          return;
        }
      else
        { // Exponential backoff of connection time out
          int backoffCount = 0x1 << (m_synRetries - m_synCount);
          m_rto = m_cnTimeout * backoffCount;
          m_synCount--;
        }

      if (m_synRetries - 1 == m_synCount)
        {
          UpdateRttHistory (s, 0, false);
        }
      else
        { // This is SYN retransmission
          UpdateRttHistory (s, 0, true);
        }

      windowSize = AdvertisedWindowSize (false);
    }
  header.SetWindowSize (windowSize);

  if (flags & TcpHeader::ACK)
    { // If sending an ACK, cancel the delay ACK as well
      m_delAckEvent.Cancel ();
      m_delAckCount = 0;
      if (m_highTxAck < header.GetAckNumber ())
        {
          m_highTxAck = header.GetAckNumber ();
        }
      if (m_sackEnabled && m_tcb->m_rxBuffer->GetSackListSize () > 0)
        {
          AddOptionSack (header);
        }
      NS_LOG_INFO ("Sending a pure ACK, acking seq " << m_tcb->m_rxBuffer->NextRxSequence ());
    }

  m_txTrace (p, header, this);

  if (m_endPoint != nullptr)
    {
      m_tcp->SendPacket (p, header, m_endPoint->GetLocalAddress (),
                         m_endPoint->GetPeerAddress (), m_boundnetdevice);
    }
  else
    {
      m_tcp->SendPacket (p, header, m_endPoint6->GetLocalAddress (),
                         m_endPoint6->GetPeerAddress (), m_boundnetdevice);
    }


  if (m_retxEvent.IsExpired () && (hasSyn || hasFin) && !isAck )
    { // Retransmit SYN / SYN+ACK / FIN / FIN+ACK to guard against lost
      NS_LOG_LOGIC ("Schedule retransmission timeout at time "
                    << Simulator::Now ().GetSeconds () << " to expire at time "
                    << (Simulator::Now () + m_rto.Get ()).GetSeconds ());
      m_retxEvent = Simulator::Schedule (m_rto, &TcpSocketBase::SendEmptyPacket, this, flags);
    }
}

/* This function closes the endpoint completely. Called upon RST_TX action. */
void
TcpSocketBase::SendRST (void)
{
  NS_LOG_FUNCTION (this);
  SendEmptyPacket (TcpHeader::RST);
  NotifyErrorClose ();
  DeallocateEndPoint ();
}

/* Deallocate the end point and cancel all the timers */
void
TcpSocketBase::DeallocateEndPoint (void)
{
  if (m_endPoint != nullptr)
    {
      CancelAllTimers ();
      m_endPoint->SetDestroyCallback (MakeNullCallback<void> ());
      m_tcp->DeAllocate (m_endPoint);
      m_endPoint = nullptr;
      m_tcp->RemoveSocket (this);
    }
  else if (m_endPoint6 != nullptr)
    {
      CancelAllTimers ();
      m_endPoint6->SetDestroyCallback (MakeNullCallback<void> ());
      m_tcp->DeAllocate (m_endPoint6);
      m_endPoint6 = nullptr;
      m_tcp->RemoveSocket (this);
    }
}

/* Configure the endpoint to a local address. Called by Connect() if Bind() didn't specify one. */
int
TcpSocketBase::SetupEndpoint ()
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4> ();
  NS_ASSERT (ipv4 != nullptr);
  if (ipv4->GetRoutingProtocol () == nullptr)
    {
      NS_FATAL_ERROR ("No Ipv4RoutingProtocol in the node");
    }
  // Create a dummy packet, then ask the routing function for the best output
  // interface's address
  Ipv4Header header;
  header.SetDestination (m_endPoint->GetPeerAddress ());
  Socket::SocketErrno errno_;
  Ptr<Ipv4Route> route;
  Ptr<NetDevice> oif = m_boundnetdevice;
  route = ipv4->GetRoutingProtocol ()->RouteOutput (Ptr<Packet> (), header, oif, errno_);
  if (route == 0)
    {
      NS_LOG_LOGIC ("Route to " << m_endPoint->GetPeerAddress () << " does not exist");
      NS_LOG_ERROR (errno_);
      m_errno = errno_;
      return -1;
    }
  NS_LOG_LOGIC ("Route exists");
  m_endPoint->SetLocalAddress (route->GetSource ());
  return 0;
}

int
TcpSocketBase::SetupEndpoint6 ()
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  NS_ASSERT (ipv6 != nullptr);
  if (ipv6->GetRoutingProtocol () == nullptr)
    {
      NS_FATAL_ERROR ("No Ipv6RoutingProtocol in the node");
    }
  // Create a dummy packet, then ask the routing function for the best output
  // interface's address
  Ipv6Header header;
  header.SetDestination (m_endPoint6->GetPeerAddress ());
  Socket::SocketErrno errno_;
  Ptr<Ipv6Route> route;
  Ptr<NetDevice> oif = m_boundnetdevice;
  route = ipv6->GetRoutingProtocol ()->RouteOutput (Ptr<Packet> (), header, oif, errno_);
  if (route == nullptr)
    {
      NS_LOG_LOGIC ("Route to " << m_endPoint6->GetPeerAddress () << " does not exist");
      NS_LOG_ERROR (errno_);
      m_errno = errno_;
      return -1;
    }
  NS_LOG_LOGIC ("Route exists");
  m_endPoint6->SetLocalAddress (route->GetSource ());
  return 0;
}

/* This function is called only if a SYN received in LISTEN state. After
   TcpSocketBase cloned, allocate a new end point to handle the incoming
   connection and send a SYN+ACK to complete the handshake. */
void
TcpSocketBase::CompleteFork ([[maybe_unused]] Ptr<Packet> p, const TcpHeader& h,
                             const Address& fromAddress, const Address& toAddress)
{
  NS_LOG_FUNCTION (this << p << h << fromAddress << toAddress);
  // Get port and address from peer (connecting host)
  if (InetSocketAddress::IsMatchingType (toAddress))
    {
      m_endPoint = m_tcp->Allocate (GetBoundNetDevice (),
                                    InetSocketAddress::ConvertFrom (toAddress).GetIpv4 (),
                                    InetSocketAddress::ConvertFrom (toAddress).GetPort (),
                                    InetSocketAddress::ConvertFrom (fromAddress).GetIpv4 (),
                                    InetSocketAddress::ConvertFrom (fromAddress).GetPort ());
      m_endPoint6 = nullptr;
    }
  else if (Inet6SocketAddress::IsMatchingType (toAddress))
    {
      m_endPoint6 = m_tcp->Allocate6 (GetBoundNetDevice (),
                                      Inet6SocketAddress::ConvertFrom (toAddress).GetIpv6 (),
                                      Inet6SocketAddress::ConvertFrom (toAddress).GetPort (),
                                      Inet6SocketAddress::ConvertFrom (fromAddress).GetIpv6 (),
                                      Inet6SocketAddress::ConvertFrom (fromAddress).GetPort ());
      m_endPoint = nullptr;
    }
  m_tcp->AddSocket (this);

  // Change the cloned socket from LISTEN state to SYN_RCVD
  NS_LOG_DEBUG ("LISTEN -> SYN_RCVD");
  m_state = SYN_RCVD;
  m_synCount = m_synRetries;
  m_dataRetrCount = m_dataRetries;
  SetupCallback ();
  // Set the sequence number and send SYN+ACK
  m_tcb->m_rxBuffer->SetNextRxSequence (h.GetSequenceNumber () + SequenceNumber32 (1));

  /* Check if we received an ECN SYN packet. Change the ECN state of receiver to ECN_IDLE if sender has sent an ECN SYN
   * packet and the traffic is ECN Capable
   */
  if (m_tcb->m_useEcn != TcpSocketState::Off &&
      (h.GetFlags () & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::CWR | TcpHeader::ECE))
    {
      SendEmptyPacket (TcpHeader::SYN | TcpHeader::ACK | TcpHeader::ECE);
      NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
      m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
    }
  else
    {
      SendEmptyPacket (TcpHeader::SYN | TcpHeader::ACK);
      m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
    }
}

void
TcpSocketBase::ConnectionSucceeded ()
{ // Wrapper to protected function NotifyConnectionSucceeded() so that it can
  // be called as a scheduled event
  NotifyConnectionSucceeded ();
  // The if-block below was moved from ProcessSynSent() to here because we need
  // to invoke the NotifySend() only after NotifyConnectionSucceeded() to
  // reflect the behaviour in the real world.
  if (GetTxAvailable () > 0)
    {
      NotifySend (GetTxAvailable ());
    }
}

void
TcpSocketBase::AddSocketTags (const Ptr<Packet> &p) const
{
  /*
   * Add tags for each socket option.
   * Note that currently the socket adds both IPv4 tag and IPv6 tag
   * if both options are set. Once the packet got to layer three, only
   * the corresponding tags will be read.
   */
  if (GetIpTos ())
    {
      SocketIpTosTag ipTosTag;
      if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && !CheckNoEcn (GetIpTos ()))
        {
          ipTosTag.SetTos (MarkEcnCodePoint (GetIpTos (), m_tcb->m_ectCodePoint));
        }
      else
        {
          // Set the last received ipTos
          ipTosTag.SetTos (GetIpTos ());
        }
      p->AddPacketTag (ipTosTag);
    }
  else
    {
      if ((m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && p->GetSize () > 0) || m_tcb->m_ecnMode == TcpSocketState::DctcpEcn)
        {
          SocketIpTosTag ipTosTag;
          ipTosTag.SetTos (MarkEcnCodePoint (GetIpTos (), m_tcb->m_ectCodePoint));
          p->AddPacketTag (ipTosTag);
        }
    }

  if (IsManualIpv6Tclass ())
    {
      SocketIpv6TclassTag ipTclassTag;
      if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && !CheckNoEcn (GetIpv6Tclass ()))
        {
          ipTclassTag.SetTclass (MarkEcnCodePoint (GetIpv6Tclass (), m_tcb->m_ectCodePoint));
        }
      else
        {
          // Set the last received ipTos
          ipTclassTag.SetTclass (GetIpv6Tclass ());
        }
      p->AddPacketTag (ipTclassTag);
    }
  else
    {
      if ((m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && p->GetSize () > 0) || m_tcb->m_ecnMode == TcpSocketState::DctcpEcn)
        {
          SocketIpv6TclassTag ipTclassTag;
          ipTclassTag.SetTclass (MarkEcnCodePoint (GetIpv6Tclass (), m_tcb->m_ectCodePoint));
          p->AddPacketTag (ipTclassTag);
        }
    }

  if (IsManualIpTtl ())
    {
      SocketIpTtlTag ipTtlTag;
      ipTtlTag.SetTtl (GetIpTtl ());
      p->AddPacketTag (ipTtlTag);
    }

  if (IsManualIpv6HopLimit ())
    {
      SocketIpv6HopLimitTag ipHopLimitTag;
      ipHopLimitTag.SetHopLimit (GetIpv6HopLimit ());
      p->AddPacketTag (ipHopLimitTag);
    }

  uint8_t priority = GetPriority ();
  if (priority)
    {
      SocketPriorityTag priorityTag;
      priorityTag.SetPriority (priority);
      p->ReplacePacketTag (priorityTag);
    }
}

/* Extract at most maxSize bytes from the TxBuffer at sequence seq, add the
    TCP header, and send to TcpL4Protocol */
uint32_t
TcpSocketBase::SendDataPacket (SequenceNumber32 seq, uint32_t maxSize, bool withAck)
{
  NS_LOG_FUNCTION (this << seq << maxSize << withAck);

  bool isStartOfTransmission = BytesInFlight () == 0U;
  TcpTxItem *outItem = m_txBuffer->CopyFromSequence (maxSize, seq);

  m_rateOps->SkbSent(outItem, isStartOfTransmission);

  bool isRetransmission = outItem->IsRetrans ();
  Ptr<Packet> p = outItem->GetPacketCopy ();
  uint32_t sz = p->GetSize (); // Size of packet
  uint8_t flags = withAck ? TcpHeader::ACK : 0;
  uint32_t remainingData = m_txBuffer->SizeFromSequence (seq + SequenceNumber32 (sz));

  // TCP sender should not send data out of the window advertised by the
  // peer when it is not retransmission.
  NS_ASSERT (isRetransmission || ((m_highRxAckMark + SequenceNumber32 (m_rWnd)) >= (seq + SequenceNumber32 (maxSize))));

  if (IsPacingEnabled ())
    {
      NS_LOG_INFO ("Pacing is enabled");
      if (m_pacingTimer.IsExpired ())
        {
          NS_LOG_DEBUG ("Current Pacing Rate " << m_tcb->m_pacingRate);
          NS_LOG_DEBUG ("Timer is in expired state, activate it " << m_tcb->m_pacingRate.Get ().CalculateBytesTxTime (sz));
          m_pacingTimer.Schedule (m_tcb->m_pacingRate.Get ().CalculateBytesTxTime (sz));
        }
      else
        {
          NS_LOG_INFO ("Timer is already in running state");
        }
    }
  else
    {
      NS_LOG_INFO ("Pacing is disabled");
    }

  if (withAck)
    {
      m_delAckEvent.Cancel ();
      m_delAckCount = 0;
    }

  if (m_tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD && m_ecnEchoSeq.Get() > m_ecnCWRSeq.Get () && !isRetransmission)
    {
      NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CWR_SENT");
      m_tcb->m_ecnState = TcpSocketState::ECN_CWR_SENT;
      m_ecnCWRSeq = seq;
      flags |= TcpHeader::CWR;
      NS_LOG_INFO ("CWR flags set");
    }

  AddSocketTags (p);

  if (m_closeOnEmpty && (remainingData == 0))
    {
      flags |= TcpHeader::FIN;
      if (m_state == ESTABLISHED)
        { // On active close: I am the first one to send FIN
          NS_LOG_DEBUG ("ESTABLISHED -> FIN_WAIT_1");
          m_state = FIN_WAIT_1;
        }
      else if (m_state == CLOSE_WAIT)
        { // On passive close: Peer sent me FIN already
          NS_LOG_DEBUG ("CLOSE_WAIT -> LAST_ACK");
          m_state = LAST_ACK;
        }
    }
  TcpHeader header;
  header.SetFlags (flags);
  header.SetSequenceNumber (seq);
  header.SetAckNumber (m_tcb->m_rxBuffer->NextRxSequence ());
  if (m_endPoint)
    {
      header.SetSourcePort (m_endPoint->GetLocalPort ());
      header.SetDestinationPort (m_endPoint->GetPeerPort ());
    }
  else
    {
      header.SetSourcePort (m_endPoint6->GetLocalPort ());
      header.SetDestinationPort (m_endPoint6->GetPeerPort ());
    }
  header.SetWindowSize (AdvertisedWindowSize ());
  AddOptions (header);

  if (m_retxEvent.IsExpired ())
    {
      // Schedules retransmit timeout. m_rto should be already doubled.

      NS_LOG_LOGIC (this << " SendDataPacket Schedule ReTxTimeout at time " <<
                    Simulator::Now ().GetSeconds () << " to expire at time " <<
                    (Simulator::Now () + m_rto.Get ()).GetSeconds () );
      m_retxEvent = Simulator::Schedule (m_rto, &TcpSocketBase::ReTxTimeout, this);
    }

  m_txTrace (p, header, this);

  if (m_endPoint)
    {
      m_tcp->SendPacket (p, header, m_endPoint->GetLocalAddress (),
                         m_endPoint->GetPeerAddress (), m_boundnetdevice);
      NS_LOG_DEBUG ("Send segment of size " << sz << " with remaining data " <<
                    remainingData << " via TcpL4Protocol to " <<  m_endPoint->GetPeerAddress () <<
                    ". Header " << header);
    }
  else
    {
      m_tcp->SendPacket (p, header, m_endPoint6->GetLocalAddress (),
                         m_endPoint6->GetPeerAddress (), m_boundnetdevice);
      NS_LOG_DEBUG ("Send segment of size " << sz << " with remaining data " <<
                    remainingData << " via TcpL4Protocol to " <<  m_endPoint6->GetPeerAddress () <<
                    ". Header " << header);
    }

  UpdateRttHistory (seq, sz, isRetransmission);

  // Update bytes sent during recovery phase
  if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY || m_tcb->m_congState == TcpSocketState::CA_CWR)
    {
      m_recoveryOps->UpdateBytesSent (sz);
    }

  // Notify the application of the data being sent unless this is a retransmit
  if (!isRetransmission)
    {
      Simulator::ScheduleNow (&TcpSocketBase::NotifyDataSent, this,
                              (seq + sz - m_tcb->m_highTxMark.Get ()));
    }
  // Update highTxMark
  m_tcb->m_highTxMark = std::max (seq + sz, m_tcb->m_highTxMark.Get ());
  return sz;
}

void
TcpSocketBase::UpdateRttHistory (const SequenceNumber32 &seq, uint32_t sz,
                                 bool isRetransmission)
{
  NS_LOG_FUNCTION (this);

  // update the history of sequence numbers used to calculate the RTT
  if (isRetransmission == false)
    { // This is the next expected one, just log at end
      m_history.push_back (RttHistory (seq, sz, Simulator::Now ()));
    }
  else
    { // This is a retransmit, find in list and mark as re-tx
      for (std::deque<RttHistory>::iterator i = m_history.begin (); i != m_history.end (); ++i)
        {
          if ((seq >= i->seq) && (seq < (i->seq + SequenceNumber32 (i->count))))
            { // Found it
              i->retx = true;
              i->count = ((seq + SequenceNumber32 (sz)) - i->seq); // And update count in hist
              break;
            }
        }
    }
}

// Note that this function did not implement the PSH flag
uint32_t
TcpSocketBase::SendPendingData (bool withAck)
{
  NS_LOG_FUNCTION (this << withAck);
  if (m_txBuffer->Size () == 0)
    {
      return false;                           // Nothing to send
    }
  if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
      NS_LOG_INFO ("TcpSocketBase::SendPendingData: No endpoint; m_shutdownSend=" << m_shutdownSend);
      return false; // Is this the right way to handle this condition?
    }

  uint32_t nPacketsSent = 0;
  uint32_t availableWindow = AvailableWindow ();

  // RFC 6675, Section (C)
  // If cwnd - pipe >= 1 SMSS, the sender SHOULD transmit one or more
  // segments as follows:
  // (NOTE: We check > 0, and do the checks for segmentSize in the following
  // else branch to control silly window syndrome and Nagle)
  while (availableWindow > 0)
    {
      if (IsPacingEnabled ())
        {
          NS_LOG_INFO ("Pacing is enabled");
          if (m_pacingTimer.IsRunning ())
            {
              NS_LOG_INFO ("Skipping Packet due to pacing" << m_pacingTimer.GetDelayLeft ());
              break;
            }
          NS_LOG_INFO ("Timer is not running");
        }

      if (m_tcb->m_congState == TcpSocketState::CA_OPEN
          && m_state == TcpSocket::FIN_WAIT_1)
        {
          NS_LOG_INFO ("FIN_WAIT and OPEN state; no data to transmit");
          break;
        }
      // (C.1) The scoreboard MUST be queried via NextSeg () for the
      //       sequence number range of the next segment to transmit (if
      //       any), and the given segment sent.  If NextSeg () returns
      //       failure (no data to send), return without sending anything
      //       (i.e., terminate steps C.1 -- C.5).
      SequenceNumber32 next;
      SequenceNumber32 nextHigh;
      bool enableRule3 = m_sackEnabled && m_tcb->m_congState == TcpSocketState::CA_RECOVERY;
      if (!m_txBuffer->NextSeg (&next, &nextHigh, enableRule3))
        {
          NS_LOG_INFO ("no valid seq to transmit, or no data available");
          break;
        }
      else
        {
          // It's time to transmit, but before do silly window and Nagle's check
          uint32_t availableData = m_txBuffer->SizeFromSequence (next);

          // If there's less app data than the full window, ask the app for more
          // data before trying to send
          if (availableData < availableWindow)
            {
              NotifySend (GetTxAvailable ());
            }

          // Stop sending if we need to wait for a larger Tx window (prevent silly window syndrome)
          // but continue if we don't have data
          if (availableWindow < m_tcb->m_segmentSize && availableData > availableWindow)
            {
              NS_LOG_LOGIC ("Preventing Silly Window Syndrome. Wait to send.");
              break; // No more
            }
          // Nagle's algorithm (RFC896): Hold off sending if there is unacked data
          // in the buffer and the amount of data to send is less than one segment
          if (!m_noDelay && UnAckDataCount () > 0 && availableData < m_tcb->m_segmentSize)
            {
              NS_LOG_DEBUG ("Invoking Nagle's algorithm for seq " << next <<
                            ", SFS: " << m_txBuffer->SizeFromSequence (next) <<
                            ". Wait to send.");
              break;
            }

          uint32_t s = std::min (availableWindow, m_tcb->m_segmentSize);
          // NextSeg () may have further constrained the segment size
          uint32_t maxSizeToSend = static_cast<uint32_t> (nextHigh - next);
          s = std::min (s, maxSizeToSend);

          // (C.2) If any of the data octets sent in (C.1) are below HighData,
          //       HighRxt MUST be set to the highest sequence number of the
          //       retransmitted segment unless NextSeg () rule (4) was
          //       invoked for this retransmission.
          // (C.3) If any of the data octets sent in (C.1) are above HighData,
          //       HighData must be updated to reflect the transmission of
          //       previously unsent data.
          //
          // These steps are done in m_txBuffer with the tags.
          if (m_tcb->m_nextTxSequence != next)
            {
              m_tcb->m_nextTxSequence = next;
            }
          if (m_tcb->m_bytesInFlight.Get () == 0)
            {
              m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_TX_START);
            }
          uint32_t sz = SendDataPacket (m_tcb->m_nextTxSequence, s, withAck);

          NS_LOG_LOGIC (" rxwin " << m_rWnd <<
                        " segsize " << m_tcb->m_segmentSize <<
                        " highestRxAck " << m_txBuffer->HeadSequence () <<
                        " pd->Size " << m_txBuffer->Size () <<
                        " pd->SFS " << m_txBuffer->SizeFromSequence (m_tcb->m_nextTxSequence));

          NS_LOG_DEBUG ("cWnd: " << m_tcb->m_cWnd <<
                        " total unAck: " << UnAckDataCount () <<
                        " sent seq " << m_tcb->m_nextTxSequence <<
                        " size " << sz);
          m_tcb->m_nextTxSequence += sz;
          ++nPacketsSent;
          if (IsPacingEnabled ())
            {
              NS_LOG_INFO ("Pacing is enabled");
              if (m_pacingTimer.IsExpired ())
                {
                  NS_LOG_DEBUG ("Current Pacing Rate " << m_tcb->m_pacingRate);
                  NS_LOG_DEBUG ("Timer is in expired state, activate it " << m_tcb->m_pacingRate.Get ().CalculateBytesTxTime (sz));
                  m_pacingTimer.Schedule (m_tcb->m_pacingRate.Get ().CalculateBytesTxTime (sz));
                  break;
                }
            }
        }

      // (C.4) The estimate of the amount of data outstanding in the
      //       network must be updated by incrementing pipe by the number
      //       of octets transmitted in (C.1).
      //
      // Done in BytesInFlight, inside AvailableWindow.
      availableWindow = AvailableWindow ();

      // (C.5) If cwnd - pipe >= 1 SMSS, return to (C.1)
      // loop again!
    }

  if (nPacketsSent > 0)
    {
      if (!m_sackEnabled)
        {
          if (!m_limitedTx)
            {
              // We can't transmit in CA_DISORDER without limitedTx active
              NS_ASSERT (m_tcb->m_congState != TcpSocketState::CA_DISORDER);
            }
        }

      NS_LOG_DEBUG ("SendPendingData sent " << nPacketsSent << " segments");
    }
  else
    {
      NS_LOG_DEBUG ("SendPendingData no segments sent");
    }
  return nPacketsSent;
}

uint32_t
TcpSocketBase::UnAckDataCount () const
{
  return m_tcb->m_highTxMark - m_txBuffer->HeadSequence ();
}

uint32_t
TcpSocketBase::BytesInFlight () const
{
  uint32_t bytesInFlight = m_txBuffer->BytesInFlight ();
  // Ugly, but we are not modifying the state; m_bytesInFlight is used
  // only for tracing purpose.
  m_tcb->m_bytesInFlight = bytesInFlight;

  NS_LOG_DEBUG ("Returning calculated bytesInFlight: " << bytesInFlight);
  return bytesInFlight;
}

uint32_t
TcpSocketBase::Window (void) const
{
  return std::min (m_rWnd.Get (), m_tcb->m_cWnd.Get ());
}

uint32_t
TcpSocketBase::AvailableWindow () const
{
  uint32_t win = Window ();             // Number of bytes allowed to be outstanding
  uint32_t inflight = BytesInFlight (); // Number of outstanding bytes
  return (inflight > win) ? 0 : win - inflight;
}

uint16_t
TcpSocketBase::AdvertisedWindowSize (bool scale) const
{
  NS_LOG_FUNCTION (this << scale);
  uint32_t w;

  // We don't want to advertise 0 after a FIN is received. So, we just use
  // the previous value of the advWnd.
  if (m_tcb->m_rxBuffer->GotFin ())
    {
      w = m_advWnd;
    }
  else
    {
      NS_ASSERT_MSG (m_tcb->m_rxBuffer->MaxRxSequence () - m_tcb->m_rxBuffer->NextRxSequence () >= 0,
                     "Unexpected sequence number values");
      w = static_cast<uint32_t> (m_tcb->m_rxBuffer->MaxRxSequence () - m_tcb->m_rxBuffer->NextRxSequence ());
    }

  // Ugly, but we are not modifying the state, that variable
  // is used only for tracing purpose.
  if (w != m_advWnd)
    {
      const_cast<TcpSocketBase*> (this)->m_advWnd = w;
    }
  if (scale)
    {
      w >>= m_rcvWindShift;
    }
  if (w > m_maxWinSize)
    {
      w = m_maxWinSize;
      NS_LOG_WARN ("Adv window size truncated to " << m_maxWinSize << "; possibly to avoid overflow of the 16-bit integer");
    }
  NS_LOG_LOGIC ("Returning AdvertisedWindowSize of " << static_cast<uint16_t> (w));
  return static_cast<uint16_t> (w);
}

// Receipt of new packet, put into Rx buffer
void
TcpSocketBase::ReceivedData (Ptr<Packet> p, const TcpHeader& tcpHeader)
{
  NS_LOG_FUNCTION (this << tcpHeader);
  NS_LOG_DEBUG ("Data segment, seq=" << tcpHeader.GetSequenceNumber () <<
                " pkt size=" << p->GetSize () );

  // Put into Rx buffer
  SequenceNumber32 expectedSeq = m_tcb->m_rxBuffer->NextRxSequence ();
  if (!m_tcb->m_rxBuffer->Add (p, tcpHeader))
    { // Insert failed: No data or RX buffer full
      if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
          SendEmptyPacket (TcpHeader::ACK | TcpHeader::ECE);
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
          m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
      else
        {
          SendEmptyPacket (TcpHeader::ACK);
        }
      return;
    }
  // Notify app to receive if necessary
  if (expectedSeq < m_tcb->m_rxBuffer->NextRxSequence ())
    { // NextRxSeq advanced, we have something to send to the app
      if (!m_shutdownRecv)
        {
          NotifyDataRecv ();
        }
      // Handle exceptions
      if (m_closeNotified)
        {
          NS_LOG_WARN ("Why TCP " << this << " got data after close notification?");
        }
      // If we received FIN before and now completed all "holes" in rx buffer,
      // invoke peer close procedure
      if (m_tcb->m_rxBuffer->Finished () && (tcpHeader.GetFlags () & TcpHeader::FIN) == 0)
        {
          DoPeerClose ();
          return;
        }
    }
  // Now send a new ACK packet acknowledging all received and delivered data
  if (m_tcb->m_rxBuffer->Size () > m_tcb->m_rxBuffer->Available () || m_tcb->m_rxBuffer->NextRxSequence () > expectedSeq + p->GetSize ())
    { // A gap exists in the buffer, or we filled a gap: Always ACK
      m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_NON_DELAYED_ACK);
      if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
          SendEmptyPacket (TcpHeader::ACK | TcpHeader::ECE);
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
          m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
      else
        {
          SendEmptyPacket (TcpHeader::ACK);
        }
    }
  else
    { // In-sequence packet: ACK if delayed ack count allows
      if (++m_delAckCount >= m_delAckMaxCount)
        {
          m_delAckEvent.Cancel ();
          m_delAckCount = 0;
          m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_NON_DELAYED_ACK);
          if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
            {
              NS_LOG_DEBUG("Congestion algo " << m_congestionControl->GetName ());
              SendEmptyPacket (TcpHeader::ACK | TcpHeader::ECE);
              NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
              m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
            }
          else
            {
              SendEmptyPacket (TcpHeader::ACK);
            }
        }
      else if (!m_delAckEvent.IsExpired ())
        {
          m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
        }
      else if (m_delAckEvent.IsExpired ())
        {
          m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
          m_delAckEvent = Simulator::Schedule (m_delAckTimeout,
                                               &TcpSocketBase::DelAckTimeout, this);
          NS_LOG_LOGIC (this << " scheduled delayed ACK at " <<
                        (Simulator::Now () + Simulator::GetDelayLeft (m_delAckEvent)).GetSeconds ());
        }
    }
}

void
TcpSocketBase::EstimateRtt (const TcpHeader& tcpHeader)
{
  SequenceNumber32 ackSeq = tcpHeader.GetAckNumber ();
  Time m = Time (0.0);

  // An ack has been received, calculate rtt and log this measurement
  // Note we use a linear search (O(n)) for this since for the common
  // case the ack'ed packet will be at the head of the list
  if (!m_history.empty ())
    {
      RttHistory& h = m_history.front ();
      if (!h.retx && ackSeq >= (h.seq + SequenceNumber32 (h.count)))
        { // Ok to use this sample
          if (m_timestampEnabled && tcpHeader.HasOption (TcpOption::TS))
            {
              Ptr<const TcpOptionTS> ts;
              ts = DynamicCast<const TcpOptionTS> (tcpHeader.GetOption (TcpOption::TS));
              m = TcpOptionTS::ElapsedTimeFromTsValue (ts->GetEcho ());
              if (m.IsZero ())
                {
                  NS_LOG_LOGIC ("TcpSocketBase::EstimateRtt - RTT calculated from TcpOption::TS is zero, approximating to 1us.");
                  m = MicroSeconds (1);
                }
            }
          else
            {
              m = Simulator::Now () - h.time; // Elapsed time
            }
        }
    }

  // Now delete all ack history with seq <= ack
  while (!m_history.empty ())
    {
      RttHistory& h = m_history.front ();
      if ((h.seq + SequenceNumber32 (h.count)) > ackSeq)
        {
          break;                                                              // Done removing
        }
      m_history.pop_front (); // Remove
    }

  if (!m.IsZero ())
    {
      m_rtt->Measurement (m);                // Log the measurement
      // RFC 6298, clause 2.4
      m_rto = Max (m_rtt->GetEstimate () + Max (m_clockGranularity, m_rtt->GetVariation () * 4), m_minRto);
      m_tcb->m_lastRtt = m_rtt->GetEstimate ();
      m_tcb->m_minRtt = std::min (m_tcb->m_lastRtt.Get (), m_tcb->m_minRtt);
      NS_LOG_INFO (this << m_tcb->m_lastRtt << m_tcb->m_minRtt);
    }
}

// Called by the ReceivedAck() when new ACK received and by ProcessSynRcvd()
// when the three-way handshake completed. This cancels retransmission timer
// and advances Tx window
void
TcpSocketBase::NewAck (SequenceNumber32 const& ack, bool resetRTO)
{
  NS_LOG_FUNCTION (this << ack);

  // Reset the data retransmission count. We got a new ACK!
  m_dataRetrCount = m_dataRetries;

  if (m_state != SYN_RCVD && resetRTO)
    { // Set RTO unless the ACK is received in SYN_RCVD state
      NS_LOG_LOGIC (this << " Cancelled ReTxTimeout event which was set to expire at " <<
                    (Simulator::Now () + Simulator::GetDelayLeft (m_retxEvent)).GetSeconds ());
      m_retxEvent.Cancel ();
      // On receiving a "New" ack we restart retransmission timer .. RFC 6298
      // RFC 6298, clause 2.4
      m_rto = Max (m_rtt->GetEstimate () + Max (m_clockGranularity, m_rtt->GetVariation () * 4), m_minRto);

      NS_LOG_LOGIC (this << " Schedule ReTxTimeout at time " <<
                    Simulator::Now ().GetSeconds () << " to expire at time " <<
                    (Simulator::Now () + m_rto.Get ()).GetSeconds ());
      m_retxEvent = Simulator::Schedule (m_rto, &TcpSocketBase::ReTxTimeout, this);
    }

  // Note the highest ACK and tell app to send more
  NS_LOG_LOGIC ("TCP " << this << " NewAck " << ack <<
                " numberAck " << (ack - m_txBuffer->HeadSequence ())); // Number bytes ack'ed

  if (GetTxAvailable () > 0)
    {
      NotifySend (GetTxAvailable ());
    }
  if (ack > m_tcb->m_nextTxSequence)
    {
      m_tcb->m_nextTxSequence = ack; // If advanced
    }
  if (m_txBuffer->Size () == 0 && m_state != FIN_WAIT_1 && m_state != CLOSING)
    { // No retransmit timer if no data to retransmit
      NS_LOG_LOGIC (this << " Cancelled ReTxTimeout event which was set to expire at " <<
                    (Simulator::Now () + Simulator::GetDelayLeft (m_retxEvent)).GetSeconds ());
      m_retxEvent.Cancel ();
    }
}

// Retransmit timeout
void
TcpSocketBase::ReTxTimeout ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " ReTxTimeout Expired at time " << Simulator::Now ().GetSeconds ());
  // If erroneous timeout in closed/timed-wait state, just return
  if (m_state == CLOSED || m_state == TIME_WAIT)
    {
      return;
    }

  if (m_state == SYN_SENT)
    {
      NS_ASSERT (m_synCount > 0);
      if (m_tcb->m_useEcn == TcpSocketState::On)
        {
          SendEmptyPacket (TcpHeader::SYN | TcpHeader::ECE | TcpHeader::CWR);
        }
      else
        {
          SendEmptyPacket (TcpHeader::SYN);
        }
      return;
    }

  // Retransmit non-data packet: Only if in FIN_WAIT_1 or CLOSING state
  if (m_txBuffer->Size () == 0)
    {
      if (m_state == FIN_WAIT_1 || m_state == CLOSING)
        { // Must have lost FIN, re-send
          SendEmptyPacket (TcpHeader::FIN);
        }
      return;
    }

  NS_LOG_DEBUG ("Checking if Connection is Established");
  // If all data are received (non-closing socket and nothing to send), just return
  if (m_state <= ESTABLISHED && m_txBuffer->HeadSequence () >= m_tcb->m_highTxMark && m_txBuffer->Size () == 0)
    {
      NS_LOG_DEBUG ("Already Sent full data" << m_txBuffer->HeadSequence () << " " << m_tcb->m_highTxMark);
      return;
    }

  if (m_dataRetrCount == 0)
    {
      NS_LOG_INFO ("No more data retries available. Dropping connection");
      NotifyErrorClose ();
      DeallocateEndPoint ();
      return;
    }
  else
    {
      --m_dataRetrCount;
    }

  uint32_t inFlightBeforeRto = BytesInFlight ();
  bool resetSack = !m_sackEnabled; // Reset SACK information if SACK is not enabled.
                                   // The information in the TcpTxBuffer is guessed, in this case.

  // Reset dupAckCount
  m_dupAckCount = 0;
  if (!m_sackEnabled)
    {
      m_txBuffer->ResetRenoSack ();
    }

  // From RFC 6675, Section 5.1
  // [RFC2018] suggests that a TCP sender SHOULD expunge the SACK
  // information gathered from a receiver upon a retransmission timeout
  // (RTO) "since the timeout might indicate that the data receiver has
  // reneged."  Additionally, a TCP sender MUST "ignore prior SACK
  // information in determining which data to retransmit."
  // It has been suggested that, as long as robust tests for
  // reneging are present, an implementation can retain and use SACK
  // information across a timeout event [Errata1610].
  // The head of the sent list will not be marked as sacked, therefore
  // will be retransmitted, if the receiver renegotiate the SACK blocks
  // that we received.
  m_txBuffer->SetSentListLost (resetSack);

  // From RFC 6675, Section 5.1
  // If an RTO occurs during loss recovery as specified in this document,
  // RecoveryPoint MUST be set to HighData.  Further, the new value of
  // RecoveryPoint MUST be preserved and the loss recovery algorithm
  // outlined in this document MUST be terminated.
  m_recover = m_tcb->m_highTxMark;
  m_recoverActive = true;

  // RFC 6298, clause 2.5, double the timer
  Time doubledRto = m_rto + m_rto;
  m_rto = Min (doubledRto, Time::FromDouble (60,  Time::S));

  // Empty RTT history
  m_history.clear ();

  // Please don't reset highTxMark, it is used for retransmission detection

  // When a TCP sender detects segment loss using the retransmission timer
  // and the given segment has not yet been resent by way of the
  // retransmission timer, decrease ssThresh
  if (m_tcb->m_congState != TcpSocketState::CA_LOSS || !m_txBuffer->IsHeadRetransmitted ())
    {
      m_tcb->m_ssThresh = m_congestionControl->GetSsThresh (m_tcb, inFlightBeforeRto);
    }

  // Cwnd set to 1 MSS
  m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_LOSS);
  m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_LOSS);
  m_tcb->m_congState = TcpSocketState::CA_LOSS;
  m_tcb->m_cWnd = m_tcb->m_segmentSize;
  m_tcb->m_cWndInfl = m_tcb->m_cWnd;

  m_pacingTimer.Cancel ();

  NS_LOG_DEBUG ("RTO. Reset cwnd to " <<  m_tcb->m_cWnd << ", ssthresh to " <<
                m_tcb->m_ssThresh << ", restart from seqnum " <<
                m_txBuffer->HeadSequence () << " doubled rto to " <<
                m_rto.Get ().GetSeconds () << " s");

  NS_ASSERT_MSG (BytesInFlight () == 0, "There are some bytes in flight after an RTO: " <<
                 BytesInFlight ());

  SendPendingData (m_connected);

  NS_ASSERT_MSG (BytesInFlight () <= m_tcb->m_segmentSize,
                 "In flight (" << BytesInFlight () <<
                 ") there is more than one segment (" << m_tcb->m_segmentSize << ")");
}

void
TcpSocketBase::DelAckTimeout (void)
{
  m_delAckCount = 0;
  m_congestionControl->CwndEvent (m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
  if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
    {
      SendEmptyPacket (TcpHeader::ACK | TcpHeader::ECE);
      m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
    }
  else
    {
      SendEmptyPacket (TcpHeader::ACK);
    }
}

void
TcpSocketBase::LastAckTimeout (void)
{
  NS_LOG_FUNCTION (this);

  m_lastAckEvent.Cancel ();
  if (m_state == LAST_ACK)
    {
      if (m_dataRetrCount == 0)
        {
          NS_LOG_INFO ("LAST-ACK: No more data retries available. Dropping connection");
          NotifyErrorClose ();
          DeallocateEndPoint ();
          return;
        }
      m_dataRetrCount--;
      SendEmptyPacket (TcpHeader::FIN | TcpHeader::ACK);
      NS_LOG_LOGIC ("TcpSocketBase " << this << " rescheduling LATO1");
      Time lastRto = m_rtt->GetEstimate () + Max (m_clockGranularity, m_rtt->GetVariation () * 4);
      m_lastAckEvent = Simulator::Schedule (lastRto, &TcpSocketBase::LastAckTimeout, this);
    }
}

// Send 1-byte data to probe for the window size at the receiver when
// the local knowledge tells that the receiver has zero window size
// C.f.: RFC793 p.42, RFC1112 sec.4.2.2.17
void
TcpSocketBase::PersistTimeout ()
{
  NS_LOG_LOGIC ("PersistTimeout expired at " << Simulator::Now ().GetSeconds ());
  m_persistTimeout = std::min (Seconds (60), Time (2 * m_persistTimeout)); // max persist timeout = 60s
  Ptr<Packet> p = m_txBuffer->CopyFromSequence (1, m_tcb->m_nextTxSequence)->GetPacketCopy ();
  m_txBuffer->ResetLastSegmentSent ();
  TcpHeader tcpHeader;
  tcpHeader.SetSequenceNumber (m_tcb->m_nextTxSequence);
  tcpHeader.SetAckNumber (m_tcb->m_rxBuffer->NextRxSequence ());
  tcpHeader.SetWindowSize (AdvertisedWindowSize ());
  if (m_endPoint != nullptr)
    {
      tcpHeader.SetSourcePort (m_endPoint->GetLocalPort ());
      tcpHeader.SetDestinationPort (m_endPoint->GetPeerPort ());
    }
  else
    {
      tcpHeader.SetSourcePort (m_endPoint6->GetLocalPort ());
      tcpHeader.SetDestinationPort (m_endPoint6->GetPeerPort ());
    }
  AddOptions (tcpHeader);
  //Send a packet tag for setting ECT bits in IP header
  if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
    {
      SocketIpTosTag ipTosTag;
      ipTosTag.SetTos (MarkEcnCodePoint (0, m_tcb->m_ectCodePoint));
      p->AddPacketTag (ipTosTag);

      SocketIpv6TclassTag ipTclassTag;
      ipTclassTag.SetTclass (MarkEcnCodePoint (0, m_tcb->m_ectCodePoint));
      p->AddPacketTag (ipTclassTag);
    }
  m_txTrace (p, tcpHeader, this);

  if (m_endPoint != nullptr)
    {
      m_tcp->SendPacket (p, tcpHeader, m_endPoint->GetLocalAddress (),
                         m_endPoint->GetPeerAddress (), m_boundnetdevice);
    }
  else
    {
      m_tcp->SendPacket (p, tcpHeader, m_endPoint6->GetLocalAddress (),
                         m_endPoint6->GetPeerAddress (), m_boundnetdevice);
    }

  NS_LOG_LOGIC ("Schedule persist timeout at time "
                << Simulator::Now ().GetSeconds () << " to expire at time "
                << (Simulator::Now () + m_persistTimeout).GetSeconds ());
  m_persistEvent = Simulator::Schedule (m_persistTimeout, &TcpSocketBase::PersistTimeout, this);
}

void
TcpSocketBase::DoRetransmit ()
{
  NS_LOG_FUNCTION (this);
  bool res;
  SequenceNumber32 seq;
  SequenceNumber32 seqHigh;
  uint32_t maxSizeToSend;

  // Find the first segment marked as lost and not retransmitted. With Reno,
  // that should be the head
  res = m_txBuffer->NextSeg (&seq, &seqHigh, false);
  if (!res)
    {
      // We have already retransmitted the head. However, we still received
      // three dupacks, or the RTO expired, but no data to transmit.
      // Therefore, re-send again the head.
      seq = m_txBuffer->HeadSequence ();
      maxSizeToSend = m_tcb->m_segmentSize;
    }
  else
    {
      // NextSeg() may constrain the segment size when res is true
      maxSizeToSend = static_cast<uint32_t> (seqHigh - seq);
    }
  NS_ASSERT (m_sackEnabled || seq == m_txBuffer->HeadSequence ());

  NS_LOG_INFO ("Retransmitting " << seq);
  // Update the trace and retransmit the segment
  m_tcb->m_nextTxSequence = seq;
  uint32_t sz = SendDataPacket (m_tcb->m_nextTxSequence, maxSizeToSend, true);

  NS_ASSERT (sz > 0);
}

void
TcpSocketBase::CancelAllTimers ()
{
  m_retxEvent.Cancel ();
  m_persistEvent.Cancel ();
  m_delAckEvent.Cancel ();
  m_lastAckEvent.Cancel ();
  m_timewaitEvent.Cancel ();
  m_sendPendingDataEvent.Cancel ();
  m_pacingTimer.Cancel ();
}

/* Move TCP to Time_Wait state and schedule a transition to Closed state */
void
TcpSocketBase::TimeWait ()
{
  NS_LOG_DEBUG (TcpStateName[m_state] << " -> TIME_WAIT");
  m_state = TIME_WAIT;
  CancelAllTimers ();
  if (!m_closeNotified)
    {
      // Technically the connection is not fully closed, but we notify now
      // because an implementation (real socket) would behave as if closed.
      // Notify normal close when entering TIME_WAIT or leaving LAST_ACK.
      NotifyNormalClose ();
      m_closeNotified = true;
    }
  // Move from TIME_WAIT to CLOSED after 2*MSL. Max segment lifetime is 2 min
  // according to RFC793, p.28
  m_timewaitEvent = Simulator::Schedule (Seconds (2 * m_msl),
                                         &TcpSocketBase::CloseAndNotify, this);
}

/* Below are the attribute get/set functions */

void
TcpSocketBase::SetSndBufSize (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_txBuffer->SetMaxBufferSize (size);
}

uint32_t
TcpSocketBase::GetSndBufSize (void) const
{
  return m_txBuffer->MaxBufferSize ();
}

void
TcpSocketBase::SetRcvBufSize (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  uint32_t oldSize = GetRcvBufSize ();

  m_tcb->m_rxBuffer->SetMaxBufferSize (size);

  /* The size has (manually) increased. Actively inform the other end to prevent
   * stale zero-window states.
   */
  if (oldSize < size && m_connected)
    {
      if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
          SendEmptyPacket (TcpHeader::ACK | TcpHeader::ECE);
          NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
          m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
      else
        {
          SendEmptyPacket (TcpHeader::ACK);
        }
    }
}

uint32_t
TcpSocketBase::GetRcvBufSize (void) const
{
  return m_tcb->m_rxBuffer->MaxBufferSize ();
}

void
TcpSocketBase::SetSegSize (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_tcb->m_segmentSize = size;
  m_txBuffer->SetSegmentSize (size);

  NS_ABORT_MSG_UNLESS (m_state == CLOSED, "Cannot change segment size dynamically.");
}

uint32_t
TcpSocketBase::GetSegSize (void) const
{
  return m_tcb->m_segmentSize;
}

void
TcpSocketBase::SetConnTimeout (Time timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  m_cnTimeout = timeout;
}

Time
TcpSocketBase::GetConnTimeout (void) const
{
  return m_cnTimeout;
}

void
TcpSocketBase::SetSynRetries (uint32_t count)
{
  NS_LOG_FUNCTION (this << count);
  m_synRetries = count;
}

uint32_t
TcpSocketBase::GetSynRetries (void) const
{
  return m_synRetries;
}

void
TcpSocketBase::SetDataRetries (uint32_t retries)
{
  NS_LOG_FUNCTION (this << retries);
  m_dataRetries = retries;
}

uint32_t
TcpSocketBase::GetDataRetries (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dataRetries;
}

void
TcpSocketBase::SetDelAckTimeout (Time timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  m_delAckTimeout = timeout;
}

Time
TcpSocketBase::GetDelAckTimeout (void) const
{
  return m_delAckTimeout;
}

void
TcpSocketBase::SetDelAckMaxCount (uint32_t count)
{
  NS_LOG_FUNCTION (this << count);
  m_delAckMaxCount = count;
}

uint32_t
TcpSocketBase::GetDelAckMaxCount (void) const
{
  return m_delAckMaxCount;
}

void
TcpSocketBase::SetTcpNoDelay (bool noDelay)
{
  NS_LOG_FUNCTION (this << noDelay);
  m_noDelay = noDelay;
}

bool
TcpSocketBase::GetTcpNoDelay (void) const
{
  return m_noDelay;
}

void
TcpSocketBase::SetPersistTimeout (Time timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  m_persistTimeout = timeout;
}

Time
TcpSocketBase::GetPersistTimeout (void) const
{
  return m_persistTimeout;
}

bool
TcpSocketBase::SetAllowBroadcast (bool allowBroadcast)
{
  // Broadcast is not implemented. Return true only if allowBroadcast==false
  return (!allowBroadcast);
}

bool
TcpSocketBase::GetAllowBroadcast (void) const
{
  return false;
}

void
TcpSocketBase::AddOptions (TcpHeader& header)
{
  NS_LOG_FUNCTION (this << header);

  if (m_timestampEnabled)
    {
      AddOptionTimestamp (header);
    }
}

void
TcpSocketBase::ProcessOptionWScale (const Ptr<const TcpOption> option)
{
  NS_LOG_FUNCTION (this << option);

  Ptr<const TcpOptionWinScale> ws = DynamicCast<const TcpOptionWinScale> (option);

  // In naming, we do the contrary of RFC 1323. The received scaling factor
  // is Rcv.Wind.Scale (and not Snd.Wind.Scale)
  m_sndWindShift = ws->GetScale ();

  if (m_sndWindShift > 14)
    {
      NS_LOG_WARN ("Possible error; m_sndWindShift exceeds 14: " << m_sndWindShift);
      m_sndWindShift = 14;
    }

  NS_LOG_INFO (m_node->GetId () << " Received a scale factor of " <<
               static_cast<int> (m_sndWindShift));
}

uint8_t
TcpSocketBase::CalculateWScale () const
{
  NS_LOG_FUNCTION (this);
  uint32_t maxSpace = m_tcb->m_rxBuffer->MaxBufferSize ();
  uint8_t scale = 0;

  while (maxSpace > m_maxWinSize)
    {
      maxSpace = maxSpace >> 1;
      ++scale;
    }

  if (scale > 14)
    {
      NS_LOG_WARN ("Possible error; scale exceeds 14: " << scale);
      scale = 14;
    }

  NS_LOG_INFO ("Node " << m_node->GetId () << " calculated wscale factor of " <<
               static_cast<int> (scale) << " for buffer size " << m_tcb->m_rxBuffer->MaxBufferSize ());
  return scale;
}

void
TcpSocketBase::AddOptionWScale (TcpHeader &header)
{
  NS_LOG_FUNCTION (this << header);
  NS_ASSERT (header.GetFlags () & TcpHeader::SYN);

  Ptr<TcpOptionWinScale> option = CreateObject<TcpOptionWinScale> ();

  // In naming, we do the contrary of RFC 1323. The sended scaling factor
  // is Snd.Wind.Scale (and not Rcv.Wind.Scale)

  m_rcvWindShift = CalculateWScale ();
  option->SetScale (m_rcvWindShift);

  header.AppendOption (option);

  NS_LOG_INFO (m_node->GetId () << " Send a scaling factor of " <<
               static_cast<int> (m_rcvWindShift));
}

uint32_t
TcpSocketBase::ProcessOptionSack (const Ptr<const TcpOption> option)
{
  NS_LOG_FUNCTION (this << option);

  Ptr<const TcpOptionSack> s = DynamicCast<const TcpOptionSack> (option);
  return m_txBuffer->Update (s->GetSackList (), MakeCallback (&TcpRateOps::SkbDelivered, m_rateOps));
}

void
TcpSocketBase::ProcessOptionSackPermitted (const Ptr<const TcpOption> option)
{
  NS_LOG_FUNCTION (this << option);

  Ptr<const TcpOptionSackPermitted> s = DynamicCast<const TcpOptionSackPermitted> (option);

  NS_ASSERT (m_sackEnabled == true);
  NS_LOG_INFO (m_node->GetId () << " Received a SACK_PERMITTED option " << s);
}

void
TcpSocketBase::AddOptionSackPermitted (TcpHeader &header)
{
  NS_LOG_FUNCTION (this << header);
  NS_ASSERT (header.GetFlags () & TcpHeader::SYN);

  Ptr<TcpOptionSackPermitted> option = CreateObject<TcpOptionSackPermitted> ();
  header.AppendOption (option);
  NS_LOG_INFO (m_node->GetId () << " Add option SACK-PERMITTED");
}

void
TcpSocketBase::AddOptionSack (TcpHeader& header)
{
  NS_LOG_FUNCTION (this << header);

  // Calculate the number of SACK blocks allowed in this packet
  uint8_t optionLenAvail = header.GetMaxOptionLength () - header.GetOptionLength ();
  uint8_t allowedSackBlocks = (optionLenAvail - 2) / 8;

  TcpOptionSack::SackList sackList = m_tcb->m_rxBuffer->GetSackList ();
  if (allowedSackBlocks == 0 || sackList.empty ())
    {
      NS_LOG_LOGIC ("No space available or sack list empty, not adding sack blocks");
      return;
    }

  // Append the allowed number of SACK blocks
  Ptr<TcpOptionSack> option = CreateObject<TcpOptionSack> ();
  TcpOptionSack::SackList::iterator i;
  for (i = sackList.begin (); allowedSackBlocks > 0 && i != sackList.end (); ++i)
    {
      option->AddSackBlock (*i);
      allowedSackBlocks--;
    }

  header.AppendOption (option);
  NS_LOG_INFO (m_node->GetId () << " Add option SACK " << *option);
}

void
TcpSocketBase::ProcessOptionTimestamp (const Ptr<const TcpOption> option,
                                       const SequenceNumber32 &seq)
{
  NS_LOG_FUNCTION (this << option);

  Ptr<const TcpOptionTS> ts = DynamicCast<const TcpOptionTS> (option);

  // This is valid only when no overflow occurs. It happens
  // when a connection last longer than 50 days.
  if (m_tcb->m_rcvTimestampValue > ts->GetTimestamp ())
    {
      // Do not save a smaller timestamp (probably there is reordering)
      return;
    }

  m_tcb->m_rcvTimestampValue = ts->GetTimestamp ();
  m_tcb->m_rcvTimestampEchoReply = ts->GetEcho ();

  if (seq == m_tcb->m_rxBuffer->NextRxSequence () && seq <= m_highTxAck)
    {
      m_timestampToEcho = ts->GetTimestamp ();
    }

  NS_LOG_INFO (m_node->GetId () << " Got timestamp=" <<
               m_timestampToEcho << " and Echo="     << ts->GetEcho ());
}

void
TcpSocketBase::AddOptionTimestamp (TcpHeader& header)
{
  NS_LOG_FUNCTION (this << header);

  Ptr<TcpOptionTS> option = CreateObject<TcpOptionTS> ();

  option->SetTimestamp (TcpOptionTS::NowToTsValue ());
  option->SetEcho (m_timestampToEcho);

  header.AppendOption (option);
  NS_LOG_INFO (m_node->GetId () << " Add option TS, ts=" <<
               option->GetTimestamp () << " echo=" << m_timestampToEcho);
}

void TcpSocketBase::UpdateWindowSize (const TcpHeader &header)
{
  NS_LOG_FUNCTION (this << header);
  //  If the connection is not established, the window size is always
  //  updated
  uint32_t receivedWindow = header.GetWindowSize ();
  receivedWindow <<= m_sndWindShift;
  NS_LOG_INFO ("Received (scaled) window is " << receivedWindow << " bytes");
  if (m_state < ESTABLISHED)
    {
      m_rWnd = receivedWindow;
      NS_LOG_LOGIC ("State less than ESTABLISHED; updating rWnd to " << m_rWnd);
      return;
    }

  // Test for conditions that allow updating of the window
  // 1) segment contains new data (advancing the right edge of the receive
  // buffer),
  // 2) segment does not contain new data but the segment acks new data
  // (highest sequence number acked advances), or
  // 3) the advertised window is larger than the current send window
  bool update = false;
  if (header.GetAckNumber () == m_highRxAckMark && receivedWindow > m_rWnd)
    {
      // right edge of the send window is increased (window update)
      update = true;
    }
  if (header.GetAckNumber () > m_highRxAckMark)
    {
      m_highRxAckMark = header.GetAckNumber ();
      update = true;
    }
  if (header.GetSequenceNumber () > m_highRxMark)
    {
      m_highRxMark = header.GetSequenceNumber ();
      update = true;
    }
  if (update == true)
    {
      m_rWnd = receivedWindow;
      NS_LOG_LOGIC ("updating rWnd to " << m_rWnd);
    }
}

void
TcpSocketBase::SetMinRto (Time minRto)
{
  NS_LOG_FUNCTION (this << minRto);
  m_minRto = minRto;
}

Time
TcpSocketBase::GetMinRto (void) const
{
  return m_minRto;
}

void
TcpSocketBase::SetClockGranularity (Time clockGranularity)
{
  NS_LOG_FUNCTION (this << clockGranularity);
  m_clockGranularity = clockGranularity;
}

Time
TcpSocketBase::GetClockGranularity (void) const
{
  return m_clockGranularity;
}

Ptr<TcpTxBuffer>
TcpSocketBase::GetTxBuffer (void) const
{
  return m_txBuffer;
}

Ptr<TcpRxBuffer>
TcpSocketBase::GetRxBuffer (void) const
{
  return m_tcb->m_rxBuffer;
}

void
TcpSocketBase::SetRetxThresh (uint32_t retxThresh)
{
  m_retxThresh = retxThresh;
  m_txBuffer->SetDupAckThresh (retxThresh);
}

void
TcpSocketBase::UpdatePacingRateTrace (DataRate oldValue, DataRate newValue)
{
  m_pacingRateTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateCwnd (uint32_t oldValue, uint32_t newValue)
{
  m_cWndTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateCwndInfl (uint32_t oldValue, uint32_t newValue)
{
  m_cWndInflTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateSsThresh (uint32_t oldValue, uint32_t newValue)
{
  m_ssThTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateCongState (TcpSocketState::TcpCongState_t oldValue,
                                TcpSocketState::TcpCongState_t newValue)
{
  m_congStateTrace (oldValue, newValue);
}

 void
TcpSocketBase::UpdateEcnState (TcpSocketState::EcnState_t oldValue,
                                TcpSocketState::EcnState_t newValue)
{
  m_ecnStateTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateNextTxSequence (SequenceNumber32 oldValue,
                                     SequenceNumber32 newValue)

{
  m_nextTxSequenceTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateHighTxMark (SequenceNumber32 oldValue, SequenceNumber32 newValue)
{
  m_highTxMarkTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateBytesInFlight (uint32_t oldValue, uint32_t newValue)
{
  m_bytesInFlightTrace (oldValue, newValue);
}

void
TcpSocketBase::UpdateRtt (Time oldValue, Time newValue)
{
  m_lastRttTrace (oldValue, newValue);
}

void
TcpSocketBase::SetCongestionControlAlgorithm (Ptr<TcpCongestionOps> algo)
{
  NS_LOG_FUNCTION (this << algo);
  m_congestionControl = algo;
  m_congestionControl->Init (m_tcb);
}

void
TcpSocketBase::SetRecoveryAlgorithm (Ptr<TcpRecoveryOps> recovery)
{
  NS_LOG_FUNCTION (this << recovery);
  m_recoveryOps = recovery;
}

Ptr<TcpSocketBase>
TcpSocketBase::Fork (void)
{
  return CopyObject<TcpSocketBase> (this);
}

uint32_t
TcpSocketBase::SafeSubtraction (uint32_t a, uint32_t b)
{
  if (a > b)
    {
      return a-b;
    }

  return 0;
}

void
TcpSocketBase::NotifyPacingPerformed (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_INFO ("Performing Pacing");
  SendPendingData (m_connected);
}

bool
TcpSocketBase::IsPacingEnabled (void) const
{
  if (!m_tcb->m_pacing)
    {
      return false;
    }
  else
    {
      if (m_tcb->m_paceInitialWindow)
        {
          return true;
        }
      SequenceNumber32 highTxMark = m_tcb->m_highTxMark; // cast traced value
      if (highTxMark.GetValue () > (GetInitialCwnd () * m_tcb->m_segmentSize))
        {
          return true;
        }
    }
  return false;
}

void
TcpSocketBase::UpdatePacingRate (void)
{
  NS_LOG_FUNCTION (this << m_tcb);

  // According to Linux, set base pacing rate to (cwnd * mss) / srtt
  //
  // In (early) slow start, multiply base by the slow start factor.
  // In late slow start and congestion avoidance, multiply base by
  // the congestion avoidance factor.
  // Comment from Linux code regarding early/late slow start:
  // Normal Slow Start condition is (tp->snd_cwnd < tp->snd_ssthresh)
  // If snd_cwnd >= (tp->snd_ssthresh / 2), we are approaching
  // end of slow start and should slow down.

  // Similar to Linux, do not update pacing rate here if the
  // congestion control implements TcpCongestionOps::CongControl ()
  if (m_congestionControl->HasCongControl () || !m_tcb->m_pacing) return;

  double factor;
  if (m_tcb->m_cWnd < m_tcb->m_ssThresh/2)
    {
      NS_LOG_DEBUG ("Pacing according to slow start factor; " << m_tcb->m_cWnd << " " << m_tcb->m_ssThresh);
      factor = static_cast<double> (m_tcb->m_pacingSsRatio)/100;
    }
  else
    {
      NS_LOG_DEBUG ("Pacing according to congestion avoidance factor; " << m_tcb->m_cWnd << " " << m_tcb->m_ssThresh);
      factor = static_cast<double> (m_tcb->m_pacingCaRatio)/100;
    }
  Time lastRtt = m_tcb->m_lastRtt.Get (); // Get underlying Time value
  NS_LOG_DEBUG ("Last RTT is " << lastRtt.GetSeconds ());
  
  // Multiply by 8 to convert from bytes per second to bits per second
  DataRate pacingRate ((std::max (m_tcb->m_cWnd, m_tcb->m_bytesInFlight) * 8 * factor) / lastRtt.GetSeconds ());
  if (pacingRate < m_tcb->m_maxPacingRate)
    {
      NS_LOG_DEBUG ("Pacing rate updated to: " << pacingRate);
      m_tcb->m_pacingRate = pacingRate;
    }
  else
    {
      NS_LOG_DEBUG ("Pacing capped by max pacing rate: " << m_tcb->m_maxPacingRate);
      m_tcb->m_pacingRate = m_tcb->m_maxPacingRate;
    }
}

void
TcpSocketBase::SetPacingStatus (bool pacing)
{
  NS_LOG_FUNCTION (this << pacing);
  m_tcb->m_pacing = pacing;
}

void
TcpSocketBase::SetPaceInitialWindow (bool paceWindow)
{
  NS_LOG_FUNCTION (this << paceWindow);
  m_tcb->m_paceInitialWindow = paceWindow;
}

void
TcpSocketBase::SetUseEcn (TcpSocketState::UseEcn_t useEcn)
{
  NS_LOG_FUNCTION (this << useEcn);
  m_tcb->m_useEcn = useEcn;
}

uint32_t
TcpSocketBase::GetRWnd (void) const
{
  return m_rWnd.Get ();
}

SequenceNumber32
TcpSocketBase::GetHighRxAck (void) const
{
  return m_highRxAckMark.Get ();
}


//RttHistory methods
RttHistory::RttHistory (SequenceNumber32 s, uint32_t c, Time t)
  : seq (s),
    count (c),
    time (t),
    retx (false)
{
}

RttHistory::RttHistory (const RttHistory& h)
  : seq (h.seq),
    count (h.count),
    time (h.time),
    retx (h.retx)
{
}

} // namespace ns3
