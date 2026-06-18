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
#define __STDC_LIMIT_MACROS
#include "ns3/test.h"
#include "ns3/node-container.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/log.h"
#include "ns3/queue.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/ipv4-end-point.h"
#include "ns3/ipv6-end-point.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/tcp-rx-buffer.h"
#include "ns3/rtt-estimator.h"

#include "tcp-general-test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpGeneralTest");

TcpGeneralTest::TcpGeneralTest (const std::string &desc)
  : TestCase (desc),
    m_congControlTypeId (TcpNewReno::GetTypeId ()),
    m_recoveryTypeId (TcpClassicRecovery::GetTypeId ()),
    m_remoteAddr (Ipv4Address::GetAny (), 4477)
{
  NS_LOG_FUNCTION (this << desc);
}

TcpGeneralTest::~TcpGeneralTest ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
TcpGeneralTest::ReceivePacket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  Ptr<Packet> packet;
  Address from;

  while ((packet = socket->RecvFrom (from)))
    {
      if (packet->GetSize () == 0)
        { //EOF
          break;
        }
    }
}

void
TcpGeneralTest::SendPacket (Ptr<Socket> socket, uint32_t pktSize,
                            uint32_t pktCount, Time pktInterval )
{
  NS_LOG_FUNCTION (this << " " << pktSize << " " << pktCount << " " <<
                   pktInterval.GetSeconds ());
  if (pktCount > 0)
    {
      socket->Send (Create<Packet> (pktSize));
      Simulator::Schedule (pktInterval, &TcpGeneralTest::SendPacket, this,
                           socket, pktSize, pktCount - 1, pktInterval);
    }
  else
    {
      socket->Close ();
    }
}

void
TcpGeneralTest::DoTeardown (void)
{
  FinalChecks ();

  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}

void
TcpGeneralTest::ConfigureEnvironment ()
{
  NS_LOG_FUNCTION (this);

  SetCongestionControl (m_congControlTypeId);
  SetRecoveryAlgorithm (m_recoveryTypeId);
  SetPropagationDelay (MilliSeconds (500));
  SetTransmitStart (Seconds (10));
  SetAppPktSize (500);
  SetAppPktCount (10);
  SetAppPktInterval (MilliSeconds (1));
  SetMTU (1500);
}

void
TcpGeneralTest::ConfigureProperties ()
{
  NS_LOG_FUNCTION (this);
  SetInitialCwnd (SENDER, 1);
  SetInitialSsThresh (SENDER, UINT32_MAX);
  SetSegmentSize (SENDER, 500);
  SetSegmentSize (RECEIVER, 500);
}

void
TcpGeneralTest::DoRun (void)
{
  ConfigureEnvironment ();

  NS_LOG_INFO ("Create nodes.");
  NodeContainer nodes;
  nodes.Create (2);

  InternetStackHelper internet;
  internet.Install (nodes);

  Packet::EnablePrinting ();

  Ptr<SimpleChannel> channel = CreateChannel ();

  SimpleNetDeviceHelper helperChannel;
  helperChannel.SetNetDevicePointToPointMode (true);

  NetDeviceContainer net = helperChannel.Install (nodes, channel);

  Ptr<ErrorModel> receiverEM = CreateReceiverErrorModel ();
  Ptr<ErrorModel> senderEM   = CreateSenderErrorModel ();

  Ptr<SimpleNetDevice> senderDev = DynamicCast<SimpleNetDevice> (net.Get (0));
  Ptr<SimpleNetDevice> receiverDev = DynamicCast<SimpleNetDevice> (net.Get (1));

  senderDev->SetMtu (m_mtu);
  senderDev->GetQueue ()->TraceConnect ("Drop", "SENDER",
                                        MakeCallback (&TcpGeneralTest::QueueDropCb, this));
  senderDev->TraceConnect ("PhyRxDrop", "sender",
                           MakeCallback (&TcpGeneralTest::PhyDropCb, this));

  receiverDev->SetMtu (m_mtu);
  receiverDev->GetQueue ()->TraceConnect ("Drop", "RECEIVER",
                                          MakeCallback (&TcpGeneralTest::QueueDropCb, this));
  receiverDev->TraceConnect ("PhyRxDrop", "RECEIVER",
                             MakeCallback (&TcpGeneralTest::PhyDropCb, this));

  senderDev->SetReceiveErrorModel (senderEM);
  receiverDev->SetReceiveErrorModel (receiverEM);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = ipv4.Assign (net);
  Ipv4Address serverAddress = i.GetAddress (1);
  //Ipv4Address clientAddress = i.GetAddress (0);

  NS_LOG_INFO ("Create sockets.");
  //Receiver socket on n1
  m_receiverSocket = CreateReceiverSocket (nodes.Get (1));

  m_receiverSocket->SetRecvCallback (MakeCallback (&TcpGeneralTest::ReceivePacket, this));
  m_receiverSocket->SetAcceptCallback (
    MakeNullCallback<bool, Ptr<Socket>, const Address &> (),
    MakeCallback (&TcpGeneralTest::HandleAccept, this));
  m_receiverSocket->SetCloseCallbacks (MakeCallback (&TcpGeneralTest::NormalCloseCb, this),
                                       MakeCallback (&TcpGeneralTest::ErrorCloseCb, this));
  m_receiverSocket->SetRcvAckCb (MakeCallback (&TcpGeneralTest::RcvAckCb, this));
  m_receiverSocket->SetProcessedAckCb (MakeCallback (&TcpGeneralTest::ProcessedAckCb, this));
  m_receiverSocket->SetAfterRetransmitCb (MakeCallback (&TcpGeneralTest::AfterRetransmitCb, this));
  m_receiverSocket->SetBeforeRetransmitCb (MakeCallback (&TcpGeneralTest::BeforeRetransmitCb, this));
  m_receiverSocket->SetForkCb (MakeCallback (&TcpGeneralTest::ForkCb, this));
  m_receiverSocket->SetUpdateRttHistoryCb (MakeCallback (&TcpGeneralTest::UpdateRttHistoryCb, this));
  m_receiverSocket->TraceConnectWithoutContext ("Tx",
                                                MakeCallback (&TcpGeneralTest::TxPacketCb, this));
  m_receiverSocket->TraceConnectWithoutContext ("Rx",
                                                MakeCallback (&TcpGeneralTest::RxPacketCb, this));

  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 4477);
  m_receiverSocket->Bind (local);

  m_senderSocket = CreateSenderSocket (nodes.Get (0));
  m_senderSocket->SetCloseCallbacks (MakeCallback (&TcpGeneralTest::NormalCloseCb, this),
                                     MakeCallback (&TcpGeneralTest::ErrorCloseCb, this));
  m_senderSocket->SetRcvAckCb (MakeCallback (&TcpGeneralTest::RcvAckCb, this));
  m_senderSocket->SetProcessedAckCb (MakeCallback (&TcpGeneralTest::ProcessedAckCb, this));
  m_senderSocket->SetAfterRetransmitCb (MakeCallback (&TcpGeneralTest::AfterRetransmitCb, this));
  m_senderSocket->SetBeforeRetransmitCb (MakeCallback (&TcpGeneralTest::BeforeRetransmitCb, this));
  m_senderSocket->SetDataSentCallback (MakeCallback (&TcpGeneralTest::DataSentCb, this));
  m_senderSocket->SetUpdateRttHistoryCb (MakeCallback (&TcpGeneralTest::UpdateRttHistoryCb, this));
  m_senderSocket->TraceConnectWithoutContext ("CongestionWindow",
                                              MakeCallback (&TcpGeneralTest::CWndTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("CongestionWindowInflated",
                                              MakeCallback (&TcpGeneralTest::CWndInflTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("SlowStartThreshold",
                                              MakeCallback (&TcpGeneralTest::SsThreshTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("CongState",
                                              MakeCallback (&TcpGeneralTest::CongStateTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("Tx",
                                              MakeCallback (&TcpGeneralTest::TxPacketCb, this));
  m_senderSocket->TraceConnectWithoutContext ("Rx",
                                              MakeCallback (&TcpGeneralTest::RxPacketCb, this));
  m_senderSocket->TraceConnectWithoutContext ("RTT",
                                              MakeCallback (&TcpGeneralTest::RttTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("BytesInFlight",
                                              MakeCallback (&TcpGeneralTest::BytesInFlightTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("RTO",
                                              MakeCallback (&TcpGeneralTest::RtoTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("NextTxSequence",
                                              MakeCallback (&TcpGeneralTest::NextTxSeqTrace, this));
  m_senderSocket->TraceConnectWithoutContext ("HighestSequence",
                                              MakeCallback (&TcpGeneralTest::HighestTxSeqTrace, this));
  m_senderSocket->m_rateOps->TraceConnectWithoutContext ("TcpRateUpdated",
                                              MakeCallback (&TcpGeneralTest::RateUpdatedTrace, this));
  m_senderSocket->m_rateOps->TraceConnectWithoutContext ("TcpRateSampleUpdated",
                                              MakeCallback (&TcpGeneralTest::RateSampleUpdatedTrace, this));


  m_remoteAddr = InetSocketAddress (serverAddress, 4477);

  ConfigureProperties ();

  m_receiverSocket->Listen ();
  m_receiverSocket->ShutdownSend ();

  Simulator::Schedule (Seconds (0.0),
                       &TcpGeneralTest::DoConnect, this);
  Simulator::ScheduleWithContext (nodes.Get (0)->GetId (),
                                  m_startTime, &TcpGeneralTest::SendPacket, this,
                                  m_senderSocket, m_pktSize, m_pktCount, m_interPacketInterval);

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
}

void
TcpGeneralTest::DoConnect ()
{
  NS_LOG_INFO (this);
  m_senderSocket->Connect (m_remoteAddr);
}

void
TcpGeneralTest::HandleAccept (Ptr<Socket> socket, const Address& from)
{
  (void) from;
  socket->SetRecvCallback (MakeCallback (&TcpGeneralTest::ReceivePacket, this));
  socket->SetCloseCallbacks (MakeCallback (&TcpGeneralTest::NormalCloseCb, this),
                             MakeCallback (&TcpGeneralTest::ErrorCloseCb, this));

}

Ptr<SimpleChannel>
TcpGeneralTest::CreateChannel ()
{
  Ptr<SimpleChannel> ch = CreateObject <SimpleChannel> ();

  ch->SetAttribute ("Delay", TimeValue (m_propagationDelay));

  return ch;
}

Ptr<TcpSocketMsgBase>
TcpGeneralTest::CreateSocket (Ptr<Node> node, TypeId socketType,
                              TypeId congControl)
{
  return CreateSocket (node, socketType, congControl, m_recoveryTypeId);
}

Ptr<TcpSocketMsgBase>
TcpGeneralTest::CreateSocket (Ptr<Node> node, TypeId socketType,
                              TypeId congControl, TypeId recoveryAlgorithm)
{
  ObjectFactory rttFactory;
  ObjectFactory congestionAlgorithmFactory;
  ObjectFactory recoveryAlgorithmFactory;
  ObjectFactory socketFactory;

  rttFactory.SetTypeId (RttMeanDeviation::GetTypeId ());
  congestionAlgorithmFactory.SetTypeId (congControl);
  recoveryAlgorithmFactory.SetTypeId (recoveryAlgorithm);
  socketFactory.SetTypeId (socketType);

  Ptr<RttEstimator> rtt = rttFactory.Create<RttEstimator> ();
  Ptr<TcpSocketMsgBase> socket = DynamicCast<TcpSocketMsgBase> (socketFactory.Create ());
  Ptr<TcpCongestionOps> algo = congestionAlgorithmFactory.Create<TcpCongestionOps> ();
  Ptr<TcpRecoveryOps> recovery = recoveryAlgorithmFactory.Create<TcpRecoveryOps> ();

  socket->SetNode (node);
  socket->SetTcp (node->GetObject<TcpL4Protocol> ());
  socket->SetRtt (rtt);
  socket->SetCongestionControlAlgorithm (algo);
  socket->SetRecoveryAlgorithm (recovery);
  return socket;
}

Ptr<ErrorModel>
TcpGeneralTest::CreateSenderErrorModel ()
{
  return nullptr;
}

Ptr<ErrorModel>
TcpGeneralTest::CreateReceiverErrorModel ()
{
  return nullptr;
}

Ptr<TcpSocketMsgBase>
TcpGeneralTest::CreateSenderSocket (Ptr<Node> node)
{
  return CreateSocket (node, TcpSocketMsgBase::GetTypeId (), m_congControlTypeId, m_recoveryTypeId);
}

Ptr<TcpSocketMsgBase>
TcpGeneralTest::CreateReceiverSocket (Ptr<Node> node)
{
  return CreateSocket (node, TcpSocketMsgBase::GetTypeId (), m_congControlTypeId, m_recoveryTypeId);
}

void
TcpGeneralTest::QueueDropCb ( std::string context, Ptr<const Packet> p)
{
  if (context.compare ("SENDER") == 0)
    {
      QueueDrop (SENDER);
    }
  else if (context.compare ("RECEIVER") == 0)
    {
      QueueDrop (RECEIVER);
    }
  else
    {
      NS_FATAL_ERROR ("Packet dropped in a queue, but queue not recognized");
    }
}

void
TcpGeneralTest::PhyDropCb (std::string context, [[maybe_unused]] Ptr<const Packet> p)
{
  if (context.compare ("SENDER") == 0)
    {
      PhyDrop (SENDER);
    }
  else if (context.compare ("RECEIVER") == 0)
    {
      PhyDrop (RECEIVER);
    }
  else
    {
      NS_FATAL_ERROR ("Packet dropped in a queue, but queue not recognized");
    }
}

void
TcpGeneralTest::NormalCloseCb (Ptr<Socket> socket)
{
  if (socket->GetNode () ==  m_receiverSocket->GetNode ())
    {
      NormalClose (RECEIVER);
    }
  else if (socket->GetNode () == m_senderSocket->GetNode ())
    {
      NormalClose (SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Closed socket, but not recognized");
    }
}

void
TcpGeneralTest::UpdateRttHistoryCb (Ptr<const TcpSocketBase> tcp,
                                    const SequenceNumber32 & seq, uint32_t sz,
                                    bool isRetransmission)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      UpdatedRttHistory (seq, sz, isRetransmission, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      UpdatedRttHistory (seq, sz, isRetransmission, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Closed socket, but not recognized");
    }
}

void
TcpGeneralTest::AfterRetransmitCb (const Ptr<const TcpSocketState> tcb,
                                   const Ptr<const TcpSocketBase> tcp)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      AfterRTOExpired (tcb, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      AfterRTOExpired (tcb, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Closed socket, but not recognized");
    }
}

void
TcpGeneralTest::BeforeRetransmitCb (const Ptr<const TcpSocketState> tcb,
                                    const Ptr<const TcpSocketBase> tcp)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      BeforeRTOExpired (tcb, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      BeforeRTOExpired (tcb, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Closed socket, but not recognized");
    }
}

void
TcpGeneralTest::DataSentCb (Ptr<Socket> socket, uint32_t size)
{
  if (socket->GetNode () == m_receiverSocket->GetNode ())
    {
      DataSent (size, RECEIVER);
    }
  else if (socket->GetNode () == m_senderSocket->GetNode ())
    {
      DataSent (size, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Closed socket, but not recognized");
    }
}

void
TcpGeneralTest::ErrorCloseCb (Ptr<Socket> socket)
{
  if (socket->GetNode () == m_receiverSocket->GetNode ())
    {
      ErrorClose (RECEIVER);
    }
  else if (socket->GetNode () == m_senderSocket->GetNode ())
    {
      ErrorClose (SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Closed socket, but not recognized");
    }
}

void
TcpGeneralTest::Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);
}

void
TcpGeneralTest::Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);
}

void
TcpGeneralTest::RcvAckCb (const Ptr<const Packet> p, const TcpHeader& h,
                          const Ptr<const TcpSocketBase> tcp)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      RcvAck (tcp->m_tcb, h, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      RcvAck (tcp->m_tcb, h, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Received ACK but socket not recognized");
    }
}

void
TcpGeneralTest::TxPacketCb (const Ptr<const Packet> p,
                            const TcpHeader &h, const Ptr<const TcpSocketBase> tcp)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      Tx (p, h, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      Tx (p, h, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Received ACK but socket not recognized");
    }
}

void
TcpGeneralTest::RxPacketCb (const Ptr<const Packet> p, const TcpHeader &h,
                            const Ptr<const TcpSocketBase> tcp)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      Rx (p, h, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      Rx (p, h, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Received ACK but socket not recognized");
    }
}

void
TcpGeneralTest::ProcessedAckCb (Ptr<const Packet> p, const TcpHeader& h,
                                Ptr<const TcpSocketBase> tcp)
{
  if (tcp->GetNode () == m_receiverSocket->GetNode ())
    {
      ProcessedAck (tcp->m_tcb, h, RECEIVER);
    }
  else if (tcp->GetNode () == m_senderSocket->GetNode ())
    {
      ProcessedAck (tcp->m_tcb, h, SENDER);
    }
  else
    {
      NS_FATAL_ERROR ("Received ACK but socket not recognized");
    }
}

void
TcpGeneralTest::ForkCb (Ptr<TcpSocketMsgBase> tcp)
{
  NS_LOG_FUNCTION (this << tcp);

  m_receiverSocket = tcp;
}

uint32_t
TcpGeneralTest::GetReTxThreshold (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_retxThresh;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_retxThresh;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

uint32_t
TcpGeneralTest::GetDupAckCount (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_dupAckCount;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_dupAckCount;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

uint32_t
TcpGeneralTest::GetDelAckCount (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_delAckMaxCount;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_delAckMaxCount;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Time
TcpGeneralTest::GetDelAckTimeout (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->GetDelAckTimeout ();
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->GetDelAckTimeout ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

uint32_t
TcpGeneralTest::GetSegSize (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->GetSegSize ();
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->GetSegSize ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

SequenceNumber32
TcpGeneralTest::GetHighestTxMark (SocketWho who)
{
  return GetTcb (who)->m_highTxMark;
}

uint32_t
TcpGeneralTest::GetInitialCwnd (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->GetInitialCwnd ();
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->GetInitialCwnd ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

uint32_t
TcpGeneralTest::GetInitialSsThresh (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->GetInitialSSThresh ();
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->GetInitialSSThresh ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Time
TcpGeneralTest::GetRto (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_rto.Get ();
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_rto.Get ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Time
TcpGeneralTest::GetMinRto (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_minRto;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_minRto;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Time
TcpGeneralTest::GetConnTimeout (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_cnTimeout;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_cnTimeout;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Ptr<RttEstimator>
TcpGeneralTest::GetRttEstimator (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_rtt;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_rtt;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Time
TcpGeneralTest::GetClockGranularity (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_clockGranularity;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_clockGranularity;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

TcpSocket::TcpStates_t
TcpGeneralTest::GetTcpState (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_state.Get ();
    }
  else if (who == RECEIVER)
    {

      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_state.Get ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

uint32_t
TcpGeneralTest::GetRWnd (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_rWnd.Get ();
    }
  else if (who == RECEIVER)
    {

      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_rWnd.Get ();
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

EventId
TcpGeneralTest::GetPersistentEvent (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_persistEvent;
    }
  else if (who == RECEIVER)
    {

      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_persistEvent;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Time
TcpGeneralTest::GetPersistentTimeout (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_persistTimeout;
    }
  else if (who == RECEIVER)
    {

      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_persistTimeout;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Ptr<TcpSocketState>
TcpGeneralTest::GetTcb (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_tcb;
    }
  else if (who == RECEIVER)
    {

      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_tcb;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

Ptr<TcpRxBuffer>
TcpGeneralTest::GetRxBuffer (SocketWho who)
{
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_tcb->m_rxBuffer;
    }
  else if (who == RECEIVER)
    {

      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_tcb->m_rxBuffer;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

 Ptr<TcpTxBuffer>
 TcpGeneralTest::GetTxBuffer (SocketWho who)
 {
  if (who == SENDER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_senderSocket)->m_txBuffer;
    }
  else if (who == RECEIVER)
    {
      return DynamicCast<TcpSocketMsgBase> (m_receiverSocket)->m_txBuffer;
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
 }


void
TcpGeneralTest::SetRcvBufSize (SocketWho who, uint32_t size)
{
  if (who == SENDER)
    {
      m_senderSocket->SetRcvBufSize (size);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetRcvBufSize (size);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

void
TcpGeneralTest::SetSegmentSize (SocketWho who, uint32_t segmentSize)
{
  if (who == SENDER)
    {
      m_senderSocket->SetSegSize (segmentSize);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetSegSize (segmentSize);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

void
TcpGeneralTest::SetInitialCwnd (SocketWho who, uint32_t initialCwnd)
{
  if (who == SENDER)
    {
      m_senderSocket->SetInitialCwnd (initialCwnd);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetInitialCwnd (initialCwnd);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}
void
TcpGeneralTest::SetDelAckMaxCount (SocketWho who, uint32_t count)
{
  if (who == SENDER)
    {
      m_senderSocket->SetDelAckMaxCount (count);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetDelAckMaxCount (count);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}
void
TcpGeneralTest::SetUseEcn (SocketWho who, TcpSocketState::UseEcn_t useEcn)
{
  if (who == SENDER)
    {
      m_senderSocket->SetUseEcn (useEcn);
    }
   else if (who == RECEIVER)
    {
      m_receiverSocket->SetUseEcn (useEcn);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

void
TcpGeneralTest::SetPacingStatus (SocketWho who, bool pacing)
{
  if (who == SENDER)
    {
      m_senderSocket->SetPacingStatus (pacing);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetPacingStatus (pacing);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

void
TcpGeneralTest::SetPaceInitialWindow (SocketWho who, bool paceWindow)
{
  if (who == SENDER)
    {
      m_senderSocket->SetPaceInitialWindow (paceWindow);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetPaceInitialWindow (paceWindow);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

void
TcpGeneralTest::SetInitialSsThresh (SocketWho who, uint32_t initialSsThresh)
{
  if (who == SENDER)
    {
      m_senderSocket->SetInitialSSThresh (initialSsThresh);
    }
  else if (who == RECEIVER)
    {
      m_receiverSocket->SetInitialSSThresh (initialSsThresh);
    }
  else
    {
      NS_FATAL_ERROR ("Not defined");
    }
}

NS_OBJECT_ENSURE_REGISTERED (TcpSocketMsgBase);

TypeId
TcpSocketMsgBase::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpSocketMsgBase")
    .SetParent<TcpSocketBase> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpSocketMsgBase> ()
  ;
  return tid;
}

Ptr<TcpSocketBase>
TcpSocketMsgBase::Fork (void)
{
  return CopyObject<TcpSocketMsgBase> (this);
}

void
TcpSocketMsgBase::SetRcvAckCb (AckManagementCb cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_rcvAckCb = cb;
}

void
TcpSocketMsgBase::SetProcessedAckCb (AckManagementCb cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_processedAckCb = cb;
}

void
TcpSocketMsgBase::SetAfterRetransmitCb (RetrCb cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_afterRetrCallback = cb;
}

void
TcpSocketMsgBase::SetBeforeRetransmitCb (RetrCb cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_beforeRetrCallback = cb;
}

void
TcpSocketMsgBase::ReceivedAck (Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
  NS_ASSERT (!(m_rcvAckCb.IsNull () || m_processedAckCb.IsNull ()));
  m_rcvAckCb (packet, tcpHeader, this);

  TcpSocketBase::ReceivedAck (packet, tcpHeader);

  m_processedAckCb (packet, tcpHeader, this);
}

void
TcpSocketMsgBase::ReTxTimeout ()
{
  m_beforeRetrCallback (m_tcb, this);
  TcpSocketBase::ReTxTimeout ();
  m_afterRetrCallback (m_tcb, this);
}

void
TcpSocketMsgBase::SetForkCb (Callback<void, Ptr<TcpSocketMsgBase> > cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_forkCb = cb;
}

void
TcpSocketMsgBase::SetUpdateRttHistoryCb (UpdateRttCallback cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_updateRttCb = cb;
}

void
TcpSocketMsgBase::UpdateRttHistory (const SequenceNumber32 &seq, uint32_t sz,
                                    bool isRetransmission)
{
  TcpSocketBase::UpdateRttHistory (seq, sz, isRetransmission);
  if (!m_updateRttCb.IsNull ())
    {
      m_updateRttCb (this, seq, sz, isRetransmission);
    }
}

void
TcpSocketMsgBase::CompleteFork (Ptr<Packet> p, const TcpHeader &tcpHeader,
                                const Address &fromAddress, const Address &toAddress)
{
  TcpSocketBase::CompleteFork (p, tcpHeader, fromAddress, toAddress);

  if (!m_forkCb.IsNull ())
    {
      m_forkCb (this);
    }
}

NS_OBJECT_ENSURE_REGISTERED (TcpSocketSmallAcks);

TypeId
TcpSocketSmallAcks::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpSocketSmallAcks")
    .SetParent<TcpSocketMsgBase> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpSocketSmallAcks> ()
  ;
  return tid;
}

/*
 * Send empty packet, copied/pasted from TcpSocketBase
 *
 * The rationale for copying/pasting is that we need to edit a little the
 * code inside. Since there isn't a well-defined division of duties,
 * we are forced to do this.
 */
void
TcpSocketSmallAcks::SendEmptyPacket (uint8_t flags)
{
  Ptr<Packet> p = Create<Packet> ();
  TcpHeader header;
  SequenceNumber32 s = m_tcb->m_nextTxSequence;

  /*
   * Add tags for each socket option.
   * Note that currently the socket adds both IPv4 tag and IPv6 tag
   * if both options are set. Once the packet got to layer three, only
   * the corresponding tags will be read.
   */
  if (GetIpTos ())
    {
      SocketIpTosTag ipTosTag;
      ipTosTag.SetTos (GetIpTos ());
      p->AddPacketTag (ipTosTag);
    }

  if (IsManualIpv6Tclass ())
    {
      SocketIpv6TclassTag ipTclassTag;
      ipTclassTag.SetTclass (GetIpv6Tclass ());
      p->AddPacketTag (ipTclassTag);
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

  if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
      NS_LOG_WARN ("Failed to send empty packet due to null endpoint");
      return;
    }
  if (flags & TcpHeader::FIN)
    {
      flags |= TcpHeader::ACK;
    }
  else if (m_state == FIN_WAIT_1 || m_state == LAST_ACK || m_state == CLOSING)
    {
      ++s;
    }

  bool hasSyn = flags & TcpHeader::SYN;
  bool hasFin = flags & TcpHeader::FIN;
  bool isAck = flags == TcpHeader::ACK;

  header.SetFlags (flags);
  header.SetSequenceNumber (s);

  // Actual division in small acks.
  if (hasSyn || hasFin)
    {
      header.SetAckNumber (m_tcb->m_rxBuffer->NextRxSequence ());
    }
  else
    {
      SequenceNumber32 ackSeq;

      ackSeq = m_lastAckedSeq + m_bytesToAck;

      if (m_bytesLeftToBeAcked == 0 && m_tcb->m_rxBuffer->NextRxSequence () > m_lastAckedSeq)
        {
          m_bytesLeftToBeAcked = m_tcb->m_rxBuffer->NextRxSequence ().GetValue () - m_lastAckedSeq.GetValue ();
          m_bytesLeftToBeAcked -= m_bytesToAck;
          NS_LOG_DEBUG ("Setting m_bytesLeftToBeAcked to " << m_bytesLeftToBeAcked);
        }
      else if (m_bytesLeftToBeAcked > 0 && m_tcb->m_rxBuffer->NextRxSequence () > m_lastAckedSeq)
        {
          m_bytesLeftToBeAcked -= m_bytesToAck;
          NS_LOG_DEBUG ("Decrementing m_bytesLeftToBeAcked to " << m_bytesLeftToBeAcked);
        }

      NS_LOG_LOGIC ("Acking up to " << ackSeq << " remaining bytes: " << m_bytesLeftToBeAcked);

      header.SetAckNumber (ackSeq);
      m_lastAckedSeq = ackSeq;
    }

  // end of division in small acks

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
  header.SetWindowSize (AdvertisedWindowSize ());

  // RFC 6298, clause 2.4
  m_rto = Max (m_rtt->GetEstimate () + Max (m_clockGranularity, m_rtt->GetVariation () * 4), m_minRto);

  if (hasSyn)
    {
      if (m_synCount == 0)
        { // No more connection retries, give up
          NS_LOG_LOGIC ("Connection failed.");
          m_rtt->Reset (); //According to recommendation -> RFC 6298
          CloseAndNotify ();
          return;
        }
      else
        { // Exponential backoff of connection time out
          int backoffCount = 0x1 << (m_synRetries - m_synCount);
          m_rto = m_cnTimeout * backoffCount;
          m_synCount--;
        }
    }
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

  m_txTrace (p, header, this);

  if (flags & TcpHeader::ACK)
    { // If sending an ACK, cancel the delay ACK as well
      m_delAckEvent.Cancel ();
      m_delAckCount = 0;
    }
  if (m_retxEvent.IsExpired () && (hasSyn || hasFin) && !isAck )
    { // Retransmit SYN / SYN+ACK / FIN / FIN+ACK to guard against lost
      NS_LOG_LOGIC ("Schedule retransmission timeout at time "
                    << Simulator::Now ().GetSeconds () << " to expire at time "
                    << (Simulator::Now () + m_rto.Get ()).GetSeconds ());
      m_retxEvent = Simulator::Schedule (m_rto, &TcpSocketSmallAcks::SendEmptyPacket, this, flags);
    }

  // send another ACK if bytes remain
  if (m_bytesLeftToBeAcked > m_bytesToAck && m_tcb->m_rxBuffer->NextRxSequence () > m_lastAckedSeq && !hasFin)
    {
      NS_LOG_DEBUG ("Recursing to call SendEmptyPacket() again with m_bytesLeftToBeAcked = " << m_bytesLeftToBeAcked);
      SendEmptyPacket (flags);
    }
}

Ptr<TcpSocketBase>
TcpSocketSmallAcks::Fork (void)
{
  return CopyObject<TcpSocketSmallAcks> (this);
}

