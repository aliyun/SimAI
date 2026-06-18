/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
 *               2010      NICTA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Quincy Tse <quincy.tse@nicta.com.au>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/string.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/adhoc-wifi-mac.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/interference-helper.h"
#include "ns3/yans-error-rate-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/test.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/config.h"
#include "ns3/error-model.h"
#include "ns3/socket.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/wifi-spectrum-signal-parameters.h"
#include "ns3/yans-wifi-phy.h"
#include "ns3/mgt-headers.h"
#include "ns3/ht-configuration.h"
#include "ns3/wifi-ppdu.h"
#include "ns3/wifi-psdu.h"
#include "ns3/vht-phy.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/frame-exchange-manager.h"
#include "ns3/wifi-default-protection-manager.h"
#include "ns3/wifi-default-ack-manager.h"

using namespace ns3;

//Helper function to assign streams to random variables, to control
//randomness in the tests
static void
AssignWifiRandomStreams (Ptr<WifiMac> mac, int64_t stream)
{
  int64_t currentStream = stream;
  PointerValue ptr;
  if (!mac->GetQosSupported ())
    {
      mac->GetAttribute ("Txop", ptr);
      Ptr<Txop> txop = ptr.Get<Txop> ();
      currentStream += txop->AssignStreams (currentStream);
    }
  else
    {
      mac->GetAttribute ("VO_Txop", ptr);
      Ptr<QosTxop> vo_txop = ptr.Get<QosTxop> ();
      currentStream += vo_txop->AssignStreams (currentStream);

      mac->GetAttribute ("VI_Txop", ptr);
      Ptr<QosTxop> vi_txop = ptr.Get<QosTxop> ();
      currentStream += vi_txop->AssignStreams (currentStream);

      mac->GetAttribute ("BE_Txop", ptr);
      Ptr<QosTxop> be_txop = ptr.Get<QosTxop> ();
      currentStream += be_txop->AssignStreams (currentStream);

      mac->GetAttribute ("BK_Txop", ptr);
      Ptr<QosTxop> bk_txop = ptr.Get<QosTxop> ();
      bk_txop->AssignStreams (currentStream);
    }
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Test
 */
class WifiTest : public TestCase
{
public:
  WifiTest ();

  void DoRun (void) override;


private:
  /// Run one function
  void RunOne (void);
  /**
   * Create one function
   * \param pos the position
   * \param channel the wifi channel
   */
  void CreateOne (Vector pos, Ptr<YansWifiChannel> channel);
  /**
   * Send one packet function
   * \param dev the device
   */
  void SendOnePacket (Ptr<WifiNetDevice> dev);

  ObjectFactory m_manager; ///< manager
  ObjectFactory m_mac; ///< MAC
  ObjectFactory m_propDelay; ///< propagation delay
};

WifiTest::WifiTest ()
  : TestCase ("Wifi")
{
}

void
WifiTest::SendOnePacket (Ptr<WifiNetDevice> dev)
{
  Ptr<Packet> p = Create<Packet> ();
  dev->Send (p, dev->GetBroadcast (), 1);
}

void
WifiTest::CreateOne (Vector pos, Ptr<YansWifiChannel> channel)
{
  Ptr<Node> node = CreateObject<Node> ();
  Ptr<WifiNetDevice> dev = CreateObject<WifiNetDevice> ();

  Ptr<WifiMac> mac = m_mac.Create<WifiMac> ();
  mac->SetDevice (dev);
  mac->SetAddress (Mac48Address::Allocate ());
  mac->ConfigureStandard (WIFI_STANDARD_80211a);
  Ptr<FrameExchangeManager> fem = mac->GetFrameExchangeManager ();
  Ptr<WifiProtectionManager> protectionManager = CreateObject<WifiDefaultProtectionManager> ();
  protectionManager->SetWifiMac (mac);
  fem->SetProtectionManager (protectionManager);
  Ptr<WifiAckManager> ackManager = CreateObject<WifiDefaultAckManager> ();
  ackManager->SetWifiMac (mac);
  fem->SetAckManager (ackManager);

  Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel> ();
  Ptr<YansWifiPhy> phy = CreateObject<YansWifiPhy> ();
  Ptr<InterferenceHelper> interferenceHelper = CreateObject<InterferenceHelper> ();
  phy->SetInterferenceHelper (interferenceHelper);
  Ptr<ErrorRateModel> error = CreateObject<YansErrorRateModel> ();
  phy->SetErrorRateModel (error);
  phy->SetChannel (channel);
  phy->SetDevice (dev);
  phy->ConfigureStandard (WIFI_STANDARD_80211a);
  Ptr<WifiRemoteStationManager> manager = m_manager.Create<WifiRemoteStationManager> ();

  mobility->SetPosition (pos);
  node->AggregateObject (mobility);
  dev->SetMac (mac);
  dev->SetPhy (phy);
  dev->SetRemoteStationManager (manager);
  node->AddDevice (dev);

  Simulator::Schedule (Seconds (1.0), &WifiTest::SendOnePacket, this, dev);
}

void
WifiTest::RunOne (void)
{
  Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel> ();
  Ptr<PropagationDelayModel> propDelay = m_propDelay.Create<PropagationDelayModel> ();
  Ptr<PropagationLossModel> propLoss = CreateObject<RandomPropagationLossModel> ();
  channel->SetPropagationDelayModel (propDelay);
  channel->SetPropagationLossModel (propLoss);

  CreateOne (Vector (0.0, 0.0, 0.0), channel);
  CreateOne (Vector (5.0, 0.0, 0.0), channel);
  CreateOne (Vector (5.0, 0.0, 0.0), channel);

  Simulator::Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
}

void
WifiTest::DoRun (void)
{
  m_mac.SetTypeId ("ns3::AdhocWifiMac");
  m_propDelay.SetTypeId ("ns3::ConstantSpeedPropagationDelayModel");

  m_manager.SetTypeId ("ns3::ArfWifiManager");
  RunOne ();
  m_manager.SetTypeId ("ns3::AarfWifiManager");
  RunOne ();
  m_manager.SetTypeId ("ns3::ConstantRateWifiManager");
  RunOne ();
  m_manager.SetTypeId ("ns3::OnoeWifiManager");
  RunOne ();
  m_manager.SetTypeId ("ns3::AmrrWifiManager");
  RunOne ();
  m_manager.SetTypeId ("ns3::IdealWifiManager");
  RunOne ();

  m_mac.SetTypeId ("ns3::AdhocWifiMac");
  RunOne ();
  m_mac.SetTypeId ("ns3::ApWifiMac");
  RunOne ();
  m_mac.SetTypeId ("ns3::StaWifiMac");
  RunOne ();


  m_propDelay.SetTypeId ("ns3::RandomPropagationDelayModel");
  m_mac.SetTypeId ("ns3::AdhocWifiMac");
  RunOne ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Qos Utils Is Old Packet Test
 */
class QosUtilsIsOldPacketTest : public TestCase
{
public:
  QosUtilsIsOldPacketTest () : TestCase ("QosUtilsIsOldPacket")
  {
  }
  void DoRun (void) override
  {
    //startingSeq=0, seqNum=2047
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (0, 2047), false, "2047 is new in comparison to 0");
    //startingSeq=0, seqNum=2048
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (0, 2048), true, "2048 is old in comparison to 0");
    //startingSeq=2048, seqNum=0
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (2048, 0), true, "0 is old in comparison to 2048");
    //startingSeq=4095, seqNum=0
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (4095, 0), false, "0 is new in comparison to 4095");
    //startingSeq=0, seqNum=4095
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (0, 4095), true, "4095 is old in comparison to 0");
    //startingSeq=4095 seqNum=2047
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (4095, 2047), true, "2047 is old in comparison to 4095");
    //startingSeq=2048 seqNum=4095
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (2048, 4095), false, "4095 is new in comparison to 2048");
    //startingSeq=2049 seqNum=0
    NS_TEST_EXPECT_MSG_EQ (QosUtilsIsOldPacket (2049, 0), false, "0 is new in comparison to 2049");
  }
};


/**
 * See \bugid{991}
 */
class InterferenceHelperSequenceTest : public TestCase
{
public:
  InterferenceHelperSequenceTest ();

  void DoRun (void) override;


private:
  /**
   * Create one function
   * \param pos the position
   * \param channel the wifi channel
   * \returns the node
   */
  Ptr<Node> CreateOne (Vector pos, Ptr<YansWifiChannel> channel);
  /**
   * Send one packet function
   * \param dev the device
   */
  void SendOnePacket (Ptr<WifiNetDevice> dev);
  /**
   * Switch channel function
   * \param dev the device
   */
  void SwitchCh (Ptr<WifiNetDevice> dev);

  ObjectFactory m_manager; ///< manager
  ObjectFactory m_mac; ///< MAC
  ObjectFactory m_propDelay; ///< propagation delay
};

InterferenceHelperSequenceTest::InterferenceHelperSequenceTest ()
  : TestCase ("InterferenceHelperSequence")
{
}

void
InterferenceHelperSequenceTest::SendOnePacket (Ptr<WifiNetDevice> dev)
{
  Ptr<Packet> p = Create<Packet> (1000);
  dev->Send (p, dev->GetBroadcast (), 1);
}

void
InterferenceHelperSequenceTest::SwitchCh (Ptr<WifiNetDevice> dev)
{
  Ptr<WifiPhy> p = dev->GetPhy ();
  p->SetOperatingChannel (WifiPhy::ChannelTuple {40, 0, (int)(WIFI_PHY_BAND_5GHZ), 0});
}

Ptr<Node>
InterferenceHelperSequenceTest::CreateOne (Vector pos, Ptr<YansWifiChannel> channel)
{
  Ptr<Node> node = CreateObject<Node> ();
  Ptr<WifiNetDevice> dev = CreateObject<WifiNetDevice> ();

  Ptr<WifiMac> mac = m_mac.Create<WifiMac> ();
  mac->SetDevice (dev);
  mac->SetAddress (Mac48Address::Allocate ());
  mac->ConfigureStandard (WIFI_STANDARD_80211a);
  Ptr<FrameExchangeManager> fem = mac->GetFrameExchangeManager ();
  Ptr<WifiProtectionManager> protectionManager = CreateObject<WifiDefaultProtectionManager> ();
  protectionManager->SetWifiMac (mac);
  fem->SetProtectionManager (protectionManager);
  Ptr<WifiAckManager> ackManager = CreateObject<WifiDefaultAckManager> ();
  ackManager->SetWifiMac (mac);
  fem->SetAckManager (ackManager);

  Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel> ();
  Ptr<YansWifiPhy> phy = CreateObject<YansWifiPhy> ();
  Ptr<InterferenceHelper> interferenceHelper = CreateObject<InterferenceHelper> ();
  phy->SetInterferenceHelper (interferenceHelper);
  Ptr<ErrorRateModel> error = CreateObject<YansErrorRateModel> ();
  phy->SetErrorRateModel (error);
  phy->SetChannel (channel);
  phy->SetDevice (dev);
  phy->SetMobility (mobility);
  phy->ConfigureStandard (WIFI_STANDARD_80211a);
  Ptr<WifiRemoteStationManager> manager = m_manager.Create<WifiRemoteStationManager> ();

  mobility->SetPosition (pos);
  node->AggregateObject (mobility);
  dev->SetMac (mac);
  dev->SetPhy (phy);
  dev->SetRemoteStationManager (manager);
  node->AddDevice (dev);

  return node;
}

void
InterferenceHelperSequenceTest::DoRun (void)
{
  m_mac.SetTypeId ("ns3::AdhocWifiMac");
  m_propDelay.SetTypeId ("ns3::ConstantSpeedPropagationDelayModel");
  m_manager.SetTypeId ("ns3::ConstantRateWifiManager");

  Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel> ();
  Ptr<PropagationDelayModel> propDelay = m_propDelay.Create<PropagationDelayModel> ();
  Ptr<MatrixPropagationLossModel> propLoss = CreateObject<MatrixPropagationLossModel> ();
  channel->SetPropagationDelayModel (propDelay);
  channel->SetPropagationLossModel (propLoss);

  Ptr<Node> rxOnly = CreateOne (Vector (0.0, 0.0, 0.0), channel);
  Ptr<Node> senderA = CreateOne (Vector (5.0, 0.0, 0.0), channel);
  Ptr<Node> senderB = CreateOne (Vector (-5.0, 0.0, 0.0), channel);

  propLoss->SetLoss (senderB->GetObject<MobilityModel> (), rxOnly->GetObject<MobilityModel> (), 0, true);
  propLoss->SetDefaultLoss (999);

  Simulator::Schedule (Seconds (1.0),
                       &InterferenceHelperSequenceTest::SendOnePacket, this,
                       DynamicCast<WifiNetDevice> (senderB->GetDevice (0)));

  Simulator::Schedule (Seconds (1.0000001),
                       &InterferenceHelperSequenceTest::SwitchCh, this,
                       DynamicCast<WifiNetDevice> (rxOnly->GetDevice (0)));

  Simulator::Schedule (Seconds (5.0),
                       &InterferenceHelperSequenceTest::SendOnePacket, this,
                       DynamicCast<WifiNetDevice> (senderA->GetDevice (0)));

  Simulator::Schedule (Seconds (7.0),
                       &InterferenceHelperSequenceTest::SendOnePacket, this,
                       DynamicCast<WifiNetDevice> (senderB->GetDevice (0)));

  Simulator::Stop (Seconds (100.0));
  Simulator::Run ();

  Simulator::Destroy ();
}

//-----------------------------------------------------------------------------
/**
 * Make sure that when multiple broadcast packets are queued on the same
 * device in a short succession, that:
 * 1) no backoff occurs if the frame arrives and the idle time >= DIFS or AIFSn
 *    (this is 'DCF immediate access', Figure 9-3 of IEEE 802.11-2012)
 * 2) a backoff occurs for the second frame that arrives (this is clearly
 *    stated in Sec. 9.3.4.2 of IEEE 802.11-2012, (basic access, which
 *    applies to group-addressed frames) where it states
 *    "If, under these conditions, the medium is determined by the CS
 *    mechanism to be busy when a STA desires to initiate the initial frame
 *    of a frame exchange sequence (described in Annex G), exclusive of the
 *    CF period, the random backoff procedure described in 9.3.4.3
 *    shall be followed."
 *    and from 9.3.4.3
 *    "The result of this procedure is that transmitted
 *    frames from a STA are always separated by at least one backoff interval."
 *
 * The observed behavior is that the first frame will be sent immediately,
 * and the frames are spaced by (backoff + DIFS) time intervals
 * (where backoff is a random number of slot sizes up to maximum CW)
 *
 * The following test case should _not_ generate virtual collision for
 * the second frame.  The seed and run numbers were pick such that the
 * second frame gets backoff = 1 slot.
 *
 *                      frame 1, frame 2
 *                      arrive                DIFS = 2 x slot + SIFS
 *                      |                          = 2 x 9us + 16us for 11a
 *                      |                    <----------->
 *                      V                                 <-backoff->
 * time  |--------------|-------------------|-------------|----------->
 *       0              1s                  1.001408s     1.001442s  |1.001451s
 *                      ^                   ^                        ^
 *                      start TX            finish TX                start TX
 *                      frame 1             frame 1                  frame 2
 *                      ^
 *                      frame 2
 *                      backoff = 1 slot
 *
 * The buggy behavior observed in prior versions was shown by picking
 * RngSeedManager::SetRun (17);
 * which generated a 0 slot backoff for frame 2.  Then, frame 2
 * experiences a virtual collision and re-selects the backoff again.
 * As a result, the _actual_ backoff experience by frame 2 is less likely
 * to be 0 since that would require two successions of 0 backoff (one that
 * generates the virtual collision and one after the virtual collision).
 *
 * See \bugid{555} for past behavior.
 */

class DcfImmediateAccessBroadcastTestCase : public TestCase
{
public:
  DcfImmediateAccessBroadcastTestCase ();

  void DoRun (void) override;


private:
  /**
   * Send one packet function
   * \param dev the device
   */
  void SendOnePacket (Ptr<WifiNetDevice> dev);

  ObjectFactory m_manager; ///< manager
  ObjectFactory m_mac; ///< MAC
  ObjectFactory m_propDelay; ///< propagation delay

  Time m_firstTransmissionTime; ///< first transmission time
  Time m_secondTransmissionTime; ///< second transmission time
  unsigned int m_numSentPackets; ///< number of sent packets

  /**
   * Notify Phy transmit begin
   * \param p the packet
   * \param txPowerW the tx power
   */
  void NotifyPhyTxBegin (Ptr<const Packet> p, double txPowerW);
};

DcfImmediateAccessBroadcastTestCase::DcfImmediateAccessBroadcastTestCase ()
  : TestCase ("Test case for DCF immediate access with broadcast frames")
{
}

void
DcfImmediateAccessBroadcastTestCase::NotifyPhyTxBegin (Ptr<const Packet> p, double txPowerW)
{
  if (m_numSentPackets == 0)
    {
      m_numSentPackets++;
      m_firstTransmissionTime = Simulator::Now ();
    }
  else if (m_numSentPackets == 1)
    {
      m_secondTransmissionTime = Simulator::Now ();
    }
}

void
DcfImmediateAccessBroadcastTestCase::SendOnePacket (Ptr<WifiNetDevice> dev)
{
  Ptr<Packet> p = Create<Packet> (1000);
  dev->Send (p, dev->GetBroadcast (), 1);
}

void
DcfImmediateAccessBroadcastTestCase::DoRun (void)
{
  m_mac.SetTypeId ("ns3::AdhocWifiMac");
  m_propDelay.SetTypeId ("ns3::ConstantSpeedPropagationDelayModel");
  m_manager.SetTypeId ("ns3::ConstantRateWifiManager");

  //Assign a seed and run number, and later fix the assignment of streams to
  //WiFi random variables, so that the first backoff used is one slot
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (40);  // a value of 17 will result in zero slots

  Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel> ();
  Ptr<PropagationDelayModel> propDelay = m_propDelay.Create<PropagationDelayModel> ();
  Ptr<PropagationLossModel> propLoss = CreateObject<RandomPropagationLossModel> ();
  channel->SetPropagationDelayModel (propDelay);
  channel->SetPropagationLossModel (propLoss);

  Ptr<Node> txNode = CreateObject<Node> ();
  Ptr<WifiNetDevice> txDev = CreateObject<WifiNetDevice> ();
  Ptr<WifiMac> txMac = m_mac.Create<WifiMac> ();
  txMac->SetDevice (txDev);
  txMac->ConfigureStandard (WIFI_STANDARD_80211a);
  Ptr<FrameExchangeManager> fem = txMac->GetFrameExchangeManager ();
  Ptr<WifiProtectionManager> protectionManager = CreateObject<WifiDefaultProtectionManager> ();
  protectionManager->SetWifiMac (txMac);
  fem->SetProtectionManager (protectionManager);
  Ptr<WifiAckManager> ackManager = CreateObject<WifiDefaultAckManager> ();
  ackManager->SetWifiMac (txMac);
  fem->SetAckManager (ackManager);


  //Fix the stream assignment to the Dcf Txop objects (backoffs)
  //The below stream assignment will result in the Txop object
  //using a backoff value of zero for this test when the
  //Txop::EndTxNoAck() calls to StartBackoffNow()
  AssignWifiRandomStreams (txMac, 23);

  Ptr<ConstantPositionMobilityModel> txMobility = CreateObject<ConstantPositionMobilityModel> ();
  Ptr<YansWifiPhy> txPhy = CreateObject<YansWifiPhy> ();
  Ptr<InterferenceHelper> txInterferenceHelper = CreateObject<InterferenceHelper> ();
  txPhy->SetInterferenceHelper (txInterferenceHelper);
  Ptr<ErrorRateModel> txError = CreateObject<YansErrorRateModel> ();
  txPhy->SetErrorRateModel (txError);
  txPhy->SetChannel (channel);
  txPhy->SetDevice (txDev);
  txPhy->SetMobility (txMobility);
  txPhy->ConfigureStandard (WIFI_STANDARD_80211a);

  txPhy->TraceConnectWithoutContext ("PhyTxBegin", MakeCallback (&DcfImmediateAccessBroadcastTestCase::NotifyPhyTxBegin, this));

  txMobility->SetPosition (Vector (0.0, 0.0, 0.0));
  txNode->AggregateObject (txMobility);
  txMac->SetAddress (Mac48Address::Allocate ());
  txDev->SetMac (txMac);
  txDev->SetPhy (txPhy);
  txDev->SetRemoteStationManager (m_manager.Create<WifiRemoteStationManager> ());
  txNode->AddDevice (txDev);

  m_firstTransmissionTime = Seconds (0.0);
  m_secondTransmissionTime = Seconds (0.0);
  m_numSentPackets = 0;

  Simulator::Schedule (Seconds (1.0), &DcfImmediateAccessBroadcastTestCase::SendOnePacket, this, txDev);
  Simulator::Schedule (Seconds (1.0) + MicroSeconds (1), &DcfImmediateAccessBroadcastTestCase::SendOnePacket, this, txDev);

  Simulator::Stop (Seconds (2.0));
  Simulator::Run ();
  Simulator::Destroy ();

  // First packet is transmitted a DIFS after the packet is queued. A DIFS
  // is 2 slots (2 * 9 = 18 us) plus a SIFS (16 us), i.e., 34 us
  Time expectedFirstTransmissionTime = Seconds (1.0) + MicroSeconds (34);

  //First packet has 1408 us of transmit time.   Slot time is 9 us.
  //Backoff is 1 slots.  SIFS is 16 us.  DIFS is 2 slots = 18 us.
  //Should send next packet at 1408 us + (1 * 9 us) + 16 us + (2 * 9) us
  //1451 us after the first one.
  uint32_t expectedWait1 = 1408 + (1 * 9) + 16 + (2 * 9);
  Time expectedSecondTransmissionTime = expectedFirstTransmissionTime + MicroSeconds (expectedWait1);
  NS_TEST_ASSERT_MSG_EQ (m_firstTransmissionTime, expectedFirstTransmissionTime, "The first transmission time not correct!");

  NS_TEST_ASSERT_MSG_EQ (m_secondTransmissionTime, expectedSecondTransmissionTime, "The second transmission time not correct!");
}

//-----------------------------------------------------------------------------
/**
 * Make sure that when changing the fragmentation threshold during the simulation,
 * the TCP transmission does not unexpectedly stop.
 *
 * The scenario considers a TCP transmission between a 802.11b station and a 802.11b
 * access point. After the simulation has begun, the fragmentation threshold is set at
 * a value lower than the packet size. It then checks whether the TCP transmission
 * continues after the fragmentation threshold modification.
 *
 * See \bugid{730}
 */

class Bug730TestCase : public TestCase
{
public:
  Bug730TestCase ();
  virtual ~Bug730TestCase ();

  void DoRun (void) override;


private:
  uint32_t m_received; ///< received

  /**
   * Receive function
   * \param context the context
   * \param p the packet
   * \param adr the address
   */
  void Receive (std::string context, Ptr<const Packet> p, const Address &adr);

};

Bug730TestCase::Bug730TestCase ()
  : TestCase ("Test case for Bug 730"),
    m_received (0)
{
}

Bug730TestCase::~Bug730TestCase ()
{
}

void
Bug730TestCase::Receive (std::string context, Ptr<const Packet> p, const Address &adr)
{
  if ((p->GetSize () == 1460) && (Simulator::Now () > Seconds (20)))
    {
      m_received++;
    }
}


void
Bug730TestCase::DoRun (void)
{
  m_received = 0;

  NodeContainer wifiStaNode;
  wifiStaNode.Create (1);

  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211b);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("DsssRate1Mbps"),
                                "ControlMode", StringValue ("DsssRate1Mbps"));

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevices;
  staDevices = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid),
               "BeaconGeneration", BooleanValue (true));

  NetDeviceContainer apDevices;
  apDevices = wifi.Install (phy, mac, wifiApNode);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  Ptr<WifiNetDevice> ap_device = DynamicCast<WifiNetDevice> (apDevices.Get (0));
  Ptr<WifiNetDevice> sta_device = DynamicCast<WifiNetDevice> (staDevices.Get (0));

  PacketSocketAddress socket;
  socket.SetSingleDevice (sta_device->GetIfIndex ());
  socket.SetPhysicalAddress (ap_device->GetAddress ());
  socket.SetProtocol (1);

  // give packet socket powers to nodes.
  PacketSocketHelper packetSocket;
  packetSocket.Install (wifiStaNode);
  packetSocket.Install (wifiApNode);

  Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
  client->SetAttribute ("PacketSize", UintegerValue (1460));
  client->SetRemote (socket);
  wifiStaNode.Get (0)->AddApplication (client);
  client->SetStartTime (Seconds (1));
  client->SetStopTime (Seconds (51.0));

  Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
  server->SetLocal (socket);
  wifiApNode.Get (0)->AddApplication (server);
  server->SetStartTime (Seconds (0.0));
  server->SetStopTime (Seconds (52.0));

  Config::Connect ("/NodeList/*/ApplicationList/0/$ns3::PacketSocketServer/Rx", MakeCallback (&Bug730TestCase::Receive, this));

  Simulator::Schedule (Seconds (10.0), Config::Set, "/NodeList/0/DeviceList/0/RemoteStationManager/FragmentationThreshold", StringValue ("800"));

  Simulator::Stop (Seconds (55));
  Simulator::Run ();

  Simulator::Destroy ();

  bool result = (m_received > 0);
  NS_TEST_ASSERT_MSG_EQ (result, true, "packet reception unexpectedly stopped after adapting fragmentation threshold!");
}

//-----------------------------------------------------------------------------
/**
 * Make sure that fragmentation works with QoS stations.
 *
 * The scenario considers a TCP transmission between an 802.11n station and an 802.11n
 * access point.
 */

class QosFragmentationTestCase : public TestCase
{
public:
  QosFragmentationTestCase ();
  virtual ~QosFragmentationTestCase ();

  void DoRun (void) override;


private:
  uint32_t m_received; ///< received packets
  uint32_t m_fragments; ///< transmitted fragments

  /**
   * Receive function
   * \param context the context
   * \param p the packet
   * \param adr the address
   */
  void Receive (std::string context, Ptr<const Packet> p, const Address &adr);

  /**
   * Callback invoked when PHY transmits a packet
   * \param context the context
   * \param p the packet
   * \param power the tx power
   */
  void Transmit (std::string context, Ptr<const Packet> p, double power);
};

QosFragmentationTestCase::QosFragmentationTestCase ()
  : TestCase ("Test case for fragmentation with QoS stations"),
    m_received (0),
    m_fragments (0)
{
}

QosFragmentationTestCase::~QosFragmentationTestCase ()
{
}

void
QosFragmentationTestCase::Receive (std::string context, Ptr<const Packet> p, const Address &adr)
{
  if (p->GetSize () == 1400)
    {
      m_received++;
    }
}

void
QosFragmentationTestCase::Transmit (std::string context, Ptr<const Packet> p, double power)
{
  WifiMacHeader hdr;
  p->PeekHeader (hdr);
  if (hdr.IsQosData ())
    {
      NS_TEST_EXPECT_MSG_LT_OR_EQ (p->GetSize (), 400, "Unexpected fragment size");
      m_fragments++;
    }
}

void
QosFragmentationTestCase::DoRun (void)
{
  NodeContainer wifiStaNode;
  wifiStaNode.Create (1);

  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("HtMcs7"));

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevices;
  staDevices = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid),
               "BeaconGeneration", BooleanValue (true));

  NetDeviceContainer apDevices;
  apDevices = wifi.Install (phy, mac, wifiApNode);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  Ptr<WifiNetDevice> ap_device = DynamicCast<WifiNetDevice> (apDevices.Get (0));
  Ptr<WifiNetDevice> sta_device = DynamicCast<WifiNetDevice> (staDevices.Get (0));

  // set the TXOP limit on BE AC
  PointerValue ptr;
  sta_device->GetMac ()->GetAttribute ("BE_Txop", ptr);
  ptr.Get<QosTxop> ()->SetTxopLimit (MicroSeconds (3008));

  PacketSocketAddress socket;
  socket.SetSingleDevice (sta_device->GetIfIndex ());
  socket.SetPhysicalAddress (ap_device->GetAddress ());
  socket.SetProtocol (1);

  // give packet socket powers to nodes.
  PacketSocketHelper packetSocket;
  packetSocket.Install (wifiStaNode);
  packetSocket.Install (wifiApNode);

  Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
  client->SetAttribute ("PacketSize", UintegerValue (1400));
  client->SetAttribute ("MaxPackets", UintegerValue (1));
  client->SetRemote (socket);
  wifiStaNode.Get (0)->AddApplication (client);
  client->SetStartTime (Seconds (1));
  client->SetStopTime (Seconds (3.0));

  Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
  server->SetLocal (socket);
  wifiApNode.Get (0)->AddApplication (server);
  server->SetStartTime (Seconds (0.0));
  server->SetStopTime (Seconds (4.0));

  Config::Connect ("/NodeList/*/ApplicationList/0/$ns3::PacketSocketServer/Rx", MakeCallback (&QosFragmentationTestCase::Receive, this));

  Config::Set ("/NodeList/0/DeviceList/0/RemoteStationManager/FragmentationThreshold", StringValue ("400"));
  Config::Connect ("/NodeList/0/DeviceList/0/Phy/PhyTxBegin", MakeCallback (&QosFragmentationTestCase::Transmit, this));

  Simulator::Stop (Seconds (5));
  Simulator::Run ();

  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_received, 1, "Unexpected number of received packets");
  NS_TEST_ASSERT_MSG_EQ (m_fragments, 4, "Unexpected number of transmitted fragments");
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Set Channel Frequency Test
 */
class SetChannelFrequencyTest : public TestCase
{
public:
  SetChannelFrequencyTest ();

  void DoRun (void) override;


private:
  /**
   * Get yans wifi phy function
   * \param nc the device collection
   * \returns the wifi phy
   */
  Ptr<YansWifiPhy> GetYansWifiPhyPtr (const NetDeviceContainer &nc) const;

};

SetChannelFrequencyTest::SetChannelFrequencyTest ()
  : TestCase ("Test case for setting WifiPhy channel and frequency")
{
}

Ptr<YansWifiPhy>
SetChannelFrequencyTest::GetYansWifiPhyPtr (const NetDeviceContainer &nc) const
{
  Ptr<WifiNetDevice> wnd = nc.Get (0)->GetObject<WifiNetDevice> ();
  Ptr<WifiPhy> wp = wnd->GetPhy ();
  return wp->GetObject<YansWifiPhy> ();
}

void
SetChannelFrequencyTest::DoRun ()
{
  NodeContainer wifiStaNode;
  wifiStaNode.Create (1);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  // Configure and declare other generic components of this example
  Ssid ssid;
  ssid = Ssid ("wifi-phy-configuration");
  WifiMacHelper macSta;
  macSta.SetType ("ns3::StaWifiMac",
                  "Ssid", SsidValue (ssid),
                  "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevice;
  Ptr<YansWifiPhy> phySta;

  // Cases taken from src/wifi/examples/wifi-phy-configuration.cc example
  {
    // case 0:
    // Default configuration, without WifiHelper::SetStandard or WifiHelper
    phySta = CreateObject<YansWifiPhy> ();
    // The default results in an invalid configuration
    NS_TEST_ASSERT_MSG_EQ (phySta->GetOperatingChannel ().IsSet (), false, "default configuration");
  }
  {
    // case 1:
    WifiHelper wifi;
    wifi.SetStandard (WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager ("ns3::ArfWifiManager");
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    // We expect channel 36, width 20, frequency 5180
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "default configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "default configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "default configuration");
  }
  {
    // case 2:
    WifiHelper wifi;
    wifi.SetStandard (WIFI_STANDARD_80211b);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    // We expect channel 1, width 22, frequency 2412
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 1, "802.11b configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 22, "802.11b configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 2412, "802.11b configuration");
  }
  {
    // case 3:
    WifiHelper wifi;
    wifi.SetStandard (WIFI_STANDARD_80211g);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    // We expect channel 1, width 20, frequency 2412
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 1, "802.11g configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11g configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 2412, "802.11g configuration");
  }
  {
    // case 4:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211n);
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "802.11n-5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11n-5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "802.11n-5GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 5:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211n);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 1, "802.11n-2.4GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11n-2.4GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 2412, "802.11n-2.4GHz configuration");
  }
  {
    // case 6:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211ac);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 42, "802.11ac configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 80, "802.11ac configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5210, "802.11ac configuration");
  }
  {
    // case 7:
    // By default, WifiHelper will use WIFI_PHY_STANDARD_80211ax
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_2_4GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 1, "802.11ax-2.4GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11ax-2.4GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 2412, "802.11ax-2.4GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 8:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 42, "802.11ax-5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 80, "802.11ax-5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5210, "802.11ax-5GHz configuration");
  }
  {
    // case 9:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_6GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 7, "802.11ax-6GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 80, "802.11ax-6GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5975, "802.11ax-6GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 10:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211p);
    phy.Set ("ChannelSettings", StringValue ("{0, 10, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 172, "802.11p 10Mhz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 10, "802.11p 10Mhz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5860, "802.11p 10Mhz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 11:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211p);
    phy.Set ("ChannelSettings", StringValue ("{0, 5, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 171, "802.11p 5Mhz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 5, "802.11p 5Mhz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5860, "802.11p 5Mhz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 12:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211n);
    phy.Set ("ChannelSettings", StringValue ("{44, 20, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 44, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5220, "802.11 5GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 13:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    phy.Set ("ChannelSettings", StringValue ("{44, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    // Post-install reconfiguration to channel number 40
    std::ostringstream path;
    path << "/NodeList/*/DeviceList/" << staDevice.Get(0)->GetIfIndex () << "/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/ChannelSettings";
    Config::Set (path.str(), StringValue ("{40, 0, BAND_5GHZ, 0}"));
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 40, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5200, "802.11 5GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 14:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    phy.Set ("ChannelSettings", StringValue ("{44, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    // Post-install reconfiguration to a 40 MHz channel
    std::ostringstream path;
    path << "/NodeList/*/DeviceList/" << staDevice.Get(0)->GetIfIndex () << "/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/ChannelSettings";
    Config::Set (path.str(), StringValue ("{46, 0, BAND_5GHZ, 0}"));
    // Although channel 44 is configured originally for 20 MHz, we
    // allow it to be used for 40 MHz here
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 46, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 40, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5230, "802.11 5GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 15:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211n);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    phySta->SetAttribute ("ChannelSettings", StringValue ("{3, 20, BAND_2_4GHZ, 0}"));
    return;
    // Post-install reconfiguration to a 40 MHz channel
    std::ostringstream path;
    path << "/NodeList/*/DeviceList/" << staDevice.Get(0)->GetIfIndex () << "/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/ChannelSettings";
    Config::Set (path.str(), StringValue ("{4, 40, BAND_2_4GHZ, 0}"));
    // Although channel 44 is configured originally for 20 MHz, we
    // allow it to be used for 40 MHz here
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 4, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 40, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 2427, "802.11 5GHz configuration");
    phy.Set ("ChannelSettings", StringValue ("{0, 0, BAND_UNSPECIFIED, 0}")); // restore default
  }
  {
    // case 16:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    // Test that setting Frequency to a non-standard value will throw an exception
    wifi.SetStandard (WIFI_STANDARD_80211n);
    phy.Set ("ChannelSettings", StringValue ("{44, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    bool exceptionThrown = false;
    try
      {
        phySta->SetAttribute ("ChannelSettings", StringValue ("{45, 0, BAND_5GHZ, 0}"));
      }
    catch (const std::runtime_error&)
      {
        exceptionThrown = true;
      }
    // We expect that an exception is thrown
    NS_TEST_ASSERT_MSG_EQ (exceptionThrown, true, "802.11 5GHz configuration");
  }
  {
    // case 17:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211n);
    phy.Set ("ChannelSettings", StringValue ("{44, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    // Test that setting channel to a standard value will set the
    // frequency correctly
    phySta->SetAttribute ("ChannelSettings", StringValue ("{100, 0, BAND_5GHZ, 0}"));
    // We expect frequency to be 5500 due to channel number being 100
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 100, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5500, "802.11 5GHz configuration");
  }
  {
    // case 18:
    // Set a wrong channel after initialization
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    wifi.SetStandard (WIFI_STANDARD_80211n);
    phy.Set ("ChannelSettings", StringValue ("{44, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    bool exceptionThrown = false;
    try
      {
        phySta->SetOperatingChannel (WifiPhy::ChannelTuple {99, 40, WIFI_PHY_BAND_5GHZ, 0});
      }
    catch (const std::runtime_error&)
      {
        exceptionThrown = true;
      }
    // We expect that an exception is thrown
    NS_TEST_ASSERT_MSG_EQ (exceptionThrown, true, "802.11 5GHz configuration");
  }
  {
    // case 19:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    // Test how channel number behaves when frequency is non-standard
    wifi.SetStandard (WIFI_STANDARD_80211n);
    phy.Set ("ChannelSettings", StringValue ("{44, 0, BAND_5GHZ, 0}"));
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    bool exceptionThrown = false;
    try
      {
        phySta->SetAttribute ("ChannelSettings", StringValue ("{45, 0, BAND_5GHZ, 0}"));
      }
    catch (const std::runtime_error&)
      {
        exceptionThrown = true;
      }
    // We expect that an exception is thrown due to unknown channel number 45
    NS_TEST_ASSERT_MSG_EQ (exceptionThrown, true, "802.11 5GHz configuration");
    phySta->SetAttribute ("ChannelSettings", StringValue ("{36, 0, BAND_5GHZ, 0}"));
    // We expect channel number to be 36 due to known center frequency 5180
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "802.11 5GHz configuration");
    exceptionThrown = false;
    try
      {
        phySta->SetAttribute ("ChannelSettings", StringValue ("{43, 0, BAND_5GHZ, 0}"));
      }
    catch (const std::runtime_error&)
      {
        exceptionThrown = true;
      }
    // We expect that an exception is thrown due to unknown channel number 43
    NS_TEST_ASSERT_MSG_EQ (exceptionThrown, true, "802.11 5GHz configuration");
    phySta->SetAttribute ("ChannelSettings", StringValue ("{36, 0, BAND_5GHZ, 0}"));
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "802.11 5GHz configuration");
  }
  {
    // case 20:
    WifiHelper wifi;
    wifi.SetRemoteStationManager ("ns3::IdealWifiManager");
    phy.Set ("ChannelSettings", StringValue ("{40, 0, BAND_5GHZ, 0}"));
    wifi.SetStandard (WIFI_STANDARD_80211n);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 40, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5200, "802.11 5GHz configuration");
    // Set both channel and frequency to consistent values after initialization
    wifi.SetStandard (WIFI_STANDARD_80211n);
    staDevice = wifi.Install (phy, macSta, wifiStaNode.Get (0));
    phySta = GetYansWifiPhyPtr (staDevice);
    phySta->SetAttribute ("ChannelSettings", StringValue ("{40, 0, BAND_5GHZ, 0}"));
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 40, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5200, "802.11 5GHz configuration");

    phySta->SetAttribute ("ChannelSettings", StringValue ("{36, 0, BAND_5GHZ, 0}"));
    // We expect channel number to be 36
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "802.11 5GHz configuration");
    phySta->SetAttribute ("ChannelSettings", StringValue ("{40, 0, BAND_5GHZ, 0}"));
    // We expect channel number to be 40
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 40, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5200, "802.11 5GHz configuration");
    bool exceptionThrown = false;
    try
      {
        phySta->SetAttribute ("ChannelSettings", StringValue ("{45, 0, BAND_5GHZ, 0}"));
      }
    catch (const std::runtime_error&)
      {
        exceptionThrown = true;
      }
    phySta->SetAttribute ("ChannelSettings", StringValue ("{36, 0, BAND_5GHZ, 0}"));
    // We expect channel number to be 36 and an exception to be thrown
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (exceptionThrown, true, "802.11 5GHz configuration");
    phySta->SetAttribute ("ChannelSettings", StringValue ("{36, 0, BAND_5GHZ, 0}"));
    exceptionThrown = false;
    try
      {
        phySta->SetAttribute ("ChannelSettings", StringValue ("{43, 0, BAND_5GHZ, 0}"));
      }
    catch (const std::runtime_error&)
      {
        exceptionThrown = true;
      }
    // We expect channel number to be 36 and an exception to be thrown
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelNumber (), 36, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetChannelWidth (), 20, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (phySta->GetFrequency (), 5180, "802.11 5GHz configuration");
    NS_TEST_ASSERT_MSG_EQ (exceptionThrown, true, "802.11 5GHz configuration");
  }

  Simulator::Destroy ();
}

//-----------------------------------------------------------------------------
/**
 * Make sure that when virtual collision occurs the wifi remote station manager
 * is triggered and the retry counter is increased.
 *
 * See \bugid{2222}
 */

class Bug2222TestCase : public TestCase
{
public:
  Bug2222TestCase ();
  virtual ~Bug2222TestCase ();

  void DoRun (void) override;


private:
  uint32_t m_countInternalCollisions; ///< count internal collisions

  /**
   * Transmit data failed function
   * \param context the context
   * \param adr the MAC address
   */
  void TxDataFailedTrace (std::string context, Mac48Address adr);
};

Bug2222TestCase::Bug2222TestCase ()
  : TestCase ("Test case for Bug 2222"),
    m_countInternalCollisions (0)
{
}

Bug2222TestCase::~Bug2222TestCase ()
{
}

void
Bug2222TestCase::TxDataFailedTrace (std::string context, Mac48Address adr)
{
  //Indicate the long retry counter has been increased in the wifi remote station manager
  m_countInternalCollisions++;
}

void
Bug2222TestCase::DoRun (void)
{
  m_countInternalCollisions = 0;

  //Generate same backoff for AC_VI and AC_VO
  //The below combination will work
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 100;

  NodeContainer wifiNodes;
  wifiNodes.Create (2);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("OfdmRate54Mbps"),
                                "ControlMode", StringValue ("OfdmRate24Mbps"));
  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");
  mac.SetType ("ns3::AdhocWifiMac",
               "QosSupported", BooleanValue (true));

  NetDeviceContainer wifiDevices;
  wifiDevices = wifi.Install (phy, mac, wifiNodes);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (wifiDevices, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (10.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiNodes);

  Ptr<WifiNetDevice> device1 = DynamicCast<WifiNetDevice> (wifiDevices.Get (0));
  Ptr<WifiNetDevice> device2 = DynamicCast<WifiNetDevice> (wifiDevices.Get (1));

  PacketSocketAddress socket;
  socket.SetSingleDevice (device1->GetIfIndex ());
  socket.SetPhysicalAddress (device2->GetAddress ());
  socket.SetProtocol (1);

  PacketSocketHelper packetSocket;
  packetSocket.Install (wifiNodes);

  Ptr<PacketSocketClient> clientLowPriority = CreateObject<PacketSocketClient> ();
  clientLowPriority->SetAttribute ("PacketSize", UintegerValue (1460));
  clientLowPriority->SetAttribute ("MaxPackets", UintegerValue (1));
  clientLowPriority->SetAttribute ("Priority", UintegerValue (4)); //AC_VI
  clientLowPriority->SetRemote (socket);
  wifiNodes.Get (0)->AddApplication (clientLowPriority);
  clientLowPriority->SetStartTime (Seconds (0.0));
  clientLowPriority->SetStopTime (Seconds (1.0));

  Ptr<PacketSocketClient> clientHighPriority = CreateObject<PacketSocketClient> ();
  clientHighPriority->SetAttribute ("PacketSize", UintegerValue (1460));
  clientHighPriority->SetAttribute ("MaxPackets", UintegerValue (1));
  clientHighPriority->SetAttribute ("Priority", UintegerValue (6)); //AC_VO
  clientHighPriority->SetRemote (socket);
  wifiNodes.Get (0)->AddApplication (clientHighPriority);
  clientHighPriority->SetStartTime (Seconds (0.0));
  clientHighPriority->SetStopTime (Seconds (1.0));

  Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
  server->SetLocal (socket);
  wifiNodes.Get (1)->AddApplication (server);
  server->SetStartTime (Seconds (0.0));
  server->SetStopTime (Seconds (1.0));

  Config::Connect ("/NodeList/*/DeviceList/*/RemoteStationManager/MacTxDataFailed", MakeCallback (&Bug2222TestCase::TxDataFailedTrace, this));

  Simulator::Stop (Seconds (1.0));
  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_countInternalCollisions, 1, "unexpected number of internal collisions!");
}

//-----------------------------------------------------------------------------
/**
 * Make sure that the correct channel width and center frequency have been set
 * for OFDM basic rate transmissions and BSS channel widths larger than 20 MHz.
 *
 * The scenario considers a UDP transmission between a 40 MHz 802.11ac station and a
 * 40 MHz 802.11ac access point. All transmission parameters are checked so as
 * to ensure that only 2 {starting frequency, channelWidth, Number of subbands
 * in SpectrumModel, modulation type} tuples are used.
 *
 * See \bugid{2843}
 */

class Bug2843TestCase : public TestCase
{
public:
  Bug2843TestCase ();
  virtual ~Bug2843TestCase ();
  void DoRun (void) override;

private:
  /**
   * A tuple of {starting frequency, channelWidth, Number of subbands in SpectrumModel, modulation type}
   */
  typedef std::tuple<double, uint16_t, uint32_t, WifiModulationClass> FreqWidthSubbandModulationTuple;
  std::vector<FreqWidthSubbandModulationTuple> m_distinctTuples; ///< vector of distinct {starting frequency, channelWidth, Number of subbands in SpectrumModel, modulation type} tuples

  /**
   * Stores the distinct {starting frequency, channelWidth, Number of subbands in SpectrumModel, modulation type} tuples
   * that have been used during the testcase run.
   * \param context the context
   * \param txParams spectrum signal parameters set by transmitter
   */
  void StoreDistinctTuple (std::string context, Ptr<SpectrumSignalParameters> txParams);
  /**
   * Triggers the arrival of a burst of 1000 Byte-long packets in the source device
   * \param numPackets number of packets in burst
   * \param sourceDevice pointer to the source NetDevice
   * \param destination address of the destination device
   */
  void SendPacketBurst (uint8_t numPackets, Ptr<NetDevice> sourceDevice, Address& destination) const;

  uint16_t m_channelWidth; ///< channel width (in MHz)
};

Bug2843TestCase::Bug2843TestCase ()
  : TestCase ("Test case for Bug 2843"),
    m_channelWidth (20)
{
}

Bug2843TestCase::~Bug2843TestCase ()
{
}

void
Bug2843TestCase::StoreDistinctTuple (std::string context,  Ptr<SpectrumSignalParameters> txParams)
{
  // Extract starting frequency and number of subbands
  Ptr<const SpectrumModel> c = txParams->psd->GetSpectrumModel ();
  std::size_t numBands = c->GetNumBands ();
  double startingFreq = c->Begin ()->fl;

  // Get channel bandwidth and modulation class
  Ptr<const WifiSpectrumSignalParameters> wifiTxParams = DynamicCast<WifiSpectrumSignalParameters> (txParams);

  Ptr<WifiPpdu> ppdu = wifiTxParams->ppdu->Copy ();
  WifiTxVector txVector = ppdu->GetTxVector ();
  m_channelWidth = txVector.GetChannelWidth ();
  WifiModulationClass modulationClass = txVector.GetMode ().GetModulationClass ();

  // Build a tuple and check if seen before (if so store it)
  FreqWidthSubbandModulationTuple tupleForCurrentTx = std::make_tuple (startingFreq, m_channelWidth, numBands, modulationClass);
  bool found = false;
  for (std::vector<FreqWidthSubbandModulationTuple>::const_iterator it = m_distinctTuples.begin (); it != m_distinctTuples.end (); it++)
    {
      if (*it == tupleForCurrentTx)
        {
          found = true;
        }
    }
  if (!found)
    {
      m_distinctTuples.push_back (tupleForCurrentTx);
    }
}

void
Bug2843TestCase::SendPacketBurst (uint8_t numPackets, Ptr<NetDevice> sourceDevice,
                                  Address& destination) const
{
  for (uint8_t i = 0; i < numPackets; i++)
    {
      Ptr<Packet> pkt = Create<Packet> (1000);  // 1000 dummy bytes of data
      sourceDevice->Send (pkt, destination, 0);
    }
}

void
Bug2843TestCase::DoRun (void)
{
  uint16_t channelWidth = 40; // at least 40 MHz expected here

  NodeContainer wifiStaNode;
  wifiStaNode.Create (1);

  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  SpectrumWifiPhyHelper spectrumPhy;
  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  lossModel->SetFrequency (5.190e9);
  spectrumChannel->AddPropagationLossModel (lossModel);

  Ptr<ConstantSpeedPropagationDelayModel> delayModel
    = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  spectrumPhy.SetChannel (spectrumChannel);
  spectrumPhy.SetErrorRateModel ("ns3::NistErrorRateModel");
  spectrumPhy.Set ("ChannelSettings", StringValue ("{38, 40, BAND_5GHZ, 0}"));
  spectrumPhy.Set ("TxPowerStart", DoubleValue (10));
  spectrumPhy.Set ("TxPowerEnd", DoubleValue (10));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("VhtMcs8"),
                                "ControlMode", StringValue ("VhtMcs8"),
                                "RtsCtsThreshold", StringValue ("500")); // so as to force RTS/CTS for data frames

  WifiMacHelper mac;
  mac.SetType ("ns3::StaWifiMac");
  NetDeviceContainer staDevice;
  staDevice = wifi.Install (spectrumPhy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac");
  NetDeviceContainer apDevice;
  apDevice = wifi.Install (spectrumPhy, mac, wifiApNode);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0)); // put close enough in order to use MCS
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  // Send two 5 packet-bursts
  Simulator::Schedule (Seconds (0.5), &Bug2843TestCase::SendPacketBurst, this, 5, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (0.6), &Bug2843TestCase::SendPacketBurst, this, 5, apDevice.Get (0), staDevice.Get (0)->GetAddress ());

  Config::Connect ("/ChannelList/*/$ns3::MultiModelSpectrumChannel/TxSigParams", MakeCallback (&Bug2843TestCase::StoreDistinctTuple, this));

  Simulator::Stop (Seconds (0.8));
  Simulator::Run ();

  Simulator::Destroy ();

  // {starting frequency, channelWidth, Number of subbands in SpectrumModel, modulation type} tuples
  std::size_t numberTuples = m_distinctTuples.size ();
  NS_TEST_ASSERT_MSG_EQ (numberTuples, 2, "Only two distinct tuples expected");
  NS_TEST_ASSERT_MSG_EQ (std::get<0> (m_distinctTuples[0]) - 20e6, std::get<0> (m_distinctTuples[1]), "The starting frequency of the first tuple should be shifted 20 MHz to the right wrt second tuple");
  // Note that the first tuple should the one initiated by the beacon, i.e. non-HT OFDM (20 MHz)
  NS_TEST_ASSERT_MSG_EQ (std::get<1> (m_distinctTuples[0]), 20, "First tuple's channel width should be 20 MHz");
  NS_TEST_ASSERT_MSG_EQ (std::get<2> (m_distinctTuples[0]), 193, "First tuple should have 193 subbands (64+DC, 20MHz+DC, inband and 64*2 out-of-band, 20MHz on each side)");
  NS_TEST_ASSERT_MSG_EQ (std::get<3> (m_distinctTuples[0]), WifiModulationClass::WIFI_MOD_CLASS_OFDM, "First tuple should be OFDM");
  // Second tuple
  NS_TEST_ASSERT_MSG_EQ (std::get<1> (m_distinctTuples[1]), channelWidth, "Second tuple's channel width should be 40 MHz");
  NS_TEST_ASSERT_MSG_EQ (std::get<2> (m_distinctTuples[1]), 385, "Second tuple should have 385 subbands (128+DC, 40MHz+DC, inband and 128*2 out-of-band, 40MHz on each side)");
  NS_TEST_ASSERT_MSG_EQ (std::get<3> (m_distinctTuples[1]), WifiModulationClass::WIFI_MOD_CLASS_VHT, "Second tuple should be VHT_OFDM");
}

//-----------------------------------------------------------------------------
/**
 * Make sure that the channel width and the channel number can be changed at runtime.
 *
 * The scenario considers an access point and a station using a 20 MHz channel width.
 * After 1s, we change the channel width and the channel number to use a 40 MHz channel.
 * The tests checks the operational channel width sent in Beacon frames
 * and verify that the association procedure is executed twice.
 *
 * See \bugid{2831}
 */

class Bug2831TestCase : public TestCase
{
public:
  Bug2831TestCase ();
  virtual ~Bug2831TestCase ();
  void DoRun (void) override;

private:
  /**
   * Function called to change the supported channel width at runtime
   */
  void ChangeSupportedChannelWidth (void);
  /**
   * Callback triggered when a packet is received by the PHYs
   * \param context the context
   * \param p the received packet
   * \param rxPowersW the received power per channel band in watts
   */
  void RxCallback (std::string context, Ptr<const Packet> p, RxPowerWattPerChannelBand rxPowersW);

  Ptr<YansWifiPhy> m_apPhy; ///< AP PHY
  Ptr<YansWifiPhy> m_staPhy; ///< STA PHY

  uint16_t m_assocReqCount; ///< count number of association requests
  uint16_t m_assocRespCount; ///< count number of association responses
  uint16_t m_countOperationalChannelWidth20; ///< count number of beacon frames announcing a 20 MHz operating channel width
  uint16_t m_countOperationalChannelWidth40; ///< count number of beacon frames announcing a 40 MHz operating channel width
};

Bug2831TestCase::Bug2831TestCase ()
  : TestCase ("Test case for Bug 2831"),
    m_assocReqCount (0),
    m_assocRespCount (0),
    m_countOperationalChannelWidth20 (0),
    m_countOperationalChannelWidth40 (0)
{
}

Bug2831TestCase::~Bug2831TestCase ()
{
}

void
Bug2831TestCase::ChangeSupportedChannelWidth ()
{
  m_apPhy->SetOperatingChannel (WifiPhy::ChannelTuple {38, 40, (int)(WIFI_PHY_BAND_5GHZ), 0});
  m_staPhy->SetOperatingChannel (WifiPhy::ChannelTuple {38, 40, (int)(WIFI_PHY_BAND_5GHZ), 0});
}

void
Bug2831TestCase::RxCallback (std::string context, Ptr<const Packet> p, RxPowerWattPerChannelBand rxPowersW)
{
  Ptr<Packet> packet = p->Copy ();
  WifiMacHeader hdr;
  packet->RemoveHeader (hdr);
  if (hdr.IsAssocReq ())
    {
      m_assocReqCount++;
    }
  else if (hdr.IsAssocResp ())
    {
      m_assocRespCount++;
    }
  else if (hdr.IsBeacon ())
    {
      MgtBeaconHeader beacon;
      packet->RemoveHeader (beacon);
      HtOperation htOperation = beacon.GetHtOperation ();
      if (htOperation.GetStaChannelWidth () > 0)
        {
          m_countOperationalChannelWidth40++;
        }
      else
        {
          m_countOperationalChannelWidth20++;
        }
    }
}

void
Bug2831TestCase::DoRun (void)
{
  Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel> ();
  ObjectFactory propDelay;
  propDelay.SetTypeId ("ns3::ConstantSpeedPropagationDelayModel");
  Ptr<PropagationDelayModel> propagationDelay = propDelay.Create<PropagationDelayModel> ();
  Ptr<PropagationLossModel> propagationLoss = CreateObject<FriisPropagationLossModel> ();
  channel->SetPropagationDelayModel (propagationDelay);
  channel->SetPropagationLossModel (propagationLoss);

  Ptr<Node> apNode = CreateObject<Node> ();
  Ptr<WifiNetDevice> apDev = CreateObject<WifiNetDevice> ();
  apDev->SetStandard (WIFI_STANDARD_80211ax);
  Ptr<HtConfiguration> apHtConfiguration = CreateObject<HtConfiguration> ();
  apDev->SetHtConfiguration (apHtConfiguration);
  ObjectFactory mac;
  mac.SetTypeId ("ns3::ApWifiMac");
  mac.Set ("EnableBeaconJitter", BooleanValue (false));
  mac.Set ("QosSupported", BooleanValue (true));
  Ptr<WifiMac> apMac = mac.Create<WifiMac> ();
  apMac->SetDevice (apDev);
  apMac->SetAddress (Mac48Address::Allocate ());
  apMac->ConfigureStandard (WIFI_STANDARD_80211ax);
  Ptr<FrameExchangeManager> fem = apMac->GetFrameExchangeManager ();
  Ptr<WifiProtectionManager> protectionManager = CreateObject<WifiDefaultProtectionManager> ();
  protectionManager->SetWifiMac (apMac);
  fem->SetProtectionManager (protectionManager);
  Ptr<WifiAckManager> ackManager = CreateObject<WifiDefaultAckManager> ();
  ackManager->SetWifiMac (apMac);
  fem->SetAckManager (ackManager);

  Ptr<Node> staNode = CreateObject<Node> ();
  Ptr<WifiNetDevice> staDev = CreateObject<WifiNetDevice> ();
  staDev->SetStandard (WIFI_STANDARD_80211ax);
  Ptr<HtConfiguration> staHtConfiguration = CreateObject<HtConfiguration> ();
  staDev->SetHtConfiguration (staHtConfiguration);
  mac.SetTypeId ("ns3::StaWifiMac");
  Ptr<WifiMac> staMac = mac.Create<WifiMac> ();
  staMac->SetDevice (staDev);
  staMac->SetAddress (Mac48Address::Allocate ());
  staMac->ConfigureStandard (WIFI_STANDARD_80211ax);
  fem = staMac->GetFrameExchangeManager ();
  protectionManager = CreateObject<WifiDefaultProtectionManager> ();
  protectionManager->SetWifiMac (staMac);
  fem->SetProtectionManager (protectionManager);
  ackManager = CreateObject<WifiDefaultAckManager> ();
  ackManager->SetWifiMac (staMac);
  fem->SetAckManager (ackManager);

  Ptr<ConstantPositionMobilityModel> apMobility = CreateObject<ConstantPositionMobilityModel> ();
  apMobility->SetPosition (Vector (0.0, 0.0, 0.0));
  apNode->AggregateObject (apMobility);

  m_apPhy = CreateObject<YansWifiPhy> ();
  Ptr<InterferenceHelper> apInterferenceHelper = CreateObject<InterferenceHelper> ();
  m_apPhy->SetInterferenceHelper (apInterferenceHelper);
  Ptr<ErrorRateModel> apErrorModel = CreateObject<YansErrorRateModel> ();
  m_apPhy->SetErrorRateModel (apErrorModel);
  m_apPhy->SetChannel (channel);
  m_apPhy->SetMobility (apMobility);
  m_apPhy->SetDevice (apDev);
  m_apPhy->ConfigureStandard (WIFI_STANDARD_80211ax);
  m_apPhy->SetOperatingChannel (WifiPhy::ChannelTuple {36, 20, (int)(WIFI_PHY_BAND_5GHZ), 0});

  Ptr<ConstantPositionMobilityModel> staMobility = CreateObject<ConstantPositionMobilityModel> ();
  staMobility->SetPosition (Vector (1.0, 0.0, 0.0));
  staNode->AggregateObject (staMobility);

  m_staPhy = CreateObject<YansWifiPhy> ();
  Ptr<InterferenceHelper> staInterferenceHelper = CreateObject<InterferenceHelper> ();
  m_staPhy->SetInterferenceHelper (staInterferenceHelper);
  Ptr<ErrorRateModel> staErrorModel = CreateObject<YansErrorRateModel> ();
  m_staPhy->SetErrorRateModel (staErrorModel);
  m_staPhy->SetChannel (channel);
  m_staPhy->SetMobility (staMobility);
  m_staPhy->SetDevice (apDev);
  m_staPhy->ConfigureStandard (WIFI_STANDARD_80211ax);
  m_staPhy->SetOperatingChannel (WifiPhy::ChannelTuple {36, 20, (int)(WIFI_PHY_BAND_5GHZ), 0});

  apDev->SetMac (apMac);
  apDev->SetPhy (m_apPhy);
  ObjectFactory manager;
  manager.SetTypeId ("ns3::ConstantRateWifiManager");
  apDev->SetRemoteStationManager (manager.Create<WifiRemoteStationManager> ());
  apNode->AddDevice (apDev);

  staDev->SetMac (staMac);
  staDev->SetPhy (m_staPhy);
  staDev->SetRemoteStationManager (manager.Create<WifiRemoteStationManager> ());
  staNode->AddDevice (staDev);

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyRxBegin", MakeCallback (&Bug2831TestCase::RxCallback, this));

  Simulator::Schedule (Seconds (1.0), &Bug2831TestCase::ChangeSupportedChannelWidth, this);

  Simulator::Stop (Seconds (3.0));
  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_assocReqCount, 2, "Second Association request not received");
  NS_TEST_ASSERT_MSG_EQ (m_assocRespCount, 2, "Second Association response not received");
  NS_TEST_ASSERT_MSG_EQ (m_countOperationalChannelWidth20, 10, "Incorrect operational channel width before channel change");
  NS_TEST_ASSERT_MSG_EQ (m_countOperationalChannelWidth40, 20, "Incorrect operational channel width after channel change");
}

//-----------------------------------------------------------------------------
/**
 * Make sure that Wifi STA is correctly associating to the best AP (i.e.,
 * nearest from STA). We consider 3 AP and 1 STA. This test case consisted of
 * three sub tests:
 *   - The best AP sends its beacon later than the other APs. STA is expected
 *     to associate to the best AP.
 *   - The STA is using active scanning instead of passive, the rest of the
 *     APs works normally. STA is expected to associate to the best AP
 *   - The nearest AP is turned off after sending beacon and while STA is
 *     still scanning. STA is expected to associate to the second best AP.
 *
 * See \bugid{2399}
 * \todo Add explicit association refusal test if ns-3 implemented it.
 */

class StaWifiMacScanningTestCase : public TestCase
{
public:
  StaWifiMacScanningTestCase ();
  virtual ~StaWifiMacScanningTestCase ();
  void DoRun (void) override;

private:
  /**
   * Callback function on STA assoc event
   * \param context context string
   * \param bssid the associated AP's bssid
   */
  void AssocCallback (std::string context, Mac48Address bssid);
  /**
   * Turn beacon generation on the AP node
   * \param apNode the AP node
   */
  void TurnBeaconGenerationOn (Ptr<Node> apNode);
  /**
   * Turn the AP node off
   * \param apNode the AP node
   */
  void TurnApOff (Ptr<Node> apNode);
  /**
   * Setup test
   * \param nearestApBeaconGeneration set BeaconGeneration attribute of the nearest AP
   * \param staActiveProbe set ActiveProbing attribute of the STA
   * \return node container containing all nodes
   */
  NodeContainer Setup (bool nearestApBeaconGeneration, bool staActiveProbe);

  Mac48Address m_associatedApBssid; ///< Associated AP's bssid
};

StaWifiMacScanningTestCase::StaWifiMacScanningTestCase ()
  : TestCase ("Test case for StaWifiMac scanning capability")
{
}

StaWifiMacScanningTestCase::~StaWifiMacScanningTestCase ()
{
}

void
StaWifiMacScanningTestCase::AssocCallback (std::string context, Mac48Address bssid)
{
  m_associatedApBssid = bssid;
}

void
StaWifiMacScanningTestCase::TurnBeaconGenerationOn (Ptr<Node> apNode)
{
  Ptr<WifiNetDevice> netDevice = DynamicCast<WifiNetDevice> (apNode->GetDevice (0));
  Ptr<ApWifiMac> mac = DynamicCast<ApWifiMac> (netDevice->GetMac ());
  mac->SetAttribute ("BeaconGeneration", BooleanValue (true));
}

void
StaWifiMacScanningTestCase::TurnApOff (Ptr<Node> apNode)
{
  Ptr<WifiNetDevice> netDevice = DynamicCast<WifiNetDevice> (apNode->GetDevice (0));
  Ptr<WifiPhy> phy = netDevice->GetPhy ();
  phy->SetOffMode ();
}

NodeContainer
StaWifiMacScanningTestCase::Setup (bool nearestApBeaconGeneration, bool staActiveProbe)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 1;

  NodeContainer apNodes;
  apNodes.Create (2);

  Ptr<Node> apNodeNearest = CreateObject<Node> ();
  Ptr<Node> staNode = CreateObject<Node> ();

  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager");

  WifiMacHelper mac;
  NetDeviceContainer apDevice, apDeviceNearest;
  mac.SetType ("ns3::ApWifiMac",
               "BeaconGeneration", BooleanValue (true));
  apDevice = wifi.Install (phy, mac, apNodes);
  mac.SetType ("ns3::ApWifiMac",
               "BeaconGeneration", BooleanValue (nearestApBeaconGeneration));
  apDeviceNearest = wifi.Install (phy, mac, apNodeNearest);

  NetDeviceContainer staDevice;
  mac.SetType ("ns3::StaWifiMac",
               "ActiveProbing", BooleanValue (staActiveProbe));
  staDevice = wifi.Install (phy, mac, staNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevice, streamNumber);
  wifi.AssignStreams (apDeviceNearest, streamNumber + 1);
  wifi.AssignStreams (staDevice, streamNumber + 2);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));  // Furthest AP
  positionAlloc->Add (Vector (10.0, 0.0, 0.0)); // Second nearest AP
  positionAlloc->Add (Vector (5.0, 5.0, 0.0));  // Nearest AP
  positionAlloc->Add (Vector (6.0, 5.0, 0.0));  // STA
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (apNodes);
  mobility.Install (apNodeNearest);
  mobility.Install (staNode);

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/Assoc", MakeCallback (&StaWifiMacScanningTestCase::AssocCallback, this));

  NodeContainer allNodes = NodeContainer (apNodes, apNodeNearest, staNode);
  return allNodes;
}

void
StaWifiMacScanningTestCase::DoRun (void)
{
  {
    NodeContainer nodes = Setup (false, false);
    Ptr<Node> nearestAp = nodes.Get (2);
    Mac48Address nearestApAddr = DynamicCast<WifiNetDevice> (nearestAp->GetDevice (0))->GetMac ()->GetAddress ();

    Simulator::Schedule (Seconds (0.05), &StaWifiMacScanningTestCase::TurnBeaconGenerationOn, this, nearestAp);

    Simulator::Stop (Seconds (0.2));
    Simulator::Run ();
    Simulator::Destroy ();

    NS_TEST_ASSERT_MSG_EQ (m_associatedApBssid, nearestApAddr, "STA is associated to the wrong AP");
  }
  m_associatedApBssid = Mac48Address ();
  {
    NodeContainer nodes = Setup (true, true);
    Ptr<Node> nearestAp = nodes.Get (2);
    Mac48Address nearestApAddr = DynamicCast<WifiNetDevice> (nearestAp->GetDevice (0))->GetMac ()->GetAddress ();

    Simulator::Stop (Seconds (0.2));
    Simulator::Run ();
    Simulator::Destroy ();

    NS_TEST_ASSERT_MSG_EQ (m_associatedApBssid, nearestApAddr, "STA is associated to the wrong AP");
  }
  m_associatedApBssid = Mac48Address ();
  {
    NodeContainer nodes = Setup (true, false);
    Ptr<Node> nearestAp = nodes.Get (2);
    Mac48Address secondNearestApAddr = DynamicCast<WifiNetDevice> (nodes.Get (1)->GetDevice (0))->GetMac ()->GetAddress ();

    Simulator::Schedule (Seconds (0.1), &StaWifiMacScanningTestCase::TurnApOff, this, nearestAp);

    Simulator::Stop (Seconds (1.5));
    Simulator::Run ();
    Simulator::Destroy ();

    NS_TEST_ASSERT_MSG_EQ (m_associatedApBssid, secondNearestApAddr, "STA is associated to the wrong AP");
  }
}

//-----------------------------------------------------------------------------
/**
 * Make sure that the ADDBA handshake process is protected.
 *
 * The scenario considers an access point and a station. It utilizes
 * ReceiveListErrorModel to drop by force ADDBA request on STA or ADDBA
 * response on AP. The AP sends 5 packets of each 1000 bytes (thus generating
 * BA agreement), 2 times during the test at 0.5s and 0.8s. We only drop the
 * first ADDBA request/response of the first BA negotiation. Therefore, we
 * expect that the packets still in queue after the failed BA agreement will be
 * sent with normal MPDU, and packets queued after that should be sent with
 * A-MPDU.
 *
 * This test consider 2 cases:
 *
 *   1. ADDBA request packets are blocked on receive at STA, triggering
 *      transmission failure at AP
 *   2. ADDBA response packets are blocked on receive at AP, STA stops
 *      retransmission of ADDBA response
 *
 * See \bugid{2470}
 */

class Bug2470TestCase : public TestCase
{
public:
  Bug2470TestCase ();
  virtual ~Bug2470TestCase ();
  void DoRun (void) override;

private:
  /**
   * Callback when ADDBA state changed
   * \param context node context
   * \param t the time the state changed
   * \param recipient the MAC address of the recipient
   * \param tid the TID
   * \param state the state
   */
  void AddbaStateChangedCallback (std::string context, Time t, Mac48Address recipient, uint8_t tid, OriginatorBlockAckAgreement::State state);
  /**
   * Callback when packet is received
   * \param context node context
   * \param p the received packet
   * \param channelFreqMhz the channel frequency in MHz
   * \param txVector the TX vector
   * \param aMpdu the A-MPDU info
   * \param signalNoise the signal noise in dBm
   * \param staId the STA-ID
   */
  void RxCallback (std::string context, Ptr<const Packet> p, uint16_t channelFreqMhz, WifiTxVector txVector, MpduInfo aMpdu, SignalNoiseDbm signalNoise, uint16_t staId);
  /**
   * Callback when packet is dropped
   * \param context node context
   * \param p the failed packet
   * \param snr the SNR of the failed packet in linear scale
   */
  void RxErrorCallback (std::string context, Ptr<const Packet> p, double snr);
  /**
   * Triggers the arrival of a burst of 1000 Byte-long packets in the source device
   * \param numPackets number of packets in burst
   * \param sourceDevice pointer to the source NetDevice
   * \param destination address of the destination device
   */
  void SendPacketBurst (uint32_t numPackets, Ptr<NetDevice> sourceDevice, Address& destination) const;
  /**
   * Run subtest for this test suite
   * \param apErrorModel ErrorModel used for AP
   * \param staErrorModel ErrorModel used for STA
   */
  void RunSubtest (PointerValue apErrorModel, PointerValue staErrorModel);

  uint16_t m_receivedNormalMpduCount; ///< Count received normal MPDU packets on STA
  uint16_t m_receivedAmpduCount;      ///< Count received A-MPDU packets on STA
  uint16_t m_failedActionCount;       ///< Count failed ADDBA request/response
  uint16_t m_addbaEstablishedCount;   ///< Count number of times ADDBA state machine is in established state
  uint16_t m_addbaPendingCount;       ///< Count number of times ADDBA state machine is in pending state
  uint16_t m_addbaRejectedCount;      ///< Count number of times ADDBA state machine is in rejected state
  uint16_t m_addbaNoReplyCount;       ///< Count number of times ADDBA state machine is in no_reply state
  uint16_t m_addbaResetCount;         ///< Count number of times ADDBA state machine is in reset state
};

Bug2470TestCase::Bug2470TestCase ()
  : TestCase ("Test case for Bug 2470"),
    m_receivedNormalMpduCount (0),
    m_receivedAmpduCount (0),
    m_failedActionCount (0),
    m_addbaEstablishedCount (0),
    m_addbaPendingCount (0),
    m_addbaRejectedCount (0),
    m_addbaNoReplyCount (0),
    m_addbaResetCount (0)
{
}

Bug2470TestCase::~Bug2470TestCase ()
{
}

void
Bug2470TestCase::AddbaStateChangedCallback (std::string context, Time t, Mac48Address recipient, uint8_t tid, OriginatorBlockAckAgreement::State state)
{
  switch (state)
    {
    case OriginatorBlockAckAgreement::ESTABLISHED:
      m_addbaEstablishedCount++;
      break;
    case OriginatorBlockAckAgreement::PENDING:
      m_addbaPendingCount++;
      break;
    case OriginatorBlockAckAgreement::REJECTED:
      m_addbaRejectedCount++;
      break;
    case OriginatorBlockAckAgreement::NO_REPLY:
      m_addbaNoReplyCount++;
      break;
    case OriginatorBlockAckAgreement::RESET:
      m_addbaResetCount++;
      break;
    }
}

void
Bug2470TestCase::RxCallback (std::string context, Ptr<const Packet> p, uint16_t channelFreqMhz, WifiTxVector txVector, MpduInfo aMpdu, SignalNoiseDbm signalNoise, uint16_t staId)
{
  Ptr<Packet> packet = p->Copy ();
  if (aMpdu.type != MpduType::NORMAL_MPDU)
    {
      m_receivedAmpduCount++;
    }
  else
    {
      WifiMacHeader hdr;
      packet->RemoveHeader (hdr);
      if (hdr.IsData ())
        {
          m_receivedNormalMpduCount++;
        }
    }
}

void
Bug2470TestCase::RxErrorCallback (std::string context, Ptr<const Packet> p, double snr)
{
  Ptr<Packet> packet = p->Copy ();
  WifiMacHeader hdr;
  packet->RemoveHeader (hdr);
  if (hdr.IsAction ())
    {
      m_failedActionCount++;
    }
}

void
Bug2470TestCase::SendPacketBurst (uint32_t numPackets, Ptr<NetDevice> sourceDevice,
                                  Address& destination) const
{
  for (uint32_t i = 0; i < numPackets; i++)
    {
      Ptr<Packet> pkt = Create<Packet> (1000); // 1000 dummy bytes of data
      sourceDevice->Send (pkt, destination, 0);
    }
}

void
Bug2470TestCase::RunSubtest (PointerValue apErrorModel, PointerValue staErrorModel)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 200;

  NodeContainer wifiApNode, wifiStaNode;
  wifiApNode.Create (1);
  wifiStaNode.Create (1);

  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("HtMcs7"),
                                "ControlMode", StringValue ("HtMcs7"));

  WifiMacHelper mac;
  NetDeviceContainer apDevice;
  phy.Set ("PostReceptionErrorModel", apErrorModel);
  phy.Set ("ChannelSettings", StringValue ("{36, 20, BAND_5GHZ, 0}"));
  mac.SetType ("ns3::ApWifiMac", "EnableBeaconJitter", BooleanValue (false));
  apDevice = wifi.Install (phy, mac, wifiApNode);

  NetDeviceContainer staDevice;
  phy.Set ("PostReceptionErrorModel", staErrorModel);
  mac.SetType ("ns3::StaWifiMac");
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevice, streamNumber);
  wifi.AssignStreams (staDevice, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/MonitorSnifferRx", MakeCallback (&Bug2470TestCase::RxCallback, this));
  Config::Connect ("/NodeList/*/DeviceList/*/Phy/State/RxError", MakeCallback (&Bug2470TestCase::RxErrorCallback, this));
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/BE_Txop/BlockAckManager/AgreementState", MakeCallback (&Bug2470TestCase::AddbaStateChangedCallback, this));

  Simulator::Schedule (Seconds (0.5), &Bug2470TestCase::SendPacketBurst, this, 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (0.5) + MicroSeconds (5), &Bug2470TestCase::SendPacketBurst, this, 4, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (0.8), &Bug2470TestCase::SendPacketBurst, this, 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (0.8) + MicroSeconds (5), &Bug2470TestCase::SendPacketBurst, this, 4, apDevice.Get (0), staDevice.Get (0)->GetAddress ());

  Simulator::Stop (Seconds (1.0));
  Simulator::Run ();
  Simulator::Destroy ();
}

void
Bug2470TestCase::DoRun (void)
{
  // Create ReceiveListErrorModel to corrupt ADDBA req packet. We use ReceiveListErrorModel
  // instead of ListErrorModel since packet UID is incremented between simulations. But
  // problem may occur because of random stream, therefore we suppress usage of RNG as
  // much as possible (i.e., removing beacon jitter).
  Ptr<ReceiveListErrorModel> staPem = CreateObject<ReceiveListErrorModel> ();
  std::list<uint32_t> blackList;
  // Block ADDBA request 6 times (== maximum number of MAC frame transmissions in the ADDBA response timeout interval)
  blackList.push_back (9);
  blackList.push_back (10);
  blackList.push_back (11);
  blackList.push_back (12);
  blackList.push_back (13);
  blackList.push_back (14);
  staPem->SetList (blackList);

  {
    RunSubtest (PointerValue (), PointerValue (staPem));
    NS_TEST_ASSERT_MSG_EQ (m_failedActionCount, 6, "ADDBA request packets are not failed");
    // There are two sets of 5 packets to be transmitted. The first 5 packets should be sent by normal
    // MPDU because of failed ADDBA handshake. For the second set, the first packet should be sent by
    // normal MPDU, and the rest with A-MPDU. In total we expect to receive 2 normal MPDU packets and
    // 8 A-MPDU packets.
    NS_TEST_ASSERT_MSG_EQ (m_receivedNormalMpduCount, 2, "Receiving incorrect number of normal MPDU packet on subtest 1");
    NS_TEST_ASSERT_MSG_EQ (m_receivedAmpduCount, 8, "Receiving incorrect number of A-MPDU packet on subtest 1");

    NS_TEST_ASSERT_MSG_EQ (m_addbaEstablishedCount, 1, "Incorrect number of times the ADDBA state machine was in established state on subtest 1");
    NS_TEST_ASSERT_MSG_EQ (m_addbaPendingCount, 1, "Incorrect number of times the ADDBA state machine was in pending state on subtest 1");
    NS_TEST_ASSERT_MSG_EQ (m_addbaRejectedCount, 0, "Incorrect number of times the ADDBA state machine was in rejected state on subtest 1");
    NS_TEST_ASSERT_MSG_EQ (m_addbaNoReplyCount, 0, "Incorrect number of times the ADDBA state machine was in no_reply state on subtest 1");
    NS_TEST_ASSERT_MSG_EQ (m_addbaResetCount, 0, "Incorrect number of times the ADDBA state machine was in reset state on subtest 1");
  }

  m_receivedNormalMpduCount = 0;
  m_receivedAmpduCount = 0;
  m_failedActionCount = 0;
  m_addbaEstablishedCount = 0;
  m_addbaPendingCount = 0;
  m_addbaRejectedCount = 0;
  m_addbaNoReplyCount = 0;
  m_addbaResetCount = 0;

  Ptr<ReceiveListErrorModel> apPem = CreateObject<ReceiveListErrorModel> ();
  blackList.clear ();
  // Block ADDBA request 4 times (== maximum number of MAC frame transmissions in the ADDBA response timeout interval)
  blackList.push_back (5);
  blackList.push_back (6);
  blackList.push_back (7);
  blackList.push_back (9);
  apPem->SetList (blackList);

  {
    RunSubtest (PointerValue (apPem), PointerValue ());
    NS_TEST_ASSERT_MSG_EQ (m_failedActionCount, 4, "ADDBA response packets are not failed");
    // Similar to subtest 1, we also expect to receive 6 normal MPDU packets and 4 A-MPDU packets.
    NS_TEST_ASSERT_MSG_EQ (m_receivedNormalMpduCount, 6, "Receiving incorrect number of normal MPDU packet on subtest 2");
    NS_TEST_ASSERT_MSG_EQ (m_receivedAmpduCount, 4, "Receiving incorrect number of A-MPDU packet on subtest 2");

    NS_TEST_ASSERT_MSG_EQ (m_addbaEstablishedCount, 1, "Incorrect number of times the ADDBA state machine was in established state on subtest 2");
    NS_TEST_ASSERT_MSG_EQ (m_addbaPendingCount, 1, "Incorrect number of times the ADDBA state machine was in pending state on subtest 2");
    NS_TEST_ASSERT_MSG_EQ (m_addbaRejectedCount, 0, "Incorrect number of times the ADDBA state machine was in rejected state on subtest 2");
    NS_TEST_ASSERT_MSG_EQ (m_addbaNoReplyCount, 1, "Incorrect number of times the ADDBA state machine was in no_reply state on subtest 2");
    NS_TEST_ASSERT_MSG_EQ (m_addbaResetCount, 0, "Incorrect number of times the ADDBA state machine was in reset state on subtest 2");
  }

  // TODO: In the second test set, it does not go to reset state since ADDBA response is received after timeout (NO_REPLY)
  // but before it does not enter RESET state. More tests should be written to verify all possible scenarios.
}

//-----------------------------------------------------------------------------
/**
 * Make sure that Ideal rate manager recovers when the station is moving away from the access point.
 *
 * The scenario considers an access point and a moving station.
 * Initially, the station is located at 1 meter from the access point.
 * After 1s, the station moves away from the access for 0.5s to
 * reach a point away of 50 meters from the access point.
 * The tests checks the Ideal rate manager is reset once it has
 * failed to transmit a data packet, so that the next data packets
 * can be successfully transmitted using a lower modulation.
 *
 * See \issueid{40}
 */

class Issue40TestCase : public TestCase
{
public:
  Issue40TestCase ();
  virtual ~Issue40TestCase ();
  void DoRun (void) override;

private:
  /**
   * Run one function
   * \param useAmpdu flag to indicate whether the test should be run with A-MPDU
   */
  void RunOne (bool useAmpdu);

  /**
   * Callback when packet is successfully received
   * \param context node context
   * \param p the received packet
   */
  void RxSuccessCallback (std::string context, Ptr<const Packet> p);
  /**
   * Triggers the arrival of 1000 Byte-long packets in the source device
   * \param numPackets number of packets in burst
   * \param sourceDevice pointer to the source NetDevice
   * \param destination address of the destination device
   */
   void SendPackets (uint8_t numPackets, Ptr<NetDevice> sourceDevice, Address& destination);
  /**
   * Transmit final data failed function
   * \param context the context
   * \param address the MAC address
   */
  void TxFinalDataFailedCallback (std::string context, Mac48Address address);

  uint16_t m_rxCount; ///< Count number of successfully received data packets
  uint16_t m_txCount; ///< Count number of transmitted data packets
  uint16_t m_txMacFinalDataFailedCount; ///< Count number of unsuccessfully transmitted data packets
};

Issue40TestCase::Issue40TestCase ()
  : TestCase ("Test case for issue #40"),
    m_rxCount (0),
    m_txCount (0),
    m_txMacFinalDataFailedCount (0)
{
}

Issue40TestCase::~Issue40TestCase ()
{
}

void
Issue40TestCase::RxSuccessCallback (std::string context, Ptr<const Packet> p)
{
  m_rxCount++;
}

void
Issue40TestCase::SendPackets (uint8_t numPackets, Ptr<NetDevice> sourceDevice, Address& destination)
{
  for (uint8_t i = 0; i < numPackets; i++)
    {
      Ptr<Packet> pkt = Create<Packet> (1000); // 1000 dummy bytes of data
      sourceDevice->Send (pkt, destination, 0);
      m_txCount++;
    }
}

void
Issue40TestCase::TxFinalDataFailedCallback (std::string context, Mac48Address address)
{
  m_txMacFinalDataFailedCount++;
}

void
Issue40TestCase::RunOne (bool useAmpdu)
{
  m_rxCount = 0;
  m_txCount = 0;
  m_txMacFinalDataFailedCount = 0;

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 100;

  NodeContainer wifiApNode, wifiStaNode;
  wifiApNode.Create (1);
  wifiStaNode.Create (1);

  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  WifiMacHelper mac;
  NetDeviceContainer apDevice;
  mac.SetType ("ns3::ApWifiMac");
  apDevice = wifi.Install (phy, mac, wifiApNode);

  NetDeviceContainer staDevice;
  mac.SetType ("ns3::StaWifiMac");
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevice, streamNumber);
  wifi.AssignStreams (staDevice, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (10.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);

  mobility.SetMobilityModel("ns3::WaypointMobilityModel");
  mobility.Install (wifiStaNode);

  Config::Connect ("/NodeList/*/DeviceList/*/RemoteStationManager/MacTxFinalDataFailed", MakeCallback (&Issue40TestCase::TxFinalDataFailedCallback, this));
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/MacRx", MakeCallback (&Issue40TestCase::RxSuccessCallback, this));
              
  Ptr<WaypointMobilityModel> staWaypointMobility = DynamicCast<WaypointMobilityModel>(wifiStaNode.Get(0)->GetObject<MobilityModel>());
  staWaypointMobility->AddWaypoint (Waypoint (Seconds(1.0), Vector (10.0, 0.0, 0.0)));
  staWaypointMobility->AddWaypoint (Waypoint (Seconds(1.5), Vector (50.0, 0.0, 0.0)));

  if (useAmpdu)
    {
      // Disable use of BAR that are sent with the lowest modulation so that we can also reproduce the problem with A-MPDU, i.e. the lack of feedback about SNR change
      Ptr<WifiNetDevice> ap_device = DynamicCast<WifiNetDevice> (apDevice.Get (0));
      PointerValue ptr;
      ap_device->GetMac ()->GetAttribute ("BE_Txop", ptr);
      ptr.Get<QosTxop> ()->SetAttribute ("UseExplicitBarAfterMissedBlockAck", BooleanValue (false));
  }

  // Transmit a first data packet before the station moves: it should be sent with a high modulation and successfully received
  Simulator::Schedule (Seconds (0.5), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());

  // Transmit a second data packet once the station is away from the access point: it should be sent with the same high modulation and be unsuccessfully received
  Simulator::Schedule (Seconds (2.0), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());

  // Keep on transmitting data packets while the station is away from the access point: it should be sent with a lower modulation and be successfully received
  Simulator::Schedule (Seconds (2.1), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (2.2), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (2.3), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (2.4), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (2.5), &Issue40TestCase::SendPackets, this, useAmpdu ? 2 : 1, apDevice.Get (0), staDevice.Get (0)->GetAddress ());

  Simulator::Stop (Seconds (3.0));
  Simulator::Run ();

  NS_TEST_ASSERT_MSG_EQ (m_txCount, (useAmpdu ? 14 : 7), "Incorrect number of transmitted packets");
  NS_TEST_ASSERT_MSG_EQ (m_rxCount, (useAmpdu ? 12 : 6), "Incorrect number of successfully received packets");
  NS_TEST_ASSERT_MSG_EQ (m_txMacFinalDataFailedCount, 1, "Incorrect number of dropped TX packets");

  Simulator::Destroy ();
}

void
Issue40TestCase::DoRun (void)
{
  //Test without A-MPDU
  RunOne (false);

  //Test with A-MPDU
  RunOne (true);
}

//-----------------------------------------------------------------------------
/**
 * Make sure that Ideal rate manager is able to handle non best-effort traffic.
 *
 * The scenario considers an access point and a fixed station.
 * The station first sends a best-effort packet to the access point,
 * for which Ideal rate manager should select a VHT rate. Then,
 * the station sends a non best-effort (voice) packet to the access point,
 * and since SNR is unchanged, the same VHT rate should be used.
 *
 * See \issueid{169}
 */

class Issue169TestCase : public TestCase
{
public:
  Issue169TestCase ();
  virtual ~Issue169TestCase ();
  void DoRun (void) override;

private:
  /**
   * Triggers the transmission of a 1000 Byte-long data packet from the source device
   * \param numPackets number of packets in burst
   * \param sourceDevice pointer to the source NetDevice
   * \param destination address of the destination device
   * \param priority the priority of the packets to send
   */
   void SendPackets (uint8_t numPackets, Ptr<NetDevice> sourceDevice, Address& destination, uint8_t priority);

  /**
   * Callback that indicates a PSDU is being transmitted
   * \param context the context
   * \param psdus the PSDU map to transmit
   * \param txVector the TX vector
   * \param txPowerW the TX power (W)
   */
  void TxCallback (std::string context, WifiConstPsduMap psdus, WifiTxVector txVector, double txPowerW);
};

Issue169TestCase::Issue169TestCase ()
  : TestCase ("Test case for issue #169")
{
}

Issue169TestCase::~Issue169TestCase ()
{
}

void
Issue169TestCase::SendPackets (uint8_t numPackets, Ptr<NetDevice> sourceDevice, Address& destination, uint8_t priority)
{
  SocketPriorityTag priorityTag;
  priorityTag.SetPriority (priority);
  for (uint8_t i = 0; i < numPackets; i++)
    {
      Ptr<Packet> packet = Create<Packet> (1000); // 1000 dummy bytes of data
      packet->AddPacketTag (priorityTag);
      sourceDevice->Send (packet, destination, 0);
    }
}

void
Issue169TestCase::TxCallback (std::string context, WifiConstPsduMap psdus, WifiTxVector txVector, double txPowerW)
{
  if (psdus.begin()->second->GetSize () >= 1000)
    {
      NS_TEST_ASSERT_MSG_EQ (txVector.GetMode ().GetModulationClass (), WifiModulationClass::WIFI_MOD_CLASS_VHT, "Ideal rate manager selected incorrect modulation class");
    }
}

void
Issue169TestCase::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 100;

  NodeContainer wifiApNode, wifiStaNode;
  wifiApNode.Create (1);
  wifiStaNode.Create (1);

  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  WifiMacHelper mac;
  NetDeviceContainer apDevice;
  mac.SetType ("ns3::ApWifiMac");
  apDevice = wifi.Install (phy, mac, wifiApNode);

  NetDeviceContainer staDevice;
  mac.SetType ("ns3::StaWifiMac");
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevice, streamNumber);
  wifi.AssignStreams (staDevice, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxPsduBegin", MakeCallback (&Issue169TestCase::TxCallback, this));

  //Send best-effort packet (i.e. priority 0)
  Simulator::Schedule (Seconds (0.5), &Issue169TestCase::SendPackets, this, 1, apDevice.Get (0), staDevice.Get (0)->GetAddress (), 0);

  //Send non best-effort (voice) packet (i.e. priority 6)
  Simulator::Schedule (Seconds (1.0), &Issue169TestCase::SendPackets, this, 1, apDevice.Get (0), staDevice.Get (0)->GetAddress (), 6);

  Simulator::Stop (Seconds (2.0));
  Simulator::Run ();

  Simulator::Destroy ();
}


//-----------------------------------------------------------------------------
/**
 * Make sure that Ideal rate manager properly selects MCS based on the configured channel width.
 *
 * The scenario considers an access point and a fixed station.
 * The access point first sends a 80 MHz PPDU to the station,
 * for which Ideal rate manager should select VH-MCS 0 based
 * on the distance (no interference generated in this test). Then,
 * the access point sends a 20 MHz PPDU to the station,
 * which corresponds to a SNR 6 dB higher than previously, hence
 * VHT-MCS 2 should be selected. Finally, the access point sends a
 * 40 MHz PPDU to the station, which means corresponds to a SNR 3 dB
 * lower than previously, hence VHT-MCS 1 should be selected.
 */

class IdealRateManagerChannelWidthTest : public TestCase
{
public:
  IdealRateManagerChannelWidthTest ();
  virtual ~IdealRateManagerChannelWidthTest ();
  void DoRun (void) override;

private:
  /**
   * Change the configured channel width for all nodes
   * \param channelWidth the channel width (in MHz)
   */
  void ChangeChannelWidth (uint16_t channelWidth);

  /**
   * Triggers the transmission of a 1000 Byte-long data packet from the source device
   * \param sourceDevice pointer to the source NetDevice
   * \param destination address of the destination device
   */
   void SendPacket (Ptr<NetDevice> sourceDevice, Address& destination);

  /**
   * Callback that indicates a PSDU is being transmitted
   * \param context the context
   * \param psduMap the PSDU map to transmit
   * \param txVector the TX vector
   * \param txPowerW the TX power (W)
   */
  void TxCallback (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);

  /**
   * Check if the selected WifiMode is correct
   * \param expectedMode the expected WifiMode
   */
  void CheckLastSelectedMode (WifiMode expectedMode);

  WifiMode m_txMode; ///< Store the last selected mode to send data packet
};

IdealRateManagerChannelWidthTest::IdealRateManagerChannelWidthTest ()
  : TestCase ("Test case for use of channel bonding with Ideal rate manager")
{
}

IdealRateManagerChannelWidthTest::~IdealRateManagerChannelWidthTest ()
{
}

void
IdealRateManagerChannelWidthTest::ChangeChannelWidth (uint16_t channelWidth)
{
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
               StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));
}

void
IdealRateManagerChannelWidthTest::SendPacket (Ptr<NetDevice> sourceDevice, Address& destination)
{
  Ptr<Packet> packet = Create<Packet> (1000);
  sourceDevice->Send (packet, destination, 0);
}

void
IdealRateManagerChannelWidthTest::TxCallback (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
{
  if (psduMap.begin ()->second->GetSize () >= 1000)
    {
      m_txMode = txVector.GetMode ();
    }
}

void
IdealRateManagerChannelWidthTest::CheckLastSelectedMode (WifiMode expectedMode)
{
  NS_TEST_ASSERT_MSG_EQ (m_txMode, expectedMode, "Last selected WifiMode " << m_txMode << " does not match expected WifiMode " << expectedMode);
}

void
IdealRateManagerChannelWidthTest::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 100;

  NodeContainer wifiApNode, wifiStaNode;
  wifiApNode.Create (1);
  wifiStaNode.Create (1);

  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  WifiMacHelper mac;
  NetDeviceContainer apDevice;
  mac.SetType ("ns3::ApWifiMac");
  apDevice = wifi.Install (phy, mac, wifiApNode);

  NetDeviceContainer staDevice;
  mac.SetType ("ns3::StaWifiMac");
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevice, streamNumber);
  wifi.AssignStreams (staDevice, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (50.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxPsduBegin", MakeCallback (&IdealRateManagerChannelWidthTest::TxCallback, this));

  //Set channel width to 80 MHz & send packet
  Simulator::Schedule (Seconds (0.5), &IdealRateManagerChannelWidthTest::ChangeChannelWidth, this, 80);
  Simulator::Schedule (Seconds (1.0), &IdealRateManagerChannelWidthTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  //Selected rate should be VHT-MCS 1
  Simulator::Schedule (Seconds (1.1), &IdealRateManagerChannelWidthTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs1 ());

  //Set channel width to 20 MHz & send packet
  Simulator::Schedule (Seconds (1.5), &IdealRateManagerChannelWidthTest::ChangeChannelWidth, this, 20);
  Simulator::Schedule (Seconds (2.0), &IdealRateManagerChannelWidthTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  //Selected rate should be VHT-MCS 3 since SNR should be 6 dB higher than previously
  Simulator::Schedule (Seconds (2.1), &IdealRateManagerChannelWidthTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs3 ());

  //Set channel width to 40 MHz & send packet
  Simulator::Schedule (Seconds (2.5), &IdealRateManagerChannelWidthTest::ChangeChannelWidth, this, 40);
  Simulator::Schedule (Seconds (3.0), &IdealRateManagerChannelWidthTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  //Selected rate should be VHT-MCS 2 since SNR should be 3 dB lower than previously
  Simulator::Schedule (Seconds (3.1), &IdealRateManagerChannelWidthTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs2 ());

  Simulator::Stop (Seconds (3.2));
  Simulator::Run ();

  Simulator::Destroy ();
}


//-----------------------------------------------------------------------------
/**
 * Test to validate that Ideal rate manager properly selects TXVECTOR in scenarios where MIMO is used.
 * The test consider both balanced and unbalanced MIMO settings, and verify ideal picks the correct number
 * of spatial streams and the correct MCS, taking into account potential diversity in AWGN channels when the
 * number of antenna at the receiver is higher than the number of spatial streams used for the transmission.
 */

class IdealRateManagerMimoTest : public TestCase
{
public:
  IdealRateManagerMimoTest ();
  virtual ~IdealRateManagerMimoTest ();
  void DoRun (void) override;

private:
  /**
   * Change the configured MIMO  settings  for AP node
   * \param antennas the number of active antennas
   * \param maxStreams the maximum number of allowed spatial streams
   */
  void SetApMimoSettings (uint8_t antennas, uint8_t maxStreams);
  /**
   * Change the configured MIMO  settings  for STA node
   * \param antennas the number of active antennas
   * \param maxStreams the maximum number of allowed spatial streams
   */
  void SetStaMimoSettings (uint8_t antennas, uint8_t maxStreams);
  /**
   * Triggers the transmission of a 1000 Byte-long data packet from the source device
   * \param sourceDevice pointer to the source NetDevice
   * \param destination address of the destination device
   */
   void SendPacket (Ptr<NetDevice> sourceDevice, Address& destination);

  /**
   * Callback that indicates a PSDU is being transmitted
   * \param context the context
   * \param psdus the PSDU map to transmit
   * \param txVector the TX vector
   * \param txPowerW the TX power (W)
   */
  void TxCallback (std::string context, WifiConstPsduMap psdus, WifiTxVector txVector, double txPowerW);

  /**
   * Check if the selected WifiMode is correct
   * \param expectedMode the expected WifiMode
   */
  void CheckLastSelectedMode (WifiMode expectedMode);
  /**
   * Check if the selected Nss is correct
   * \param expectedNss the expected Nss
   */
  void CheckLastSelectedNss (uint8_t expectedNss);

  WifiTxVector m_txVector; ///< Store the last TXVECTOR used to transmit Data
};

IdealRateManagerMimoTest::IdealRateManagerMimoTest ()
  : TestCase ("Test case for use of imbalanced MIMO settings with Ideal rate manager")
{
}

IdealRateManagerMimoTest::~IdealRateManagerMimoTest ()
{
}

void
IdealRateManagerMimoTest::SetApMimoSettings (uint8_t antennas, uint8_t maxStreams)
{
  Config::Set ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/Antennas", UintegerValue (antennas));
  Config::Set ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/MaxSupportedTxSpatialStreams", UintegerValue (maxStreams));
  Config::Set ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/MaxSupportedRxSpatialStreams", UintegerValue (maxStreams));
}

void
IdealRateManagerMimoTest::SetStaMimoSettings (uint8_t antennas, uint8_t maxStreams)
{
  Config::Set ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Phy/Antennas", UintegerValue (antennas));
  Config::Set ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Phy/MaxSupportedTxSpatialStreams", UintegerValue (maxStreams));
  Config::Set ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Phy/MaxSupportedRxSpatialStreams", UintegerValue (maxStreams));
}

void
IdealRateManagerMimoTest::SendPacket (Ptr<NetDevice> sourceDevice, Address& destination)
{
  Ptr<Packet> packet = Create<Packet> (1000);
  sourceDevice->Send (packet, destination, 0);
}

void
IdealRateManagerMimoTest::TxCallback (std::string context, WifiConstPsduMap psdus, WifiTxVector txVector, double txPowerW)
{
  if (psdus.begin ()->second->GetSize () >= 1000)
    {
      m_txVector = txVector;
    }
}

void
IdealRateManagerMimoTest::CheckLastSelectedNss (uint8_t expectedNss)
{
  NS_TEST_ASSERT_MSG_EQ (m_txVector.GetNss (), expectedNss, "Last selected Nss " << m_txVector.GetNss () << " does not match expected Nss " << expectedNss);
}

void
IdealRateManagerMimoTest::CheckLastSelectedMode (WifiMode expectedMode)
{
  NS_TEST_ASSERT_MSG_EQ (m_txVector.GetMode (), expectedMode, "Last selected WifiMode " << m_txVector.GetMode () << " does not match expected WifiMode " << expectedMode);
}

void
IdealRateManagerMimoTest::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 100;

  NodeContainer wifiApNode, wifiStaNode;
  wifiApNode.Create (1);
  wifiStaNode.Create (1);

  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  WifiMacHelper mac;
  NetDeviceContainer apDevice;
  mac.SetType ("ns3::ApWifiMac");
  apDevice = wifi.Install (phy, mac, wifiApNode);

  NetDeviceContainer staDevice;
  mac.SetType ("ns3::StaWifiMac");
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevice, streamNumber);
  wifi.AssignStreams (staDevice, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (40.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxPsduBegin", MakeCallback (&IdealRateManagerMimoTest::TxCallback, this));


  // TX: 1 antenna
  Simulator::Schedule (Seconds (0.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 1, 1);
  // RX: 1 antenna
  Simulator::Schedule (Seconds (0.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 1, 1);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (1.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (1.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since both TX and RX support a single antenna
  Simulator::Schedule (Seconds (1.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be VHT-MCS 2 because of settings and distance between TX and RX
  Simulator::Schedule (Seconds (1.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs2 ());


  // TX: 1 antenna
  Simulator::Schedule (Seconds (1.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 1, 1);
  // RX: 2 antennas, but only supports 1 spatial stream
  Simulator::Schedule (Seconds (1.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 2, 1);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (2.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (2.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since both TX and RX support a single antenna
  Simulator::Schedule (Seconds (2.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be increased to VHT-MCS 3 because of RX diversity resulting in SNR improvement of about 3dB
  Simulator::Schedule (Seconds (2.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs3 ());


  // TX: 1 antenna
  Simulator::Schedule (Seconds (2.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 1, 1);
  // RX: 2 antennas, and supports 2 spatial streams
  Simulator::Schedule (Seconds (2.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 2, 2);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (3.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (3.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since TX supports a single antenna
  Simulator::Schedule (Seconds (3.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be as previously
  Simulator::Schedule (Seconds (3.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs3 ());


  // TX: 2 antennas, but only supports 1 spatial stream
  Simulator::Schedule (Seconds (3.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 2, 1);
  // RX: 1 antenna
  Simulator::Schedule (Seconds (3.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 1, 1);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (4.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (4.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since both TX and RX support a single antenna
  Simulator::Schedule (Seconds (4.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be VHT-MCS 2 because we do no longer have diversity in this scenario (more antennas at TX does not result in SNR improvement in AWGN channel)
  Simulator::Schedule (Seconds (4.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs2 ());


  // TX: 2 antennas, but only supports 1 spatial stream
  Simulator::Schedule (Seconds (4.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 2, 1);
  // RX: 2 antennas, but only supports 1 spatial stream
  Simulator::Schedule (Seconds (4.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 2, 1);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (5.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (5.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since both TX and RX support a single antenna
  Simulator::Schedule (Seconds (5.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be increased to VHT-MCS 3 because of RX diversity resulting in SNR improvement of about 3dB (more antennas at TX does not result in SNR improvement in AWGN channel)
  Simulator::Schedule (Seconds (5.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs3 ());


  // TX: 2 antennas, but only supports 1 spatial stream
  Simulator::Schedule (Seconds (5.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 2, 1);
  // RX: 2 antennas, and supports 2 spatial streams
  Simulator::Schedule (Seconds (5.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 2, 2);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (6.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (6.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since TX supports a single antenna
  Simulator::Schedule (Seconds (6.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be as previously
  Simulator::Schedule (Seconds (6.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs3 ());


  // TX: 2 antennas, and supports 2 spatial streams
  Simulator::Schedule (Seconds (6.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 2, 2);
  // RX: 1 antenna
  Simulator::Schedule (Seconds (6.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 1, 1);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (7.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (7.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since RX supports a single antenna
  Simulator::Schedule (Seconds (7.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be VHT-MCS 2 because we do no longer have diversity in this scenario (more antennas at TX does not result in SNR improvement in AWGN channel)
  Simulator::Schedule (Seconds (7.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs2 ());


  // TX: 2 antennas, and supports 2 spatial streams
  Simulator::Schedule (Seconds (7.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 2, 2);
  // RX: 2 antennas, but only supports 1 spatial stream
  Simulator::Schedule (Seconds (7.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 2, 1);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (8.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (8.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 1 since RX supports a single antenna
  Simulator::Schedule (Seconds (8.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  // Selected rate should be increased to VHT-MCS 3 because of RX diversity resulting in SNR improvement of about 3dB (more antennas at TX does not result in SNR improvement in AWGN channel)
  Simulator::Schedule (Seconds (8.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs3 ());


  // TX: 2 antennas, and supports 2 spatial streams
  Simulator::Schedule (Seconds (8.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 2, 2);
  // RX: 2 antennas, and supports 2 spatial streams
  Simulator::Schedule (Seconds (8.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 2, 2);
  // Send packets (2 times to get one feedback)
  Simulator::Schedule (Seconds (9.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (9.1), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  // Selected NSS should be 2 since both TX and RX support 2 antennas
  Simulator::Schedule (Seconds (9.2), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 2);
  // Selecte rate should be the same as without diversity, as it uses 2 spatial streams so there is no more benefits from diversity in AWGN channels
  Simulator::Schedule (Seconds (9.2), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs2 ());


  // Verify we can go back to initial situation
  Simulator::Schedule (Seconds (9.9), &IdealRateManagerMimoTest::SetApMimoSettings, this, 1, 1);
  Simulator::Schedule (Seconds (9.9), &IdealRateManagerMimoTest::SetStaMimoSettings, this, 1, 1);
  Simulator::Schedule (Seconds (10.0), &IdealRateManagerMimoTest::SendPacket, this, apDevice.Get (0), staDevice.Get (0)->GetAddress ());
  Simulator::Schedule (Seconds (10.1), &IdealRateManagerMimoTest::CheckLastSelectedNss, this, 1);
  Simulator::Schedule (Seconds (10.1), &IdealRateManagerMimoTest::CheckLastSelectedMode, this, VhtPhy::GetVhtMcs2 ());

  Simulator::Stop (Seconds (10.2));
  Simulator::Run ();
  Simulator::Destroy ();
}

//-----------------------------------------------------------------------------
/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Data rate verification test for MCSs of different RU sizes
 */
class HeRuMcsDataRateTestCase : public TestCase
{
public:
  HeRuMcsDataRateTestCase ();

private:
  /**
   * Compare the data rate computed for the provided combination with standard defined one.
   * \param ruType the RU type
   * \param mcs the modulation and coding scheme (as a string, e.g. HeMcs0)
   * \param nss the number of spatial streams
   * \param guardInterval the guard interval to use
   * \param expectedDataRate the expected data rate in 100 kbps units (minimum granularity in standard tables)
   * \returns true if data rates are the same, false otherwise
   */
  bool CheckDataRate (HeRu::RuType ruType, std::string mcs, uint8_t nss, uint16_t guardInterval, uint16_t expectedDataRate);
  void DoRun (void) override;
};

HeRuMcsDataRateTestCase::HeRuMcsDataRateTestCase ()
  : TestCase ("Check data rates for different RU types.")
{
}

bool
HeRuMcsDataRateTestCase::CheckDataRate (HeRu::RuType ruType, std::string mcs, uint8_t nss, uint16_t guardInterval, uint16_t expectedDataRate)
{
  uint16_t approxWidth = HeRu::GetBandwidth (ruType);
  WifiMode mode (mcs);
  uint64_t dataRate = round (mode.GetDataRate (approxWidth, guardInterval, nss) / 100000.0);
  NS_ABORT_MSG_IF (dataRate > 65535, "Rate is way too high");
  if (static_cast<uint16_t> (dataRate) != expectedDataRate)
    {
      std::cerr << "RU=" << ruType
                << " mode=" << mode
                << " Nss=" << +nss
                << " guardInterval=" << guardInterval
                << " expected=" << expectedDataRate << " x100kbps"
                << " computed=" << static_cast<uint16_t> (dataRate) << " x100kbps"
                << std::endl;
      return false;
    }
  return true;
}

void
HeRuMcsDataRateTestCase::DoRun (void)
{
  bool retval = true;

  //26-tone RU, browse over all MCSs, GIs and Nss's (up to 4, current max)
  retval = retval
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs0", 1,  800,   9)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs1", 1, 1600,  17)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs2", 1, 3200,  23)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs3", 1, 3200,  30)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs4", 2, 1600, 100)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs5", 3, 1600, 200)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs6", 4, 1600, 300)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs7", 4, 3200, 300)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs8", 4, 1600, 400)
      && CheckDataRate (HeRu::RU_26_TONE,  "HeMcs9", 4, 3200, 400)
      && CheckDataRate (HeRu::RU_26_TONE, "HeMcs10", 4, 1600, 500)
      && CheckDataRate (HeRu::RU_26_TONE, "HeMcs11", 4, 3200, 500);

  NS_TEST_EXPECT_MSG_EQ (retval, true, "26-tone RU  data rate verification for different MCSs, GIs, and Nss's failed");

  //Check other RU sizes
  retval = retval
      && CheckDataRate (   HeRu::RU_52_TONE, "HeMcs2", 1, 1600,   50)
      && CheckDataRate (  HeRu::RU_106_TONE, "HeMcs9", 1,  800,  500)
      && CheckDataRate (  HeRu::RU_242_TONE, "HeMcs5", 1, 1600,  650)
      && CheckDataRate (  HeRu::RU_484_TONE, "HeMcs3", 1, 1600,  650)
      && CheckDataRate (  HeRu::RU_996_TONE, "HeMcs5", 1, 3200, 2450)
      && CheckDataRate (HeRu::RU_2x996_TONE, "HeMcs3", 1, 3200, 2450);

  NS_TEST_EXPECT_MSG_EQ (retval, true, "Data rate verification for RUs above 52-tone RU (included) failed");
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Test Suite
 */
class WifiTestSuite : public TestSuite
{
public:
  WifiTestSuite ();
};

WifiTestSuite::WifiTestSuite ()
  : TestSuite ("wifi-devices", UNIT)
{
  AddTestCase (new WifiTest, TestCase::QUICK);
  AddTestCase (new QosUtilsIsOldPacketTest, TestCase::QUICK);
  AddTestCase (new InterferenceHelperSequenceTest, TestCase::QUICK); //Bug 991
  AddTestCase (new DcfImmediateAccessBroadcastTestCase, TestCase::QUICK);
  AddTestCase (new Bug730TestCase, TestCase::QUICK); //Bug 730
  AddTestCase (new QosFragmentationTestCase, TestCase::QUICK);
  AddTestCase (new SetChannelFrequencyTest, TestCase::QUICK);
  AddTestCase (new Bug2222TestCase, TestCase::QUICK); //Bug 2222
  AddTestCase (new Bug2843TestCase, TestCase::QUICK); //Bug 2843
  AddTestCase (new Bug2831TestCase, TestCase::QUICK); //Bug 2831
  AddTestCase (new StaWifiMacScanningTestCase, TestCase::QUICK); //Bug 2399
  AddTestCase (new Bug2470TestCase, TestCase::QUICK); //Bug 2470
  AddTestCase (new Issue40TestCase, TestCase::QUICK); //Issue #40
  AddTestCase (new Issue169TestCase, TestCase::QUICK); //Issue #169
  AddTestCase (new IdealRateManagerChannelWidthTest, TestCase::QUICK);
  AddTestCase (new IdealRateManagerMimoTest, TestCase::QUICK);
  AddTestCase (new HeRuMcsDataRateTestCase, TestCase::QUICK);
}

static WifiTestSuite g_wifiTestSuite; ///< the test suite
