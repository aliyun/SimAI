/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#include <cmath>
#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/node-list.h"
#include "ns3/mobility-model.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/wave-net-device.h"
#include "ns3/wave-mac-helper.h"
#include "ns3/wave-helper.h"

using namespace ns3;

/**
 * \ingroup wave
 * \defgroup wave-test wave module tests
 */


/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief This test case tests the channel coordination.
 * In particular, it checks the following:
 * - channel interval calculation including CCH Interval, SCH Interval,
 *   Guard Interval and Sync Interval
 * - current interval state for current time and future time
 * - channel coordination events notified at the correct time.
 */
class ChannelCoordinationTestCase : public TestCase
{
public:
  ChannelCoordinationTestCase (void);
  virtual ~ChannelCoordinationTestCase (void);

  // below three methods are used in CoordinationTestListener
  /**
   * Notify CCS start now function
   * \param duration the duration
   */
  void NotifyCchStartNow (Time duration);
  /**
   * Notify SCH start now function
   * \param duration the duration
   */
  void NotifySchStartNow (Time duration);
  /**
   * Notify guard start now function
   * \param duration the duration
   * \param inCchInterval the CCH interval
   */
  void NotifyGuardStartNow (Time duration, bool inCchInterval);
private:
  /**
   * Test interval after function
   * \param cchi the CCHI
   * \param schi the SCHI
   * \param guardi the guard
   */
  void TestIntervalAfter (bool cchi, bool schi, bool guardi);
  virtual void DoRun (void);
  Ptr<ChannelCoordinator> m_coordinator; ///< coordinator

};

/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief CoordinationTestListener is used to test channel coordination events
 */
class CoordinationTestListener : public ChannelCoordinationListener
{
public:
  /**
   * Constructor
   *
   * \param coordinatorTest channel coordination test case
   */
  CoordinationTestListener (ChannelCoordinationTestCase *coordinatorTest)
    : m_coordinatorTest (coordinatorTest)
  {
  }
  virtual ~CoordinationTestListener (void)
  {
  }
  virtual void NotifyCchSlotStart (Time duration)
  {
    m_coordinatorTest->NotifyCchStartNow (duration);
  }
  virtual void NotifySchSlotStart (Time duration)
  {
    m_coordinatorTest->NotifySchStartNow (duration);
  }
  virtual void NotifyGuardSlotStart (Time duration, bool cchi)
  {
    m_coordinatorTest->NotifyGuardStartNow (duration, cchi);
  }
  ChannelCoordinationTestCase *m_coordinatorTest; ///< coordinator test
};

ChannelCoordinationTestCase::ChannelCoordinationTestCase (void)
  : TestCase ("channel-coordination")
{
}
ChannelCoordinationTestCase::~ChannelCoordinationTestCase (void)
{
}

void
ChannelCoordinationTestCase::TestIntervalAfter (bool cchi, bool schi, bool guardi)
{
  uint32_t now = Now ().GetMilliSeconds ();
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->IsCchInterval (), cchi, "now is " << now  << "ms "
                         "check whether is CCH interval");
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->IsSchInterval (), schi, "now is " << now  << "ms "
                         "check whether is SCH interval");
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->IsGuardInterval (), guardi, "now is " << now  << "ms "
                         "check whether is Guard interval");
}
void
ChannelCoordinationTestCase::NotifyCchStartNow (Time duration)
{
  // this method shall be called at 4ms, 104ms, ... synci * n + guardi
  // synci is sync interval with default value 100ms
  // guardi is guard interval with default value 4ms
  // n is sequence number
  int64_t now = Now ().GetMilliSeconds ();
  int64_t synci = m_coordinator->GetSyncInterval ().GetMilliSeconds ();
  int64_t guardi = m_coordinator->GetGuardInterval ().GetMilliSeconds ();
  bool test = (((now - guardi) % synci) == 0);
  NS_TEST_EXPECT_MSG_EQ (test, true, "the time of now shall be synci * n + guardi");

  // besides that, the argument duration shall be cchi - guardi
  Time d = m_coordinator->GetCchInterval () - m_coordinator->GetGuardInterval ();
  NS_TEST_EXPECT_MSG_EQ ((duration == d), true, "the duration shall be cchi - guardi");
}

void
ChannelCoordinationTestCase::NotifySchStartNow (Time duration)
{
  // this method shall be called at 54ms, 154ms, ... synci * n + cchi + guardi
  // synci is sync interval with default value 100ms
  // cchi is CCH interval with default value 50ms
  // guardi is guard interval with default value 4ms
  // n is sequence number
  int64_t now = Now ().GetMilliSeconds ();
  int64_t synci = m_coordinator->GetSyncInterval ().GetMilliSeconds ();
  int64_t cchi = m_coordinator->GetCchInterval ().GetMilliSeconds ();
  int64_t guardi = m_coordinator->GetGuardInterval ().GetMilliSeconds ();
  bool test = ((now - guardi - cchi) % synci == 0);
  NS_TEST_EXPECT_MSG_EQ (test, true, "the time of now shall be synci * n + cchi + guardi");

  // besides that, the argument duration shall be schi - guardi
  Time d = m_coordinator->GetSchInterval () - m_coordinator->GetGuardInterval ();
  NS_TEST_EXPECT_MSG_EQ ((duration == d), true, "the duration shall be schi - guardi");
}

void
ChannelCoordinationTestCase::NotifyGuardStartNow (Time duration, bool inCchInterval)
{
  int64_t now = Now ().GetMilliSeconds ();
  int64_t sync = m_coordinator->GetSyncInterval ().GetMilliSeconds ();
  int64_t cchi = m_coordinator->GetCchInterval ().GetMilliSeconds ();
  bool test = false;
  if (inCchInterval)
    {
      // if cchi, this method will be called at 0ms, 100ms, sync * n
      test = ((now % sync) == 0);
      NS_TEST_EXPECT_MSG_EQ (test, true, "the time of now shall be sync * n");
    }
  else
    {
      // if schi, this method will be called at 50ms, 150ms, sync * n + cchi
      test = (((now - cchi) % sync) == 0);
      NS_TEST_EXPECT_MSG_EQ (test, true, "the time of now shall be sync * n");
    }
  // the duration shall be guardi
  test = (duration == m_coordinator->GetGuardInterval ());
  NS_TEST_EXPECT_MSG_EQ (test, true, "the duration shall be guard interval");
}

void
ChannelCoordinationTestCase::DoRun ()
{
  // first test configure method
  m_coordinator = CreateObject<ChannelCoordinator> ();
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->GetCchInterval (), MilliSeconds (50), "normally CCH interval is 50ms");
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->GetSchInterval (), MilliSeconds (50), "normally SCH interval is 50ms");
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->GetSyncInterval (), MilliSeconds (100), "normally Sync interval is 50ms");
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->GetGuardInterval (), MilliSeconds (4), "normally Guard interval is 50ms");
  m_coordinator->SetCchInterval (MilliSeconds (40));
  m_coordinator->SetSchInterval (MilliSeconds (60));
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->IsValidConfig (), true, "valid configuration of channel intervals");
  m_coordinator->SetCchInterval (MilliSeconds (40));
  m_coordinator->SetSchInterval (MilliSeconds (50));
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->IsValidConfig (), false, "invalid configuration of channel intervals");
  m_coordinator->SetGuardInterval (MilliSeconds (50));
  m_coordinator->SetCchInterval (MilliSeconds (40));
  m_coordinator->SetSchInterval (MilliSeconds (50));
  NS_TEST_EXPECT_MSG_EQ (m_coordinator->IsValidConfig (), false, "invalid configuration of channel intervals");

  // second test member method
  m_coordinator = CreateObject<ChannelCoordinator> ();
  Simulator::Schedule (MilliSeconds (0), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (1), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (3), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (4), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, false);
  Simulator::Schedule (MilliSeconds (5), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, false);
  Simulator::Schedule (MilliSeconds (50), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, true);
  Simulator::Schedule (MilliSeconds (51), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, true);
  Simulator::Schedule (MilliSeconds (53), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, true);
  Simulator::Schedule (MilliSeconds (54), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, false);
  Simulator::Schedule (MilliSeconds (55), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, false);
  Simulator::Schedule (MilliSeconds (100), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (200), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (201), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (203), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Schedule (MilliSeconds (204), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, false);
  Simulator::Schedule (MilliSeconds (205), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, false);
  Simulator::Schedule (MilliSeconds (250), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, true);
  Simulator::Schedule (MilliSeconds (251), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, true);
  Simulator::Schedule (MilliSeconds (253), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, true);
  Simulator::Schedule (MilliSeconds (254), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, false);
  Simulator::Schedule (MilliSeconds (255), &ChannelCoordinationTestCase::TestIntervalAfter, this, false, true, false);
  Simulator::Schedule (MilliSeconds (300), &ChannelCoordinationTestCase::TestIntervalAfter, this, true, false, true);
  Simulator::Stop (Seconds (1.0));
  Simulator::Run ();
  Simulator::Destroy ();

  m_coordinator = CreateObject<ChannelCoordinator> ();
  // third test channel coordination events
  Ptr<CoordinationTestListener> ptr = Create<CoordinationTestListener> (this);
  m_coordinator->RegisterListener (ptr);
  Simulator::Stop (Seconds (100.0));
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief Test Case Helper
 */
class TestCaseHelper
{
public:
  /**
   * Create WAVE device function
   * \param nodesNumber the number of nodes
   * \returns the collection of nodes
   */
  static NetDeviceContainer  CreatWaveDevice (uint32_t nodesNumber = 2);
};

#define PI 3.14159265

NetDeviceContainer
TestCaseHelper::CreatWaveDevice (uint32_t nodesNumber)
{
  NodeContainer nodes;
  nodes.Create (nodesNumber);

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);
  // this is a circle with radius 10
  Ptr<MobilityModel> model = NodeList::GetNode (0)->GetObject<MobilityModel> ();
  model->SetPosition (Vector (0, 0, 0));
  for (uint32_t n = 1; n != nodesNumber; ++n)
    {
      double angle = (PI / 180) * (360 / (nodesNumber - 1) * n);
      double x = cos (angle) * 10;
      double y = sin (angle) * 10;
      Ptr<MobilityModel> model = NodeList::GetNode (n)->GetObject<MobilityModel> ();
      model->SetPosition (Vector (x, y,0));
    }

  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWavePhyHelper wifiPhy =  YansWavePhyHelper::Default ();
  wifiPhy.SetChannel (wifiChannel.Create ());
  QosWaveMacHelper waveMac = QosWaveMacHelper::Default ();
  WaveHelper waveHelper = WaveHelper::Default ();
  NetDeviceContainer devices = waveHelper.Install (wifiPhy, waveMac, nodes);
  return devices;
}

/**
 *  route packets or frames in different approaches
 *  see 1609.4-2010 chapter 5.3.4
 */
class ChannelRoutingTestCase : public TestCase
{
public:
  ChannelRoutingTestCase (void);
  virtual ~ChannelRoutingTestCase (void);

  /**
   * Send IP-based packets
   *
   * \param shouldSucceed is used to check whether packet sent should be successful.
   * \param ipv6 is IPv6?
   */
  void SendIp (bool shouldSucceed, bool ipv6);
  /**
   * Send WSMP or other packets
   * \param shouldSucceed is used to check whether packet sent should be successful.
   * \param txInfo transmit info
   */
  void SendWsmp (bool shouldSucceed, const TxInfo &txInfo);
  /**
   * Send VSA management frames
   * \param shouldSucceed is used to check whether packet sent should be successful.
   * \param vsaInfo VSA info
   */
  void SendWsa (bool shouldSucceed, const VsaInfo &vsaInfo);

private:
  virtual void DoRun (void);
  /**
   * Receive function
   * \param dev the device
   * \param pkt the packet
   * \param mode the mode
   * \param sender the sender address
   * \returns true if successful
   */
  bool Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender);
  /**
   * Receive VSA function
   * \param pkt the packet
   * \param address the address
   * \returns true if successful
   */
  bool ReceiveVsa (Ptr<const Packet> pkt,const Address & address, uint32_t, uint32_t);

  Ptr<WaveNetDevice>  m_sender; ///< sender
};

ChannelRoutingTestCase::ChannelRoutingTestCase (void)
  : TestCase ("channel-routing")
{

}
ChannelRoutingTestCase::~ChannelRoutingTestCase (void)
{

}
void
ChannelRoutingTestCase::SendWsmp (bool shouldSucceed, const TxInfo &txInfo)
{
  Ptr<Packet> packet = Create<Packet> (100);
  const Address dest = Mac48Address::GetBroadcast ();
  uint16_t protocol = 0x80dd; // any number is OK even ipv4 and ipv6
  bool result = m_sender->SendX (packet, dest, protocol, txInfo);
  NS_TEST_EXPECT_MSG_EQ (result, shouldSucceed, "test SendWsmp method error");
}
void
ChannelRoutingTestCase::SendIp (bool shouldSucceed, bool ipv6)
{
  Ptr<Packet> packet = Create<Packet> (100);
  const Address dest = Mac48Address::GetBroadcast ();
  const static uint16_t IPv4_PROT_NUMBER = 0x0800;
  const static uint16_t IPv6_PROT_NUMBER = 0x86DD;
  uint16_t protocol = ipv6 ? IPv6_PROT_NUMBER : IPv4_PROT_NUMBER;
  bool result = m_sender->Send (packet, dest, protocol);
  NS_TEST_EXPECT_MSG_EQ (result, shouldSucceed, "test SendIp method error");
}

void
ChannelRoutingTestCase::SendWsa (bool shouldSucceed, const VsaInfo &vsaInfo)
{
  bool result = m_sender->StartVsa (vsaInfo);
  NS_TEST_EXPECT_MSG_EQ (result, shouldSucceed, "test SendWsa method error");
}

void
ChannelRoutingTestCase::DoRun ()
{
  //  check SendX method for WSMP packets
  {
    NetDeviceContainer devices = TestCaseHelper::CreatWaveDevice (1);
    m_sender = DynamicCast<WaveNetDevice> (devices.Get (0));

    Simulator::Schedule (Seconds (0.1), &ChannelRoutingTestCase::SendWsmp, this, true, TxInfo (CCH));
    Simulator::Schedule (Seconds (0.1), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH1));
    Simulator::Schedule (Seconds (0.1), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH2));

    const SchInfo schInfo = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (0.2), &WaveNetDevice::StartSch, m_sender, schInfo);

    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, true, TxInfo (CCH));
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, true, TxInfo (SCH1));
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH2));
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH3));
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH4));
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH5));
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH6));

    // invalid channel number
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (0));
    // invalid user priority
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (CCH, 8));
    // invalid tx parameters
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (CCH, 7, WifiMode ("OfdmRate6Mbps"), WIFI_PREAMBLE_LONG, 7));
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (CCH, 7, WifiMode ("OfdmRate3MbpsBW10MHz"), WIFI_PREAMBLE_LONG, 10));
    // valid tx parameters
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsmp, this, true, TxInfo (CCH, 7, WifiMode ("OfdmRate3MbpsBW10MHz"), WIFI_PREAMBLE_LONG, 7));
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsmp, this, true, TxInfo (CCH, 7, WifiMode (), WIFI_PREAMBLE_LONG, 8));

    // release channel access at 0.6s
    Simulator::Schedule (Seconds (0.5), &WaveNetDevice::StopSch, m_sender, SCH1);

    // the packet will be dropped because channel access is not assigned again
    Simulator::Schedule (Seconds (0.6), &ChannelRoutingTestCase::SendWsmp, this, true, TxInfo (CCH));
    Simulator::Schedule (Seconds (0.6), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH1));
    Simulator::Schedule (Seconds (0.6), &ChannelRoutingTestCase::SendWsmp, this, false, TxInfo (SCH2));

    Simulator::Stop (Seconds (1.0));
    Simulator::Run ();
    Simulator::Destroy ();
  }

  // check Send method for IP-based packets
  {
    NetDeviceContainer devices = TestCaseHelper::CreatWaveDevice (1);
    m_sender = DynamicCast<WaveNetDevice> (devices.Get (0));

    bool ipv6 = true, ipv4 = false;
    Simulator::Schedule (Seconds (0.1), &ChannelRoutingTestCase::SendIp, this, false, ipv6);
    Simulator::Schedule (Seconds (0.1), &ChannelRoutingTestCase::SendIp, this, false, ipv4);

    const SchInfo schInfo = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (0.2), &WaveNetDevice::StartSch, m_sender, schInfo);

    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendIp, this, false, ipv6);
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendIp, this, false, ipv4);

    TxProfile txProfile = TxProfile (SCH1);
    Simulator::Schedule (Seconds (0.4), &WaveNetDevice::RegisterTxProfile, m_sender, txProfile);

    Simulator::Schedule (Seconds (0.5), &ChannelRoutingTestCase::SendIp, this, true, ipv6);
    Simulator::Schedule (Seconds (0.5), &ChannelRoutingTestCase::SendIp, this, true, ipv4);

    // unregister txprofile
    Simulator::Schedule (Seconds (0.5), &WaveNetDevice::DeleteTxProfile, m_sender,SCH1);

    Simulator::Schedule (Seconds (0.6), &ChannelRoutingTestCase::SendIp, this, false, ipv6);
    Simulator::Schedule (Seconds (0.6), &ChannelRoutingTestCase::SendIp, this, false, ipv4);

    // release channel access
    // mac entities have no channel resource even phy device has ability to send
    Simulator::Schedule (Seconds (0.7),&WaveNetDevice::StopSch, m_sender, SCH1);

    Simulator::Schedule (Seconds (0.8), &ChannelRoutingTestCase::SendIp, this, false, ipv6);
    Simulator::Schedule (Seconds (0.8), &ChannelRoutingTestCase::SendIp, this, false, ipv4);

    Simulator::Stop (Seconds (1.0));
    Simulator::Run ();
    Simulator::Destroy ();
  }

  // check StartVsa method for WSA management frames
  {
    NetDeviceContainer devices = TestCaseHelper::CreatWaveDevice (1);
    m_sender = DynamicCast<WaveNetDevice> (devices.Get (0));

    Ptr<Packet> packet = Create<Packet> (100);
    const Mac48Address dest = Mac48Address::GetBroadcast ();
    VsaInfo vsaInfo = VsaInfo (dest, OrganizationIdentifier (), 3, packet, SCH1, 50, VSA_TRANSMIT_IN_BOTHI);
    Simulator::Schedule (Seconds (0.1), &ChannelRoutingTestCase::SendWsa, this, false, vsaInfo);

    vsaInfo.channelNumber = 0;
    Simulator::Schedule (Seconds (0.2), &ChannelRoutingTestCase::SendWsa, this, false, vsaInfo);

    vsaInfo.channelNumber = CCH;
    Simulator::Schedule (Seconds (0.3), &ChannelRoutingTestCase::SendWsa, this, true, vsaInfo);
    Simulator::Schedule (Seconds (0.39), &WaveNetDevice::StopVsa, m_sender, CCH);

    const SchInfo schInfo = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (0.4), &WaveNetDevice::StartSch, m_sender, schInfo);
    vsaInfo.channelNumber = CCH;
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsa, this, true, vsaInfo);
    vsaInfo.channelNumber = SCH1;
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsa, this, true, vsaInfo);
    vsaInfo.channelNumber = SCH2;
    Simulator::Schedule (Seconds (0.4), &ChannelRoutingTestCase::SendWsa, this, false, vsaInfo);

    Simulator::Schedule (Seconds (0.49), &WaveNetDevice::StopVsa, m_sender, CCH);
    Simulator::Schedule (Seconds (0.49), &WaveNetDevice::StopVsa, m_sender, SCH1);
    Simulator::Schedule (Seconds (0.49),&WaveNetDevice::StopSch, m_sender, SCH1);

    vsaInfo.channelNumber = CCH;
    Simulator::Schedule (Seconds (0.5), &ChannelRoutingTestCase::SendWsa, this, true, vsaInfo);
    vsaInfo.channelNumber = SCH1;
    Simulator::Schedule (Seconds (0.5), &ChannelRoutingTestCase::SendWsa, this, false, vsaInfo);
    vsaInfo.channelNumber = SCH2;
    Simulator::Schedule (Seconds (0.5), &ChannelRoutingTestCase::SendWsa, this, false, vsaInfo);

    Simulator::Stop (Seconds (1.0));
    Simulator::Run ();
    Simulator::Destroy ();
  }
}

/**
 * This test case tests channel access assignments which is done by
 * StartSch and StopSch method of WaveNetDevice.
 * channel access assignments include ContinuousAccess, ExtendedAccess,
 * and AlternatingAccess.
 * The results of this test case depend on the implementation of ChannelScheduler
 * In this case, the results depend on class "DefaultChannelScheduler".
 */
class ChannelAccessTestCase : public TestCase
{
public:
  ChannelAccessTestCase (void);
  virtual ~ChannelAccessTestCase (void);
private:
  /**
   * Test continuous function
   * \param info the schedule info
   * \param shouldSucceed true if it should succeed
   */
  void TestContinuous (SchInfo &info, bool shouldSucceed);
  /**
   * Test continuous after function
   * \param channelNumber the channel number
   * \param isAccessAssigned true if access assigned
   */
  void TestContinuousAfter (uint32_t channelNumber, bool isAccessAssigned);
  /**
   * Test extended function
   * \param info the schedule info
   * \param shouldSucceed true if it should succeed
   */
  void TestExtended (SchInfo &info, bool shouldSucceed);
  /**
   * Test extended after function
   * \param channelNumber the channel number
   * \param isAccessAssigned true if access assigned
   */
  void TestExtendedAfter (uint32_t channelNumber, bool isAccessAssigned);
  /**
   * Test aternating function
   * \param info the schedule info
   * \param shouldSucceed true if it should succeed
   */
  void TestAlternating (SchInfo &info, bool shouldSucceed);
  /**
   * Test alternating after function
   * \param channelNumber the channel number
   * \param isAccessAssigned true if access assigned
   */
  void TestAlternatingAfter (uint32_t channelNumber, bool isAccessAssigned);

  /**
   * Send X function
   * \param channel the channel number
   * \param receiverId
   */
  void SendX (uint32_t channel, uint32_t receiverId);
  /**
   * Receive function
   * \param dev the device
   * \param pkt the packet
   * \param mode the mode
   * \param sender the sender address
   * \returns true if successful
   */
  bool Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender);

  virtual void DoRun (void);

  NetDeviceContainer m_devices; ///< the devices
  Ptr<WaveNetDevice>  m_sender; ///< sender
  uint32_t m_received; ///< received
};

ChannelAccessTestCase::ChannelAccessTestCase (void)
  : TestCase ("channel-access")
{
}
ChannelAccessTestCase::~ChannelAccessTestCase (void)
{

}
void
ChannelAccessTestCase::TestContinuous (SchInfo &info, bool shouldSucceed)
{
  bool result = m_sender->StartSch (info);
  NS_TEST_EXPECT_MSG_EQ (result, shouldSucceed, "TestContinuous fail at " << Now ().As (Time::S));
}
void
ChannelAccessTestCase::TestContinuousAfter (uint32_t channelNumber, bool isAccessAssigned)
{
  bool result = m_sender->GetChannelScheduler ()->IsContinuousAccessAssigned (channelNumber);
  NS_TEST_EXPECT_MSG_EQ (result, isAccessAssigned, "TestContinuousAfter fail at " << Now ().As (Time::S));
}
void
ChannelAccessTestCase::TestExtended (SchInfo &info, bool shouldSucceed)
{
  bool result = m_sender->StartSch (info);
  NS_TEST_EXPECT_MSG_EQ (result, shouldSucceed, "TestExtended fail at " << Now ().As (Time::S));
}
void
ChannelAccessTestCase::TestExtendedAfter (uint32_t channelNumber, bool isAccessAssigned)
{
  bool result = m_sender->GetChannelScheduler ()->IsExtendedAccessAssigned (channelNumber);
  NS_TEST_EXPECT_MSG_EQ (result, isAccessAssigned, "TestExtendedAfter fail at " << Now ().As (Time::S));
}

void
ChannelAccessTestCase::TestAlternating (SchInfo &info, bool shouldSucceed)
{
  bool result = m_sender->StartSch (info);
  NS_TEST_EXPECT_MSG_EQ (result, shouldSucceed, "TestAlternating fail at " << Now ().As (Time::S));
}
void
ChannelAccessTestCase::TestAlternatingAfter (uint32_t channelNumber, bool isAccessAssigned)
{
  bool result = m_sender->GetChannelScheduler ()->IsAlternatingAccessAssigned (channelNumber);
  NS_TEST_EXPECT_MSG_EQ (result, isAccessAssigned, "TestAlternating fail at " << Now ().As (Time::S));
}

void
ChannelAccessTestCase::SendX  (uint32_t channel, uint32_t receiverId)
{
  const static uint16_t WSMP_PROT_NUMBER = 0x88DC;
  const Mac48Address dest = Mac48Address::GetBroadcast ();
  const TxInfo txInfo = TxInfo (channel);

  uint8_t *data = new uint8_t [112];
  data [0] = (receiverId >> 24) & 0xFF;
  data [1] = (receiverId >> 16) & 0xFF;
  data [2] = (receiverId >> 8) & 0xFF;
  data [3] = (receiverId >> 0) & 0xFF;

  uint64_t ts = Simulator::Now ().GetTimeStep ();
  data [4] = (ts >> 56) & 0xFF;
  data [5] = (ts >> 48) & 0xFF;
  data [6] = (ts >> 40) & 0xFF;
  data [7] = (ts >> 32) & 0xFF;
  data [8] = (ts >> 24) & 0xFF;
  data [9] = (ts >> 16) & 0xFF;
  data [10] = (ts >> 8) & 0xFF;
  data [11] = (ts >> 0) & 0xFF;

  Ptr<Packet> p = Create<Packet> (data, 112);

  m_sender->SendX  (p, dest, WSMP_PROT_NUMBER, txInfo);

  delete [] data;
}

bool
ChannelAccessTestCase::Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender)
{
  uint8_t *data = new uint8_t [pkt->GetSize ()];
  pkt->CopyData(data, pkt->GetSize ());

  uint32_t seq = data [0];
  seq <<= 8;
  seq |= data [1];
  seq <<= 8;
  seq |= data [2];
  seq <<= 8;
  seq |= data [3];

  uint64_t ts = data [4];
  ts <<= 8;
  ts |= data [5];
  ts <<= 8;
  ts |= data [6];
  ts <<= 8;
  ts |= data [7];
  ts <<= 8;
  ts |= data [8];
  ts <<= 8;
  ts |= data [9];
  ts <<= 8;
  ts |= data [10];
  ts <<= 8;
  ts |= data [11];
  Time sendTime = TimeStep (ts);

  delete [] data;

//  SeqTsHeader seqTs;
//  ConstCast<Packet> (pkt)->RemoveHeader (seqTs);
  uint32_t curNodeId =  dev->GetNode ()->GetId ();
  NS_TEST_EXPECT_MSG_EQ (curNodeId, seq, "fail to assign channel access");
  m_received++;
  return true;
}

void
ChannelAccessTestCase::DoRun ()
{
  // test ContinuousAccess in the sender side
  {
    m_devices = TestCaseHelper::CreatWaveDevice (1);
    m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));

    // there is no need for assigning CCH continuous access.
    SchInfo info = SchInfo (CCH, false, EXTENDED_CONTINUOUS);
    Simulator::Schedule (Seconds (1), &ChannelAccessTestCase::TestContinuous, this, info, false);

    info = SchInfo (SCH1, false, EXTENDED_CONTINUOUS);
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestContinuous, this, info, true);

    // BE ATTENTION !!!
    // because channel access is assigned in non-immediate mode, the first CCH Interval will be
    // the wait time with DefaultCchAccess assigned, thus there is no ContinuousAccess assigned.
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestContinuousAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.01), &ChannelAccessTestCase::TestContinuousAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.049), &ChannelAccessTestCase::TestContinuousAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.05), &ChannelAccessTestCase::TestContinuousAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.051), &ChannelAccessTestCase::TestContinuousAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.99), &ChannelAccessTestCase::TestContinuousAfter, this, SCH1, true);

    // it's OK to assign same access again,
    Simulator::Schedule (Seconds (3), &ChannelAccessTestCase::TestContinuous, this, info, true);
    // fail to assign continuous access for other SCH if current channel is assigned
    info = SchInfo (SCH2, false, EXTENDED_CONTINUOUS);
    Simulator::Schedule (Seconds (4), &ChannelAccessTestCase::TestContinuous, this, info, false);

    // then we release channel access at 0.5s
    Simulator::Schedule (Seconds (5), &WaveNetDevice::StopSch, m_sender, SCH1);

    info = SchInfo (SCH2, false, EXTENDED_CONTINUOUS);
    // succeed to assign access for other SCH is previous SCH access is released
    Simulator::Schedule (Seconds (6), &ChannelAccessTestCase::TestContinuous, this, info, true);

    Simulator::Stop (Seconds (7.0));
    Simulator::Run ();
    Simulator::Destroy ();
  }

  // test ContinuousAccess again in the receiver side
  {
    m_devices = TestCaseHelper::CreatWaveDevice (8);
    m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));
    m_received = 0;

    for (uint32_t i = 1; i != 8; ++i)
      {
        Ptr<WaveNetDevice> device = DynamicCast<WaveNetDevice> (m_devices.Get (i));
        device->SetReceiveCallback (MakeCallback (&ChannelAccessTestCase::Receive, this));

        // at 0s, receivers are assigned ContinuousAccess from CCH, SCH1 to SCH6
        static std::vector<uint32_t> WaveChannels = ChannelManager::GetWaveChannels ();
        uint32_t channel = WaveChannels[i - 1];
        const SchInfo info = SchInfo (channel, false, EXTENDED_CONTINUOUS);
        Simulator::Schedule (Seconds (0), &WaveNetDevice::StartSch, device, info);
      }

    // at 0s, the sender is assigned DefaultCchAccess, so only node-1 can receive packets.
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, CCH, 1);
    // if receivers assigned for SCH access can receive packets, there shall be crashed
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH1, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    // at 1s, the sender is assigned ContinuousAccess for SCH1, so only node-2 can receive packets.
    SchInfo info = SchInfo (SCH1, false, EXTENDED_CONTINUOUS);
    Simulator::Schedule (Seconds (1), &WaveNetDevice::StartSch, m_sender, info);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH1, 2);
    // other channel access cannot receive packets
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, CCH, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    Simulator::Stop (Seconds (10.0));
    Simulator::Run ();
    Simulator::Destroy ();

    NS_TEST_EXPECT_MSG_EQ (m_received, 2, "test ContinuousAccess fail in receive side");
  }

  // test ExtendedAccess in the sender side
  {
    m_devices = TestCaseHelper::CreatWaveDevice (1);
    m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));

    // there is no need for assigning CCH extended access.
    SchInfo info = SchInfo (CCH, false, 10);
    Simulator::Schedule (Seconds (1), &ChannelAccessTestCase::TestExtended, this, info, false);

    info = SchInfo (SCH1, false, 10);
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestExtended, this, info, true);
    // succeed because request for extends 8 can be fulfilled by previous extends 10..
    info = SchInfo (SCH1, false, 8);
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestExtended, this, info, true);
    // fail because request for extends 12 cannot be fulfilled by previous extends 10..
    info = SchInfo (SCH1, false, 12);
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestExtended, this, info, false);

    // BE ATTENTION !!!
    // because channel access is assigned in non-immediate mode, the first CCH Interval will be
    // the wait time with DefaultCchAccess assigned, while there is no ExtendedAccess assigned.
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.01), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.049), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.05), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (2.051), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.99), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, true);

    // the end of extended access is (2s +  100ms + 100ms * 10) = 3.1s
    Simulator::Schedule (Seconds (3), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (3.1), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (3.2), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, false);
    Simulator::Schedule (Seconds (3.3), &ChannelAccessTestCase::TestExtendedAfter, this, SCH1, false);

    // succeed to assign extended access for other SCH since previous extended access is released automatically
    info = SchInfo (SCH2, false, 10);
    Simulator::Schedule (Seconds (4), &ChannelAccessTestCase::TestExtended, this, info, true);

    // stop it at 5s even the end of extended access is (4s + 100ms + 100ms * 10) = 5.1s
    Simulator::Schedule (Seconds (5), &WaveNetDevice::StopSch, m_sender, SCH2);

    Simulator::Schedule (Seconds (5), &ChannelAccessTestCase::TestExtendedAfter, this, SCH2, false);
    Simulator::Schedule (Seconds (5.1), &ChannelAccessTestCase::TestExtendedAfter, this, SCH2, false);
    Simulator::Schedule (Seconds (5.2), &ChannelAccessTestCase::TestExtendedAfter, this, SCH2, false);

    Simulator::Stop (Seconds (6.0));
    Simulator::Run ();
    Simulator::Destroy ();
  }

  // test ExtendedAccess again in the receiver side
  {
    m_devices = TestCaseHelper::CreatWaveDevice (8);
    m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));
    m_received = 0;

    for (uint32_t i = 1; i != 8; ++i)
      {
        Ptr<WaveNetDevice> device = DynamicCast<WaveNetDevice> (m_devices.Get (i));
        device->SetReceiveCallback (MakeCallback (&ChannelAccessTestCase::Receive, this));

        // at 0s, receivers are assigned ContinuosAccess from CCH, SCH1 to SCH6
        static std::vector<uint32_t> WaveChannels = ChannelManager::GetWaveChannels ();
        uint32_t channel = WaveChannels[i - 1];
        const SchInfo info = SchInfo (channel, false, EXTENDED_CONTINUOUS);
        Simulator::Schedule (Seconds (0), &WaveNetDevice::StartSch, device, info);
      }

    // at 0s, the sender is assigned DefaultCchAccess, so only node-1 can receive packets.
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, CCH, 1);
    // if receivers assigned for SCH access can receive packets, there shall be crashed
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH1, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    // at 1s, the sender is assigned ExtendedAccess for SCH1 with extends 10,
    //, so only node-2 can receive packets from 1s - 2.1s ( 1s + 100ms + 100ms * 10)
    SchInfo info = SchInfo (SCH1, false, 10);
    Simulator::Schedule (Seconds (1), &WaveNetDevice::StartSch, m_sender, info);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH1, 2);
    // other channel access cannot receive packets
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, CCH, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);
    // at 2.2s the node-2 cannot receive this packet because of extended access released in node-0
    // but sended is assigned DefaultCchAccess again, thus node-1 can receive broadcasted packets.
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, CCH, 1);
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, SCH1, 0);
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (2.2), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    Simulator::Stop (Seconds (10.0));
    Simulator::Run ();
    Simulator::Destroy ();

    NS_TEST_EXPECT_MSG_EQ (m_received, 3, "test ExtendedAccess fail in receive side");
  }

  // test AlternatingAccess in the sender side
  {
    m_devices = TestCaseHelper::CreatWaveDevice (1);
    m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));

    // there is no need for assigning CCH alternating  access.
    SchInfo info = SchInfo (CCH, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (1), &ChannelAccessTestCase::TestAlternating, this, info, false);

    info = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestAlternating, this, info, true);

    // BE ATTENTION !!!
    // No matter whether channel access is assigned in immediate mode or non-immediate mode,
    // the channel access will assigned immediately which is different from the test results in
    // ExtendedAccess assignment and ContinuousAccess assignment.
    Simulator::Schedule (Seconds (2), &ChannelAccessTestCase::TestAlternatingAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.01), &ChannelAccessTestCase::TestAlternatingAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.049), &ChannelAccessTestCase::TestAlternatingAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.05), &ChannelAccessTestCase::TestAlternatingAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.051), &ChannelAccessTestCase::TestAlternatingAfter, this, SCH1, true);
    Simulator::Schedule (Seconds (2.99), &ChannelAccessTestCase::TestAlternatingAfter, this, SCH1, true);

    Simulator::Schedule (Seconds (3), &ChannelAccessTestCase::TestAlternating, this, info, true);
    info = SchInfo (SCH2, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (3), &ChannelAccessTestCase::TestAlternating, this, info, false);
    info = SchInfo (0, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (3), &ChannelAccessTestCase::TestAlternating, this, info, false);

    // then we release channel access at 0.5s
    Simulator::Schedule (Seconds (4), &WaveNetDevice::StopSch, m_sender, SCH1);

    info = SchInfo (SCH2, false, EXTENDED_ALTERNATING);
    // succeed to assign access for other SCH is previous SCH access is released
    Simulator::Schedule (Seconds (5), &ChannelAccessTestCase::TestAlternating, this, info, true);

    Simulator::Stop (Seconds (6.0));
    Simulator::Run ();
    Simulator::Destroy ();
  }

  // test AlternatingAccess again in the receiver side
  {
    m_devices = TestCaseHelper::CreatWaveDevice (8);
    m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));
    m_received = 0;

    for (uint32_t i = 1; i != 8; ++i)
      {
        Ptr<WaveNetDevice> device = DynamicCast<WaveNetDevice> (m_devices.Get (i));
        device->SetReceiveCallback (MakeCallback (&ChannelAccessTestCase::Receive, this));

        // at 0s, receivers are assigned ContinuosAccess from CCH, SCH1 to SCH6
        static std::vector<uint32_t> WaveChannels = ChannelManager::GetWaveChannels ();
        uint32_t channel = WaveChannels[i - 1];
        const SchInfo info = SchInfo (channel, false, EXTENDED_CONTINUOUS);
        Simulator::Schedule (Seconds (0), &WaveNetDevice::StartSch, device, info);
      }

    // at 0s, the sender is assigned DefaultCchAccess, so only node-1 can receive packets.
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, CCH, 1);
    // if receivers assigned for SCH access can receive packets, there shall be crashed
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH1, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (0.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    // at 1s, the sender is assigned ContinuosAccess for SCH1, so only node-2 can receive packets.
    SchInfo info = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (1), &WaveNetDevice::StartSch, m_sender, info);
    // node-1 (assigned CCH access) and node-2 (assigned SCH1 access) can receive packets
    // in different channel interval
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH1, 2);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, CCH, 1);
    // other channel access cannot receive packets
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (1.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    Simulator::Schedule (Seconds (2), &WaveNetDevice::StopSch, m_sender, SCH1);
    // if ContinuousAccess for SCH1 is released, node-2 cannot receive packets again
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, CCH, 1);
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, SCH1, 0);
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, SCH2, 0);
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, SCH3, 0);
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, SCH4, 0);
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, SCH5, 0);
    Simulator::Schedule (Seconds (2.1), &ChannelAccessTestCase::SendX, this, SCH6, 0);

    Simulator::Stop (Seconds (10.0));
    Simulator::Run ();
    Simulator::Destroy ();

    NS_TEST_EXPECT_MSG_EQ (m_received, 4, "test AlternatingAccess fail in receive side");
  }
}

/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief The Annex C of  IEEE 1609.4 : "Avoiding transmission at scheduled guard intervals"
 * This feature is implemented in WaveMacLow::StartTransmission method
 */
class AnnexC_TestCase : public TestCase
{
public:
  AnnexC_TestCase ();
  virtual ~AnnexC_TestCase ();
private:
  virtual void DoRun (void);

  /**
   * Send packet function
   * \param packetSize the packet size
   * \param txInfo the transmit info
   * \param sequence the sequence
   */
  void SendPacket  (uint32_t packetSize, const TxInfo & txInfo, uint32_t sequence);
  /**
   * Receive function
   * \param dev the device
   * \param pkt the packet
   * \param mode the mode
   * \param sender the sender address
   * \returns true if successful
   */
  bool Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender);

  NetDeviceContainer m_devices; ///< devices
  Ptr<WaveNetDevice>  m_sender; ///< sender
  Ptr<WaveNetDevice>  m_receiver; ///< receiver
};

AnnexC_TestCase::AnnexC_TestCase (void)
  : TestCase ("annex-c")
{
}

AnnexC_TestCase::~AnnexC_TestCase (void)
{
}

void
AnnexC_TestCase::SendPacket  (uint32_t packetSize, const TxInfo & txInfo, uint32_t sequence)
{
  const static uint16_t WSMP_PROT_NUMBER = 0x88DC;
  const Mac48Address dest = Mac48Address::ConvertFrom (m_receiver->GetAddress ());

  uint8_t *data = new uint8_t [packetSize];
  data [0] = (sequence >> 24) & 0xFF;
  data [1] = (sequence >> 16) & 0xFF;
  data [2] = (sequence >> 8) & 0xFF;
  data [3] = (sequence >> 0) & 0xFF;

  uint64_t ts = Simulator::Now ().GetTimeStep ();
  data [4] = (ts >> 56) & 0xFF;
  data [5] = (ts >> 48) & 0xFF;
  data [6] = (ts >> 40) & 0xFF;
  data [7] = (ts >> 32) & 0xFF;
  data [8] = (ts >> 24) & 0xFF;
  data [9] = (ts >> 16) & 0xFF;
  data [10] = (ts >> 8) & 0xFF;
  data [11] = (ts >> 0) & 0xFF;

  Ptr<Packet> p = Create<Packet> (data, packetSize);

  m_sender->SendX  (p, dest, WSMP_PROT_NUMBER, txInfo);

  delete [] data;
}

bool
AnnexC_TestCase::Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender)
{
  uint8_t *data = new uint8_t [pkt->GetSize ()];
  pkt->CopyData(data, pkt->GetSize ());

  uint32_t seq = data [0];
  seq <<= 8;
  seq |= data [1];
  seq <<= 8;
  seq |= data [2];
  seq <<= 8;
  seq |= data [3];

  uint64_t ts = data [4];
  ts <<= 8;
  ts |= data [5];
  ts <<= 8;
  ts |= data [6];
  ts <<= 8;
  ts |= data [7];
  ts <<= 8;
  ts |= data [8];
  ts <<= 8;
  ts |= data [9];
  ts <<= 8;
  ts |= data [10];
  ts <<= 8;
  ts |= data [11];
  Time sendTime = TimeStep (ts);

  delete [] data;

  Time curTime = Now ();
  Time duration = curTime - sendTime;

  if (seq == 1)
    {
      NS_TEST_EXPECT_MSG_GT (duration, ChannelCoordinator::GetDefaultSchInterval (), "fail to test Annex C when packet sequence is " << seq);
    }
  else if (seq == 2)
    {
      NS_TEST_EXPECT_MSG_LT (duration, ChannelCoordinator::GetDefaultSchInterval (), "fail to test Annex C when packet sequence is " << seq);
    }
  else if (seq == 3)
    {
      NS_TEST_EXPECT_MSG_GT (duration, ChannelCoordinator::GetDefaultCchInterval (), "fail to test Annex C when packet sequence is " << seq);
    }
  else if (seq == 4)
    {
      NS_TEST_EXPECT_MSG_LT (duration, ChannelCoordinator::GetDefaultCchInterval (), "fail to test Annex C when packet sequence is " << seq);
    }
  return true;
}

void
AnnexC_TestCase::DoRun (void)
{
  m_devices = TestCaseHelper::CreatWaveDevice (2);
  m_sender = DynamicCast<WaveNetDevice> (m_devices.Get (0));
  m_receiver = DynamicCast<WaveNetDevice> (m_devices.Get (1));
  m_receiver->SetReceiveCallback (MakeCallback (&AnnexC_TestCase::Receive, this));

  // at 0s, the receiver is assigned AlternatingAccess  for SCH1
  SchInfo infoReceiver = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
  Simulator::Schedule (MilliSeconds (0), &WaveNetDevice::StartSch, m_receiver, infoReceiver);

  // at 0s, the sender is assigned AlternatingAccess for SCH1
  SchInfo infoSender = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
  Simulator::Schedule (MilliSeconds (0), &WaveNetDevice::StartSch, m_sender, infoSender);

  TxInfo txInfo = TxInfo (CCH, 0, WifiMode ("OfdmRate3MbpsBW10MHz"), WIFI_PREAMBLE_LONG, 0);
  // the packet size with 2312 bytes costs 6.42s, which will cancel this transmission in the CCH Interval
  // so the receiver will receive this packet in next CCH Interval
  Simulator::Schedule (MilliSeconds (45), &AnnexC_TestCase::SendPacket, this, 2304, txInfo, 1);

  // the packet size with 312 bytes costs 1.104ms, which will not cancel transmission in the CCH Interval
  // so the receiver can this packet is this CCH Interval
  Simulator::Schedule (MilliSeconds (145), &AnnexC_TestCase::SendPacket, this, 312, txInfo, 2);

  txInfo = TxInfo (SCH1, 0, WifiMode ("OfdmRate3MbpsBW10MHz"), WIFI_PREAMBLE_LONG, 0);
  // the packet size with 2312 bytes costs 6.42ms, which will cancel this transmission in the SCH Interval
  // so the receiver will receive this packet in next SCH Interval
  Simulator::Schedule (MilliSeconds (295), &AnnexC_TestCase::SendPacket, this, 2304, txInfo, 3);

  // the packet size with 312 bytes costs 1.104ms, which will not cancel transmission in the SCH Interval
  // so the receiver can this packet is this SCH Interval
  Simulator::Schedule (MilliSeconds (395), &AnnexC_TestCase::SendPacket, this, 312, txInfo, 4);

  Simulator::Stop (Seconds (1.0));
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief Wave Mac Test Suite
 */
class WaveMacTestSuite : public TestSuite
{
public:
  WaveMacTestSuite ();
};

WaveMacTestSuite::WaveMacTestSuite ()
  : TestSuite ("wave-mac-extension", UNIT)
{
  // TestDuration for TestCase can be QUICK, EXTENSIVE or TAKES_FOREVER
  AddTestCase (new ChannelCoordinationTestCase, TestCase::QUICK);
  AddTestCase (new ChannelRoutingTestCase, TestCase::QUICK);
  AddTestCase (new ChannelAccessTestCase, TestCase::QUICK);
  AddTestCase (new AnnexC_TestCase, TestCase::QUICK);
}

// Do not forget to allocate an instance of this TestSuite
static WaveMacTestSuite waveMacTestSuite; ///< the test suite
