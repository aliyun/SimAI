/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 University of Washington
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
 * Authors: Sébastien Deronne <sebastien.deronne@gmail.com>
 *          Scott Carpenter <scarpenter44@windstream.net>
 */

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/config.h"
#include "ns3/ssid.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/mobility-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/constant-obss-pd-algorithm.h"
#include "ns3/he-configuration.h"
#include "ns3/wifi-utils.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("InterBssTestSuite");

static uint32_t
ConvertContextToNodeId (std::string context)
{
  std::string sub = context.substr (10);
  uint32_t pos = sub.find ("/Device");
  uint32_t nodeId = atoi (sub.substr (0, pos).c_str ());
  return nodeId;
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Test
 *
 * This test case tests the transmission of inter-BSS cases
 * and verify behavior of 11ax OBSS_PD spatial reuse.
 *
 * The topology for this test case is made of three networks, each with one AP and one STA:
 *
 *  AP  --d1--  STA1  --d2--  AP2  --d3-- STA2 --d4--  AP3  --d5-- STA3
 *  TX1         RX1           TX2         RX2          TX3         RX3
 *
 * Main parameters:
 *  OBSS_PD level = -72dbm
 *  Received Power by TX1 from TX2 = [-62dbm, -82dbm]
 *  Received SINR by RX1 from TX1 > 3dB (enough to pass MCS0 reception)
 *  Received SINR by RX2 from TX2 > 3dB (enough to pass MCS0 reception)
 *  Received SINR by RX3 from TX3 > 3dB (enough to pass MCS0 reception)
 *  TX1/RX1 BSS Color = 1
 *  TX2/RX2 transmission PPDU BSS Color = [2 0]
 *  TX3/RX3 BSS color = 3 (BSS 3 only used to test some corner cases)
 *  PHY = 11ax, MCS 0, 80MHz
 */

class TestInterBssConstantObssPdAlgo : public TestCase
{
public:
  TestInterBssConstantObssPdAlgo ();
  ~TestInterBssConstantObssPdAlgo ();

  void DoRun (void) override;

private:
  /**
   * Send one packet function
   * \param tx_dev the transmitting device
   * \param rx_dev the receiving device
   * \param payloadSize the payload size
   */
  void SendOnePacket (Ptr<WifiNetDevice> tx_dev, Ptr<WifiNetDevice> rx_dev, uint32_t payloadSize);

  /**
   * Allocate the node positions
   * \param d1 distance d1 (in meters)
   * \param d2 distance d2 (in meters)
   * \param d3 distance d3 (in meters)
   * \param d4 distance d4 (in meters)
   * \param d5 distance d5 (in meters)
   * \return the node positions
   */
  Ptr<ListPositionAllocator> AllocatePositions (double d1, double d2, double d3, double d4, double d5);

  /**
   * Set the expected transmit power in dBm
   * \param txPowerDbm the transmit power in dBm
   */
  void SetExpectedTxPower (double txPowerDbm);

  /**
   * Setup the simulation
   */
  void SetupSimulation ();

  /**
   * Check the results
   */
  void CheckResults ();

  /**
   * Reset the results
   */
  void ResetResults ();

  /**
   * Clear the drop reasons
   */
  void ClearDropReasons ();

  /**
   * Run one function
   */
  void RunOne ();

  /**
   * Check if the Phy State for a device is an expected value
   * \param device the device to check
   * \param expectedState the expected PHY state
   */
  void CheckPhyState (Ptr<WifiNetDevice> device, WifiPhyState expectedState);

  /**
   * Check if the Phy drop reasons for a device are as expected
   * \param device the device to check
   * \param expectedDropReasons the expected PHY drop reasons
   */
  void CheckPhyDropReasons (Ptr<WifiNetDevice> device, std::vector<WifiPhyRxfailureReason> expectedDropReasons);

  /**
   * Notify Phy transmit begin
   * \param context the context
   * \param p the packet
   * \param txPowerW the tx power
   */
  void NotifyPhyTxBegin (std::string context, Ptr<const Packet> p, double txPowerW);

  /**
   * Notify Phy receive ends
   * \param context the context
   * \param p the packet
   */
  void NotifyPhyRxEnd (std::string context, Ptr<const Packet> p);

  /**
   * Notify Phy receive drops
   * \param context the context
   * \param p the packet
   * \param reason the reason why it was dropped
   */
  void NotifyPhyRxDrop (std::string context, Ptr<const Packet> p, WifiPhyRxfailureReason reason);

  unsigned int m_numSta1PacketsSent; ///< number of sent packets from STA1
  unsigned int m_numSta2PacketsSent; ///< number of sent packets from STA2
  unsigned int m_numAp1PacketsSent; ///< number of sent packets from AP1
  unsigned int m_numAp2PacketsSent; ///< number of sent packets from AP2

  unsigned int m_numSta1PacketsReceived; ///< number of received packets from STA1
  unsigned int m_numSta2PacketsReceived; ///< number of received packets from STA2
  unsigned int m_numAp1PacketsReceived; ///< number of received packets from AP1
  unsigned int m_numAp2PacketsReceived; ///< number of received packets from AP2

  std::vector<WifiPhyRxfailureReason> m_dropReasonsSta1; ///< drop reasons for STA1
  std::vector<WifiPhyRxfailureReason> m_dropReasonsSta2; ///< drop reasons for STA2
  std::vector<WifiPhyRxfailureReason> m_dropReasonsAp1; ///< drop reasons for AP1
  std::vector<WifiPhyRxfailureReason> m_dropReasonsAp2; ///< drop reasons for AP2

  unsigned int m_payloadSize1; ///< size in bytes of packet payload in BSS 1
  unsigned int m_payloadSize2; ///< size in bytes of packet payload in BSS 2
  unsigned int m_payloadSize3; ///< size in bytes of packet payload in BSS 3

  NetDeviceContainer m_staDevices; ///< STA devices
  NetDeviceContainer m_apDevices;  ///< AP devices

  double m_txPowerDbm;         ///< configured transmit power in dBm
  double m_obssPdLevelDbm;     ///< OBSS-PD level in dBm
  double m_obssRxPowerDbm;     ///< forced RX power in dBm for OBSS
  double m_expectedTxPowerDbm; ///< expected transmit power in dBm

  uint8_t m_bssColor1; ///< color for BSS 1
  uint8_t m_bssColor2; ///< color for BSS 2
  uint8_t m_bssColor3; ///< color for BSS 3
};

TestInterBssConstantObssPdAlgo::TestInterBssConstantObssPdAlgo ()
  : TestCase ("InterBssConstantObssPd"),
    m_numSta1PacketsSent (0),
    m_numSta2PacketsSent (0),
    m_numAp1PacketsSent (0),
    m_numAp2PacketsSent (0),
    m_numSta1PacketsReceived (0),
    m_numSta2PacketsReceived (0),
    m_numAp1PacketsReceived (0),
    m_numAp2PacketsReceived (0),
    m_payloadSize1 (1000),
    m_payloadSize2 (1500),
    m_payloadSize3 (2000),
    m_txPowerDbm (15),
    m_obssPdLevelDbm (-72),
    m_obssRxPowerDbm (-82),
    m_expectedTxPowerDbm (15),
    m_bssColor1 (1),
    m_bssColor2 (2),
    m_bssColor3 (3)
{
}

TestInterBssConstantObssPdAlgo::~TestInterBssConstantObssPdAlgo ()
{
  ClearDropReasons ();
}

Ptr<ListPositionAllocator>
TestInterBssConstantObssPdAlgo::AllocatePositions (double d1, double d2, double d3, double d4, double d5)
{
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));  // AP1
  positionAlloc->Add (Vector (d1 + d2, 0.0, 0.0));  // AP2
  positionAlloc->Add (Vector (d1 + d2 + d3 + d4, 0.0, 0.0));  // AP3
  positionAlloc->Add (Vector (d1, 0.0, 0.0));  // STA1
  positionAlloc->Add (Vector (d1 + d2 + d3, 0.0, 0.0));  // STA2
  positionAlloc->Add (Vector (d1 + d2 + d3 + d4 + d5, 0.0, 0.0));  // STA3
  return positionAlloc;
}

void
TestInterBssConstantObssPdAlgo::SetupSimulation ()
{
  Ptr<WifiNetDevice> ap_device1 = DynamicCast<WifiNetDevice> (m_apDevices.Get (0));
  Ptr<WifiNetDevice> ap_device2 = DynamicCast<WifiNetDevice> (m_apDevices.Get (1));
  Ptr<WifiNetDevice> ap_device3 = DynamicCast<WifiNetDevice> (m_apDevices.Get (2));
  Ptr<WifiNetDevice> sta_device1 = DynamicCast<WifiNetDevice> (m_staDevices.Get (0));
  Ptr<WifiNetDevice> sta_device2 = DynamicCast<WifiNetDevice> (m_staDevices.Get (1));
  Ptr<WifiNetDevice> sta_device3 = DynamicCast<WifiNetDevice> (m_staDevices.Get (2));

  bool expectFilter = (m_bssColor1 != 0) && (m_bssColor2 != 0);
  bool expectPhyReset = expectFilter && (m_obssPdLevelDbm >= m_obssRxPowerDbm);
  std::vector<WifiPhyRxfailureReason> dropReasons;
  WifiPhyState stateDuringPayloadNeighboringBss = expectFilter ? WifiPhyState::CCA_BUSY : WifiPhyState::RX;
  if (expectFilter)
    {
      dropReasons.push_back (FILTERED);
    }
  if (expectPhyReset)
    {
      dropReasons.push_back (OBSS_PD_CCA_RESET);
    }

  // In order to have all ADDBA handshakes established, each AP and STA sends a packet.

  Simulator::Schedule (Seconds (0.25), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device1, sta_device1, m_payloadSize1);
  Simulator::Schedule (Seconds (0.5), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device1, ap_device1, m_payloadSize1);
  Simulator::Schedule (Seconds (0.75), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device2, sta_device2, m_payloadSize2);
  Simulator::Schedule (Seconds (1), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device2, ap_device2, m_payloadSize2);
  Simulator::Schedule (Seconds (1.25), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device3, sta_device3, m_payloadSize3);
  Simulator::Schedule (Seconds (1.5), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device3, ap_device3, m_payloadSize3);

  // We test PHY state and verify whether a CCA reset did occur.

  // AP2 sends a packet 0.5s later.
  Simulator::Schedule (Seconds (2.0), &TestInterBssConstantObssPdAlgo::ClearDropReasons, this);
  Simulator::Schedule (Seconds (2.0), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device2, sta_device2, m_payloadSize2);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (10), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device2, WifiPhyState::TX);
  // All other PHYs should have stay idle until 4us (preamble detection time).
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (13), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device1, WifiPhyState::IDLE);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (13), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device2, WifiPhyState::IDLE);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (13), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, WifiPhyState::IDLE);
  // All PHYs should be receiving the PHY header (i.e. PHY state is CCA_BUSY) if preamble has been detected (always the case in this test).
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (14), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device1, WifiPhyState::CCA_BUSY);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (14), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device2, WifiPhyState::CCA_BUSY);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (14), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, WifiPhyState::CCA_BUSY);
  // PHYs of AP1 and STA1 should be idle after HE-SIG-A if they were reset by OBSS_PD SR, otherwise they should be CCA_busy until beginning of payload.
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (43), &TestInterBssConstantObssPdAlgo::CheckPhyDropReasons, this, sta_device1, dropReasons);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (43), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device1, expectPhyReset ? WifiPhyState::IDLE : WifiPhyState::CCA_BUSY);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (43), &TestInterBssConstantObssPdAlgo::CheckPhyDropReasons, this, ap_device1, dropReasons);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (43), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, expectPhyReset ? WifiPhyState::IDLE : WifiPhyState::CCA_BUSY);
  // PHYs of AP1 and STA1 should be idle if they were reset by OBSS_PD SR, otherwise they should be CCA_busy/Rx (since filtered/not filtered, resp.).
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (54), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device1, expectPhyReset ? WifiPhyState::IDLE : stateDuringPayloadNeighboringBss);
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (54), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, expectPhyReset ? WifiPhyState::IDLE : stateDuringPayloadNeighboringBss);
  // STA2 should be receiving
  Simulator::Schedule (Seconds (2.0) + MicroSeconds (54), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device2, WifiPhyState::RX);

  // We test whether two networks can transmit simultaneously, and whether transmit power restrictions are applied.

  // AP2 sends another packet 0.1s later.
  Simulator::Schedule (Seconds (2.1), &TestInterBssConstantObssPdAlgo::ClearDropReasons, this);
  Simulator::Schedule (Seconds (2.1), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device2, sta_device2, m_payloadSize2);
  // STA1 sends a packet 42us later (i.e. right after HE-SIG-A of AP2). Even though AP2 is still transmitting, STA1 can transmit simultaneously if it's PHY was reset by OBSS_PD SR.
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (42), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device1, ap_device1, m_payloadSize1);
  if (expectPhyReset)
    {
      // In this case, we check the TX power is restricted (and set the expected value slightly before transmission should occur)
      double expectedTxPower = std::min (m_txPowerDbm, 21 - (m_obssPdLevelDbm + 82));
      Simulator::Schedule (Seconds (2.1) + MicroSeconds (41), &TestInterBssConstantObssPdAlgo::SetExpectedTxPower, this, expectedTxPower);
    }
  // Check simultaneous transmissions
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (100), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device2, WifiPhyState::TX);
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (100), &TestInterBssConstantObssPdAlgo::CheckPhyDropReasons, this, sta_device1, dropReasons);
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (100), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device1, expectPhyReset ? WifiPhyState::TX : stateDuringPayloadNeighboringBss);
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (100), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device2, WifiPhyState::RX);
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (100), &TestInterBssConstantObssPdAlgo::CheckPhyDropReasons, this, ap_device1, dropReasons);
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (100), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, stateDuringPayloadNeighboringBss);
  Simulator::Schedule (Seconds (2.1) + MicroSeconds (142), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, expectPhyReset ? WifiPhyState::RX : stateDuringPayloadNeighboringBss);

  // AP2 sends another packet 0.1s later, and STA1 wanting to send a packet during the payload of the former.
  Simulator::Schedule (Seconds (2.2), &TestInterBssConstantObssPdAlgo::ClearDropReasons, this);
  Simulator::Schedule (Seconds (2.2), &TestInterBssConstantObssPdAlgo::SetExpectedTxPower, this, m_txPowerDbm);
  Simulator::Schedule (Seconds (2.2), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device2, sta_device2, m_payloadSize2);
  // STA1 sends a packet 90us later (i.e. during payload of AP2). Even though AP2 is still transmitting, STA1 can transmit simultaneously if it's PHY was reset by OBSS_PD SR.
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (90), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device1, ap_device1, m_payloadSize1);
  if (expectPhyReset)
    {
      // In this case, we check the TX power is restricted (and set the expected value slightly before transmission should occur)
      double expectedTxPower = std::min (m_txPowerDbm, 21 - (m_obssPdLevelDbm + 82));
      Simulator::Schedule (Seconds (2.2) + MicroSeconds (89), &TestInterBssConstantObssPdAlgo::SetExpectedTxPower, this, expectedTxPower);
    }
  // Check simultaneous transmissions
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (105), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device2, WifiPhyState::TX);
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (105), &TestInterBssConstantObssPdAlgo::CheckPhyDropReasons, this, sta_device1, dropReasons);
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (105), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device1, expectPhyReset ? WifiPhyState::TX : stateDuringPayloadNeighboringBss);
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (105), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, sta_device2, WifiPhyState::RX);
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (105), &TestInterBssConstantObssPdAlgo::CheckPhyDropReasons, this, ap_device1, dropReasons);
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (105), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, stateDuringPayloadNeighboringBss);
  Simulator::Schedule (Seconds (2.2) + MicroSeconds (195), &TestInterBssConstantObssPdAlgo::CheckPhyState, this, ap_device1, expectPhyReset ? WifiPhyState::RX : stateDuringPayloadNeighboringBss);


  // Verify transmit power restrictions are not applied if access to the channel is requested after ignored OBSS transmissions.

  Simulator::Schedule (Seconds (2.3), &TestInterBssConstantObssPdAlgo::SetExpectedTxPower, this, m_txPowerDbm);
  // AP2 sends another packet 0.1s later. Power restriction should not be applied.
  Simulator::Schedule (Seconds (2.3), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device2, sta_device2, m_payloadSize2);
  // STA1 sends a packet 0.1s later. Power restriction should not be applied.
  Simulator::Schedule (Seconds (2.4), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device1, ap_device1, m_payloadSize1);

  // Verify a scenario that involves 3 networks in order to verify corner cases for transmit power restrictions.
  // First, there is a transmission on network 2 from STA to AP, followed by a response from AP to STA.
  // During that time, the STA on network 1 has a packet to send and request access to the channel.
  // If a CCA reset occurred, it starts deferring while transmissions are ongoing from network 2.
  // Before its backoff expires, a transmission on network 3 occurs, also eventually triggering another CCA reset (depending on the scenario that is being run).
  // This test checks whether this sequence preserves transmit power restrictions if CCA resets occurred, since STA 1 has been deferring during ignored OBSS transmissions.

  Simulator::Schedule (Seconds (2.5), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, sta_device2, ap_device2, m_payloadSize2 / 10);
  Simulator::Schedule (Seconds (2.5) + MicroSeconds (15), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device2, sta_device2, m_payloadSize2 / 10);
  Simulator::Schedule (Seconds (2.5) + MicroSeconds (270), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device1, sta_device1, m_payloadSize1 / 10);
  Simulator::Schedule (Seconds (2.5) + MicroSeconds (300), &TestInterBssConstantObssPdAlgo::SendOnePacket, this, ap_device3, sta_device3, m_payloadSize3 / 10);
  if (expectPhyReset)
    {
      // In this case, we check the TX power is restricted (and set the expected value slightly before transmission should occur)
      double expectedTxPower = std::min (m_txPowerDbm, 21 - (m_obssPdLevelDbm + 82));
      Simulator::Schedule (Seconds (2.5) + MicroSeconds (338), &TestInterBssConstantObssPdAlgo::SetExpectedTxPower, this, expectedTxPower);
    }

  Simulator::Stop (Seconds (2.6));
}

void
TestInterBssConstantObssPdAlgo::ResetResults ()
{
  m_numSta1PacketsSent = 0;
  m_numSta2PacketsSent = 0;
  m_numAp1PacketsSent = 0;
  m_numAp2PacketsSent = 0;
  m_numSta1PacketsReceived = 0;
  m_numSta2PacketsReceived = 0;
  m_numAp1PacketsReceived = 0;
  m_numAp2PacketsReceived = 0;
  ClearDropReasons ();
  m_expectedTxPowerDbm = m_txPowerDbm;
}

void
TestInterBssConstantObssPdAlgo::ClearDropReasons ()
{
  m_dropReasonsSta1.clear ();
  m_dropReasonsSta2.clear ();
  m_dropReasonsAp1.clear ();
  m_dropReasonsAp2.clear ();
}

void
TestInterBssConstantObssPdAlgo::CheckResults ()
{
  NS_TEST_ASSERT_MSG_EQ (m_numSta1PacketsSent, 4, "The number of packets sent by STA1 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numSta2PacketsSent, 2, "The number of packets sent by STA2 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numAp1PacketsSent, 2, "The number of packets sent by AP1 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numAp2PacketsSent, 6, "The number of packets sent by AP2 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numSta1PacketsReceived, 2, "The number of packets received by STA1 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numSta2PacketsReceived, 6, "The number of packets received by STA2 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numAp1PacketsReceived, 4, "The number of packets received by AP1 is not correct!");
  NS_TEST_ASSERT_MSG_EQ (m_numAp2PacketsReceived, 2, "The number of packets received by AP2 is not correct!");
}

void
TestInterBssConstantObssPdAlgo::NotifyPhyTxBegin (std::string context, Ptr<const Packet> p, double txPowerW)
{
  uint32_t idx = ConvertContextToNodeId (context);
  uint32_t pktSize = p->GetSize () - 38;
  if ((idx == 0) && ((pktSize == m_payloadSize1) || (pktSize == (m_payloadSize1 / 10))))
    {
      m_numSta1PacketsSent++;
      NS_TEST_EXPECT_MSG_EQ (TestDoubleIsEqual (WToDbm (txPowerW), m_expectedTxPowerDbm, 1e-12), true, "Tx power is not correct!");
    }
  else if ((idx == 1) && ((pktSize == m_payloadSize2) || (pktSize == (m_payloadSize2 / 10))))
    {
      m_numSta2PacketsSent++;
      NS_TEST_EXPECT_MSG_EQ (TestDoubleIsEqual (WToDbm (txPowerW), m_expectedTxPowerDbm, 1e-12), true, "Tx power is not correct!");
    }
  else if ((idx == 3) && ((pktSize == m_payloadSize1) || (pktSize == (m_payloadSize1 / 10))))
    {
      m_numAp1PacketsSent++;
      NS_TEST_EXPECT_MSG_EQ (TestDoubleIsEqual (WToDbm (txPowerW), m_expectedTxPowerDbm, 1e-12), true, "Tx power is not correct!");
    }
  else if ((idx == 4) && ((pktSize == m_payloadSize2) || (pktSize == (m_payloadSize2 / 10))))
    {
      m_numAp2PacketsSent++;
      NS_TEST_EXPECT_MSG_EQ (TestDoubleIsEqual (WToDbm (txPowerW), m_expectedTxPowerDbm, 1e-12), true, "Tx power is not correct!");
    }
}

void
TestInterBssConstantObssPdAlgo::NotifyPhyRxEnd (std::string context, Ptr<const Packet> p)
{
  uint32_t idx = ConvertContextToNodeId (context);
  uint32_t pktSize = p->GetSize () - 38;
  if ((idx == 0) && ((pktSize == m_payloadSize1) || (pktSize == (m_payloadSize1 / 10))))
    {
      m_numSta1PacketsReceived++;
    }
  else if ((idx == 1) && ((pktSize == m_payloadSize2) || (pktSize == (m_payloadSize2 / 10))))
    {
      m_numSta2PacketsReceived++;
    }
  else if ((idx == 3) && ((pktSize == m_payloadSize1) || (pktSize == (m_payloadSize1 / 10))))
    {
      m_numAp1PacketsReceived++;
    }
  else if ((idx == 4) && ((pktSize == m_payloadSize2) || (pktSize == (m_payloadSize2 / 10))))
    {
      m_numAp2PacketsReceived++;
    }
}

void
TestInterBssConstantObssPdAlgo::NotifyPhyRxDrop (std::string context, Ptr<const Packet> p,
                                                 WifiPhyRxfailureReason reason)
{
  uint32_t idx = ConvertContextToNodeId (context);
  uint32_t pktSize = p->GetSize () - 38;
  if ((idx == 0) && ((pktSize != m_payloadSize1) && (pktSize != (m_payloadSize1 / 10))))
    {
      m_dropReasonsSta1.push_back (reason);
    }
  else if ((idx == 1) && ((pktSize != m_payloadSize2) && (pktSize != (m_payloadSize2 / 10))))
    {
      m_dropReasonsSta2.push_back (reason);
    }
  else if ((idx == 3) && ((pktSize != m_payloadSize1) && (pktSize != (m_payloadSize1 / 10))))
    {
      m_dropReasonsAp1.push_back (reason);
    }
  else if ((idx == 4) && ((pktSize != m_payloadSize2) && (pktSize != (m_payloadSize2 / 10))))
    {
      m_dropReasonsAp2.push_back (reason);
    }
}

void
TestInterBssConstantObssPdAlgo::SendOnePacket (Ptr<WifiNetDevice> tx_dev, Ptr<WifiNetDevice> rx_dev, uint32_t payloadSize)
{
  Ptr<Packet> p = Create<Packet> (payloadSize);
  tx_dev->Send (p, rx_dev->GetAddress (), 1);
}

void
TestInterBssConstantObssPdAlgo::SetExpectedTxPower (double txPowerDbm)
{
  m_expectedTxPowerDbm = txPowerDbm;
}

void
TestInterBssConstantObssPdAlgo::CheckPhyState (Ptr<WifiNetDevice> device, WifiPhyState expectedState)
{
  WifiPhyState currentState;
  PointerValue ptr;
  Ptr<WifiPhy> phy = DynamicCast<WifiPhy> (device->GetPhy ());
  phy->GetAttribute ("State", ptr);
  Ptr <WifiPhyStateHelper> state = DynamicCast <WifiPhyStateHelper> (ptr.Get<WifiPhyStateHelper> ());
  currentState = state->GetState ();
  NS_TEST_ASSERT_MSG_EQ (currentState, expectedState, "PHY State " << currentState << " does not match expected state " << expectedState << " at " << Simulator::Now ());
}

void
TestInterBssConstantObssPdAlgo::CheckPhyDropReasons (Ptr<WifiNetDevice> device,
                                                     std::vector<WifiPhyRxfailureReason> expectedDropReasons)
{
  std::vector<WifiPhyRxfailureReason> currentDropReasons;
  uint32_t nodeId = device->GetNode ()->GetId ();
  switch (nodeId)
    {
      case 0: //STA1
        currentDropReasons = m_dropReasonsSta1;
        break;
      case 1: //STA2
        currentDropReasons = m_dropReasonsSta2;
        break;
      case 3: //AP1
        currentDropReasons = m_dropReasonsAp1;
        break;
      case 4: //AP2
        currentDropReasons = m_dropReasonsAp2;
        break;
      default: //others, no attribute
        return;
    }
  NS_TEST_ASSERT_MSG_EQ (currentDropReasons.size (), expectedDropReasons.size (), "Number of drop reasons " << currentDropReasons.size () << " does not match expected one " << expectedDropReasons.size () << " at " << Simulator::Now ());
  for (std::size_t i = 0; i < currentDropReasons.size (); ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (currentDropReasons[i], expectedDropReasons[i], "Drop reason " << i << ": " << currentDropReasons[i] << " does not match expected reason " << expectedDropReasons[i] << " at " << Simulator::Now ());
    }
}

void
TestInterBssConstantObssPdAlgo::RunOne (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (3);
  int64_t streamNumber = 50;
  
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_MaxAmpduSize", UintegerValue (0));

  ResetResults ();

  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (3);

  NodeContainer wifiApNodes;
  wifiApNodes.Create (3);

  Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel> ();
  lossModel->SetDefaultLoss (m_txPowerDbm - m_obssRxPowerDbm); //Force received RSSI to be equal to m_obssRxPowerDbm

  SpectrumWifiPhyHelper phy;
  phy.DisablePreambleDetectionModel ();
  phy.SetFrameCaptureModel ("ns3::SimpleFrameCaptureModel");
  Ptr<MultiModelSpectrumChannel> channel = CreateObject<MultiModelSpectrumChannel> ();
  channel->SetPropagationDelayModel (CreateObject<ConstantSpeedPropagationDelayModel> ());
  channel->AddPropagationLossModel (lossModel);
  phy.SetChannel (channel);
  phy.Set ("TxPowerStart", DoubleValue (m_txPowerDbm));
  phy.Set ("TxPowerEnd", DoubleValue (m_txPowerDbm));
  phy.Set ("ChannelSettings", StringValue ("{36, 20, BAND_5GHZ, 0}"));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("HeMcs5"),
                                "ControlMode", StringValue ("HeMcs0"));

  wifi.SetObssPdAlgorithm ("ns3::ConstantObssPdAlgorithm",
                           "ObssPdLevel", DoubleValue (m_obssPdLevelDbm));

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid));
  m_staDevices = wifi.Install (phy, mac, wifiStaNodes);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (m_staDevices, streamNumber);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid));
  m_apDevices = wifi.Install (phy, mac, wifiApNodes);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (m_apDevices, streamNumber);

  for (uint32_t i = 0; i < m_apDevices.GetN (); i++)
    {
      Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice> (m_apDevices.Get (i));
      Ptr<HeConfiguration> heConfiguration = device->GetHeConfiguration ();
      if (i == 0)
        {
          heConfiguration->SetAttribute ("BssColor", UintegerValue (m_bssColor1));
        }
      else if (i == 1)
        {
          heConfiguration->SetAttribute ("BssColor", UintegerValue (m_bssColor2));
        }
      else
        {
          heConfiguration->SetAttribute ("BssColor", UintegerValue (m_bssColor3));
        }
    }

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = AllocatePositions (10, 50, 10, 50, 10); //distances do not really matter since we set RSSI per TX-RX pair to have full control
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNodes);
  mobility.Install (wifiStaNodes);

  lossModel->SetLoss (wifiStaNodes.Get (0)->GetObject<MobilityModel> (), wifiApNodes.Get (0)->GetObject<MobilityModel> (), m_txPowerDbm + 30); //Low attenuation for IBSS transmissions
  lossModel->SetLoss (wifiStaNodes.Get (1)->GetObject<MobilityModel> (), wifiApNodes.Get (1)->GetObject<MobilityModel> (), m_txPowerDbm + 30); //Low attenuation for IBSS transmissions
  lossModel->SetLoss (wifiStaNodes.Get (2)->GetObject<MobilityModel> (), wifiApNodes.Get (2)->GetObject<MobilityModel> (), m_txPowerDbm + 30); //Low attenuation for IBSS transmissions

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin", MakeCallback (&TestInterBssConstantObssPdAlgo::NotifyPhyTxBegin, this));
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxEnd", MakeCallback (&TestInterBssConstantObssPdAlgo::NotifyPhyRxEnd, this));
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop", MakeCallback (&TestInterBssConstantObssPdAlgo::NotifyPhyRxDrop, this));

  SetupSimulation ();

  Simulator::Run ();
  Simulator::Destroy ();

  CheckResults ();
}

void
TestInterBssConstantObssPdAlgo::DoRun (void)
{
  //Test case 1: CCA CS Threshold = m_obssRxPowerDbm < m_obssPdLevelDbm
  m_obssPdLevelDbm = -72;
  m_obssRxPowerDbm = -82;
  m_bssColor1 = 1;
  m_bssColor2 = 2;
  m_bssColor3 = 3;
  RunOne ();

  //Test case 2: CCA CS Threshold < m_obssPdLevelDbm < m_obssRxPowerDbm
  m_obssPdLevelDbm = -72;
  m_obssRxPowerDbm = -62;
  m_bssColor1 = 1;
  m_bssColor2 = 2;
  m_bssColor3 = 3;
  RunOne ();

  //Test case 3: CCA CS Threshold < m_obssPdLevelDbm = m_obssRxPowerDbm
  m_obssPdLevelDbm = -72;
  m_obssRxPowerDbm = -72;
  m_bssColor1 = 1;
  m_bssColor2 = 2;
  m_bssColor3 = 3;
  RunOne ();

  //Test case 4: CCA CS Threshold = m_obssRxPowerDbm < m_obssPdLevelDbm with BSS color 2 and 3 set to 0
  m_obssPdLevelDbm = -72;
  m_obssRxPowerDbm = -82;
  m_bssColor1 = 1;
  m_bssColor2 = 0;
  m_bssColor3 = 0;
  RunOne ();

  //Test case 5: CCA CS Threshold = m_obssRxPowerDbm < m_obssPdLevelDbm with BSS color 1 set to 0
  m_obssPdLevelDbm = -72;
  m_obssRxPowerDbm = -82;
  m_bssColor1 = 0;
  m_bssColor2 = 2;
  m_bssColor3 = 3;
  RunOne ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Inter BSS Test Suite
 */

class InterBssTestSuite : public TestSuite
{
public:
  InterBssTestSuite ();
};

InterBssTestSuite::InterBssTestSuite ()
  : TestSuite ("wifi-inter-bss", UNIT)
{
  AddTestCase (new TestInterBssConstantObssPdAlgo, TestCase::QUICK);
}

// Do not forget to allocate an instance of this TestSuite
static InterBssTestSuite interBssTestSuite;
