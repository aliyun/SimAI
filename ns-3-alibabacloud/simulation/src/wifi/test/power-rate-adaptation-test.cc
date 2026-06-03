/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universidad de la República - Uruguay
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
 * Author: Matías Richart <mrichart@fing.edu.uy>
 */

#include "ns3/node.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-phy.h"
#include "ns3/adhoc-wifi-mac.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/frame-exchange-manager.h"
#include "ns3/interference-helper.h"
#include "ns3/wifi-default-protection-manager.h"
#include "ns3/wifi-default-ack-manager.h"

using namespace ns3;

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Power Rate Adaptation Test
 */
class PowerRateAdaptationTest : public TestCase
{
public:
  PowerRateAdaptationTest ();

  void DoRun (void) override;
private:
  /// Test parf function
  void TestParf ();
  /// Test aparf function
  void TestAparf ();
  /// Test rrpaa function
  void TestRrpaa ();
  /**
   * Configure nde function
   * \returns the node
   */
  Ptr<Node> ConfigureNode ();

  ObjectFactory m_manager; ///< manager
};

PowerRateAdaptationTest::PowerRateAdaptationTest ()
  : TestCase ("PowerRateAdaptation")
{
}

Ptr<Node>
PowerRateAdaptationTest::ConfigureNode ()
{
  /*
   * Create channel model. Is is necessary to configure correctly the phy layer.
   */
  Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel> ();

  /*
   * Create mac layer. We use Adhoc because association is not needed to get supported rates.
   */
  Ptr<WifiNetDevice> dev = CreateObject<WifiNetDevice> ();
  Ptr<AdhocWifiMac> mac = CreateObject<AdhocWifiMac> ();
  mac->SetDevice (dev);
  mac->ConfigureStandard (WIFI_STANDARD_80211a);
  Ptr<FrameExchangeManager> fem = mac->GetFrameExchangeManager ();

  Ptr<WifiProtectionManager> protectionManager = CreateObject<WifiDefaultProtectionManager> ();
  protectionManager->SetWifiMac (mac);
  fem->SetProtectionManager (protectionManager);

  Ptr<WifiAckManager> ackManager = CreateObject<WifiDefaultAckManager> ();
  ackManager->SetWifiMac (mac);
  fem->SetAckManager (ackManager);

  /*
   * Create mobility model. Is needed by the phy layer for transmission.
   */
  Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel> ();

  /*
   * Create and configure phy layer.
   */
  Ptr<YansWifiPhy> phy = CreateObject<YansWifiPhy> ();
  Ptr<InterferenceHelper> interferenceHelper = CreateObject<InterferenceHelper> ();
  phy->SetInterferenceHelper (interferenceHelper);
  phy->SetChannel (channel);
  phy->SetDevice (dev);
  phy->SetMobility (mobility);
  phy->ConfigureStandard (WIFI_STANDARD_80211a);

  /*
   * Configure power control parameters.
   */
  phy->SetNTxPower (18);
  phy->SetTxPowerStart (0);
  phy->SetTxPowerEnd (17);

  /*
   * Create manager.
   */
  Ptr<WifiRemoteStationManager> manager = m_manager.Create<WifiRemoteStationManager> ();

  /*
   * Create and configure node. Add mac and phy layer and the manager.
   */
  Ptr<Node> node = CreateObject<Node> ();
  mac->SetAddress (Mac48Address::Allocate ());
  dev->SetMac (mac);
  dev->SetPhy (phy);
  dev->SetRemoteStationManager (manager);
  node->AddDevice (dev);

  return node;
}

void
PowerRateAdaptationTest::TestParf ()
{
  m_manager.SetTypeId ("ns3::ParfWifiManager");
  Ptr<Node> node = ConfigureNode ();
  Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice> (node->GetDevice (0));
  Ptr<WifiRemoteStationManager> manager = dev->GetRemoteStationManager ();

  /*
   * Configure thresholds for rate and power control.
   */
  manager->SetAttribute ("AttemptThreshold",UintegerValue (15));
  manager->SetAttribute ("SuccessThreshold",UintegerValue (10));

  /*
   * Create a dummy packet to simulate transmission.
   */
  Mac48Address remoteAddress = Mac48Address::Allocate ();
  WifiMacHeader packetHeader;
  packetHeader.SetAddr1 (remoteAddress);
  packetHeader.SetType (WIFI_MAC_DATA);
  packetHeader.SetQosTid (0);
  Ptr<Packet> packet = Create<Packet> (10);
  Ptr<WifiMacQueueItem> mpdu = Create<WifiMacQueueItem> (packet, packetHeader);
  WifiMode ackMode;

  /*
   * To initialize the manager we need to generate a transmission.
   */
  Ptr<Packet> p = Create<Packet> ();
  dev->Send (p, remoteAddress, 1);

  //-----------------------------------------------------------------------------------------------------

  /*
   * Parf initiates with maximal rate and power.
   */
  WifiTxVector txVector = manager->GetDataTxVector (packetHeader);
  WifiMode mode = txVector.GetMode ();
  int power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Initial data rate wrong");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Initial power level wrong");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 10 consecutive successful transmissions parf increase rate or decrease power.
   * As we are at maximal rate, the power should be decreased. recoveryPower=true.
   */
  for (int i = 0; i < 10; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are using recovery power, one failure make power increase.
   *
   */
  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 15 transmissions attempts parf increase rate or decrease power.
   * As we are at maximal rate, the power should be decreased. recoveryPower=true.
   */
  for (int i = 0; i < 7; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
      manager->ReportDataFailed (mpdu);
    }
  manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are using recovery power, one failure make power increase. recoveryPower=false.
   */

  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After two consecutive fails the rate is decreased or the power increased.
   * As we are at maximal power, the rate should be decreased.
   */
  manager->ReportDataFailed (mpdu);
  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 48000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 10 consecutive successful transmissions parf increase rate or decrease power.
   * As we are not at maximal rate, the rate is increased again. recoveryRate=true.
   */
  for (int i = 0; i < 10; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are using recovery rate, one failure make rate decrease. recoveryRate=false.
   */

  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 48000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 10 consecutive successful transmissions parf increase rate or decrease power.
   * As we are not at maximal rate, the rate is increased again. recoveryRate=true.
   */
  for (int i = 0; i < 10; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 10 consecutive successful transmissions parf increase rate or decrease power.
   * As we are at maximal rate, the power is decreased. recoveryRate=false, recoveryPower=true.
   */
  for (int i = 0; i < 10; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "PARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * One successful transmissions after a power decrease make recoverPower=false.
   * So we need two consecutive failures to increase power again.
   */
  manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);

  for (int i = 0; i < 2; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "PARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "PARF: Incorrect value of power level");

  Simulator::Destroy ();
}

void
PowerRateAdaptationTest::TestAparf ()
{
  m_manager.SetTypeId ("ns3::AparfWifiManager");
  Ptr<Node> node = ConfigureNode ();
  Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice> (node->GetDevice (0));
  Ptr<WifiRemoteStationManager> manager = dev->GetRemoteStationManager ();

  /*
   * Configure thresholds for rate and power control.
   */
  manager->SetAttribute ("SuccessThreshold1",UintegerValue (3));
  manager->SetAttribute ("SuccessThreshold2",UintegerValue (10));
  manager->SetAttribute ("FailThreshold",UintegerValue (1));
  manager->SetAttribute ("PowerThreshold",UintegerValue (10));

  /*
   * Create a dummy packet to simulate transmission.
   */
  Mac48Address remoteAddress = Mac48Address::Allocate ();
  WifiMacHeader packetHeader;
  packetHeader.SetAddr1 (remoteAddress);
  packetHeader.SetType (WIFI_MAC_DATA);
  packetHeader.SetQosTid (0);
  Ptr<Packet> packet = Create<Packet> (10);
  Ptr<WifiMacQueueItem> mpdu = Create<WifiMacQueueItem> (packet, packetHeader);
  WifiMode ackMode;

  /*
   * To initialize the manager we need to generate a transmission.
   */
  Ptr<Packet> p = Create<Packet> ();
  dev->Send (p, remoteAddress, 1);

  //-----------------------------------------------------------------------------------------------------

  /*
   * Aparf initiates with maximal rate and power.
   */
  WifiTxVector txVector = manager->GetDataTxVector (packetHeader);
  WifiMode mode = txVector.GetMode ();
  int power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Initial data rate wrong");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "APARF: Initial power level wrong");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As Aparf starts in state High, after 3 consecutive successful transmissions aparf increase rate or decrease power.
   * As we are at maximal rate, the power should be decreased.
   * Change to state Spread.
   */
  for (int i = 0; i < 3; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * One failure make the power to be increased again.
   * Change to state Low.
   */
  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are in state Low we need 10 successful transmissions to increase rate or decrease power.
   * As we are at maximal rate, the power should be decreased.
   * Change to state Spread.
   */
  for (int i = 0; i < 10; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * One more successful transmission make to change to state High.
   * Two more successful transmissions make power decrease.
   */

  for (int i = 0; i < 3; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 15, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are in state High we need 3 successful transmissions to increase rate or decrease power.
   * After 16*3 successful transmissions power is decreased to zero.
   */
  for (int i = 0; i < 16 * 3; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 0, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After one fail the rate is decreased or the power increased.
   * As we are at minimal power, the power should be increased.
   */
  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 1, "Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After one fail the rate is decreased or the power increased.
   * After 16 failed transmissions power is increase to 17.
   */
  for (int i = 0; i < 16; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After one fail the rate is decreased or the power increased.
   * As we are at maximal power, the rate should be decreased.
   * Set critical rate to 54 Mbps.
   */
  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 48000000, "Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are in state High we need 3 successful transmissions to increase rate or decrease power.
   * As rate critical is set, after 3 successful transmissions power is decreased.
   */
  for (int i = 0; i < 3; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 48000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "APARF: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As we are in state High we need 3 successful transmissions to increase rate or decrease power.
   * After 10 power changes critical rate is reset.
   * So after 10*3 successful transmissions critical rate is set to 0.
   * And 3 successful transmissions more will make power increase to maximum and rate increase to the critical rate.
   */
  for (int i = 0; i < 9 * 3; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 48000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 7, "APARF: Incorrect value of power level");

  for (int i = 0; i < 3; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth ()), 54000000, "APARF: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "APARF: Incorrect value of power level");

  Simulator::Destroy ();
}

void
PowerRateAdaptationTest::TestRrpaa ()
{
  m_manager.SetTypeId ("ns3::RrpaaWifiManager");
  Ptr<Node> node = ConfigureNode ();
  Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice> (node->GetDevice (0));
  Ptr<WifiRemoteStationManager> manager = dev->GetRemoteStationManager ();

  /*
   * Configure constants for rate and power control.
   */
  manager->SetAttribute ("Basic", BooleanValue (true));
  manager->SetAttribute ("Alpha", DoubleValue (1.25));
  manager->SetAttribute ("Beta", DoubleValue (2));
  manager->SetAttribute ("Tau", DoubleValue (0.015));
  /*
   * Constants for the Probabilistic Decision Table.
   * We set both to 1 to avoid random behaviour in tests.
   */
  manager->SetAttribute ("Gamma", DoubleValue (1));
  manager->SetAttribute ("Delta", DoubleValue (1));

  /*
   * Create a dummy packet to simulate transmission.
   */
  Mac48Address remoteAddress = Mac48Address::Allocate ();
  WifiMacHeader packetHeader;
  packetHeader.SetAddr1 (remoteAddress);
  packetHeader.SetType (WIFI_MAC_DATA);
  packetHeader.SetQosTid (0);
  Ptr<Packet> packet = Create<Packet> (10);
  Ptr<WifiMacQueueItem> mpdu = Create<WifiMacQueueItem> (packet, packetHeader);
  WifiMode ackMode;

  /*
   * To initialize the manager we need to generate a transmission.
   */
  Ptr<Packet> p = Create<Packet> ();
  dev->Send (p, remoteAddress, 1);

  /**
   * This will be the thresholds table.
   * The parameters of the table are:
   * - Estimation Window Size (EWND)
   * - Maximum Tolerable Loss Probability Threshold (MTL)
   * - Opportunistic Rate Increase (and Power Decrease) Probability Threshold (ORI)
   *
   * We also calculate the needed success and failures to generate
   * a rate or power change:
   * Rate Increase or Power Decrease limits (RI-PD)
   * Rate Decrease or Power Increase limits (RD-PI)
   * Power Decrease limits (PD)
   *
   * Mode	           EWND  MTL       ORI        RI-PD  RD-PI  PD
   *                                            succ   fails  succ fails
   * OfdmRate6Mbps   8     1         0.19861    7      8      2
   * OfdmRate9Mbps   11    0.397219  0.14556    10     5      7    2
   * OfdmRate12Mbps  15    0.291121  0.189753   13     5      11   3
   * OfdmRate18Mbps  21    0.379507  0.13624    19     8      14   3
   * OfdmRate24Mbps  27    0.27248   0.174216   23     8      20   5
   * OfdmRate36Mbps  37    0.348432  0.120773   33     13     25   5
   * OfdmRate48Mbps  45    0.241546  0.0523952  43     11     35   3
   * OfdmRate54Mbps  50    0.10479   0          50     6      45   5
   *
   */

  //-----------------------------------------------------------------------------------------------------

  /*
   * RRPAA initiates with minimal rate and maximal power.
   */
  WifiTxVector txVector = manager->GetDataTxVector (packetHeader);
  WifiMode mode = txVector.GetMode ();
  int power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 6000000, "RRPAA: Initial data rate wrong"); //802.11a minimal rate is 6Mbps
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Initial power level wrong");

  //-----------------------------------------------------------------------------------------------------

  /*
   * As RRPAA starts with the 6Mbps rate, 7 successful transmissions are needed for RRPAA to increase rate.
   * 1/8 = 0.125
   */

  /**
   * Test that 6 is not enough.
   */
  for (int i = 0; i < 6; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 6000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  /**
   * Test that 7 is enough.
   */
  manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 9000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * 5 failures are needed to make the rate decrease again.
   * 5/11 = 0.45
   */
  for (int i = 0; i < 4; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 9000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  manager->ReportDataFailed (mpdu);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 6000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /**
   * Increase rate until maximal rate.
   */

  for (int i = 0; i < 7; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 9000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 10; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 12000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 13; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 18000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 19; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 24000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 23; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 33; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 48000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 43; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 54000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /**
   * Now we need more successful transmissions to make power decrease.
   * As we are at maximal rate, the power is decreased when it is sure that the failures
   * will not generate a rate decrease.
   */

  for (int i = 0; i < 49; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 54000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 54000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /**
   * As rate hasn't change the same amount of success are needed.
   * After 16*45 successful transmissions power is decreased to zero.
   */

  for (int i = 0; i < 16 * 50; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 54000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 0, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /**
   * After 6 failures the power should be increased.
   */

  for (int i = 0; i < 6; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 54000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 1, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 16*6 failed transmissions power is increase to 17.
   */

  for (int i = 0; i < 16 * 6; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 54000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * After 6 more failures the rate should be decreased.
   */

  for (int i = 0; i < 6; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 48000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  /*
   * Now 11 failures are needed to decrease rate again.
   */

  for (int i = 0; i < 11; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * Test power decrement when loss probability is between MTL and ORI.
   * As we are at rate 36 Mbps we need at least 25 successful transmissions
   * and 5 failures.
   */

  for (int i = 0; i < 25; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 17, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 5; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 5; i++)
    {
      manager->ReportDataFailed (mpdu);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 16, "RRPAA: Incorrect value of power level");

  for (int i = 0; i < 25; i++)
    {
      manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 15, "RRPAA: Incorrect value of power level");

  //-----------------------------------------------------------------------------------------------------

  /*
   * Repeat the previous test until power 0 is reached.
   */

  for (int i = 0; i < 16; i++)
    {
      for (int j = 0; j < 25; j++)
        {
          manager->ReportDataOk (mpdu, 0, ackMode, 0, txVector);
        }

      for (int j = 0; j < 5; j++)
        {
          manager->ReportDataFailed (mpdu);
        }
    }

  txVector = manager->GetDataTxVector (packetHeader);
  mode = txVector.GetMode ();
  power = (int) txVector.GetTxPowerLevel ();

  NS_TEST_ASSERT_MSG_EQ (mode.GetDataRate (txVector.GetChannelWidth (), txVector.GetGuardInterval (), 1), 36000000, "RRPAA: Incorrect vale of data rate");
  NS_TEST_ASSERT_MSG_EQ (power, 0, "RRPAA: Incorrect value of power level");

  Simulator::Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();

}

void
PowerRateAdaptationTest::DoRun (void)
{

  TestParf ();
  TestAparf ();
  TestRrpaa ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Power Rate Adaptation Test Suite
 */
class PowerRateAdaptationTestSuite : public TestSuite
{
public:
  PowerRateAdaptationTestSuite ();
};

PowerRateAdaptationTestSuite::PowerRateAdaptationTestSuite ()
  : TestSuite ("wifi-power-rate-adaptation", UNIT)
{
  AddTestCase (new PowerRateAdaptationTest, TestCase::QUICK);
}

static PowerRateAdaptationTestSuite g_powerRateAdaptationTestSuite; ///< the test suite
