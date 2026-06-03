/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Ritsumeikan University, Shiga, Japan
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
 * Author:
 *  Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */

#include <ns3/log.h>
#include <ns3/core-module.h>
#include <ns3/lr-wpan-module.h>
#include <ns3/propagation-loss-model.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/simulator.h>
#include <ns3/single-model-spectrum-channel.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/packet.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("lr-wpan-slotted-csma-test");

/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief Test the correct allocation of DIRECT transmissions in the
 *        contention access period (CAP) of the superframe
 *        (Slotted CSMA-CA algorithm).
 */
class LrWpanSlottedCsmacaTestCase : public TestCase
{
public:
  LrWpanSlottedCsmacaTestCase ();
  virtual ~LrWpanSlottedCsmacaTestCase ();



private:
  /**
   * \brief Function called when McpsDataConfirm is hit.
   * \param testcase The TestCase.
   * \param dev The LrWpanNetDevice.
   * \param params The McpsDataConfirm parameters.
   */
  static void TransEndIndication (LrWpanSlottedCsmacaTestCase *testcase,
                                  Ptr<LrWpanNetDevice> dev,
                                  McpsDataConfirmParams params);
  /**
   * \brief Function called when McpsDataIndication is hit.
   * \param testcase The TestCase.
   * \param dev The LrWpanNetDevice.
   * \param params The McpsDataIndication parameters.
   * \param p The received packet.
   */
  static void DataIndicationCoordinator (LrWpanSlottedCsmacaTestCase *testcase,
                                         Ptr<LrWpanNetDevice> dev,
                                         McpsDataIndicationParams params,
                                         Ptr<Packet> p);
  /**
   * \brief Function called when MlmeStartConfirm is hit.
   * \param testcase The TestCase.
   * \param dev The LrWpanNetDevice.
   * \param params The MlmeStartConfirm parameters.
   */
  static void StartConfirm (LrWpanSlottedCsmacaTestCase *testcase,
                            Ptr<LrWpanNetDevice> dev,
                            MlmeStartConfirmParams params);

  /**
   * \brief Function called on each Superframe status change (CAP|CFP|INACTIVE).
   * \param testcase The TestCase.
   * \param dev The LrWpanNetDevice.
   * \param oldValue The previous superframe status.
   * \param newValue THe new superframe status.
   */
  static void IncomingSuperframeStatus (LrWpanSlottedCsmacaTestCase *testcase,
                                        Ptr<LrWpanNetDevice> dev,
                                        SuperframeStatus oldValue,
                                        SuperframeStatus newValue);

  /**
   * \brief Function called to indicated the calculated transaction cost in slotted CSMA-CA
   * \param testcase The TestCase.
   * \param dev The LrWpanNetDevice.
   * \param trans The transaction cost in symbols.
   */
  static void TransactionCost (LrWpanSlottedCsmacaTestCase *testcase,
                               Ptr<LrWpanNetDevice> dev,
                               uint32_t trans);

  virtual void DoRun (void);

  Time m_startCap;      //!< The time of the start of the Contention Access Period (CAP).
  Time m_apBoundary;    //!< Indicates the time after the calculation of the transaction cost (A boundary of an Active Period in the CAP)
  Time m_sentTime;      //!< Indicates the time after a successful transmission.
  uint32_t m_transCost; //!< The current transaction cost in symbols.
};


LrWpanSlottedCsmacaTestCase::LrWpanSlottedCsmacaTestCase ()
  : TestCase ("Lrwpan: Slotted CSMA-CA test")
{
  m_transCost = 0;
}

LrWpanSlottedCsmacaTestCase::~LrWpanSlottedCsmacaTestCase ()
{

}

void
LrWpanSlottedCsmacaTestCase::TransEndIndication (LrWpanSlottedCsmacaTestCase *testcase, Ptr<LrWpanNetDevice> dev, McpsDataConfirmParams params)
{
  // In the case of transmissions with the acknowledgment flag activated, the transmission is only
  // successful if the acknowledgment was received.
  if (params.m_status == LrWpanMcpsDataConfirmStatus::IEEE_802_15_4_SUCCESS)
    {
      NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << "s Transmission successfully sent");
      testcase->m_sentTime = Simulator::Now ();
    }
}

void
LrWpanSlottedCsmacaTestCase::DataIndicationCoordinator (LrWpanSlottedCsmacaTestCase *testcase, Ptr<LrWpanNetDevice> dev, McpsDataIndicationParams params, Ptr<Packet> p)
{
  NS_LOG_UNCOND (Simulator::Now ().As(Time::S) << "s Coordinator Received DATA packet (size " << p->GetSize () << " bytes)");
}

void
LrWpanSlottedCsmacaTestCase::StartConfirm (LrWpanSlottedCsmacaTestCase *testcase, Ptr<LrWpanNetDevice> dev, MlmeStartConfirmParams params)
{
  NS_LOG_UNCOND (Simulator::Now ().As(Time::S) << "s Beacon Sent");
}

void
LrWpanSlottedCsmacaTestCase::IncomingSuperframeStatus (LrWpanSlottedCsmacaTestCase *testcase, Ptr<LrWpanNetDevice> dev,SuperframeStatus oldValue, SuperframeStatus newValue)
{
  if (newValue == SuperframeStatus::CAP)
    {
      testcase->m_startCap = Simulator::Now ();
      NS_LOG_UNCOND (Simulator::Now ().As(Time::S) << "s Incoming superframe CAP starts");
    }
}

void
LrWpanSlottedCsmacaTestCase::TransactionCost (LrWpanSlottedCsmacaTestCase *testcase, Ptr<LrWpanNetDevice> dev, uint32_t trans)
{
  testcase->m_apBoundary = Simulator::Now ();
  testcase->m_transCost = trans;
  NS_LOG_UNCOND (Simulator::Now ().As(Time::S) << "s Transaction Cost is:" << trans);

}





void
LrWpanSlottedCsmacaTestCase::DoRun ()
{
  // Create 2 nodes, and a NetDevice for each one
  Ptr<Node> n0 = CreateObject <Node> ();
  Ptr<Node> n1 = CreateObject <Node> ();

  Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice> ();

  dev0->SetAddress (Mac16Address ("00:01"));
  dev1->SetAddress (Mac16Address ("00:02"));

  // Each device must be attached to the same channel
  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  Ptr<LogDistancePropagationLossModel> propModel = CreateObject<LogDistancePropagationLossModel> ();
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);

  dev0->SetChannel (channel);
  dev1->SetChannel (channel);

  // To complete configuration, a LrWpanNetDevice must be added to a node
  n0->AddDevice (dev0);
  n1->AddDevice (dev1);


  // Set mobility
  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0,0,0));
  dev0->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();

  sender1Mobility->SetPosition (Vector (0,10,0));
  dev1->GetPhy ()->SetMobility (sender1Mobility);


  // MAC layer and CSMA-CA callback hooks

  MlmeStartConfirmCallback cb0;
  cb0 = MakeBoundCallback (&LrWpanSlottedCsmacaTestCase::StartConfirm, this, dev0);
  dev0->GetMac ()->SetMlmeStartConfirmCallback (cb0);

  McpsDataConfirmCallback cb1;
  cb1 = MakeBoundCallback (&LrWpanSlottedCsmacaTestCase::TransEndIndication, this, dev1);
  dev1->GetMac ()->SetMcpsDataConfirmCallback (cb1);

  LrWpanMacTransCostCallback cb2;
  cb2 = MakeBoundCallback (&LrWpanSlottedCsmacaTestCase::TransactionCost, this, dev1);
  dev1->GetCsmaCa ()->SetLrWpanMacTransCostCallback (cb2);

  McpsDataIndicationCallback cb5;
  cb5 = MakeBoundCallback (&LrWpanSlottedCsmacaTestCase::DataIndicationCoordinator, this, dev0);
  dev0->GetMac ()->SetMcpsDataIndicationCallback (cb5);


  // Connect to trace in the MAC layer
  dev1->GetMac ()->TraceConnectWithoutContext ("MacIncSuperframeStatus",
                                               MakeBoundCallback (&LrWpanSlottedCsmacaTestCase::IncomingSuperframeStatus, this, dev1));


  // Manual Device Association
  // Note: We manually associate dev1 device to a PAN coordinator
  //       because currently there is no automatic association behavior;
  //       The PAN COORDINATOR does not need to associate, set
  //       PAN Id or its own coordinator id, these are set
  //       by the MLME-start.request primitive when used.

  dev1->GetMac ()->SetPanId (5);
  dev1->GetMac ()->SetAssociatedCoor (Mac16Address ("00:01"));


  // Dev0 sets the start time for beacons
  MlmeStartRequestParams params;
  params.m_panCoor = true;
  params.m_PanId = 5;
  params.m_bcnOrd = 14;
  params.m_sfrmOrd = 6;
  Simulator::ScheduleWithContext (1, Seconds (2.0),
                                  &LrWpanMac::MlmeStartRequest,
                                  dev0->GetMac (), params);

  // Dev1 sets the transmission of data packet

  Ptr<Packet> p1 = Create<Packet> (5);  // 5 bytes of dummy data
  McpsDataRequestParams params2;
  params2.m_dstPanId = 5;
  params2.m_srcAddrMode = SHORT_ADDR;
  params2.m_dstAddrMode = SHORT_ADDR;
  params2.m_dstAddr = Mac16Address ("00:01");
  params2.m_msduHandle = 0;


  // Beacon-enabled | Device to Coordinator | Direct transmission
  Simulator::ScheduleWithContext (1, Seconds (2.93),
                                  &LrWpanMac::McpsDataRequest,
                                  dev1->GetMac (), params2, p1);


  Simulator::Stop (Seconds (4));
  Simulator::Run ();

  Time activePeriodsSum;
  Time transactionTime;
  uint64_t      symbolRate;
  uint32_t activePeriodSize = 20;
  double boundary;


  // Verifies that the CCA checks and the rest of the transaction runs
  // on a boundary of an Active Period in the slotted CSMA-CA.

  symbolRate = (uint64_t) dev1->GetMac ()->GetPhy ()->GetDataOrSymbolRate (false);
  activePeriodsSum = m_apBoundary - m_startCap;
  boundary =  (activePeriodsSum.GetMicroSeconds () * 1000 * 1000 * symbolRate) % activePeriodSize;

  NS_TEST_EXPECT_MSG_EQ (boundary, 0, "Error, the transaction is not calculated on a boundary of an Active Period in the CAP");


  // Slotted CSMA-CA needs to precalculate the cost of the transaction to ensure there
  // is enough time in the CAP to complete the transmission. The following checks that such
  // pre-calculation matches the time it took to complete the transmission.

  // The calculated transaction includes the IFS time, so we need to subtract its value to compare it.
  // MPDU = MAC Header + MSDU (payload)
  // Mac Header = 13 bytes
  // If the MPDU is > aMaxSIFSFrameSize (18 bytes) then IFS = LIFS (40 symbols), else  IFS = SIFS (12 symbols)

  uint32_t ifsSize;
  if (p1->GetSize () > 18)
    {
      ifsSize = 40;
    }
  else
    {
      ifsSize = 12;
    }

  // The transaction cost here includes the ifsSize and the turnAroundTime (Tx->Rx)
  // therefore we subtract these before the final comparison
  //
  //  Transmission Start                     Transmission End
  //     |                                     |
  //     +-------+--------------------+--------+------------------------+------+
  //     | 2 CCA |  TurnAround(Rx->Tx)| Data   |  TurnAround(Tx->Rx)    |  IFS |
  //     +-------+--------------------+--------+------------------------+------+

  // TODO: This test need some rework to make it more clear

  transactionTime = Seconds ((double)(m_transCost-(ifsSize + 12))  / symbolRate);
  NS_LOG_UNCOND ("Transmission start time(On a boundary): "<<m_apBoundary.As(Time::S));
  NS_LOG_UNCOND ("Transmission End time (McpsData.confirm): "<<m_sentTime.As(Time::S));

  NS_TEST_EXPECT_MSG_EQ (m_sentTime,(m_apBoundary + transactionTime),"Error, the transaction time is not the expected value");

  Simulator::Destroy ();

}


/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan Slotted CSMA-CA TestSuite
 */

class LrWpanSlottedCsmacaTestSuite : public TestSuite
{
public:
  LrWpanSlottedCsmacaTestSuite ();
};

LrWpanSlottedCsmacaTestSuite::LrWpanSlottedCsmacaTestSuite ()
  : TestSuite ("lr-wpan-slotted-csmaca", UNIT)
{
  AddTestCase (new LrWpanSlottedCsmacaTestCase, TestCase::QUICK);
}

static LrWpanSlottedCsmacaTestSuite lrWpanSlottedCsmacaTestSuite; //!< Static variable for test initialization


