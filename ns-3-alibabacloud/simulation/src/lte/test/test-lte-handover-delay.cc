/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Magister Solutions
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
 * Author: Budiarto Herman <budiarto.herman@magister.fi>
 *         Alexander Krotov <krotov@iitp.ru>
 */

#include <ns3/test.h>

#include <ns3/log.h>
#include <ns3/nstime.h>
#include <ns3/callback.h>
#include <ns3/config.h>
#include <ns3/boolean.h>
#include <ns3/simulator.h>

#include <ns3/node-container.h>
#include <ns3/net-device-container.h>
#include <ns3/ipv4-interface-container.h>

#include <ns3/lte-helper.h>
#include <ns3/point-to-point-epc-helper.h>
#include <ns3/internet-stack-helper.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/ipv4-address-helper.h>
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/mobility-helper.h>

#include <ns3/data-rate.h>
#include <ns3/ipv4-static-routing.h>
#include <ns3/position-allocator.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteHandoverDelayTest");

/**
 * \ingroup tests lte-test
 *
 * \brief Verifying that the time needed for handover is under a
 * specified threshold.
 */

class LteHandoverDelayTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param numberOfComponentCarriers number of component carriers
   * \param useIdealRrc if true, use the ideal RRC
   * \param handoverTime the time of handover
   * \param delayThreshold the delay threshold
   * \param simulationDuration duration of the simulation
   */
  LteHandoverDelayTestCase (uint8_t numberOfComponentCarriers, bool useIdealRrc, Time handoverTime,
                            Time delayThreshold, Time simulationDuration)
    : TestCase ("Verifying that the time needed for handover is under a specified threshold"),
      m_numberOfComponentCarriers (numberOfComponentCarriers),
      m_useIdealRrc (useIdealRrc),
      m_handoverTime (handoverTime),
      m_delayThreshold (delayThreshold),
      m_simulationDuration (simulationDuration),
      m_ueHandoverStart (Seconds (0)),
      m_enbHandoverStart (Seconds (0))
  {}

private:
  virtual void DoRun (void);

  /**
   * UE handover start callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellid the cell ID
   * \param rnti the RNTI
   * \param targetCellId the target cell ID
   */
  void UeHandoverStartCallback (std::string context, uint64_t imsi,
                                uint16_t cellid, uint16_t rnti, uint16_t targetCellId);
  /**
   * UE handover end OK callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellid the cell ID
   * \param rnti the RNTI
   */
  void UeHandoverEndOkCallback (std::string context, uint64_t imsi,
                                uint16_t cellid, uint16_t rnti);
  /**
   * ENB handover start callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellid the cell ID
   * \param rnti the RNTI
   * \param targetCellId the target cell ID
   */
  void EnbHandoverStartCallback (std::string context, uint64_t imsi,
                                 uint16_t cellid, uint16_t rnti, uint16_t targetCellId);
  /**
   * ENB handover end OK callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellid the cell ID
   * \param rnti the RNTI
   */
  void EnbHandoverEndOkCallback (std::string context, uint64_t imsi,
                                 uint16_t cellid, uint16_t rnti);

  uint8_t m_numberOfComponentCarriers; ///< Number of component carriers
  bool m_useIdealRrc; ///< use ideal RRC?
  Time m_handoverTime; ///< handover time
  Time m_delayThreshold; ///< the delay threshold
  Time m_simulationDuration; ///< the simulation duration

  Time m_ueHandoverStart; ///< UE handover start time
  Time m_enbHandoverStart; ///< ENB handover start time
};


void
LteHandoverDelayTestCase::DoRun ()
{
  NS_LOG_INFO ("-----test case: ideal RRC = " << m_useIdealRrc << " handover time = "
                                              << m_handoverTime.As (Time::S) << "-----");

  /*
   * Helpers.
   */
  auto epcHelper = CreateObject<PointToPointEpcHelper> ();

  auto lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetEpcHelper (epcHelper);
  lteHelper->SetAttribute ("UseIdealRrc", BooleanValue (m_useIdealRrc));
  lteHelper->SetAttribute ("NumberOfComponentCarriers", UintegerValue (m_numberOfComponentCarriers));

  auto ccHelper = CreateObject<CcHelper> ();
  ccHelper->SetUlEarfcn (100 + 18000);
  ccHelper->SetDlEarfcn (100);
  ccHelper->SetUlBandwidth (25);
  ccHelper->SetDlBandwidth (25);
  ccHelper->SetNumberOfComponentCarriers (m_numberOfComponentCarriers);

  /*
   * Physical layer.
   *
   * eNodeB 0                    UE                      eNodeB 1
   *
   *    x ----------------------- x ----------------------- x
   *              500 m                      500 m
   */
  // Create nodes.
  NodeContainer enbNodes;
  enbNodes.Create (2);
  auto ueNode = CreateObject<Node> ();

  // Setup mobility
  auto posAlloc = CreateObject<ListPositionAllocator> ();
  posAlloc->Add (Vector (0, 0, 0));
  posAlloc->Add (Vector (1000, 0, 0));
  posAlloc->Add (Vector (500, 0, 0));

  MobilityHelper mobilityHelper;
  mobilityHelper.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobilityHelper.SetPositionAllocator (posAlloc);
  mobilityHelper.Install (enbNodes);
  mobilityHelper.Install (ueNode);

  /*
   * Link layer.
   */
  auto enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  auto ueDev = lteHelper->InstallUeDevice (ueNode).Get (0);

  /*
   * Network layer.
   */
  InternetStackHelper inetStackHelper;
  inetStackHelper.Install (ueNode);
  Ipv4InterfaceContainer ueIfs;
  ueIfs = epcHelper->AssignUeIpv4Address (ueDev);

  // Setup traces.
  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
                   MakeCallback (&LteHandoverDelayTestCase::UeHandoverStartCallback, this));
  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
                   MakeCallback (&LteHandoverDelayTestCase::UeHandoverEndOkCallback, this));

  Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                   MakeCallback (&LteHandoverDelayTestCase::EnbHandoverStartCallback, this));
  Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                   MakeCallback (&LteHandoverDelayTestCase::EnbHandoverEndOkCallback, this));

  // Prepare handover.
  lteHelper->AddX2Interface (enbNodes);
  lteHelper->Attach (ueDev, enbDevs.Get (0));
  lteHelper->HandoverRequest (m_handoverTime, ueDev, enbDevs.Get (0), enbDevs.Get (1));

  // Run simulation.
  Simulator::Stop (m_simulationDuration);
  Simulator::Run ();
  Simulator::Destroy ();

} // end of void LteHandoverDelayTestCase::DoRun ()


void
LteHandoverDelayTestCase::UeHandoverStartCallback (std::string context,
                                                   uint64_t imsi, uint16_t cellid, uint16_t rnti, uint16_t targetCellId)
{
  NS_LOG_FUNCTION (this << context);
  m_ueHandoverStart = Simulator::Now ();
}

void
LteHandoverDelayTestCase::UeHandoverEndOkCallback (std::string context,
                                                   uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << context);
  NS_ASSERT (m_ueHandoverStart > Seconds (0));
  Time delay = Simulator::Now () - m_ueHandoverStart;

  NS_LOG_DEBUG (this << " UE delay = " << delay.As (Time::S));
  NS_TEST_ASSERT_MSG_LT (delay, m_delayThreshold,
                         "UE handover delay is higher than the allowed threshold "
                         << "(ideal RRC = " << m_useIdealRrc
                         << " handover time = " << m_handoverTime.As (Time::S) << ")");
}


void
LteHandoverDelayTestCase::EnbHandoverStartCallback (std::string context,
                                                    uint64_t imsi, uint16_t cellid, uint16_t rnti, uint16_t targetCellId)
{
  NS_LOG_FUNCTION (this << context);
  m_enbHandoverStart = Simulator::Now ();
}

void
LteHandoverDelayTestCase::EnbHandoverEndOkCallback (std::string context,
                                                    uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << context);
  NS_ASSERT (m_enbHandoverStart > Seconds (0));
  Time delay = Simulator::Now () - m_enbHandoverStart;

  NS_LOG_DEBUG (this << " eNodeB delay = " << delay.As (Time::S));
  NS_TEST_ASSERT_MSG_LT (delay, m_delayThreshold,
                         "eNodeB handover delay is higher than the allowed threshold "
                         << "(ideal RRC = " << m_useIdealRrc
                         << " handover time = " << m_handoverTime.As (Time::S) << ")");
}



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Handover Delay Test Suite
 */

static class LteHandoverDelayTestSuite : public TestSuite
{
public:
  LteHandoverDelayTestSuite ()
    : TestSuite ("lte-handover-delay", TestSuite::SYSTEM)
  {
    //LogComponentEnable ("LteHandoverDelayTest", LOG_PREFIX_TIME);
    //LogComponentEnable ("LteHandoverDelayTest", LOG_DEBUG);
    //LogComponentEnable ("LteHandoverDelayTest", LOG_INFO);

    // HANDOVER DELAY TEST CASES WITH IDEAL RRC (THRESHOLD = 0.005 sec)

    for (Time handoverTime = Seconds (0.100); handoverTime < Seconds (0.110);
         handoverTime += Seconds (0.001))
      {
        // arguments: useIdealRrc, handoverTime, delayThreshold, simulationDuration
        AddTestCase (
          new LteHandoverDelayTestCase (1, true, handoverTime, Seconds (0.005),
                                        Seconds (0.200)), TestCase::QUICK);
        AddTestCase (
          new LteHandoverDelayTestCase (2, true, handoverTime, Seconds (0.005),
                                        Seconds (0.200)), TestCase::QUICK);
        AddTestCase (
          new LteHandoverDelayTestCase (4, true, handoverTime, Seconds (0.005),
                                        Seconds (0.200)), TestCase::QUICK);
      }

    // HANDOVER DELAY TEST CASES WITH REAL RRC (THRESHOLD = 0.020 sec)

    for (Time handoverTime = Seconds (0.100); handoverTime < Seconds (0.110);
         handoverTime += Seconds (0.001))
      {
        // arguments: useIdealRrc, handoverTime, delayThreshold, simulationDuration
        AddTestCase (
          new LteHandoverDelayTestCase (1, false, handoverTime, Seconds (0.020),
                                        Seconds (0.200)), TestCase::QUICK);
        AddTestCase (
          new LteHandoverDelayTestCase (2, false, handoverTime, Seconds (0.020),
                                        Seconds (0.200)), TestCase::QUICK);
        AddTestCase (
          new LteHandoverDelayTestCase (4, false, handoverTime, Seconds (0.020),
                                        Seconds (0.200)), TestCase::QUICK);
      }
  }
} g_lteHandoverDelayTestSuite; ///< the test suite
