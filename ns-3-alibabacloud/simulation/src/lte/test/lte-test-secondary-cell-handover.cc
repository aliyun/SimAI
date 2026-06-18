/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Alexander Krotov
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 *
 */

#include <ns3/friis-spectrum-propagation-loss.h>
#include <ns3/log.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-helper.h>
#include <ns3/test.h>
#include <ns3/point-to-point-epc-helper.h>
#include <ns3/mobility-helper.h>
#include <ns3/boolean.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/log.h>
#include <ns3/simulator.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/internet-stack-helper.h>
#include <ns3/lte-ue-rrc.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteSecondaryCellHandoverTest");

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test measurement-based handover to secondary cell.
 */

class LteSecondaryCellHandoverTestCase : public TestCase {
public:
  /**
   * \brief Creates an instance of the measurement-based secondary cell handover test case.
   * \param name name of the test case
   * \param useIdealRrc if true, simulation uses Ideal RRC protocol, otherwise
   *                    simulation uses Real RRC protocol
   */

  LteSecondaryCellHandoverTestCase (std::string name, bool useIdealRrc);

  /**
   * \brief Shutdown cellId by reducing its power to 1 dBm.
   * \param cellId ID of the cell to shutdown
   */
  void ShutdownCell (uint32_t cellId);

  /**
   * \brief Callback method indicating start of UE handover
   * \param imsi The IMSI
   * \param sourceCellId The source cell ID
   * \param rnti The RNTI
   * \param targetCellId The target cell ID
   */
  void UeHandoverStartCallback (uint64_t imsi,
                                uint16_t sourceCellId, uint16_t rnti,
                                uint16_t targetCellId);



private:
  /**
   * \brief Run a simulation.
   */
  virtual void DoRun ();

  /**
   * \brief Verify that handover has occurred during the simulation.
   */
  virtual void DoTeardown ();

  bool m_useIdealRrc; ///< whether LTE is configured to use ideal RRC
  uint8_t m_numberOfComponentCarriers; ///< Number of component carriers

  Ptr<LteEnbNetDevice> m_sourceEnbDev; ///< Source eNB device

  bool m_hasUeHandoverStarted; ///< true if UE started handover
};

LteSecondaryCellHandoverTestCase::LteSecondaryCellHandoverTestCase (std::string name, bool useIdealRrc)
    : TestCase {name},
      m_useIdealRrc {useIdealRrc},
      m_numberOfComponentCarriers {2},
      m_hasUeHandoverStarted {false}
{
}

void
LteSecondaryCellHandoverTestCase::ShutdownCell (uint32_t cellId)
{
  Ptr<LteEnbPhy> phy = m_sourceEnbDev->GetPhy (cellId - 1);
  phy->SetTxPower (1);
}

void
LteSecondaryCellHandoverTestCase::UeHandoverStartCallback (uint64_t imsi,
                            uint16_t sourceCellId, uint16_t rnti,
                            uint16_t targetCellId)
{
  NS_LOG_FUNCTION (this << imsi << sourceCellId << rnti << targetCellId);
  m_hasUeHandoverStarted = true;
}

void
LteSecondaryCellHandoverTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << GetName ());

  Config::SetDefault ("ns3::LteEnbNetDevice::DlEarfcn", UintegerValue (100));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlEarfcn", UintegerValue (100 + 18000));
  Config::SetDefault ("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue (25));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue (25));
  Config::SetDefault ("ns3::LteUeNetDevice::DlEarfcn", UintegerValue (100));

  // Create helpers.
  auto lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetAttribute ("PathlossModel", TypeIdValue (ns3::FriisSpectrumPropagationLossModel::GetTypeId ()));
  lteHelper->SetAttribute ("UseIdealRrc", BooleanValue (m_useIdealRrc));
  lteHelper->SetAttribute ("NumberOfComponentCarriers", UintegerValue (m_numberOfComponentCarriers));

  // Configure handover algorithm.
  lteHelper->SetHandoverAlgorithmType ("ns3::A3RsrpHandoverAlgorithm");
  lteHelper->SetHandoverAlgorithmAttribute ("Hysteresis", DoubleValue (1.5));
  lteHelper->SetHandoverAlgorithmAttribute ("TimeToTrigger", TimeValue (MilliSeconds (128)));

  auto epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // Create nodes.
  auto enbNode = CreateObject<Node> ();
  auto ueNode = CreateObject<Node> ();

  // Setup node mobility.
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNode);
  mobility.Install (ueNode);

  // Physical layer.
  m_sourceEnbDev = DynamicCast<LteEnbNetDevice> (lteHelper->InstallEnbDevice (enbNode).Get (0));
  auto ueDevs = lteHelper->InstallUeDevice (ueNode);
  auto ueDev = DynamicCast<LteUeNetDevice> (ueDevs.Get (0));

  // Network layer.
  InternetStackHelper internet;
  internet.Install (ueNode);
  epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDev));

  uint16_t sourceCellId = m_sourceEnbDev->GetCellId ();
  Simulator::Schedule (Seconds (0.5), &LteSecondaryCellHandoverTestCase::ShutdownCell, this, sourceCellId);

  // Setup traces.
  ueDev->GetRrc ()->TraceConnectWithoutContext ("HandoverStart", MakeCallback (&LteSecondaryCellHandoverTestCase::UeHandoverStartCallback, this));

  std::map<uint8_t, Ptr<ComponentCarrierUe>> ueCcMap = ueDev->GetCcMap ();
  ueDev->SetDlEarfcn (ueCcMap.at (0)->GetDlEarfcn ());
  lteHelper->Attach (ueDev, m_sourceEnbDev, 0);

  // Run simulation.
  Simulator::Stop (Seconds (1));
  Simulator::Run ();
  Simulator::Destroy ();
}

void
LteSecondaryCellHandoverTestCase::DoTeardown ()
{
  NS_LOG_FUNCTION (this);
  NS_TEST_ASSERT_MSG_EQ (m_hasUeHandoverStarted, true, "Handover did not occur");
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief LTE measurement-based handover to secondary cell test suite.
 */
class LteSecondaryCellHandoverTestSuite : public TestSuite {
public:
    LteSecondaryCellHandoverTestSuite ();
};

LteSecondaryCellHandoverTestSuite::LteSecondaryCellHandoverTestSuite ()
    : TestSuite {"lte-secondary-cell-handover", SYSTEM}
{
  AddTestCase (new LteSecondaryCellHandoverTestCase ("Ideal RRC", true), TestCase::QUICK);
  AddTestCase (new LteSecondaryCellHandoverTestCase ("Real RRC", false), TestCase::QUICK);
}

static LteSecondaryCellHandoverTestSuite g_lteSecondaryCellHandoverTestSuiteInstance;
