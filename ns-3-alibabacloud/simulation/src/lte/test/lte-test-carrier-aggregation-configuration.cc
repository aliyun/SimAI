/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Zoraze Ali <zoraze.ali@cttc.es>
 *
 */

#include <ns3/object.h>
#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/simulator.h>
#include <ns3/ptr.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/node-container.h>
#include <ns3/mobility-helper.h>
#include <ns3/net-device-container.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/lte-helper.h>
#include <ns3/lte-spectrum-value-helper.h>
#include <ns3/callback.h>

using namespace ns3;

/**
 * This test suite verifies following two things:
 *
 *  1. When CA is enabled and UE carriers configuration is different than the default one,
 *     we check that the UE is configured properly once it receives 
 *     RRC Connection Reconfiguration message from eNB.
 *
 *  2. A user can configure 2 or more eNBs and UEs with different configuration parameters,
 *     i.e, each eNB and UE can have different EARFCN and Bandwidths and a UE connects to
 *     an eNB with similar DL EARFCN.
 *     Here we check it with CA enabled but the end results will be the same if CA is not
 *     enabled and we have more than one eNBs and UEs with different configurations.
 *
 * Since we do not need EPC to test the configuration, this test only simulates
 * the LTE radio access with RLC SM.
 *
 * Test 1 tests that the UE is configured properly after receiving RRC Connection
 * Reconfiguration message from the eNB, which will overwrite UE default configuration
 * done in LteHelper for the sake of creating PHY and MAC instances equal to the number
 * of component carriers.
 *
 * Test 2 tests that in a simulation scenario every eNB or UE can be configured with
 * different EARFCNs and Bandwidths. This will check that the eNBs and UEs configuration
 * is not static, as reported in BUG 2840.
 */

/// ConfigToCheck structure
struct ConfigToCheck
{
  uint16_t m_dlBandwidth; ///< Downlink bandwidth
  uint16_t m_ulBandwidth; ///< Uplink bandwidth
  uint32_t m_dlEarfcn; ///< Downlink EARFCN
  uint32_t m_ulEarfcn; ///< Uplink EARFCN
};

NS_LOG_COMPONENT_DEFINE ("TestCarrierAggregationConfig");

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Carrier aggregation configuration test case.
 */
class CarrierAggregationConfigTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param numberOfNodes Total Number of eNBs and UEs
   * \param numberOfComponentCarriers  Total number of component carriers
   * \param configToCheck Vector containing all the configurations to check
   * \param simulationDuration Duration of the simulation
   */
  CarrierAggregationConfigTestCase (uint32_t numberOfNodes, uint16_t numberOfComponentCarriers, std::vector<ConfigToCheck> configToCheck, Time simulationDuration)
    : TestCase (BuildNameString (numberOfNodes, numberOfComponentCarriers, configToCheck, simulationDuration)),
      m_numberOfNodes(numberOfNodes),
      m_numberOfComponentCarriers(numberOfComponentCarriers),
      m_configToCheck(configToCheck),
      m_simulationDuration (simulationDuration)
  {
    m_connectionCounter = 0.0;
  }

private:
  virtual void DoRun (void);

  /**
   * Build name string function
   *
   * \param numberOfNodes Total Number of eNBs and UEs
   * \param numberOfComponentCarriers  Total number of component carriers
   * \param configToCheck Vector containing all the configurations to check
   * \param simulationDuration Duration of the simulation
   * \returns the name string
   */
  std::string BuildNameString (uint32_t numberOfNodes, uint16_t numberOfComponentCarriers, std::vector<ConfigToCheck> configToCheck, Time simulationDuration);
  /**
   * Evaluate function
   *
   * \param context The context
   * \param ueRrc Pointer to the UE RRC
   * \param sCellToAddModList List of the configuration parameters for secondary cell
   */
  void Evaluate (std::string context, Ptr<LteUeRrc> ueRrc, std::list<LteRrcSap::SCellToAddMod> sCellToAddModList);
  /**
   * Equally spaced component carriers function
   *
   * \return Vector of maps containing the per component carrier configuration
   */
  std::vector<std::map<uint16_t, ConfigToCheck> > EquallySpacedCcs ();

  uint32_t m_numberOfNodes; ///< Numer of nodes
  uint16_t m_numberOfComponentCarriers; ///< Number of component carriers
  std::vector<ConfigToCheck> m_configToCheck; ///< Vector containing all the configurations to check
  uint32_t m_connectionCounter; ///< Connection counter
  Time m_simulationDuration; ///< Simulation duration
  std::vector<std::map<uint16_t, ConfigToCheck> > m_configToCheckContainer; ///< Vector of maps containing the per component carrier configuration
};

std::string
CarrierAggregationConfigTestCase::BuildNameString (uint32_t numberOfNodes, uint16_t numberOfComponentCarriers, std::vector<ConfigToCheck> configToCheck, Time simulationDuration)
{
  std::ostringstream oss;
  oss << " nodes " << numberOfNodes << " carriers " << numberOfComponentCarriers
      << " configurations " << configToCheck.size ()
      << " duration " << simulationDuration;
  return oss.str ();
}

std::vector<std::map<uint16_t, ConfigToCheck> >
CarrierAggregationConfigTestCase::EquallySpacedCcs ()
{
  std::vector<std::map<uint16_t, ConfigToCheck> > configToCheckContainer;

  for (auto &it: m_configToCheck)
    {
      std::map<uint16_t, ConfigToCheck> ccmap;
      uint32_t ulEarfcn = it.m_ulEarfcn;
      uint32_t dlEarfcn = it.m_dlEarfcn;
      uint32_t maxBandwidthRb = std::max<uint32_t> (it.m_ulBandwidth, it.m_dlBandwidth);

      // Convert bandwidth from RBs to kHz
      uint32_t maxBandwidthKhz = LteSpectrumValueHelper::GetChannelBandwidth (maxBandwidthRb) / 1e3;

      for (uint16_t i = 0; i < m_numberOfComponentCarriers; i++)
        {
          // Make sure we stay within the same band.
          if (LteSpectrumValueHelper::GetUplinkCarrierBand (ulEarfcn) !=
              LteSpectrumValueHelper::GetUplinkCarrierBand (it.m_ulEarfcn)
           || LteSpectrumValueHelper::GetDownlinkCarrierBand (dlEarfcn) !=
              LteSpectrumValueHelper::GetDownlinkCarrierBand (it.m_dlEarfcn))
            {
              NS_FATAL_ERROR ("Band is not wide enough to allocate " << m_numberOfComponentCarriers << " CCs");
            }

      ConfigToCheck cc;
      cc.m_dlBandwidth = it.m_dlBandwidth;
      cc.m_dlEarfcn = dlEarfcn;
      cc.m_ulBandwidth = it.m_ulBandwidth;
      cc.m_ulEarfcn = ulEarfcn;

      ccmap.insert (std::pair<uint16_t, ConfigToCheck> (i, cc));

      NS_LOG_INFO ("UL BW: " << it.m_ulBandwidth <<
                   ", DL BW: " << it.m_dlBandwidth <<
                   ", UL Earfcn: " << ulEarfcn <<
                   ", DL Earfcn: " << dlEarfcn);

      // The spacing between the center frequencies of two contiguous CCs should be multiple of 300 kHz.
      // Round spacing up to 300 kHz.
      uint32_t frequencyShift = 300 * (1 + (maxBandwidthKhz - 1) / 300);

      // Unit of EARFCN corresponds to 100kHz.
      uint32_t earfcnShift = frequencyShift / 100;
      ulEarfcn += earfcnShift;
      dlEarfcn += earfcnShift;
    }

    configToCheckContainer.push_back (ccmap);
  }

  return configToCheckContainer;
}

void
CarrierAggregationConfigTestCase::Evaluate (std::string context, Ptr<LteUeRrc> ueRrc, std::list<LteRrcSap::SCellToAddMod> sCellToAddModList)
{
  NS_LOG_INFO ("Secondary carriers configured");

  uint16_t cellId = ueRrc->GetCellId ();
  NS_LOG_INFO ("cellId " << cellId);
  NS_LOG_INFO ("m_configToCheckContainer size " << m_configToCheckContainer.size ());

  ++m_connectionCounter;

  std::map<uint16_t, ConfigToCheck> configToCheckMap;

  if (cellId == 1)
    {
      configToCheckMap = m_configToCheckContainer[cellId-1];
    }
  else
    {
      uint16_t n1 = std::max(cellId, m_numberOfComponentCarriers);
      uint16_t n2 = std::min(cellId, m_numberOfComponentCarriers);
      configToCheckMap = m_configToCheckContainer[n1-n2];
    }

  NS_LOG_INFO ("PCarrier - UL BW: " << static_cast<uint16_t> (ueRrc->GetUlBandwidth ()) <<
               ", DL BW: " << static_cast<uint16_t> (ueRrc->GetDlBandwidth ()) <<
               ", UL Earfcn: " << ueRrc->GetUlEarfcn () <<
               ", DL Earfcn: " << ueRrc->GetDlEarfcn ());

  for (auto scell: sCellToAddModList)
    {
      NS_LOG_INFO ("SCarrier - UL BW: " << static_cast<uint16_t> (scell.radioResourceConfigCommonSCell.ulConfiguration.ulFreqInfo.ulBandwidth)<<
                   ", DL BW: " << static_cast<uint16_t> (scell.radioResourceConfigCommonSCell.nonUlConfiguration.dlBandwidth) <<
                   ", UL Earfcn: " << scell.radioResourceConfigCommonSCell.ulConfiguration.ulFreqInfo.ulCarrierFreq <<
                   ", DL Earfcn: " << scell.cellIdentification.dlCarrierFreq);
    }

  ConfigToCheck pCConfig = configToCheckMap[0]; // Primary Carrier
  ConfigToCheck sCConfig; // Secondary Carriers

  NS_TEST_ASSERT_MSG_EQ (pCConfig.m_dlBandwidth, static_cast<uint16_t> (ueRrc->GetDlBandwidth ()),
                         "Primary Carrier DL bandwidth configuration failed");
  NS_TEST_ASSERT_MSG_EQ (pCConfig.m_ulBandwidth, static_cast<uint16_t> (ueRrc->GetUlBandwidth ()),
                         "Primary Carrier UL bandwidth configuration failed");
  NS_TEST_ASSERT_MSG_EQ (pCConfig.m_dlEarfcn, ueRrc->GetDlEarfcn (),
                         "Primary Carrier DL EARFCN configuration failed");
  NS_TEST_ASSERT_MSG_EQ (pCConfig.m_ulEarfcn, ueRrc->GetUlEarfcn (),
                         "Primary Carrier UL EARFCN configuration failed");

  uint32_t ConfigToCheckMapIndex = 1;

  for (auto scell: sCellToAddModList)
    {
      sCConfig = configToCheckMap[ConfigToCheckMapIndex];

      NS_TEST_ASSERT_MSG_EQ (sCConfig.m_dlBandwidth, static_cast<uint16_t> (scell.radioResourceConfigCommonSCell.nonUlConfiguration.dlBandwidth),
                             "Secondary Carrier DL bandwidth configuration failed");
      NS_TEST_ASSERT_MSG_EQ (sCConfig.m_ulBandwidth, static_cast<uint16_t> (scell.radioResourceConfigCommonSCell.ulConfiguration.ulFreqInfo.ulBandwidth),
                             "Secondary Carrier UL bandwidth configuration failed");
      NS_TEST_ASSERT_MSG_EQ (sCConfig.m_dlEarfcn, scell.cellIdentification.dlCarrierFreq,
                             "Secondary Carrier DL EARFCN configuration failed");
      NS_TEST_ASSERT_MSG_EQ (sCConfig.m_ulEarfcn, scell.radioResourceConfigCommonSCell.ulConfiguration.ulFreqInfo.ulCarrierFreq,
                             "Secondary Carrier UL EARFCN configuration failed");
      ConfigToCheckMapIndex++;
    }
}

void
CarrierAggregationConfigTestCase::DoRun ()
{
  Config::SetDefault ("ns3::LteHelper::UseCa", BooleanValue (true));
  Config::SetDefault ("ns3::LteHelper::NumberOfComponentCarriers", UintegerValue (m_numberOfComponentCarriers));
  Config::SetDefault ("ns3::LteHelper::EnbComponentCarrierManager", StringValue ("ns3::RrComponentCarrierManager"));

  int64_t stream = 1;

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (m_numberOfNodes);
  ueNodes.Create (m_numberOfNodes);

  uint32_t totalNumberOfNodes = enbNodes.GetN () + ueNodes.GetN ();

  // Install Mobility Model
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  for (uint16_t i = 0; i < totalNumberOfNodes; i++)
    {
      positionAlloc->Add (Vector (2 * i, 0, 0));
    }

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);

  for (uint32_t n = 0; n < m_numberOfNodes; ++n)
    {
      mobility.Install (enbNodes.Get (n));
      mobility.Install (ueNodes.Get (n));
    }

  ConfigToCheck configuration;
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;

  // Set bandwidth, EARFCN and install nodes (eNB and UE)
  for (uint32_t i = 0; i < m_configToCheck.size (); ++i)
    {
      configuration = m_configToCheck[i];

      lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (configuration.m_dlBandwidth));
      lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (configuration.m_ulBandwidth));
      lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (configuration.m_dlEarfcn));
      lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (configuration.m_ulEarfcn));
      lteHelper->SetUeDeviceAttribute ("DlEarfcn", UintegerValue (configuration.m_dlEarfcn));
      enbDevs.Add (lteHelper->InstallEnbDevice (enbNodes.Get (i)));
      lteHelper->AssignStreams (enbDevs, stream);
      ueDevs.Add (lteHelper->InstallUeDevice (ueNodes.Get (i)));
      lteHelper->AssignStreams (ueDevs, stream);
    }

  // Calculate the DlBandwidth, UlBandwidth, DlEarfcn and UlEarfcn to which the values from UE RRC would be compared
  m_configToCheckContainer = EquallySpacedCcs ();

  // Attach a UE to an eNB
  for(uint32_t k = 0; k < m_numberOfNodes; ++k)
    {
      lteHelper->Attach (ueDevs.Get (k), enbDevs.Get (k));
    }

   // Activate a data radio bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/SCarrierConfigured",
                   MakeCallback (&CarrierAggregationConfigTestCase::Evaluate, this));

  Simulator::Stop (m_simulationDuration);
  Simulator::Run ();

  NS_TEST_ASSERT_MSG_EQ (m_connectionCounter, ueNodes.GetN (), "Not all the UEs were connected");

  Simulator::Destroy ();
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Carrier aggregation configuration test suite.
 */
class CarrierAggregationConfigTestSuite : public TestSuite
{
public:
  CarrierAggregationConfigTestSuite ();
};

CarrierAggregationConfigTestSuite::CarrierAggregationConfigTestSuite ()
  : TestSuite ("lte-carrier-aggregation-configuration", SYSTEM)
{
  std::vector<ConfigToCheck> configToCheck;

  // Test1 with 1 eNB and 1 UE.
  // We put a configuration different than the default configuration done in LteHelper for the sake of
  // creating PHY and MAC instances equal to the number of component carriers.

  ConfigToCheck configToCheckTest1;
  configToCheckTest1.m_dlBandwidth = 50;
  configToCheckTest1.m_ulBandwidth = 50;
  configToCheckTest1.m_dlEarfcn = 300;
  configToCheckTest1.m_ulEarfcn = 300+18000;
  configToCheck.push_back (configToCheckTest1);
  uint32_t numberOfNodes = 1;
  uint16_t numberOfComponentCarriers = 2;
  Time simulationDuration = Seconds (1);

  AddTestCase (new CarrierAggregationConfigTestCase (numberOfNodes, numberOfComponentCarriers, configToCheck, simulationDuration), TestCase::QUICK);

//   configToCheck.erase(configToCheck.begin(), configToCheck.end());
  configToCheck.clear ();

  // Test2 with 2 eNBs and 2 UEs.
  // We decrease the bandwidth so not to exceed maximum band bandwidth of 20 MHz

  configToCheckTest1.m_dlBandwidth = 25;
  configToCheckTest1.m_ulBandwidth = 25;
  configToCheckTest1.m_dlEarfcn = 300;
  configToCheckTest1.m_ulEarfcn = 300+18000;
  configToCheck.push_back (configToCheckTest1);

  ConfigToCheck configToCheckTest2;
  configToCheckTest2.m_dlBandwidth = 25;
  configToCheckTest2.m_ulBandwidth = 25;
  configToCheckTest2.m_dlEarfcn = 502;
  configToCheckTest2.m_ulEarfcn = 502+18000;
  configToCheck.push_back (configToCheckTest2);
  numberOfNodes = 2;
  simulationDuration = Seconds (2);

  AddTestCase (new CarrierAggregationConfigTestCase (numberOfNodes, numberOfComponentCarriers, configToCheck, simulationDuration), TestCase::QUICK);
}

static CarrierAggregationConfigTestSuite g_carrierAggregationConfigTestSuite;
