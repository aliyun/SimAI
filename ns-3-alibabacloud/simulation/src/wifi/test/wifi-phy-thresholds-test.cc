/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/wifi-spectrum-value-helper.h"
#include "ns3/spectrum-wifi-phy.h"
#include "ns3/interference-helper.h"
#include "ns3/nist-error-rate-model.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-spectrum-signal-parameters.h"
#include "ns3/wifi-utils.h"
#include "ns3/wifi-psdu.h"
#include "ns3/ofdm-ppdu.h"
#include "ns3/ofdm-phy.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiPhyThresholdsTest");

static const uint8_t CHANNEL_NUMBER = 36;
static const uint32_t FREQUENCY = 5180; //MHz
static const uint16_t CHANNEL_WIDTH = 20; //MHz

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Phy Threshold Test base class
 */
class WifiPhyThresholdsTest : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param test_name the test name
   */
  WifiPhyThresholdsTest (std::string test_name);
  /**
   * Destructor
   */
  virtual ~WifiPhyThresholdsTest ();

protected:
  /**
   * Make wifi signal function
   * \param txPowerWatts the transmit power in watts
   * \returns Ptr<SpectrumSignalParameters>
   */
  virtual Ptr<SpectrumSignalParameters> MakeWifiSignal (double txPowerWatts);
  /**
   * Make foreign signal function
   * \param txPowerWatts the transmit power in watts
   * \returns Ptr<SpectrumSignalParameters>
   */
  virtual Ptr<SpectrumSignalParameters> MakeForeignSignal (double txPowerWatts);
  /**
   * Send signal function
   * \param txPowerWatts the transmit power in watts
   * \param wifiSignal whether the signal is a wifi signal or not
   */
  virtual void SendSignal (double txPowerWatts, bool wifiSignal);
  /**
   * PHY receive success callback function
   * \param psdu the PSDU
   * \param rxSignalInfo the info on the received signal (\see RxSignalInfo)
   * \param txVector the transmit vector
   * \param statusPerMpdu reception status per MPDU
   */
  virtual void RxSuccess (Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo,
                          WifiTxVector txVector, std::vector<bool> statusPerMpdu);
  /**
   * PHY receive failure callback function
   * \param psdu the PSDU
   */
  virtual void RxFailure (Ptr<WifiPsdu> psdu);
  /**
   * PHY dropped packet callback function
   * \param p the packet
   * \param reason the reason
   */
  void RxDropped (Ptr<const Packet> p, WifiPhyRxfailureReason reason);
  /**
   * PHY state changed callback function
   * \param start the start time of the new state
   * \param duration the duration of the new state
   * \param newState the new state
   */
  virtual void PhyStateChanged (Time start, Time duration, WifiPhyState newState);

  Ptr<SpectrumWifiPhy> m_phy; ///< PHY object
  uint32_t m_rxSuccess;       ///< count number of successfully received packets
  uint32_t m_rxFailure;       ///< count number of unsuccessfully received packets
  uint32_t m_rxDropped;       ///< count number of dropped packets
  uint32_t m_stateChanged;    ///< count number of PHY state change
  uint32_t m_rxStateCount;    ///< count number of PHY state change to RX state
  uint32_t m_idleStateCount;    ///< count number of PHY state change to IDLE state
  uint32_t m_ccabusyStateCount; ///< count number of PHY state change to CCA_BUSY state

private:
  void DoSetup (void) override;
  void DoTeardown (void) override;
};

WifiPhyThresholdsTest::WifiPhyThresholdsTest (std::string test_name)
  : TestCase (test_name),
    m_rxSuccess (0),
    m_rxFailure (0),
    m_rxDropped (0),
    m_stateChanged (0),
    m_rxStateCount (0),
    m_idleStateCount (0),
    m_ccabusyStateCount (0)
{
}

WifiPhyThresholdsTest::~WifiPhyThresholdsTest ()
{
}

Ptr<SpectrumSignalParameters>
WifiPhyThresholdsTest::MakeWifiSignal (double txPowerWatts)
{
  WifiTxVector txVector = WifiTxVector (OfdmPhy::GetOfdmRate6Mbps (), 0, WIFI_PREAMBLE_LONG, 800, 1, 1, 0, 20, false);

  Ptr<Packet> pkt = Create<Packet> (1000);
  WifiMacHeader hdr;

  hdr.SetType (WIFI_MAC_QOSDATA);
  hdr.SetQosTid (0);

  Ptr<WifiPsdu> psdu = Create<WifiPsdu> (pkt, hdr);
  Time txDuration = m_phy->CalculateTxDuration (psdu->GetSize (), txVector, m_phy->GetPhyBand ());

  Ptr<WifiPpdu> ppdu = Create<OfdmPpdu> (psdu, txVector, WIFI_PHY_BAND_5GHZ, 0);

  Ptr<SpectrumValue> txPowerSpectrum = WifiSpectrumValueHelper::CreateHeOfdmTxPowerSpectralDensity (FREQUENCY, CHANNEL_WIDTH, txPowerWatts, CHANNEL_WIDTH);
  Ptr<WifiSpectrumSignalParameters> txParams = Create<WifiSpectrumSignalParameters> ();
  txParams->psd = txPowerSpectrum;
  txParams->txPhy = 0;
  txParams->duration = txDuration;
  txParams->ppdu = ppdu;
  return txParams;
}

Ptr<SpectrumSignalParameters>
WifiPhyThresholdsTest::MakeForeignSignal (double txPowerWatts)
{
  Ptr<SpectrumValue> txPowerSpectrum = WifiSpectrumValueHelper::CreateHeOfdmTxPowerSpectralDensity (FREQUENCY, CHANNEL_WIDTH, txPowerWatts, CHANNEL_WIDTH);
  Ptr<SpectrumSignalParameters> txParams = Create<SpectrumSignalParameters> ();
  txParams->psd = txPowerSpectrum;
  txParams->txPhy = 0;
  txParams->duration = Seconds (0.5);
  return txParams;
}

void
WifiPhyThresholdsTest::SendSignal (double txPowerWatts, bool wifiSignal)
{
  if (wifiSignal)
    {
      m_phy->StartRx (MakeWifiSignal (txPowerWatts));
    }
  else
    {
      m_phy->StartRx (MakeForeignSignal (txPowerWatts));
    }
}

void
WifiPhyThresholdsTest::RxSuccess (Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo,
                                  WifiTxVector txVector, std::vector<bool> statusPerMpdu)
{
  NS_LOG_FUNCTION (this << *psdu << rxSignalInfo << txVector);
  m_rxSuccess++;
}

void
WifiPhyThresholdsTest::RxFailure (Ptr<WifiPsdu> psdu)
{
  NS_LOG_FUNCTION (this << *psdu);
  m_rxFailure++;
}

void
WifiPhyThresholdsTest::RxDropped (Ptr<const Packet> p, WifiPhyRxfailureReason reason)
{
  NS_LOG_FUNCTION (this << p << reason);
  m_rxDropped++;
}

void
WifiPhyThresholdsTest::PhyStateChanged (Time start, Time duration, WifiPhyState newState)
{
  NS_LOG_FUNCTION (this << start << duration << newState);
  m_stateChanged++;
  if (newState == WifiPhyState::IDLE)
    {
      m_idleStateCount++;
    }
  else if (newState == WifiPhyState::RX)
    {
      m_rxStateCount++;
    }
  else if (newState == WifiPhyState::CCA_BUSY)
    {
      m_ccabusyStateCount++;
    }
}

void
WifiPhyThresholdsTest::DoSetup (void)
{
  m_phy = CreateObject<SpectrumWifiPhy> ();
  m_phy->ConfigureStandard (WIFI_STANDARD_80211ax);
  Ptr<InterferenceHelper> interferenceHelper = CreateObject<InterferenceHelper> ();
  m_phy->SetInterferenceHelper (interferenceHelper);
  Ptr<ErrorRateModel> error = CreateObject<NistErrorRateModel> ();
  m_phy->SetErrorRateModel (error);
  m_phy->SetOperatingChannel (WifiPhy::ChannelTuple {CHANNEL_NUMBER, 0, WIFI_PHY_BAND_5GHZ, 0});
  m_phy->SetReceiveOkCallback (MakeCallback (&WifiPhyThresholdsTest::RxSuccess, this));
  m_phy->SetReceiveErrorCallback (MakeCallback (&WifiPhyThresholdsTest::RxFailure, this));
  m_phy->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&WifiPhyThresholdsTest::RxDropped, this));
  m_phy->GetState ()->TraceConnectWithoutContext ("State", MakeCallback (&WifiPhyThresholdsTest::PhyStateChanged, this));
}

void
WifiPhyThresholdsTest::DoTeardown (void)
{
  m_phy->Dispose ();
  m_phy = 0;
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Phy Threshold Weak Wifi Signal Test
 *
 * This test makes sure PHY ignores a Wi-Fi signal
 * if its received power lower than RxSensitivity.
 */
class WifiPhyThresholdsWeakWifiSignalTest : public WifiPhyThresholdsTest
{
public:
  WifiPhyThresholdsWeakWifiSignalTest ();
  virtual ~WifiPhyThresholdsWeakWifiSignalTest ();
  virtual void DoRun (void);
};

WifiPhyThresholdsWeakWifiSignalTest::WifiPhyThresholdsWeakWifiSignalTest ()
  : WifiPhyThresholdsTest ("WifiPhy reception thresholds: test weak wifi signal reception")
{
}

WifiPhyThresholdsWeakWifiSignalTest::~WifiPhyThresholdsWeakWifiSignalTest ()
{
}

void
WifiPhyThresholdsWeakWifiSignalTest::DoRun (void)
{
  double txPowerWatts = DbmToW (-110);

  Simulator::Schedule (Seconds (1), &WifiPhyThresholdsWeakWifiSignalTest::SendSignal, this, txPowerWatts, true);

  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_rxDropped + m_rxSuccess + m_rxFailure, 0, "Reception should not have been triggered if packet is weaker than RxSensitivity threshold");
  NS_TEST_ASSERT_MSG_EQ (m_stateChanged, 0, "State should stay idle if reception involves a signal weaker than RxSensitivity threshold");
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Phy Threshold Weak Foreign Signal Test
 *
 * This test makes sure PHY keeps the state as IDLE if reception involves
 * a foreign signal with a received power lower than CcaEdThreshold.
 */
class WifiPhyThresholdsWeakForeignSignalTest : public WifiPhyThresholdsTest
{
public:
  WifiPhyThresholdsWeakForeignSignalTest ();
  virtual ~WifiPhyThresholdsWeakForeignSignalTest ();
  virtual void DoRun (void);
};

WifiPhyThresholdsWeakForeignSignalTest::WifiPhyThresholdsWeakForeignSignalTest ()
  : WifiPhyThresholdsTest ("WifiPhy reception thresholds: test weak foreign signal reception")
{
}

WifiPhyThresholdsWeakForeignSignalTest::~WifiPhyThresholdsWeakForeignSignalTest ()
{
}

void
WifiPhyThresholdsWeakForeignSignalTest::DoRun (void)
{
  double txPowerWatts = DbmToW (-90);

  Simulator::Schedule (Seconds (1), &WifiPhyThresholdsWeakForeignSignalTest::SendSignal, this, txPowerWatts, false);

  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_rxDropped + m_rxSuccess + m_rxFailure, 0, "Reception of non-wifi packet should not be triggered");
  NS_TEST_ASSERT_MSG_EQ (m_stateChanged, 0, "State should stay idle if reception involves a signal weaker than RxSensitivity threshold");
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Phy Threshold Strong Wifi Signal Test
 *
 * This test makes sure PHY processes a Wi-Fi signal
 * with a received power higher than RxSensitivity.
 */
class WifiPhyThresholdsStrongWifiSignalTest : public WifiPhyThresholdsTest
{
public:
  WifiPhyThresholdsStrongWifiSignalTest ();
  virtual ~WifiPhyThresholdsStrongWifiSignalTest ();
  virtual void DoRun (void);
};

WifiPhyThresholdsStrongWifiSignalTest::WifiPhyThresholdsStrongWifiSignalTest ()
  : WifiPhyThresholdsTest ("WifiPhy reception thresholds: test strong wifi signal reception")
{
}

WifiPhyThresholdsStrongWifiSignalTest::~WifiPhyThresholdsStrongWifiSignalTest ()
{
}

void
WifiPhyThresholdsStrongWifiSignalTest::DoRun (void)
{
  double txPowerWatts = DbmToW (-60);

  Simulator::Schedule (Seconds (1), &WifiPhyThresholdsStrongWifiSignalTest::SendSignal, this, txPowerWatts, true);

  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_rxDropped + m_rxFailure, 0, "Packet reception should have been successfull");
  NS_TEST_ASSERT_MSG_EQ (m_rxSuccess, 1, "Packet should have been successfully received");
  NS_TEST_ASSERT_MSG_EQ (m_ccabusyStateCount, 2, "State should have moved to CCA_BUSY once");
  NS_TEST_ASSERT_MSG_EQ (m_stateChanged, 4, "State should have moved to CCA_BUSY, then to RX and finally back to IDLE");
  NS_TEST_ASSERT_MSG_EQ (m_rxStateCount, 1, "State should have moved to RX once");
  NS_TEST_ASSERT_MSG_EQ (m_idleStateCount, 1, "State should have moved to IDLE once");
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Phy Threshold Strong Foreign Signal Test
 *
 * This test makes sure PHY declare the state as CCA_BUSY if reception involves
 * a foreign signal with a received power higher than CcaEdThreshold.
 */
class WifiPhyThresholdsStrongForeignSignalTest : public WifiPhyThresholdsTest
{
public:
  WifiPhyThresholdsStrongForeignSignalTest ();
  virtual ~WifiPhyThresholdsStrongForeignSignalTest ();
  virtual void DoRun (void);
};

WifiPhyThresholdsStrongForeignSignalTest::WifiPhyThresholdsStrongForeignSignalTest ()
  : WifiPhyThresholdsTest ("WifiPhy reception thresholds: test weak foreign signal reception")
{
}

WifiPhyThresholdsStrongForeignSignalTest::~WifiPhyThresholdsStrongForeignSignalTest ()
{
}

void
WifiPhyThresholdsStrongForeignSignalTest::DoRun (void)
{
  double txPowerWatts = DbmToW (-60);

  Simulator::Schedule (Seconds (1), &WifiPhyThresholdsStrongForeignSignalTest::SendSignal, this, txPowerWatts, false);

  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_rxDropped + m_rxSuccess + m_rxFailure, 0, "Reception of non-wifi packet should not be triggered");
  NS_TEST_ASSERT_MSG_EQ (m_idleStateCount, 1, "State should have moved to CCA-BUSY then back to IDLE");
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Phy Thresholds Test Suite
 */
class WifiPhyThresholdsTestSuite : public TestSuite
{
public:
  WifiPhyThresholdsTestSuite ();
};

WifiPhyThresholdsTestSuite::WifiPhyThresholdsTestSuite ()
  : TestSuite ("wifi-phy-thresholds", UNIT)
{
  AddTestCase (new WifiPhyThresholdsWeakWifiSignalTest, TestCase::QUICK);
  AddTestCase (new WifiPhyThresholdsWeakForeignSignalTest, TestCase::QUICK);
  AddTestCase (new WifiPhyThresholdsStrongWifiSignalTest, TestCase::QUICK);
  AddTestCase (new WifiPhyThresholdsStrongForeignSignalTest, TestCase::QUICK);
}

static WifiPhyThresholdsTestSuite wifiPhyThresholdsTestSuite; ///< the test suite
