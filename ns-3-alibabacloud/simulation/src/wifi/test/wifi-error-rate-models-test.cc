/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 University of Washington
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
 * Authors: Tom Henderson (tomhend@u.washington.edu)
 *          Sébastien Deronne (sebastien.deronne@gmail.com)
 */

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/nist-error-rate-model.h"
#include "ns3/yans-error-rate-model.h"
#include "ns3/dsss-error-rate-model.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-utils.h"
#include "ns3/table-based-error-rate-model.h"
#include "ns3/he-phy.h" //includes HT and VHT
#include "ns3/interference-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiErrorRateModelsTest");

static double
FromRss (double rssDbw)
{
  // SINR is based on receiver noise figure of 7 dB and thermal noise
  // of -100.5522786 dBm in this 22 MHz bandwidth at 290K
  double noisePowerDbw = -100.5522786 + 7;

  double sinrDb = rssDbw - noisePowerDbw;
  // return SINR expressed as ratio
  return pow (10.0, sinrDb / 10.0);
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Error Rate Models Test Case Dsss
 */
class WifiErrorRateModelsTestCaseDsss : public TestCase
{
public:
  WifiErrorRateModelsTestCaseDsss ();
  virtual ~WifiErrorRateModelsTestCaseDsss ();

private:
  void DoRun (void) override;
};

WifiErrorRateModelsTestCaseDsss::WifiErrorRateModelsTestCaseDsss ()
  : TestCase ("WifiErrorRateModel test case DSSS")
{
}

WifiErrorRateModelsTestCaseDsss::~WifiErrorRateModelsTestCaseDsss ()
{
}

void
WifiErrorRateModelsTestCaseDsss::DoRun (void)
{
  // 1024 bytes plus headers
  uint64_t size = (1024 + 40 + 14) * 8;
  // Spot test some values returned from DsssErrorRateModel
  // Values taken from sample 80211b.c program used in validation paper
  double value;
  // DBPSK
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-105.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0, 1e-13, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-100.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 1.5e-13, 1e-13, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-99.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.0003, 0.0001, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-98.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.202, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-97.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.813, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-96.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.984, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-95.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.999, 0.001, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDbpskSuccessRate (FromRss (-90.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 1, 0.001, "Not equal within tolerance");

  // DQPSK
  //
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-96.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0, 1e-13, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-95.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4.5e-6, 1e-6, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-94.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.036, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-93.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.519, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-92.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.915, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-91.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.993, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-90.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.999, 0.001, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskSuccessRate (FromRss (-89.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 1, 0.001, "Not equal within tolerance");

#ifdef HAVE_GSL
  // DQPSK_CCK5.5
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-94.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0, 1e-13, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-93.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 6.6e-14, 5e-14, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-92.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.0001, 0.00005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-91.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.132, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-90.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.744, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-89.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.974, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-88.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.999, 0.001, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (FromRss (-87.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 1, 0.001, "Not equal within tolerance");

  // DQPSK_CCK11
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-91.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0, 1e-14, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-90.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4.7e-14, 1e-14, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-89.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 8.85e-5, 1e-5, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-88.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.128, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-87.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.739, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-86.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.973, 0.005, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-85.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 0.999, 0.001, "Not equal within tolerance");
  value = DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (FromRss (-84.0), size);
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 1, 0.001, "Not equal within tolerance");
#endif
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Error Rate Models Test Case Nist
 */
class WifiErrorRateModelsTestCaseNist : public TestCase
{
public:
  WifiErrorRateModelsTestCaseNist ();
  virtual ~WifiErrorRateModelsTestCaseNist ();

private:
  void DoRun (void) override;
};

WifiErrorRateModelsTestCaseNist::WifiErrorRateModelsTestCaseNist ()
  : TestCase ("WifiErrorRateModel test case NIST")
{
}

WifiErrorRateModelsTestCaseNist::~WifiErrorRateModelsTestCaseNist ()
{
}

void
WifiErrorRateModelsTestCaseNist::DoRun (void)
{
  uint32_t frameSize = 2000;
  WifiTxVector txVector;
  Ptr<NistErrorRateModel> nist = CreateObject<NistErrorRateModel> ();

  double ps; // probability of success
  double snr; // dB

  // Spot test some values returned from NistErrorRateModel
  // values can be generated by the example program ofdm-validation.cc
  snr = 2.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate6Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 2.04e-10, 1e-10, "Not equal within tolerance");
  snr = 3.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate6Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.020, 0.001, "Not equal within tolerance");
  snr = 4.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate6Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.885, 0.001, "Not equal within tolerance");
  snr = 5.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate6Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.997, 0.001, "Not equal within tolerance");

  snr = 6.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate9Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.097, 0.001, "Not equal within tolerance");
  snr = 7.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate9Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.918, 0.001, "Not equal within tolerance");
  snr = 8.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate9Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.998, 0.001, "Not equal within tolerance");
  snr = 9.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate9Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");

  snr = 6.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate12Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.0174, 0.001, "Not equal within tolerance");
  snr = 7.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate12Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.881, 0.001, "Not equal within tolerance");
  snr = 8.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate12Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.997, 0.001, "Not equal within tolerance");
  snr = 9.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate12Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");

  snr = 8.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate18Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 2.85e-6, 1e-6, "Not equal within tolerance");
  snr = 9.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate18Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.623, 0.001, "Not equal within tolerance");
  snr = 10.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate18Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.985, 0.001, "Not equal within tolerance");
  snr = 11.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate18Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");

  snr = 12.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate24Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 2.22e-7, 1e-7, "Not equal within tolerance");
  snr = 13.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate24Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.495, 0.001, "Not equal within tolerance");
  snr = 14.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate24Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.974, 0.001, "Not equal within tolerance");
  snr = 15.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate24Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");

  snr = 15.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate36Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.012, 0.001, "Not equal within tolerance");
  snr = 16.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate36Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.818, 0.001, "Not equal within tolerance");
  snr = 17.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate36Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.993, 0.001, "Not equal within tolerance");
  snr = 18.5;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate36Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");

  snr = 20.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate48Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 1.3e-4, 1e-4, "Not equal within tolerance");
  snr = 21.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate48Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.649, 0.001, "Not equal within tolerance");
  snr = 22.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate48Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.983, 0.001, "Not equal within tolerance");
  snr = 23.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate48Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");

  snr = 21.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate54Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 5.44e-8, 1e-8, "Not equal within tolerance");
  snr = 22.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate54Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.410, 0.001, "Not equal within tolerance");
  snr = 23.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate54Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.958, 0.001, "Not equal within tolerance");
  snr = 24.0;
  ps = nist->GetChunkSuccessRate (WifiMode ("OfdmRate54Mbps"), txVector, std::pow (10.0, snr / 10.0), frameSize * 8);
  NS_TEST_ASSERT_MSG_EQ_TOL (ps, 0.999, 0.001, "Not equal within tolerance");
}

class TestInterferenceHelper : public InterferenceHelper
{
public:
  using InterferenceHelper::InterferenceHelper;
  using InterferenceHelper::CalculatePayloadChunkSuccessRate;
  using InterferenceHelper::CalculateSnr;
};

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Error Rate Models Test Case MIMO
 */
class WifiErrorRateModelsTestCaseMimo : public TestCase
{
public:
  WifiErrorRateModelsTestCaseMimo ();
  virtual ~WifiErrorRateModelsTestCaseMimo ();

private:
  void DoRun (void) override;
};

WifiErrorRateModelsTestCaseMimo::WifiErrorRateModelsTestCaseMimo ()
  : TestCase ("WifiErrorRateModel test case MIMO")
{
}

WifiErrorRateModelsTestCaseMimo::~WifiErrorRateModelsTestCaseMimo ()
{
}

void
WifiErrorRateModelsTestCaseMimo::DoRun (void)
{
  TestInterferenceHelper interference;
  interference.SetNoiseFigure (0);
  WifiMode mode = HtPhy::GetHtMcs0 ();
  WifiTxVector txVector;

  txVector.SetMode (mode);
  txVector.SetTxPowerLevel (0);
  txVector.SetChannelWidth (20);
  txVector.SetNss (1);
  txVector.SetNTx (1);

  interference.SetNumberOfReceiveAntennas (1);
  Ptr<NistErrorRateModel> nist = CreateObject<NistErrorRateModel> ();
  interference.SetErrorRateModel (nist);

  // SISO: initial SNR set to 4dB
  double initialSnr = 4.0;
  double snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr, 0.1, "Attempt to set initial SNR to known value failed");
  Time duration = MilliSeconds (2);
  double chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_EQ_TOL (chunkSuccess, 0.905685, 0.000001, "CSR not within tolerance for SISO");
  double sisoChunkSuccess = chunkSuccess;

  // MIMO 2x1:2: expect no SNR gain in AWGN channel
  txVector.SetNss (2);
  txVector.SetNTx (2);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr, 0.1, "SNR not within tolerance for 2x1:2 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_EQ_TOL (chunkSuccess, 0.905685, 0.000001, "CSR not within tolerance for SISO");

  // MIMO 1x2:1: expect that SNR is increased by a factor of 3 dB (10 log 2/1) compared to SISO thanks to RX diversity
  txVector.SetNss (1);
  txVector.SetNTx (1);
  interference.SetNumberOfReceiveAntennas (2);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 3, 0.1, "SNR not within tolerance for 1x2:1 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 1x2:1 MIMO");

  // MIMO 2x2:1: expect that SNR is increased by a factor of 3 dB (10 log 2/1) compared to SISO thanks to RX diversity
  txVector.SetNss (1);
  txVector.SetNTx (2);
  interference.SetNumberOfReceiveAntennas (2);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 3, 0.1, "SNR not equal within tolerance for 2x2:1 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 2x2:1 MIMO");

 // MIMO 2x2:2: expect no SNR gain in AWGN channel
  txVector.SetNss (2);
  txVector.SetNTx (2);
  interference.SetNumberOfReceiveAntennas (2);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr, 0.1, "SNR not equal within tolerance for 2x2:2 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_EQ_TOL (chunkSuccess, sisoChunkSuccess, 0.000001, "CSR not within tolerance for 2x2:2 MIMO");

  // MIMO 3x3:1: expect that SNR is increased by a factor of 4.8 dB (10 log 3/1) compared to SISO thanks to RX diversity
  txVector.SetNss (1);
  txVector.SetNTx (3);
  interference.SetNumberOfReceiveAntennas (3);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 4.8, 0.1, "SNR not within tolerance for 3x3:1 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 3x3:1 MIMO");

  // MIMO 3x3:2: expect that SNR is increased by a factor of 1.8 dB (10 log 3/2) compared to SISO thanks to RX diversity
  txVector.SetNss (2);
  txVector.SetNTx (3);
  interference.SetNumberOfReceiveAntennas (3);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 1.8, 0.1, "SNR not within tolerance for 3x3:2 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 3x3:2 MIMO");

  // MIMO 3x3:3: expect no SNR gain in AWGN channel
  txVector.SetNss (3);
  txVector.SetNTx (3);
  interference.SetNumberOfReceiveAntennas (3);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr, 0.1, "SNR not within tolerance for 3x3:3 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_EQ_TOL (chunkSuccess, sisoChunkSuccess, 0.000001, "CSR not equal within tolerance for 3x3:3 MIMO");

  // MIMO 4x4:1: expect that SNR is increased by a factor of 6 dB (10 log 4/1) compared to SISO thanks to RX diversity
  txVector.SetNss (1);
  txVector.SetNTx (4);
  interference.SetNumberOfReceiveAntennas (4);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 6, 0.1, "SNR not within tolerance for 4x4:1 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 4x4:1 MIMO");

  // MIMO 4x4:2: expect that SNR is increased by a factor of 3 dB (10 log 4/2) compared to SISO thanks to RX diversity
  txVector.SetNss (2);
  txVector.SetNTx (4);
  interference.SetNumberOfReceiveAntennas (4);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 3, 0.1, "SNR not within tolerance for 4x4:2 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 4x4:2 MIMO");

  // MIMO 4x4:3: expect that SNR is increased by a factor of 1.2 dB (10 log 4/3) compared to SISO thanks to RX diversity
  txVector.SetNss (3);
  txVector.SetNTx (4);
  interference.SetNumberOfReceiveAntennas (4);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr + 1.2, 0.1, "SNR not within tolerance for 4x4:3 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_GT (chunkSuccess, sisoChunkSuccess, "CSR not within tolerance for 4x4:1 MIMO");

  // MIMO 4x4:4: expect no SNR gain in AWGN channel
  txVector.SetNss (4);
  txVector.SetNTx (4);
  interference.SetNumberOfReceiveAntennas (4);
  snr = interference.CalculateSnr (0.001, 0.001 / DbToRatio (initialSnr), txVector.GetChannelWidth (), txVector.GetNss ());
  NS_TEST_ASSERT_MSG_EQ_TOL (RatioToDb (snr), initialSnr, 0.1, "SNR not within tolerance for 4x4:4 MIMO");
  chunkSuccess = interference.CalculatePayloadChunkSuccessRate (snr, duration, txVector);
  NS_TEST_ASSERT_MSG_EQ_TOL (chunkSuccess, sisoChunkSuccess, 0.000001, "CSR not within tolerance for 4x4:4 MIMO");
}

/**
 * map of PER values that have been manually computed for a given MCS, size (in bytes) and SNR (in dB) in order to verify against the PER calculated by the model
 */
std::map<std::pair <uint8_t /* mcs */, uint32_t /* size */>, std::map<double /* snr */, double /* per */> > expectedTableValues =
{
/* MCS 0 - 1458 bytes */
    {std::make_pair (0, 1458), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 1.00000},
        {-3.00, 1.00000},
        {-2.75, 1.00000},
        {-2.50, 1.00000},
        {-2.25, 1.00000},
        {-2.00, 1.00000},
        {-1.75, 1.00000},
        {-1.50, 1.00000},
        {-1.25, 1.00000},
        {-1.00, 1.00000},
        {-0.75, 0.99700},
        {-0.50, 0.99400},
        {-0.25, 0.90625},
        {0.00, 0.81850},
        {0.25, 0.55465},
        {0.50, 0.29080},
        {0.75, 0.17855},
        {1.00, 0.06630},
        {1.25, 0.03875},
        {1.50, 0.01120},
        {1.75, 0.00635},
        {2.00, 0.00150},
        {2.25, 0.00083},
        {2.50, 0.00015},
        {2.75, 0.00008},
        {3.00, 0.00001},
        {3.25, 0.00000},
        {3.50, 0.00000},
        {3.75, 0.00000},
        {4.00, 0.00000},
        {4.25, 0.00000},
        {4.50, 0.00000},
        {4.75, 0.00000},
        {5.00, 0.00000},
        {5.25, 0.00000},
        {5.50, 0.00000},
        {5.75, 0.00000},
        {6.00, 0.00000},
        {6.25, 0.00000},
        {6.50, 0.00000},
        {6.75, 0.00000},
        {7.00, 0.00000},
        {7.25, 0.00000},
        {7.50, 0.00000},
        {7.75, 0.00000},
        {8.00, 0.00000},
        {8.25, 0.00000},
        {8.50, 0.00000},
        {8.75, 0.00000},
        {9.00, 0.00000},
        {9.25, 0.00000},
        {9.50, 0.00000},
        {9.75, 0.00000},
        {10.00, 0.00000},
        {10.25, 0.00000},
        {10.50, 0.00000},
        {10.75, 0.00000},
        {11.00, 0.00000},
        {11.25, 0.00000},
        {11.50, 0.00000},
        {11.75, 0.00000},
        {12.00, 0.00000},
        {12.25, 0.00000},
        {12.50, 0.00000},
        {12.75, 0.00000},
        {13.00, 0.00000},
        {13.25, 0.00000},
        {13.50, 0.00000},
        {13.75, 0.00000},
        {14.00, 0.00000},
        {14.25, 0.00000},
        {14.50, 0.00000},
        {14.75, 0.00000},
        {15.00, 0.00000},
        {15.25, 0.00000},
        {15.50, 0.00000},
        {15.75, 0.00000},
        {16.00, 0.00000},
        {16.25, 0.00000},
        {16.50, 0.00000},
        {16.75, 0.00000},
        {17.00, 0.00000},
        {17.25, 0.00000},
        {17.50, 0.00000},
        {17.75, 0.00000},
        {18.00, 0.00000},
        {18.25, 0.00000},
        {18.50, 0.00000},
        {18.75, 0.00000},
        {19.00, 0.00000},
        {19.25, 0.00000},
        {19.50, 0.00000},
        {19.75, 0.00000},
        {20.00, 0.00000},
        {20.25, 0.00000},
        {20.50, 0.00000},
        {20.75, 0.00000},
        {21.00, 0.00000},
        {21.25, 0.00000},
        {21.50, 0.00000},
        {21.75, 0.00000},
        {22.00, 0.00000},
        {22.25, 0.00000},
        {22.50, 0.00000},
        {22.75, 0.00000},
        {23.00, 0.00000},
        {23.25, 0.00000},
        {23.50, 0.00000},
        {23.75, 0.00000},
        {24.00, 0.00000},
        {24.25, 0.00000},
        {24.50, 0.00000},
        {24.75, 0.00000},
        {25.00, 0.00000},
        {25.25, 0.00000},
        {25.50, 0.00000},
        {25.75, 0.00000},
        {26.00, 0.00000},
        {26.25, 0.00000},
        {26.50, 0.00000},
        {26.75, 0.00000},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
/* MCS 0 - 32 bytes */
    {std::make_pair (0, 32), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 0.99750},
        {-3.00, 0.99500},
        {-2.75, 0.96790},
        {-2.50, 0.94080},
        {-2.25, 0.88335},
        {-2.00, 0.82590},
        {-1.75, 0.70770},
        {-1.50, 0.58950},
        {-1.25, 0.44890},
        {-1.00, 0.30830},
        {-0.75, 0.21685},
        {-0.50, 0.12540},
        {-0.25, 0.07990},
        {0.00, 0.03440},
        {0.25, 0.02145},
        {0.50, 0.00850},
        {0.75, 0.00500},
        {1.00, 0.00150},
        {1.25, 0.00087},
        {1.50, 0.00024},
        {1.75, 0.00017},
        {2.00, 0.00009},
        {2.25, 0.00005},
        {2.50, 0.00000},
        {2.75, 0.00000},
        {3.00, 0.00000},
        {3.25, 0.00000},
        {3.50, 0.00000},
        {3.75, 0.00000},
        {4.00, 0.00000},
        {4.25, 0.00000},
        {4.50, 0.00000},
        {4.75, 0.00000},
        {5.00, 0.00000},
        {5.25, 0.00000},
        {5.50, 0.00000},
        {5.75, 0.00000},
        {6.00, 0.00000},
        {6.25, 0.00000},
        {6.50, 0.00000},
        {6.75, 0.00000},
        {7.00, 0.00000},
        {7.25, 0.00000},
        {7.50, 0.00000},
        {7.75, 0.00000},
        {8.00, 0.00000},
        {8.25, 0.00000},
        {8.50, 0.00000},
        {8.75, 0.00000},
        {9.00, 0.00000},
        {9.25, 0.00000},
        {9.50, 0.00000},
        {9.75, 0.00000},
        {10.00, 0.00000},
        {10.25, 0.00000},
        {10.50, 0.00000},
        {10.75, 0.00000},
        {11.00, 0.00000},
        {11.25, 0.00000},
        {11.50, 0.00000},
        {11.75, 0.00000},
        {12.00, 0.00000},
        {12.25, 0.00000},
        {12.50, 0.00000},
        {12.75, 0.00000},
        {13.00, 0.00000},
        {13.25, 0.00000},
        {13.50, 0.00000},
        {13.75, 0.00000},
        {14.00, 0.00000},
        {14.25, 0.00000},
        {14.50, 0.00000},
        {14.75, 0.00000},
        {15.00, 0.00000},
        {15.25, 0.00000},
        {15.50, 0.00000},
        {15.75, 0.00000},
        {16.00, 0.00000},
        {16.25, 0.00000},
        {16.50, 0.00000},
        {16.75, 0.00000},
        {17.00, 0.00000},
        {17.25, 0.00000},
        {17.50, 0.00000},
        {17.75, 0.00000},
        {18.00, 0.00000},
        {18.25, 0.00000},
        {18.50, 0.00000},
        {18.75, 0.00000},
        {19.00, 0.00000},
        {19.25, 0.00000},
        {19.50, 0.00000},
        {19.75, 0.00000},
        {20.00, 0.00000},
        {20.25, 0.00000},
        {20.50, 0.00000},
        {20.75, 0.00000},
        {21.00, 0.00000},
        {21.25, 0.00000},
        {21.50, 0.00000},
        {21.75, 0.00000},
        {22.00, 0.00000},
        {22.25, 0.00000},
        {22.50, 0.00000},
        {22.75, 0.00000},
        {23.00, 0.00000},
        {23.25, 0.00000},
        {23.50, 0.00000},
        {23.75, 0.00000},
        {24.00, 0.00000},
        {24.25, 0.00000},
        {24.50, 0.00000},
        {24.75, 0.00000},
        {25.00, 0.00000},
        {25.25, 0.00000},
        {25.50, 0.00000},
        {25.75, 0.00000},
        {26.00, 0.00000},
        {26.25, 0.00000},
        {26.50, 0.00000},
        {26.75, 0.00000},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
/* MCS 0 - 1000 bytes */
    {std::make_pair (0, 1000), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 1.00000},
        {-3.00, 1.00000},
        {-2.75, 1.00000},
        {-2.50, 1.00000},
        {-2.25, 1.00000},
        {-2.00, 1.00000},
        {-1.75, 1.00000},
        {-1.50, 1.00000},
        {-1.25, 1.00000},
        {-1.00, 1.00000},
        {-0.75, 0.98140},
        {-0.50, 0.97007},
        {-0.25, 0.80280},
        {0.00, 0.68977},
        {0.25, 0.42581},
        {0.50, 0.20997},
        {0.75, 0.12620},
        {1.00, 0.04596},
        {1.25, 0.02674},
        {1.50, 0.00770},
        {1.75, 0.00436},
        {2.00, 0.00103},
        {2.25, 0.00057},
        {2.50, 0.00010},
        {2.75, 0.00005},
        {3.00, 0.00001},
        {3.25, 0.00000},
        {3.50, 0.00000},
        {3.75, 0.00000},
        {4.00, 0.00000},
        {4.25, 0.00000},
        {4.50, 0.00000},
        {4.75, 0.00000},
        {5.00, 0.00000},
        {5.25, 0.00000},
        {5.50, 0.00000},
        {5.75, 0.00000},
        {6.00, 0.00000},
        {6.25, 0.00000},
        {6.50, 0.00000},
        {6.75, 0.00000},
        {7.00, 0.00000},
        {7.25, 0.00000},
        {7.50, 0.00000},
        {7.75, 0.00000},
        {8.00, 0.00000},
        {8.25, 0.00000},
        {8.50, 0.00000},
        {8.75, 0.00000},
        {9.00, 0.00000},
        {9.25, 0.00000},
        {9.50, 0.00000},
        {9.75, 0.00000},
        {10.00, 0.00000},
        {10.25, 0.00000},
        {10.50, 0.00000},
        {10.75, 0.00000},
        {11.00, 0.00000},
        {11.25, 0.00000},
        {11.50, 0.00000},
        {11.75, 0.00000},
        {12.00, 0.00000},
        {12.25, 0.00000},
        {12.50, 0.00000},
        {12.75, 0.00000},
        {13.00, 0.00000},
        {13.25, 0.00000},
        {13.50, 0.00000},
        {13.75, 0.00000},
        {14.00, 0.00000},
        {14.25, 0.00000},
        {14.50, 0.00000},
        {14.75, 0.00000},
        {15.00, 0.00000},
        {15.25, 0.00000},
        {15.50, 0.00000},
        {15.75, 0.00000},
        {16.00, 0.00000},
        {16.25, 0.00000},
        {16.50, 0.00000},
        {16.75, 0.00000},
        {17.00, 0.00000},
        {17.25, 0.00000},
        {17.50, 0.00000},
        {17.75, 0.00000},
        {18.00, 0.00000},
        {18.25, 0.00000},
        {18.50, 0.00000},
        {18.75, 0.00000},
        {19.00, 0.00000},
        {19.25, 0.00000},
        {19.50, 0.00000},
        {19.75, 0.00000},
        {20.00, 0.00000},
        {20.25, 0.00000},
        {20.50, 0.00000},
        {20.75, 0.00000},
        {21.00, 0.00000},
        {21.25, 0.00000},
        {21.50, 0.00000},
        {21.75, 0.00000},
        {22.00, 0.00000},
        {22.25, 0.00000},
        {22.50, 0.00000},
        {22.75, 0.00000},
        {23.00, 0.00000},
        {23.25, 0.00000},
        {23.50, 0.00000},
        {23.75, 0.00000},
        {24.00, 0.00000},
        {24.25, 0.00000},
        {24.50, 0.00000},
        {24.75, 0.00000},
        {25.00, 0.00000},
        {25.25, 0.00000},
        {25.50, 0.00000},
        {25.75, 0.00000},
        {26.00, 0.00000},
        {26.25, 0.00000},
        {26.50, 0.00000},
        {26.75, 0.00000},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
/* MCS 0 - 1 byte */
    {std::make_pair (0, 1), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 0.17075},
        {-3.00, 0.15260},
        {-2.75, 0.10190},
        {-2.50, 0.08455},
        {-2.25, 0.06494},
        {-2.00, 0.05316},
        {-1.75, 0.03771},
        {-1.50, 0.02744},
        {-1.25, 0.01845},
        {-1.00, 0.01145},
        {-0.75, 0.00761},
        {-0.50, 0.00418},
        {-0.25, 0.00260},
        {0.00, 0.00110},
        {0.25, 0.00068},
        {0.50, 0.00027},
        {0.75, 0.00016},
        {1.00, 0.00005},
        {1.25, 0.00003},
        {1.50, 0.00000},
        {1.75, 0.00000},
        {2.00, 0.00000},
        {2.25, 0.00000},
        {2.50, 0.00000},
        {2.75, 0.00000},
        {3.00, 0.00000},
        {3.25, 0.00000},
        {3.50, 0.00000},
        {3.75, 0.00000},
        {4.00, 0.00000},
        {4.25, 0.00000},
        {4.50, 0.00000},
        {4.75, 0.00000},
        {5.00, 0.00000},
        {5.25, 0.00000},
        {5.50, 0.00000},
        {5.75, 0.00000},
        {6.00, 0.00000},
        {6.25, 0.00000},
        {6.50, 0.00000},
        {6.75, 0.00000},
        {7.00, 0.00000},
        {7.25, 0.00000},
        {7.50, 0.00000},
        {7.75, 0.00000},
        {8.00, 0.00000},
        {8.25, 0.00000},
        {8.50, 0.00000},
        {8.75, 0.00000},
        {9.00, 0.00000},
        {9.25, 0.00000},
        {9.50, 0.00000},
        {9.75, 0.00000},
        {10.00, 0.00000},
        {10.25, 0.00000},
        {10.50, 0.00000},
        {10.75, 0.00000},
        {11.00, 0.00000},
        {11.25, 0.00000},
        {11.50, 0.00000},
        {11.75, 0.00000},
        {12.00, 0.00000},
        {12.25, 0.00000},
        {12.50, 0.00000},
        {12.75, 0.00000},
        {13.00, 0.00000},
        {13.25, 0.00000},
        {13.50, 0.00000},
        {13.75, 0.00000},
        {14.00, 0.00000},
        {14.25, 0.00000},
        {14.50, 0.00000},
        {14.75, 0.00000},
        {15.00, 0.00000},
        {15.25, 0.00000},
        {15.50, 0.00000},
        {15.75, 0.00000},
        {16.00, 0.00000},
        {16.25, 0.00000},
        {16.50, 0.00000},
        {16.75, 0.00000},
        {17.00, 0.00000},
        {17.25, 0.00000},
        {17.50, 0.00000},
        {17.75, 0.00000},
        {18.00, 0.00000},
        {18.25, 0.00000},
        {18.50, 0.00000},
        {18.75, 0.00000},
        {19.00, 0.00000},
        {19.25, 0.00000},
        {19.50, 0.00000},
        {19.75, 0.00000},
        {20.00, 0.00000},
        {20.25, 0.00000},
        {20.50, 0.00000},
        {20.75, 0.00000},
        {21.00, 0.00000},
        {21.25, 0.00000},
        {21.50, 0.00000},
        {21.75, 0.00000},
        {22.00, 0.00000},
        {22.25, 0.00000},
        {22.50, 0.00000},
        {22.75, 0.00000},
        {23.00, 0.00000},
        {23.25, 0.00000},
        {23.50, 0.00000},
        {23.75, 0.00000},
        {24.00, 0.00000},
        {24.25, 0.00000},
        {24.50, 0.00000},
        {24.75, 0.00000},
        {25.00, 0.00000},
        {25.25, 0.00000},
        {25.50, 0.00000},
        {25.75, 0.00000},
        {26.00, 0.00000},
        {26.25, 0.00000},
        {26.50, 0.00000},
        {26.75, 0.00000},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
/* MCS 0 - 2000 bytes */
    {std::make_pair (0, 2000), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 1.00000},
        {-3.00, 1.00000},
        {-2.75, 1.00000},
        {-2.50, 1.00000},
        {-2.25, 1.00000},
        {-2.00, 1.00000},
        {-1.75, 1.00000},
        {-1.50, 1.00000},
        {-1.25, 1.00000},
        {-1.00, 1.00000},
        {-0.75, 0.99965},
        {-0.50, 0.99910},
        {-0.25, 0.96111},
        {0.00, 0.90376},
        {0.25, 0.67031},
        {0.50, 0.37584},
        {0.75, 0.23647},
        {1.00, 0.08981},
        {1.25, 0.05277},
        {1.50, 0.01533},
        {1.75, 0.00870},
        {2.00, 0.00206},
        {2.25, 0.00113},
        {2.50, 0.00021},
        {2.75, 0.00011},
        {3.00, 0.00001},
        {3.25, 0.00000},
        {3.50, 0.00000},
        {3.75, 0.00000},
        {4.00, 0.00000},
        {4.25, 0.00000},
        {4.50, 0.00000},
        {4.75, 0.00000},
        {5.00, 0.00000},
        {5.25, 0.00000},
        {5.50, 0.00000},
        {5.75, 0.00000},
        {6.00, 0.00000},
        {6.25, 0.00000},
        {6.50, 0.00000},
        {6.75, 0.00000},
        {7.00, 0.00000},
        {7.25, 0.00000},
        {7.50, 0.00000},
        {7.75, 0.00000},
        {8.00, 0.00000},
        {8.25, 0.00000},
        {8.50, 0.00000},
        {8.75, 0.00000},
        {9.00, 0.00000},
        {9.25, 0.00000},
        {9.50, 0.00000},
        {9.75, 0.00000},
        {10.00, 0.00000},
        {10.25, 0.00000},
        {10.50, 0.00000},
        {10.75, 0.00000},
        {11.00, 0.00000},
        {11.25, 0.00000},
        {11.50, 0.00000},
        {11.75, 0.00000},
        {12.00, 0.00000},
        {12.25, 0.00000},
        {12.50, 0.00000},
        {12.75, 0.00000},
        {13.00, 0.00000},
        {13.25, 0.00000},
        {13.50, 0.00000},
        {13.75, 0.00000},
        {14.00, 0.00000},
        {14.25, 0.00000},
        {14.50, 0.00000},
        {14.75, 0.00000},
        {15.00, 0.00000},
        {15.25, 0.00000},
        {15.50, 0.00000},
        {15.75, 0.00000},
        {16.00, 0.00000},
        {16.25, 0.00000},
        {16.50, 0.00000},
        {16.75, 0.00000},
        {17.00, 0.00000},
        {17.25, 0.00000},
        {17.50, 0.00000},
        {17.75, 0.00000},
        {18.00, 0.00000},
        {18.25, 0.00000},
        {18.50, 0.00000},
        {18.75, 0.00000},
        {19.00, 0.00000},
        {19.25, 0.00000},
        {19.50, 0.00000},
        {19.75, 0.00000},
        {20.00, 0.00000},
        {20.25, 0.00000},
        {20.50, 0.00000},
        {20.75, 0.00000},
        {21.00, 0.00000},
        {21.25, 0.00000},
        {21.50, 0.00000},
        {21.75, 0.00000},
        {22.00, 0.00000},
        {22.25, 0.00000},
        {22.50, 0.00000},
        {22.75, 0.00000},
        {23.00, 0.00000},
        {23.25, 0.00000},
        {23.50, 0.00000},
        {23.75, 0.00000},
        {24.00, 0.00000},
        {24.25, 0.00000},
        {24.50, 0.00000},
        {24.75, 0.00000},
        {25.00, 0.00000},
        {25.25, 0.00000},
        {25.50, 0.00000},
        {25.75, 0.00000},
        {26.00, 0.00000},
        {26.25, 0.00000},
        {26.50, 0.00000},
        {26.75, 0.00000},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
/* MCS 7 - 1500 bytes */
    {std::make_pair (7, 1500), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 1.00000},
        {-3.00, 1.00000},
        {-2.75, 1.00000},
        {-2.50, 1.00000},
        {-2.25, 1.00000},
        {-2.00, 1.00000},
        {-1.75, 1.00000},
        {-1.50, 1.00000},
        {-1.25, 1.00000},
        {-1.00, 1.00000},
        {-0.75, 1.00000},
        {-0.50, 1.00000},
        {-0.25, 1.00000},
        {0.00, 1.00000},
        {0.25, 1.00000},
        {0.50, 1.00000},
        {0.75, 1.00000},
        {1.00, 1.00000},
        {1.25, 1.00000},
        {1.50, 1.00000},
        {1.75, 1.00000},
        {2.00, 1.00000},
        {2.25, 1.00000},
        {2.50, 1.00000},
        {2.75, 1.00000},
        {3.00, 1.00000},
        {3.25, 1.00000},
        {3.50, 1.00000},
        {3.75, 1.00000},
        {4.00, 1.00000},
        {4.25, 1.00000},
        {4.50, 1.00000},
        {4.75, 1.00000},
        {5.00, 1.00000},
        {5.25, 1.00000},
        {5.50, 1.00000},
        {5.75, 1.00000},
        {6.00, 1.00000},
        {6.25, 1.00000},
        {6.50, 1.00000},
        {6.75, 1.00000},
        {7.00, 1.00000},
        {7.25, 1.00000},
        {7.50, 1.00000},
        {7.75, 1.00000},
        {8.00, 1.00000},
        {8.25, 1.00000},
        {8.50, 1.00000},
        {8.75, 1.00000},
        {9.00, 1.00000},
        {9.25, 1.00000},
        {9.50, 1.00000},
        {9.75, 1.00000},
        {10.00, 1.00000},
        {10.25, 1.00000},
        {10.50, 1.00000},
        {10.75, 1.00000},
        {11.00, 1.00000},
        {11.25, 1.00000},
        {11.50, 1.00000},
        {11.75, 1.00000},
        {12.00, 1.00000},
        {12.25, 1.00000},
        {12.50, 1.00000},
        {12.75, 1.00000},
        {13.00, 1.00000},
        {13.25, 1.00000},
        {13.50, 1.00000},
        {13.75, 1.00000},
        {14.00, 1.00000},
        {14.25, 1.00000},
        {14.50, 1.00000},
        {14.75, 1.00000},
        {15.00, 1.00000},
        {15.25, 1.00000},
        {15.50, 1.00000},
        {15.75, 1.00000},
        {16.00, 1.00000},
        {16.25, 1.00000},
        {16.50, 1.00000},
        {16.75, 1.00000},
        {17.00, 1.00000},
        {17.25, 1.00000},
        {17.50, 1.00000},
        {17.75, 0.99057},
        {18.00, 0.98075},
        {18.25, 0.86664},
        {18.50, 0.74920},
        {18.75, 0.54857},
        {19.00, 0.34531},
        {19.25, 0.23624},
        {19.50, 0.12672},
        {19.75, 0.08164},
        {20.00, 0.03650},
        {20.25, 0.02340},
        {20.50, 0.01029},
        {20.75, 0.00653},
        {21.00, 0.00278},
        {21.25, 0.00165},
        {21.50, 0.00051},
        {21.75, 0.00030},
        {22.00, 0.00009},
        {22.25, 0.00005},
        {22.50, 0.00001},
        {22.75, 0.00000},
        {23.00, 0.00000},
        {23.25, 0.00000},
        {23.50, 0.00000},
        {23.75, 0.00000},
        {24.00, 0.00000},
        {24.25, 0.00000},
        {24.50, 0.00000},
        {24.75, 0.00000},
        {25.00, 0.00000},
        {25.25, 0.00000},
        {25.50, 0.00000},
        {25.75, 0.00000},
        {26.00, 0.00000},
        {26.25, 0.00000},
        {26.50, 0.00000},
        {26.75, 0.00000},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
/* MCS 8 - 1500 bytes */
    {std::make_pair (8, 1500), {
        {-4.00, 1.00000},
        {-3.75, 1.00000},
        {-3.50, 1.00000},
        {-3.25, 1.00000},
        {-3.00, 1.00000},
        {-2.75, 1.00000},
        {-2.50, 1.00000},
        {-2.25, 1.00000},
        {-2.00, 1.00000},
        {-1.75, 1.00000},
        {-1.50, 1.00000},
        {-1.25, 1.00000},
        {-1.00, 1.00000},
        {-0.75, 1.00000},
        {-0.50, 1.00000},
        {-0.25, 1.00000},
        {0.00, 1.00000},
        {0.25, 1.00000},
        {0.50, 1.00000},
        {0.75, 1.00000},
        {1.00, 1.00000},
        {1.25, 1.00000},
        {1.50, 1.00000},
        {1.75, 1.00000},
        {2.00, 1.00000},
        {2.25, 1.00000},
        {2.50, 1.00000},
        {2.75, 1.00000},
        {3.00, 1.00000},
        {3.25, 1.00000},
        {3.50, 1.00000},
        {3.75, 1.00000},
        {4.00, 1.00000},
        {4.25, 1.00000},
        {4.50, 1.00000},
        {4.75, 1.00000},
        {5.00, 1.00000},
        {5.25, 1.00000},
        {5.50, 1.00000},
        {5.75, 1.00000},
        {6.00, 1.00000},
        {6.25, 1.00000},
        {6.50, 1.00000},
        {6.75, 1.00000},
        {7.00, 1.00000},
        {7.25, 1.00000},
        {7.50, 1.00000},
        {7.75, 1.00000},
        {8.00, 1.00000},
        {8.25, 1.00000},
        {8.50, 1.00000},
        {8.75, 1.00000},
        {9.00, 1.00000},
        {9.25, 1.00000},
        {9.50, 1.00000},
        {9.75, 1.00000},
        {10.00, 1.00000},
        {10.25, 1.00000},
        {10.50, 1.00000},
        {10.75, 1.00000},
        {11.00, 1.00000},
        {11.25, 1.00000},
        {11.50, 1.00000},
        {11.75, 1.00000},
        {12.00, 1.00000},
        {12.25, 1.00000},
        {12.50, 1.00000},
        {12.75, 1.00000},
        {13.00, 1.00000},
        {13.25, 1.00000},
        {13.50, 1.00000},
        {13.75, 1.00000},
        {14.00, 1.00000},
        {14.25, 1.00000},
        {14.50, 1.00000},
        {14.75, 1.00000},
        {15.00, 1.00000},
        {15.25, 1.00000},
        {15.50, 1.00000},
        {15.75, 1.00000},
        {16.00, 1.00000},
        {16.25, 1.00000},
        {16.50, 1.00000},
        {16.75, 1.00000},
        {17.00, 1.00000},
        {17.25, 1.00000},
        {17.50, 1.00000},
        {17.75, 1.00000},
        {18.00, 1.00000},
        {18.25, 1.00000},
        {18.50, 1.00000},
        {18.75, 1.00000},
        {19.00, 1.00000},
        {19.25, 1.00000},
        {19.50, 1.00000},
        {19.75, 1.00000},
        {20.00, 1.00000},
        {20.25, 1.00000},
        {20.50, 1.00000},
        {20.75, 1.00000},
        {21.00, 1.00000},
        {21.25, 0.99918},
        {21.50, 0.99833},
        {21.75, 0.97191},
        {22.00, 0.94458},
        {22.25, 0.81436},
        {22.50, 0.68127},
        {22.75, 0.52168},
        {23.00, 0.36056},
        {23.25, 0.25114},
        {23.50, 0.14127},
        {23.75, 0.09509},
        {24.00, 0.04883},
        {24.25, 0.03234},
        {24.50, 0.01584},
        {24.75, 0.01060},
        {25.00, 0.00535},
        {25.25, 0.00345},
        {25.50, 0.00154},
        {25.75, 0.00096},
        {26.00, 0.00037},
        {26.25, 0.00022},
        {26.50, 0.00007},
        {26.75, 0.00004},
        {27.00, 0.00000},
        {27.25, 0.00000},
        {27.50, 0.00000},
        {27.75, 0.00000},
        {28.00, 0.00000},
        {28.25, 0.00000},
        {28.50, 0.00000},
        {28.75, 0.00000},
        {29.00, 0.00000},
        {29.25, 0.00000},
        {29.50, 0.00000},
        {29.75, 0.00000},
        {30.00, 0.00000},
    }},
};

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Table-based Error Rate Models Test Case
 */
class TableBasedErrorRateTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param testName the test name
   * \param mode the WifiMode to use for the test
   * \param size the number of bytes to use for the test
   */
  TableBasedErrorRateTestCase (const std::string &testName, WifiMode mode, uint32_t size);
  virtual ~TableBasedErrorRateTestCase ();

private:
  void DoRun (void) override;

  std::string m_testName; ///< The name of the test to run
  WifiMode m_mode;        ///< The WifiMode to test
  uint32_t m_size;        ///< The size (in bytes) to test
};

TableBasedErrorRateTestCase::TableBasedErrorRateTestCase (const std::string &testName, WifiMode mode, uint32_t size)
  : TestCase (testName),
    m_testName (testName),
    m_mode (mode),
    m_size (size)
{
}

TableBasedErrorRateTestCase::~TableBasedErrorRateTestCase ()
{
}

void
TableBasedErrorRateTestCase::DoRun (void)
{
  //LogComponentEnable ("WifiErrorRateModelsTest", LOG_LEVEL_ALL);
  //LogComponentEnable ("TableBasedErrorRateModel", LOG_LEVEL_ALL);
  //LogComponentEnable ("YansErrorRateModel", LOG_LEVEL_ALL);

  Ptr<TableBasedErrorRateModel> table = CreateObject<TableBasedErrorRateModel> ();
  WifiTxVector txVector;
  txVector.SetMode (m_mode);

  // Spot test some values returned from TableBasedErrorRateModel
  for (double snr = -4; snr <= 30; snr += 0.25)
    {
      double expectedValue = 0;
      if (m_mode.GetMcsValue () > ERROR_TABLE_BCC_MAX_NUM_MCS)
        {
          Ptr<YansErrorRateModel> yans = CreateObject<YansErrorRateModel> ();
          expectedValue = 1 - yans->GetChunkSuccessRate (m_mode, txVector, std::pow (10, snr / 10), m_size * 8);
        }
      else
        {
          auto it = expectedTableValues.find (std::make_pair (m_mode.GetMcsValue (), m_size));
          if (it != expectedTableValues.end ())
            {
              auto itValue = it->second.find (snr);
              if (itValue != it->second.end ())
                {
                  expectedValue = itValue->second;
                }
              else
                {
                  NS_FATAL_ERROR ("SNR value " << snr << " dB not found!");
                }
            }
          else
            {
              NS_FATAL_ERROR ("No expected value found for the combination MCS " << +m_mode.GetMcsValue () << " and size " << m_size << " bytes");
            }
        }
      double per = 1 - table->GetChunkSuccessRate (m_mode, txVector, std::pow (10, snr / 10), m_size * 8);
      NS_LOG_INFO (m_testName << ": snr=" << snr << "dB per=" << per << " expectedPER=" << expectedValue);
      NS_TEST_ASSERT_MSG_EQ_TOL (per, expectedValue, 1e-5, "Not equal within tolerance");
    }
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi Error Rate Models Test Suite
 */
class WifiErrorRateModelsTestSuite : public TestSuite
{
public:
  WifiErrorRateModelsTestSuite ();
};

WifiErrorRateModelsTestSuite::WifiErrorRateModelsTestSuite ()
  : TestSuite ("wifi-error-rate-models", UNIT)
{
  AddTestCase (new WifiErrorRateModelsTestCaseDsss, TestCase::QUICK);
  AddTestCase (new WifiErrorRateModelsTestCaseNist, TestCase::QUICK);
  AddTestCase (new WifiErrorRateModelsTestCaseMimo, TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedHtMcs0-1458bytes", HtPhy::GetHtMcs0 (), 1458), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedHtMcs0-32bytes", HtPhy::GetHtMcs0 (), 32), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedHtMcs0-1000bytes", HtPhy::GetHtMcs0 (), 1000), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedHtMcs0-1byte", HtPhy::GetHtMcs0 (), 1), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedHtMcs0-2000bytes", HtPhy::GetHtMcs0 (), 2000), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedHtMcs7-1500bytes", HtPhy::GetHtMcs7 (), 1500), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedVhtMcs0-1458bytes", VhtPhy::GetVhtMcs0 (), 1458), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedVhtMcs0-32bytes", VhtPhy::GetVhtMcs0 (), 32), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedVhtMcs0-1000bytes", VhtPhy::GetVhtMcs0 (), 1000), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedVhtMcs0-1byte", VhtPhy::GetVhtMcs0 (), 1), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedVhtMcs0-2000bytes", VhtPhy::GetVhtMcs0 (), 2000), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("DefaultTableBasedVhtMcs8-1500bytes", VhtPhy::GetVhtMcs8 (), 1500), TestCase::QUICK);
  AddTestCase (new TableBasedErrorRateTestCase ("FallbackTableBasedHeMcs11-1458bytes", HePhy::GetHeMcs11 (), 1458), TestCase::QUICK);
}

static WifiErrorRateModelsTestSuite wifiErrorRateModelsTestSuite; ///< the test suite
