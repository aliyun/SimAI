/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Orange Labs
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
 * Author: Rediet <getachew.redieteab@orange.com>
 */

#include <cmath>
#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/fatal-error.h"
#include "ns3/wifi-spectrum-value-helper.h"
#include "ns3/wifi-standards.h"
#include "ns3/wifi-phy-band.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiTransmitMaskTest");

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test checks if Wifi spectrum values for OFDM are generated properly.
 * Different test cases are configured by defining different standards and bandwidth.
 */
class WifiOfdmMaskSlopesTestCase : public TestCase
{
public:
  /**
   * typedef for a pair of sub-band index and relative power value (dBr)
   */
  typedef std::pair<uint32_t, double> IndexPowerPair;

  /**
   * typedef for a vector of pairs of sub-band index and relative power value (dBr)
   */
  typedef std::vector<IndexPowerPair> IndexPowerVect;

  /**
   * Constructor
   *
   * \param str test reference name
   * \param standard selected standard
   * \param band selected PHY band
   * \param bw bandwidth
   * \param maskRefsLeft vector of expected power values and corresponding indexes of generated PSD
   *                     (only start and stop indexes/values given) for left guard bandwidth
   * \param maskRefsRight vector of expected power values and corresponding indexes of generated PSD
   *                      (only start and stop indexes/values given) for right guard bandwidth
   * \param tol tolerance (in dB)
   */
  WifiOfdmMaskSlopesTestCase (const char* str, WifiStandard standard, WifiPhyBand band, uint8_t bw,
                              IndexPowerVect maskRefsLeft, IndexPowerVect maskRefsRight, double tol);
  virtual ~WifiOfdmMaskSlopesTestCase ();

protected:
  Ptr<SpectrumValue> m_actualSpectrum; ///< actual spectrum value
  IndexPowerVect m_expectedLeftPsd;    ///< expected power values for left guard bandwidth
  IndexPowerVect m_expectedRightPsd;   ///< expected power values for right guard bandwidth
  double m_tolerance;                  ///< tolerance (in dB)

private:
  void DoRun (void) override;
  /**
   * Interpolate PSD values for indexes between provided start and stop and append to provided
   * vector.
   *
   * \param vect vector of sub-band index and relative power value pairs to which interpolated values
                 should be appended
   * \param start pair of sub-band index and relative power value (dBr) for interval start
   * \param stop pair of sub-band index and relative power value (dBr) for interval stop
   * \param tol tolerance (in dB)
  */
  static void InterpolateAndAppendValues (IndexPowerVect &vect, IndexPowerPair start, IndexPowerPair stop,
                                          double tol);
};

WifiOfdmMaskSlopesTestCase::WifiOfdmMaskSlopesTestCase (const char* str, WifiStandard standard, WifiPhyBand band, uint8_t bw,
                                                        IndexPowerVect maskRefsLeft, IndexPowerVect maskRefsRight, double tol)
  :   TestCase (std::string ("SpectrumValue ") + str)
{
  NS_LOG_FUNCTION (this << str << standard << band << +bw << tol);
  NS_ASSERT (maskRefsLeft.size () % 2 == 0 && maskRefsRight.size () % 2 == 0); //start/stop pairs expected
  uint16_t freq = 5170 + (bw / 2); // so as to have 5180/5190/5210/5250 for 20/40/80/160
  double refTxPowerW = 1; // have to work in dBr when comparing though
  m_tolerance = tol; // in dB
  double outerBandMaximumRejection = -40; // in dBr

  switch (standard)
    {
    case WIFI_STANDARD_80211p:
      NS_ASSERT ((bw == 5) || (bw == 10));
      freq = 5860;
      m_actualSpectrum = WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity (freq, bw, refTxPowerW, bw, -20.0, -28.0, outerBandMaximumRejection);
      break;

    // 11g and 11a
    case WIFI_STANDARD_80211g:
      freq = 2412;
    // no break on purpose
    case WIFI_STANDARD_80211a:
      NS_ASSERT (bw == 20);
      m_actualSpectrum = WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity (freq, bw, refTxPowerW, bw, -20.0, -28.0, outerBandMaximumRejection);
      break;

    // 11n
    case WIFI_STANDARD_80211n:
      if (band == WIFI_PHY_BAND_2_4GHZ)
        {
          freq = 2402 + (bw / 2); //so as to have 2412/2422 for 20/40
          outerBandMaximumRejection = -45;
        }
      NS_ASSERT (bw == 20 || bw == 40);
      m_actualSpectrum = WifiSpectrumValueHelper::CreateHtOfdmTxPowerSpectralDensity (freq, bw, refTxPowerW, bw, -20.0, -28.0, outerBandMaximumRejection);
      break;

    // 11ac
    case WIFI_STANDARD_80211ac:
      NS_ASSERT (bw == 20 || bw == 40 || bw == 80 || bw == 160);
      m_actualSpectrum = WifiSpectrumValueHelper::CreateHtOfdmTxPowerSpectralDensity (freq, bw, refTxPowerW, bw, -20.0, -28.0, outerBandMaximumRejection);
      break;

    // 11ax
    case WIFI_STANDARD_80211ax:
      if (band == WIFI_PHY_BAND_2_4GHZ)
        {
          NS_ASSERT (bw != 160); // not enough space in 2.4 GHz bands
          freq = 2402 + (bw / 2); //so as to have 2412/2422 for 20/40
          outerBandMaximumRejection = -45;
        }
      NS_ASSERT (bw == 20 || bw == 40 || bw == 80 || bw == 160);
      m_actualSpectrum = WifiSpectrumValueHelper::CreateHeOfdmTxPowerSpectralDensity (freq, bw, refTxPowerW, bw, -20.0, -28.0, outerBandMaximumRejection);
      break;

    // other
    default:
      NS_FATAL_ERROR ("Standard unknown or non-OFDM");
      break;
    }

  NS_LOG_INFO ("Build expected left PSD");
  for (uint32_t i = 0; i < maskRefsLeft.size (); i = i + 2)
    {
      InterpolateAndAppendValues (m_expectedLeftPsd, maskRefsLeft[i], maskRefsLeft[i + 1], tol);
    }
  NS_ASSERT (m_expectedLeftPsd.size () == (m_expectedLeftPsd.back ().first - m_expectedLeftPsd.front ().first + 1));

  NS_LOG_INFO ("Build expected right PSD");
  for (uint32_t i = 0; i < maskRefsRight.size (); i = i + 2)
    {
      InterpolateAndAppendValues (m_expectedRightPsd, maskRefsRight[i], maskRefsRight[i + 1], tol);
    }
  NS_ASSERT (m_expectedRightPsd.size () == (m_expectedRightPsd.back ().first - m_expectedRightPsd.front ().first + 1));

}

WifiOfdmMaskSlopesTestCase::~WifiOfdmMaskSlopesTestCase ()
{
}

void
WifiOfdmMaskSlopesTestCase::InterpolateAndAppendValues (IndexPowerVect &vect, IndexPowerPair start, IndexPowerPair stop,
                                                        double tol)
{
  NS_LOG_FUNCTION (start.first << start.second << stop.first << stop.second);
  NS_ASSERT (start.first <= stop.first);

  if (start.first == stop.first) //only one point, no need to interpolate
    {
      NS_ASSERT (start.second == stop.second);
      vect.push_back (start);
      NS_LOG_LOGIC ("Append (" << start.first << ", " << stop.second << ")");
      return;
    }

  double slope = (stop.second - start.second) / (stop.first - start.first);
  for (uint32_t i = start.first; i <= stop.first; i++)
    {
      double val = start.second + slope * (i - start.first);
      vect.push_back (std::make_pair (i, val));
      NS_LOG_LOGIC ("Append (" << i << ", " << val << ")");
    }
  NS_ASSERT (vect.back ().first == stop.first
             && TestDoubleIsEqual (vect.back ().second, stop.second, tol));
}

void
WifiOfdmMaskSlopesTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  double currentPowerDbr = 0.0; //have to work in dBr so as to compare with expected slopes
  double maxPowerW = (*m_actualSpectrum)[0];
  for (Values::const_iterator vit = m_actualSpectrum->ConstValuesBegin (); vit != m_actualSpectrum->ConstValuesEnd (); vit++)
    {
      maxPowerW = std::max (maxPowerW, *vit);
    }

  NS_LOG_INFO ("Compare expected left PSD");
  for (IndexPowerVect::const_iterator it = m_expectedLeftPsd.begin (); it != m_expectedLeftPsd.end (); it++)
    {
      currentPowerDbr = 10.0 * std::log10 ((*m_actualSpectrum)[it->first] / maxPowerW);
      NS_LOG_LOGIC ("For " << it->first << ", expected: " << it->second << " vs obtained: " << currentPowerDbr);
      NS_TEST_EXPECT_MSG_EQ_TOL (currentPowerDbr, it->second, m_tolerance,
                                 "Spectrum value mismatch for left guard band (" << it->first << ")");
    }
  NS_LOG_INFO ("Compare expected right PSD");
  for (IndexPowerVect::const_iterator it = m_expectedRightPsd.begin (); it != m_expectedRightPsd.end (); it++)
    {
      currentPowerDbr = 10.0 * std::log10 ((*m_actualSpectrum)[it->first] / maxPowerW);
      NS_LOG_LOGIC ("For " << it->first << ", expected: " << it->second << " vs obtained: " << currentPowerDbr);
      NS_TEST_EXPECT_MSG_EQ_TOL (currentPowerDbr, it->second, m_tolerance,
                                 "Spectrum value mismatch for right guard band (" << it->first << ")");
    }
}



/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test suite for checking the consistency of different OFDM-based transmit masks.
 */
class WifiTransmitMaskTestSuite : public TestSuite
{
public:
  WifiTransmitMaskTestSuite ();
};

static WifiTransmitMaskTestSuite g_WifiTransmitMaskTestSuite;

WifiTransmitMaskTestSuite::WifiTransmitMaskTestSuite ()
  : TestSuite ("wifi-transmit-mask", UNIT)
{
//  LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);
//  LogComponentEnable ("WifiTransmitMaskTest", logLevel);
//  LogComponentEnable ("WifiSpectrumValueHelper", logLevel);

  NS_LOG_INFO ("Creating WifiTransmitMaskTestSuite");

  WifiOfdmMaskSlopesTestCase::IndexPowerVect maskSlopesLeft;
  WifiOfdmMaskSlopesTestCase::IndexPowerVect maskSlopesRight;
  double tol = 0.001; // in dB

  // ============================================================================================
  // 11p 5MHz
  NS_LOG_FUNCTION ("Check slopes for 11p 5MHz");
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (31, -28.375)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (32, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (60, -20.276)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (63, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (64, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (69, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (123, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (128, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (129, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (132, -20.276)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (160, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (161, -28.375)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (192, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11p 5MHz", WIFI_STANDARD_80211p, WIFI_PHY_BAND_5GHZ,
                                               5, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11p 10MHz
  NS_LOG_FUNCTION ("Check slopes for 11p 10MHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (31, -28.375)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (32, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (60, -20.276)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (63, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (64, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (69, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (123, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (128, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (129, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (132, -20.276)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (160, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (161, -28.375)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (192, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11p 10MHz", WIFI_STANDARD_80211p, WIFI_PHY_BAND_5GHZ,
                                               10, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);


  // ============================================================================================
  // 11a
  NS_LOG_FUNCTION ("Check slopes for 11a");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (31, -28.375)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (32, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (60, -20.276)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (63, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (64, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (69, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (123, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (128, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (129, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (132, -20.276)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (160, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (161, -28.375)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (192, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11a", WIFI_STANDARD_80211a, WIFI_PHY_BAND_5GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11g
  NS_LOG_FUNCTION ("Check slopes for 11g");
  // same slppes as 11g
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11g", WIFI_STANDARD_80211g, WIFI_PHY_BAND_2_4GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);


  // ============================================================================================
  // 11n 20MHz @ 2.4GHz
  NS_LOG_FUNCTION ("Check slopes for 11n 20MHz @ 2.4GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -45.000)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (31, -28.531)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (32, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (60, -20.276)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (62, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (67, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (125, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (130, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (132, -20.276)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (160, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (161, -28.531)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (192, -45.000)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11n_2.4GHz 20MHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_2_4GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11n 20MHz @ 5GHz
  NS_LOG_FUNCTION ("Check slopes for 11n 20MHz @ 5GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (31, -28.375)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (32, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (60, -20.276)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (62, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (67, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (125, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (130, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (132, -20.276)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (160, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (161, -28.375)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (192, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11n_5GHz 20MHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_5GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11n 40MHz @ 2.4GHz
  NS_LOG_FUNCTION ("Check slopes for 11n 40MHz @ 2.4GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -45.000)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (63, -28.266)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (64, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (124, -20.131)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (125, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (125, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (126, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (131, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (253, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (258, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (259, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (259, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (260, -20.131)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (320, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (321, -28.266)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (384, -45.000)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11n_2.4GHz 40MHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_2_4GHZ,
                                               40, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11n 20MHz @ 5GHz
  NS_LOG_FUNCTION ("Check slopes for 11n 40MHz @ 5GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (63, -28.188)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (64, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (124, -20.131)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (125, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (125, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (126, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (131, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (253, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (258, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (259, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (259, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (260, -20.131)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (320, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (321, -28.188)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (384, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11n_5GHz 40MHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_5GHZ,
                                               40, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);


  // ============================================================================================
  // 11ac 20MHz
  NS_LOG_FUNCTION ("Check slopes for 11ac 20MHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (31, -28.375)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (32, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (60, -20.276)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (61, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (62, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (67, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (125, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (130, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (131, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (132, -20.276)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (160, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (161, -28.375)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (192, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ac 20MHz", WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ac 20MHz
  NS_LOG_FUNCTION ("Check slopes for 11ac 40MHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (63, -28.188)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (64, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (124, -20.131)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (125, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (125, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (126, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (131, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (253, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (258, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (259, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (259, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (260, -20.131)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (320, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (321, -28.188)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (384, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ac 40MHz", WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ,
                                               40, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ac 80MHz
  NS_LOG_FUNCTION ("Check slopes for 11ac 80MHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (127, -28.094)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (128, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (252, -20.064)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (253, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (253, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (254, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (259, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (509, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (514, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (515, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (515, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (516, -20.064)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (640, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (641, -28.094)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (768, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ac 80MHz", WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ,
                                               80, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ac 20MHz
  NS_LOG_FUNCTION ("Check slopes for 11ac 160MHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (255, -28.047)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (256, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (508, -20.032)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (509, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (509, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (510, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (515, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (1021, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (1026, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (1027, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (1027, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (1028, -20.032)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (1280, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (1281, -28.047)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (1536, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ac 160MHz", WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ,
                                               160, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);


  // ============================================================================================
  // 11ax 20MHz @ 2.4GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 20MHz @ 2.4GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -45.000)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (127, -28.133)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (128, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (252, -20.064)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (253, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (255, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (256, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (261, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (507, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (512, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (513, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (515, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (516, -20.064)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (640, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (641, -28.133)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (768, -45.000)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_2.4GHz 20MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_2_4GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ax 20MHz @ 5GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 20MHz @ 5GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (127, -28.094)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (128, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (252, -20.064)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (253, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (255, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (256, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (261, -3.333)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (507, -3.333)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (512, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (513, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (515, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (516, -20.064)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (640, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (641, -28.094)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (768, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_5GHz 20MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_5GHZ,
                                               20, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ax 40MHz @ 2.4GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 40MHz @ 2.4GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -45.000)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (255, -28.066)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (256, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (505, -20.032)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (506, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (510, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (511, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (523, -1.538)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (1013, -1.538)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (1025, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (1026, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (1030, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (1031, -20.032)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (1280, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (1281, -28.066)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (1536, -45.000)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_2.4GHz 40MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_2_4GHZ,
                                               40, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ax 40MHz @ 5GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 40MHz @ 5GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (255, -28.047)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (256, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (505, -20.032)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (506, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (510, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (511, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (523, -1.538)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (1013, -1.538)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (1025, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (1026, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (1030, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (1031, -20.032)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (1280, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (1281, -28.047)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (1536, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_5GHz 40MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_5GHZ,
                                               40, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ax 80MHz @ 2.4GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 80MHz @ 2.4GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -45.000)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (511, -28.033)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (512, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (1017, -20.016)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (1018, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (1022, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (1023, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (1035, -1.538)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (2037, -1.538)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (2049, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (2050, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (2054, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (2055, -20.016)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (2560, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (2561, -28.033)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (3072, -45.000)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_2.4GHz 80MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_2_4GHZ,
                                               80, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ax 80MHz @ 5GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 80MHz @ 5GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (511, -28.023)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (512, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (1017, -20.016)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (1018, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (1022, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (1023, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (1035, -1.538)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (2037, -1.538)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (2049, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (2050, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (2054, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (2055, -20.016)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (2560, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (2561, -28.023)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (3072, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_5GHz 80MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_5GHZ,
                                               80, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  // 11ax 160MHz @ 2.4GHz -> not enough space so skip

  // 11ax 160MHz @ 5GHz
  NS_LOG_FUNCTION ("Check slopes for 11ax 160MHz @ 5GHz");
  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
  maskSlopesLeft.push_back (std::make_pair (0, -40.0)); // Outer band left (start)
  maskSlopesLeft.push_back (std::make_pair (1023, -28.012)); // Outer band left (stop)
  maskSlopesLeft.push_back (std::make_pair (1024, -28.000)); // Middle band left (start)
  maskSlopesLeft.push_back (std::make_pair (2041, -20.008)); // Middle band left (stop)
  maskSlopesLeft.push_back (std::make_pair (2042, -20.0)); // Flat junction band left (start)
  maskSlopesLeft.push_back (std::make_pair (2046, -20.0)); // Flat junction band left (stop)
  maskSlopesLeft.push_back (std::make_pair (2047, -20.0)); // Inner band left (start)
  maskSlopesLeft.push_back (std::make_pair (2059, -1.538)); // Inner band left (stop)
  maskSlopesRight.push_back (std::make_pair (4085, -1.538)); // Inner band right (start)
  maskSlopesRight.push_back (std::make_pair (4097, -20.0)); // Inner band right (stop)
  maskSlopesRight.push_back (std::make_pair (4098, -20.0)); // Flat junction band right (start)
  maskSlopesRight.push_back (std::make_pair (4102, -20.0)); // Flat junction band right (stop)
  maskSlopesRight.push_back (std::make_pair (4103, -20.008)); // Middle band right (start)
  maskSlopesRight.push_back (std::make_pair (5120, -28.000)); // Middle band right (stop)
  maskSlopesRight.push_back (std::make_pair (5121, -28.012)); // Outer band right (start)
  maskSlopesRight.push_back (std::make_pair (6144, -40.0)); // Outer band right (stop)
  AddTestCase (new WifiOfdmMaskSlopesTestCase ("11ax_5GHz 160MHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_5GHZ,
                                               160, maskSlopesLeft, maskSlopesRight, tol),
               TestCase::QUICK);

  maskSlopesLeft.clear ();
  maskSlopesRight.clear ();
}
