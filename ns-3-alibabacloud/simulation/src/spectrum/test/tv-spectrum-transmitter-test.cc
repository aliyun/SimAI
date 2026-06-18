/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 * Author: Benjamin Cizdziel <ben.cizdziel@gmail.com>
 */

#include <ns3/test.h>
#include <ns3/log.h>
#include <ns3/spectrum-value.h>
#include <ns3/enum.h>
#include <ns3/double.h>
#include <ns3/tv-spectrum-transmitter.h>

NS_LOG_COMPONENT_DEFINE ("TvSpectrumTransmitterTest");

using namespace ns3;

const double TOLERANCE = 1e-15;
// Bug 2094: Adjust floating point comparison epsilon based on inputs.
//           Follows http://realtimecollisiondetection.net/blog/?p=89
double epsilon;

/**
 * \ingroup spectrum-tests
 *
 * This test verifies the accuracy of the spectrum/PSD model in the 
 * TvSpectrumTransmitter class. To do so, it tests if the max power spectral 
 * density, start frequency, and end frequency comply with expected values. 
 * Values for TV/modulation type, start frequency, channel bandwidth, and 
 * base PSD are swept and tested for each case.
 */
class TvSpectrumTransmitterTestCase : public TestCase
{
public:
  /**
   * Constructor
   * \param startFrequency Start frequency.
   * \param channelBandwidth Channel Bandwidth.
   * \param basePsd Base Power Spectral Density (PSD).
   * \param tvType TV type.
   */
  TvSpectrumTransmitterTestCase (double startFrequency, 
                            double channelBandwidth, 
                            double basePsd, 
                            TvSpectrumTransmitter::TvType tvType);
  virtual ~TvSpectrumTransmitterTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Build the test name
   * \param tvType TV type.
   * \param startFrequency Start frequency.
   * \param channelBandwidth Channel Bandwidth.
   * \param basePsd Base Power Spectral Density (PSD).
   * \return The test name
   */
  static std::string Name (TvSpectrumTransmitter::TvType tvType, 
                           double startFrequency, 
                           double channelBandwidth, 
                           double basePsd);

  double m_startFrequency;      //!< Start frequency.
  double m_channelBandwidth;    //!< Channel Bandwidth.
  double m_basePsd;             //!< Base Power Spectral Density (PSD).
  TvSpectrumTransmitter::TvType m_tvType; //!< TV type.
};


std::string 
TvSpectrumTransmitterTestCase::Name (TvSpectrumTransmitter::TvType tvType, 
                                     double startFrequency, 
                                     double channelBandwidth, 
                                     double basePsd)
{
  std::ostringstream oss;
  oss << "TV type = " << tvType << ", "
      << "start frequency = " << startFrequency << " Hz, "
      << "channel bandwidth = " << channelBandwidth << " Hz, "
      << "base PSD = " << basePsd << " dBm per Hz";
  return oss.str();
}

TvSpectrumTransmitterTestCase::TvSpectrumTransmitterTestCase (double startFrequency,
						    double channelBandwidth,
						    double basePsd,
						    TvSpectrumTransmitter::TvType tvType)
  : TestCase (Name (tvType, startFrequency, channelBandwidth, basePsd)),
    m_startFrequency (startFrequency),
    m_channelBandwidth (channelBandwidth),
    m_basePsd (basePsd),
    m_tvType (tvType)
{
}

TvSpectrumTransmitterTestCase::~TvSpectrumTransmitterTestCase ()
{
}

void
TvSpectrumTransmitterTestCase::DoRun (void)
{  
  NS_LOG_FUNCTION (m_startFrequency << m_basePsd << m_tvType);

  /* TV transmitter setup */
  Ptr<TvSpectrumTransmitter> phy = CreateObject<TvSpectrumTransmitter>();
  phy->SetAttribute ("StartFrequency", DoubleValue (m_startFrequency));
  phy->SetAttribute ("ChannelBandwidth", DoubleValue (m_channelBandwidth));
  phy->SetAttribute ("BasePsd", DoubleValue (m_basePsd)); 
  phy->SetAttribute ("TvType", EnumValue (m_tvType));
  phy->CreateTvPsd ();

  /* Test max PSD value */
  Ptr<SpectrumValue> psd = phy->GetTxPsd ();
  Values::const_iterator psdIter = psd->ConstValuesBegin ();
  double maxValue = 0;
  while (psdIter != psd->ConstValuesEnd ())
    {
      if (*psdIter > maxValue)
        {
          maxValue = *psdIter;
        }
      ++psdIter;
    }
  double basePsdWattsHz = pow (10.0, (m_basePsd - 30) / 10.0); // convert dBm to W/Hz
  if (m_tvType == TvSpectrumTransmitter::TVTYPE_8VSB) // pilot has highest PSD
    {
      double expectedPsd = (0.502 * basePsdWattsHz) + (21.577 * basePsdWattsHz);
      epsilon = TOLERANCE * std::max (1.0, std::max (maxValue, expectedPsd));
      NS_TEST_ASSERT_MSG_EQ_TOL (maxValue, 
                                 expectedPsd, 
                                 epsilon,
                                 "peak PSD value (" << maxValue << ") is incorrect");
    }
  else // highest PSD is base PSD
    {
      epsilon = TOLERANCE * std::max (1.0, std::max (maxValue, basePsdWattsHz));
      NS_TEST_ASSERT_MSG_EQ_TOL (maxValue, 
                                 basePsdWattsHz, 
                                 epsilon, 
                                 "peak PSD value (" << maxValue << ") is incorrect");
    }

  /* Test frequency range */
  Bands::const_iterator bandStart = psd->ConstBandsBegin ();
  Bands::const_iterator bandEnd = psd->ConstBandsEnd ();
  epsilon = TOLERANCE * std::max (1.0, std::max ((*bandStart).fc, m_startFrequency));
  NS_TEST_ASSERT_MSG_EQ_TOL ((*bandStart).fc, 
                             m_startFrequency, 
                             epsilon, 
                             "start frequency value (" << (*bandStart).fc << ") is incorrect");
  epsilon = TOLERANCE * std::max (1.0, std::max ((*bandStart).fc, (m_startFrequency + m_channelBandwidth)));
  NS_TEST_ASSERT_MSG_EQ_TOL ((*(bandEnd - 1)).fc, 
                             m_startFrequency + m_channelBandwidth, 
                             epsilon, 
                             "end frequency value (" << (*(bandEnd - 1)).fc << ") is incorrect");
}


/**
 * \ingroup spectrum-tests
 *
 * Test suite for the TvSpectrumTransmitter class
 */
class TvSpectrumTransmitterTestSuite : public TestSuite
{
public:
  TvSpectrumTransmitterTestSuite ();
};

TvSpectrumTransmitterTestSuite::TvSpectrumTransmitterTestSuite ()
  : TestSuite ("tv-spectrum-transmitter", UNIT)
{
  NS_LOG_INFO ("creating TvSpectrumTransmitterTestSuite");
  for (double startFreq = 100; startFreq < 1e15; startFreq *= 10)
    {
      for (double bandwidth = 100; bandwidth < 1e15; bandwidth *= 10)
        {
          for (double psd = -100; psd <= 100; psd += 20)
            {
              AddTestCase (new TvSpectrumTransmitterTestCase (startFreq, 
                                                              bandwidth, 
                                                              psd, 
                                                              TvSpectrumTransmitter::TVTYPE_8VSB), 
                           TestCase::QUICK);
            }
        }
    }
  for (double startFreq = 100; startFreq < 1e15; startFreq *= 10)
    {
      for (double bandwidth = 100; bandwidth < 1e15; bandwidth *= 10)
        {
          for (double psd = -100; psd <= 100; psd += 20)
            {
              AddTestCase (new TvSpectrumTransmitterTestCase (startFreq, 
                                                              bandwidth, 
                                                              psd, 
                                                              TvSpectrumTransmitter::TVTYPE_COFDM), 
                           TestCase::QUICK);
            }
        }
    }
  for (double startFreq = 100; startFreq < 1e15; startFreq *= 10)
    {
      for (double bandwidth = 100; bandwidth < 1e15; bandwidth *= 10)
        {
          for (double psd = -100; psd <= 100; psd += 20)
            {
              AddTestCase (new TvSpectrumTransmitterTestCase (startFreq, 
                                                              bandwidth, 
                                                              psd, 
                                                              TvSpectrumTransmitter::TVTYPE_ANALOG), 
                           TestCase::QUICK);
            }
        }
    }
}

/// Static variable for test initialization
static TvSpectrumTransmitterTestSuite g_tvSpectrumTransmitterTestSuite;
