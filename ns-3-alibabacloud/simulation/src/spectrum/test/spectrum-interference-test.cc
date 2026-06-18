/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include <ns3/object.h>
#include <ns3/spectrum-interference.h>
#include <ns3/spectrum-error-model.h>
#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/simulator.h>
#include <ns3/packet.h>
#include <ns3/ptr.h>
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SpectrumInterferenceTest");



/**
 * \ingroup spectrum-tests
 *
 * \brief Spectrum Interference Test
 */
class SpectrumInterferenceTestCase : public TestCase
{
public:
  /**
   * Constructor
   * \param s Spectrum value
   * \param txBytes number of bytes to transmit
   * \param rxCorrect extected Rx bytes
   * \param name test name
   */
  SpectrumInterferenceTestCase (Ptr<SpectrumValue> s, uint32_t txBytes, bool rxCorrect, std::string name);
  virtual ~SpectrumInterferenceTestCase ();
  virtual void DoRun (void);
  /**
   * Retrieve the test results
   * \param si SpectrumInterference instance
   */
  void RetrieveTestResult (SpectrumInterference* si);

private:
  Ptr<SpectrumValue> m_s;             //!< Spectrum value
  uint32_t m_txBytes;                 //!< number of bytes to transmit
  uint32_t m_rxCorrectKnownOutcome;   //!< extected Rx bytes
  Ptr<const SpectrumModel> m_mySpectrumModel; //!< Spectrum model pointer
};



SpectrumInterferenceTestCase::SpectrumInterferenceTestCase (Ptr<SpectrumValue> s, uint32_t txBytes, bool rxCorrect, std::string name)
  : TestCase (name),
    m_s (s),
    m_txBytes (txBytes),
    m_rxCorrectKnownOutcome (rxCorrect),
    m_mySpectrumModel (s->GetSpectrumModel ())
{
}

SpectrumInterferenceTestCase::~SpectrumInterferenceTestCase ()
{
}



void
SpectrumInterferenceTestCase::DoRun (void)
{
  Ptr<SpectrumValue> n = Create<SpectrumValue> (m_mySpectrumModel);
  Ptr<SpectrumValue> i1 = Create<SpectrumValue> (m_mySpectrumModel);
  Ptr<SpectrumValue> i2 = Create<SpectrumValue> (m_mySpectrumModel);
  Ptr<SpectrumValue> i3 = Create<SpectrumValue> (m_mySpectrumModel);
  Ptr<SpectrumValue> i4 = Create<SpectrumValue> (m_mySpectrumModel);


  (*n)[0] = 5.000000000000e-19;
  (*n)[1] = 4.545454545455e-19;

  (*i1)[0] = 5.000000000000e-18;
  (*i2)[0] = 5.000000000000e-16;
  (*i3)[0] = 1.581138830084e-16;
  (*i4)[0] = 7.924465962306e-17;
  (*i1)[1] = 1.437398936440e-18;
  (*i2)[1] = 5.722388235428e-16;
  (*i3)[1] = 7.204059965732e-17;
  (*i4)[1] = 5.722388235428e-17;


  SpectrumInterference si;
  si.SetErrorModel (CreateObject<ShannonSpectrumErrorModel> ());
  si.SetNoisePowerSpectralDensity (n);

  Time ts  = Seconds (1);
  Time ds  = Seconds (1);
  Time ti1 = Seconds (0);
  Time di1 = Seconds (3);
  Time ti2 = Seconds (0.7);
  Time di2 = Seconds (1);
  Time ti3 = Seconds (1.2);
  Time di3 = Seconds (1);
  Time ti4 = Seconds (1.5);
  Time di4 = Seconds (0.1);

  Simulator::Schedule (ts,  &SpectrumInterference::AddSignal, &si, m_s,  ds);
  Simulator::Schedule (ti1, &SpectrumInterference::AddSignal, &si, i1, di1);
  Simulator::Schedule (ti2, &SpectrumInterference::AddSignal, &si, i2, di2);
  Simulator::Schedule (ti3, &SpectrumInterference::AddSignal, &si, i3, di3);
  Simulator::Schedule (ti4, &SpectrumInterference::AddSignal, &si, i4, di4);

  Ptr<Packet> p = Create<Packet> (m_txBytes);
  Simulator::Schedule (ts,  &SpectrumInterference::StartRx, &si, p, m_s);
  Simulator::Schedule (ts + ds,  &SpectrumInterferenceTestCase::RetrieveTestResult, this, &si);

  Simulator::Run ();
  // the above will return and after RetrieveTestResults have
  // been called and after all signals have expired
  Simulator::Destroy ();
}



void
SpectrumInterferenceTestCase::RetrieveTestResult (SpectrumInterference* si)
{
  NS_TEST_ASSERT_MSG_EQ (si->EndRx (), m_rxCorrectKnownOutcome, "no message");
}



/**
 * \ingroup spectrum-tests
 *
 * \brief Spectrum Interference TestSuite
 */
class SpectrumInterferenceTestSuite : public TestSuite
{
public:
  SpectrumInterferenceTestSuite ();
};

SpectrumInterferenceTestSuite::SpectrumInterferenceTestSuite ()
  : TestSuite ("spectrum-interference", UNIT)
{

  NS_LOG_INFO ("creating SpectrumInterferenceTestSuite");

  Ptr<const SpectrumModel> m;

  Bands bands;
  struct BandInfo bi;

  bi.fl = 2.400e9;
  bi.fc = 2.410e9;
  bi.fh = 2.420e9;
  bands.push_back (bi);

  bi.fl = 2.420e9;
  bi.fc = 2.431e9;
  bi.fh = 2.442e9;
  bands.push_back (bi);

  m = Create<SpectrumModel> (bands);


  double b; // max deliverable bytes

  const double e = 1e-5; // max tolerated relative error for
                         // deliverable bytes

  // Power Spectral Density of the signal of interest  = [-46 -48] dBm;
  Ptr<SpectrumValue> s1  = Create<SpectrumValue> (m);
  (*s1)[0] = 1.255943215755e-15;
  (*s1)[1] = 7.204059965732e-16;
  b = 10067205.5632012;
  AddTestCase (new SpectrumInterferenceTestCase (s1,   0, true,            "sdBm  = [-46 -48]  tx bytes: 1"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s1, static_cast<uint32_t> (b * 0.5 + 0.5), true,   "sdBm  = [-46 -48]  tx bytes: b*0.5"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s1, static_cast<uint32_t> (b * (1 - e) + 0.5), true, "sdBm  = [-46 -48]  tx bytes: b*(1-e)"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s1, static_cast<uint32_t> (b * (1 + e) + 0.5), false, "sdBm  = [-46 -48]  tx bytes: b*(1+e)"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s1, static_cast<uint32_t> (b * 1.5 + 0.5), false,   "sdBm  = [-46 -48]  tx bytes: b*1.5"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s1, 0xffffffff, false,     "sdBm  = [-46 -48]  tx bytes: 2^32-1"), TestCase::QUICK);

  // Power Spectral Density of the signal of interest  =  [-63 -61] dBm;
  Ptr<SpectrumValue> s2  = Create<SpectrumValue> (m);
  (*s2)[0] = 2.505936168136e-17;
  (*s2)[1] = 3.610582885110e-17;
  b = 882401.591840728;
  AddTestCase (new SpectrumInterferenceTestCase (s2,   1, true,            "sdBm  = [-63 -61]  tx bytes: 1"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s2, static_cast<uint32_t> (b * 0.5 + 0.5), true,   "sdBm  = [-63 -61]  tx bytes: b*0.5"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s2, static_cast<uint32_t> (b * (1 - e) + 0.5), true, "sdBm  = [-63 -61]  tx bytes: b*(1-e)"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s2, static_cast<uint32_t> (b * (1 + e) + 0.5), false, "sdBm  = [-63 -61]  tx bytes: b*(1+e)"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s2, static_cast<uint32_t> (b * 1.5 + 0.5), false,   "sdBm  = [-63 -61]  tx bytes: b*1.5"), TestCase::QUICK);
  AddTestCase (new SpectrumInterferenceTestCase (s2, 0xffffffff, false,     "sdBm  = [-63 -61]  tx bytes: 2^32-1"), TestCase::QUICK);

}

/// Static variable for test initialization
static SpectrumInterferenceTestSuite spectrumInterferenceTestSuite;
