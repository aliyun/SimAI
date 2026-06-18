/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
#include <ns3/string.h>
#include <iostream>
#include <ns3/math.h>
#include <ns3/spectrum-model-ism2400MHz-res1MHz.h>
#include <ns3/spectrum-model-300kHz-300GHz-log.h>
#include <ns3/wifi-spectrum-value-helper.h>
#include <ns3/single-model-spectrum-channel.h>
#include <ns3/waveform-generator.h>
#include <ns3/spectrum-analyzer.h>
#include <string>
#include <iomanip>
#include <ns3/friis-spectrum-propagation-loss.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/spectrum-helper.h>
#include <ns3/adhoc-aloha-noack-ideal-phy-helper.h>
#include <ns3/mobility-helper.h>
#include <ns3/data-rate.h>
#include <ns3/uinteger.h>
#include <ns3/packet-socket-helper.h>
#include <ns3/packet-socket-address.h>
#include <ns3/packet-socket-client.h>
#include <ns3/config.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SpectrumIdealPhyTest");

static uint64_t g_rxBytes;
static double g_bandwidth = 20e6; // Hz

void
PhyRxEndOkTrace (std::string context, Ptr<const Packet> p)
{
  g_rxBytes += p->GetSize ();
}


/**
 * \ingroup spectrum-tests
 *
 * \brief Ideal Spectrum PHY Test
 */
class SpectrumIdealPhyTestCase : public TestCase
{
public:
  /**
   * Constructor
   * \param snrLinear SNR (linear)
   * \param phyRate PHY rate (bps)
   * \param rateIsAchievable Check if the rate is achievable
   * \param channelType Channel type
   */
  SpectrumIdealPhyTestCase (double snrLinear,
			    uint64_t phyRate,
			    bool rateIsAchievable,
			    std::string channelType);
  virtual ~SpectrumIdealPhyTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Get the test name
   * \param channelType Channel type
   * \param snrLinear SNR (linear)
   * \param phyRate PHY rate (bps)
   * \return the test name
   */
  static std::string Name (std::string channelType, double snrLinear, uint64_t phyRate);
  
  double      m_snrLinear;        //!< SNR (linear)
  uint64_t    m_phyRate;          //!< PHY rate (bps)
  bool        m_rateIsAchievable; //!< Check if the rate is achievable
  std::string m_channelType;      //!< Channel type
};

std::string 
SpectrumIdealPhyTestCase::Name (std::string channelType, double snrLinear, uint64_t phyRate)
{
  std::ostringstream oss;
  oss << channelType
      << " snr = " << snrLinear << " (linear), "
      << " phyRate = " << phyRate << " bps";
  return oss.str();
}


SpectrumIdealPhyTestCase::SpectrumIdealPhyTestCase (double snrLinear,
						    uint64_t phyRate,
						    bool rateIsAchievable,
						    std::string channelType)
  : TestCase (Name (channelType, snrLinear, phyRate)),
    m_snrLinear (snrLinear),
    m_phyRate (phyRate),
    m_rateIsAchievable (rateIsAchievable),
    m_channelType (channelType)
{
}

SpectrumIdealPhyTestCase::~SpectrumIdealPhyTestCase ()
{
}


void
SpectrumIdealPhyTestCase::DoRun (void)
{  
  NS_LOG_FUNCTION (m_snrLinear << m_phyRate);
  double txPowerW = 0.1; 
  // for the noise, we use the Power Spectral Density of thermal noise
  // at room temperature. The value of the PSD will be constant over the band of interest.
  const double k = 1.381e-23; //Boltzmann's constant
  const double T = 290; // temperature in Kelvin
  double noisePsdValue = k * T; // W/Hz
  double lossLinear = (txPowerW) / (m_snrLinear * noisePsdValue * g_bandwidth); 
  double lossDb = 10 * std::log10 (lossLinear);
  uint64_t phyRate = m_phyRate; // bps
  uint32_t pktSize = 50; // bytes

  uint32_t numPkts = 200; //desired number of packets in the
			  //test. Directly related with the accuracy
			  //of the measurement. 

  double testDuration = (numPkts * pktSize * 8.0) / phyRate;
  NS_LOG_INFO ("test duration = " << std::fixed << testDuration);

  NodeContainer c;
  c.Create (2);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");


  mobility.Install (c);


  SpectrumChannelHelper channelHelper;
  channelHelper.SetChannel (m_channelType);
  channelHelper.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  Ptr<MatrixPropagationLossModel> propLoss = CreateObject<MatrixPropagationLossModel> ();  
  propLoss->SetLoss (c.Get(0)->GetObject<MobilityModel> (), c.Get(1)->GetObject<MobilityModel> (), lossDb, true);
  channelHelper.AddPropagationLoss (propLoss);
  Ptr<SpectrumChannel> channel = channelHelper.Create ();


  WifiSpectrumValue5MhzFactory sf;

  uint32_t channelNumber = 1;
  Ptr<SpectrumValue> txPsd =  sf.CreateTxPowerSpectralDensity (txPowerW, channelNumber);

  Ptr<SpectrumValue> noisePsd = sf.CreateConstant (noisePsdValue);

  AdhocAlohaNoackIdealPhyHelper deviceHelper;
  deviceHelper.SetChannel (channel);
  deviceHelper.SetTxPowerSpectralDensity (txPsd);
  deviceHelper.SetNoisePowerSpectralDensity (noisePsd);
  deviceHelper.SetPhyAttribute ("Rate", DataRateValue (DataRate (phyRate)));
  NetDeviceContainer devices = deviceHelper.Install (c);

  PacketSocketHelper packetSocket;
  packetSocket.Install (c);

  PacketSocketAddress socket;
  socket.SetSingleDevice (devices.Get (0)->GetIfIndex ());
  socket.SetPhysicalAddress (devices.Get (1)->GetAddress ());
  socket.SetProtocol (1);

  Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
  client->SetRemote (socket);
  client->SetAttribute ("Interval", TimeValue (Seconds (double (pktSize*8) / (1.2*double (phyRate)))));
  client->SetAttribute ("PacketSize", UintegerValue (pktSize));
  client->SetAttribute ("MaxPackets", UintegerValue (0));
  client->SetStartTime(Seconds (0.0));
  client->SetStopTime(Seconds (testDuration));
  c.Get (0)->AddApplication (client);

  Config::Connect ("/NodeList/*/DeviceList/*/Phy/RxEndOk", MakeCallback (&PhyRxEndOkTrace));

  g_rxBytes = 0;
  Simulator::Stop (Seconds (testDuration+0.000000001));
  Simulator::Run ();
  double throughputBps = (g_rxBytes * 8.0) / testDuration;

  std::clog.unsetf(std::ios_base::floatfield);
  
  if (m_rateIsAchievable)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL (throughputBps, m_phyRate, m_phyRate*0.01, "throughput does not match PHY rate");
    }
  else
    {
      NS_TEST_ASSERT_MSG_EQ (throughputBps, 0.0, "PHY rate is not achievable but throughput is non-zero");    
    }

  Simulator::Destroy ();
}



/**
 * \ingroup spectrum-tests
 *
 * \brief Ideal Spectrum PHY TestSuite
 */
class SpectrumIdealPhyTestSuite : public TestSuite
{
public:
  SpectrumIdealPhyTestSuite ();
};

SpectrumIdealPhyTestSuite::SpectrumIdealPhyTestSuite ()
  : TestSuite ("spectrum-ideal-phy", SYSTEM)
{

  NS_LOG_INFO ("creating SpectrumIdealPhyTestSuite");
    
  for (double snr = 0.01; snr <= 10 ; snr *= 2)
    {          
      double achievableRate = g_bandwidth*log2(1+snr);      
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*0.1),  true,  "ns3::SingleModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*0.5),  true,  "ns3::SingleModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*0.95), true,  "ns3::SingleModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*1.05), false,  "ns3::SingleModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*2),    false,  "ns3::SingleModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*4),    false,  "ns3::SingleModelSpectrumChannel"), TestCase::QUICK);
    }
  for (double snr = 0.01; snr <= 10 ; snr *= 10)
    {          
      double achievableRate = g_bandwidth*log2(1+snr);      
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*0.1),  true,  "ns3::MultiModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*0.5),  true,  "ns3::MultiModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*0.95), true,  "ns3::MultiModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*1.05), false,  "ns3::MultiModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*2),    false,  "ns3::MultiModelSpectrumChannel"), TestCase::QUICK);
      AddTestCase (new SpectrumIdealPhyTestCase (snr, static_cast<uint64_t> (achievableRate*4),    false,  "ns3::MultiModelSpectrumChannel"), TestCase::QUICK);
    }
}

/// Static variable for test initialization
static SpectrumIdealPhyTestSuite g_spectrumIdealPhyTestSuite;
