/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 * Modified by Marco Miozzo <mmiozzo@ctt.es>
 *     Extend to Data and SRS frames
 */

#include "ns3/simulator.h"

#include "ns3/log.h"

#include "ns3/spectrum-test.h"

#include "ns3/lte-phy-tag.h"
#include "lte-test-ue-phy.h"
#include "ns3/lte-spectrum-signal-parameters.h"

#include "lte-test-uplink-sinr.h"

#include <ns3/lte-helper.h>
#include <ns3/lte-chunk-processor.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteUplinkSinrTest");

/**
 * Test 1.2 SINR calculation in uplink
 */

/**
 * TestSuite
 */
LteUplinkSinrTestSuite::LteUplinkSinrTestSuite ()
  : TestSuite ("lte-uplink-sinr", SYSTEM)
{
  /**
   * Build Spectrum Model values for the TX signal
   */
  Ptr<SpectrumModel> sm;

  Bands bands;
  BandInfo bi;

  bi.fl = 2.400e9;
  bi.fc = 2.410e9;
  bi.fh = 2.420e9;
  bands.push_back (bi);

  bi.fl = 2.420e9;
  bi.fc = 2.431e9;
  bi.fh = 2.442e9;
  bands.push_back (bi);

  sm = Create<SpectrumModel> (bands);

  /**
   * TX signals #1: Power Spectral Density (W/Hz) of the signals of interest = [-46 -inf] and [-inf -48] dBm and BW = [20 22] MHz
   */
  Ptr<SpectrumValue> rxPsd1 = Create<SpectrumValue> (sm);
  (*rxPsd1)[0] = 1.255943215755e-15;
  (*rxPsd1)[1] = 0.0;

  Ptr<SpectrumValue> rxPsd2 = Create<SpectrumValue> (sm);
  (*rxPsd2)[0] = 0.0;
  (*rxPsd2)[1] = 7.204059965732e-16;

  Ptr<SpectrumValue> theoreticalSinr1 = Create<SpectrumValue> (sm);
  (*theoreticalSinr1)[0] = 3.72589167251055;
  (*theoreticalSinr1)[1] = 3.72255684126076;

  AddTestCase (new LteUplinkDataSinrTestCase (rxPsd1, rxPsd2, theoreticalSinr1, "sdBm = [-46 -inf] and [-inf -48]"), TestCase::QUICK);
  
  AddTestCase (new LteUplinkSrsSinrTestCase (rxPsd1, rxPsd2, theoreticalSinr1, "sdBm = [-46 -inf] and [-inf -48]"), TestCase::QUICK);

  /**
   * TX signals #2: Power Spectral Density of the signals of interest = [-63 -inf] and [-inf -61] dBm and BW = [20 22] MHz
   */
  Ptr<SpectrumValue> rxPsd3 = Create<SpectrumValue> (sm);
  (*rxPsd3)[0] = 2.505936168136e-17;
  (*rxPsd3)[1] = 0.0;

  Ptr<SpectrumValue> rxPsd4 = Create<SpectrumValue> (sm);
  (*rxPsd4)[0] = 0.0;
  (*rxPsd4)[1] = 3.610582885110e-17;

  Ptr<SpectrumValue> theoreticalSinr2 = Create<SpectrumValue> (sm);
  (*theoreticalSinr2)[0] = 0.0743413124381667;
  (*theoreticalSinr2)[1] = 0.1865697965291756;

  AddTestCase (new LteUplinkDataSinrTestCase (rxPsd3, rxPsd4, theoreticalSinr2, "sdBm = [-63 -inf] and [-inf -61]"), TestCase::QUICK);
  
  AddTestCase (new LteUplinkSrsSinrTestCase (rxPsd3, rxPsd4, theoreticalSinr2, "sdBm = [-63 -inf] and [-inf -61]"), TestCase::QUICK);

}

static LteUplinkSinrTestSuite lteUplinkSinrTestSuite;


/**
 * TestCase Data
 */

LteUplinkDataSinrTestCase::LteUplinkDataSinrTestCase (Ptr<SpectrumValue> sv1, Ptr<SpectrumValue> sv2, Ptr<SpectrumValue> sinr, std::string name)
  : TestCase ("SINR calculation in uplink data frame: " + name),
    m_sv1 (sv1),
    m_sv2 (sv2),
    m_sm (sv1->GetSpectrumModel ()),
    m_expectedSinr (sinr)
{
  NS_LOG_INFO ("Creating LteUplinkDataSinrTestCase");
}

LteUplinkDataSinrTestCase::~LteUplinkDataSinrTestCase ()
{
}


void
LteUplinkDataSinrTestCase::DoRun (void)
{
  /**
   * Instantiate a single receiving LteSpectrumPhy
   */
  Ptr<LteSpectrumPhy> dlPhy = CreateObject<LteSpectrumPhy> ();
  Ptr<LteSpectrumPhy> ulPhy = CreateObject<LteSpectrumPhy> ();
  Ptr<LteTestUePhy> uePhy = CreateObject<LteTestUePhy> (dlPhy, ulPhy);
  uint16_t cellId = 100;
  dlPhy->SetCellId (cellId);
  ulPhy->SetCellId (cellId);

  Ptr<LteChunkProcessor> chunkProcessor = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher actualSinrCatcher;
  chunkProcessor->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &actualSinrCatcher));
  ulPhy->AddDataSinrChunkProcessor (chunkProcessor);

  /**
  * Generate several calls to LteSpectrumPhy::StartRx corresponding to 
  * several signals. One will be the signal of interest, i.e., the
  *  LteSpectrumSignalParametersDataFrame of the Packet burst
  * will have the same CellId of the receiving PHY; the others will have 
  * a different CellId and hence will be the interfering signals
  */

  // Number of packet bursts (2 data + 4 interferences)
  const int numOfDataPbs = 2;
  const int numOfIntfPbs = 4;
  const int numOfPbs = numOfDataPbs + numOfIntfPbs;

  // Number of packets in the packet bursts
  const int numOfPkts = 10;

  // Packet bursts
  Ptr<PacketBurst> packetBursts[numOfPbs];

  // Packets
  Ptr<Packet> pkt[numOfPbs][numOfPkts];

  // Bursts cellId
  uint16_t pbCellId[numOfPbs];
  


  /**
   * Build packet burst (Data and interference)
   */
  int pb = 0;
  for ( int dataPb = 0 ; dataPb < numOfDataPbs ; dataPb++, pb++ )
    {
      // Create packet burst
      packetBursts[pb] = CreateObject<PacketBurst> ();
      pbCellId[pb] = cellId;
      // Create packets and add them to the burst
      for ( int i = 0 ; i < numOfPkts ; i++ )
        {
          pkt[pb][i] = Create<Packet> (1000);

          packetBursts[pb]->AddPacket ( pkt[pb][i] );
        }
    }
  for ( int intfPb = 0 ; intfPb < numOfIntfPbs ; intfPb++, pb++ )
    {
      // Create packet burst
      packetBursts[pb] = CreateObject<PacketBurst> ();
      pbCellId[pb] = cellId * (pb + 1);

      // Create packets and add them to the burst
      for ( int i = 0 ; i < numOfPkts ; i++ )
        {
          pkt[pb][i] = Create<Packet> (1000);

          packetBursts[pb]->AddPacket ( pkt[pb][i] );
        }
    }


  Ptr<SpectrumValue> noisePsd = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i1 = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i2 = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i3 = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i4 = Create<SpectrumValue> (m_sm);

  (*noisePsd)[0] = 5.000000000000e-19;
  (*noisePsd)[1] = 4.545454545455e-19;

  (*i1)[0] = 5.000000000000e-18;
  (*i2)[0] = 5.000000000000e-16;
  (*i3)[0] = 1.581138830084e-16;
  (*i4)[0] = 7.924465962306e-17;
  (*i1)[1] = 1.437398936440e-18;
  (*i2)[1] = 5.722388235428e-16;
  (*i3)[1] = 7.204059965732e-17;
  (*i4)[1] = 5.722388235428e-17;

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

  ulPhy->SetNoisePowerSpectralDensity (noisePsd);

  /**
   * Schedule the reception of the data signals plus the interference signals
   */

  // 2 UEs send data to the eNB through 2 subcarriers
  Ptr<LteSpectrumSignalParametersDataFrame> sp1 = Create<LteSpectrumSignalParametersDataFrame> ();
  sp1->psd = m_sv1;
  sp1->txPhy = 0;
  sp1->duration = ds;
  sp1->packetBurst = packetBursts[0];
  sp1->cellId = pbCellId[0];
  Simulator::Schedule (ts, &LteSpectrumPhy::StartRx, ulPhy, sp1);

  Ptr<LteSpectrumSignalParametersDataFrame> sp2 = Create<LteSpectrumSignalParametersDataFrame> ();
  sp2->psd = m_sv2;
  sp2->txPhy = 0;
  sp2->duration = ds;
  sp2->packetBurst = packetBursts[1];
  sp2->cellId = pbCellId[1];
  Simulator::Schedule (ts, &LteSpectrumPhy::StartRx, ulPhy, sp2);


  Ptr<LteSpectrumSignalParametersDataFrame> ip1 = Create<LteSpectrumSignalParametersDataFrame> ();
  ip1->psd = i1;
  ip1->txPhy = 0;
  ip1->duration = di1;
  ip1->packetBurst = packetBursts[2];
  ip1->cellId = pbCellId[2];
  Simulator::Schedule (ti1, &LteSpectrumPhy::StartRx, ulPhy, ip1);

  Ptr<LteSpectrumSignalParametersDataFrame> ip2 = Create<LteSpectrumSignalParametersDataFrame> ();
  ip2->psd = i2;
  ip2->txPhy = 0;
  ip2->duration = di2;
  ip2->packetBurst = packetBursts[3];
  ip2->cellId = pbCellId[3];
  Simulator::Schedule (ti2, &LteSpectrumPhy::StartRx, ulPhy, ip2);

  Ptr<LteSpectrumSignalParametersDataFrame> ip3 = Create<LteSpectrumSignalParametersDataFrame> ();
  ip3->psd = i3;
  ip3->txPhy = 0;
  ip3->duration = di3;
  ip3->packetBurst = packetBursts[4];
  ip3->cellId = pbCellId[4];
  Simulator::Schedule (ti3, &LteSpectrumPhy::StartRx, ulPhy, ip3);

  Ptr<LteSpectrumSignalParametersDataFrame> ip4 = Create<LteSpectrumSignalParametersDataFrame> ();
  ip4->psd = i4;
  ip4->txPhy = 0;
  ip4->duration = di4;
  ip4->packetBurst = packetBursts[5];
  ip4->cellId = pbCellId[5];
  Simulator::Schedule (ti4, &LteSpectrumPhy::StartRx, ulPhy, ip4);

  Simulator::Stop (Seconds (5.0));
  Simulator::Run ();

  NS_LOG_INFO ("Data Frame - Theoretical SINR: " << *m_expectedSinr);
  NS_LOG_INFO ("Data Frame - Calculated SINR: " << *(actualSinrCatcher.GetValue ()));

  NS_TEST_EXPECT_MSG_NE (actualSinrCatcher.GetValue (), 0, "no actual SINR reported");

  NS_TEST_ASSERT_MSG_SPECTRUM_VALUE_EQ_TOL (*(actualSinrCatcher.GetValue ()), *m_expectedSinr, 0.0000001, "Data Frame - Wrong SINR !");
  ulPhy->Dispose ();
  Simulator::Destroy ();
  
}


/**
* TestCase Srs
*/

LteUplinkSrsSinrTestCase::LteUplinkSrsSinrTestCase (Ptr<SpectrumValue> sv1, Ptr<SpectrumValue> sv2, Ptr<SpectrumValue> sinr, std::string name)
  : TestCase ("SINR calculation in uplink srs frame: " + name),
    m_sv1 (sv1),
    m_sv2 (sv2),
    m_sm (sv1->GetSpectrumModel ()),
    m_expectedSinr (sinr)
{
  NS_LOG_INFO ("Creating LteUplinkSrsSinrTestCase");
}

LteUplinkSrsSinrTestCase::~LteUplinkSrsSinrTestCase ()
{
}

void
LteUplinkSrsSinrTestCase::ReportSinr (const SpectrumValue& sinr)
{
  m_actualSinr = sinr.Copy ();
}

void
LteUplinkSrsSinrTestCase::DoRun (void)
{
  /**
  * Instantiate a single receiving LteSpectrumPhy
  */
  
  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  // lteHelper->EnableLogComponents ();
  Ptr<LteSpectrumPhy> dlPhy = CreateObject<LteSpectrumPhy> ();
  Ptr<LteSpectrumPhy> ulPhy = CreateObject<LteSpectrumPhy> ();
  Ptr<LteTestUePhy> uePhy = CreateObject<LteTestUePhy> (dlPhy, ulPhy);
  uint16_t cellId = 100;
  dlPhy->SetCellId (cellId);
  ulPhy->SetCellId (cellId);
  
  Ptr<LteChunkProcessor> chunkProcessor = Create<LteChunkProcessor> ();
  chunkProcessor->AddCallback (MakeCallback (&LteUplinkSrsSinrTestCase::ReportSinr, this));
  ulPhy->AddCtrlSinrChunkProcessor (chunkProcessor);

  /**
  * Generate several calls to LteSpectrumPhy::StartRx corresponding to 
  * several signals.
  * One will be the signal of interest, i.e., the
  *  LteSpectrumSignalParametersUlSrsFrame of the first signal will have the
  *   same CellId of the receiving PHY; the others will have a different 
  *   CellId and hence will be the interfering signals
  */
  
  // Number of packet bursts (2 data + 4 interferences)
  int numOfDataSignals = 2;
  int numOfIntfSignals = 4;
  int numOfSignals = numOfDataSignals + numOfIntfSignals;

  uint16_t pbCellId[numOfSignals];
  
  
  
  /**
  * Build packet burst (Data and interference)
  */
  int pb = 0;
  for ( int dataPb = 0 ; dataPb < numOfDataSignals ; dataPb++, pb++ )
    {
      pbCellId[pb] = cellId;

    }
  for ( int intfPb = 0 ; intfPb < numOfIntfSignals ; intfPb++, pb++ )
    {

      pbCellId[pb] = cellId * (pb + 1);

    }
  
  
  Ptr<SpectrumValue> noisePsd = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i1 = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i2 = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i3 = Create<SpectrumValue> (m_sm);
  Ptr<SpectrumValue> i4 = Create<SpectrumValue> (m_sm);
  
  (*noisePsd)[0] = 5.000000000000e-19;
  (*noisePsd)[1] = 4.545454545455e-19;
  
  (*i1)[0] = 5.000000000000e-18;
  (*i2)[0] = 5.000000000000e-16;
  (*i3)[0] = 1.581138830084e-16;
  (*i4)[0] = 7.924465962306e-17;
  (*i1)[1] = 1.437398936440e-18;
  (*i2)[1] = 5.722388235428e-16;
  (*i3)[1] = 7.204059965732e-17;
  (*i4)[1] = 5.722388235428e-17;
  
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
  
  ulPhy->SetNoisePowerSpectralDensity (noisePsd);
  
  /**
  * Schedule the reception of the data signals plus the interference signals
  */
  
  // 2 UEs send data to the eNB through 2 subcarriers
  Ptr<LteSpectrumSignalParametersUlSrsFrame> sp1 = Create<LteSpectrumSignalParametersUlSrsFrame> ();
  sp1->psd = m_sv1;
  sp1->txPhy = 0;
  sp1->duration = ds;
  sp1->cellId = pbCellId[0];
  Simulator::Schedule (ts, &LteSpectrumPhy::StartRx, ulPhy, sp1);
  
  Ptr<LteSpectrumSignalParametersUlSrsFrame> sp2 = Create<LteSpectrumSignalParametersUlSrsFrame> ();
  sp2->psd = m_sv2;
  sp2->txPhy = 0;
  sp2->duration = ds;
  sp2->cellId = pbCellId[1];
  Simulator::Schedule (ts, &LteSpectrumPhy::StartRx, ulPhy, sp2);
  
  
  Ptr<LteSpectrumSignalParametersUlSrsFrame> ip1 = Create<LteSpectrumSignalParametersUlSrsFrame> ();
  ip1->psd = i1;
  ip1->txPhy = 0;
  ip1->duration = di1;
  ip1->cellId = pbCellId[2];
  Simulator::Schedule (ti1, &LteSpectrumPhy::StartRx, ulPhy, ip1);
  
  Ptr<LteSpectrumSignalParametersUlSrsFrame> ip2 = Create<LteSpectrumSignalParametersUlSrsFrame> ();
  ip2->psd = i2;
  ip2->txPhy = 0;
  ip2->duration = di2;
  ip2->cellId = pbCellId[3];
  Simulator::Schedule (ti2, &LteSpectrumPhy::StartRx, ulPhy, ip2);
  
  Ptr<LteSpectrumSignalParametersUlSrsFrame> ip3 = Create<LteSpectrumSignalParametersUlSrsFrame> ();
  ip3->psd = i3;
  ip3->txPhy = 0;
  ip3->duration = di3;
  ip3->cellId = pbCellId[4];
  Simulator::Schedule (ti3, &LteSpectrumPhy::StartRx, ulPhy, ip3);
  
  Ptr<LteSpectrumSignalParametersUlSrsFrame> ip4 = Create<LteSpectrumSignalParametersUlSrsFrame> ();
  ip4->psd = i4;
  ip4->txPhy = 0;
  ip4->duration = di4;
  ip4->cellId = pbCellId[5];
  Simulator::Schedule (ti4, &LteSpectrumPhy::StartRx, ulPhy, ip4);
  
  Simulator::Stop (Seconds (5.0));
  Simulator::Run ();
  
  NS_ASSERT_MSG (m_actualSinr != 0, "no actual SINR reported");
  
  NS_LOG_INFO ("SRS Frame - Theoretical SINR: " << *m_expectedSinr);
  NS_LOG_INFO ("SRS Frame - Calculated SINR: " << *m_actualSinr);

  NS_TEST_ASSERT_MSG_SPECTRUM_VALUE_EQ_TOL (*m_actualSinr, *m_expectedSinr, 0.0000001, "Data Frame - Wrong SINR !");
  ulPhy->Dispose ();
  Simulator::Destroy ();
  
}
