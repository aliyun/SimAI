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
 */

#include "ns3/simulator.h"
#include "ns3/log.h"

#include "ns3/lte-rlc-header.h"
#include "ns3/lte-rlc-um.h"

#include "lte-test-rlc-um-transmitter.h"
#include "lte-test-entities.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteRlcUmTransmitterTest");

/**
 * TestSuite 4.1.1 RLC UM: Only transmitter
 */

LteRlcUmTransmitterTestSuite::LteRlcUmTransmitterTestSuite ()
  : TestSuite ("lte-rlc-um-transmitter", SYSTEM)
{
  // LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);
  // LogComponentEnable ("LteRlcUmTransmitterTest", logLevel);

  // NS_LOG_INFO ("Creating LteRlcUmTransmitterTestSuite");

  AddTestCase (new LteRlcUmTransmitterOneSduTestCase ("One SDU, one PDU"), TestCase::QUICK);
  AddTestCase (new LteRlcUmTransmitterSegmentationTestCase ("Segmentation"), TestCase::QUICK);
  AddTestCase (new LteRlcUmTransmitterConcatenationTestCase ("Concatenation"), TestCase::QUICK);
  AddTestCase (new LteRlcUmTransmitterReportBufferStatusTestCase ("ReportBufferStatus primitive"), TestCase::QUICK);

}

static LteRlcUmTransmitterTestSuite lteRlcUmTransmitterTestSuite;


LteRlcUmTransmitterTestCase::LteRlcUmTransmitterTestCase (std::string name)
  : TestCase (name)
{
  // NS_LOG_UNCOND ("Creating LteRlcUmTransmitterTestCase: " + name);
}

LteRlcUmTransmitterTestCase::~LteRlcUmTransmitterTestCase ()
{
}

void
LteRlcUmTransmitterTestCase::DoRun (void)
{
  // LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);
  // LogComponentEnable ("LteRlcUmTransmitterTest", logLevel);
  // LogComponentEnable ("LteTestEntities", logLevel);
  // LogComponentEnable ("LteRlc", logLevel);
  // LogComponentEnable ("LteRlcUm", logLevel);
  // LogComponentEnable ("LteRlcHeader", logLevel);

  uint16_t rnti = 1111;
  uint8_t lcid = 222;

  Packet::EnablePrinting ();

  // Create topology

  // Create transmission PDCP test entity
  txPdcp = CreateObject<LteTestPdcp> ();

  // Create transmission RLC entity
  txRlc = CreateObject<LteRlcUm> ();
  txRlc->SetRnti (rnti);
  txRlc->SetLcId (lcid);

  // Create transmission MAC test entity
  txMac = CreateObject<LteTestMac> ();
  txMac->SetRlcHeaderType (LteTestMac::UM_RLC_HEADER);

  // Connect SAPs: PDCP (TX) <-> RLC (Tx) <-> MAC (Tx)
  txPdcp->SetLteRlcSapProvider (txRlc->GetLteRlcSapProvider ());
  txRlc->SetLteRlcSapUser (txPdcp->GetLteRlcSapUser ());

  txRlc->SetLteMacSapProvider (txMac->GetLteMacSapProvider ());
  txMac->SetLteMacSapUser (txRlc->GetLteMacSapUser ());

}

void
LteRlcUmTransmitterTestCase::CheckDataReceived (Time time, std::string shouldReceived, std::string assertMsg)
{
  Simulator::Schedule (time, &LteRlcUmTransmitterTestCase::DoCheckDataReceived, this, shouldReceived, assertMsg);
}

void
LteRlcUmTransmitterTestCase::DoCheckDataReceived (std::string shouldReceived, std::string assertMsg)
{
  NS_TEST_ASSERT_MSG_EQ (shouldReceived, txMac->GetDataReceived (), assertMsg);
}


/**
 * Test 4.1.1.1 One SDU, One PDU
 */
LteRlcUmTransmitterOneSduTestCase::LteRlcUmTransmitterOneSduTestCase (std::string name)
  : LteRlcUmTransmitterTestCase (name)
{
}

LteRlcUmTransmitterOneSduTestCase::~LteRlcUmTransmitterOneSduTestCase ()
{
}

void
LteRlcUmTransmitterOneSduTestCase::DoRun (void)
{
  // Create topology
  LteRlcUmTransmitterTestCase::DoRun ();

  //
  // a) One SDU generates one PDU
  //

  // PDCP entity sends data
  txPdcp->SendData (Seconds (0.100), "ABCDEFGHIJKLMNOPQRSTUVWXYZ");

  // MAC entity sends TxOpp to RLC entity
  txMac->SendTxOpportunity (Seconds (0.150), 28);
  CheckDataReceived (Seconds (0.200), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "SDU is not OK");

  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * Test 4.1.1.2 Segmentation (One SDU => n PDUs)
 */
LteRlcUmTransmitterSegmentationTestCase::LteRlcUmTransmitterSegmentationTestCase (std::string name)
  : LteRlcUmTransmitterTestCase (name)
{
}

LteRlcUmTransmitterSegmentationTestCase::~LteRlcUmTransmitterSegmentationTestCase ()
{
}

void
LteRlcUmTransmitterSegmentationTestCase::DoRun (void)
{
  // Create topology
  LteRlcUmTransmitterTestCase::DoRun ();

  //
  // b) Segmentation: one SDU generates n PDUs
  //

  // PDCP entity sends data
  txPdcp->SendData (Seconds (0.100), "ABCDEFGHIJKLMNOPQRSTUVWXYZ");

  // MAC entity sends small TxOpp to RLC entity generating four segments
  txMac->SendTxOpportunity (Seconds (0.150), 10);
  CheckDataReceived (Seconds (0.200), "ABCDEFGH", "Segment #1 is not OK");

  txMac->SendTxOpportunity (Seconds (0.200), 10);
  CheckDataReceived (Seconds (0.250), "IJKLMNOP", "Segment #2 is not OK");

  txMac->SendTxOpportunity (Seconds (0.300), 10);
  CheckDataReceived (Seconds (0.350), "QRSTUVWX", "Segment #3 is not OK");

  txMac->SendTxOpportunity (Seconds (0.400), 4);
  CheckDataReceived (Seconds (0.450), "YZ", "Segment #4 is not OK");

  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * Test 4.1.1.3 Concatenation (n SDUs => One PDU)
 */
LteRlcUmTransmitterConcatenationTestCase::LteRlcUmTransmitterConcatenationTestCase (std::string name)
  : LteRlcUmTransmitterTestCase (name)
{
}

LteRlcUmTransmitterConcatenationTestCase::~LteRlcUmTransmitterConcatenationTestCase ()
{
}

void
LteRlcUmTransmitterConcatenationTestCase::DoRun (void)
{
  // Create topology
  LteRlcUmTransmitterTestCase::DoRun ();

  //
  // c) Concatenation: n SDUs generate one PDU
  //

  // PDCP entity sends three data packets
  txPdcp->SendData (Seconds (0.100), "ABCDEFGH");
  txPdcp->SendData (Seconds (0.150), "IJKLMNOPQR");
  txPdcp->SendData (Seconds (0.200), "STUVWXYZ");

  // MAC entity sends TxOpp to RLC entity generating only one concatenated PDU
  txMac->SendTxOpportunity (Seconds (0.250), 31);
  CheckDataReceived (Seconds (0.300), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Concatenation is not OK");

  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * Test 4.1.1.4 Report Buffer Status (test primitive parameters)
 */
LteRlcUmTransmitterReportBufferStatusTestCase::LteRlcUmTransmitterReportBufferStatusTestCase (std::string name)
  : LteRlcUmTransmitterTestCase (name)
{
}

LteRlcUmTransmitterReportBufferStatusTestCase::~LteRlcUmTransmitterReportBufferStatusTestCase ()
{
}

void
LteRlcUmTransmitterReportBufferStatusTestCase::DoRun (void)
{
  // Create topology
  LteRlcUmTransmitterTestCase::DoRun ();

  //
  // d) Test the parameters of the ReportBufferStatus primitive
  //

  // PDCP entity sends data
  txPdcp->SendData (Seconds (0.100), "ABCDEFGHIJ"); // 10
  txPdcp->SendData (Seconds (0.150), "KLMNOPQRS");  // 9
  txPdcp->SendData (Seconds (0.200), "TUVWXYZ");    // 7

  txMac->SendTxOpportunity (Seconds (0.250), (2+2) + (10+6));
  CheckDataReceived (Seconds (0.300), "ABCDEFGHIJKLMNOP", "SDU is not OK");

  txPdcp->SendData (Seconds (0.350), "ABCDEFGH");     // 8
  txPdcp->SendData (Seconds (0.400), "IJKLMNOPQRST"); // 12
  txPdcp->SendData (Seconds (0.450), "UVWXYZ");       // 6

  txMac->SendTxOpportunity (Seconds (0.500), 2 + 3);
  CheckDataReceived (Seconds (0.550), "QRS", "SDU is not OK");

  txPdcp->SendData (Seconds (0.600), "ABCDEFGH");     // 8
  txPdcp->SendData (Seconds (0.650), "IJKLMNOPQRST"); // 12
  txPdcp->SendData (Seconds (0.700), "UVWXYZ");       // 6

  txPdcp->SendData (Seconds (0.750), "ABCDEFGHIJ");   // 10
  txPdcp->SendData (Seconds (0.800), "KLMNOPQRST");   // 10
  txPdcp->SendData (Seconds (0.850), "UVWXYZ");       // 6

  txMac->SendTxOpportunity (Seconds (0.900), 2 + 7);
  CheckDataReceived (Seconds (0.950), "TUVWXYZ", "SDU is not OK");

  txMac->SendTxOpportunity (Seconds (1.000), (2+2) + (8+2));
  CheckDataReceived (Seconds (1.050), "ABCDEFGHIJ", "SDU is not OK");

  txPdcp->SendData (Seconds (1.100), "ABCDEFGHIJ");   // 10
  txPdcp->SendData (Seconds (1.150), "KLMNOPQRST");   // 10
  txPdcp->SendData (Seconds (1.200), "UVWXYZ");       // 6

  txMac->SendTxOpportunity (Seconds (1.250), 2 + 2);
  CheckDataReceived (Seconds (1.300), "KL", "SDU is not OK");

  txMac->SendTxOpportunity (Seconds (1.350), 2 + 3);
  CheckDataReceived (Seconds (1.400), "MNO", "SDU is not OK");

  txMac->SendTxOpportunity (Seconds (1.450), 2 + 5);
  CheckDataReceived (Seconds (1.500), "PQRST", "SDU is not OK");

  txMac->SendTxOpportunity (Seconds (1.550), (2+2+1+2+1+2+1) + (6+8+12+6+10+10+3));
  CheckDataReceived (Seconds (1.600), "UVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVW", "SDU is not OK");

  txMac->SendTxOpportunity (Seconds (1.650), (2+2+1+2) + (3+10+10+6));
  CheckDataReceived (Seconds (1.700), "XYZABCDEFGHIJKLMNOPQRSTUVWXYZ", "SDU is not OK");

  Simulator::Run ();
  Simulator::Destroy ();
}
