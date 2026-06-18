/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 *         Nicola Baldo <nbaldo@cttc.es>
 */

#include "ns3/config.h"
#include "ns3/simulator.h"
#include "ns3/pointer.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/error-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/radio-bearer-stats-calculator.h"
#include "ns3/lte-rlc-header.h"
#include "ns3/lte-rlc-um.h"
#include "ns3/config-store.h"

#include "lte-test-rlc-am-e2e.h"
#include "lte-simple-helper.h"
#include "lte-test-entities.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteRlcAmE2eTest");

LteRlcAmE2eTestSuite::LteRlcAmE2eTestSuite ()
  : TestSuite ("lte-rlc-am-e2e", SYSTEM)
{
  // NS_LOG_INFO ("Creating LteRlcAmE2eTestSuite");

  double losses[] = {0.0, 0.05,  0.10, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95};
  uint32_t runs[] = {1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 11110,
                     12221, 13332, 14443, 15554, 16665, 17776, 18887, 19998, 21109, 22220,
                     23331, 24442, 25553, 26664, 27775, 28886, 29997, 31108, 32219, 33330};

  for ( uint32_t l = 0; l < (sizeof (losses) / sizeof (double)); l++ )
    {
      for ( uint32_t s = 0; s < (sizeof (runs) / sizeof (uint32_t)); s++ )
        {
          for (uint32_t sduArrivalType = 0; sduArrivalType <= 1; ++sduArrivalType)
            {
              std::ostringstream name;
              name << " losses = " << losses[l] * 100 << "%; run = " << runs[s];

              bool bulkSduArrival;
              switch (sduArrivalType)
                {
                  case 0:
                    bulkSduArrival = false;
                    name << "; continuous SDU arrival";
                    break;
                  case 1:
                    bulkSduArrival = true;
                    name << "; bulk SDU arrival";
                    break;
                  default:
                    NS_FATAL_ERROR ("unsupported option");
                    break;
                }

              TestCase::TestDuration testDuration;
              if (l == 1 && s == 0)
                {
                  testDuration = TestCase::QUICK;
                }
              else if (s <= 4)
                {
                  testDuration = TestCase::EXTENSIVE;
                }
              else
                {
                  testDuration = TestCase::TAKES_FOREVER;
                }
              AddTestCase (new LteRlcAmE2eTestCase (name.str (), runs[s], losses[l], bulkSduArrival), testDuration);
            }
        }
    }
}

static LteRlcAmE2eTestSuite lteRlcAmE2eTestSuite;

LteRlcAmE2eTestCase::LteRlcAmE2eTestCase (std::string name, uint32_t run, double losses, bool bulkSduArrival)
  : TestCase (name),
    m_run (run),
    m_losses (losses),
    m_bulkSduArrival (bulkSduArrival),
    m_dlDrops (0),
    m_ulDrops (0)
{
  NS_LOG_INFO ("Creating LteRlcAmTestingTestCase: " + name);
}

LteRlcAmE2eTestCase::~LteRlcAmE2eTestCase ()
{}


void
LteRlcAmE2eTestCase::DlDropEvent (Ptr<const Packet> p)
{
  // NS_LOG_FUNCTION (this);
  m_dlDrops++;
}

void
LteRlcAmE2eTestCase::UlDropEvent (Ptr<const Packet> p)
{
  // NS_LOG_FUNCTION (this);
  m_ulDrops++;
}

void
LteRlcAmE2eTestCase::DoRun (void)
{
  uint16_t numberOfNodes = 1;

  // LogLevel level = (LogLevel) (LOG_LEVEL_ALL | LOG_PREFIX_TIME | LOG_PREFIX_NODE | LOG_PREFIX_FUNC);
  // LogComponentEnable ("LteRlcAmE2eTest", level);
  // LogComponentEnable ("ErrorModel", level);
  // LogComponentEnable ("LteSimpleHelper", level);
  // LogComponentEnable ("LteSimpleNetDevice", level);
  // LogComponentEnable ("SimpleNetDevice", level);
  // LogComponentEnable ("SimpleChannel", level);
  // LogComponentEnable ("LteTestEntities", level);
  // LogComponentEnable ("LtePdcp", level);
  // LogComponentEnable ("LteRlc", level);
  // LogComponentEnable ("LteRlcUm", level);
  // LogComponentEnable ("LteRlcAm", level);

  Config::SetGlobal ("RngRun", UintegerValue (m_run));
  Config::SetDefault ("ns3::LteRlcAm::PollRetransmitTimer", TimeValue (MilliSeconds (20)));
  Config::SetDefault ("ns3::LteRlcAm::ReorderingTimer", TimeValue (MilliSeconds (10)));
  Config::SetDefault ("ns3::LteRlcAm::StatusProhibitTimer", TimeValue (MilliSeconds (40)));
  // This test was written for an unlimited transmit buffer (special value of 0)
  Config::SetDefault ("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue (0));

  Ptr<LteSimpleHelper> lteSimpleHelper = CreateObject<LteSimpleHelper> ();
  // lteSimpleHelper->EnableLogComponents ();
  // lteSimpleHelper->EnableTraces ();

  lteSimpleHelper->SetAttribute ("RlcEntity", StringValue ("RlcAm"));

  // eNB and UE nodes
  NodeContainer ueNodes;
  NodeContainer enbNodes;
  enbNodes.Create (numberOfNodes);
  ueNodes.Create (numberOfNodes);

  // Install LTE Devices to the nodes
  NetDeviceContainer enbLteDevs = lteSimpleHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueLteDevs = lteSimpleHelper->InstallUeDevice (ueNodes);

  // Note: Just one eNB and UE is supported. Everything is done in InstallEnbDevice and InstallUeDevice

  // Attach one UE per eNodeB
  // for (uint16_t i = 0; i < numberOfNodes; i++)
  //   {
  //     lteSimpleHelper->Attach (ueLteDevs.Get(i), enbLteDevs.Get(i));
  //   }

  //   lteSimpleHelper->ActivateEpsBearer (ueLteDevs, EpsBearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT), EpcTft::Default ());


  // Error models: downlink and uplink
  Ptr<RateErrorModel> dlEm = CreateObject<RateErrorModel> ();
  // fix the stream so that subsequent test cases get a number from the same stream
  // if RngRun is different, the number shall then be different
  dlEm->AssignStreams (3);
  dlEm->SetAttribute ("ErrorRate", DoubleValue (m_losses));
  dlEm->SetAttribute ("ErrorUnit", StringValue ("ERROR_UNIT_PACKET"));

//   Ptr<RateErrorModel> ueEm = CreateObjectWithAttributes<RateErrorModel> ("RanVar", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1.0]"));
//   ueEm->SetAttribute ("ErrorRate", DoubleValue (m_losses));
//   ueEm->SetAttribute ("ErrorUnit", StringValue ("ERROR_UNIT_PACKET"));

  // The below hooks will cause drops and receptions to be counted
  ueLteDevs.Get (0)->SetAttribute ("ReceiveErrorModel", PointerValue (dlEm));
  ueLteDevs.Get (0)->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&LteRlcAmE2eTestCase::DlDropEvent, this));
//   enbLteDevs.Get (0)->SetAttribute ("ReceiveErrorModel", PointerValue (enbEm));
//   enbLteDevs.Get (0)->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&LteRlcAmE2eTestCase::EnbDropEvent, this));



  uint32_t sduSizeBytes = 100;
  uint32_t numSdu = 1000;
  double sduStartTimeSeconds = 0.100;
  double sduStopTimeSeconds;
  double sduArrivalTimeSeconds;
  uint32_t dlTxOppSizeBytes = 150;
  double dlTxOpprTimeSeconds = 0.003;
  uint32_t ulTxOppSizeBytes = 140;
  double ulTxOpprTimeSeconds = 0.003;

  if (m_bulkSduArrival)
    {
      sduStopTimeSeconds = sduStartTimeSeconds + 0.010;
    }
  else
    {
      sduStopTimeSeconds = sduStartTimeSeconds + 10;
    }
  sduArrivalTimeSeconds = (sduStopTimeSeconds - sduStartTimeSeconds) / numSdu;



  // Sending packets from RRC layer
  lteSimpleHelper->m_enbRrc->SetArrivalTime (Seconds (sduArrivalTimeSeconds));
  lteSimpleHelper->m_enbRrc->SetPduSize (sduSizeBytes);

  // MAC sends transmission opportunities (TxOpp)
  lteSimpleHelper->m_enbMac->SetTxOppSize (dlTxOppSizeBytes);
  lteSimpleHelper->m_enbMac->SetTxOppTime (Seconds (dlTxOpprTimeSeconds));
  lteSimpleHelper->m_enbMac->SetTxOpportunityMode (LteTestMac::AUTOMATIC_MODE);

  // MAC sends transmission opportunities (TxOpp)
  lteSimpleHelper->m_ueMac->SetTxOppSize (ulTxOppSizeBytes);
  lteSimpleHelper->m_ueMac->SetTxOppTime (Seconds (ulTxOpprTimeSeconds));
  lteSimpleHelper->m_ueMac->SetTxOpportunityMode (LteTestMac::AUTOMATIC_MODE);

  // Start/Stop pseudo-application at RRC layer
  Simulator::Schedule (Seconds (sduStartTimeSeconds), &LteTestRrc::Start, lteSimpleHelper->m_enbRrc);
  Simulator::Schedule (Seconds (sduStopTimeSeconds), &LteTestRrc::Stop, lteSimpleHelper->m_enbRrc);


  double maxDlThroughput = (dlTxOppSizeBytes / (dlTxOppSizeBytes + 4.0)) * (dlTxOppSizeBytes / dlTxOpprTimeSeconds) * (1.0 - m_losses);
  const double statusProhibitSeconds = 0.020;
  double pollFrequency = (1.0 / dlTxOpprTimeSeconds) * (1 - m_losses);
  double statusFrequency = std::min (pollFrequency, 1.0 / statusProhibitSeconds);
  const uint32_t numNackSnPerStatusPdu = (ulTxOppSizeBytes * 8 - 14) / 10;
  double maxRetxThroughput = ((double)numNackSnPerStatusPdu * (double)dlTxOppSizeBytes) * statusFrequency;
  double throughput = std::min (maxDlThroughput, maxRetxThroughput);
  double totBytes = ((sduSizeBytes) * (sduStopTimeSeconds - sduStartTimeSeconds) / sduArrivalTimeSeconds);


  // note: the throughput estimation is valid only for the full buffer
  // case. However, the test sends a finite number of SDUs. Hence, the
  // estimated throughput will only be effective at the beginning of
  // the test. Towards the end of the test, two issues are present:
  //   1) no new data is transmitted, hence less feedback is sent,
  //      hence the transmission rate for the last PDUs to be
  //      retransmitted is much lower. This effect can be best noteed
  //      at very high loss rates, and can be adjusted by timers and
  //      params.
  //   2) throuhgput is not meaningful, you need to evaluate the time
  //      it takes for all PDUs to be (re)transmitted successfully,
  //      i.e., how long it takes for the TX and reTX queues to deplete.

  // Estimating correctly this effect would require a complex stateful
  // model (e.g., a Markov chain model) so to avoid the hassle we just
  // use a margin here which we empirically determine as something we
  // think reasonable based on the PDU loss rate
  Time margin;
  if (m_losses < 0.07)
    {
      margin = Seconds (0.500);
    }
  else if (m_losses < 0.20)
    {
      margin = Seconds (1);
    }
  else if (m_losses < 0.50)
    {
      margin = Seconds (2);
    }
  else if (m_losses < 0.70)
    {
      margin = Seconds (10);
    }
  else if (m_losses < 0.91)
    {
      margin = Seconds (20);
    }
  else // 0.95
    {
      margin = Seconds (30);
    }
  Time stopTime = Seconds (std::max (sduStartTimeSeconds + totBytes / throughput, sduStopTimeSeconds)) + margin;

  NS_LOG_INFO ("statusFrequency=" << statusFrequency << ", maxDlThroughput=" << maxDlThroughput << ", maxRetxThroughput=" << maxRetxThroughput << ", totBytes=" << totBytes << ", stopTime=" << stopTime.As (Time::S));

  Simulator::Stop (stopTime);
  Simulator::Run ();

  uint32_t txEnbRrcPdus = lteSimpleHelper->m_enbRrc->GetTxPdus ();
  uint32_t rxUeRrcPdus = lteSimpleHelper->m_ueRrc->GetRxPdus ();

  uint32_t txEnbRlcPdus = lteSimpleHelper->m_enbMac->GetTxPdus ();
  uint32_t rxUeRlcPdus = lteSimpleHelper->m_ueMac->GetRxPdus ();

  NS_LOG_INFO ("Run = " << m_run);
  NS_LOG_INFO ("Loss rate (%) = " << uint32_t (m_losses * 100));


  NS_LOG_INFO ("RLC PDUs   TX: " << txEnbRlcPdus
                                 << "   RX: " << rxUeRlcPdus
                                 << "   LOST: " << m_dlDrops
                                 << " (" << (100.0 * (double) m_dlDrops) / txEnbRlcPdus << "%)");

  NS_TEST_ASSERT_MSG_EQ (txEnbRlcPdus,  rxUeRlcPdus + m_dlDrops, "lost RLC PDUs don't match TX + RX");

  NS_LOG_INFO ("eNB tx RRC count = " << txEnbRrcPdus);
  NS_LOG_INFO ("UE rx RRC count = " << rxUeRrcPdus);


  NS_TEST_ASSERT_MSG_EQ (txEnbRrcPdus, rxUeRrcPdus,
                         "TX PDUs (" << txEnbRrcPdus << ") != RX PDUs (" << rxUeRrcPdus << ")");

  Simulator::Destroy ();
}
