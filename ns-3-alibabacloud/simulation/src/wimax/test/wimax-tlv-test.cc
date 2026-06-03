/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2009 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 *
 */
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/ipcs-classifier-record.h"
#include "ns3/ptr.h"
#include "ns3/packet.h"
#include "ns3/wimax-helper.h"
#include "ns3/wimax-tlv.h"
#include "ns3/service-flow.h"
#include "ns3/cs-parameters.h"

using namespace ns3;

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Test the wimax tlv implementation.
 */
class Ns3WimaxCsParamTlvTestCase : public TestCase
{
public:
  Ns3WimaxCsParamTlvTestCase ();
  virtual ~Ns3WimaxCsParamTlvTestCase ();

private:
  virtual void DoRun (void);

};

Ns3WimaxCsParamTlvTestCase::Ns3WimaxCsParamTlvTestCase ()
  : TestCase ("Test the CS parameters tlv implementation.")
{
}

Ns3WimaxCsParamTlvTestCase::~Ns3WimaxCsParamTlvTestCase ()
{
}

void
Ns3WimaxCsParamTlvTestCase::DoRun (void)
{

  IpcsClassifierRecord classifier (Ipv4Address ("10.0.0.0"),
                                   Ipv4Mask ("255.0.0.0"),
                                   Ipv4Address ("11.0.0.0"),
                                   Ipv4Mask ("255.0.0.0"),
                                   1000,
                                   1100,
                                   3000,
                                   3100,
                                   17,
                                   1);

  classifier.AddSrcAddr (Ipv4Address ("1.0.0.0"), Ipv4Mask ("255.0.0.0"));
  classifier.AddDstAddr (Ipv4Address ("16.0.0.0"), Ipv4Mask ("255.0.0.0"));
  classifier.AddProtocol (6);
  classifier.AddSrcPortRange (1, 2);
  classifier.AddDstPortRange (4000, 4100);
  classifier.SetIndex (1);

  CsParameters csParam (CsParameters::ADD, classifier);

  SfVectorTlvValue sfVectorTlvValue;
  sfVectorTlvValue.Add (csParam.ToTlv ());

  Tlv tlvSent (145, sfVectorTlvValue.GetSerializedSize (), sfVectorTlvValue);
  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (tlvSent);

  Tlv tlvReceived;
  packet->RemoveHeader (tlvReceived);
  if (tlvReceived.GetType () == Tlv::UPLINK_SERVICE_FLOW)
    {
      SfVectorTlvValue * sfVecValue =
        (SfVectorTlvValue *) tlvReceived.PeekValue ();
      for (std::vector<Tlv*>::const_iterator iter = sfVecValue->Begin (); iter
           != sfVecValue->End (); ++iter)
        {
          if ((*iter)->GetType () == SfVectorTlvValue::IPV4_CS_Parameters)
            {
              CsParameters csParamsRecv (*(*iter));
              IpcsClassifierRecord classifier =
                csParamsRecv.GetPacketClassifierRule ();

              NS_TEST_ASSERT_MSG_EQ (!classifier.CheckMatch (Ipv4Address ("10.1.1.1"),
                                                             Ipv4Address ("16.1.1.1"),
                                                             1050,
                                                             3050,
                                                             17), false, "The classifier address did not match.");
              NS_TEST_ASSERT_MSG_EQ (!classifier.CheckMatch (Ipv4Address ("10.1.5.1"),
                                                             Ipv4Address ("11.1.1.23"),
                                                             1070,
                                                             3040,
                                                             6), false, "The classifier address did not match.");
              NS_TEST_ASSERT_MSG_EQ (classifier.CheckMatch (Ipv4Address ("11.1.1.1"),
                                                            Ipv4Address ("17.1.1.1"),
                                                            1050,
                                                            3050,
                                                            17), false, "The classifier addresses matched.");
              NS_TEST_ASSERT_MSG_EQ (classifier.CheckMatch (Ipv4Address ("10.1.1.1"),
                                                            Ipv4Address ("16.1.1.1"),
                                                            1050,
                                                            3050,
                                                            8), false, "The classifier addresses matched.");
            }
        }
    }
}

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Test the service flow tlv implementation.
 */
class Ns3WimaxSfTlvTestCase : public TestCase
{
public:
  Ns3WimaxSfTlvTestCase ();
  virtual ~Ns3WimaxSfTlvTestCase ();

private:
  virtual void DoRun (void);

};

Ns3WimaxSfTlvTestCase::Ns3WimaxSfTlvTestCase ()
  : TestCase ("Test the service flow tlv implementation.")
{
}

Ns3WimaxSfTlvTestCase::~Ns3WimaxSfTlvTestCase ()
{
}

void
Ns3WimaxSfTlvTestCase::DoRun (void)
{
  IpcsClassifierRecord classifier = IpcsClassifierRecord ();
  CsParameters csParam (CsParameters::ADD, classifier);
  ServiceFlow sf = ServiceFlow (ServiceFlow::SF_DIRECTION_DOWN);

  sf.SetSfid (100);
  sf.SetConvergenceSublayerParam (csParam);
  sf.SetCsSpecification (ServiceFlow::IPV4);
  sf.SetServiceSchedulingType (ServiceFlow::SF_TYPE_UGS);
  sf.SetMaxSustainedTrafficRate (1000000);
  sf.SetMinReservedTrafficRate (1000000);
  sf.SetMinTolerableTrafficRate (1000000);
  sf.SetMaximumLatency (10);
  sf.SetMaxTrafficBurst (1000);
  sf.SetTrafficPriority (1);

  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (sf.ToTlv ());

  Tlv tlvReceived;
  packet->RemoveHeader (tlvReceived);

  ServiceFlow sfRecv = ServiceFlow (tlvReceived);

  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetDirection (), ServiceFlow::SF_DIRECTION_DOWN, "The sfRecv had the wrong direction.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetSfid (), 100, "The sfRecv had the wrong sfid.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetCsSpecification (), ServiceFlow::IPV4, "The sfRecv had the wrong cs specification.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetServiceSchedulingType (), ServiceFlow::SF_TYPE_UGS, "The sfRecv had the wrong service scheduling type.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetMaxSustainedTrafficRate (), 1000000, "The sfRecv had the wrong maximum sustained traffic rate.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetMinReservedTrafficRate (), 1000000, "The sfRecv had the wrong minimum reserved traffic rate.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetMinTolerableTrafficRate (), 1000000, "The sfRecv had the wrong minimum tolerable traffic rate.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetMaximumLatency (), 10, "The sfRecv had the wrong maximum latency.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetMaxTrafficBurst (), 1000, "The sfRecv had the wrong maximum traffic burst.");
  NS_TEST_ASSERT_MSG_EQ (sfRecv.GetTrafficPriority (), 1, "The sfRecv had the wrong traffic priority.");
}

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Ns3 Wimax Tlv Test Suite
 */
class Ns3WimaxTlvTestSuite : public TestSuite
{
public:
  Ns3WimaxTlvTestSuite ();
};

Ns3WimaxTlvTestSuite::Ns3WimaxTlvTestSuite ()
  : TestSuite ("wimax-tlv", UNIT)
{
  AddTestCase (new Ns3WimaxCsParamTlvTestCase, TestCase::QUICK);
  AddTestCase (new Ns3WimaxSfTlvTestCase, TestCase::QUICK);
}

static Ns3WimaxTlvTestSuite ns3WimaxTlvTestSuite; ///< the test suite
