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
 *
 */
#include "ns3/test.h"
#include "ns3/service-flow.h"
#include "ns3/mac-messages.h"

using namespace ns3;

/**
 * \ingroup wimax
 * \defgroup wimax-test wimax module tests
 */


/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Test the DSA request message.
 */
class DsaRequestTestCase : public TestCase
{
public:
  DsaRequestTestCase ();
  virtual ~DsaRequestTestCase ();

private:
  virtual void DoRun (void);

};

DsaRequestTestCase::DsaRequestTestCase ()
  : TestCase ("Test the DSA request messages")
{
}

DsaRequestTestCase::~DsaRequestTestCase ()
{
}

void
DsaRequestTestCase::DoRun (void)
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

  DsaReq dsaReq (sf);
  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (dsaReq);

  DsaReq dsaReqRecv;
  packet->RemoveHeader (dsaReqRecv);

  ServiceFlow sfRecv = dsaReqRecv.GetServiceFlow ();

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
 * \brief Ns3 Wimax Mac Messages Test Suite
 */
class Ns3WimaxMacMessagesTestSuite : public TestSuite
{
public:
  Ns3WimaxMacMessagesTestSuite ();
};

Ns3WimaxMacMessagesTestSuite::Ns3WimaxMacMessagesTestSuite ()
  : TestSuite ("wimax-mac-messages", UNIT)
{
  AddTestCase (new DsaRequestTestCase, TestCase::QUICK);
}

static Ns3WimaxMacMessagesTestSuite ns3WimaxMacMessagesTestSuite; ///< the test suite
