/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Hajime Tazaki
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
 * Author: Hajime Tazaki (tazaki@sfc.wide.ad.jp)
 */

//-----------------------------------------------------------------------------
// Unit tests
//-----------------------------------------------------------------------------

#include "ns3/test.h"
#include "ns3/rocketfuel-topology-reader.h"
#include "ns3/node-container.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/attribute.h"
#include "ns3/object-factory.h"
#include "ns3/simulator.h"

using namespace ns3;
/**
 * \ingroup topology
 * \defgroup topology-test Topology module tests
 */

/**
 * \file
 * \ingroup topology-test
 * ns3::RockefuelTopologyReader test suite.
 */

/**
 * \ingroup topology-test
 * \ingroup tests

 * \brief Rocketfuel Topology Reader Test
 */
class RocketfuelTopologyReaderTest : public TestCase
{
public:
  RocketfuelTopologyReaderTest ();
private:
  virtual void DoRun (void);
};

RocketfuelTopologyReaderTest::RocketfuelTopologyReaderTest ()
  : TestCase ("RocketfuelTopologyReaderTest")
{
}


void
RocketfuelTopologyReaderTest::DoRun (void)
{
  Ptr<RocketfuelTopologyReader> inFile;
  NodeContainer nodes;

  std::string input ("./src/topology-read/examples/RocketFuel_toposample_1239_weights.txt");

  inFile = CreateObject<RocketfuelTopologyReader> ();
  inFile->SetFileName (input);

  if (inFile != 0)
    {
      nodes = inFile->Read ();
    }

  NS_TEST_ASSERT_MSG_NE (nodes.GetN (), 0, "Problems reading node information the topology file..");

  NS_TEST_ASSERT_MSG_NE (inFile->LinksSize (), 0, "Problems reading the topology file.");

  NS_TEST_EXPECT_MSG_EQ (nodes.GetN (),315, "noes");
  NS_TEST_EXPECT_MSG_EQ (inFile->LinksSize (),972, "links");
  Simulator::Destroy ();
}

/**
 * \ingroup topology-test
 * \ingroup tests
 *
 * \brief Rocketfuel Topology Reader TestSuite
 */
class RocketfuelTopologyReaderTestSuite : public TestSuite
{
public:
  RocketfuelTopologyReaderTestSuite ();
private:
};

RocketfuelTopologyReaderTestSuite::RocketfuelTopologyReaderTestSuite ()
  : TestSuite ("rocketfuel-topology-reader", UNIT)
{
  AddTestCase (new RocketfuelTopologyReaderTest (), TestCase::QUICK);
}

static RocketfuelTopologyReaderTestSuite g_rocketfuelTopologyReaderTestSuite; //!< Static variable for test initialization
