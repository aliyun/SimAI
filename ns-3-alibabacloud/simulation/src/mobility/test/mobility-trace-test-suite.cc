/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington
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
 * Author: Mitch Watrous (watrous@u.washington.edu)
 */

#include <string>

#include "ns3/ascii-test.h"
#include "ns3/trace-helper.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/simulator.h"
#include "ns3/rectangle.h"
#include "ns3/mobility-model.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/test.h"

using namespace ns3;


/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Mobility Trace Test Case
 */

class MobilityTraceTestCase : public TestCase
{
public:
  MobilityTraceTestCase ();
  virtual ~MobilityTraceTestCase ();

private:
  virtual void DoRun (void);
};

MobilityTraceTestCase::MobilityTraceTestCase ()
  :
  TestCase ("Mobility Trace Test Case")
{
}

MobilityTraceTestCase::~MobilityTraceTestCase ()
{
}

void
MobilityTraceTestCase::DoRun (void)
{
  //***************************************************************************
  // Create the new mobility trace.
  //***************************************************************************

  NodeContainer sta;
  sta.Create (4);
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (1.0),
                                 "MinY", DoubleValue (1.0),
                                 "DeltaX", DoubleValue (5.0),
                                 "DeltaY", DoubleValue (5.0),
                                 "GridWidth", UintegerValue (3),
                                 "LayoutType", StringValue ("RowFirst"));
  mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                             "Mode", StringValue ("Time"),
                             "Time", StringValue ("2s"),
                             "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=1.0]"),
                             "Bounds", RectangleValue (Rectangle (0.0, 20.0, 0.0, 20.0)));
  mobility.Install (sta);
  // Set mobility random number streams to fixed values
  mobility.AssignStreams (sta, 0);

  SetDataDir (NS_TEST_SOURCEDIR);
  std::string referenceMobilityFilePath = CreateDataDirFilename ("mobility-trace-example.mob");
  std::string testMobilityFilePath      = CreateTempDirFilename ("mobility-trace-test.mob");

  AsciiTraceHelper ascii;
  MobilityHelper::EnableAsciiAll (ascii.CreateFileStream (testMobilityFilePath));
  Simulator::Stop (Seconds (5.0));
  Simulator::Run ();
  Simulator::Destroy ();


  //***************************************************************************
  // Test the new mobility trace against the reference mobility trace.
  //***************************************************************************

  NS_ASCII_TEST_EXPECT_EQ (testMobilityFilePath, referenceMobilityFilePath);
}


/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Mobility Trace Test Suite
 */

class MobilityTraceTestSuite : public TestSuite
{
public:
  MobilityTraceTestSuite ();
};

MobilityTraceTestSuite::MobilityTraceTestSuite ()
  : TestSuite ("mobility-trace", UNIT)
{
  AddTestCase (new MobilityTraceTestCase, TestCase::QUICK);
}

static MobilityTraceTestSuite mobilityTraceTestSuite; ///< the test suite
