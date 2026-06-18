/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Andrea Sacco
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
 * Author: Andrea Sacco <andrea.sacco85@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simple-device-energy-model.h"
#include "ns3/li-ion-energy-source.h"
#include "ns3/node.h"
#include "ns3/simulator.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LiIonEnergySourceTestSuite");


/**
 * \ingroup energy-tests
 *
 * \brief LiIon battery Test
 */
class LiIonEnergyTestCase : public TestCase
{
public:
  LiIonEnergyTestCase ();
  ~LiIonEnergyTestCase ();

  void DoRun (void);

  Ptr<Node> m_node; //!< Node to aggreagte the source to.
};

LiIonEnergyTestCase::LiIonEnergyTestCase ()
  : TestCase ("Li-Ion energy source test case")
{
}

LiIonEnergyTestCase::~LiIonEnergyTestCase ()
{
  m_node = 0;
}

void
LiIonEnergyTestCase::DoRun ()
{
  m_node = CreateObject<Node> ();

  Ptr<SimpleDeviceEnergyModel> sem = CreateObject<SimpleDeviceEnergyModel> ();
  Ptr<LiIonEnergySource> es = CreateObject<LiIonEnergySource> ();

  es->SetNode (m_node);
  sem->SetEnergySource (es);
  es->AppendDeviceEnergyModel (sem);
  m_node->AggregateObject (es);

  Time now = Simulator::Now ();

  // discharge at 2.33 A for 1700 seconds
  sem->SetCurrentA (2.33);
  now += Seconds (1701);

  Simulator::Stop (now);
  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ_TOL (es->GetSupplyVoltage (), 3.6, 1.0e-3,
                             "Incorrect consumed energy!");
}

/**
 * \ingroup energy-tests
 *
 * \brief LiIon battery TestSuite
 */
class LiIonEnergySourceTestSuite : public TestSuite
{
public:
  LiIonEnergySourceTestSuite ();
};

LiIonEnergySourceTestSuite::LiIonEnergySourceTestSuite ()
  : TestSuite ("li-ion-energy-source", UNIT)
{
  AddTestCase (new LiIonEnergyTestCase, TestCase::QUICK);
}

/// create an instance of the test suite
static LiIonEnergySourceTestSuite g_liIonEnergySourceTestSuite;
