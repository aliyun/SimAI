/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
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
 * Author: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/basic-energy-harvester.h"
#include "ns3/basic-energy-source.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("BasicEnergyHarvesterTestSuite");

/**
 * \ingroup energy-tests
 *
 * \brief Energy Harvester Test
 */
class BasicEnergyHarvesterTestCase : public TestCase
{
public:
	BasicEnergyHarvesterTestCase ();
  ~BasicEnergyHarvesterTestCase ();

  void DoRun (void);

  double m_timeS;     //!< Time, in seconds
  double m_tolerance; //!< Tolerance for energy estimation

  ObjectFactory m_energySource; //!< Energy source factory
  ObjectFactory m_energyHarvester; //!< Energy harvester factory

};

BasicEnergyHarvesterTestCase::BasicEnergyHarvesterTestCase ()
  : TestCase ("Basic Energy Harvester test case")
{
	m_timeS = 15; // harvest energy for 15 seconds
	m_tolerance = 1.0e-13;  //
}

BasicEnergyHarvesterTestCase::~BasicEnergyHarvesterTestCase ()
{
}

void
BasicEnergyHarvesterTestCase::DoRun ()
{
	// set types
	m_energySource.SetTypeId ("ns3::BasicEnergySource");
	m_energyHarvester.SetTypeId ("ns3::BasicEnergyHarvester");
  // create node
  Ptr<Node> node = CreateObject<Node> ();

  // create Energy Source
  Ptr<BasicEnergySource> source = m_energySource.Create<BasicEnergySource> ();
  // aggregate Energy Source to the node
  node->AggregateObject (source);

  //create energy harvester
  Ptr<BasicEnergyHarvester> harvester = m_energyHarvester.Create<BasicEnergyHarvester> ();
  // Set the Energy Harvesting update interval to a value grater than the
  // simulation duration, so that the power provided by the harvester is constant
  harvester->SetHarvestedPowerUpdateInterval (Seconds (m_timeS + 1.0));
	// Connect the Basic Energy Harvester to the Energy Source
	source->ConnectEnergyHarvester (harvester);
	harvester->SetNode (node);
	harvester->SetEnergySource (source);

  Time now = Simulator::Now ();

  /*
	 * The energy harvester will recharge the energy source for m_timeS seconds.
	 */

  // Calculate remaining energy at simulation stop time
  Simulator::Schedule (Seconds (m_timeS),
                       &BasicEnergySource::UpdateEnergySource, source);

  double timeDelta = 0.000000001; // 1 nanosecond
  // run simulation; stop just after last scheduled event
  Simulator::Stop (Seconds (m_timeS + timeDelta));
  Simulator::Run ();

  // calculate energy harvested
  double estRemainingEnergy = source->GetInitialEnergy ();
  // energy = power * time
  estRemainingEnergy += harvester->GetPower () * m_timeS;

  // obtain remaining energy from source
	double remainingEnergy = source->GetRemainingEnergy ();
	NS_LOG_DEBUG ("Remaining energy is " << remainingEnergy);
	NS_LOG_DEBUG ("Estimated remaining energy is " << estRemainingEnergy);
	NS_LOG_DEBUG ("Difference is " << estRemainingEnergy - remainingEnergy);

	Simulator::Destroy ();

	// check remaining energy
  NS_TEST_ASSERT_MSG_EQ_TOL (remainingEnergy, estRemainingEnergy, m_tolerance,
                             "Incorrect Remaining energy!");

}

/**
 * \ingroup energy-tests
 *
 * \brief Energy harvester TestSuite
 */
class BasicEnergyHarvesterTestSuite : public TestSuite
{
public:
  BasicEnergyHarvesterTestSuite ();
};

BasicEnergyHarvesterTestSuite::BasicEnergyHarvesterTestSuite ()
  : TestSuite ("basic-energy-harvester", UNIT)
{
  AddTestCase (new BasicEnergyHarvesterTestCase, TestCase::QUICK);
}

/// create an instance of the test suite
static BasicEnergyHarvesterTestSuite g_basicEnergyHarvesterTestSuite;
