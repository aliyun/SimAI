/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: John Abraham <john.abraham@gatech.edu>
 * Contributions: Eugene Kalishenko <ydginster@gmail.com> (Open Source and Linux Laboratory http://dev.osll.ru/)
 */

#include <iostream>
#include "unistd.h"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/basic-energy-source.h"
#include "ns3/simple-device-energy-model.h"

using namespace ns3;

/**
 * \ingroup netanim
 * \defgroup netanim-test animation module tests
 */


/**
 * \ingroup netanim-test
 * \ingroup tests
 *
 * \brief Abstract Animation Interface Test Case
 */
class AbstractAnimationInterfaceTestCase : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param name testcase name
   */
  AbstractAnimationInterfaceTestCase (std::string name);
  /**
   * \brief Destructor.
   */
  virtual
  ~AbstractAnimationInterfaceTestCase ();
  virtual void
  DoRun (void);

protected:

  NodeContainer m_nodes; ///< the nodes
  AnimationInterface* m_anim; ///< animation

private:

  /// Prepare network function
  virtual void PrepareNetwork () = 0;

  /// Check logic function
  virtual void CheckLogic () = 0;

  /// Check file existence
  virtual void CheckFileExistence ();

  const char* m_traceFileName; ///< trace file name
};

AbstractAnimationInterfaceTestCase::AbstractAnimationInterfaceTestCase (std::string name) :
  TestCase (name), m_anim (NULL), m_traceFileName ("netanim-test.xml")
{
}

AbstractAnimationInterfaceTestCase::~AbstractAnimationInterfaceTestCase ()
{
  delete m_anim;
}

void
AbstractAnimationInterfaceTestCase::DoRun (void)
{
  PrepareNetwork ();

  m_anim = new AnimationInterface (m_traceFileName);

  Simulator::Run ();
  CheckLogic ();
  CheckFileExistence ();
  Simulator::Destroy ();
}

void
AbstractAnimationInterfaceTestCase::CheckFileExistence ()
{
  FILE * fp = fopen (m_traceFileName, "r");
  NS_TEST_ASSERT_MSG_NE (fp, 0, "Trace file was not created");
  fclose (fp);
  unlink (m_traceFileName);
}

/**
 * \ingroup netanim-test
 * \ingroup tests
 *
 * \brief Animation Interface Test Case
 */
class AnimationInterfaceTestCase : public AbstractAnimationInterfaceTestCase
{
public:
  /**
   * \brief Constructor.
   */
  AnimationInterfaceTestCase ();

private:

  virtual void
  PrepareNetwork ();

  virtual void
  CheckLogic ();

};

AnimationInterfaceTestCase::AnimationInterfaceTestCase () :
  AbstractAnimationInterfaceTestCase ("Verify AnimationInterface")
{
}

void
AnimationInterfaceTestCase::PrepareNetwork (void)
{
  m_nodes.Create (2);
  AnimationInterface::SetConstantPosition (m_nodes.Get (0), 0 , 10);
  AnimationInterface::SetConstantPosition (m_nodes.Get (1), 1 , 10);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (m_nodes);

  InternetStackHelper stack;
  stack.Install (m_nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (m_nodes.Get (1));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (100));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (m_nodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));
}

void
AnimationInterfaceTestCase::CheckLogic (void)
{
  NS_TEST_ASSERT_MSG_EQ (m_anim->GetTracePktCount (), 16, "Expected 16 packets traced");
}

/**
 * \ingroup netanim-test
 * \ingroup tests
 *
 * \brief Animation Remaining Energy Test Case
 */
class AnimationRemainingEnergyTestCase : public AbstractAnimationInterfaceTestCase
{
public:
  /**
   * \brief Constructor.
   */
  AnimationRemainingEnergyTestCase ();

private:

  virtual void
  PrepareNetwork ();

  virtual void
  CheckLogic ();

  Ptr<BasicEnergySource> m_energySource; ///< energy source
  Ptr<SimpleDeviceEnergyModel> m_energyModel; ///< energy model
  const double m_initialEnergy; ///< initial energy
};

AnimationRemainingEnergyTestCase::AnimationRemainingEnergyTestCase () :
  AbstractAnimationInterfaceTestCase ("Verify Remaining energy tracing"),
  m_initialEnergy (100)
{
}

void
AnimationRemainingEnergyTestCase::PrepareNetwork (void)
{
  m_energySource = CreateObject<BasicEnergySource>();
  m_energyModel = CreateObject<SimpleDeviceEnergyModel>();

  m_energySource->SetInitialEnergy (m_initialEnergy);
  m_energyModel->SetEnergySource (m_energySource);
  m_energySource->AppendDeviceEnergyModel (m_energyModel);
  m_energyModel->SetCurrentA (20);

  m_nodes.Create (1);
  AnimationInterface::SetConstantPosition (m_nodes.Get (0), 0 , 10);

  // aggregate energy source to node
  m_nodes.Get (0)->AggregateObject (m_energySource);
  // once node's energy will be depleted according to the model
  Simulator::Stop (Seconds (1));
}

void
AnimationRemainingEnergyTestCase::CheckLogic (void)
{
  const double remainingEnergy = m_energySource->GetRemainingEnergy ();

  NS_TEST_ASSERT_MSG_EQ ((remainingEnergy < m_initialEnergy), true, "Energy hasn't depleted!");
  NS_TEST_ASSERT_MSG_EQ_TOL (m_anim->GetNodeEnergyFraction (m_nodes.Get (0)),
                            remainingEnergy / m_initialEnergy,
                            1.0e-13,
                            "Wrong remaining energy value was traced");
}

/**
 * \ingroup netanim-test
 * \ingroup tests
 *
 * \brief Animation Interface Test Suite
 */
static class AnimationInterfaceTestSuite : public TestSuite
{
public:
  AnimationInterfaceTestSuite () :
    TestSuite ("animation-interface", UNIT)
  {
    AddTestCase (new AnimationInterfaceTestCase (), TestCase::QUICK);
    AddTestCase (new AnimationRemainingEnergyTestCase (), TestCase::QUICK);
  }
} g_animationInterfaceTestSuite; ///< the test suite
