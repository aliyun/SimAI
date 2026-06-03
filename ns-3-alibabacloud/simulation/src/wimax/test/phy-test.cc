/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Author: Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 */
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/wimax-helper.h"
#include "ns3/snr-to-block-error-rate-manager.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WimaxPhyTest");

/*
 * Configure a network with 3 SS and 1 BS
 * Install a SIMPLE OFDM PHY layer on all nodes and check that all SSs
 * could register with the BS
 *
 */

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Ns3 Wimax Simple OFDM Test Case
 */
class Ns3WimaxSimpleOFDMTestCase : public TestCase
{
public:
  Ns3WimaxSimpleOFDMTestCase ();
  virtual ~Ns3WimaxSimpleOFDMTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Run once function
   * \param FrameDuration the frame duration
   * \returns true if successful
   */
  bool DoRunOnce (double FrameDuration);

};

Ns3WimaxSimpleOFDMTestCase::Ns3WimaxSimpleOFDMTestCase ()
  : TestCase ("Test the Phy model with different frame durations")
{
}

Ns3WimaxSimpleOFDMTestCase::~Ns3WimaxSimpleOFDMTestCase ()
{
}

bool
Ns3WimaxSimpleOFDMTestCase::DoRunOnce (double FrameDuration)
{
  WimaxHelper::SchedulerType scheduler = WimaxHelper::SCHED_TYPE_SIMPLE;
  NodeContainer ssNodes;
  NodeContainer bsNodes;
  ssNodes.Create (3);
  bsNodes.Create (1);

  WimaxHelper wimax;

  NetDeviceContainer ssDevs, bsDevs;

  ssDevs = wimax.Install (ssNodes, WimaxHelper::DEVICE_TYPE_SUBSCRIBER_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM, scheduler, FrameDuration);
  bsDevs = wimax.Install (bsNodes, WimaxHelper::DEVICE_TYPE_BASE_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM, scheduler, FrameDuration);

  Simulator::Stop (Seconds (1));
  Simulator::Run ();
  for (int i = 0; i < 3; i++)
    {
      if (ssDevs.Get (i)->GetObject<SubscriberStationNetDevice> ()->IsRegistered ()
          == false)
        {
          NS_LOG_DEBUG ("SS[" << i << "] not registered");
          return true; // Test fail because SS[i] is not registered
        }
    }
  Simulator::Destroy ();
  return (false); // Test was ok, all the SSs are registered

}

void
Ns3WimaxSimpleOFDMTestCase::DoRun (void)
{

  double
    frameDuratioTab[7] = { 0.0025, 0.004, 0.005, 0.008, 0.01, 0.0125, 0.02 };
  for (int i = 0; i < 7; i++)
    {
      NS_LOG_DEBUG ("Frame Duration = " << frameDuratioTab[i]);
      if (DoRunOnce (frameDuratioTab[i]) != false)
        {
          return;
        }
    }
}

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Test the SNr tom block error rate module
 */
class Ns3WimaxSNRtoBLERTestCase : public TestCase
{
public:
  Ns3WimaxSNRtoBLERTestCase ();
  virtual ~Ns3WimaxSNRtoBLERTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Run once function
   * \param modulationType the modulation type
   * \returns true if successful
   */
  bool DoRunOnce (uint8_t modulationType);

};

Ns3WimaxSNRtoBLERTestCase::Ns3WimaxSNRtoBLERTestCase ()
  : TestCase ("Test the SNR to block error rate module")
{
}

Ns3WimaxSNRtoBLERTestCase::~Ns3WimaxSNRtoBLERTestCase ()
{
}

bool Ns3WimaxSNRtoBLERTestCase::DoRunOnce (uint8_t modulationType)
{

  SNRToBlockErrorRateManager l_SNRToBlockErrorRateManager;
  l_SNRToBlockErrorRateManager.LoadTraces ();
  SNRToBlockErrorRateRecord * BLERRec;

  for (double i = -5; i < 40; i += 0.1)
    {
      BLERRec = l_SNRToBlockErrorRateManager.GetSNRToBlockErrorRateRecord (i,
                                                                           modulationType);
      delete BLERRec;
    }
  return false;
}

void
Ns3WimaxSNRtoBLERTestCase::DoRun (void)
{
  for (int i = 0; i < 7; i++)
    {
      DoRunOnce (i);
    }
}

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief The test suite
 */
class Ns3WimaxPhyTestSuite : public TestSuite
{
public:
  Ns3WimaxPhyTestSuite ();
};
Ns3WimaxPhyTestSuite::Ns3WimaxPhyTestSuite ()
  : TestSuite ("wimax-phy-layer", UNIT)
{
  AddTestCase (new Ns3WimaxSNRtoBLERTestCase, TestCase::QUICK);
  AddTestCase (new Ns3WimaxSimpleOFDMTestCase, TestCase::QUICK);

}

static Ns3WimaxPhyTestSuite ns3WimaxPhyTestSuite; ///< the test suite
