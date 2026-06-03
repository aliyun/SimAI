/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
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
 * Author: He Wu <mdzz@u.washington.edu>
 */

#include "ns3/rv-battery-model.h"
#include "ns3/wifi-radio-energy-model.h"
#include "ns3/rv-battery-model-helper.h"
#include "ns3/wifi-radio-energy-model-helper.h"
#include "ns3/energy-source-container.h"
#include "ns3/device-energy-model-container.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/config.h"
#include "ns3/command-line.h"
#include "ns3/string.h"
#include "ns3/yans-wifi-helper.h"
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("RvBatteryModelTestSuite");

/**
 * \ingroup energy
 * 
 * This example was originally devised as a test, then it was converted
 * to an example.
 *
 * The script tests the remaining energy for RvBatteryModel and
 * WifiRadioEnergyModel updates. It does so by
 * mimicking the procedure and results published in
 * D. Rakhmatov, S. Vrudhula, D.A. Wallach, "Battery lifetime prediction for energy-aware computing,"
 * Proceedings of the 2002 International Symposium on Low Power Electronics and Design, 2002. ISLPED '02.
 * doi: 10.1109/LPE.2002.146729
 */
class BatteryLifetimeTest
{
public:
  BatteryLifetimeTest ();
  virtual ~BatteryLifetimeTest ();

  /**
   * Creates load profiles according to
   * D. Rakhmatov, S. Vrudhula, D.A. Wallach, "Battery lifetime prediction for energy-aware computing,"
   * Proceedings of the 2002 International Symposium on Low Power Electronics and Design, 2002. ISLPED '02.
   * doi: 10.1109/LPE.2002.146729
   */
  void CreateLoadProfiles (void);

  /**
   * \param load Load value, in Amperes (A).
   * \param expLifetime Expected lifetime.
   * \return False if no error occurs.
   *
   * Runs simulation with constant load and checks the battery lifetime with
   * known results.
   */
  bool ConstantLoadTest (double load, Time expLifetime);

  /**
   * \param loads Load profile.
   * \param timeStamps Time stamps.
   * \param expLifetime Expected lifetime.
   * \returns False if no error occurs.
   *
   * Runs simulation with variable load and checks the battery lifetime with
   * known results.
   */
  bool VariableLoadTest (std::vector<double> loads,
                         std::vector<Time> timeStamps,
                         Time expLifetime);

  /// Load profile of the battery
  typedef struct LoadProfile
  {
    std::vector<double> loads;      //!< Loads container
    std::vector<Time> timeStamps;   //!< Timestamps container
    Time itsyLifetime;              //!< Expected lifetime for an ITSY battery
    Time dualFoilLifeTime;          //!< Expected lifetime for a Dualfoil battery
  } LoadProfile;

  std::vector<LoadProfile> m_loadProfiles; //!< Load profiles
  double m_alpha; //!< Alpha parameter of the battery model
  double m_beta;  //!< Beta parameter of the battery model
};

BatteryLifetimeTest::BatteryLifetimeTest ()
{
  // Itsy battery
  m_alpha = 35220;
  m_beta = 0.637;
}

BatteryLifetimeTest::~BatteryLifetimeTest ()
{
}

void
BatteryLifetimeTest::CreateLoadProfiles (void)
{
  // create set of load profiles
  LoadProfile profile;

  std::vector<double> loads;
  std::vector<Time> timeStamps;

  // C1
  loads.push_back (0.628);
  loads.push_back (0);
  loads.push_back (0.628);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (19.5 * 60));
  timeStamps.push_back (Seconds (26.0 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (55.0 * 60);     // 55.0 minutes
  profile.dualFoilLifeTime = Seconds (36.2 * 60); // 36.2 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C2
  loads.push_back (0.4947);
  loads.push_back (0);
  loads.push_back (0.4947);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (31.0 * 60));
  timeStamps.push_back (Seconds (41.3 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (73.9 * 60);     // 73.9 minutes
  profile.dualFoilLifeTime = Seconds (55.8 * 60); // 55.8 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C3
  loads.push_back (0.4256);
  loads.push_back (0);
  loads.push_back (0.4256);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (41.0 * 60));
  timeStamps.push_back (Seconds (54.6 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (88.8 * 60);     // 88.8 minutes
  profile.dualFoilLifeTime = Seconds (71.8 * 60); // 71.8 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C4
  loads.push_back (0.2923);
  loads.push_back (0);
  loads.push_back (0.2923);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (74.6 * 60));
  timeStamps.push_back (Seconds (99.5 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (137.8 * 60);      // 137.8 minutes
  profile.dualFoilLifeTime = Seconds (124.9 * 60);  // 124.9 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C5
  loads.push_back (0.2227);
  loads.push_back (0);
  loads.push_back (0.2227);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (105.7 * 60));
  timeStamps.push_back (Seconds (140.9 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (185.8 * 60);      // 185.8 minutes
  profile.dualFoilLifeTime = Seconds (176.7 * 60);  // 176.7 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C6
  loads.push_back (0.628);
  loads.push_back (0);
  loads.push_back (0.628);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (19.5 * 60));
  timeStamps.push_back (Seconds (29.9 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (58.9 * 60);     // 58.9 minutes
  profile.dualFoilLifeTime = Seconds (41.0 * 60); // 41.0 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C7
  loads.push_back (0.628);
  loads.push_back (0);
  loads.push_back (0.628);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (19.5 * 60));
  timeStamps.push_back (Seconds (22.1 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (51.1 * 60);     // 51.1 minutes
  profile.dualFoilLifeTime = Seconds (30.8 * 60); // 30.8 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C8
  loads.push_back (0.628);
  loads.push_back (0);
  loads.push_back (0.628);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (23.4 * 60));
  timeStamps.push_back (Seconds (29.9 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (55.0 * 60);     // 55.0 minutes
  profile.dualFoilLifeTime = Seconds (37.4 * 60); // 37.4 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C9
  loads.push_back (0.628);
  loads.push_back (0);
  loads.push_back (0.628);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (15.6 * 60));
  timeStamps.push_back (Seconds (22.1 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (55.0 * 60);     // 55.0 minutes
  profile.dualFoilLifeTime = Seconds (35.2 * 60); // 35.2 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C10
  loads.push_back (0.300);
  loads.push_back (0.628);
  loads.push_back (0.4947);
  loads.push_back (0.2523);
  loads.push_back (0.2341);
  loads.push_back (0.1379);
  loads.push_back (0.1139);
  loads.push_back (0.2656);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (0.5 * 60));
  timeStamps.push_back (Seconds (5.5 * 60));
  timeStamps.push_back (Seconds (10.5 * 60));
  timeStamps.push_back (Seconds (35.5 * 60));
  timeStamps.push_back (Seconds (60.5 * 60));
  timeStamps.push_back (Seconds (85.5 * 60));
  timeStamps.push_back (Seconds (110.5 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (144.3 * 60);      // 144.3 minutes
  profile.dualFoilLifeTime = Seconds (132.6 * 60);  // 132.6 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C11
  loads.push_back (0.300);
  loads.push_back (0.1139);
  loads.push_back (0.1379);
  loads.push_back (0.2341);
  loads.push_back (0.2523);
  loads.push_back (0.4947);
  loads.push_back (0.628);
  loads.push_back (0.2656);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (0.5 * 60));
  timeStamps.push_back (Seconds (25.5 * 60));
  timeStamps.push_back (Seconds (50.5 * 60));
  timeStamps.push_back (Seconds (75.5 * 60));
  timeStamps.push_back (Seconds (100.5 * 60));
  timeStamps.push_back (Seconds (105.5 * 60));
  timeStamps.push_back (Seconds (110.5 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (144.3 * 60);      // 144.3 minutes
  profile.dualFoilLifeTime = Seconds (107.4 * 60);  // 107.4 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C12
  loads.push_back (0.300);
  loads.push_back (0.1139);
  loads.push_back (0.1379);
  loads.push_back (0.2341);
  loads.push_back (0.2523);
  loads.push_back (0.4947);
  loads.push_back (0.0);
  loads.push_back (0.300);
  loads.push_back (0.628);
  loads.push_back (0.2656);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (0.5 * 60));
  timeStamps.push_back (Seconds (25.5 * 60));
  timeStamps.push_back (Seconds (50.5 * 60));
  timeStamps.push_back (Seconds (75.5 * 60));
  timeStamps.push_back (Seconds (100.5 * 60));
  timeStamps.push_back (Seconds (105.5 * 60));
  timeStamps.push_back (Seconds (130.5 * 60));
  timeStamps.push_back (Seconds (131.0 * 60));
  timeStamps.push_back (Seconds (136.0 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (169.3 * 60);      // 169.3 minutes
  profile.dualFoilLifeTime = Seconds (155.4 * 60);  // 155.4 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C13
  loads.push_back (0.300);
  timeStamps.push_back (Seconds (0));

  for (int i = 0; i < 5; i++)
    {
      loads.push_back (0.628);
      loads.push_back (0.4947);
      loads.push_back (0.2523);
      loads.push_back (0.2341);
      loads.push_back (0.1379);
      loads.push_back (0.1139);

      timeStamps.push_back (Seconds ((0.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((1.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((2.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((7.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((12.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((17.5 + i * 22.5) * 60));
    }

  loads.push_back (0.2656);
  timeStamps.push_back (Seconds (110.5 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (144.3 * 60);      // 144.3 minutes
  profile.dualFoilLifeTime = Seconds (131.7 * 60);  // 131.7 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C14, time stamp calculation in paper is off, using our own estimated value
  loads.push_back (0.300);
  timeStamps.push_back (Seconds (0));

  for (int i = 0; i < 5; i++)
    {
      loads.push_back (0.1139);
      loads.push_back (0.1379);
      loads.push_back (0.2341);
      loads.push_back (0.2523);
      loads.push_back (0.4947);
      loads.push_back (0.628);

      timeStamps.push_back (Seconds ((0.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((5.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((10.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((15.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((20.5 + i * 22.5) * 60));
      timeStamps.push_back (Seconds ((21.5 + i * 22.5) * 60));
    }

  loads.push_back (0.2656);
  timeStamps.push_back (Seconds (112.5 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (141.5 * 60);      // 141.5 minutes
  profile.dualFoilLifeTime = Seconds (126.3 * 60);  // 126.3 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C15
  loads.push_back (0.2227);
  loads.push_back (0.2045);
  loads.push_back (0.1083);
  loads.push_back (0.0843);
  loads.push_back (0.2227);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (50.0 * 60));
  timeStamps.push_back (Seconds (100.0 * 60));
  timeStamps.push_back (Seconds (150.0 * 60));
  timeStamps.push_back (Seconds (200.0 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (211.4 * 60);      // 211.4 minutes
  profile.dualFoilLifeTime = Seconds (209.2 * 60);  // 209.2 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C16
  loads.push_back (0.0843);
  loads.push_back (0.1083);
  loads.push_back (0.2045);
  loads.push_back (0.2227);
  loads.push_back (0.2227);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (50.0 * 60));
  timeStamps.push_back (Seconds (100.0 * 60));
  timeStamps.push_back (Seconds (150.0 * 60));
  timeStamps.push_back (Seconds (200.0 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (211.4 * 60);      // 211.4 minutes
  profile.dualFoilLifeTime = Seconds (200.7 * 60);  // 200.7 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C17
  loads.push_back (0.0843);
  loads.push_back (0.1083);
  loads.push_back (0.2045);
  loads.push_back (0.0);
  loads.push_back (0.2227);
  loads.push_back (0.2227);

  timeStamps.push_back (Seconds (0));
  timeStamps.push_back (Seconds (50.0 * 60));
  timeStamps.push_back (Seconds (100.0 * 60));
  timeStamps.push_back (Seconds (150.0 * 60));
  timeStamps.push_back (Seconds (200.0 * 60));
  timeStamps.push_back (Seconds (250.0 * 60));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (261.4 * 60);      // 261.4 minutes
  profile.dualFoilLifeTime = Seconds (251.2 * 60);  // 251.2 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C18
  for (int i = 0; i < 10; i++)
    {
      loads.push_back (0.0843);
      loads.push_back (0.1083);
      loads.push_back (0.2045);
      loads.push_back (0.2227);

      timeStamps.push_back (Seconds ((0.0 + i * 20.0) * 60));
      timeStamps.push_back (Seconds ((5.0 + i * 20.0) * 60));
      timeStamps.push_back (Seconds ((10.0 + i * 20.0) * 60));
      timeStamps.push_back (Seconds ((15.0 + i * 20.0) * 60));
    }

  loads.push_back (0.2227);
  timeStamps.push_back (Seconds (200.0));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (211.4 * 60);      // 211.4 minutes
  profile.dualFoilLifeTime = Seconds (204.6 * 60);  // 204.6 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C19
  for (int i = 0; i < 10; i++)
    {
      loads.push_back (0.0755);
      loads.push_back (0.0949);
      loads.push_back (0.2045);
      loads.push_back (0.2227);

      timeStamps.push_back (Seconds ((0.0 + i * 20.0) * 60));
      timeStamps.push_back (Seconds ((5.0 + i * 20.0) * 60));
      timeStamps.push_back (Seconds ((10.0 + i * 20.0) * 60));
      timeStamps.push_back (Seconds ((15.0 + i * 20.0) * 60));
    }

  loads.push_back (0.2227);
  timeStamps.push_back (Seconds (200.0));

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (216.4 * 60);      // 216.4 minutes
  profile.dualFoilLifeTime = Seconds (208.7 * 60);  // 208.7 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C20
  for (int i = 0; i < 50; i++)
    {
      loads.push_back (0.4947);
      loads.push_back (0.628);

      timeStamps.push_back (Seconds ((0.0 + i * 2.0) * 60));
      timeStamps.push_back (Seconds ((1.0 + i * 2.0) * 60));
    }

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (55.3 * 60);       // 55.3 minutes
  profile.dualFoilLifeTime = Seconds (33.2 * 60);   // 33.2 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C21
  for (int i = 0; i < 50; i++)
    {
      loads.push_back (0.4947);
      loads.push_back (0.628);
      loads.push_back (0.0576);

      timeStamps.push_back (Seconds ((0.0 + i * 3.0) * 60));
      timeStamps.push_back (Seconds ((1.0 + i * 3.0) * 60));
      timeStamps.push_back (Seconds ((2.0 + i * 3.0) * 60));
    }

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (79.6 * 60);       // 79.6 minutes
  profile.dualFoilLifeTime = Seconds (55.9 * 60);   // 55.9 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();

  // C22
  for (int i = 0; i < 150; i++)
    {
      loads.push_back (0.005 + 0.005 * i);
      timeStamps.push_back (Seconds ((0.0 + i * 1.0) * 60));
    }

  profile.loads = loads;
  profile.timeStamps = timeStamps;
  profile.itsyLifetime = Seconds (112.2 * 60);      // 112.2 minutes
  profile.dualFoilLifeTime = Seconds (94.5 * 60);   // 94.5 minutes

  m_loadProfiles.push_back (profile);

  loads.clear ();
  timeStamps.clear ();
}

int
main (int argc, char **argv)
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  NS_LOG_DEBUG ("Constant load run.");

  BatteryLifetimeTest test;
  int ret = 0;

  if (test.ConstantLoadTest (0.640, Seconds (2844.0)))
    {
      ret = 1;
      std::cerr << "Problems with constant load test (640mA)." << std::endl;
    }
  if (test.ConstantLoadTest (0.320, Seconds (6146.0)))
    {
      ret = 1;
      std::cerr << "Problems with constant load test (320mA)." << std::endl;
    }
  if (test.ConstantLoadTest (0.128, Seconds (16052.0)))
    {
      ret = 1;
      std::cerr << "Problems with constant load test (128mA)." << std::endl;
    }
  if (test.ConstantLoadTest (0.064, Seconds (32561.0)))
    {
      ret = 1;
      std::cerr << "Problems with constant load test (64mA)." << std::endl;
    }
  if (test.ConstantLoadTest (0.032, Seconds (65580.0)))
    {
      ret = 1;
      std::cerr << "Problems with constant load test (32mA)." << std::endl;
    }

  // create load profiles for variable load test
  test.CreateLoadProfiles ();

  // variable load with Itsy battery
  NS_LOG_DEBUG ("\n\nItsy");
  test.m_alpha = 35220;
  test.m_beta = 0.637;
  for (uint32_t i = 0; i < test.m_loadProfiles.size (); i++)
    {
      NS_LOG_DEBUG ("========");
      NS_LOG_DEBUG ("Variable load profile C" << i + 1);
      if (test.VariableLoadTest (test.m_loadProfiles[i].loads,
                                 test.m_loadProfiles[i].timeStamps,
                                 test.m_loadProfiles[i].itsyLifetime))
        {
          ret = 1;
          std::cerr << "Problems with variable load test (Itsy)." << std::endl;
        }
    }

  // variable load with DUALFOIL battery
  NS_LOG_DEBUG ("\n\nDUALFOIL");
  test.m_alpha = 40027;
  test.m_beta = 0.276;
  for (uint32_t i = 0; i < test.m_loadProfiles.size (); i++)
    {
      NS_LOG_DEBUG ("========");
      NS_LOG_DEBUG ("Variable load profile C" << i + 1);
      if (test.VariableLoadTest (test.m_loadProfiles[i].loads,
                                 test.m_loadProfiles[i].timeStamps,
                                 test.m_loadProfiles[i].dualFoilLifeTime))
        {
          ret = 1;
          std::cerr << "Problems with variable load test (DUALFOIL)." << std::endl;
        }
    }

  return ret;
}

bool
BatteryLifetimeTest::ConstantLoadTest (double load, Time expLifetime)
{
  // create single node
  NodeContainer c;
  c.Create (1);

  std::string phyMode ("DsssRate1Mbps");

  // disable fragmentation for frames below 2200 bytes
  Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold",
                      StringValue ("2200"));
  // turn off RTS/CTS for frames below 2200 bytes
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                      StringValue ("2200"));
  // Fix non-unicast data rate to be the same as that of unicast
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                      StringValue (phyMode));

  // install YansWifiPhy
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211b);

  YansWifiPhyHelper wifiPhy;
  /*
   * This is one parameter that matters when using FixedRssLossModel, set it to
   * zero; otherwise, gain will be added.
   */
  wifiPhy.Set ("RxGain", DoubleValue (0));
  // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add a MAC and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue (phyMode),
                                "ControlMode", StringValue (phyMode));
  // Set it to ad-hoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

  // Create and install battery model and device models
  // RV battery model
  RvBatteryModelHelper rvModelHelper;
  // Set alpha & beta values
  rvModelHelper.Set ("RvBatteryModelAlphaValue", DoubleValue (m_alpha));
  rvModelHelper.Set ("RvBatteryModelBetaValue", DoubleValue (m_beta));
  rvModelHelper.Set ("RvBatteryModelLowBatteryThreshold", DoubleValue (0.0));
  // install source
  EnergySourceContainer sources = rvModelHelper.Install (c);
  // device energy model
  WifiRadioEnergyModelHelper radioEnergyHelper;
  // set VariableLoadTestIDLE current, which will be the constant load
  radioEnergyHelper.Set ("IdleCurrentA", DoubleValue (load));
  // install on node
  DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install (devices, sources);

  // run simulation
  Simulator::Stop (Seconds (70000.0));
  Simulator::Run ();

  Time actualLifetime;
  Ptr<RvBatteryModel> srcPtr = DynamicCast<RvBatteryModel> (sources.Get (0));
  actualLifetime = srcPtr->GetLifetime ();

  NS_LOG_DEBUG ("Expected lifetime = " << expLifetime.As (Time::S));
  NS_LOG_DEBUG ("Actual lifetime = " << actualLifetime.As (Time::S));

  Simulator::Destroy ();

  if (actualLifetime != expLifetime)
    {
      std::cerr << "ConstantLoadTest: Incorrect lifetime for load " << load << std::endl;
      return true;
    }

  return false; // error free
}

bool
BatteryLifetimeTest::VariableLoadTest (std::vector<double> loads,
                                       std::vector<Time> timeStamps,
                                       Time expLifetime)
{
  NS_ASSERT (loads.size () == timeStamps.size ());

  // create single node
  NodeContainer c;
  c.Create (1);

  std::string phyMode ("DsssRate1Mbps");

  // disable fragmentation for frames below 2200 bytes
  Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold",
                      StringValue ("2200"));
  // turn off RTS/CTS for frames below 2200 bytes
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                      StringValue ("2200"));
  // Fix non-unicast data rate to be the same as that of unicast
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                      StringValue (phyMode));

  // install YansWifiPhy
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211b);

  YansWifiPhyHelper wifiPhy;
  /*
   * This is one parameter that matters when using FixedRssLossModel, set it to
   * zero; otherwise, gain will be added.
   */
  wifiPhy.Set ("RxGain", DoubleValue (0));
  // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add a MAC and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue (phyMode),
                                "ControlMode", StringValue (phyMode));
  // Set it to ad-hoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

  // Create and install battery model and device models
  // RV battery model
  RvBatteryModelHelper rvModelHelper;
  // Set alpha & beta values
  rvModelHelper.Set ("RvBatteryModelAlphaValue", DoubleValue (m_alpha));
  rvModelHelper.Set ("RvBatteryModelBetaValue", DoubleValue (m_beta));
  rvModelHelper.Set ("RvBatteryModelLowBatteryThreshold", DoubleValue (0.0));
  // install source
  EnergySourceContainer sources = rvModelHelper.Install (c);
  // device energy model
  WifiRadioEnergyModelHelper radioEnergyHelper;
  // set VariableLoadTestIDLE current, which will be the constant load
  radioEnergyHelper.Set ("IdleCurrentA", DoubleValue (loads[0]));
  // install on node
  DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install (devices, sources);


  Ptr<WifiRadioEnergyModel> wifiDevicePtr = DynamicCast<WifiRadioEnergyModel> (deviceModels.Get (0));
  // schedule load change events
  for (uint32_t i = 1; i < loads.size (); i++)
    {
      Simulator::Schedule (timeStamps[i], &WifiRadioEnergyModel::SetIdleCurrentA,
                           wifiDevicePtr, loads[i]);
    }

  // run simulation
  Simulator::Stop (Seconds (70000.0));
  Simulator::Run ();

  Time actualLifetime;
  Ptr<RvBatteryModel> srcPtr = DynamicCast<RvBatteryModel> (sources.Get (0));
  actualLifetime = srcPtr->GetLifetime ();

  NS_LOG_DEBUG ("Expected lifetime = " << expLifetime.As (Time::S));
  NS_LOG_DEBUG ("Actual lifetime = " << actualLifetime.As (Time::S));
  NS_LOG_DEBUG ("Difference = " << (expLifetime - actualLifetime).As (Time::S));

  Simulator::Destroy ();

  // error tolerance = 120s
  if (Abs (actualLifetime - expLifetime)  > Seconds (120))
    {
      std::cerr << "VariableLoadTest: Incorrect lifetime." << std::endl;
      return true;
    }

  return false;   // error free
}

