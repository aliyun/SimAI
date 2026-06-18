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

#include "ns3/basic-energy-source.h"
#include "ns3/wifi-radio-energy-model.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/wifi-radio-energy-model-helper.h"
#include "ns3/energy-source-container.h"
#include "ns3/device-energy-model-container.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/yans-wifi-helper.h"
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("BasicEnergyModelTestSuite");

/**
 * Test case of update remaining energy for BasicEnergySource and
 * WifiRadioEnergyModel.
 */
class BasicEnergyUpdateTest
{
public:
  BasicEnergyUpdateTest ();
  virtual ~BasicEnergyUpdateTest ();

  /**
   * Performs some tests involving state updates and the relative energy consumption
   * \return true is some error happened.
   */
  bool DoRun (void);

private:
  /**
   * \param state Radio state to switch to.
   * \return False if no error occurs.
   *
   * Runs simulation for a while, check if final state & remaining energy is
   * correctly updated.
   */
  bool StateSwitchTest (WifiPhyState state);

private:
  double m_timeS;     //!< Time in seconds
  double m_tolerance; //!< Tolerance for power estimation

  ObjectFactory m_energySource;       //!< Energy source factory
  ObjectFactory m_deviceEnergyModel;  //!< Device energy model factory
};

BasicEnergyUpdateTest::BasicEnergyUpdateTest ()
{
  m_timeS = 15.5; // idle for 15 seconds before changing state
  m_tolerance = 1.0e-5;  //
}

BasicEnergyUpdateTest::~BasicEnergyUpdateTest ()
{
}

bool
BasicEnergyUpdateTest::DoRun (void)
{
  // set types
  m_energySource.SetTypeId ("ns3::BasicEnergySource");
  m_deviceEnergyModel.SetTypeId ("ns3::WifiRadioEnergyModel");

  // run state switch tests
  if (StateSwitchTest (WifiPhyState::IDLE))
    {
      return 1;
      std::cerr << "Problem with state switch test (WifiPhy idle)." << std::endl;
    }
  if (StateSwitchTest (WifiPhyState::CCA_BUSY))
    {
      return 1;
      std::cerr << "Problem with state switch test (WifiPhy cca busy)." << std::endl;
    }
  if (StateSwitchTest (WifiPhyState::TX))
    {
      return 1;
      std::cerr << "Problem with state switch test (WifiPhy tx)." << std::endl;
    }
  if (StateSwitchTest (WifiPhyState::RX))
    {
      return 1;
      std::cerr << "Problem with state switch test (WifiPhy rx)." << std::endl;
    }
  if (StateSwitchTest (WifiPhyState::SWITCHING))
    {
      return 1;
      std::cerr << "Problem with state switch test (WifiPhy switching)." << std::endl;
    }
  if (StateSwitchTest (WifiPhyState::SLEEP))
    {
      return 1;
      std::cerr << "Problem with state switch test (WifiPhy sleep)." << std::endl;
    }
  return 0;
}

bool
BasicEnergyUpdateTest::StateSwitchTest (WifiPhyState state)
{
  // create node
  Ptr<Node> node = CreateObject<Node> ();

  // create energy source
  Ptr<BasicEnergySource> source = m_energySource.Create<BasicEnergySource> ();
  source->SetInitialEnergy (50);
  // aggregate energy source to node
  node->AggregateObject (source);
  source->SetNode (node);

  // create device energy model
  Ptr<WifiRadioEnergyModel> model =
    m_deviceEnergyModel.Create<WifiRadioEnergyModel> ();
  // set energy source pointer
  model->SetEnergySource (source);
  // add device energy model to model list in energy source
  source->AppendDeviceEnergyModel (model);

  // retrieve device energy model from energy source
  DeviceEnergyModelContainer models =
    source->FindDeviceEnergyModels ("ns3::WifiRadioEnergyModel");
  // check list
  if ((models.GetN () == 0))
    {
      std::cerr << "Model list is empty!." << std::endl;
      return true;
    }
  // get pointer
  Ptr<WifiRadioEnergyModel> devModel =
    DynamicCast<WifiRadioEnergyModel> (models.Get (0));
  // check pointer
  if ((devModel == 0))
    {
      std::cerr << "NULL pointer to device model!." << std::endl;
      return true;
    }

  /*
   * The radio will stay IDLE for m_timeS seconds. Then it will switch into a
   * different state.
   */

  // schedule change of state
  Simulator::Schedule (Seconds (m_timeS),
                       &WifiRadioEnergyModel::ChangeState, devModel, state);

  // Calculate remaining energy at simulation stop time
  Simulator::Schedule (Seconds (m_timeS * 2),
                       &BasicEnergySource::UpdateEnergySource, source);

  double timeDelta = 0.000000001; // 1 nanosecond
  // run simulation; stop just after last scheduled event
  Simulator::Stop (Seconds (m_timeS * 2 + timeDelta));
  Simulator::Run ();

  // energy = current * voltage * time

  // calculate idle power consumption
  double estRemainingEnergy = source->GetInitialEnergy ();
  double voltage = source->GetSupplyVoltage ();
  estRemainingEnergy -= devModel->GetIdleCurrentA () * voltage * m_timeS;

  // calculate new state power consumption
  double current = 0.0;
  switch (state)
    {
    case WifiPhyState::IDLE:
      current = devModel->GetIdleCurrentA ();
      break;
    case WifiPhyState::CCA_BUSY:
      current = devModel->GetCcaBusyCurrentA ();
      break;
    case WifiPhyState::TX:
      current = devModel->GetTxCurrentA ();
      break;
    case WifiPhyState::RX:
      current = devModel->GetRxCurrentA ();
      break;
    case WifiPhyState::SWITCHING:
      current = devModel->GetSwitchingCurrentA ();
      break;
    case WifiPhyState::SLEEP:
      current = devModel->GetSleepCurrentA ();
      break;
    case WifiPhyState::OFF:
      current = 0;
      break;
    default:
      NS_FATAL_ERROR ("Undefined radio state: " << state);
      break;
    }
  estRemainingEnergy -= current * voltage * m_timeS;
  estRemainingEnergy = std::max (0.0, estRemainingEnergy);

  // obtain remaining energy from source
  double remainingEnergy = source->GetRemainingEnergy ();
  NS_LOG_DEBUG ("Remaining energy is " << remainingEnergy);
  NS_LOG_DEBUG ("Estimated remaining energy is " << estRemainingEnergy);
  NS_LOG_DEBUG ("Difference is " << estRemainingEnergy - remainingEnergy);

  // check remaining energy
  if ((remainingEnergy > (estRemainingEnergy + m_tolerance))
      || (remainingEnergy < (estRemainingEnergy - m_tolerance)))
    {
      std::cerr << "Incorrect remaining energy!" << std::endl;
      return true;
    }

  // obtain radio state
  WifiPhyState endState = devModel->GetCurrentState ();
  NS_LOG_DEBUG ("Radio state is " << endState);
  // check end state
  if (endState != state)
    {
      std::cerr << "Incorrect end state!" << std::endl;
      return true;
    }
  Simulator::Destroy ();

  return false; // no error
}

// -------------------------------------------------------------------------- //

/**
 * Test case of energy depletion handling for BasicEnergySource and
 * WifiRadioEnergyModel.
 */
class BasicEnergyDepletionTest
{
public:
  BasicEnergyDepletionTest ();
  virtual ~BasicEnergyDepletionTest ();

  /**
   * Performs some tests involving energy depletion
   * \return true is some error happened.
   */
  bool DoRun (void);

private:
  /**
   * Callback invoked when energy is drained from source.
   */
  void DepletionHandler (void);

  /**
   * \param simTimeS Simulation time, in seconds.
   * \param updateIntervalS Device model update interval, in seconds.
   * \return False if all is good.
   *
   * Runs simulation with specified simulation time and update interval.
   */
  bool DepletionTestCase (double simTimeS, double updateIntervalS);

private:
  int m_numOfNodes;         //!< number of nodes in simulation
  int m_callbackCount;      //!< counter for # of callbacks invoked
  double m_simTimeS;        //!< maximum simulation time, in seconds
  double m_timeStepS;       //!< simulation time step size, in seconds
  double m_updateIntervalS; //!< update interval of each device model

};

BasicEnergyDepletionTest::BasicEnergyDepletionTest ()
{
  m_numOfNodes = 10;
  m_callbackCount = 0;
  m_simTimeS = 4.5;
  m_timeStepS = 0.5;
  m_updateIntervalS = 1.5;
}

BasicEnergyDepletionTest::~BasicEnergyDepletionTest ()
{
}

bool
BasicEnergyDepletionTest::DoRun (void)
{
  /*
   * Run simulation with different simulation time and update interval.
   */
  uint8_t ret = 0;

  for (double simTimeS = 0.0; simTimeS <= m_simTimeS; simTimeS += m_timeStepS)
    {
      for (double updateIntervalS = 0.5; updateIntervalS <= m_updateIntervalS;
           updateIntervalS += m_timeStepS)
        {
          if (DepletionTestCase (simTimeS, updateIntervalS))
            {
              ret = 1;
              std::cerr << "Depletion test case problem." << std::endl;
            }
          // reset callback count
          m_callbackCount = 0;
        }
    }
  return ret;
}

void
BasicEnergyDepletionTest::DepletionHandler (void)
{
  m_callbackCount++;
}

bool
BasicEnergyDepletionTest::DepletionTestCase (double simTimeS,
                                             double updateIntervalS)
{
  // create node
  NodeContainer c;
  c.Create (m_numOfNodes);

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

  /*
   * Create and install energy source and a single basic radio energy model on
   * the node using helpers.
   */
  // source helper
  BasicEnergySourceHelper basicSourceHelper;
  // set energy to 0 so that we deplete energy at the beginning of simulation
  basicSourceHelper.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (0.0));
  // set update interval
  basicSourceHelper.Set ("PeriodicEnergyUpdateInterval",
                         TimeValue (Seconds (updateIntervalS)));
  // install source
  EnergySourceContainer sources = basicSourceHelper.Install (c);

  // device energy model helper
  WifiRadioEnergyModelHelper radioEnergyHelper;
  // set energy depletion callback
  WifiRadioEnergyModel::WifiRadioEnergyDepletionCallback callback =
    MakeCallback (&BasicEnergyDepletionTest::DepletionHandler, this);
  radioEnergyHelper.SetDepletionCallback (callback);
  // install on node
  DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install (devices, sources);

  // run simulation
  Simulator::Stop (Seconds (simTimeS));
  Simulator::Run ();
  Simulator::Destroy ();

  NS_LOG_DEBUG ("Simulation time = " << simTimeS << "s");
  NS_LOG_DEBUG ("Update interval = " << updateIntervalS << "s");
  NS_LOG_DEBUG ("Expected callback count is " << m_numOfNodes);
  NS_LOG_DEBUG ("Actual callback count is " << m_callbackCount);

  // check result, call back should only be invoked once
  if (m_numOfNodes != m_callbackCount)
    {
      std::cerr << "Not all callbacks are invoked!" << std::endl;
      return true;
    }

  return false;
}

// -------------------------------------------------------------------------- //

int
main (int argc, char **argv)
{
  BasicEnergyUpdateTest testEnergyUpdate;
  if (testEnergyUpdate.DoRun ())
    {
      return 1;
    }

  BasicEnergyDepletionTest testEnergyDepletion;
  if (testEnergyDepletion.DoRun ())
    {
      return 1;
    }

  return 0;
}
