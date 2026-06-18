/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
 */

/**
* This example shows how to configure the 3GPP channel model classes to
* compute the SNR between two nodes.
* The simulation involves two static nodes which are placed at a certain
* distance from each other and communicates through a wireless channel at
* 2 GHz with a bandwidth of 18 MHz. The default propagation environment is
* 3D-urban macro (UMa) and it can be configured changing the value of the
* string "scenario".
* Each node hosts a SimpleNetDevice and has an antenna array with 4 elements.
*/

#include "ns3/core-module.h"
#include "ns3/three-gpp-channel-model.h"
#include "ns3/uniform-planar-array.h"
#include <fstream>
#include "ns3/three-gpp-spectrum-propagation-loss-model.h"
#include "ns3/net-device.h"
#include "ns3/simple-net-device.h"
#include "ns3/node.h"
#include "ns3/node-container.h"
#include "ns3/mobility-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/lte-spectrum-value-helper.h"
#include "ns3/channel-condition-model.h"
#include "ns3/three-gpp-propagation-loss-model.h"

NS_LOG_COMPONENT_DEFINE ("ThreeGppChannelExample");

using namespace ns3;

static Ptr<ThreeGppPropagationLossModel> m_propagationLossModel; //!< the PropagationLossModel object
static Ptr<ThreeGppSpectrumPropagationLossModel> m_spectrumLossModel; //!< the SpectrumPropagationLossModel object

/**
 * \brief A structure that holds the parameters for the
 * ComputeSnr function. In this way the problem with the limited
 * number of parameters of method Schedule is avoided.
 */
struct ComputeSnrParams
{
  Ptr<MobilityModel> txMob; //!< the tx mobility model
  Ptr<MobilityModel> rxMob; //!< the rx mobility model
  double txPow; //!< the tx power in dBm
  double noiseFigure; //!< the noise figure in dB
  Ptr<PhasedArrayModel> txAntenna; //!< the tx antenna array
  Ptr<PhasedArrayModel> rxAntenna; //!< the rx antenna array

  /**
   * \brief Constructor
   * \param pTxMob the tx mobility model
   * \param pRxMob the rx mobility model
   * \param pTxPow the tx power in dBm
   * \param pNoiseFigure the noise figure in dB
   * \param pTxAntenna the tx antenna array
   * \param pRxAntenna the rx antenna array
   */
  ComputeSnrParams (Ptr<MobilityModel> pTxMob, Ptr<MobilityModel> pRxMob, double pTxPow, double pNoiseFigure,
                    Ptr<PhasedArrayModel> pTxAntenna, Ptr<PhasedArrayModel> pRxAntenna)
  {
    txMob = pTxMob;
    rxMob = pRxMob;
    txPow = pTxPow;
    noiseFigure = pNoiseFigure;
    txAntenna = pTxAntenna;
    rxAntenna = pRxAntenna;
  }
};

/**
 * Perform the beamforming using the DFT beamforming method
 * \param thisDevice the device performing the beamforming
 * \param thisAntenna the antenna object associated to thisDevice
 * \param otherDevice the device towards which point the beam
 */
static void
DoBeamforming (Ptr<NetDevice> thisDevice, Ptr<PhasedArrayModel> thisAntenna, Ptr<NetDevice> otherDevice)
{
  PhasedArrayModel::ComplexVector antennaWeights;

  // retrieve the position of the two devices
  Vector aPos = thisDevice->GetNode ()->GetObject<MobilityModel> ()->GetPosition ();
  Vector bPos = otherDevice->GetNode ()->GetObject<MobilityModel> ()->GetPosition ();

  // compute the azimuth and the elevation angles
  Angles completeAngle (bPos,aPos);
  double hAngleRadian = completeAngle.GetAzimuth ();
  
  double vAngleRadian = completeAngle.GetInclination (); // the elevation angle

  // retrieve the number of antenna elements
  int totNoArrayElements = thisAntenna->GetNumberOfElements ();

  // the total power is divided equally among the antenna elements
  double power = 1 / sqrt (totNoArrayElements);

  // compute the antenna weights
  for (int ind = 0; ind < totNoArrayElements; ind++)
    {
      Vector loc = thisAntenna->GetElementLocation (ind);
      double phase = -2 * M_PI * (sin (vAngleRadian) * cos (hAngleRadian) * loc.x
                                  + sin (vAngleRadian) * sin (hAngleRadian) * loc.y
                                  + cos (vAngleRadian) * loc.z);
      antennaWeights.push_back (exp (std::complex<double> (0, phase)) * power);
    }

  // store the antenna weights
  thisAntenna->SetBeamformingVector (antennaWeights);
}

/**
 * Compute the average SNR
 * \param params A structure that holds the parameters that are needed to perform calculations in ComputeSnr
 */
static void
ComputeSnr (ComputeSnrParams& params)
{
  // Create the tx PSD using the LteSpectrumValueHelper
  // 100 RBs corresponds to 18 MHz (1 RB = 180 kHz)
  // EARFCN 100 corresponds to 2125.00 MHz
  std::vector<int> activeRbs0 (100);
  for (int i = 0; i < 100 ; i++)
  {
    activeRbs0[i] = i;
  }
  Ptr<SpectrumValue> txPsd = LteSpectrumValueHelper::CreateTxPowerSpectralDensity (2100, 100, params.txPow, activeRbs0);
  Ptr<SpectrumValue> rxPsd = txPsd->Copy ();
  NS_LOG_DEBUG ("Average tx power " << 10*log10(Sum (*txPsd) * 180e3) << " dB");

  // create the noise PSD
  Ptr<SpectrumValue> noisePsd = LteSpectrumValueHelper::CreateNoisePowerSpectralDensity (2100, 100, params.noiseFigure);
  NS_LOG_DEBUG ("Average noise power " << 10*log10 (Sum (*noisePsd) * 180e3) << " dB");

  // apply the pathloss
  double propagationGainDb = m_propagationLossModel->CalcRxPower (0, params.txMob, params.rxMob);
  NS_LOG_DEBUG ("Pathloss " << -propagationGainDb << " dB");
  double propagationGainLinear = std::pow (10.0, (propagationGainDb) / 10.0);
  *(rxPsd) *= propagationGainLinear;

  NS_ASSERT_MSG (params.txAntenna, "params.txAntenna is nullptr!");
  NS_ASSERT_MSG (params.rxAntenna, "params.rxAntenna is nullptr!");

  // apply the fast fading and the beamforming gain
  rxPsd = m_spectrumLossModel->CalcRxPowerSpectralDensity (rxPsd, params.txMob, params.rxMob, params.txAntenna, params.rxAntenna);
  NS_LOG_DEBUG ("Average rx power " << 10 * log10 (Sum (*rxPsd) * 180e3) << " dB");

  // compute the SNR
  NS_LOG_DEBUG ("Average SNR " << 10 * log10 (Sum (*rxPsd) / Sum (*noisePsd)) << " dB");

  // print the SNR and pathloss values in the snr-trace.txt file
  std::ofstream f;
  f.open ("snr-trace.txt", std::ios::out | std::ios::app);
  f << Simulator::Now ().GetSeconds () << " " << 10 * log10 (Sum (*rxPsd) / Sum (*noisePsd)) << " " << propagationGainDb << std::endl;
  f.close ();
}

int
main (int argc, char *argv[])
{
  double frequency = 2125.0e6; // operating frequency in Hz (corresponds to EARFCN 2100)
  double txPow = 49.0; // tx power in dBm
  double noiseFigure = 9.0; // noise figure in dB
  double distance = 10.0; // distance between tx and rx nodes in meters
  uint32_t simTime = 10000; // simulation time in milliseconds
  uint32_t timeRes = 10; // time resolution in milliseconds
  std::string scenario = "UMa"; // 3GPP propagation scenario

  Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds (1))); // update the channel at each iteration
  Config::SetDefault ("ns3::ThreeGppChannelConditionModel::UpdatePeriod", TimeValue(MilliSeconds (0.0))); // do not update the channel condition

  RngSeedManager::SetSeed(1);
  RngSeedManager::SetRun(1);

  // create and configure the factories for the channel condition and propagation loss models
  ObjectFactory propagationLossModelFactory;
  ObjectFactory channelConditionModelFactory;
  if (scenario == "RMa")
  {
    propagationLossModelFactory.SetTypeId (ThreeGppRmaPropagationLossModel::GetTypeId ());
    channelConditionModelFactory.SetTypeId (ThreeGppRmaChannelConditionModel::GetTypeId ());
  }
  else if (scenario == "UMa")
  {
    propagationLossModelFactory.SetTypeId (ThreeGppUmaPropagationLossModel::GetTypeId ());
    channelConditionModelFactory.SetTypeId (ThreeGppUmaChannelConditionModel::GetTypeId ());
  }
  else if (scenario == "UMi-StreetCanyon")
  {
    propagationLossModelFactory.SetTypeId (ThreeGppUmiStreetCanyonPropagationLossModel::GetTypeId ());
    channelConditionModelFactory.SetTypeId (ThreeGppUmiStreetCanyonChannelConditionModel::GetTypeId ());
  }
  else if (scenario == "InH-OfficeOpen")
  {
    propagationLossModelFactory.SetTypeId (ThreeGppIndoorOfficePropagationLossModel::GetTypeId ());
    channelConditionModelFactory.SetTypeId (ThreeGppIndoorOpenOfficeChannelConditionModel::GetTypeId ());
  }
  else if (scenario == "InH-OfficeMixed")
  {
    propagationLossModelFactory.SetTypeId (ThreeGppIndoorOfficePropagationLossModel::GetTypeId ());
    channelConditionModelFactory.SetTypeId (ThreeGppIndoorMixedOfficeChannelConditionModel::GetTypeId ());
  }
  else
  {
    NS_FATAL_ERROR ("Unknown scenario");
  }

  // create the propagation loss model
  m_propagationLossModel = propagationLossModelFactory.Create<ThreeGppPropagationLossModel> ();
  m_propagationLossModel->SetAttribute ("Frequency", DoubleValue (frequency));
  m_propagationLossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false));

  // create the spectrum propagation loss model
  m_spectrumLossModel = CreateObject<ThreeGppSpectrumPropagationLossModel> ();
  m_spectrumLossModel->SetChannelModelAttribute ("Frequency", DoubleValue (frequency));
  m_spectrumLossModel->SetChannelModelAttribute ("Scenario", StringValue (scenario));

  // create the channel condition model and associate it with the spectrum and
  // propagation loss model
  Ptr<ChannelConditionModel> condModel = channelConditionModelFactory.Create<ThreeGppChannelConditionModel> ();
  m_spectrumLossModel->SetChannelModelAttribute ("ChannelConditionModel", PointerValue (condModel));
  m_propagationLossModel->SetChannelConditionModel (condModel);

  // create the tx and rx nodes
  NodeContainer nodes;
  nodes.Create (2);

  // create the tx and rx devices
  Ptr<SimpleNetDevice> txDev = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleNetDevice> rxDev = CreateObject<SimpleNetDevice> ();

  // associate the nodes and the devices
  nodes.Get (0)->AddDevice (txDev);
  txDev->SetNode (nodes.Get (0));
  nodes.Get (1)->AddDevice (rxDev);
  rxDev->SetNode (nodes.Get (1));

  // create the tx and rx mobility models, set the positions
  Ptr<MobilityModel> txMob = CreateObject<ConstantPositionMobilityModel> ();
  txMob->SetPosition (Vector (0.0,0.0,10.0));
  Ptr<MobilityModel> rxMob = CreateObject<ConstantPositionMobilityModel> ();
  rxMob->SetPosition (Vector (distance,0.0,1.6));

  // assign the mobility models to the nodes
  nodes.Get (0)->AggregateObject (txMob);
  nodes.Get (1)->AggregateObject (rxMob);

  // create the antenna objects and set their dimensions
  Ptr<PhasedArrayModel> txAntenna = CreateObjectWithAttributes<UniformPlanarArray> ("NumColumns", UintegerValue (2), "NumRows", UintegerValue (2));
  Ptr<PhasedArrayModel> rxAntenna = CreateObjectWithAttributes<UniformPlanarArray> ("NumColumns", UintegerValue (2), "NumRows", UintegerValue (2));

  // set the beamforming vectors
  DoBeamforming (txDev, txAntenna, rxDev);
  DoBeamforming (rxDev, rxAntenna, txDev);

  for (int i = 0; i < floor (simTime / timeRes); i++)
  {
    Simulator::Schedule (MilliSeconds (timeRes*i), &ComputeSnr, ComputeSnrParams (txMob, rxMob, txPow, noiseFigure, txAntenna, rxAntenna));
  }

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
