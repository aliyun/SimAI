/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 * Author: Benjamin Cizdziel <ben.cizdziel@gmail.com>
 */

#include <iostream>
#include <stdlib.h>
#include <ns3/core-module.h>
#include <ns3/mobility-module.h>
#include <ns3/spectrum-helper.h>
#include <ns3/spectrum-analyzer-helper.h>
#include <ns3/tv-spectrum-transmitter-helper.h>

using namespace ns3;

/**
 * This example uses the TvSpectrumTransmitterHelper class to generate a random
 * number of COFDM TV transmitters within a 250 km radius around latitude
 * 48.86 degrees and longitude 2.35 degrees (Paris, France). The transmitters' 
 * frequencies and bandwidths correspond to the European TV channel allocations.
 * These TV transmitters model the digital DVB-T standard.
 *
 * A spectrum analyzer is used to measure the transmitted spectra from the
 * TV transmitters. The file "spectrum-analyzer-tv-sim-regional-0-0.tr" contains
 * its output post simulation (and can be plotted with Gnuplot or MATLAB).
 */
int main (int argc, char** argv)
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  /* random seed and run number; adjust these to change random draws */
  RngSeedManager::SetSeed(1);
  RngSeedManager::SetRun(3);

  /* nodes and positions */
  NodeContainer spectrumAnalyzerNodes;
  spectrumAnalyzerNodes.Create (1);
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> nodePositionList = CreateObject<ListPositionAllocator> ();
  Vector coordinates = GeographicPositions::GeographicToCartesianCoordinates (48.86, 
                                                                              2.35, 
                                                                              0,
                                                                              GeographicPositions::SPHERE);
  nodePositionList->Add (coordinates); // spectrum analyzer
  mobility.SetPositionAllocator (nodePositionList);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (spectrumAnalyzerNodes);

  /* channel and propagation */
  SpectrumChannelHelper channelHelper = SpectrumChannelHelper::Default ();
  channelHelper.SetChannel ("ns3::MultiModelSpectrumChannel");
  Ptr<SpectrumChannel> channel = channelHelper.Create ();
  
  /* TV transmitter setup */
  TvSpectrumTransmitterHelper tvTransHelper;
  tvTransHelper.SetChannel (channel);
  tvTransHelper.SetAttribute ("StartingTime", TimeValue (Seconds (0.1)));
  tvTransHelper.SetAttribute ("TransmitDuration", TimeValue (Seconds (0.1)));
  // 7.96 dBm/Hz from 50 kW ERP transmit power, flat 8 MHz PSD spectrum assumed for this approximation 
  tvTransHelper.SetAttribute ("BasePsd", DoubleValue (7.96));
  tvTransHelper.SetAttribute ("TvType", EnumValue (TvSpectrumTransmitter::TVTYPE_COFDM));
  tvTransHelper.SetAttribute ("Antenna", StringValue ("ns3::IsotropicAntennaModel"));

  tvTransHelper.AssignStreams (300);
  tvTransHelper.CreateRegionalTvTransmitters(TvSpectrumTransmitterHelper::REGION_EUROPE, 
                                             TvSpectrumTransmitterHelper::DENSITY_MEDIUM, 
                                             48.86, 
                                             2.35, 
                                             0,
                                             250000);

  /* frequency range for spectrum analyzer */
  std::vector<double> freqs;
  for (int i = 0; i < 6860; i = i + 5)
    {
      freqs.push_back ((i + 1740) * 1e5);
    }
  Ptr<SpectrumModel> spectrumAnalyzerFreqModel = Create<SpectrumModel> (freqs);

  /* spectrum analyzer setup */
  SpectrumAnalyzerHelper spectrumAnalyzerHelper;
  spectrumAnalyzerHelper.SetChannel (channel);
  spectrumAnalyzerHelper.SetRxSpectrumModel (spectrumAnalyzerFreqModel);
  spectrumAnalyzerHelper.SetPhyAttribute ("NoisePowerSpectralDensity", DoubleValue (4.14e-21)); // approx -174 dBm/Hz
  spectrumAnalyzerHelper.EnableAsciiAll ("spectrum-analyzer-tv-sim-regional");
  NetDeviceContainer spectrumAnalyzerDevices = spectrumAnalyzerHelper.Install (spectrumAnalyzerNodes);

  Simulator::Stop (Seconds (0.4));

  Simulator::Run ();

  Simulator::Destroy ();

  std::cout << "simulation done!" << std::endl;
  std::cout << "see spectrum analyzer output file" << std::endl;

  return 0;
}
