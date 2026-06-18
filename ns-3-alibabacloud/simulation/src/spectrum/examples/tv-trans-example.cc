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
 * This example uses the TvSpectrumTransmitterHelper class to set up two 8-VSB
 * TV transmitters with adjacent channels. Each transmitter's spectrum has a
 * bandwidth of 6 MHz. The first TV transmitter has a start frequency of
 * 524 MHz while the second has a start frequency of 530 MHz. These transmitters
 * model ATSC (North American digital TV standard) channels 23 and 24.
 *
 * A spectrum analyzer is used to measure the transmitted spectra from the
 * TV transmitters. The file "spectrum-analyzer-tv-sim-2-0.tr" contains its
 * output post simulation (and can be plotted with Gnuplot or MATLAB).
 */
int main (int argc, char** argv)
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  /* nodes and positions */
  NodeContainer tvTransmitterNodes;
  NodeContainer spectrumAnalyzerNodes;
  NodeContainer allNodes;
  tvTransmitterNodes.Create (2);
  spectrumAnalyzerNodes.Create (1);
  allNodes.Add (tvTransmitterNodes);
  allNodes.Add (spectrumAnalyzerNodes);
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> nodePositionList = CreateObject<ListPositionAllocator> ();
  nodePositionList->Add (Vector (128000.0, 0.0, 0.0)); // TV Transmitter 1; 128 km away from spectrum analyzer 
  nodePositionList->Add (Vector (0.0, 24000.0, 0.0)); // TV Transmitter 2; 24 km away from spectrum analyzer
  nodePositionList->Add (Vector (0.0, 0.0, 0.0));  // Spectrum Analyzer
  mobility.SetPositionAllocator (nodePositionList);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (allNodes);

  /* channel and propagation */
  SpectrumChannelHelper channelHelper = SpectrumChannelHelper::Default ();
  channelHelper.SetChannel ("ns3::MultiModelSpectrumChannel");
  // constant path loss added just to show capability to set different propagation loss models
  // FriisSpectrumPropagationLossModel already added by default in SpectrumChannelHelper
  channelHelper.AddSpectrumPropagationLoss ("ns3::ConstantSpectrumPropagationLossModel");
  Ptr<SpectrumChannel> channel = channelHelper.Create ();
  
  /* TV transmitter setup */
  TvSpectrumTransmitterHelper tvTransHelper;
  tvTransHelper.SetChannel (channel);
  tvTransHelper.SetAttribute ("StartFrequency", DoubleValue (524e6));
  tvTransHelper.SetAttribute ("ChannelBandwidth", DoubleValue (6e6));
  tvTransHelper.SetAttribute ("StartingTime", TimeValue (Seconds (0)));
  tvTransHelper.SetAttribute ("TransmitDuration", TimeValue (Seconds (0.2)));
  // 22.22 dBm/Hz from 1000 kW ERP transmit power, flat 6 MHz PSD spectrum assumed for this approximation 
  tvTransHelper.SetAttribute ("BasePsd", DoubleValue (22.22)); 
  tvTransHelper.SetAttribute ("TvType", EnumValue (TvSpectrumTransmitter::TVTYPE_8VSB));
  tvTransHelper.SetAttribute ("Antenna", StringValue ("ns3::IsotropicAntennaModel"));
  tvTransHelper.InstallAdjacent (tvTransmitterNodes);

  /* frequency range for spectrum analyzer */
  std::vector<double> freqs;
  for (int i = 0; i < 200; ++i)
    {
      freqs.push_back ((i + 5200) * 1e5);
    }
  Ptr<SpectrumModel> spectrumAnalyzerFreqModel = Create<SpectrumModel> (freqs);

  /* spectrum analyzer setup */
  SpectrumAnalyzerHelper spectrumAnalyzerHelper;
  spectrumAnalyzerHelper.SetChannel (channel);
  spectrumAnalyzerHelper.SetRxSpectrumModel (spectrumAnalyzerFreqModel);
  spectrumAnalyzerHelper.SetPhyAttribute ("NoisePowerSpectralDensity", DoubleValue (1e-15)); // -120 dBm/Hz
  spectrumAnalyzerHelper.EnableAsciiAll ("spectrum-analyzer-tv-sim");
  NetDeviceContainer spectrumAnalyzerDevices = spectrumAnalyzerHelper.Install (spectrumAnalyzerNodes);

  Simulator::Stop (Seconds (0.4));

  Simulator::Run ();

  Simulator::Destroy ();

  std::cout << "simulation done!" << std::endl;
  std::cout << "see spectrum analyzer output file" << std::endl;

  return 0;
}
