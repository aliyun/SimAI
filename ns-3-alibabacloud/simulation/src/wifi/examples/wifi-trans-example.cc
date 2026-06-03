/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Orange Labs
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
 * Author: Rediet <getachew.redieteab@orange.com>
 */

#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include "ns3/string.h"
#include "ns3/ssid.h"
#include "ns3/spectrum-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/spectrum-analyzer-helper.h"
#include "ns3/spectrum-channel.h"
#include "ns3/mobility-helper.h"

using namespace ns3;

/**
 * This example (inspired from tv-trans-example) enables to generate the transmitted spectra of
 * Wi-Fi stations, so as to model transmit mask imperfections of OFDM-based Wi-Fi standards.
 * Only one data packet is sent from access point to station (once association has been performed)
 * so as to reduce execution time.
 *
 * A spectrum analyzer is used to measure the transmitted spectra from Wi-Fi stations.
 * The file "spectrum-analyzer-wifi-[standard]-[bandwidth]MHz-sim-2-0.tr" contains its
 * output post simulation and use it to plot transmitted spectra with Gnuplot.
 *
 * The wifi-trans-example.sh script runs this example for all combinations, plots transmitted spectra,
 * and puts resulting png images in wifi-trans-results folder.
 */

void
SendPacket (Ptr<NetDevice> sourceDevice, Address& destination)
{
  Ptr<Packet> pkt = Create<Packet> (100);  // dummy bytes of data
  sourceDevice->Send (pkt, destination, 0);
}

int main (int argc, char** argv)
{
  std::string standard = "11a";
  int bw = 20;
  double pow = 23; //dBm
  bool verbose = false;
  CommandLine cmd (__FILE__);
  cmd.AddValue ("standard",
                "OFDM-based Wi-Fi standard [11a, 11p_10MHZ, 11p_5MHZ, 11n_2_4GHZ, 11n_5GHZ, 11ac, 11ax_2_4GHZ, 11ax_5GHZ]",
                standard);
  cmd.AddValue ("bw", "Bandwidth (consistent with standard, in MHz)", bw);
  cmd.AddValue ("txPower", "Transmit power (dBm)", pow);
  cmd.AddValue ("verbose", "Display log messages for WifiSpectrumValueHelper and SpectrumWifiPhy", verbose);
  cmd.Parse (argc,argv);

  WifiHelper wifi;
  Ssid ssid;
  std::string dataRate;
  int freq;
  Time dataStartTime = MicroSeconds (800); // leaving enough time for beacon and association procedure
  Time dataDuration = MicroSeconds (300); // leaving enough time for data transfer (+ acknowledgment)
  if (standard == "11a")
    {
      wifi.SetStandard (WIFI_STANDARD_80211a);
      ssid = Ssid ("ns380211a");
      dataRate = "OfdmRate6Mbps";
      freq = 5180;
      if (bw != 20)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11p_10MHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211p);
      ssid = Ssid ("ns380211p_10MHZ");
      dataRate = "OfdmRate3MbpsBW10MHz";
      freq = 5860;
      dataStartTime = MicroSeconds (1400);
      dataDuration = MicroSeconds (600);
      if (bw != 10)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11p_5MHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211p);
      ssid = Ssid ("ns380211p_5MHZ");
      dataRate = "OfdmRate1_5MbpsBW5MHz";
      freq = 5860;
      dataStartTime = MicroSeconds (2500);
      dataDuration = MicroSeconds (1200);
      if (bw != 5)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11n_2_4GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211n);
      ssid = Ssid ("ns380211n_2_4GHZ");
      dataRate = "HtMcs0";
      freq = 2402 + (bw / 2); //so as to have 2412/2422 for 20/40
      dataStartTime = MicroSeconds (4700);
      dataDuration = MicroSeconds (400);
      if (bw != 20 && bw != 40)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11n_5GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211n);
      ssid = Ssid ("ns380211n_5GHZ");
      dataRate = "HtMcs0";
      freq = 5170 + (bw / 2); //so as to have 5180/5190 for 20/40
      dataStartTime = MicroSeconds (1000);
      if (bw != 20 && bw != 40)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11ac")
    {
      wifi.SetStandard (WIFI_STANDARD_80211ac);
      ssid = Ssid ("ns380211ac");
      dataRate = "VhtMcs0";
      freq = 5170 + (bw / 2); //so as to have 5180/5190/5210/5250 for 20/40/80/160
      dataStartTime = MicroSeconds (1100);
      dataDuration += MicroSeconds (400); //account for ADDBA procedure
      if (bw != 20 && bw != 40 && bw != 80 && bw != 160)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11ax_2_4GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211ax);
      ssid = Ssid ("ns380211ax_2_4GHZ");
      dataRate = "HeMcs0";
      freq = 2402 + (bw / 2); //so as to have 2412/2422/2442 for 20/40/80
      dataStartTime = MicroSeconds (5500);
      dataDuration += MicroSeconds (2000); //account for ADDBA procedure
      if (bw != 20 && bw != 40 && bw != 80)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else if (standard == "11ax_5GHZ")
    {
      wifi.SetStandard (WIFI_STANDARD_80211ax);
      ssid = Ssid ("ns380211ax_5GHZ");
      dataRate = "HeMcs0";
      freq = 5170 + (bw / 2); //so as to have 5180/5190/5210/5250 for 20/40/80/160
      dataStartTime = MicroSeconds (1200);
      dataDuration += MicroSeconds (500); //account for ADDBA procedure
      if (bw != 20 && bw != 40 && bw != 80 && bw != 160)
        {
          std::cout << "Bandwidth is not compatible with standard" << std::endl;
          return 1;
        }
    }
  else
    {
      std::cout << "Unknown OFDM standard (please refer to the listed possible values)" << std::endl;
      return 1;
    }

  if (verbose)
    {
      LogComponentEnableAll (LOG_PREFIX_ALL);
      LogComponentEnable ("WifiSpectrumValueHelper", LOG_LEVEL_ALL);
      LogComponentEnable ("SpectrumWifiPhy", LOG_LEVEL_ALL);
    }

  /* nodes and positions */
  NodeContainer wifiNodes;
  NodeContainer spectrumAnalyzerNodes;
  NodeContainer allNodes;
  wifiNodes.Create (2);
  spectrumAnalyzerNodes.Create (1);
  allNodes.Add (wifiNodes);
  allNodes.Add (spectrumAnalyzerNodes);
  NodeContainer wifiStaNode;
  NodeContainer wifiApNode;
  wifiApNode.Add (wifiNodes.Get (0));
  wifiStaNode.Add (wifiNodes.Get (1));

  /* channel and propagation */
  SpectrumChannelHelper channelHelper = SpectrumChannelHelper::Default ();
  channelHelper.SetChannel ("ns3::MultiModelSpectrumChannel");
  // constant path loss added just to show capability to set different propagation loss models
  // FriisSpectrumPropagationLossModel already added by default in SpectrumChannelHelper
  channelHelper.AddSpectrumPropagationLoss ("ns3::ConstantSpectrumPropagationLossModel");
  Ptr<SpectrumChannel> channel = channelHelper.Create ();

  /* Wi-Fi transmitter setup */

  SpectrumWifiPhyHelper spectrumPhy;
  spectrumPhy.SetChannel (channel);
  spectrumPhy.SetErrorRateModel ("ns3::NistErrorRateModel");
  spectrumPhy.Set ("Frequency", UintegerValue (freq));
  spectrumPhy.Set ("ChannelWidth", UintegerValue (bw));
  spectrumPhy.Set ("TxPowerStart", DoubleValue (pow)); // dBm
  spectrumPhy.Set ("TxPowerEnd", DoubleValue (pow));

  WifiMacHelper mac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", StringValue (dataRate),
                                "ControlMode", StringValue (dataRate));

  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevice = wifi.Install (spectrumPhy, mac, wifiStaNode);
  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid),
               "EnableBeaconJitter", BooleanValue (false)); // so as to be sure that first beacon arrives quickly
  NetDeviceContainer apDevice = wifi.Install (spectrumPhy, mac, wifiApNode);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> nodePositionList = CreateObject<ListPositionAllocator> ();
  nodePositionList->Add (Vector (0.0, 1.0, 0.0)); // AP
  nodePositionList->Add (Vector (1.0, 0.0, 0.0)); // STA
  nodePositionList->Add (Vector (0.0, 0.0, 0.0));  // Spectrum Analyzer
  mobility.SetPositionAllocator (nodePositionList);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (allNodes);

  /* Need to send data packet because beacon and association frames shall be sent using lowest rate */
  // Send one data packet (this packet is sent using data rate / MCS defined above) once association is done (otherwise dropped)
  Simulator::Schedule (dataStartTime, &SendPacket, apDevice.Get (0), staDevice.Get (0)->GetAddress ());

  /* frequency range for spectrum analyzer */
  std::vector<double> freqs;
  int margin = 2; //1MHz margin on each side
  int band = (bw + margin);
  for (int i = 0; i < (4 * 10 * band); ++i) //conversion to 100kHz scale
    {
      freqs.push_back (i * 1e5 + (freq - 2 * band) * 1e6);
    }
  Ptr<SpectrumModel> spectrumAnalyzerFreqModel = Create<SpectrumModel> (freqs);

  /* spectrum analyzer setup */
  SpectrumAnalyzerHelper spectrumAnalyzerHelper;
  spectrumAnalyzerHelper.SetChannel (channel);
  spectrumAnalyzerHelper.SetRxSpectrumModel (spectrumAnalyzerFreqModel);
  spectrumAnalyzerHelper.SetPhyAttribute ("Resolution", TimeValue (MicroSeconds (4))); //enough resolution to distinguish OFDM symbols (default 1ms too long even for PPDUs)
  std::ostringstream ossFileName;
  ossFileName << "spectrum-analyzer-wifi-" << standard << "-" << bw << "MHz";
  spectrumAnalyzerHelper.EnableAsciiAll (ossFileName.str ());
  NetDeviceContainer spectrumAnalyzerDevices = spectrumAnalyzerHelper.Install (spectrumAnalyzerNodes);

  /* Let enough time for first beacon, association procedure, and first data (+acknowledgment and eventually preceding ADDBA procedure) */
  Simulator::Stop (dataStartTime + dataDuration);

  Simulator::Run ();

  /* Plot transmitted spectra with Gnuplot */
  ossFileName << "-2-0"; // append node-interface info
  std::ostringstream ossPlt;
  ossPlt << ossFileName.str () << ".plt";
  std::ofstream plotFile (ossPlt.str ());
  std::ostringstream ossPng;
  ossPng << ossFileName.str () << ".png";
  Gnuplot plot = Gnuplot (ossPng.str ());
  //Prepare 3D plot (reset previous values)
  std::ostringstream ossExtra;
  ossExtra << "file = '" << ossFileName.str () << "'";
  plot.SetExtra (ossExtra.str ());
  plot.AppendExtra ("unset surface");
  plot.AppendExtra ("set key off");
  //Configure output file as png
  plot.AppendExtra ("set term png");
  plot.AppendExtra ("set output file . '.png'");
  //Switch to 3D plot
  plot.AppendExtra ("set pm3d at s");
  plot.AppendExtra ("set palette");
  //Orient view
  plot.AppendExtra ("set view 50,50");
  //Add legends
  plot.AppendExtra ("set xlabel \"time (ms)\"");
  plot.AppendExtra ("set ylabel \"freq (MHz)\" offset 15,0,0");
  plot.AppendExtra ("set zlabel \"PSD (dBW/Hz)\" offset 15,0,0");
  //Define grid
  plot.AppendExtra ("set ytics");
  plot.AppendExtra ("set mytics 2");
  plot.AppendExtra ("set ztics");
  plot.AppendExtra ("set mztics 5");
  plot.AppendExtra ("set grid ytics mytics ztics mztics");
  //tr file name
  plot.AppendExtra ("filename = file . '.tr'");
  //Extract max power using stats (so as to normalize during display)
  plot.AppendExtra ("stats filename using 3");
  plot.AppendExtra ("refW = STATS_max");
  //Plot graph (file being defined upon gnuplot call)
  plot.AppendExtra ("splot filename using ($1*1000.0):($2/1e6):(10*log10($3/refW))");
  //Generate output and close file
  plot.GenerateOutput (plotFile);
  plotFile.close ();

  Simulator::Destroy ();

  std::cout << "Simulation done!" << std::endl;
  std::cout << "See spectrum analyzer output file: " << ossFileName.str () << ".tr" << std::endl;
  std::cout << "To generate plot simply execute the following command: gnuplot " << ossFileName.str () << ".plt" << std::endl;

  return 0;
}
