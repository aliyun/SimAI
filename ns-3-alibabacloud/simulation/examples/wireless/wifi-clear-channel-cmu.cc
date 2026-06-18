/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Boeing Company
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
 * Author: Guangyu Pei <guangyu.pei@boeing.com>
 */

#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/mobility-model.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ClearChannelCmu");

/**
 * WiFi clear channel cmu experiment class.
 * 
 * It handles the creation and run of an experiment.
 */
class Experiment
{
public:
  Experiment ();
  /**
   * Constructor.
   * \param name The name of the experiment.
   */
  Experiment (std::string name);
  /**
   * Run an experiment.
   * \param wifi      //!< The WifiHelper class.
   * \param wifiPhy   //!< The YansWifiPhyHelper class.
   * \param wifiMac   //!< The WifiMacHelper class.
   * \param wifiChannel //!< The YansWifiChannelHelper class.
   * \return the number of received packets. 
   */
  uint32_t Run (const WifiHelper &wifi, const YansWifiPhyHelper &wifiPhy,
                const WifiMacHelper &wifiMac, const YansWifiChannelHelper &wifiChannel);
private:
  /**
   * Receive a packet.
   * \param socket The recieving socket.
   */
  void ReceivePacket (Ptr<Socket> socket);
  /**
   * Set the position of a node.
   * \param node The node.
   * \param position The position of the node.
   */
  void SetPosition (Ptr<Node> node, Vector position);
  /**
   * Get the position of a node.
   * \param node The node.
   * \return the position of the node.
   */
  Vector GetPosition (Ptr<Node> node);
  /**
   * Setup the receiving socket.
   * \param node The receiving node.
   * \return the socket.
   */
  Ptr<Socket> SetupPacketReceive (Ptr<Node> node);
  /**
   * Generate the traffic.
   * \param socket The sending socket.
   * \param pktSize The packet size.
   * \param pktCount The number of packets to send.
   * \param pktInterval The time between packets.
   */
  void GenerateTraffic (Ptr<Socket> socket, uint32_t pktSize,
                        uint32_t pktCount, Time pktInterval );

  uint32_t m_pktsTotal;       //!< Total number of received packets
  Gnuplot2dDataset m_output;  //!< Output dataset.
};

Experiment::Experiment ()
{
}

Experiment::Experiment (std::string name)
  : m_output (name)
{
  m_output.SetStyle (Gnuplot2dDataset::LINES);
}

void
Experiment::SetPosition (Ptr<Node> node, Vector position)
{
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  mobility->SetPosition (position);
}

Vector
Experiment::GetPosition (Ptr<Node> node)
{
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  return mobility->GetPosition ();
}

void
Experiment::ReceivePacket (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  while ((packet = socket->Recv ()))
    {
      m_pktsTotal++;
    }
}

Ptr<Socket>
Experiment::SetupPacketReceive (Ptr<Node> node)
{
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> sink = Socket::CreateSocket (node, tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 80);
  sink->Bind (local);
  sink->SetRecvCallback (MakeCallback (&Experiment::ReceivePacket, this));
  return sink;
}

void
Experiment::GenerateTraffic (Ptr<Socket> socket, uint32_t pktSize,
                             uint32_t pktCount, Time pktInterval )
{
  if (pktCount > 0)
    {
      socket->Send (Create<Packet> (pktSize));
      Simulator::Schedule (pktInterval, &Experiment::GenerateTraffic, this,
                           socket, pktSize,pktCount - 1, pktInterval);
    }
  else
    {
      socket->Close ();
    }
}

uint32_t
Experiment::Run (const WifiHelper &wifi, const YansWifiPhyHelper &wifiPhy,
                 const WifiMacHelper &wifiMac, const YansWifiChannelHelper &wifiChannel)
{
  m_pktsTotal = 0;

  NodeContainer c;
  c.Create (2);

  InternetStackHelper internet;
  internet.Install (c);

  YansWifiPhyHelper phy = wifiPhy;
  phy.SetChannel (wifiChannel.Create ());

  WifiMacHelper mac = wifiMac;
  NetDeviceContainer devices = wifi.Install (phy, mac, c);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);

  Ipv4AddressHelper ipv4;
  NS_LOG_INFO ("Assign IP Addresses.");
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = ipv4.Assign (devices);

  Ptr<Socket> recvSink = SetupPacketReceive (c.Get (0));

  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> source = Socket::CreateSocket (c.Get (1), tid);
  InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
  source->SetAllowBroadcast (true);
  source->Connect (remote);
  uint32_t packetSize = 1014;
  uint32_t maxPacketCount = 200;
  Time interPacketInterval = Seconds (1.);
  Simulator::Schedule (Seconds (1.0), &Experiment::GenerateTraffic,
                       this, source, packetSize, maxPacketCount,interPacketInterval);
  Simulator::Run ();

  Simulator::Destroy ();

  return m_pktsTotal;
}

int main (int argc, char *argv[])
{
  std::ofstream outfile ("clear-channel.plt");
  std::vector <std::string> modes;

  modes.push_back ("DsssRate1Mbps");
  modes.push_back ("DsssRate2Mbps");
  modes.push_back ("DsssRate5_5Mbps");
  modes.push_back ("DsssRate11Mbps");

  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  Gnuplot gnuplot = Gnuplot ("clear-channel.eps");

  for (uint32_t i = 0; i < modes.size (); i++)
    {
      std::cout << modes[i] << std::endl;
      Gnuplot2dDataset dataset (modes[i]);

      for (double rss = -102.0; rss <= -80.0; rss += 0.5)
        {
          Experiment experiment;
          dataset.SetStyle (Gnuplot2dDataset::LINES);

          WifiHelper wifi;
          wifi.SetStandard (WIFI_STANDARD_80211b);
          WifiMacHelper wifiMac;
          Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                              StringValue (modes[i]));
          wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                        "DataMode",StringValue (modes[i]),
                                        "ControlMode",StringValue (modes[i]));
          wifiMac.SetType ("ns3::AdhocWifiMac");

          YansWifiPhyHelper wifiPhy;
          YansWifiChannelHelper wifiChannel;
          wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
          wifiChannel.AddPropagationLoss ("ns3::FixedRssLossModel","Rss",DoubleValue (rss));


          NS_LOG_DEBUG (modes[i]);
          experiment = Experiment (modes[i]);
          wifiPhy.Set ("TxPowerStart", DoubleValue (15.0) );
          wifiPhy.Set ("TxPowerEnd", DoubleValue (15.0) );
          wifiPhy.Set ("RxGain", DoubleValue (0) );
          wifiPhy.Set ("RxNoiseFigure", DoubleValue (7) );
          uint32_t pktsRecvd = experiment.Run (wifi, wifiPhy, wifiMac, wifiChannel);
          dataset.Add (rss, pktsRecvd);
        }

      gnuplot.AddDataset (dataset);
    }
  gnuplot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  gnuplot.SetLegend ("RSS(dBm)", "Number of packets received");
  gnuplot.SetExtra  ("set xrange [-102:-83]");
  gnuplot.GenerateOutput (outfile);
  outfile.close ();

  return 0;
}
