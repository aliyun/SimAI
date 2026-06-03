/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 University of Washington
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
 * Authors: Tom Henderson <tomhend@u.washington.edu>
 *          Matías Richart <mrichart@fing.edu.uy>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

// Test the operation of a wifi manager as the SNR is varied, and create
// a gnuplot output file for plotting.
//
// The test consists of a device acting as server and a device as client generating traffic.
//
// The output consists of a plot of the rate observed and selected at the client device.
//
// By default, the 802.11a standard using IdealWifiManager is plotted. Several command line
// arguments can change the following options:
// --wifiManager (Aarf, Aarfcd, Amrr, Arf, Cara, Ideal, Minstrel, MinstrelHt, Onoe, Rraa, ThompsonSampling)
// --standard (802.11a, 802.11b, 802.11g, 802.11p-10MHz, 802.11p-5MHz, 802.11n-5GHz, 802.11n-2.4GHz, 802.11ac, 802.11ax-6GHz, 802.11ax-5GHz, 802.11ax-2.4GHz)
// --serverShortGuardInterval and --clientShortGuardInterval (for 802.11n/ac)
// --serverNss and --clientNss (for 802.11n/ac)
// --serverChannelWidth and --clientChannelWidth (for 802.11n/ac)
// --broadcast instead of unicast (default is unicast)
// --rtsThreshold (by default, value of 99999 disables it)

#include "ns3/log.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/tuple.h"
#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/mobility-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-server.h"
#include "ns3/ht-configuration.h"
#include "ns3/he-configuration.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiManagerExample");

// 290K @ 20 MHz
const double NOISE_DBM_Hz = -174.0;
double noiseDbm = NOISE_DBM_Hz;

double g_intervalBytes = 0;
uint64_t g_intervalRate = 0;

void
PacketRx (Ptr<const Packet> pkt, const Address &addr)
{
  g_intervalBytes += pkt->GetSize ();
}

void
RateChange (uint64_t oldVal, uint64_t newVal)
{
  NS_LOG_DEBUG ("Change from " << oldVal << " to " << newVal);
  g_intervalRate = newVal;
}

/// Step structure
struct Step
{
  double stepSize; ///< step size in dBm
  double stepTime; ///< step size in seconds
};

/// StandardInfo structure
struct StandardInfo
{
  StandardInfo ()
  {
    m_name = "none";
  }
  /**
   * Constructor
   *
   * \param name reference name
   * \param standard wifi standard
   * \param band PHY band
   * \param width channel width
   * \param snrLow SNR low
   * \param snrHigh SNR high
   * \param xMin x minimum
   * \param xMax x maximum
   * \param yMax y maximum
   */
  StandardInfo (std::string name, WifiStandard standard, WifiPhyBand band, uint16_t width, double snrLow, double snrHigh, double xMin, double xMax, double yMax)
    : m_name (name),
      m_standard (standard),
      m_band (band),
      m_width (width),
      m_snrLow (snrLow),
      m_snrHigh (snrHigh),
      m_xMin (xMin),
      m_xMax (xMax),
      m_yMax (yMax)
  {
  }
  std::string m_name; ///< name
  WifiStandard m_standard; ///< standard
  WifiPhyBand m_band; ///< PHY band
  uint16_t m_width; ///< channel width
  double m_snrLow; ///< lowest SNR
  double m_snrHigh; ///< highest SNR
  double m_xMin;  ///< X minimum
  double m_xMax;  ///< X maximum
  double m_yMax;  ///< Y maximum
};

void
ChangeSignalAndReportRate (Ptr<FixedRssLossModel> rssModel, Step step, double rss, Gnuplot2dDataset& rateDataset, Gnuplot2dDataset& actualDataset)
{
  NS_LOG_FUNCTION (rssModel << step.stepSize << step.stepTime << rss);
  double snr = rss - noiseDbm;
  rateDataset.Add (snr, g_intervalRate / 1e6);
  // Calculate received rate since last interval
  double currentRate = ((g_intervalBytes * 8) / step.stepTime) / 1e6; // Mb/s
  actualDataset.Add (snr, currentRate);
  rssModel->SetRss (rss - step.stepSize);
  NS_LOG_INFO ("At time " << Simulator::Now ().As (Time::S) << "; selected rate " << (g_intervalRate / 1e6) << "; observed rate " << currentRate << "; setting new power to " << rss - step.stepSize);
  g_intervalBytes = 0;
  Simulator::Schedule (Seconds (step.stepTime), &ChangeSignalAndReportRate, rssModel, step, (rss - step.stepSize), rateDataset, actualDataset);
}

int main (int argc, char *argv[])
{
  std::vector <StandardInfo> serverStandards;
  std::vector <StandardInfo> clientStandards;
  uint32_t steps;
  uint32_t rtsThreshold = 999999;  // disabled even for large A-MPDU
  uint32_t maxAmpduSize = 65535;
  double stepSize = 1; // dBm
  double stepTime = 1; // seconds
  uint32_t packetSize = 1024; // bytes
  bool broadcast = 0;
  int ap1_x = 0;
  int ap1_y = 0;
  int sta1_x = 5;
  int sta1_y = 0;
  uint16_t serverNss = 1;
  uint16_t clientNss = 1;
  uint16_t serverShortGuardInterval = 800;
  uint16_t clientShortGuardInterval = 800;
  uint16_t serverChannelWidth = 0;  // use default for standard and band
  uint16_t clientChannelWidth = 0;  // use default for standard and band
  std::string wifiManager ("Ideal");
  std::string standard ("802.11a");
  StandardInfo serverSelectedStandard;
  StandardInfo clientSelectedStandard;
  bool infrastructure = false;
  uint32_t maxSlrc = 7;
  uint32_t maxSsrc = 7;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("maxSsrc", "The maximum number of retransmission attempts for a RTS packet", maxSsrc);
  cmd.AddValue ("maxSlrc", "The maximum number of retransmission attempts for a Data packet", maxSlrc);
  cmd.AddValue ("rtsThreshold", "RTS threshold", rtsThreshold);
  cmd.AddValue ("maxAmpduSize", "Max A-MPDU size", maxAmpduSize);
  cmd.AddValue ("stepSize", "Power between steps (dBm)", stepSize);
  cmd.AddValue ("stepTime", "Time on each step (seconds)", stepTime);
  cmd.AddValue ("broadcast", "Send broadcast instead of unicast", broadcast);
  cmd.AddValue ("serverChannelWidth", "Set channel width of the server (valid only for 802.11n or ac)", serverChannelWidth);
  cmd.AddValue ("clientChannelWidth", "Set channel width of the client (valid only for 802.11n or ac)", clientChannelWidth);
  cmd.AddValue ("serverNss", "Set nss of the server (valid only for 802.11n or ac)", serverNss);
  cmd.AddValue ("clientNss", "Set nss of the client (valid only for 802.11n or ac)", clientNss);
  cmd.AddValue ("serverShortGuardInterval", "Set short guard interval of the server (802.11n/ac/ax) in nanoseconds", serverShortGuardInterval);
  cmd.AddValue ("clientShortGuardInterval", "Set short guard interval of the client (802.11n/ac/ax) in nanoseconds", clientShortGuardInterval);
  cmd.AddValue ("standard", "Set standard (802.11a, 802.11b, 802.11g, 802.11p-10MHz, 802.11p-5MHz, 802.11n-5GHz, 802.11n-2.4GHz, 802.11ac, 802.11ax-6GHz, 802.11ax-5GHz, 802.11ax-2.4GHz)", standard);
  cmd.AddValue ("wifiManager", "Set wifi rate manager (Aarf, Aarfcd, Amrr, Arf, Cara, Ideal, Minstrel, MinstrelHt, Onoe, Rraa, ThompsonSampling)", wifiManager);
  cmd.AddValue ("infrastructure", "Use infrastructure instead of adhoc", infrastructure);
  cmd.Parse (argc,argv);

  // Print out some explanation of what this program does
  std::cout << std::endl << "This program demonstrates and plots the operation of different " << std::endl;
  std::cout << "Wi-Fi rate controls on different station configurations," << std::endl;
  std::cout << "by stepping down the received signal strength across a wide range" << std::endl;
  std::cout << "and observing the adjustment of the rate." << std::endl;
  std::cout << "Run 'wifi-manager-example --PrintHelp' to show program options." << std::endl << std::endl;

  if (infrastructure == false)
    {
      NS_ABORT_MSG_IF (serverNss != clientNss, "In ad hoc mode, we assume sender and receiver are similarly configured");
    }

  if (standard == "802.11b")
    {
      if (serverChannelWidth == 0)
        {
          serverChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211b, WIFI_PHY_BAND_2_4GHZ);
        }
      NS_ABORT_MSG_IF (serverChannelWidth != 22 && serverChannelWidth != 22, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (serverNss != 1, "Invalid nss for standard " << standard);
      if (clientChannelWidth == 0)
        {
          clientChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211b, WIFI_PHY_BAND_2_4GHZ);
        }
      NS_ABORT_MSG_IF (clientChannelWidth != 22 && clientChannelWidth != 22, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (clientNss != 1, "Invalid nss for standard " << standard);
    }
  else if (standard == "802.11a" || standard == "802.11g")
    {
      if (serverChannelWidth == 0)
        {
          serverChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211g, WIFI_PHY_BAND_2_4GHZ);
        }
      NS_ABORT_MSG_IF (serverChannelWidth != 20, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (serverNss != 1, "Invalid nss for standard " << standard);
      if (clientChannelWidth == 0)
        {
          clientChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211g, WIFI_PHY_BAND_2_4GHZ);
        }
      NS_ABORT_MSG_IF (clientChannelWidth != 20, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (clientNss != 1, "Invalid nss for standard " << standard);
    }
  else if (standard == "802.11n-5GHz" || standard == "802.11n-2.4GHz")
    {
      WifiPhyBand band = (standard == "802.11n-2.4GHz" ? WIFI_PHY_BAND_2_4GHZ : WIFI_PHY_BAND_5GHZ);
      if (serverChannelWidth == 0)
        {
          serverChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211n, band);
        }
      NS_ABORT_MSG_IF (serverChannelWidth != 20 && serverChannelWidth != 40, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (serverNss == 0 || serverNss > 4, "Invalid nss " << serverNss << " for standard " << standard);
      if (clientChannelWidth == 0)
        {
          clientChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211n, band);
        }
      NS_ABORT_MSG_IF (clientChannelWidth != 20 && clientChannelWidth != 40, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (clientNss == 0 || clientNss > 4, "Invalid nss " << clientNss << " for standard " << standard);
    }
  else if (standard == "802.11ac")
    {
      if (serverChannelWidth == 0)
        {
          serverChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ);
        }
      NS_ABORT_MSG_IF (serverChannelWidth != 20 && serverChannelWidth != 40 && serverChannelWidth != 80 && serverChannelWidth != 160, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (serverNss == 0 || serverNss > 4, "Invalid nss " << serverNss << " for standard " << standard);
      if (clientChannelWidth == 0)
        {
          clientChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ);
        }
      NS_ABORT_MSG_IF (clientChannelWidth != 20 && clientChannelWidth != 40 && clientChannelWidth != 80 && clientChannelWidth != 160, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (clientNss == 0 || clientNss > 4, "Invalid nss " << clientNss << " for standard " << standard);
    }
  else if (standard == "802.11ax-6GHz" ||standard == "802.11ax-5GHz" || standard == "802.11ax-2.4GHz")
    {
      WifiPhyBand band = (standard == "802.11ax-2.4GHz" ? WIFI_PHY_BAND_2_4GHZ : standard == "802.11ax-6GHz" ? WIFI_PHY_BAND_6GHZ : WIFI_PHY_BAND_5GHZ);
      if (serverChannelWidth == 0)
        {
          serverChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211ax, band);
        }
      NS_ABORT_MSG_IF (serverChannelWidth != 20 && serverChannelWidth != 40 && serverChannelWidth != 80 && serverChannelWidth != 160, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (serverNss == 0 || serverNss > 4, "Invalid nss " << serverNss << " for standard " << standard);
      if (clientChannelWidth == 0)
        {
          clientChannelWidth = GetDefaultChannelWidth (WIFI_STANDARD_80211ax, band);
        }
      NS_ABORT_MSG_IF (clientChannelWidth != 20 && clientChannelWidth != 40 && clientChannelWidth != 80 && clientChannelWidth != 160, "Invalid channel width for standard " << standard);
      NS_ABORT_MSG_IF (clientNss == 0 || clientNss > 4, "Invalid nss " << clientNss << " for standard " << standard);
    }

  // As channel width increases, scale up plot's yRange value
  uint32_t channelRateFactor = std::max (clientChannelWidth, serverChannelWidth) / 20;
  channelRateFactor = channelRateFactor * std::max (clientNss, serverNss);

  // The first number is channel width, second is minimum SNR, third is maximum
  // SNR, fourth and fifth provide xrange axis limits, and sixth the yaxis
  // maximum
  serverStandards.push_back (StandardInfo ("802.11a", WIFI_STANDARD_80211a, WIFI_PHY_BAND_5GHZ, 20, 3, 27, 0, 30, 60));
  serverStandards.push_back (StandardInfo ("802.11b", WIFI_STANDARD_80211b, WIFI_PHY_BAND_2_4GHZ, 22, -5, 11, -6, 15, 15));
  serverStandards.push_back (StandardInfo ("802.11g", WIFI_STANDARD_80211g, WIFI_PHY_BAND_2_4GHZ, 20, -5, 27, -6, 30, 60));
  serverStandards.push_back (StandardInfo ("802.11n-5GHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_5GHZ,  serverChannelWidth, 3, 30, 0, 35, 80 * channelRateFactor));
  serverStandards.push_back (StandardInfo ("802.11n-2.4GHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_2_4GHZ, serverChannelWidth, 3, 30, 0, 35, 80 * channelRateFactor));
  serverStandards.push_back (StandardInfo ("802.11ac", WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ, serverChannelWidth, 5, 50, 0, 55, 120 * channelRateFactor));
  serverStandards.push_back (StandardInfo ("802.11p-10MHz", WIFI_STANDARD_80211p, WIFI_PHY_BAND_5GHZ, 10, 3, 27, 0, 30, 60));
  serverStandards.push_back (StandardInfo ("802.11p-5MHz", WIFI_STANDARD_80211p, WIFI_PHY_BAND_5GHZ, 5, 3, 27, 0, 30, 60));
  serverStandards.push_back (StandardInfo ("802.11ax-6GHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_6GHZ, serverChannelWidth, 5, 55, 0, 60, 120 * channelRateFactor));
  serverStandards.push_back (StandardInfo ("802.11ax-5GHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_5GHZ, serverChannelWidth, 5, 55, 0, 60, 120 * channelRateFactor));
  serverStandards.push_back (StandardInfo ("802.11ax-2.4GHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_2_4GHZ, serverChannelWidth, 5, 55, 0, 60, 120 * channelRateFactor));

  clientStandards.push_back (StandardInfo ("802.11a", WIFI_STANDARD_80211a, WIFI_PHY_BAND_5GHZ, 20, 3, 27, 0, 30, 60));
  clientStandards.push_back (StandardInfo ("802.11b", WIFI_STANDARD_80211b, WIFI_PHY_BAND_2_4GHZ, 22, -5, 11, -6, 15, 15));
  clientStandards.push_back (StandardInfo ("802.11g", WIFI_STANDARD_80211g, WIFI_PHY_BAND_2_4GHZ, 20, -5, 27, -6, 30, 60));
  clientStandards.push_back (StandardInfo ("802.11n-5GHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_5GHZ,  clientChannelWidth, 3, 30, 0, 35, 80 * channelRateFactor));
  clientStandards.push_back (StandardInfo ("802.11n-2.4GHz", WIFI_STANDARD_80211n, WIFI_PHY_BAND_2_4GHZ, clientChannelWidth, 3, 30, 0, 35, 80 * channelRateFactor));
  clientStandards.push_back (StandardInfo ("802.11ac", WIFI_STANDARD_80211ac, WIFI_PHY_BAND_5GHZ, clientChannelWidth, 5, 50, 0, 55, 120 * channelRateFactor));
  clientStandards.push_back (StandardInfo ("802.11p-10MHz", WIFI_STANDARD_80211p, WIFI_PHY_BAND_5GHZ, 10, 3, 27, 0, 30, 60));
  clientStandards.push_back (StandardInfo ("802.11p-5MHz", WIFI_STANDARD_80211p, WIFI_PHY_BAND_5GHZ, 5, 3, 27, 0, 30, 60));
  clientStandards.push_back (StandardInfo ("802.11ax-6GHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_6GHZ, clientChannelWidth, 5, 55, 0, 60, 160 * channelRateFactor));
  clientStandards.push_back (StandardInfo ("802.11ax-5GHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_5GHZ, clientChannelWidth, 5, 55, 0, 60, 160 * channelRateFactor));
  clientStandards.push_back (StandardInfo ("802.11ax-2.4GHz", WIFI_STANDARD_80211ax, WIFI_PHY_BAND_2_4GHZ, clientChannelWidth, 5, 55, 0, 60, 160 * channelRateFactor));

  for (std::vector<StandardInfo>::size_type i = 0; i != serverStandards.size (); i++)
    {
      if (standard == serverStandards[i].m_name)
        {
          serverSelectedStandard = serverStandards[i];
        }
    }
  for (std::vector<StandardInfo>::size_type i = 0; i != clientStandards.size (); i++)
    {
      if (standard == clientStandards[i].m_name)
        {
          clientSelectedStandard = clientStandards[i];
        }
    }

  NS_ABORT_MSG_IF (serverSelectedStandard.m_name == "none", "Standard " << standard << " not found");
  NS_ABORT_MSG_IF (clientSelectedStandard.m_name == "none", "Standard " << standard << " not found");
  std::cout << "Testing " << serverSelectedStandard.m_name << " with " << wifiManager << " ..." << std::endl;
  NS_ABORT_MSG_IF (clientSelectedStandard.m_snrLow >= clientSelectedStandard.m_snrHigh, "SNR values in wrong order");
  steps = static_cast<uint32_t> (std::abs (static_cast<double> (clientSelectedStandard.m_snrHigh - clientSelectedStandard.m_snrLow ) / stepSize) + 1);
  NS_LOG_DEBUG ("Using " << steps << " steps for SNR range " << clientSelectedStandard.m_snrLow << ":" << clientSelectedStandard.m_snrHigh);
  Ptr<Node> clientNode = CreateObject<Node> ();
  Ptr<Node> serverNode = CreateObject<Node> ();

  std::string plotName = "wifi-manager-example-";
  std::string dataName = "wifi-manager-example-";
  plotName += wifiManager;
  dataName += wifiManager;
  plotName += "-";
  dataName += "-";
  plotName += standard;
  dataName += standard;
  if (standard == "802.11n-5GHz"
      || standard == "802.11n-2.4GHz"
      || standard == "802.11ac"
      || standard == "802.11ax-6GHz"
      || standard == "802.11ax-5GHz"
      || standard == "802.11ax-2.4GHz")
    {
      plotName += "-server_";
      dataName += "-server_";
      std::ostringstream oss;
      oss << serverChannelWidth << "MHz_" << serverShortGuardInterval << "ns_" << serverNss << "SS";
      plotName += oss.str ();
      dataName += oss.str ();
      plotName += "-client_";
      dataName += "-client_";
      oss.str ("");
      oss << clientChannelWidth << "MHz_" << clientShortGuardInterval << "ns_" << clientNss << "SS";
      plotName += oss.str ();
      dataName += oss.str ();
    }
  plotName += ".eps";
  dataName += ".plt";
  std::ofstream outfile (dataName.c_str ());
  Gnuplot gnuplot = Gnuplot (plotName);

  Config::SetDefault ("ns3::WifiRemoteStationManager::MaxSlrc", UintegerValue (maxSlrc));
  Config::SetDefault ("ns3::WifiRemoteStationManager::MaxSsrc", UintegerValue (maxSsrc));
  Config::SetDefault ("ns3::MinstrelWifiManager::PrintStats", BooleanValue (true));
  Config::SetDefault ("ns3::MinstrelWifiManager::PrintSamples", BooleanValue (true));
  Config::SetDefault ("ns3::MinstrelHtWifiManager::PrintStats", BooleanValue (true));

  WifiHelper wifi;
  wifi.SetStandard (serverSelectedStandard.m_standard);
  YansWifiPhyHelper wifiPhy;

  Ptr<YansWifiChannel> wifiChannel = CreateObject<YansWifiChannel> ();
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  wifiChannel->SetPropagationDelayModel (delayModel);
  Ptr<FixedRssLossModel> rssLossModel = CreateObject<FixedRssLossModel> ();
  wifiChannel->SetPropagationLossModel (rssLossModel);
  wifiPhy.SetChannel (wifiChannel);

  wifi.SetRemoteStationManager ("ns3::" + wifiManager + "WifiManager", "RtsCtsThreshold", UintegerValue (rtsThreshold));

  NetDeviceContainer serverDevice;
  NetDeviceContainer clientDevice;

  TupleValue<UintegerValue, UintegerValue, EnumValue, UintegerValue> channelValue;

  WifiMacHelper wifiMac;
  if (infrastructure)
    {
      Ssid ssid = Ssid ("ns-3-ssid");
      wifiMac.SetType ("ns3::StaWifiMac",
                       "Ssid", SsidValue (ssid));
      channelValue.Set (WifiPhy::ChannelTuple {0, serverSelectedStandard.m_width,
                                               serverSelectedStandard.m_band, 0});
      wifiPhy.Set ("ChannelSettings", channelValue);
      serverDevice = wifi.Install (wifiPhy, wifiMac, serverNode);

      wifiMac.SetType ("ns3::ApWifiMac",
                       "Ssid", SsidValue (ssid));
      channelValue.Set (WifiPhy::ChannelTuple {0, clientSelectedStandard.m_width,
                                               clientSelectedStandard.m_band, 0});
      clientDevice = wifi.Install (wifiPhy, wifiMac, clientNode);
    }
  else
    {
      wifiMac.SetType ("ns3::AdhocWifiMac");
      channelValue.Set (WifiPhy::ChannelTuple {0, serverSelectedStandard.m_width,
                                               serverSelectedStandard.m_band, 0});
      wifiPhy.Set ("ChannelSettings", channelValue);
      serverDevice = wifi.Install (wifiPhy, wifiMac, serverNode);

      channelValue.Set (WifiPhy::ChannelTuple {0, clientSelectedStandard.m_width,
                                               clientSelectedStandard.m_band, 0});
      clientDevice = wifi.Install (wifiPhy, wifiMac, clientNode);
    }

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (2);
  wifi.AssignStreams (serverDevice, 100);
  wifi.AssignStreams (clientDevice, 100);

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_MaxAmpduSize", UintegerValue (maxAmpduSize));

  Config::ConnectWithoutContext ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$ns3::" + wifiManager + "WifiManager/Rate", MakeCallback (&RateChange));

  // Configure the mobility.
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  //Initial position of AP and STA
  positionAlloc->Add (Vector (ap1_x, ap1_y, 0.0));
  NS_LOG_INFO ("Setting initial AP position to " << Vector (ap1_x, ap1_y, 0.0));
  positionAlloc->Add (Vector (sta1_x, sta1_y, 0.0));
  NS_LOG_INFO ("Setting initial STA position to " << Vector (sta1_x, sta1_y, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (clientNode);
  mobility.Install (serverNode);

  Gnuplot2dDataset rateDataset (clientSelectedStandard.m_name + std::string ("-rate selected"));
  Gnuplot2dDataset actualDataset (clientSelectedStandard.m_name + std::string ("-observed"));
  Step step;
  step.stepSize = stepSize;
  step.stepTime = stepTime;

  // Perform post-install configuration from defaults for channel width,
  // guard interval, and nss, if necessary
  // Obtain pointer to the WifiPhy
  Ptr<NetDevice> ndClient = clientDevice.Get (0);
  Ptr<NetDevice> ndServer = serverDevice.Get (0);
  Ptr<WifiNetDevice> wndClient = ndClient->GetObject<WifiNetDevice> ();
  Ptr<WifiNetDevice> wndServer = ndServer->GetObject<WifiNetDevice> ();
  Ptr<WifiPhy> wifiPhyPtrClient = wndClient->GetPhy ();
  Ptr<WifiPhy> wifiPhyPtrServer = wndServer->GetPhy ();
  uint8_t t_clientNss = static_cast<uint8_t> (clientNss);
  uint8_t t_serverNss = static_cast<uint8_t> (serverNss);
  wifiPhyPtrClient->SetNumberOfAntennas (t_clientNss);
  wifiPhyPtrClient->SetMaxSupportedTxSpatialStreams (t_clientNss);
  wifiPhyPtrClient->SetMaxSupportedRxSpatialStreams (t_clientNss);
  wifiPhyPtrServer->SetNumberOfAntennas (t_serverNss);
  wifiPhyPtrServer->SetMaxSupportedTxSpatialStreams (t_serverNss);
  wifiPhyPtrServer->SetMaxSupportedRxSpatialStreams (t_serverNss);
  // Only set the guard interval for HT and VHT modes
  if (serverSelectedStandard.m_name == "802.11n-5GHz"
      || serverSelectedStandard.m_name == "802.11n-2.4GHz"
      || serverSelectedStandard.m_name == "802.11ac")
    {
      Ptr<HtConfiguration> clientHtConfiguration = wndClient->GetHtConfiguration ();
      clientHtConfiguration->SetShortGuardIntervalSupported (clientShortGuardInterval == 400);
      Ptr<HtConfiguration> serverHtConfiguration = wndServer->GetHtConfiguration ();
      serverHtConfiguration->SetShortGuardIntervalSupported (serverShortGuardInterval == 400);
    }
  else if (serverSelectedStandard.m_name == "802.11ax-6GHz"
           || serverSelectedStandard.m_name == "802.11ax-5GHz"
           || serverSelectedStandard.m_name == "802.11ax-2.4GHz")
    {
      wndServer->GetHeConfiguration ()->SetGuardInterval (NanoSeconds (serverShortGuardInterval));
      wndClient->GetHeConfiguration ()->SetGuardInterval (NanoSeconds (clientShortGuardInterval));
    }
  NS_LOG_DEBUG ("Channel width " << wifiPhyPtrClient->GetChannelWidth () << " noiseDbm " << noiseDbm);
  NS_LOG_DEBUG ("NSS " << wifiPhyPtrClient->GetMaxSupportedTxSpatialStreams ());

  // Configure signal and noise, and schedule first iteration
  noiseDbm += 10 * log10 (clientSelectedStandard.m_width * 1000000);
  double rssCurrent = (clientSelectedStandard.m_snrHigh + noiseDbm);
  rssLossModel->SetRss (rssCurrent);
  NS_LOG_INFO ("Setting initial Rss to " << rssCurrent);
  //Move the STA by stepsSize meters every stepTime seconds
  Simulator::Schedule (Seconds (0.5 + stepTime), &ChangeSignalAndReportRate, rssLossModel, step, rssCurrent, rateDataset, actualDataset);

  PacketSocketHelper packetSocketHelper;
  packetSocketHelper.Install (serverNode);
  packetSocketHelper.Install (clientNode);

  PacketSocketAddress socketAddr;
  socketAddr.SetSingleDevice (serverDevice.Get (0)->GetIfIndex ());
  if (broadcast)
    {
      socketAddr.SetPhysicalAddress (serverDevice.Get (0)->GetBroadcast ());
    }
  else
    {
      socketAddr.SetPhysicalAddress (serverDevice.Get (0)->GetAddress ());
    }
  // Arbitrary protocol type.
  // Note: PacketSocket doesn't have any L4 multiplexing or demultiplexing
  //       The only mux/demux is based on the protocol field
  socketAddr.SetProtocol (1);

  Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
  client->SetRemote (socketAddr);
  client->SetStartTime (Seconds (0.5));  // allow simulation warmup
  client->SetAttribute ("MaxPackets", UintegerValue (0));  // unlimited
  client->SetAttribute ("PacketSize", UintegerValue (packetSize));

  // Set a maximum rate 10% above the yMax specified for the selected standard
  double rate = clientSelectedStandard.m_yMax * 1e6 * 1.10;
  double clientInterval = static_cast<double> (packetSize) * 8 / rate;
  NS_LOG_DEBUG ("Setting interval to " << clientInterval << " sec for rate of " << rate << " bits/sec");

  client->SetAttribute ("Interval", TimeValue (Seconds (clientInterval)));
  clientNode->AddApplication (client);

  Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
  server->SetLocal (socketAddr);
  server->TraceConnectWithoutContext ("Rx", MakeCallback (&PacketRx));
  serverNode->AddApplication (server);

  Simulator::Stop (Seconds ((steps + 1) * stepTime));
  Simulator::Run ();
  Simulator::Destroy ();

  gnuplot.AddDataset (rateDataset);
  gnuplot.AddDataset (actualDataset);

  std::ostringstream xMinStr, xMaxStr, yMaxStr;
  std::string xRangeStr ("set xrange [");
  xMinStr << clientSelectedStandard.m_xMin;
  xRangeStr.append (xMinStr.str ());
  xRangeStr.append (":");
  xMaxStr << clientSelectedStandard.m_xMax;
  xRangeStr.append (xMaxStr.str ());
  xRangeStr.append ("]");
  std::string yRangeStr ("set yrange [0:");
  yMaxStr << clientSelectedStandard.m_yMax;
  yRangeStr.append (yMaxStr.str ());
  yRangeStr.append ("]");

  std::string title ("Results for ");
  title.append (standard);
  title.append (" with ");
  title.append (wifiManager);
  title.append ("\\n");
  if (standard == "802.11n-5GHz"
      || standard == "802.11n-2.4GHz"
      || standard == "802.11ac"
      || standard == "802.11ax-6GHz"
      || standard == "802.11ax-5GHz"
      || standard == "802.11ax-2.4GHz")
    {
      std::ostringstream serverGiStrStr;
      std::ostringstream serverWidthStrStr;
      std::ostringstream serverNssStrStr;
      title.append ("server: width=");
      serverWidthStrStr << serverSelectedStandard.m_width;
      title.append (serverWidthStrStr.str ());
      title.append ("MHz");
      title.append (" GI=");
      serverGiStrStr << serverShortGuardInterval;
      title.append (serverGiStrStr.str ());
      title.append ("ns");
      title.append (" nss=");
      serverNssStrStr << serverNss;
      title.append (serverNssStrStr.str ());
      title.append ("\\n");
      std::ostringstream clientGiStrStr;
      std::ostringstream clientWidthStrStr;
      std::ostringstream clientNssStrStr;
      title.append ("client: width=");
      clientWidthStrStr << clientSelectedStandard.m_width;
      title.append (clientWidthStrStr.str ());
      title.append ("MHz");
      title.append (" GI=");
      clientGiStrStr << clientShortGuardInterval;
      title.append (clientGiStrStr.str ());
      title.append ("ns");
      title.append (" nss=");
      clientNssStrStr << clientNss;
      title.append (clientNssStrStr.str ());
    }
  gnuplot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  gnuplot.SetLegend ("SNR (dB)", "Rate (Mb/s)");
  gnuplot.SetTitle (title);
  gnuplot.SetExtra (xRangeStr);
  gnuplot.AppendExtra (yRangeStr);
  gnuplot.AppendExtra ("set key top left");
  gnuplot.GenerateOutput (outfile);
  outfile.close ();

  return 0;
}

