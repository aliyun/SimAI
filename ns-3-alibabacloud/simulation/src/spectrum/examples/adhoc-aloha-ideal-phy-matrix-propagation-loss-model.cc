/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */



#include <iostream>

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/spectrum-model-ism2400MHz-res1MHz.h>
#include <ns3/spectrum-model-300kHz-300GHz-log.h>
#include <ns3/wifi-spectrum-value-helper.h>
#include <ns3/single-model-spectrum-channel.h>
#include <ns3/waveform-generator.h>
#include <ns3/spectrum-analyzer.h>
#include <ns3/log.h>
#include <string>
#include <iomanip>
#include <ns3/friis-spectrum-propagation-loss.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/mobility-module.h>
#include <ns3/spectrum-helper.h>
#include <ns3/applications-module.h>
#include <ns3/adhoc-aloha-noack-ideal-phy-helper.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TestAdhocOfdmAloha");

static bool g_verbose = false;
static uint64_t g_rxBytes;

void
PhyRxEndOkTrace (std::string context, Ptr<const Packet> p)
{
  if (g_verbose)
    {
      std::cout << context << " PHY RX END OK p:" << p << std::endl;
    }
  g_rxBytes += p->GetSize ();
}



/**
 * \ingroup spectrum
 *
 * Store the last pathloss value for each TX-RX pair. This is an
 * example of how the PathlossTrace (provided by some SpectrumChannel
 * implementations) work. 
 * 
 */
class GlobalPathlossDatabase
{
public:

  /** 
   * update the pathloss value
   * 
   * \param context 
   * \param txPhy the transmitting PHY
   * \param rxPhy the receiving PHY
   * \param lossDb the loss in dB
   */
  void UpdatePathloss (std::string context, Ptr<const SpectrumPhy> txPhy, Ptr<const SpectrumPhy> rxPhy, double lossDb);

  /** 
   * print the stored pathloss values to standard output
   * 
   */
  void Print ();

private:
  std::map<uint32_t, std::map<uint32_t, double> > m_pathlossMap; //!< Path loss map
};

void
GlobalPathlossDatabase::UpdatePathloss (std::string context, 
                                        Ptr<const SpectrumPhy> txPhyConst, 
                                        Ptr<const SpectrumPhy> rxPhyConst, 
                                        double lossDb)
{
  Ptr<SpectrumPhy> txPhy = ConstCast<SpectrumPhy> (txPhyConst);
  Ptr<SpectrumPhy> rxPhy = ConstCast<SpectrumPhy> (rxPhyConst);
  uint32_t txNodeId = txPhy->GetMobility ()->GetObject<Node> ()->GetId ();
  uint32_t rxNodeId = rxPhy->GetMobility ()->GetObject<Node> ()->GetId ();
  m_pathlossMap[txNodeId][rxNodeId] = lossDb;
}

void 
GlobalPathlossDatabase::Print ()
{
  for (std::map<uint32_t, std::map<uint32_t, double> >::const_iterator txit = m_pathlossMap.begin ();
       txit != m_pathlossMap.end ();
       ++txit)
    {
      for (std::map<uint32_t, double>::const_iterator rxit = txit->second.begin ();
           rxit != txit->second.end ();
           ++rxit)
        {
          std::cout << txit->first << " --> " << rxit->first << " : " << rxit->second << " dB" << std::endl;
        }
    }
}



int main (int argc, char** argv)
{
  CommandLine cmd (__FILE__);
  double lossDb = 130;
  double txPowerW = 0.1; 
  uint64_t phyRate = 500000;
  uint32_t pktSize = 1000;
  double simDuration = 0.5;
  std::string channelType ("ns3::SingleModelSpectrumChannel");
  cmd.AddValue ("verbose", "Print trace information if true", g_verbose);
  cmd.AddValue ("lossDb", "link loss in dB", lossDb);
  cmd.AddValue ("txPowerW", "txPower in Watts", txPowerW);
  cmd.AddValue ("phyRate", "PHY rate in bps", phyRate);
  cmd.AddValue ("pktSize", "packet size in bytes", pktSize);
  cmd.AddValue ("simDuration", "duration of the simulation in seconds", simDuration);
  cmd.AddValue ("channelType", "which SpectrumChannel implementation to be used", channelType);
  cmd.Parse (argc, argv);

  NodeContainer c;
  c.Create (2);

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);
  // the actual positions are irrelevant, since we use MatrixPropagationLossModel


  SpectrumChannelHelper channelHelper;
  channelHelper.SetChannel (channelType);
  channelHelper.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  Ptr<MatrixPropagationLossModel> propLoss = CreateObject<MatrixPropagationLossModel> ();
  propLoss->SetLoss (c.Get (0)->GetObject<MobilityModel> (), c.Get (1)->GetObject<MobilityModel> (), lossDb, true);
  channelHelper.AddPropagationLoss (propLoss);
  Ptr<SpectrumChannel> channel = channelHelper.Create ();


  WifiSpectrumValue5MhzFactory sf;

  uint32_t channelNumber = 1;
  Ptr<SpectrumValue> txPsd =  sf.CreateTxPowerSpectralDensity (txPowerW, channelNumber);

  // for the noise, we use the Power Spectral Density of thermal noise
  // at room temperature. The value of the PSD will be constant over the band of interest.
  const double k = 1.381e-23; //Boltzmann's constant
  const double T = 290; // temperature in Kelvin
  double noisePsdValue = k * T; // watts per hertz
  Ptr<SpectrumValue> noisePsd = sf.CreateConstant (noisePsdValue);

  AdhocAlohaNoackIdealPhyHelper deviceHelper;
  deviceHelper.SetChannel (channel);
  deviceHelper.SetTxPowerSpectralDensity (txPsd);
  deviceHelper.SetNoisePowerSpectralDensity (noisePsd);
  deviceHelper.SetPhyAttribute ("Rate", DataRateValue (DataRate (phyRate)));
  NetDeviceContainer devices = deviceHelper.Install (c);

  PacketSocketHelper packetSocket;
  packetSocket.Install (c);

  PacketSocketAddress socket;
  socket.SetSingleDevice (devices.Get (0)->GetIfIndex ());
  socket.SetPhysicalAddress (devices.Get (1)->GetAddress ());
  socket.SetProtocol (1);

  OnOffHelper onoff ("ns3::PacketSocketFactory", Address (socket));
  onoff.SetConstantRate (DataRate (2*phyRate));
  onoff.SetAttribute ("PacketSize", UintegerValue (pktSize));

  ApplicationContainer apps = onoff.Install (c.Get (0));
  apps.Start (Seconds (0.0));
  apps.Stop (Seconds (simDuration));

  Config::Connect ("/NodeList/*/DeviceList/*/Phy/RxEndOk", MakeCallback (&PhyRxEndOkTrace));

  GlobalPathlossDatabase globalPathlossDatabase;
  Config::Connect ("/ChannelList/*/$ns3::SpectrumChannel/PathLoss", 
                   MakeCallback (&GlobalPathlossDatabase::UpdatePathloss, &globalPathlossDatabase));

  g_rxBytes = 0;
  Simulator::Stop (Seconds (simDuration + 0.000001));
  Simulator::Run ();

  if (g_verbose)
    {
      globalPathlossDatabase.Print ();

      double throughputBps = (g_rxBytes * 8.0) / simDuration;
      std::cout << "throughput:       " << throughputBps << std::endl;
      std::cout << "throughput:       " << std::setw (20) << std::fixed << throughputBps << " bps" << std::endl;
      std::cout << "phy rate  :       "   << std::setw (20) << std::fixed << phyRate*1.0 << " bps" << std::endl; 
      double rxPowerW = txPowerW / (std::pow (10.0, lossDb/10.0));
      double capacity = 20e6*log2 (1.0 + (rxPowerW/20.0e6)/noisePsdValue);
      std::cout << "shannon capacity: "   << std::setw (20) << std::fixed << capacity <<  " bps" << std::endl; 

    }



  Simulator::Destroy ();
  return 0;
}
