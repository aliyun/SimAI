/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

/**
 * \file uan-rc-example.cc
 * \ingroup UAN
 * This example uses UanMacRc and UanMacRcGw which combined form a system
 * using what is referred to as RC-MAC.  Details of RC-MAC will be published
 * soon.  In brief terms, RC-MAC is a dual channel protocol wherein the
 * available bandwidth is dynamically divided into two channels,
 * a reservation channel and a data channel.  The network is assumed
 * to consist of a single gateway node which services several
 * non-gateway nodes.
 * 
 * Time is divided into cycles.  The non-gateway nodes transmit RTS packets
 * on the reservation channel in parallel to scheduled data transmissions
 * (scheduled in the previous cycle), and the gateway stores these requests
 * for the duration of the cycle.  At the start of the next cycle
 * the gateway node transmits a CTS which contains packet transmission times
 * for reserved packets as well as bandwidth allocation information
 * 
 * This script deploys a single gateway node (current UanMacRc only supports
 * a single gateway) in the center of a region and then distributes
 * non-gateway nodes around the gateway with a uniformly distributed range
 * between each node and the gateway.
 * 
 * The script supports two simulation types.  By default the gateway
 * dynamically determines the optimal parameter settings and
 * simulations are run with varying number of nodes (SimMin to SimMax as
 * set by the command line).  If DoNode=0 is given as a command line option
 * then the mac parameter "a" (approximate expected number of successful
 * RTS arrivals per cycle) is varied as the simulation parameter.
 * 
 * For questions about this MAC protocol email "lentracy@gmail.com"
 */

#include "uan-rc-example.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/log.h"
#include "ns3/config.h"
#include "ns3/callback.h"
#include "ns3/stats-module.h"

#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("UanRcExample");

Experiment::Experiment () 
  : m_simMin (1),
    m_simMax (1),
    m_simStep (1),
    m_numRates (1023),
    m_totalRate (4096),
    m_maxRange (3000),
    m_numNodes (15),
    m_pktSize (1000),
    m_doNode (true),
    m_sifs (Seconds (0.05)),
    m_simTime (Seconds (5000)),
    m_gnuplotfile ("uan-rc-example.gpl"),
    m_bytesTotal (0)
{ 
}

void
Experiment::ReceivePacket (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  while ((packet = socket->Recv ()))
    {
      m_bytesTotal += packet->GetSize ();
    }
}

UanTxMode
Experiment::CreateMode (uint32_t kass,
                        uint32_t fc,
                        bool upperblock,
                        std::string name)
{

  std::ostringstream buf;
  buf << name << " " << kass;

  uint32_t rate = m_totalRate/(m_numRates+1)* (kass);
  uint32_t bw = kass * m_totalRate / (m_numRates+1);
  uint32_t fcmode;
  if(upperblock)
    fcmode = (m_totalRate - bw)/2 + fc;
  else
    fcmode = (uint32_t)((-((double) m_totalRate ) + (double) bw)/2.0 + (double) fc);


  uint32_t phyrate = m_totalRate;

  UanTxMode mode;
  mode = UanTxModeFactory::CreateMode (UanTxMode::OTHER,
                                       rate,
                                       phyrate,
                                       fcmode,
                                       bw,
                                       2,
                                       buf.str ());
  return mode;
}

//Creates m_numRates different modes each dividing m_totalRate Hz (assumes 1 bit per hz)
//centered at frequency fc
void
Experiment::CreateDualModes (uint32_t fc)
{


  for (uint32_t i=1; i < m_numRates+1; i++)
    {
      m_controlModes.AppendMode (CreateMode (i, fc, false, "control "));
    }
  for (uint32_t i=m_numRates; i > 0; i--)
    {
      m_dataModes.AppendMode (CreateMode (i, fc, true, "data "));
    }
}

uint32_t
Experiment::Run (uint32_t param)
{

  UanHelper uan;

  m_bytesTotal=0;

  uint32_t nNodes;
  uint32_t a;
  if(m_doNode)
    {
      a=0;
      nNodes = param;
    }
  else
    {
      nNodes = m_numNodes;
      a = param;
    }
  Time pDelay = Seconds ((double) m_maxRange / 1500.0);

  uan.SetPhy ("ns3::UanPhyDual",
              "SupportedModesPhy1", UanModesListValue (m_dataModes),
              "SupportedModesPhy2", UanModesListValue (m_controlModes));

  uan.SetMac ("ns3::UanMacRcGw",
              "NumberOfRates", UintegerValue (m_numRates),
              "NumberOfNodes", UintegerValue (nNodes),
              "MaxReservations", UintegerValue (a),
              "SIFS", TimeValue (m_sifs),
              "MaxPropDelay", TimeValue (pDelay),
              "FrameSize", UintegerValue (m_pktSize));
  Ptr<UanChannel> chan = CreateObject<UanChannel>();

  NodeContainer sink;
  sink.Create (1);
  NetDeviceContainer sinkDev = uan.Install (sink, chan);

  uan.SetMac ("ns3::UanMacRc",
              "NumberOfRates", UintegerValue (m_numRates),
              "MaxPropDelay", TimeValue (pDelay));
  NodeContainer nodes;
  nodes.Create (nNodes);
  NetDeviceContainer devices = uan.Install (nodes, chan);

  MobilityHelper mobility;
  uint32_t depth = 70;
  Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator> ();

  Ptr<UniformRandomVariable> urv = CreateObject<UniformRandomVariable> ();
  Ptr<UniformRandomVariable> utheta = CreateObject<UniformRandomVariable> ();
  pos->Add (Vector (m_maxRange, m_maxRange, depth));

  for (uint32_t i=0; i<nNodes; i++)
    {
      double theta = utheta->GetValue (0, 2.0*M_PI);
      double r = urv->GetValue (0,m_maxRange);

      double x = m_maxRange + r*std::cos (theta);
      double y = m_maxRange + r*std::sin (theta);

      pos->Add (Vector (x, y, depth));

    }

  mobility.SetPositionAllocator (pos);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (sink);
  mobility.Install (nodes);

  PacketSocketHelper pktskth;
  pktskth.Install (nodes);
  pktskth.Install (sink);

  PacketSocketAddress socket;
  socket.SetSingleDevice (sinkDev.Get (0)->GetIfIndex ());
  socket.SetPhysicalAddress (sinkDev.Get (0)->GetAddress ());
  socket.SetProtocol (0);

  OnOffHelper app ("ns3::PacketSocketFactory", Address (socket));
  app.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  app.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
  app.SetAttribute ("DataRate", DataRateValue (m_totalRate));
  app.SetAttribute ("PacketSize", UintegerValue (m_pktSize));

  ApplicationContainer apps = app.Install (nodes);

  apps.Start (Seconds (0.5));
  apps.Stop (m_simTime + Seconds (0.5));

  Ptr<Node> sinkNode = sink.Get (0);
  TypeId psfid = TypeId::LookupByName ("ns3::PacketSocketFactory");

  Ptr<Socket> sinkSocket = Socket::CreateSocket (sinkNode, psfid);
  sinkSocket->Bind (socket);
  sinkSocket->SetRecvCallback (MakeCallback (&Experiment::ReceivePacket, this));

  Simulator::Stop (m_simTime + Seconds (0.6));
  Simulator::Run ();
  Simulator::Destroy ();

  return m_bytesTotal;
}
int
main (int argc, char *argv[])
{

  Experiment exp;
  bool quiet = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("TotalRate", "Total channel capacity", exp.m_totalRate);
  cmd.AddValue ("NumberRates", "Number of divided rates ( (NumberRates+1)%TotalRate should be 0)", exp.m_numRates);
  cmd.AddValue ("MaxRange", "Maximum range between gateway and acoustic node", exp.m_maxRange);
  cmd.AddValue ("SimMin", "Minimum parameter to test (nodes if DoNode=1, \"a\" param otherwise)", exp.m_simMin);
  cmd.AddValue ("SimMax", "Maximum parameter to test (nodes if DoNode=1, \"a\" param otherwise)", exp.m_simMax);
  cmd.AddValue ("SimStep", "Amount to increment param per trial", exp.m_simStep);
  cmd.AddValue ("DataFile", "Filename for GnuPlot", exp.m_gnuplotfile);
  cmd.AddValue ("NumberNodes", "Number of nodes (invalid for doNode=1)", exp.m_numNodes);
  cmd.AddValue ("SIFS", "SIFS time duration", exp.m_sifs);
  cmd.AddValue ("PktSize", "Packet size in bytes", exp.m_pktSize);
  cmd.AddValue ("SimTime", "Simulation time per trial", exp.m_simTime);
  cmd.AddValue ("DoNode", "1 for do max nodes simulation (invalidates AMin and AMax values)", exp.m_doNode);
  cmd.AddValue ("Quiet", "Run in quiet mode (disable logging)", quiet);
  cmd.Parse (argc, argv);

  if (!quiet)
    {
      LogComponentEnable ("UanRcExample", LOG_LEVEL_ALL);
    }

  exp.CreateDualModes (12000);

  ;

  Gnuplot2dDataset ds;
  for (uint32_t param=exp.m_simMin; param<=exp.m_simMax; param += exp.m_simStep)
    {
      uint32_t bytesRx = exp.Run (param);
      NS_LOG_DEBUG ("param=" << param << ":  Received " << bytesRx << " bytes at sink");

      double util = bytesRx*8.0/(exp.m_simTime.GetSeconds ()*exp.m_totalRate);

      ds.Add (param, util);

      SeedManager::SetRun (SeedManager::GetRun () + 1);
    }

  Gnuplot gp;
  gp.AddDataset (ds);
  std::ofstream of (exp.m_gnuplotfile.c_str ());
  if (!of.is_open ())
    {
      NS_FATAL_ERROR ("Can not open GNU Plot outfile: " << exp.m_gnuplotfile);
    }
  gp.GenerateOutput (of);

}
