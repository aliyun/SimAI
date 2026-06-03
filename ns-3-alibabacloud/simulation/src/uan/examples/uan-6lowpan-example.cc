/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Hossam Khader <hossamkhader@gmail.com>
 */

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/node-container.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/energy-source-container.h"
#include "ns3/uan-helper.h"
#include "ns3/uan-channel.h"
#include "ns3/acoustic-modem-energy-model-helper.h"
#include "ns3/sixlowpan-helper.h"
#include "ns3/sixlowpan-net-device.h"

using namespace ns3;

/**
 *
 * This example shows the usage of UDP over 6LoWPAN to transfer data.
 * Two nodes are sending their remaining energy percentage (1 byte)
 * to a gateway node, that prints the received data.
 * The transmissions are scheduled at random times to avoid collisions
 *
 */

NS_LOG_COMPONENT_DEFINE ("Uan6lowpanExample");


class UanExperiment
{
public:
  UanExperiment ();

  /**
   * Set the UAN nodes position
   */
  void SetupPositions ();

  /**
   * Set the UAN nodes energy
   */
  void SetupEnergy ();

  /**
   * Set the UAN nodes communication channels
   */
  void SetupCommunications ();

  /**
   * Set the UAN nodes communication channels
   */
  void SetupApplications ();

  /**
   * Send a packet from all the nodes
   */
  void SendPackets ();

  /**
   * Send a packet from one of the nodes
   * \param node The sending node
   * \param pkt The packet
   * \param dst the destination
   */
  void SendSinglePacket (Ptr<Node> node, Ptr<Packet> pkt, Ipv6Address dst);

  /**
   * Print the received packet
   * \param socket The receiving socket
   */
  void PrintReceivedPacket (Ptr<Socket> socket);

  /**
   * Prepare the experiment
   */
  void Prepare ();

  /**
   * Teardown the experiment
   */
  void Teardown ();

private:
  NodeContainer m_nodes; //!< UAN nodes
  std::map<Ptr<Node>, Ptr<Socket> > m_sockets; //!< send and receive sockets
};


UanExperiment::UanExperiment ()
{
}

void
UanExperiment::SetupPositions ()
{
  MobilityHelper mobilityHelper;
  mobilityHelper.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobilityHelper.Install (m_nodes);
  m_nodes.Get (0)->GetObject<MobilityModel> ()->SetPosition (Vector (0, 0, 0));
  m_nodes.Get (1)->GetObject<MobilityModel> ()->SetPosition (Vector (100, 0, 0));
  m_nodes.Get (2)->GetObject<MobilityModel> ()->SetPosition (Vector (-100, 0, 0));
}

void
UanExperiment::SetupEnergy ()
{
  BasicEnergySourceHelper energySourceHelper;
  energySourceHelper.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (900000));
  energySourceHelper.Install (m_nodes);
}

void
UanExperiment::SetupCommunications ()
{
  Ptr<UanChannel> channel = CreateObject<UanChannel> ();
  UanHelper uanHelper;
  NetDeviceContainer netDeviceContainer = uanHelper.Install (m_nodes, channel);
  EnergySourceContainer energySourceContainer;
  NodeContainer::Iterator node = m_nodes.Begin ();
  while (node != m_nodes.End ())
    {
      energySourceContainer.Add ((*node)->GetObject<EnergySourceContainer> ()->Get (0));
      node++;
    }
  AcousticModemEnergyModelHelper acousticModemEnergyModelHelper;
  acousticModemEnergyModelHelper.Install (netDeviceContainer, energySourceContainer);

  SixLowPanHelper sixLowPanHelper;
  NetDeviceContainer sixlowpanNetDevices = sixLowPanHelper.Install (netDeviceContainer);

  InternetStackHelper internetStackHelper;
  internetStackHelper.Install (m_nodes);

  Ipv6AddressHelper ipv6AddressHelper;
  ipv6AddressHelper.SetBase (Ipv6Address ("2002::"), Ipv6Prefix (64));
  ipv6AddressHelper.Assign (sixlowpanNetDevices);

  node = m_nodes.Begin ();
  while (node != m_nodes.End ())
    {
      (*node)->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("DAD", BooleanValue (false));
      (*node)->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("ReachableTime", TimeValue (Seconds (3600)));
      (*node)->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("RetransmissionTime", TimeValue (Seconds (1000)));
      node++;
    }
}

void
UanExperiment::PrintReceivedPacket (Ptr<Socket> socket)
{
  Address srcAddress;
  while (socket->GetRxAvailable () > 0)
    {
      Ptr<Packet> packet = socket->RecvFrom (srcAddress);
      uint8_t energyReading;
      packet->CopyData (&energyReading, 1);

      if(Inet6SocketAddress::IsMatchingType (srcAddress))
        {
          NS_LOG_UNCOND ( "Time: " << Simulator::Now ().GetHours () << "h" << " | Node: " <<
                          Inet6SocketAddress::ConvertFrom (srcAddress).GetIpv6 () << " | Energy: " <<
                          +energyReading << "%");
        }
    }
}

void
UanExperiment::SetupApplications ()
{
  NodeContainer::Iterator node = m_nodes.Begin ();
  while (node != m_nodes.End ())
    {
      m_sockets[*node] = Socket::CreateSocket (*node, TypeId::LookupByName ("ns3::UdpSocketFactory"));
      if((*node)->GetObject<Ipv6> () != NULL)
        {
          Inet6SocketAddress ipv6_local = Inet6SocketAddress (Ipv6Address::GetAny (), 9);
          m_sockets[*node]->Bind (ipv6_local);
        }

      m_sockets[*node]->SetRecvCallback (MakeCallback (&UanExperiment::PrintReceivedPacket, this));
      node++;
    }
}

void
UanExperiment::SendPackets ()
{
  Ptr<UniformRandomVariable> uniformRandomVariable = CreateObject<UniformRandomVariable> ();

  NodeContainer::Iterator node = m_nodes.Begin ();
  Ipv6Address dst = (*node)->GetObject<Ipv6L3Protocol> ()->GetInterface (1)->GetAddress (1).GetAddress ();
  node++;
  while (node != m_nodes.End ())
    {
      uint8_t energy = ((*node)->GetObject<EnergySourceContainer> ()->Get (0)->GetEnergyFraction ()) * 100;

      Ptr<Packet> pkt = Create<Packet> (&energy, 1);

      double time = uniformRandomVariable->GetValue (0, 60);
      Simulator::Schedule (Seconds (time), &UanExperiment::SendSinglePacket, this, *node, pkt, dst);
      node++;
    }
  Simulator::Schedule (Hours (2), &UanExperiment::SendPackets, this);
}

void
UanExperiment::SendSinglePacket (Ptr<Node> node, Ptr<Packet> pkt, Ipv6Address dst)
{
  NS_LOG_UNCOND ( Simulator::Now ().GetHours () << "h" << " packet sent to " << dst );
  Inet6SocketAddress ipv6_destination = Inet6SocketAddress (Ipv6Address::ConvertFrom (dst), 9);
  m_sockets[node]->SendTo (pkt, 0, ipv6_destination);
}

void
UanExperiment::Prepare ()
{
  m_nodes.Create (3);
  SetupPositions ();
  SetupEnergy ();
  SetupCommunications ();
  SetupApplications ();
  SendPackets ();
}

void
UanExperiment::Teardown ()
{
  std::map<Ptr<Node>, Ptr<Socket> >::iterator socket;

  for (socket = m_sockets.begin (); socket != m_sockets.end (); socket++)
    {
      socket->second->Close ();
    }
}

int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  UanExperiment experiment;
  experiment.Prepare ();

  Simulator::Stop (Days (50));
  Simulator::Run ();
  Simulator::Destroy ();

  experiment.Teardown ();

  return 0;
}
