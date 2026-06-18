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
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-address.h"

using namespace ns3;

/**
 *
 * This example shows the usage of raw packets transfer data.
 * Two nodes are sending their remaining energy percentage (1 byte)
 * to a gateway node, that prints the received data.
 * The transmissions are scheduled at random times to avoid collisions
 *
 */

NS_LOG_COMPONENT_DEFINE ("UanRawExample");


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
  void SendSinglePacket (Ptr<Node> node, Ptr<Packet> pkt, Mac8Address dst);

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
}

void
UanExperiment::PrintReceivedPacket (Ptr<Socket> socket)
{
  Address srcAddress;
  while (socket->GetRxAvailable () > 0)
    {
      Ptr<Packet> packet = socket->RecvFrom (srcAddress);
      PacketSocketAddress packetSocketAddress = PacketSocketAddress::ConvertFrom (srcAddress);
      srcAddress = packetSocketAddress.GetPhysicalAddress ();
      uint8_t energyReading;
      packet->CopyData (&energyReading, 1);

      if(Mac8Address::IsMatchingType (srcAddress))
        {
          NS_LOG_UNCOND ( "Time: " << Simulator::Now ().GetHours () << "h" << " | Node: " <<
                          Mac8Address::ConvertFrom (srcAddress) << " | Energy: " <<
                          +energyReading << "%");
        }
    }
}

void
UanExperiment::SetupApplications ()
{
  NodeContainer::Iterator node = m_nodes.Begin ();
  PacketSocketHelper packetSocketHelper;
  while (node != m_nodes.End ())
    {
      packetSocketHelper.Install (*node);
      PacketSocketAddress socketAddress;
      socketAddress.SetSingleDevice ((*node)->GetDevice (0)->GetIfIndex ());
      socketAddress.SetProtocol (0);
      m_sockets[*node] = Socket::CreateSocket (*node, TypeId::LookupByName ("ns3::PacketSocketFactory"));
      m_sockets[*node]->Bind ();
      m_sockets[*node]->Connect (socketAddress);
      m_sockets[*node]->SetRecvCallback (MakeCallback (&UanExperiment::PrintReceivedPacket, this));
      node++;
    }
}

void
UanExperiment::SendPackets ()
{
  Ptr<UniformRandomVariable> uniformRandomVariable = CreateObject<UniformRandomVariable> ();

  NodeContainer::Iterator node = m_nodes.Begin ();
  Mac8Address dst = Mac8Address::ConvertFrom ((*node)->GetDevice (0)->GetAddress ());
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
UanExperiment::SendSinglePacket (Ptr<Node> node, Ptr<Packet> pkt, Mac8Address dst)
{
  NS_LOG_UNCOND ( Simulator::Now ().GetHours () << "h" << " packet sent to " << dst );
  PacketSocketAddress socketAddress;
  socketAddress.SetSingleDevice (node->GetDevice (0)->GetIfIndex ());
  socketAddress.SetPhysicalAddress (dst);
  socketAddress.SetProtocol (0);
  m_sockets[node]->SendTo (pkt, 0, socketAddress);
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
