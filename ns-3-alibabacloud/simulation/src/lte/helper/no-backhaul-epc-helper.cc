/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 *         (based on the original point-to-point-epc-helper.cc)
 */

#include "ns3/boolean.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/packet-socket-address.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv6-static-routing-helper.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/epc-enb-application.h"
#include "ns3/epc-pgw-application.h"
#include "ns3/epc-sgw-application.h"
#include "ns3/epc-mme-application.h"
#include "ns3/epc-x2.h"
#include "ns3/lte-enb-rrc.h"
#include "ns3/epc-ue-nas.h"
#include "ns3/lte-enb-net-device.h"
#include "ns3/lte-ue-net-device.h"

#include "ns3/no-backhaul-epc-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NoBackhaulEpcHelper");

NS_OBJECT_ENSURE_REGISTERED (NoBackhaulEpcHelper);


NoBackhaulEpcHelper::NoBackhaulEpcHelper () 
  : m_gtpuUdpPort (2152),  // fixed by the standard
    m_s11LinkDataRate (DataRate ("10Gb/s")),
    m_s11LinkDelay (Seconds (0)),
    m_s11LinkMtu (3000),
    m_gtpcUdpPort (2123),  // fixed by the standard
    m_s5LinkDataRate (DataRate ("10Gb/s")),
    m_s5LinkDelay (Seconds (0)),
    m_s5LinkMtu (3000)
{
  NS_LOG_FUNCTION (this);
  // To access the attribute value within the constructor
  ObjectBase::ConstructSelf (AttributeConstructionList ());

  int retval;

  // since we use point-to-point links for links between the core network nodes,
  // we use a /30 subnet which can hold exactly two addresses
  // (remember that net broadcast and null address are not valid)
  m_x2Ipv4AddressHelper.SetBase ("12.0.0.0", "255.255.255.252");
  m_s11Ipv4AddressHelper.SetBase ("13.0.0.0", "255.255.255.252");
  m_s5Ipv4AddressHelper.SetBase ("14.0.0.0", "255.255.255.252");

  // we use a /8 net for all UEs
  m_uePgwAddressHelper.SetBase ("7.0.0.0", "255.0.0.0");

  // we use a /64 IPv6 net all UEs
  m_uePgwAddressHelper6.SetBase ("7777:f00d::", Ipv6Prefix (64));

  // Create PGW, SGW and MME nodes
  m_pgw = CreateObject<Node> ();
  m_sgw = CreateObject<Node> ();
  m_mme = CreateObject<Node> ();
  InternetStackHelper internet;
  internet.Install (m_pgw);
  internet.Install (m_sgw);
  internet.Install (m_mme);

  // The Tun device resides in different 64 bit subnet.
  // We must create an unique route to tun device for all the packets destined
  // to all 64 bit IPv6 prefixes of UEs, based by the unique 48 bit network prefix of this EPC network
  Ipv6StaticRoutingHelper ipv6RoutingHelper;
  Ptr<Ipv6StaticRouting> pgwStaticRouting = ipv6RoutingHelper.GetStaticRouting (m_pgw->GetObject<Ipv6> ());
  pgwStaticRouting->AddNetworkRouteTo ("7777:f00d::", Ipv6Prefix (64), Ipv6Address ("::"), 1, 0);

  // create TUN device implementing tunneling of user data over GTP-U/UDP/IP in the PGW
  m_tunDevice = CreateObject<VirtualNetDevice> ();

  // allow jumbo packets
  m_tunDevice->SetAttribute ("Mtu", UintegerValue (30000));

  // yes we need this
  m_tunDevice->SetAddress (Mac48Address::Allocate ());

  m_pgw->AddDevice (m_tunDevice);
  NetDeviceContainer tunDeviceContainer;
  tunDeviceContainer.Add (m_tunDevice);
  // the TUN device is on the same subnet as the UEs, so when a packet
  // addressed to an UE arrives at the intenet to the WAN interface of
  // the PGW it will be forwarded to the TUN device. 
  Ipv4InterfaceContainer tunDeviceIpv4IfContainer = AssignUeIpv4Address (tunDeviceContainer);  


  // the TUN device for IPv6 address is on the different subnet as the
  // UEs, it will forward the UE packets as we have inserted the route
  // for all UEs at the time of assigning UE addresses
  Ipv6InterfaceContainer tunDeviceIpv6IfContainer = AssignUeIpv6Address (tunDeviceContainer);


  //Set Forwarding of the IPv6 interface
  tunDeviceIpv6IfContainer.SetForwarding (0,true);
  tunDeviceIpv6IfContainer.SetDefaultRouteInAllNodes (0);

  // Create S5 link between PGW and SGW
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (m_s5LinkDataRate));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (m_s5LinkMtu));
  p2ph.SetChannelAttribute ("Delay", TimeValue (m_s5LinkDelay));
  NetDeviceContainer pgwSgwDevices = p2ph.Install (m_pgw, m_sgw);
  NS_LOG_LOGIC ("IPv4 ifaces of the PGW after installing p2p dev: " << m_pgw->GetObject<Ipv4> ()->GetNInterfaces ());
  NS_LOG_LOGIC ("IPv4 ifaces of the SGW after installing p2p dev: " << m_sgw->GetObject<Ipv4> ()->GetNInterfaces ());
  Ptr<NetDevice> pgwDev = pgwSgwDevices.Get (0);
  Ptr<NetDevice> sgwDev = pgwSgwDevices.Get (1);
  m_s5Ipv4AddressHelper.NewNetwork ();
  Ipv4InterfaceContainer pgwSgwIpIfaces = m_s5Ipv4AddressHelper.Assign (pgwSgwDevices);
  NS_LOG_LOGIC ("IPv4 ifaces of the PGW after assigning Ipv4 addr to S5 dev: " << m_pgw->GetObject<Ipv4> ()->GetNInterfaces ());
  NS_LOG_LOGIC ("IPv4 ifaces of the SGW after assigning Ipv4 addr to S5 dev: " << m_sgw->GetObject<Ipv4> ()->GetNInterfaces ());

  Ipv4Address pgwS5Address = pgwSgwIpIfaces.GetAddress (0);
  Ipv4Address sgwS5Address = pgwSgwIpIfaces.GetAddress (1);

  // Create S5-U socket in the PGW
  Ptr<Socket> pgwS5uSocket = Socket::CreateSocket (m_pgw, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = pgwS5uSocket->Bind (InetSocketAddress (pgwS5Address, m_gtpuUdpPort));
  NS_ASSERT (retval == 0);

  // Create S5-C socket in the PGW
  Ptr<Socket> pgwS5cSocket = Socket::CreateSocket (m_pgw, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = pgwS5cSocket->Bind (InetSocketAddress (pgwS5Address, m_gtpcUdpPort));
  NS_ASSERT (retval == 0);

  // Create EpcPgwApplication
  m_pgwApp = CreateObject<EpcPgwApplication> (m_tunDevice, pgwS5Address, pgwS5uSocket, pgwS5cSocket);
  m_pgw->AddApplication (m_pgwApp);

  // Connect EpcPgwApplication and virtual net device for tunneling
  m_tunDevice->SetSendCallback (MakeCallback (&EpcPgwApplication::RecvFromTunDevice, m_pgwApp));


  // Create S5-U socket in the SGW
  Ptr<Socket> sgwS5uSocket = Socket::CreateSocket (m_sgw, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = sgwS5uSocket->Bind (InetSocketAddress (sgwS5Address, m_gtpuUdpPort));
  NS_ASSERT (retval == 0);

  // Create S5-C socket in the SGW
  Ptr<Socket> sgwS5cSocket = Socket::CreateSocket (m_sgw, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = sgwS5cSocket->Bind (InetSocketAddress (sgwS5Address, m_gtpcUdpPort));
  NS_ASSERT (retval == 0);

  // Create S1-U socket in the SGW
  Ptr<Socket> sgwS1uSocket = Socket::CreateSocket (m_sgw, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = sgwS1uSocket->Bind (InetSocketAddress (Ipv4Address::GetAny (), m_gtpuUdpPort));
  NS_ASSERT (retval == 0);

  // Create EpcSgwApplication
  m_sgwApp = CreateObject<EpcSgwApplication> (sgwS1uSocket, sgwS5Address, sgwS5uSocket, sgwS5cSocket);
  m_sgw->AddApplication (m_sgwApp);
  m_sgwApp->AddPgw (pgwS5Address);
  m_pgwApp->AddSgw (sgwS5Address);


  // Create S11 link between MME and SGW
  PointToPointHelper s11P2ph;
  s11P2ph.SetDeviceAttribute ("DataRate", DataRateValue (m_s11LinkDataRate));
  s11P2ph.SetDeviceAttribute ("Mtu", UintegerValue (m_s11LinkMtu));
  s11P2ph.SetChannelAttribute ("Delay", TimeValue (m_s11LinkDelay));
  NetDeviceContainer mmeSgwDevices = s11P2ph.Install (m_mme, m_sgw);
  NS_LOG_LOGIC ("MME's IPv4 ifaces after installing p2p dev: " << m_mme->GetObject<Ipv4> ()->GetNInterfaces ());
  NS_LOG_LOGIC ("SGW's IPv4 ifaces after installing p2p dev: " << m_sgw->GetObject<Ipv4> ()->GetNInterfaces ());
  Ptr<NetDevice> mmeDev = mmeSgwDevices.Get (0);
  Ptr<NetDevice> sgwS11Dev = mmeSgwDevices.Get (1);
  m_s11Ipv4AddressHelper.NewNetwork ();
  Ipv4InterfaceContainer mmeSgwIpIfaces = m_s11Ipv4AddressHelper.Assign (mmeSgwDevices);
  NS_LOG_LOGIC ("MME's IPv4 ifaces after assigning Ipv4 addr to S11 dev: " << m_mme->GetObject<Ipv4> ()->GetNInterfaces ());
  NS_LOG_LOGIC ("SGW's IPv4 ifaces after assigning Ipv4 addr to S11 dev: " << m_sgw->GetObject<Ipv4> ()->GetNInterfaces ());

  Ipv4Address mmeS11Address = mmeSgwIpIfaces.GetAddress (0);
  Ipv4Address sgwS11Address = mmeSgwIpIfaces.GetAddress (1);

  // Create S11 socket in the MME
  Ptr<Socket> mmeS11Socket = Socket::CreateSocket (m_mme, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = mmeS11Socket->Bind (InetSocketAddress (mmeS11Address, m_gtpcUdpPort));
  NS_ASSERT (retval == 0);

  // Create S11 socket in the SGW
  Ptr<Socket> sgwS11Socket = Socket::CreateSocket (m_sgw, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  retval = sgwS11Socket->Bind (InetSocketAddress (sgwS11Address, m_gtpcUdpPort));
  NS_ASSERT (retval == 0);


  // Create MME Application and connect with SGW via S11 interface
  m_mmeApp = CreateObject<EpcMmeApplication> ();
  m_mme->AddApplication (m_mmeApp);
  m_mmeApp->AddSgw (sgwS11Address, mmeS11Address, mmeS11Socket);
  m_sgwApp->AddMme (mmeS11Address, sgwS11Socket);
}

NoBackhaulEpcHelper::~NoBackhaulEpcHelper ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
NoBackhaulEpcHelper::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static TypeId tid = TypeId ("ns3::NoBackhaulEpcHelper")
    .SetParent<EpcHelper> ()
    .SetGroupName("Lte")
    .AddConstructor<NoBackhaulEpcHelper> ()
    .AddAttribute ("S5LinkDataRate",
                   "The data rate to be used for the next S5 link to be created",
                   DataRateValue (DataRate ("10Gb/s")),
                   MakeDataRateAccessor (&NoBackhaulEpcHelper::m_s5LinkDataRate),
                   MakeDataRateChecker ())
    .AddAttribute ("S5LinkDelay",
                   "The delay to be used for the next S5 link to be created",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&NoBackhaulEpcHelper::m_s5LinkDelay),
                   MakeTimeChecker ())
    .AddAttribute ("S5LinkMtu",
                   "The MTU of the next S5 link to be created",
                   UintegerValue (2000),
                   MakeUintegerAccessor (&NoBackhaulEpcHelper::m_s5LinkMtu),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("S11LinkDataRate", 
                   "The data rate to be used for the next S11 link to be created",
                   DataRateValue (DataRate ("10Gb/s")),
                   MakeDataRateAccessor (&NoBackhaulEpcHelper::m_s11LinkDataRate),
                   MakeDataRateChecker ())
    .AddAttribute ("S11LinkDelay", 
                   "The delay to be used for the next S11 link to be created",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&NoBackhaulEpcHelper::m_s11LinkDelay),
                   MakeTimeChecker ())
    .AddAttribute ("S11LinkMtu", 
                   "The MTU of the next S11 link to be created.",
                   UintegerValue (2000),
                   MakeUintegerAccessor (&NoBackhaulEpcHelper::m_s11LinkMtu),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("X2LinkDataRate",
                   "The data rate to be used for the next X2 link to be created",
                   DataRateValue (DataRate ("10Gb/s")),
                   MakeDataRateAccessor (&NoBackhaulEpcHelper::m_x2LinkDataRate),
                   MakeDataRateChecker ())
    .AddAttribute ("X2LinkDelay",
                   "The delay to be used for the next X2 link to be created",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&NoBackhaulEpcHelper::m_x2LinkDelay),
                   MakeTimeChecker ())
    .AddAttribute ("X2LinkMtu",
                   "The MTU of the next X2 link to be created. Note that, because of some big X2 messages, you need a big MTU.",
                   UintegerValue (3000),
                   MakeUintegerAccessor (&NoBackhaulEpcHelper::m_x2LinkMtu),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("X2LinkPcapPrefix",
                   "Prefix for Pcap generated by X2 link",
                   StringValue ("x2"),
                   MakeStringAccessor (&NoBackhaulEpcHelper::m_x2LinkPcapPrefix),
                   MakeStringChecker ())
    .AddAttribute ("X2LinkEnablePcap",
                   "Enable Pcap for X2 link",
                   BooleanValue (false),
                   MakeBooleanAccessor (&NoBackhaulEpcHelper::m_x2LinkEnablePcap),
                   MakeBooleanChecker ())
  ;
  return tid;
}

TypeId
NoBackhaulEpcHelper::GetInstanceTypeId () const
{
  return GetTypeId ();
}

void
NoBackhaulEpcHelper::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_tunDevice->SetSendCallback (MakeNullCallback<bool, Ptr<Packet>, const Address&, const Address&, uint16_t> ());
  m_tunDevice = 0;
  m_sgwApp = 0;
  m_sgw->Dispose ();
  m_pgwApp = 0;
  m_pgw->Dispose ();
  m_mmeApp = 0;
  m_mme->Dispose ();
}


void
NoBackhaulEpcHelper::AddEnb (Ptr<Node> enb, Ptr<NetDevice> lteEnbNetDevice, std::vector<uint16_t> cellIds)
{
  NS_LOG_FUNCTION (this << enb << lteEnbNetDevice << cellIds.size ());
  NS_ASSERT (enb == lteEnbNetDevice->GetNode ());

  int retval;

  // add an IPv4 stack to the previously created eNB
  InternetStackHelper internet;
  internet.Install (enb);
  NS_LOG_LOGIC ("number of Ipv4 ifaces of the eNB after node creation: " << enb->GetObject<Ipv4> ()->GetNInterfaces ());

  // create LTE socket for the ENB 
  Ptr<Socket> enbLteSocket = Socket::CreateSocket (enb, TypeId::LookupByName ("ns3::PacketSocketFactory"));
  PacketSocketAddress enbLteSocketBindAddress;
  enbLteSocketBindAddress.SetSingleDevice (lteEnbNetDevice->GetIfIndex ());
  enbLteSocketBindAddress.SetProtocol (Ipv4L3Protocol::PROT_NUMBER);
  retval = enbLteSocket->Bind (enbLteSocketBindAddress);
  NS_ASSERT (retval == 0);
  PacketSocketAddress enbLteSocketConnectAddress;
  enbLteSocketConnectAddress.SetPhysicalAddress (Mac48Address::GetBroadcast ());
  enbLteSocketConnectAddress.SetSingleDevice (lteEnbNetDevice->GetIfIndex ());
  enbLteSocketConnectAddress.SetProtocol (Ipv4L3Protocol::PROT_NUMBER);
  retval = enbLteSocket->Connect (enbLteSocketConnectAddress);
  NS_ASSERT (retval == 0);  

  // create LTE socket for the ENB 
  Ptr<Socket> enbLteSocket6 = Socket::CreateSocket (enb, TypeId::LookupByName ("ns3::PacketSocketFactory"));
  PacketSocketAddress enbLteSocketBindAddress6;
  enbLteSocketBindAddress6.SetSingleDevice (lteEnbNetDevice->GetIfIndex ());
  enbLteSocketBindAddress6.SetProtocol (Ipv6L3Protocol::PROT_NUMBER);
  retval = enbLteSocket6->Bind (enbLteSocketBindAddress6);
  NS_ASSERT (retval == 0);  
  PacketSocketAddress enbLteSocketConnectAddress6;
  enbLteSocketConnectAddress6.SetPhysicalAddress (Mac48Address::GetBroadcast ());
  enbLteSocketConnectAddress6.SetSingleDevice (lteEnbNetDevice->GetIfIndex ());
  enbLteSocketConnectAddress6.SetProtocol (Ipv6L3Protocol::PROT_NUMBER);
  retval = enbLteSocket6->Connect (enbLteSocketConnectAddress6);
  NS_ASSERT (retval == 0);  

  NS_LOG_INFO ("Create EpcEnbApplication for cell ID " << cellIds.at (0));
  Ptr<EpcEnbApplication> enbApp = CreateObject<EpcEnbApplication> (enbLteSocket, enbLteSocket6, cellIds.at (0));
  enb->AddApplication (enbApp);
  NS_ASSERT (enb->GetNApplications () == 1);
  NS_ASSERT_MSG (enb->GetApplication (0)->GetObject<EpcEnbApplication> () != 0, "cannot retrieve EpcEnbApplication");
  NS_LOG_LOGIC ("enb: " << enb << ", enb->GetApplication (0): " << enb->GetApplication (0));

  NS_LOG_INFO ("Create EpcX2 entity");
  Ptr<EpcX2> x2 = CreateObject<EpcX2> ();
  enb->AggregateObject (x2);
}


void
NoBackhaulEpcHelper::AddX2Interface (Ptr<Node> enb1, Ptr<Node> enb2)
{
  NS_LOG_FUNCTION (this << enb1 << enb2);

  // Create a point to point link between the two eNBs with
  // the corresponding new NetDevices on each side
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (m_x2LinkDataRate));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (m_x2LinkMtu));
  p2ph.SetChannelAttribute ("Delay", TimeValue (m_x2LinkDelay));
  NetDeviceContainer enbDevices = p2ph.Install (enb1, enb2);
  NS_LOG_LOGIC ("number of Ipv4 ifaces of the eNB #1 after installing p2p dev: " << enb1->GetObject<Ipv4> ()->GetNInterfaces ());
  NS_LOG_LOGIC ("number of Ipv4 ifaces of the eNB #2 after installing p2p dev: " << enb2->GetObject<Ipv4> ()->GetNInterfaces ());

  if (m_x2LinkEnablePcap)
    {
      p2ph.EnablePcapAll (m_x2LinkPcapPrefix);
    }

  m_x2Ipv4AddressHelper.NewNetwork ();
  Ipv4InterfaceContainer enbIpIfaces = m_x2Ipv4AddressHelper.Assign (enbDevices);
  NS_LOG_LOGIC ("number of Ipv4 ifaces of the eNB #1 after assigning Ipv4 addr to X2 dev: " << enb1->GetObject<Ipv4> ()->GetNInterfaces ());
  NS_LOG_LOGIC ("number of Ipv4 ifaces of the eNB #2 after assigning Ipv4 addr to X2 dev: " << enb2->GetObject<Ipv4> ()->GetNInterfaces ());

  Ipv4Address enb1X2Address = enbIpIfaces.GetAddress (0);
  Ipv4Address enb2X2Address = enbIpIfaces.GetAddress (1);

  // Add X2 interface to both eNBs' X2 entities
  Ptr<EpcX2> enb1X2 = enb1->GetObject<EpcX2> ();
  Ptr<EpcX2> enb2X2 = enb2->GetObject<EpcX2> ();

  Ptr<NetDevice> enb1LteDev = enb1->GetDevice (0);
  Ptr<NetDevice> enb2LteDev = enb2->GetDevice (0);

  DoAddX2Interface (enb1X2, enb1LteDev, enb1X2Address, enb2X2, enb2LteDev, enb2X2Address);
}

void
NoBackhaulEpcHelper::DoAddX2Interface (const Ptr<EpcX2> &enb1X2, const Ptr<NetDevice> &enb1LteDev,
                                       const Ipv4Address &enb1X2Address,
                                       const Ptr<EpcX2> &enb2X2, const Ptr<NetDevice> &enb2LteDev,
                                       const Ipv4Address &enb2X2Address) const
{
  NS_LOG_FUNCTION (this);

  Ptr<LteEnbNetDevice> enb1LteDevice = enb1LteDev->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbNetDevice> enb2LteDevice = enb2LteDev->GetObject<LteEnbNetDevice> ();

  NS_ABORT_MSG_IF (enb1LteDevice == nullptr , "Unable to find LteEnbNetDevice for the first eNB");
  NS_ABORT_MSG_IF (enb2LteDevice == nullptr , "Unable to find LteEnbNetDevice for the second eNB");

  std::vector<uint16_t> enb1CellIds = enb1LteDevice->GetCellIds ();
  std::vector<uint16_t> enb2CellIds = enb2LteDevice->GetCellIds ();

  uint16_t enb1CellId = enb1CellIds.at (0);
  uint16_t enb2CellId = enb2CellIds.at (0);

  NS_LOG_LOGIC ("LteEnbNetDevice #1 = " << enb1LteDev << " - CellId = " << enb1CellId);
  NS_LOG_LOGIC ("LteEnbNetDevice #2 = " << enb2LteDev << " - CellId = " << enb2CellId);

  enb1X2->AddX2Interface (enb1CellId, enb1X2Address, enb2CellIds, enb2X2Address);
  enb2X2->AddX2Interface (enb2CellId, enb2X2Address, enb1CellIds, enb1X2Address);

  enb1LteDevice->GetRrc ()->AddX2Neighbour (enb2CellId);
  enb2LteDevice->GetRrc ()->AddX2Neighbour (enb1CellId);
}


void 
NoBackhaulEpcHelper::AddUe (Ptr<NetDevice> ueDevice, uint64_t imsi)
{
  NS_LOG_FUNCTION (this << imsi << ueDevice);

  m_mmeApp->AddUe (imsi);
  m_pgwApp->AddUe (imsi);
}

uint8_t
NoBackhaulEpcHelper::ActivateEpsBearer (Ptr<NetDevice> ueDevice, uint64_t imsi,
                                        Ptr<EpcTft> tft, EpsBearer bearer)
{
  NS_LOG_FUNCTION (this << ueDevice << imsi);

  // we now retrieve the IPv4/IPv6 address of the UE and notify it to the SGW;
  // we couldn't do it before since address assignment is triggered by
  // the user simulation program, rather than done by the EPC   
  Ptr<Node> ueNode = ueDevice->GetNode (); 
  Ptr<Ipv4> ueIpv4 = ueNode->GetObject<Ipv4> ();
  Ptr<Ipv6> ueIpv6 = ueNode->GetObject<Ipv6> ();
  NS_ASSERT_MSG (ueIpv4 != 0 || ueIpv6 != 0, "UEs need to have IPv4/IPv6 installed before EPS bearers can be activated");

  if (ueIpv4)
    {
      int32_t interface =  ueIpv4->GetInterfaceForDevice (ueDevice);
      if (interface >= 0 && ueIpv4->GetNAddresses (interface) == 1)
        {
          Ipv4Address ueAddr = ueIpv4->GetAddress (interface, 0).GetLocal ();
          NS_LOG_LOGIC (" UE IPv4 address: " << ueAddr);
          m_pgwApp->SetUeAddress (imsi, ueAddr);
        }
    }
  if (ueIpv6)
    {
      int32_t interface6 =  ueIpv6->GetInterfaceForDevice (ueDevice);
      if (interface6 >= 0 && ueIpv6->GetNAddresses (interface6) == 2)
        {
          Ipv6Address ueAddr6 = ueIpv6->GetAddress (interface6, 1).GetAddress ();
          NS_LOG_LOGIC (" UE IPv6 address: " << ueAddr6);
          m_pgwApp->SetUeAddress6 (imsi, ueAddr6);
        }
    }
  uint8_t bearerId = m_mmeApp->AddBearer (imsi, tft, bearer);
  DoActivateEpsBearerForUe (ueDevice, tft, bearer);

  return bearerId;
}

void
NoBackhaulEpcHelper::DoActivateEpsBearerForUe (const Ptr<NetDevice> &ueDevice,
                                               const Ptr<EpcTft> &tft,
                                               const EpsBearer &bearer) const
{
  NS_LOG_FUNCTION (this);
  Ptr<LteUeNetDevice> ueLteDevice = DynamicCast<LteUeNetDevice> (ueDevice);
  if (ueLteDevice == nullptr)
    {
      // You may wonder why this is not an assert. Well, take a look in epc-test-s1u-downlink
      // and -uplink: we are using CSMA to simulate UEs.
      NS_LOG_WARN ("Unable to find LteUeNetDevice while activating the EPS bearer");
    }
  else
    {
      Simulator::ScheduleNow (&EpcUeNas::ActivateEpsBearer, ueLteDevice->GetNas (), bearer, tft);
    }
}

Ptr<Node>
NoBackhaulEpcHelper::GetPgwNode () const
{
  return m_pgw;
}

Ipv4InterfaceContainer 
NoBackhaulEpcHelper::AssignUeIpv4Address (NetDeviceContainer ueDevices)
{
  return m_uePgwAddressHelper.Assign (ueDevices);
}

Ipv6InterfaceContainer 
NoBackhaulEpcHelper::AssignUeIpv6Address (NetDeviceContainer ueDevices)
{
  for (NetDeviceContainer::Iterator iter = ueDevices.Begin ();
      iter != ueDevices.End ();
      iter ++)
    {
      Ptr<Icmpv6L4Protocol> icmpv6 = (*iter)->GetNode ()->GetObject<Icmpv6L4Protocol> ();
      icmpv6->SetAttribute ("DAD", BooleanValue (false));
    }
  return m_uePgwAddressHelper6.Assign (ueDevices);
}

Ipv4Address
NoBackhaulEpcHelper::GetUeDefaultGatewayAddress ()
{
  // return the address of the tun device
  return m_pgw->GetObject<Ipv4> ()->GetAddress (1, 0).GetLocal ();
}

Ipv6Address
NoBackhaulEpcHelper::GetUeDefaultGatewayAddress6 ()
{
  // return the address of the tun device
  return m_pgw->GetObject<Ipv6> ()->GetAddress (1, 1).GetAddress ();
}


Ptr<Node>
NoBackhaulEpcHelper::GetSgwNode () const
{
  return m_sgw;
}


void
NoBackhaulEpcHelper::AddS1Interface (Ptr<Node> enb, Ipv4Address enbAddress, Ipv4Address sgwAddress, std::vector<uint16_t> cellIds)
{
  NS_LOG_FUNCTION (this << enb << enbAddress << sgwAddress << cellIds.size ());

  // create S1-U socket for the ENB
  Ptr<Socket> enbS1uSocket = Socket::CreateSocket (enb, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  int retval = enbS1uSocket->Bind (InetSocketAddress (enbAddress, m_gtpuUdpPort));
  NS_ASSERT (retval == 0);

  Ptr<EpcEnbApplication> enbApp = enb->GetApplication (0)->GetObject<EpcEnbApplication> ();
  NS_ASSERT_MSG (enbApp != 0, "EpcEnbApplication not available");
  enbApp->AddS1Interface (enbS1uSocket, enbAddress, sgwAddress);

  NS_LOG_INFO ("Connect S1-AP interface");
  for (uint16_t cellId : cellIds)
    {
      NS_LOG_DEBUG ("Adding MME and SGW for cell ID " << cellId);
      m_mmeApp->AddEnb (cellId, enbAddress, enbApp->GetS1apSapEnb ());
      m_sgwApp->AddEnb (cellId, enbAddress, sgwAddress);
    }
  enbApp->SetS1apSapMme (m_mmeApp->GetS1apSapMme ());
}

} // namespace ns3
