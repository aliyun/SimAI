/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 UPB
 * Copyright (c) 2017 NITK Surathkal
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
 * Author: Radu Lupu <rlupu@elcom.pub.ro>
 *         Ankit Deepak <adadeepak8@gmail.com>
 *         Deepti Rajagopal <deeptir96@gmail.com>
 *
 */

#include <algorithm>
#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/nstime.h"
#include "ns3/ipv4-packet-info-tag.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/ipv4.h"
#include "dhcp-server.h"
#include "dhcp-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DhcpServer");
NS_OBJECT_ENSURE_REGISTERED (DhcpServer);

TypeId
DhcpServer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DhcpServer")
    .SetParent<Application> ()
    .AddConstructor<DhcpServer> ()
    .SetGroupName ("Internet-Apps")
    .AddAttribute ("LeaseTime",
                   "Lease for which address will be leased.",
                   TimeValue (Seconds (30)),
                   MakeTimeAccessor (&DhcpServer::m_lease),
                   MakeTimeChecker ())
    .AddAttribute ("RenewTime",
                   "Time after which client should renew.",
                   TimeValue (Seconds (15)),
                   MakeTimeAccessor (&DhcpServer::m_renew),
                   MakeTimeChecker ())
    .AddAttribute ("RebindTime",
                   "Time after which client should rebind.",
                   TimeValue (Seconds (25)),
                   MakeTimeAccessor (&DhcpServer::m_rebind),
                   MakeTimeChecker ())
    .AddAttribute ("PoolAddresses",
                   "Pool of addresses to provide on request.",
                   Ipv4AddressValue (),
                   MakeIpv4AddressAccessor (&DhcpServer::m_poolAddress),
                   MakeIpv4AddressChecker ())
    .AddAttribute ("FirstAddress",
                   "The First valid address that can be given.",
                   Ipv4AddressValue (),
                   MakeIpv4AddressAccessor (&DhcpServer::m_minAddress),
                   MakeIpv4AddressChecker ())
    .AddAttribute ("LastAddress",
                   "The Last valid address that can be given.",
                   Ipv4AddressValue (),
                   MakeIpv4AddressAccessor (&DhcpServer::m_maxAddress),
                   MakeIpv4AddressChecker ())
    .AddAttribute ("PoolMask",
                   "Mask of the pool of addresses.",
                   Ipv4MaskValue (),
                   MakeIpv4MaskAccessor (&DhcpServer::m_poolMask),
                   MakeIpv4MaskChecker ())
    .AddAttribute ("Gateway",
                   "Address of default gateway",
                   Ipv4AddressValue (),
                   MakeIpv4AddressAccessor (&DhcpServer::m_gateway),
                   MakeIpv4AddressChecker ())
  ;
  return tid;
}

DhcpServer::DhcpServer ()
{
  NS_LOG_FUNCTION (this);
}

DhcpServer::~DhcpServer ()
{
  NS_LOG_FUNCTION (this);
}

void
DhcpServer::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  Application::DoDispose ();
}

void DhcpServer::StartApplication (void)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (m_minAddress < m_maxAddress,"Invalid Address range");

  Ipv4Address myOwnAddress;

  if (m_socket)
    {
      NS_ABORT_MSG ("DHCP daemon is not (yet) meant to be started twice or more.");
    }

  uint32_t addrIndex;

  //add the DHCP local address to the leased addresses list, if it is defined!
  Ptr<Ipv4> ipv4 = GetNode ()->GetObject<Ipv4> ();
  int32_t ifIndex = ipv4->GetInterfaceForPrefix (m_poolAddress, m_poolMask);

  if (ifIndex < 0)
    {
      NS_ABORT_MSG ("DHCP daemon must be run on the same subnet it is assigning the addresses.");
    }

  for (addrIndex = 0; addrIndex < ipv4->GetNAddresses (ifIndex); addrIndex++)
    {
      if (ipv4->GetAddress (ifIndex, addrIndex).GetLocal ().CombineMask (m_poolMask) == m_poolAddress &&
          ipv4->GetAddress (ifIndex, addrIndex).GetLocal ().Get () >= m_minAddress.Get () &&
          ipv4->GetAddress (ifIndex, addrIndex).GetLocal ().Get () <= m_maxAddress.Get ())
        {
          // set infinite GRANTED_LEASED_TIME for my address

          myOwnAddress = ipv4->GetAddress (ifIndex, addrIndex).GetLocal ();
          m_leasedAddresses[Address ()] = std::make_pair (myOwnAddress, 0xffffffff);
          break;
        }
    }

  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  m_socket = Socket::CreateSocket (GetNode (), tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), PORT);
  m_socket->SetAllowBroadcast (true);
  m_socket->BindToNetDevice (ipv4->GetNetDevice (ifIndex));
  m_socket->Bind (local);
  m_socket->SetRecvPktInfo (true);

  uint32_t range = m_maxAddress.Get () - m_minAddress.Get () + 1;
  for (uint32_t searchSeq = 0; searchSeq < range; searchSeq ++)
    {
      Ipv4Address poolAddress = Ipv4Address (m_minAddress.Get () + searchSeq);
      if (poolAddress != myOwnAddress)
        {
          NS_LOG_LOGIC ("Adding " << poolAddress << " to the pool");
          m_availableAddresses.push_back (poolAddress);
        }
    }

  m_socket->SetRecvCallback (MakeCallback (&DhcpServer::NetHandler, this));
  m_expiredEvent = Simulator::Schedule (Seconds (1), &DhcpServer::TimerHandler, this);
}

void DhcpServer::StopApplication ()
{
  NS_LOG_FUNCTION (this);

  if (m_socket != 0)
    {
      m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    }

  m_leasedAddresses.clear ();
  m_expiredEvent.Cancel ();
}

void DhcpServer::TimerHandler ()
{
  NS_LOG_FUNCTION (this);

  // Set up timeout events and release of unsolicited addresses from the list
  LeasedAddressIter i;
  for (i = m_leasedAddresses.begin (); i != m_leasedAddresses.end (); i++)
    {
      // update the address state
      if (i->second.second != 0xffffffff && i->second.second != 0)
        {
          i->second.second--;
          if (i->second.second == 0)
            {
              NS_LOG_INFO ("Address leased state expired, address removed - " <<
                           "chaddr: " << i->first <<
                           " IP address " << i->second.first);
              i->second.second = 0;
              m_expiredAddresses.push_front (i->first);
            }
        }
    }
  m_expiredEvent = Simulator::Schedule (Seconds (1), &DhcpServer::TimerHandler, this);
}

void DhcpServer::NetHandler (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  DhcpHeader header;
  Ptr<Packet> packet = 0;
  Address from;
  packet = m_socket->RecvFrom (from);

  InetSocketAddress senderAddr = InetSocketAddress::ConvertFrom (from);

  Ipv4PacketInfoTag interfaceInfo;
  if (!packet->RemovePacketTag (interfaceInfo))
    {
      NS_ABORT_MSG ("No incoming interface on DHCP message, aborting.");
    }
  uint32_t incomingIf = interfaceInfo.GetRecvIf ();
  Ptr<NetDevice> iDev = GetNode ()->GetDevice (incomingIf);

  if (packet->RemoveHeader (header) == 0)
    {
      return;
    }
  if (header.GetType () == DhcpHeader::DHCPDISCOVER)
    {
      SendOffer (iDev, header, senderAddr);
    }
  if (header.GetType () == DhcpHeader::DHCPREQ && (header.GetReq ()).Get () >= m_minAddress.Get () && (header.GetReq ()).Get () <= m_maxAddress.Get ())
    {
      SendAck (iDev, header, senderAddr);
    }
}

void DhcpServer::SendOffer (Ptr<NetDevice> iDev, DhcpHeader header, InetSocketAddress from)
{
  NS_LOG_FUNCTION (this << iDev << header << from);

  DhcpHeader newDhcpHeader;
  Address sourceChaddr = header.GetChaddr ();
  uint32_t tran = header.GetTran ();
  Ptr<Packet> packet = 0;
  Ipv4Address offeredAddress;

  NS_LOG_INFO ("DHCP DISCOVER from: " << from.GetIpv4 () << " source port: " <<  from.GetPort ());

  LeasedAddressIter iter = m_leasedAddresses.find (sourceChaddr);
  if (iter != m_leasedAddresses.end ())
    {
      // We know this client from some time ago
      if (m_leasedAddresses[sourceChaddr].second != 0 && m_leasedAddresses[sourceChaddr].second != 0xffffffff)
        {
          NS_LOG_LOGIC ("This client is sending a DISCOVER but it has still a lease active - perhaps it didn't shut down gracefully: " << sourceChaddr);
        }

      m_expiredAddresses.remove (sourceChaddr);
      offeredAddress = m_leasedAddresses[sourceChaddr].first;

    }
  else
    {
      // No previous record of the client, we must find a suitable address and create a record.
      if (!m_availableAddresses.empty ())
        {
          // use an address never used before (if there is one)
          offeredAddress = m_availableAddresses.front ();
          m_availableAddresses.pop_front ();
        }
      else
        {
          // there's still hope: reuse the old ones.
          if (!m_expiredAddresses.empty ())
            {
              Address oldestChaddr = m_expiredAddresses.back ();
              m_expiredAddresses.pop_back ();
              offeredAddress = m_leasedAddresses[oldestChaddr].first;
              m_leasedAddresses.erase (oldestChaddr);
            }
        }
    }

  if (offeredAddress != Ipv4Address ())
    {
      m_leasedAddresses[sourceChaddr] = std::make_pair (offeredAddress, m_lease.GetSeconds ());

      packet = Create<Packet> ();
      newDhcpHeader.ResetOpt ();
      newDhcpHeader.SetType (DhcpHeader::DHCPOFFER);
      newDhcpHeader.SetChaddr (sourceChaddr);
      newDhcpHeader.SetYiaddr (offeredAddress);

      Ptr<Ipv4> ipv4 = GetNode ()->GetObject<Ipv4> ();
      Ipv4Address myAddress = ipv4->SelectSourceAddress (iDev, offeredAddress, Ipv4InterfaceAddress::InterfaceAddressScope_e::GLOBAL);

      newDhcpHeader.SetDhcps (myAddress);
      newDhcpHeader.SetMask (m_poolMask.Get ());
      newDhcpHeader.SetTran (tran);
      newDhcpHeader.SetLease (m_lease.GetSeconds ());
      newDhcpHeader.SetRenew (m_renew.GetSeconds ());
      newDhcpHeader.SetRebind (m_rebind.GetSeconds ());
      newDhcpHeader.SetTime ();
      if (m_gateway != Ipv4Address ())
        {
          newDhcpHeader.SetRouter (m_gateway);
        }
      packet->AddHeader (newDhcpHeader);

      if ((m_socket->SendTo (packet, 0, InetSocketAddress (Ipv4Address ("255.255.255.255"), from.GetPort ()))) >= 0)
        {
          NS_LOG_INFO ("DHCP OFFER" << " Offered Address: " << offeredAddress);
        }
      else
        {
          NS_LOG_INFO ("Error while sending DHCP OFFER");
        }
    }
}

void DhcpServer::SendAck (Ptr<NetDevice> iDev, DhcpHeader header, InetSocketAddress from)
{
  NS_LOG_FUNCTION (this << iDev << header << from);

  DhcpHeader newDhcpHeader;
  Address sourceChaddr = header.GetChaddr ();
  uint32_t tran = header.GetTran ();
  Ptr<Packet> packet = 0;
  Ipv4Address address = header.GetReq ();

  NS_LOG_INFO ("DHCP REQUEST from: " << from.GetIpv4 () <<
               " source port: " <<  from.GetPort () <<
               " - refreshed addr: " << address);

  LeasedAddressIter iter;
  iter = m_leasedAddresses.find (sourceChaddr);
  if (iter != m_leasedAddresses.end ())
    {
      // update the lease time of this address - send ACK
      (iter->second.second) += m_lease.GetSeconds ();
      packet = Create<Packet> ();
      newDhcpHeader.ResetOpt ();
      newDhcpHeader.SetType (DhcpHeader::DHCPACK);
      newDhcpHeader.SetChaddr (sourceChaddr);
      newDhcpHeader.SetYiaddr (address);
      newDhcpHeader.SetTran (tran);
      newDhcpHeader.SetTime ();
      packet->AddHeader (newDhcpHeader);
      if (from.GetIpv4 () != address)
        {
          m_socket->SendTo (packet, 0, InetSocketAddress (Ipv4Address ("255.255.255.255"), from.GetPort ()));
        }
      else
        {
          m_socket->SendTo (packet, 0, from);
        }
    }
  else
    {
      // Deleted or expired lease - send NACK
      packet = Create<Packet> ();
      newDhcpHeader.ResetOpt ();
      newDhcpHeader.SetType (DhcpHeader::DHCPNACK);
      newDhcpHeader.SetChaddr (sourceChaddr);
      newDhcpHeader.SetYiaddr (address);
      newDhcpHeader.SetTran (tran);
      newDhcpHeader.SetTime ();
      packet->AddHeader (newDhcpHeader);
      if (from.GetIpv4 () != address)
        {
          m_socket->SendTo (packet, 0, InetSocketAddress (Ipv4Address ("255.255.255.255"), from.GetPort ()));
        }
      else
        {
          m_socket->SendTo (packet, 0, from);
        }
      NS_LOG_INFO ("IP addr does not exists or released!");
    }
}

void DhcpServer::AddStaticDhcpEntry (Address chaddr, Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << chaddr << addr);
  Address cleanedCaddr;

  NS_ASSERT_MSG (addr.Get () >= m_minAddress.Get () && addr.Get () <= m_maxAddress.Get (),
                 "Required address is not in the pool " << addr << " is not in [" << m_minAddress << ", " << m_maxAddress << "]");

  // We need to cleanup the type from the stored chaddr, or later we'll fail to compare it.
  // Moreover, the length is always 16, because chaddr is 16 bytes.
  uint8_t buffer[Address::MAX_SIZE];
  std::memset (buffer, 0, Address::MAX_SIZE);
  uint32_t len = chaddr.CopyTo (buffer);
  NS_ASSERT_MSG (len <= 16, "DHCP server can not handle a chaddr larger than 16 bytes");
  cleanedCaddr.CopyFrom (buffer, 16);

  NS_ASSERT_MSG (m_leasedAddresses.find (cleanedCaddr) == m_leasedAddresses.end (),
                 "Client has already an active lease: " << m_leasedAddresses[cleanedCaddr].first);

  AvailableAddress::iterator it = find (m_availableAddresses.begin (), m_availableAddresses.end (), addr);
  NS_ASSERT_MSG (it == m_availableAddresses.end (),
                 "Required address is not available (perhaps it has been already assigned): " << addr);

  m_availableAddresses.remove (addr);
  m_leasedAddresses[cleanedCaddr] = std::make_pair (addr, 0xffffffff);
}

} // Namespace ns3
