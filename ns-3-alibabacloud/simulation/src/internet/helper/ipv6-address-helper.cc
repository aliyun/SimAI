/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/ptr.h"
#include "ns3/node.h"
#include "ns3/net-device.h"
#include "ns3/loopback-net-device.h"
#include "ns3/mac16-address.h"
#include "ns3/mac48-address.h"
#include "ns3/mac64-address.h"
#include "ns3/ipv6.h"
#include "ns3/ipv6-address-generator.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/traffic-control-layer.h"
#include "ns3/net-device-queue-interface.h"

#include "ipv6-address-helper.h"

namespace ns3 
{

NS_LOG_COMPONENT_DEFINE ("Ipv6AddressHelper");

Ipv6AddressHelper::Ipv6AddressHelper ()
{
  NS_LOG_FUNCTION (this);
  m_network = Ipv6Address ("2001:db8::");
  m_prefix = 64;
  m_address = Ipv6Address ("::1");
  m_base = m_address;
}

Ipv6AddressHelper::Ipv6AddressHelper (Ipv6Address network, Ipv6Prefix prefix,
                                      Ipv6Address base)
{
  NS_LOG_FUNCTION (this << network << prefix << base);

  m_network = network;
  m_prefix = prefix;
  m_address = base;
  m_base = base;

  NS_ASSERT_MSG (m_network == network.CombinePrefix (prefix),
                 "Ipv6AddressHelper: network address and prefix mismatch: " << m_network << " " << m_prefix);

  NS_ASSERT_MSG (base.CombinePrefix (prefix) == Ipv6Address::GetZero (),
                 "Ipv6AddressHelper: base address and prefix mismatch: " << base << " " << m_prefix);
}

void Ipv6AddressHelper::SetBase (Ipv6Address network, Ipv6Prefix prefix,
                                 Ipv6Address base)
{
  NS_LOG_FUNCTION (this << network << prefix << base);

  m_network = network;
  m_prefix = prefix;
  m_address = base;
  m_base = base;

  NS_ASSERT_MSG (m_network == network.CombinePrefix (prefix),
                 "Ipv6AddressHelper::SetBase(): network address and prefix mismatch: " << m_network << " " << m_prefix);

  NS_ASSERT_MSG (base.CombinePrefix (prefix) == Ipv6Address::GetZero (),
                 "Ipv6AddressHelper::SetBase(): base address and prefix mismatch: " << base << " " << m_prefix);
}


Ipv6Address Ipv6AddressHelper::NewAddress (Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  if (Mac64Address::IsMatchingType (addr))
    {
      Ipv6Address address = Ipv6Address::MakeAutoconfiguredAddress (Mac64Address::ConvertFrom (addr), m_network);
      Ipv6AddressGenerator::AddAllocated (address);
      return address;
    }
  else if (Mac48Address::IsMatchingType (addr))
    {
      Ipv6Address address = Ipv6Address::MakeAutoconfiguredAddress (Mac48Address::ConvertFrom (addr), m_network);
      Ipv6AddressGenerator::AddAllocated (address);
      return address;
    }
  else if (Mac16Address::IsMatchingType (addr))
    {
      Ipv6Address address = Ipv6Address::MakeAutoconfiguredAddress (Mac16Address::ConvertFrom (addr), m_network);
      Ipv6AddressGenerator::AddAllocated (address);
      return address;
    }
  else if (Mac8Address::IsMatchingType (addr))
    {
      Ipv6Address address = Ipv6Address::MakeAutoconfiguredAddress (Mac8Address::ConvertFrom (addr), m_network);
      Ipv6AddressGenerator::AddAllocated (address);
      return address;
    }
  else
    {
      NS_FATAL_ERROR ("Did not pass in a valid Mac Address (8, 16, 48 or 64 bits)");
    }
  /* never reached */
  return Ipv6Address ("::");
}

Ipv6Address Ipv6AddressHelper::NewAddress (void)
{
  NS_LOG_FUNCTION (this);
//
// The way this is expected to be used is that an address and network number
// are initialized, and then NewAddress() is called repeatedly to allocate and
// get new addresses on a given subnet.  The client will expect that the first
// address she gets back is the one she used to initialize the generator with.
// This implies that this operation is a post-increment.

  uint8_t netBuf[16];
  uint8_t hostBuf[16];
  uint8_t addrBuf[16];
  m_network.GetBytes (netBuf);
  m_address.GetBytes (hostBuf);

  NS_ASSERT_MSG (m_address.CombinePrefix (m_prefix) == Ipv6Address::GetZero (),
                 "Ipv6AddressHelper::NewAddress(): Too many hosts in the network: " << m_address << " " << m_prefix);

  for (uint8_t i=0; i<16; i++)
    {
      addrBuf[i] = netBuf[i] | hostBuf[i];
    }

  Ipv6Address addr = Ipv6Address (addrBuf);

  // Remember: hostBuf[15] is the Least Significant Byte.
  uint16_t sum;
  sum = static_cast<uint16_t> (hostBuf[15]) + 1;
  hostBuf[15] += 1;
  for (uint8_t index=0; index<15; index++)
    {
      if (sum > hostBuf[15-index])
        {
          sum = static_cast<uint16_t> (hostBuf[14-index]) + 1;
          hostBuf[14-index] += 1;
        }
      else
        {
          break;
        }
    }
  m_address = Ipv6Address (hostBuf);

  Ipv6AddressGenerator::AddAllocated (addr);
  return addr;
}

void Ipv6AddressHelper::NewNetwork (void)
{
  NS_LOG_FUNCTION (this);

  uint8_t netBuf[16];
  uint8_t addBuf[16];
  m_network.GetBytes (netBuf);

  uint8_t prefixIndex = (m_prefix.GetPrefixLength ()-1)/8;
  uint8_t prefixPosition = (8-(m_prefix.GetPrefixLength ()%8))%8;

  for (uint8_t index=0; index<16; index++)
    {
      addBuf[index] = 0;
      if (index==prefixIndex)
        {
          addBuf[index] = (1<<prefixPosition);
        }
    }

  uint16_t sum[16];
  for (uint8_t index=0; index<16; index++)
    {
      sum[index] = static_cast<uint16_t> (netBuf[index]) + static_cast<uint16_t> (addBuf[index]);
      netBuf[index] += addBuf[index];
    }

  for (uint8_t index=0; index<15; index++)
    {
      if (sum[15-index] > netBuf[15-index])
        {
          sum[14-index] = static_cast<uint16_t> (netBuf[14-index]) + 1;
          netBuf[14-index] += 1;
        }
    }

  m_network = Ipv6Address (netBuf);
  m_address = m_base;
}

Ipv6InterfaceContainer Ipv6AddressHelper::Assign (const NetDeviceContainer &c)
{
  NS_LOG_FUNCTION (this);
  std::vector<bool> withConfiguration;
  for (uint32_t i = 0; i < c.GetN (); ++i) 
    {
      withConfiguration.push_back (true);
    }
  return Assign (c, withConfiguration);
}

Ipv6InterfaceContainer Ipv6AddressHelper::Assign (const NetDeviceContainer &c, std::vector<bool> withConfiguration)
{
  NS_LOG_FUNCTION (this);
  std::vector<bool> onLink;
  for (uint32_t i = 0; i < c.GetN (); ++i)
    {
      onLink.push_back (true);
    }
  return Assign (c, withConfiguration, onLink);
}

Ipv6InterfaceContainer Ipv6AddressHelper::Assign (const NetDeviceContainer &c, std::vector<bool> withConfiguration, std::vector<bool> onLink)
{
  NS_LOG_FUNCTION (this);
  Ipv6InterfaceContainer retval;
  for (uint32_t i = 0; i < c.GetN (); ++i)
    {
      Ptr<NetDevice> device = c.Get (i);

      Ptr<Node> node = device->GetNode ();
      NS_ASSERT_MSG (node, "Ipv6AddressHelper::Allocate (): Bad node");

      Ptr<Ipv6> ipv6 = node->GetObject<Ipv6> ();
      NS_ASSERT_MSG (ipv6, "Ipv6AddressHelper::Allocate (): Bad ipv6");

      int32_t ifIndex = ipv6->GetInterfaceForDevice (device);
      if (ifIndex == -1)
        {
          ifIndex = ipv6->AddInterface (device);
        }
      NS_ASSERT_MSG (ifIndex >= 0, "Ipv6AddressHelper::Allocate (): "
                     "Interface index not found");

      // the first round is to make sure that the interface is set up, including its link-local addresses.
      ipv6->SetUp (ifIndex);

      ipv6->SetMetric (ifIndex, 1);

      if (withConfiguration.at (i))
        {
          Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (NewAddress (device->GetAddress ()), Ipv6Prefix (64), onLink.at(i));
          ipv6->AddAddress (ifIndex, ipv6Addr, onLink.at(i));
        }

      ipv6->SetUp (ifIndex);
      retval.Add (ipv6, ifIndex);

      // Install the default traffic control configuration if the traffic
      // control layer has been aggregated, if this is not
      // a loopback interface, and there is no queue disc installed already
      Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer> ();
      if (tc && DynamicCast<LoopbackNetDevice> (device) == 0 && tc->GetRootQueueDiscOnDevice (device) == 0)
        {
          Ptr<NetDeviceQueueInterface> ndqi = device->GetObject<NetDeviceQueueInterface> ();
          // It is useless to install a queue disc if the device has no
          // NetDeviceQueueInterface attached: the device queue is never
          // stopped and every packet enqueued in the queue disc is
          // immediately dequeued, hence there will never be backlog
          if (ndqi)
            {
              std::size_t nTxQueues = ndqi->GetNTxQueues ();
              NS_LOG_LOGIC ("Installing default traffic control configuration ("
                            << nTxQueues << " device queue(s))");
              TrafficControlHelper tcHelper = TrafficControlHelper::Default (nTxQueues);
              tcHelper.Install (device);
            }
        }
    }
  return retval;
}

// Helper API that is redundant with Assign (c, false);
Ipv6InterfaceContainer Ipv6AddressHelper::AssignWithoutAddress (const NetDeviceContainer &c)
{
  NS_LOG_FUNCTION (this);
  std::vector<bool> withConfiguration;
  for (uint32_t i = 0; i < c.GetN (); ++i) 
    {
      withConfiguration.push_back (false);
    }
  return Assign (c, withConfiguration);
}

Ipv6InterfaceContainer Ipv6AddressHelper::AssignWithoutOnLink (const NetDeviceContainer &c)
{
  NS_LOG_FUNCTION (this);
  std::vector<bool> withConfiguration;
  std::vector<bool> onLink;
  for (uint32_t i = 0; i < c.GetN (); ++i)
    {
      withConfiguration.push_back (true);
      onLink.push_back (false);
    }
  return Assign (c, withConfiguration, onLink);
}

} /* namespace ns3 */

