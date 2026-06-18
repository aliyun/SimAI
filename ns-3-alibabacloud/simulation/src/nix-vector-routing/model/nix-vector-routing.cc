/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Georgia Institute of Technology
 * Copyright (c) 2021 NITK Surathkal
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
 * This file is adapted from the old ipv4-nix-vector-routing.cc.
 *
 * Authors: Josh Pelkey <jpelkey@gatech.edu>
 * 
 * Modified by: Ameya Deshpande <ameyanrd@outlook.com>
 */

#include <queue>
#include <iomanip>

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/names.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/loopback-net-device.h"

#include "nix-vector-routing.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NixVectorRouting");

NS_OBJECT_TEMPLATE_CLASS_DEFINE (NixVectorRouting, Ipv4RoutingProtocol);
NS_OBJECT_TEMPLATE_CLASS_DEFINE (NixVectorRouting, Ipv6RoutingProtocol);

#ifdef NS3_MTP
template <typename T>
std::atomic<bool> NixVectorRouting<T>::g_isCacheDirty (false);

template <typename T>
std::atomic<bool> NixVectorRouting<T>::g_cacheFlushing (false);

template <typename T>
std::atomic<bool> NixVectorRouting<T>::g_isMapBuilt (false);

template <typename T>
std::atomic<bool> NixVectorRouting<T>::g_mapBuilding (false);
#else
template <typename T>
bool NixVectorRouting<T>::g_isCacheDirty = false;
#endif

// Epoch starts from one to make it easier to spot an uninitialized NixVector during debug.
template <typename T>
uint32_t NixVectorRouting<T>::g_epoch = 1;

template <typename T>
typename NixVectorRouting<T>::IpAddressToNodeMap NixVectorRouting<T>::g_ipAddressToNodeMap;

template <typename T>
typename NixVectorRouting<T>::NetDeviceToIpInterfaceMap NixVectorRouting<T>::g_netdeviceToIpInterfaceMap;

template <typename T>
TypeId 
NixVectorRouting<T>::GetTypeId (void)
{
  std::string Tname = GetTypeParamName<NixVectorRouting<T> > ();
  std::string name = (Tname == "Ipv4RoutingProtocol" ? "Ipv4" : "Ipv6");
  static TypeId tid = TypeId ("ns3::" + name + "NixVectorRouting")
    .SetParent<T> ()
    .SetGroupName ("NixVectorRouting")
    .template AddConstructor<NixVectorRouting<T> > ()
  ;
  return tid;
}

template <typename T>
NixVectorRouting<T>::NixVectorRouting ()
  : m_totalNeighbors (0)
{
  NS_LOG_FUNCTION_NOARGS ();
}

template <typename T>
NixVectorRouting<T>::~NixVectorRouting ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

template <typename T>
void
NixVectorRouting<T>::SetIpv4 (Ptr<Ip> ipv4)
{
  NS_ASSERT (ipv4 != 0);
  NS_ASSERT (m_ip == 0);
  NS_LOG_DEBUG ("Created Ipv4NixVectorProtocol");

  m_ip = ipv4;
}

template <typename T>
void
NixVectorRouting<T>::SetIpv6 (Ptr<Ip> ipv6)
{
  NS_ASSERT (ipv6 != 0);
  NS_ASSERT (m_ip == 0);
  NS_LOG_DEBUG ("Created Ipv6NixVectorProtocol");

  m_ip = ipv6;
}

template <typename T>
void
NixVectorRouting<T>::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  for (uint32_t i = 0 ; i < m_ip->GetNInterfaces (); i++)
    {
      m_ip->SetForwarding (i, true);
    }

  T::DoInitialize ();
}

template <typename T>
void 
NixVectorRouting<T>::DoDispose ()
{
  NS_LOG_FUNCTION_NOARGS ();

  m_node = 0;
  m_ip = 0;

  T::DoDispose ();
}

template <typename T>
void
NixVectorRouting<T>::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION_NOARGS ();

  m_node = node;
}

template <typename T>
void
NixVectorRouting<T>::FlushGlobalNixRoutingCache (void) const
{
  NS_LOG_FUNCTION_NOARGS ();

  NodeList::Iterator listEnd = NodeList::End ();
  for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
    {
      Ptr<Node> node = *i;
      Ptr<NixVectorRouting<T> > rp = node->GetObject<NixVectorRouting> ();
      if (!rp)
        {
          continue;
        }
      NS_LOG_LOGIC ("Flushing Nix caches.");
      rp->FlushNixCache ();
      rp->FlushIpRouteCache ();
      rp->m_totalNeighbors = 0;
    }

  // IP address to node mapping is potentially invalid so clear it.
  // Will be repopulated in lazy evaluation when mapping is needed.
  g_ipAddressToNodeMap.clear ();
#ifdef NS3_MTP
  g_isMapBuilt.store (false, std::memory_order_release);
#endif
}

template <typename T>
void
NixVectorRouting<T>::FlushNixCache (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  m_nixCache.clear ();
}

template <typename T>
void
NixVectorRouting<T>::FlushIpRouteCache (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  m_ipRouteCache.clear ();
}

template <typename T>
Ptr<NixVector>
NixVectorRouting<T>::GetNixVector (Ptr<Node> source, IpAddress dest, Ptr<NetDevice> oif) const
{
  NS_LOG_FUNCTION (this << source << dest << oif);

  Ptr<NixVector> nixVector = Create<NixVector> ();
  nixVector->SetEpoch (g_epoch);

  // not in cache, must build the nix vector
  // First, we have to figure out the nodes 
  // associated with these IPs
  Ptr<Node> destNode = GetNodeByIp (dest);
  if (destNode == 0)
    {
      NS_LOG_ERROR ("No routing path exists");
      return 0;
    }

  // if source == dest, then we have a special case
  /// \internal
  /// Do not process packets to self (see \bugid{1308})
  if (source == destNode)
    {
      NS_LOG_DEBUG ("Do not process packets to self");
      return 0;
    }
  else
    {
      // otherwise proceed as normal 
      // and build the nix vector
      std::vector< Ptr<Node> > parentVector;

      if (BFS (NodeList::GetNNodes (), source, destNode, parentVector, oif))
        {
          if (BuildNixVector (parentVector, source->GetId (), destNode->GetId (), nixVector))
            {
              return nixVector;
            }
          else
            {
              NS_LOG_ERROR ("No routing path exists");
              return 0;
            }
        }
      else
        {
          NS_LOG_ERROR ("No routing path exists");
          return 0;
        }

    }
}

template <typename T>
Ptr<NixVector>
NixVectorRouting<T>::GetNixVectorInCache (const IpAddress &address, bool &foundInCache) const
{
  NS_LOG_FUNCTION (this << address);

  CheckCacheStateAndFlush ();

  typename NixMap_t::iterator iter = m_nixCache.find (address);
  if (iter != m_nixCache.end ())
    {
      NS_LOG_LOGIC ("Found Nix-vector in cache.");
      foundInCache = true;
      return iter->second;
    }

  // not in cache
  foundInCache = false;
  return 0;
}

template <typename T>
Ptr<typename NixVectorRouting<T>::IpRoute>
NixVectorRouting<T>::GetIpRouteInCache (IpAddress address)
{
  NS_LOG_FUNCTION (this << address);

  CheckCacheStateAndFlush ();

  typename IpRouteMap_t::iterator iter = m_ipRouteCache.find (address);
  if (iter != m_ipRouteCache.end ())
    {
      NS_LOG_LOGIC ("Found IpRoute in cache.");
      return iter->second;
    }

  // not in cache
  return 0;
}

template <typename T>
bool
NixVectorRouting<T>::BuildNixVector (const std::vector< Ptr<Node> > & parentVector, uint32_t source, uint32_t dest, Ptr<NixVector> nixVector) const
{
  NS_LOG_FUNCTION (this << parentVector << source << dest << nixVector);

  if (source == dest)
    {
      return true;
    }

  if (parentVector.at (dest) == 0)
    {
      return false;
    }

  Ptr<Node> parentNode = parentVector.at (dest);

  uint32_t numberOfDevices = parentNode->GetNDevices ();
  uint32_t destId = 0;
  uint32_t totalNeighbors = 0;

  // scan through the net devices on the T node
  // and then look at the nodes adjacent to them
  for (uint32_t i = 0; i < numberOfDevices; i++)
    {
      // Get a net device from the node
      // as well as the channel, and figure
      // out the adjacent net devices
      Ptr<NetDevice> localNetDevice = parentNode->GetDevice (i);
      if (localNetDevice->IsBridge ())
        {
          continue;
        }
      Ptr<Channel> channel = localNetDevice->GetChannel ();
      if (channel == 0)
        {
          continue;
        }

      // this function takes in the local net dev, and channel, and
      // writes to the netDeviceContainer the adjacent net devs
      NetDeviceContainer netDeviceContainer;
      GetAdjacentNetDevices (localNetDevice, channel, netDeviceContainer);

      // Finally we can get the adjacent nodes
      // and scan through them.  If we find the 
      // node that matches "dest" then we can add 
      // the index  to the nix vector.
      // the index corresponds to the neighbor index
      uint32_t offset = 0;
      for (NetDeviceContainer::Iterator iter = netDeviceContainer.Begin (); iter != netDeviceContainer.End (); iter++)
        {
          Ptr<Node> remoteNode = (*iter)->GetNode ();

          if (remoteNode->GetId () == dest)
            {
              destId = totalNeighbors + offset;
            }
          offset += 1;
        }

      totalNeighbors += netDeviceContainer.GetN ();
    }
  NS_LOG_LOGIC ("Adding Nix: " << destId << " with " 
                               << nixVector->BitCount (totalNeighbors) << " bits, for node " << parentNode->GetId ());
  nixVector->AddNeighborIndex (destId, nixVector->BitCount (totalNeighbors));

  // recurse through T vector, grabbing the path
  // and building the nix vector
  BuildNixVector (parentVector, source, (parentVector.at (dest))->GetId (), nixVector);
  return true;
}

template <typename T>
void
NixVectorRouting<T>::GetAdjacentNetDevices (Ptr<NetDevice> netDevice, Ptr<Channel> channel, NetDeviceContainer & netDeviceContainer) const
{
  NS_LOG_FUNCTION (this << netDevice << channel);

  Ptr<IpInterface> netDeviceInterface = GetInterfaceByNetDevice (netDevice);
  if (netDeviceInterface == 0 || !netDeviceInterface->IsUp ())
    {
      NS_LOG_LOGIC ("IpInterface either doesn't exist or is down");
      return;
    }

  uint32_t netDeviceAddresses = netDeviceInterface->GetNAddresses ();

  for (std::size_t i = 0; i < channel->GetNDevices (); i++)
    {
      Ptr<NetDevice> remoteDevice = channel->GetDevice (i);
      if (remoteDevice != netDevice)
        {
          // Compare if the remoteDevice shares a common subnet with remoteDevice
          Ptr<IpInterface> remoteDeviceInterface = GetInterfaceByNetDevice (remoteDevice);
          if (remoteDeviceInterface == 0 || !remoteDeviceInterface->IsUp ())
            {
              NS_LOG_LOGIC ("IpInterface either doesn't exist or is down");
              continue;
            }

          uint32_t remoteDeviceAddresses = remoteDeviceInterface->GetNAddresses ();
          bool commonSubnetFound = false;

          for (uint32_t j = 0; j < netDeviceAddresses; ++j)
            {
              IpInterfaceAddress netDeviceIfAddr = netDeviceInterface->GetAddress (j);
              if constexpr (!IsIpv4::value)
                {
                  if (netDeviceIfAddr.GetScope () == Ipv6InterfaceAddress::LINKLOCAL)
                    {
                      continue;
                    }
                }
              for (uint32_t k = 0; k < remoteDeviceAddresses; ++k)
                {
                  IpInterfaceAddress remoteDeviceIfAddr = remoteDeviceInterface->GetAddress (k);
                  if constexpr (!IsIpv4::value)
                    {
                      if (remoteDeviceIfAddr.GetScope () == Ipv6InterfaceAddress::LINKLOCAL)
                        {
                          continue;
                        }
                    }
                  if (netDeviceIfAddr.IsInSameSubnet (remoteDeviceIfAddr.GetAddress ()))
                    {
                      commonSubnetFound = true;
                      break;
                    }
                }

              if (commonSubnetFound)
                {
                  break;
                }
            }

          if (!commonSubnetFound)
            {
              continue;
            }

          Ptr<BridgeNetDevice> bd = NetDeviceIsBridged (remoteDevice);
          // we have a bridged device, we need to add all 
          // bridged devices
          if (bd)
            {
              NS_LOG_LOGIC ("Looking through bridge ports of bridge net device " << bd);
              for (uint32_t j = 0; j < bd->GetNBridgePorts (); ++j)
                {
                  Ptr<NetDevice> ndBridged = bd->GetBridgePort (j);
                  if (ndBridged == remoteDevice)
                    {
                      NS_LOG_LOGIC ("That bridge port is me, don't walk backward");
                      continue;
                    }
                  Ptr<Channel> chBridged = ndBridged->GetChannel ();
                  if (chBridged == 0)
                    {
                      continue;
                    }
                  GetAdjacentNetDevices (ndBridged, chBridged, netDeviceContainer);
                }
            }
          else
            {
              netDeviceContainer.Add (channel->GetDevice (i));
            }
        }
    }
}

template <typename T>
void
NixVectorRouting<T>::BuildIpAddressToNodeMap (void) const
{
  NS_LOG_FUNCTION_NOARGS ();

  for (NodeList::Iterator it = NodeList::Begin (); it != NodeList::End (); ++it)
    {
      Ptr<Node> node = *it;
      Ptr<IpL3Protocol> ip = node->GetObject<IpL3Protocol> ();

      if(ip)
        {
          uint32_t numberOfDevices = node->GetNDevices ();

          for (uint32_t deviceId = 0; deviceId < numberOfDevices; deviceId++)
            {
              Ptr<NetDevice> device = node->GetDevice (deviceId);

              // If this is not a loopback device add the IP address to the map
              if ( !DynamicCast<LoopbackNetDevice>(device) )
                {
                  int32_t interfaceIndex = (ip)->GetInterfaceForDevice (node->GetDevice (deviceId));
                  if (interfaceIndex != -1)
                    {
                      g_netdeviceToIpInterfaceMap[device] = (ip)->GetInterface (interfaceIndex);

                      uint32_t numberOfAddresses = ip->GetNAddresses (interfaceIndex);
                      for (uint32_t addressIndex = 0; addressIndex < numberOfAddresses; addressIndex++)
                        {
                          IpInterfaceAddress ifAddr = ip->GetAddress (interfaceIndex, addressIndex);
                          IpAddress addr = ifAddr.GetAddress ();

                          NS_ABORT_MSG_IF (g_ipAddressToNodeMap.count (addr),
                                          "Duplicate IP address (" << addr << ") found during NIX Vector map construction for node " << node->GetId ());

                          NS_LOG_LOGIC ("Adding IP address " << addr << " for node " << node->GetId () << " to NIX Vector IP address to node map");
                          g_ipAddressToNodeMap[addr] = node;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
Ptr<Node>
NixVectorRouting<T>::GetNodeByIp (IpAddress dest) const
{
  NS_LOG_FUNCTION (this << dest);

  // Populate lookup table if is empty.
#ifdef NS3_MTP
  if (!g_isMapBuilt.load (std::memory_order_acquire))
    {
      if (g_mapBuilding.exchange (true, std::memory_order_relaxed))
        {
          while (!g_isMapBuilt.load (std::memory_order_acquire))
            ;
        }
      else
        {
          BuildIpAddressToNodeMap ();
          g_isMapBuilt.store (true, std::memory_order_release);
          g_mapBuilding.store (false, std::memory_order_release);
        }
    }
#else
  if ( g_ipAddressToNodeMap.empty () )
    {
      BuildIpAddressToNodeMap ();
    }
#endif

  Ptr<Node> destNode;

  typename IpAddressToNodeMap::iterator iter = g_ipAddressToNodeMap.find(dest);

  if(iter == g_ipAddressToNodeMap.end ())
    {
      NS_LOG_ERROR ("Couldn't find dest node given the IP" << dest);
      destNode = 0;
    }
  else
    {
      destNode = iter -> second;
    }

  return destNode;
}

template <typename T>
Ptr<typename NixVectorRouting<T>::IpInterface>
NixVectorRouting<T>::GetInterfaceByNetDevice (Ptr<NetDevice> netDevice) const
{
  // Populate lookup table if is empty.
#ifdef NS3_MTP
  if (!g_isMapBuilt.load (std::memory_order_acquire))
    {
      if (g_mapBuilding.exchange (true, std::memory_order_relaxed))
        {
          while (!g_isMapBuilt.load (std::memory_order_acquire))
            ;
        }
      else
        {
          BuildIpAddressToNodeMap ();
          g_isMapBuilt.store (true, std::memory_order_release);
          g_mapBuilding.store (false, std::memory_order_release);
        }
    }
#else
  if ( g_netdeviceToIpInterfaceMap.empty () )
    {
      BuildIpAddressToNodeMap ();
    }
#endif

  Ptr<IpInterface> ipInterface;

  typename NetDeviceToIpInterfaceMap::iterator iter = g_netdeviceToIpInterfaceMap.find(netDevice);

  if(iter == g_netdeviceToIpInterfaceMap.end ())
    {
      NS_LOG_ERROR ("Couldn't find IpInterface node given the NetDevice" << netDevice);
      ipInterface = 0;
    }
  else
    {
      ipInterface = iter -> second;
    }

  return ipInterface;
}

template <typename T>
uint32_t
NixVectorRouting<T>::FindTotalNeighbors (Ptr<Node> node) const
{
  NS_LOG_FUNCTION (this << node);

  uint32_t numberOfDevices = node->GetNDevices ();
  uint32_t totalNeighbors = 0;

  // scan through the net devices on the T node
  // and then look at the nodes adjacent to them
  for (uint32_t i = 0; i < numberOfDevices; i++)
    {
      // Get a net device from the node
      // as well as the channel, and figure
      // out the adjacent net devices
      Ptr<NetDevice> localNetDevice = node->GetDevice (i);
      Ptr<Channel> channel = localNetDevice->GetChannel ();
      if (channel == 0)
        {
          continue;
        }

      // this function takes in the local net dev, and channel, and
      // writes to the netDeviceContainer the adjacent net devs
      NetDeviceContainer netDeviceContainer;
      GetAdjacentNetDevices (localNetDevice, channel, netDeviceContainer);

      totalNeighbors += netDeviceContainer.GetN ();
    }

  return totalNeighbors;
}

template <typename T>
Ptr<BridgeNetDevice>
NixVectorRouting<T>::NetDeviceIsBridged (Ptr<NetDevice> nd) const
{
  NS_LOG_FUNCTION (this << nd);

  Ptr<Node> node = nd->GetNode ();
  uint32_t nDevices = node->GetNDevices ();

  //
  // There is no bit on a net device that says it is being bridged, so we have
  // to look for bridges on the node to which the device is attached.  If we
  // find a bridge, we need to look through its bridge ports (the devices it
  // bridges) to see if we find the device in question.
  //
  for (uint32_t i = 0; i < nDevices; ++i)
    {
      Ptr<NetDevice> ndTest = node->GetDevice (i);
      NS_LOG_LOGIC ("Examine device " << i << " " << ndTest);

      if (ndTest->IsBridge ())
        {
          NS_LOG_LOGIC ("device " << i << " is a bridge net device");
          Ptr<BridgeNetDevice> bnd = ndTest->GetObject<BridgeNetDevice> ();
          NS_ABORT_MSG_UNLESS (bnd, "NixVectorRouting::NetDeviceIsBridged (): GetObject for <BridgeNetDevice> failed");

          for (uint32_t j = 0; j < bnd->GetNBridgePorts (); ++j)
            {
              NS_LOG_LOGIC ("Examine bridge port " << j << " " << bnd->GetBridgePort (j));
              if (bnd->GetBridgePort (j) == nd)
                {
                  NS_LOG_LOGIC ("Net device " << nd << " is bridged by " << bnd);
                  return bnd;
                }
            }
        }
    }
  NS_LOG_LOGIC ("Net device " << nd << " is not bridged");
  return 0;
}

template <typename T>
uint32_t
NixVectorRouting<T>::FindNetDeviceForNixIndex (Ptr<Node> node, uint32_t nodeIndex, IpAddress & gatewayIp) const
{
  NS_LOG_FUNCTION (this << node << nodeIndex << gatewayIp);

  uint32_t numberOfDevices = node->GetNDevices ();
  uint32_t index = 0;
  uint32_t totalNeighbors = 0;

  // scan through the net devices on the parent node
  // and then look at the nodes adjacent to them
  for (uint32_t i = 0; i < numberOfDevices; i++)
    {
      // Get a net device from the node
      // as well as the channel, and figure
      // out the adjacent net devices
      Ptr<NetDevice> localNetDevice = node->GetDevice (i);
      Ptr<Channel> channel = localNetDevice->GetChannel ();
      if (channel == 0)
        {
          continue;
        }

      // this function takes in the local net dev, and channel, and
      // writes to the netDeviceContainer the adjacent net devs
      NetDeviceContainer netDeviceContainer;
      GetAdjacentNetDevices (localNetDevice, channel, netDeviceContainer);

      // check how many neighbors we have
      if (nodeIndex < (totalNeighbors + netDeviceContainer.GetN ()))
        {
          // found the proper net device
          index = i;
          Ptr<NetDevice> gatewayDevice = netDeviceContainer.Get (nodeIndex-totalNeighbors);
          Ptr<IpInterface> gatewayInterface = GetInterfaceByNetDevice (gatewayDevice);
          IpInterfaceAddress ifAddr = gatewayInterface->GetAddress (0);
          gatewayIp = ifAddr.GetAddress ();
          break;
        }
      totalNeighbors += netDeviceContainer.GetN ();
    }

  return index;
}

template <typename T>
Ptr<typename NixVectorRouting<T>::IpRoute>
NixVectorRouting<T>::RouteOutput (Ptr<Packet> p, const IpHeader &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)
{
  NS_LOG_FUNCTION (this << header << oif);

  Ptr<IpRoute> rtentry;
  Ptr<NixVector> nixVectorInCache;
  Ptr<NixVector> nixVectorForPacket;

  CheckCacheStateAndFlush ();

  IpAddress destAddress = header.GetDestination ();

  NS_LOG_DEBUG ("Dest IP from header: " << destAddress);

  if (destAddress.IsLocalhost ())
    {
      rtentry = Create<IpRoute> ();
      rtentry->SetSource (IpAddress::GetLoopback ());
      rtentry->SetDestination (destAddress);
      rtentry->SetGateway (IpAddress::GetZero ());
      for (uint32_t i = 0 ; i < m_ip->GetNInterfaces (); i++)
        {
          Ptr<LoopbackNetDevice> loNetDevice = DynamicCast<LoopbackNetDevice> (m_ip->GetNetDevice (i));
          if (loNetDevice)
            {
              rtentry->SetOutputDevice (loNetDevice);
              break;
            }
        }
      return rtentry;
    }

  if constexpr (!IsIpv4::value)
    {
      /* when sending on link-local multicast, there have to be interface specified */
      if (destAddress.IsLinkLocalMulticast ())
        {
          NS_ASSERT_MSG (oif, "Try to send on link-local multicast address, and no interface index is given!");
          rtentry = Create<IpRoute> ();
          rtentry->SetSource (m_ip->SourceAddressSelection (m_ip->GetInterfaceForDevice (oif), destAddress));
          rtentry->SetDestination (destAddress);
          rtentry->SetGateway (Ipv6Address::GetZero ());
          rtentry->SetOutputDevice (oif);
          return rtentry;
        }
    }
  // Check the Nix cache
  bool foundInCache = false;
  nixVectorInCache = GetNixVectorInCache (destAddress, foundInCache);

  // not in cache
  if (!foundInCache)
    {
      NS_LOG_LOGIC ("Nix-vector not in cache, build: ");
      // Build the nix-vector, given this node and the
      // dest IP address
      nixVectorInCache = GetNixVector (m_node, destAddress, oif);
      if (nixVectorInCache)
      {
        // cache it
        m_nixCache.insert (typename NixMap_t::value_type (destAddress, nixVectorInCache));
      }
    }

  // path exists
  if (nixVectorInCache)
    {
      NS_LOG_LOGIC ("Nix-vector contents: " << *nixVectorInCache);

      // create a new nix vector to be used,
      // we want to keep the cached version clean
      nixVectorForPacket = nixVectorInCache->Copy ();

      // Get the interface number that we go out of, by extracting
      // from the nix-vector
      if (m_totalNeighbors == 0)
        {
          m_totalNeighbors = FindTotalNeighbors (m_node);
        }

      // Get the interface number that we go out of, by extracting
      // from the nix-vector
      uint32_t numberOfBits = nixVectorForPacket->BitCount (m_totalNeighbors);
      uint32_t nodeIndex = nixVectorForPacket->ExtractNeighborIndex (numberOfBits);

      // Search here in a cache for this node index
      // and look for a IpRoute
      rtentry = GetIpRouteInCache (destAddress);

      if (!rtentry || !(rtentry->GetOutputDevice () == oif))
        {
          // not in cache or a different specified output
          // device is to be used

          // first, make sure we erase existing (incorrect)
          // rtentry from the map
          if (rtentry)
            {
              m_ipRouteCache.erase (destAddress);
            }

          NS_LOG_LOGIC ("IpRoute not in cache, build: ");
          IpAddress gatewayIp;
          uint32_t index = FindNetDeviceForNixIndex (m_node, nodeIndex, gatewayIp);
          int32_t interfaceIndex = 0;

          if (!oif)
            {
              interfaceIndex = (m_ip)->GetInterfaceForDevice (m_node->GetDevice (index));
            }
          else
            {
              interfaceIndex = (m_ip)->GetInterfaceForDevice (oif);
            }

          NS_ASSERT_MSG (interfaceIndex != -1, "Interface index not found for device");

          IpAddress sourceIPAddr = m_ip->SourceAddressSelection (interfaceIndex, destAddress);

          // start filling in the IpRoute info
          rtentry = Create<IpRoute> ();
          rtentry->SetSource (sourceIPAddr);

          rtentry->SetGateway (gatewayIp);
          rtentry->SetDestination (destAddress);

          if (!oif)
            {
              rtentry->SetOutputDevice (m_ip->GetNetDevice (interfaceIndex));
            }
          else
            {
              rtentry->SetOutputDevice (oif);
            }

          sockerr = Socket::ERROR_NOTERROR;

          // add rtentry to cache
          m_ipRouteCache.insert (typename IpRouteMap_t::value_type (destAddress, rtentry));
        }

      NS_LOG_LOGIC ("Nix-vector contents: " << *nixVectorInCache << " : Remaining bits: " << nixVectorForPacket->GetRemainingBits ());

      // Add  nix-vector in the packet class
      // make sure the packet exists first
      if (p)
        {
          NS_LOG_LOGIC ("Adding Nix-vector to packet: " << *nixVectorForPacket);
          p->SetNixVector (nixVectorForPacket);
        }
    }
  else // path doesn't exist
    {
      NS_LOG_ERROR ("No path to the dest: " << destAddress);
      sockerr = Socket::ERROR_NOROUTETOHOST;
    }

  return rtentry;
}

template <typename T>
bool
NixVectorRouting<T>::RouteInput (Ptr<const Packet> p, const IpHeader &header, Ptr<const NetDevice> idev,
                                 UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                                 LocalDeliverCallback lcb, ErrorCallback ecb)
{
  NS_LOG_FUNCTION (this << p << header << header.GetSource () << header.GetDestination () << idev);

  CheckCacheStateAndFlush ();

  NS_ASSERT (m_ip != 0);
  // Check if input device supports IP
  NS_ASSERT (m_ip->GetInterfaceForDevice (idev) >= 0);
  uint32_t iif = m_ip->GetInterfaceForDevice (idev);
  // Check if input device supports IP
  NS_ASSERT (iif >= 0);

  IpAddress destAddress = header.GetDestination ();

  if constexpr (IsIpv4::value)
    {
      // Local delivery
      if (m_ip->IsDestinationAddress (destAddress, iif))
        {
          if (!lcb.IsNull ())
            {
              NS_LOG_LOGIC ("Local delivery to " << destAddress);
              p->SetNixVector (nullptr);
              lcb (p, header, iif);
              return true;
            }
          else
            {
              // The local delivery callback is null.  This may be a multicast
              // or broadcast packet, so return false so that another
              // multicast routing protocol can handle it.  It should be possible
              // to extend this to explicitly check whether it is a unicast
              // packet, and invoke the error callback if so
              return false;
            }
        }
    }
  else
    {
      if (destAddress.IsMulticast ())
        {
          NS_LOG_LOGIC ("Multicast route not supported by Nix-Vector routing " << destAddress);
          return false; // Let other routing protocols try to handle this
        }

      // Check if input device supports IP forwarding
      if (m_ip->IsForwarding (iif) == false)
        {
          NS_LOG_LOGIC ("Forwarding disabled for this interface");
          if (!ecb.IsNull ())
            {
              ecb (p, header, Socket::ERROR_NOROUTETOHOST);
            }
          return true;
        }
    }

  Ptr<IpRoute> rtentry;

  // Get the nix-vector from the packet
  Ptr<NixVector> nixVector = p->GetNixVector ();

  // If nixVector isn't in packet, something went wrong
  NS_ASSERT (nixVector);

  if (nixVector->GetEpoch () != g_epoch)
    {
      NS_LOG_LOGIC ("NixVector epoch mismatch (" << nixVector->GetEpoch () << " Vs " << g_epoch
                                                 << ") - rebuilding it");
      nixVector = GetNixVector (m_node, destAddress, 0);
      p->SetNixVector (nixVector);
    }

  // Get the interface number that we go out of, by extracting
  // from the nix-vector
  if (m_totalNeighbors == 0)
    {
      m_totalNeighbors = FindTotalNeighbors (m_node);
    }
  uint32_t numberOfBits = nixVector->BitCount (m_totalNeighbors);
  uint32_t nodeIndex = nixVector->ExtractNeighborIndex (numberOfBits);

  rtentry = GetIpRouteInCache (destAddress);
  // not in cache
  if (!rtentry)
    {
      NS_LOG_LOGIC ("IpRoute not in cache, build: ");
      IpAddress gatewayIp;
      uint32_t index = FindNetDeviceForNixIndex (m_node, nodeIndex, gatewayIp);
      uint32_t interfaceIndex = (m_ip)->GetInterfaceForDevice (m_node->GetDevice (index));
      IpInterfaceAddress ifAddr = m_ip->GetAddress (interfaceIndex, 0);

      // start filling in the IpRoute info
      rtentry = Create<IpRoute> ();
      rtentry->SetSource (ifAddr.GetAddress ());

      rtentry->SetGateway (gatewayIp);
      rtentry->SetDestination (destAddress);
      rtentry->SetOutputDevice (m_ip->GetNetDevice (interfaceIndex));

      // add rtentry to cache
      m_ipRouteCache.insert (typename IpRouteMap_t::value_type (destAddress, rtentry));
    }

  NS_LOG_LOGIC ("At Node " << m_node->GetId () << ", Extracting " << numberOfBits <<
                " bits from Nix-vector: " << nixVector << " : " << *nixVector);

  // call the unicast callback
  // local deliver is handled by Ipv4StaticRoutingImpl
  // so this code is never even called if the packet is
  // destined for this node.
  if constexpr (IsIpv4::value)
    {
      ucb (rtentry, p, header);
    }
  else
    {
      ucb (idev, rtentry, p, header);
    }

  return true;
}

template <typename T>
void
NixVectorRouting<T>::PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
  NS_LOG_FUNCTION_NOARGS ();

  CheckCacheStateAndFlush ();

  std::ostream* os = stream->GetStream ();
  // Copy the current ostream state
  std::ios oldState (nullptr);
  oldState.copyfmt (*os);

  *os << std::resetiosflags (std::ios::adjustfield) << std::setiosflags (std::ios::left);

  *os << "Node: " << m_ip->template GetObject<Node> ()->GetId ()
      << ", Time: " << Now().As (unit)
      << ", Local time: " << m_ip->template GetObject<Node> ()->GetLocalTime ().As (unit)
      << ", Nix Routing" << std::endl;

  *os << "NixCache:" << std::endl;
  if (m_nixCache.size () > 0)
    {
      *os << std::setw (30) << "Destination";
      *os << "NixVector" << std::endl;
      for (typename NixMap_t::const_iterator it = m_nixCache.begin (); it != m_nixCache.end (); it++)
        {
          std::ostringstream dest;
          dest << it->first;
          *os << std::setw (30) << dest.str ();
          if (it->second)
            {
              *os << *(it->second) << std::endl;
            }
          else
            {
              *os << "-" << std::endl;
            }
        }
    }
  
  *os << "IpRouteCache:" << std::endl;  
  if (m_ipRouteCache.size () > 0)
    {
      *os << std::setw (30) << "Destination";
      *os << std::setw (30) << "Gateway";
      *os << std::setw (30) << "Source";
      *os << "OutputDevice" << std::endl;
      for (typename IpRouteMap_t::const_iterator it = m_ipRouteCache.begin (); it != m_ipRouteCache.end (); it++)
        {
          std::ostringstream dest, gw, src;
          dest << it->second->GetDestination ();
          *os << std::setw (30) << dest.str ();
          gw << it->second->GetGateway ();
          *os << std::setw (30) << gw.str ();
          src << it->second->GetSource ();
          *os << std::setw (30) << src.str ();
          *os << "  ";
          if (Names::FindName (it->second->GetOutputDevice ()) != "")
            {
              *os << Names::FindName (it->second->GetOutputDevice ());
            }
          else
            {
              *os << it->second->GetOutputDevice ()->GetIfIndex ();
            }
          *os << std::endl;
        }
    }
  *os << std::endl;
  // Restore the previous ostream state
  (*os).copyfmt (oldState);
}

// virtual functions from Ipv4RoutingProtocol 
template <typename T>
void
NixVectorRouting<T>::NotifyInterfaceUp (uint32_t i)
{
#ifdef NS3_MTP
  g_isCacheDirty.store (true, std::memory_order_release);
#else
  g_isCacheDirty = true;
#endif
}
template <typename T>
void
NixVectorRouting<T>::NotifyInterfaceDown (uint32_t i)
{
#ifdef NS3_MTP
  g_isCacheDirty.store (true, std::memory_order_release);
#else
  g_isCacheDirty = true;
#endif
}
template <typename T>
void
NixVectorRouting<T>::NotifyAddAddress (uint32_t interface, IpInterfaceAddress address)
{
#ifdef NS3_MTP
  g_isCacheDirty.store (true, std::memory_order_release);
#else
  g_isCacheDirty = true;
#endif
}
template <typename T>
void
NixVectorRouting<T>::NotifyRemoveAddress (uint32_t interface, IpInterfaceAddress address)
{
#ifdef NS3_MTP
  g_isCacheDirty.store (true, std::memory_order_release);
#else
  g_isCacheDirty = true;
#endif
}
template <typename T>
void
NixVectorRouting<T>::NotifyAddRoute (IpAddress dst, Ipv6Prefix mask, IpAddress nextHop, uint32_t interface, IpAddress prefixToUse)
{
#ifdef NS3_MTP
  g_isCacheDirty.store (true, std::memory_order_release);
#else
  g_isCacheDirty = true;
#endif
}
template <typename T>
void
NixVectorRouting<T>::NotifyRemoveRoute (IpAddress dst, Ipv6Prefix mask, IpAddress nextHop, uint32_t interface, IpAddress prefixToUse)
{
#ifdef NS3_MTP
  g_isCacheDirty.store (true, std::memory_order_release);
#else
  g_isCacheDirty = true;
#endif
}

template <typename T>
bool
NixVectorRouting<T>::BFS (uint32_t numberOfNodes, Ptr<Node> source,
                           Ptr<Node> dest, std::vector< Ptr<Node> > & parentVector,
                           Ptr<NetDevice> oif) const
{
  NS_LOG_FUNCTION (this << numberOfNodes << source << dest << parentVector << oif);

  NS_LOG_LOGIC ("Going from Node " << source->GetId () << " to Node " << dest->GetId ());
  std::queue< Ptr<Node> > greyNodeList;  // discovered nodes with unexplored children

  // reset the parent vector
  parentVector.assign (numberOfNodes, 0); // initialize to 0

  // Add the source node to the queue, set its parent to itself
  greyNodeList.push (source);
  parentVector.at (source->GetId ()) = source;

  // BFS loop
  while (greyNodeList.size () != 0)
    {
      Ptr<Node> currNode = greyNodeList.front ();
      Ptr<IpL3Protocol> ip = currNode->GetObject<IpL3Protocol> ();
 
      if (currNode == dest) 
        {
          NS_LOG_LOGIC ("Made it to Node " << currNode->GetId ());
          return true;
        }

      // if this is the first iteration of the loop and a 
      // specific output interface was given, make sure 
      // we go this way
      if (currNode == source && oif)
        {
          // make sure that we can go this way
          if (ip)
            {
              uint32_t interfaceIndex = (ip)->GetInterfaceForDevice (oif);
              if (!(ip->IsUp (interfaceIndex)))
                {
                  NS_LOG_LOGIC ("IpInterface is down");
                  return false;
                }
            }
          if (!(oif->IsLinkUp ()))
            {
              NS_LOG_LOGIC ("Link is down.");
              return false;
            }
          Ptr<Channel> channel = oif->GetChannel ();
          if (channel == 0)
            { 
              return false;
            }

          // this function takes in the local net dev, and channel, and
          // writes to the netDeviceContainer the adjacent net devs
          NetDeviceContainer netDeviceContainer;
          GetAdjacentNetDevices (oif, channel, netDeviceContainer);

          // Finally we can get the adjacent nodes
          // and scan through them.  We push them
          // to the greyNode queue, if they aren't 
          // already there.
          for (NetDeviceContainer::Iterator iter = netDeviceContainer.Begin (); iter != netDeviceContainer.End (); iter++)
            {
              Ptr<Node> remoteNode = (*iter)->GetNode ();
              Ptr<IpInterface> remoteIpInterface = GetInterfaceByNetDevice(*iter);
              if (remoteIpInterface == 0 || !(remoteIpInterface->IsUp ()))
                {
                  NS_LOG_LOGIC ("IpInterface either doesn't exist or is down");
                  continue;
                }

              // check to see if this node has been pushed before
              // by checking to see if it has a parent
              // if it doesn't (null or 0), then set its parent and
              // push to the queue
              if (parentVector.at (remoteNode->GetId ()) == 0)
                {
                  parentVector.at (remoteNode->GetId ()) = currNode;
                  greyNodeList.push (remoteNode);
                }
            }
        }
      else
        {
          // Iterate over the current node's adjacent vertices
          // and push them into the queue
          for (uint32_t i = 0; i < (currNode->GetNDevices ()); i++)
            {
              // Get a net device from the node
              // as well as the channel, and figure
              // out the adjacent net device
              Ptr<NetDevice> localNetDevice = currNode->GetDevice (i);

              // make sure that we can go this way
              if (ip)
                {
                  uint32_t interfaceIndex = (ip)->GetInterfaceForDevice (currNode->GetDevice (i));
                  if (!(ip->IsUp (interfaceIndex)))
                    {
                      NS_LOG_LOGIC ("IpInterface is down");
                      continue;
                    }
                }
              if (!(localNetDevice->IsLinkUp ()))
                {
                  NS_LOG_LOGIC ("Link is down.");
                  continue;
                }
              Ptr<Channel> channel = localNetDevice->GetChannel ();
              if (channel == 0)
                { 
                  continue;
                }

              // this function takes in the local net dev, and channel, and
              // writes to the netDeviceContainer the adjacent net devs
              NetDeviceContainer netDeviceContainer;
              GetAdjacentNetDevices (localNetDevice, channel, netDeviceContainer);

              // Finally we can get the adjacent nodes
              // and scan through them.  We push them
              // to the greyNode queue, if they aren't 
              // already there.
              for (NetDeviceContainer::Iterator iter = netDeviceContainer.Begin (); iter != netDeviceContainer.End (); iter++)
                {
                  Ptr<Node> remoteNode = (*iter)->GetNode ();
                  Ptr<IpInterface> remoteIpInterface = GetInterfaceByNetDevice(*iter);
                  if (remoteIpInterface == 0 || !(remoteIpInterface->IsUp ()))
                    {
                      NS_LOG_LOGIC ("IpInterface either doesn't exist or is down");
                      continue;
                    }

                  // check to see if this node has been pushed before
                  // by checking to see if it has a parent
                  // if it doesn't (null or 0), then set its parent and
                  // push to the queue
                  if (parentVector.at (remoteNode->GetId ()) == 0)
                    {
                      parentVector.at (remoteNode->GetId ()) = currNode;
                      greyNodeList.push (remoteNode);
                    }
                }
            }
        }

      // Pop off the head grey node.  We have all its children.
      // It is now black.
      greyNodeList.pop ();
    }

  // Didn't find the dest...
  return false;
}

template <typename T>
void
NixVectorRouting<T>::PrintRoutingPath (Ptr<Node> source, IpAddress dest,
                                        Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
  NS_LOG_FUNCTION (this << source << dest);

  Ptr<NixVector> nixVectorInCache;
  Ptr<NixVector> nixVector;
  Ptr<IpRoute> rtentry;

  CheckCacheStateAndFlush ();

  Ptr<Node> destNode = GetNodeByIp (dest);
  if (destNode == 0)
    {
      NS_LOG_ERROR ("No routing path exists");
      return;
    }

  std::ostream* os = stream->GetStream ();
  // Copy the current ostream state
  std::ios oldState (nullptr);
  oldState.copyfmt (*os);

  *os << std::resetiosflags (std::ios::adjustfield) << std::setiosflags (std::ios::left);
  *os << "Time: " << Now().As (unit)
      << ", Nix Routing" << std::endl;
  *os << "Route path from ";
  *os << "Node " << source->GetId () << " to Node " << destNode->GetId () << ", ";
  *os << "Nix Vector: ";

  // Check the Nix cache
  bool foundInCache = true;
  nixVectorInCache = GetNixVectorInCache (dest, foundInCache);

  // not in cache
  if (!foundInCache)
    {
      NS_LOG_LOGIC ("Nix-vector not in cache, build: ");
      // Build the nix-vector, given the source node and the
      // dest IP address
      nixVectorInCache = GetNixVector (source, dest, nullptr);
    }

  if (nixVectorInCache || (!nixVectorInCache && source == destNode))
    {
      Ptr<Node> curr = source;
      uint32_t totalNeighbors = 0;

      if (nixVectorInCache)
        {
          // Make a NixVector copy to work with. This is because
          // we don't want to extract the bits from nixVectorInCache
          // which is stored in the m_nixCache.
          nixVector = nixVectorInCache->Copy ();

          *os << *nixVector;
        }
      *os << std::endl;

      if (source == destNode)
        {
          std::ostringstream addr, node;
          addr << dest;
          node << "(Node " << destNode->GetId () << ")";
          *os << std::setw (25) << addr.str ();
          *os << std::setw (10) << node.str ();
          *os << "---->   ";
          *os << std::setw (25) << addr.str ();
          *os << node.str () << std::endl;
        }

      while (curr != destNode)
        {
          totalNeighbors = FindTotalNeighbors (curr);
          // Get the number of bits required
          // to represent all the neighbors
          uint32_t numberOfBits = nixVector->BitCount (totalNeighbors);
          // Get the nixIndex
          uint32_t nixIndex = nixVector->ExtractNeighborIndex (numberOfBits);
          // gatewayIP is the IP of next
          // node on channel found from nixIndex
          IpAddress gatewayIp;
          // Get the Net Device index from the nixIndex
          uint32_t netDeviceIndex = FindNetDeviceForNixIndex (curr, nixIndex, gatewayIp);
          // Get the interfaceIndex with the help of netDeviceIndex.
          // It will be used to get the IP address on interfaceIndex
          // interface of 'curr' node.
          Ptr<IpL3Protocol> ip = curr->GetObject<IpL3Protocol> ();
          Ptr<NetDevice> outDevice = curr->GetDevice (netDeviceIndex);
          uint32_t interfaceIndex = ip->GetInterfaceForDevice (outDevice);
          IpAddress sourceIPAddr;
          if (curr == source)
            {
              sourceIPAddr = ip->SourceAddressSelection (interfaceIndex, dest);
            }
          else
            {
              // We use the first address because it's indifferent which one
              // we use to identify intermediate routers
              sourceIPAddr = ip->GetAddress (interfaceIndex, 0).GetAddress ();
            }

          std::ostringstream currAddr, currNode, nextAddr, nextNode;
          currAddr << sourceIPAddr;
          currNode << "(Node " << curr->GetId () << ")";
          *os << std::setw (25) << currAddr.str ();
          *os << std::setw (10) << currNode.str ();
          // Replace curr with the next node
          curr = GetNodeByIp (gatewayIp);
          nextAddr << ((curr == destNode) ? dest : gatewayIp);
          nextNode << "(Node " << curr->GetId () << ")";
          *os << "---->   ";
          *os << std::setw (25) << nextAddr.str ();
          *os << nextNode.str () << std::endl;
        }
        *os << std::endl;
    }
  else
    {
      *os << ")" << std::endl;
      // No Route exists
      *os << "There does not exist a path from Node " << source->GetId ()
          << " to Node " << destNode->GetId () << "." << std::endl;
    }
  // Restore the previous ostream state
  (*os).copyfmt (oldState);
}

template <typename T>
void 
NixVectorRouting<T>::CheckCacheStateAndFlush (void) const
{
#ifdef NS3_MTP
  if (g_isCacheDirty.load (std::memory_order_acquire))
    {
      if (g_cacheFlushing.exchange (true, std::memory_order_relaxed))
        {
          while (g_isCacheDirty.load (std::memory_order_acquire))
            ;
        }
      else
        {
          FlushGlobalNixRoutingCache ();
          g_isCacheDirty.store (false, std::memory_order_release);
          g_cacheFlushing.store (false, std::memory_order_release);
        }
    }
#else
  if (g_isCacheDirty)
    {
      FlushGlobalNixRoutingCache ();
      g_epoch++;
      g_isCacheDirty = false;
    }
#endif
}

/* Public template function declarations */
template void NixVectorRouting<Ipv4RoutingProtocol>::SetNode (Ptr<Node> node);
template void NixVectorRouting<Ipv6RoutingProtocol>::SetNode (Ptr<Node> node);
template void NixVectorRouting<Ipv4RoutingProtocol>::FlushGlobalNixRoutingCache (void) const;
template void NixVectorRouting<Ipv6RoutingProtocol>::FlushGlobalNixRoutingCache (void) const;
template void NixVectorRouting<Ipv4RoutingProtocol>::PrintRoutingPath (Ptr<Node> source, IpAddress dest,
                                                                       Ptr<OutputStreamWrapper> stream, Time::Unit unit) const;
template void NixVectorRouting<Ipv6RoutingProtocol>::PrintRoutingPath (Ptr<Node> source, IpAddress dest,
                                                                       Ptr<OutputStreamWrapper> stream, Time::Unit unit) const;

} // namespace ns3
