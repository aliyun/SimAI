/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Aleksey Kovalenko <kovalenko@iitp.ru>
 */

#include "ns3/peer-management-protocol.h"
#include "peer-management-protocol-mac.h"
#include "ie-dot11s-configuration.h"
#include "ie-dot11s-id.h"
#include "ns3/mesh-point-device.h"
#include "ns3/simulator.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/random-variable-stream.h"
#include "ns3/mesh-wifi-interface-mac.h"
#include "ns3/mesh-wifi-interface-mac-plugin.h"
#include "ns3/wifi-net-device.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PeerManagementProtocol");
  
namespace dot11s {
  
/***************************************************
 * PeerManager
 ***************************************************/
NS_OBJECT_ENSURE_REGISTERED (PeerManagementProtocol);

TypeId
PeerManagementProtocol::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::dot11s::PeerManagementProtocol")
    .SetParent<Object> ()
    .SetGroupName ("Mesh")
    .AddConstructor<PeerManagementProtocol> ()
    // maximum number of peer links. Now we calculate the total
    // number of peer links on all interfaces
    .AddAttribute ( "MaxNumberOfPeerLinks",
                    "Maximum number of peer links",
                    UintegerValue (32),
                    MakeUintegerAccessor (
                      &PeerManagementProtocol::m_maxNumberOfPeerLinks),
                    MakeUintegerChecker<uint8_t> ()
                    )
    .AddAttribute ( "MaxBeaconShiftValue",
                    "Maximum number of TUs for beacon shifting",
                    UintegerValue (15),
                    MakeUintegerAccessor (
                      &PeerManagementProtocol::m_maxBeaconShift),
                    MakeUintegerChecker<uint16_t> ()
                    )
    .AddAttribute ( "EnableBeaconCollisionAvoidance",
                    "Enable/Disable Beacon collision avoidance.",
                    BooleanValue (true),
                    MakeBooleanAccessor (
                      &PeerManagementProtocol::SetBeaconCollisionAvoidance, &PeerManagementProtocol::GetBeaconCollisionAvoidance),
                    MakeBooleanChecker ()
                    )
    .AddTraceSource ("LinkOpen",
                     "New peer link opened",
                     MakeTraceSourceAccessor (&PeerManagementProtocol::m_linkOpenTraceSrc),
                     "ns3::PeerManagementProtocol::LinkOpenCloseTracedCallback"
                     )
    .AddTraceSource ("LinkClose",
                     "New peer link closed",
                     MakeTraceSourceAccessor (&PeerManagementProtocol::m_linkCloseTraceSrc),
                     "ns3::PeerManagementProtocol::LinkOpenCloseTracedCallback"
                     )

  ;
  return tid;
}
PeerManagementProtocol::PeerManagementProtocol () :
  m_lastAssocId (0), m_lastLocalLinkId (1), m_enableBca (true), m_maxBeaconShift (15)
{
  m_beaconShift = CreateObject<UniformRandomVariable> ();
}
PeerManagementProtocol::~PeerManagementProtocol ()
{
  m_meshId = 0;
}
void
PeerManagementProtocol::DoDispose ()
{
  //cancel cleanup event and go through the map of peer links,
  //deleting each
  for (PeerLinksMap::iterator j = m_peerLinks.begin (); j != m_peerLinks.end (); j++)
    {
      for (PeerLinksOnInterface::iterator i = j->second.begin (); i != j->second.end (); i++)
        {
          (*i) = 0;
        }
      j->second.clear ();
    }
  m_peerLinks.clear ();
  m_plugins.clear ();
}

bool
PeerManagementProtocol::Install (Ptr<MeshPointDevice> mp)
{
  std::vector<Ptr<NetDevice> > interfaces = mp->GetInterfaces ();
  for (std::vector<Ptr<NetDevice> >::iterator i = interfaces.begin (); i != interfaces.end (); i++)
    {
      Ptr<WifiNetDevice> wifiNetDev = (*i)->GetObject<WifiNetDevice> ();
      if (wifiNetDev == 0)
        {
          return false;
        }
      Ptr<MeshWifiInterfaceMac> mac = wifiNetDev->GetMac ()->GetObject<MeshWifiInterfaceMac> ();
      if (mac == 0)
        {
          return false;
        }
      Ptr<PeerManagementProtocolMac> plugin = Create<PeerManagementProtocolMac> ((*i)->GetIfIndex (), this);
      mac->InstallPlugin (plugin);
      m_plugins[(*i)->GetIfIndex ()] = plugin;
      PeerLinksOnInterface newmap;
      m_peerLinks[(*i)->GetIfIndex ()] = newmap;
    }
  // Mesh point aggregates all installed protocols
  m_address = Mac48Address::ConvertFrom (mp->GetAddress ());
  mp->AggregateObject (this);
  return true;
}

Ptr<IeBeaconTiming>
PeerManagementProtocol::GetBeaconTimingElement (uint32_t interface)
{
  if (!GetBeaconCollisionAvoidance ())
    {
      return 0;
    }
  Ptr<IeBeaconTiming> retval = Create<IeBeaconTiming> ();
  PeerLinksMap::iterator iface = m_peerLinks.find (interface);
  NS_ASSERT (iface != m_peerLinks.end ());
  for (PeerLinksOnInterface::iterator i = iface->second.begin (); i != iface->second.end (); i++)
    {
      //If we do not know peer Assoc Id, we shall not add any info
      //to a beacon timing element
      if ((*i)->GetBeaconInterval () == Seconds (0))
        {
          //No beacon was received, do not include to the beacon timing element
          continue;
        }
      retval->AddNeighboursTimingElementUnit ((*i)->GetLocalAid (), (*i)->GetLastBeacon (),
                                              (*i)->GetBeaconInterval ());
    }
  return retval;
}
void
PeerManagementProtocol::ReceiveBeacon (uint32_t interface, Mac48Address peerAddress, Time beaconInterval, Ptr<IeBeaconTiming> timingElement)
{
  //PM STATE Machine
  //Check that a given beacon is not from our interface
  for (PeerManagementProtocolMacMap::const_iterator i = m_plugins.begin (); i != m_plugins.end (); i++)
    {
      if (i->second->GetAddress () == peerAddress)
        {
          return;
        }
    }
  Ptr<PeerLink> peerLink = FindPeerLink (interface, peerAddress);
  if (peerLink == 0)
    {
      if (ShouldSendOpen (interface, peerAddress))
        {
          peerLink = InitiateLink (interface, peerAddress, Mac48Address::GetBroadcast ());
          peerLink->MLMEActivePeerLinkOpen ();
        }
      else
        {
          return;
        }
    }
  peerLink->SetBeaconInformation (Simulator::Now (), beaconInterval);
  if (GetBeaconCollisionAvoidance ())
    {
      peerLink->SetBeaconTimingElement (*PeekPointer (timingElement));
    }
}

void
PeerManagementProtocol::ReceivePeerLinkFrame (uint32_t interface, Mac48Address peerAddress,
                                              Mac48Address peerMeshPointAddress, uint16_t aid, IePeerManagement peerManagementElement,
                                              IeConfiguration meshConfig)
{
  Ptr<PeerLink> peerLink = FindPeerLink (interface, peerAddress);
  if (peerManagementElement.SubtypeIsOpen ())
    {
      PmpReasonCode reasonCode (REASON11S_RESERVED);
      bool reject = !(ShouldAcceptOpen (interface, peerAddress, reasonCode));
      if (peerLink == 0)
        {
          peerLink = InitiateLink (interface, peerAddress, peerMeshPointAddress);
        }
      if (!reject)
        {
          peerLink->OpenAccept (peerManagementElement.GetLocalLinkId (), meshConfig, peerMeshPointAddress);
        }
      else
        {
          peerLink->OpenReject (peerManagementElement.GetLocalLinkId (), meshConfig, peerMeshPointAddress,
                                reasonCode);
        }
    }
  if (peerLink == 0)
    {
      return;
    }
  if (peerManagementElement.SubtypeIsConfirm ())
    {
      peerLink->ConfirmAccept (peerManagementElement.GetLocalLinkId (),
                               peerManagementElement.GetPeerLinkId (), aid, meshConfig, peerMeshPointAddress);
    }
  if (peerManagementElement.SubtypeIsClose ())
    {
      peerLink->Close (peerManagementElement.GetLocalLinkId (), peerManagementElement.GetPeerLinkId (),
                       peerManagementElement.GetReasonCode ());
    }
}
void
PeerManagementProtocol::ConfigurationMismatch (uint32_t interface, Mac48Address peerAddress)
{
  Ptr<PeerLink> peerLink = FindPeerLink (interface, peerAddress);
  if (peerLink != 0)
    {
      peerLink->MLMECancelPeerLink (REASON11S_MESH_CAPABILITY_POLICY_VIOLATION);
    }
}
void
PeerManagementProtocol::TransmissionFailure (uint32_t interface, Mac48Address peerAddress)
{
  NS_LOG_DEBUG ("transmission failed between "<<GetAddress () << " and " << peerAddress << " failed, link will be closed");
  Ptr<PeerLink> peerLink = FindPeerLink (interface, peerAddress);
  if (peerLink != 0)
    {
      peerLink->TransmissionFailure ();
    }
}
void
PeerManagementProtocol::TransmissionSuccess (uint32_t interface, Mac48Address peerAddress)
{
  NS_LOG_DEBUG ("transmission success "<< GetAddress () << " and " << peerAddress);
  Ptr<PeerLink> peerLink = FindPeerLink (interface, peerAddress);
  if (peerLink != 0)
    {
      peerLink->TransmissionSuccess ();
    }
}
Ptr<PeerLink>
PeerManagementProtocol::InitiateLink (uint32_t interface, Mac48Address peerAddress,
                                      Mac48Address peerMeshPointAddress)
{
  Ptr<PeerLink> new_link = CreateObject<PeerLink> ();
  //find a peer link  - it must not exist
  if (FindPeerLink (interface, peerAddress) != 0)
    {
      NS_FATAL_ERROR ("Peer link must not exist.");
    }
  // Plugin must exist
  PeerManagementProtocolMacMap::iterator plugin = m_plugins.find (interface);
  NS_ASSERT (plugin != m_plugins.end ());
  PeerLinksMap::iterator iface = m_peerLinks.find (interface);
  NS_ASSERT (iface != m_peerLinks.end ());
  new_link->SetLocalAid (m_lastAssocId++);
  new_link->SetInterface (interface);
  new_link->SetLocalLinkId (m_lastLocalLinkId++);
  new_link->SetPeerAddress (peerAddress);
  new_link->SetPeerMeshPointAddress (peerMeshPointAddress);
  new_link->SetMacPlugin (plugin->second);
  new_link->MLMESetSignalStatusCallback (MakeCallback (&PeerManagementProtocol::PeerLinkStatus, this));
  iface->second.push_back (new_link);
  return new_link;
}

Ptr<PeerLink>
PeerManagementProtocol::FindPeerLink (uint32_t interface, Mac48Address peerAddress)
{
  PeerLinksMap::iterator iface = m_peerLinks.find (interface);
  NS_ASSERT (iface != m_peerLinks.end ());
  for (PeerLinksOnInterface::iterator i = iface->second.begin (); i != iface->second.end (); i++)
    {
      if ((*i)->GetPeerAddress () == peerAddress)
        {
          if ((*i)->LinkIsIdle ())
            {
              (*i) = 0;
              (iface->second).erase (i);
              return 0;
            }
          else
            {
              return (*i);
            }
        }
    }
  return 0;
}
void
PeerManagementProtocol::SetPeerLinkStatusCallback (
  Callback<void, Mac48Address, Mac48Address, uint32_t, bool> cb)
{
  m_peerStatusCallback = cb;
}

std::vector<Mac48Address>
PeerManagementProtocol::GetPeers (uint32_t interface) const
{
  std::vector<Mac48Address> retval;
  PeerLinksMap::const_iterator iface = m_peerLinks.find (interface);
  NS_ASSERT (iface != m_peerLinks.end ());
  for (PeerLinksOnInterface::const_iterator i = iface->second.begin (); i != iface->second.end (); i++)
    {
      if ((*i)->LinkIsEstab ())
        {
          retval.push_back ((*i)->GetPeerAddress ());
        }
    }
  return retval;
}

std::vector< Ptr<PeerLink> >
PeerManagementProtocol::GetPeerLinks () const
{
  std::vector< Ptr<PeerLink> > links;

  for (PeerLinksMap::const_iterator iface = m_peerLinks.begin (); iface != m_peerLinks.end (); ++iface)
    {
      for (PeerLinksOnInterface::const_iterator i = iface->second.begin ();
           i != iface->second.end (); i++)
        if ((*i)->LinkIsEstab ())
          links.push_back (*i);
    }
  return links;
}
bool
PeerManagementProtocol::IsActiveLink (uint32_t interface, Mac48Address peerAddress)
{
  Ptr<PeerLink> peerLink = FindPeerLink (interface, peerAddress);
  if (peerLink != 0)
    {
      return (peerLink->LinkIsEstab ());
    }
  return false;
}
bool
PeerManagementProtocol::ShouldSendOpen (uint32_t interface, Mac48Address peerAddress)
{
  return (m_stats.linksTotal < m_maxNumberOfPeerLinks);
}

bool
PeerManagementProtocol::ShouldAcceptOpen (uint32_t interface, Mac48Address peerAddress,
                                          PmpReasonCode & reasonCode)
{
  if (m_stats.linksTotal >= m_maxNumberOfPeerLinks)
    {
      reasonCode = REASON11S_MESH_MAX_PEERS;
      return false;
    }
  return true;
}

void
PeerManagementProtocol::CheckBeaconCollisions (uint32_t interface)
{
  if (!GetBeaconCollisionAvoidance ())
    {
      return;
    }
  PeerLinksMap::iterator iface = m_peerLinks.find (interface);
  NS_ASSERT (iface != m_peerLinks.end ());
  NS_ASSERT (m_plugins.find (interface) != m_plugins.end ());

  std::map<uint32_t, Time>::const_iterator lastBeacon = m_lastBeacon.find (interface);
  std::map<uint32_t, Time>::const_iterator beaconInterval = m_beaconInterval.find (interface);
  if ((lastBeacon == m_lastBeacon.end ()) || (beaconInterval == m_beaconInterval.end ()))
    {
      return;
    }
  //my last beacon in 256 us units
  uint16_t lastBeaconInTimeElement = (uint16_t) ((lastBeacon->second.GetMicroSeconds () >> 8) & 0xffff);

  NS_ASSERT_MSG (TuToTime (m_maxBeaconShift) <= m_beaconInterval[interface], "Wrong beacon shift parameters");

  if (iface->second.size () == 0)
    {
      //I have no peers - may be our beacons are in collision
      ShiftOwnBeacon (interface);
      return;
    }
  //check whether all my peers receive my beacon and I'am not in collision with other beacons

  for (PeerLinksOnInterface::iterator i = iface->second.begin (); i != iface->second.end (); i++)
    {
      bool myBeaconExists = false;
      IeBeaconTiming::NeighboursTimingUnitsList neighbors = (*i)->GetBeaconTimingElement ().GetNeighboursTimingElementsList ();
      for (IeBeaconTiming::NeighboursTimingUnitsList::const_iterator j = neighbors.begin (); j != neighbors.end (); j++)
        {
          if ((*i)->GetPeerAid () == (*j)->GetAid ())
            {
              // I am presented at neighbour's list of neighbors
              myBeaconExists = true;
              continue;
            }
          if (
            ((int16_t) ((*j)->GetLastBeacon () - lastBeaconInTimeElement) >= 0) &&
            (((*j)->GetLastBeacon () - lastBeaconInTimeElement) % (4 * TimeToTu (beaconInterval->second)) == 0)
            )
            {
              ShiftOwnBeacon (interface);
              return;
            }
        }
      if (!myBeaconExists)
        {
          // If I am not present in neighbor's beacon timing element, this may be caused by collisions with
          ShiftOwnBeacon (interface);
          return;
        }
    }
}

void
PeerManagementProtocol::ShiftOwnBeacon (uint32_t interface)
{
  int shift = 0;
  do
    {
      shift = (int) m_beaconShift->GetValue ();
    }
  while (shift == 0);
  // Apply beacon shift parameters:
  PeerManagementProtocolMacMap::iterator plugin = m_plugins.find (interface);
  NS_ASSERT (plugin != m_plugins.end ());
  plugin->second->SetBeaconShift (TuToTime (shift));
}

Time
PeerManagementProtocol::TuToTime (int x)
{
  return MicroSeconds (x * 1024);
}
int
PeerManagementProtocol::TimeToTu (Time x)
{
  return (int)(x.GetMicroSeconds () / 1024);
}

void
PeerManagementProtocol::NotifyLinkOpen (Mac48Address peerMp, Mac48Address peerIface, Mac48Address myIface, uint32_t interface)
{
  NS_LOG_LOGIC ("link_open " << myIface << " " << peerIface);
  m_stats.linksOpened++;
  m_stats.linksTotal++;
  if (!m_peerStatusCallback.IsNull ())
    {
      m_peerStatusCallback (peerMp, peerIface, interface, true);
    }
  m_linkOpenTraceSrc (myIface, peerIface);
}

void
PeerManagementProtocol::NotifyLinkClose (Mac48Address peerMp, Mac48Address peerIface, Mac48Address myIface, uint32_t interface)
{
  NS_LOG_LOGIC ("link_close " << myIface << " " << peerIface);
  m_stats.linksClosed++;
  m_stats.linksTotal--;
  if (!m_peerStatusCallback.IsNull ())
    {
      m_peerStatusCallback (peerMp, peerIface, interface, false);
    }
  m_linkCloseTraceSrc (myIface, peerIface);
}

void
PeerManagementProtocol::PeerLinkStatus (uint32_t interface, Mac48Address peerAddress,
                                        Mac48Address peerMeshPointAddress, PeerLink::PeerState ostate, PeerLink::PeerState nstate)
{
  PeerManagementProtocolMacMap::iterator plugin = m_plugins.find (interface);
  NS_ASSERT (plugin != m_plugins.end ());
  NS_LOG_DEBUG ("Link between me:" << m_address << " my interface:" 
                    << plugin->second->GetAddress ()
                    << " and peer mesh point:" << peerMeshPointAddress << " and its interface:" << peerAddress
                    << ", at my interface ID:" << interface << ". State movement:" << PeerLink::PeerStateNames[ostate] 
                    << " -> " << PeerLink::PeerStateNames[nstate]);
  if ((nstate == PeerLink::ESTAB) && (ostate != PeerLink::ESTAB))
    {
      NotifyLinkOpen (peerMeshPointAddress, peerAddress, plugin->second->GetAddress (), interface);
    }
  if ((ostate == PeerLink::ESTAB) && (nstate != PeerLink::ESTAB))
    {
      NotifyLinkClose (peerMeshPointAddress, peerAddress, plugin->second->GetAddress (), interface);
    }
  if (nstate == PeerLink::IDLE)
    {
      Ptr<PeerLink> link = FindPeerLink (interface, peerAddress);
      NS_ASSERT (link == 0);
    }
}
uint8_t
PeerManagementProtocol::GetNumberOfLinks ()
{
  return m_stats.linksTotal;
}
Ptr<IeMeshId>
PeerManagementProtocol::GetMeshId () const
{
  NS_ASSERT (m_meshId != 0);
  return m_meshId;
}
void
PeerManagementProtocol::SetMeshId (std::string s)
{
  m_meshId = Create<IeMeshId> (s);
}
Mac48Address
PeerManagementProtocol::GetAddress ()
{
  return m_address;
}
void
PeerManagementProtocol::NotifyBeaconSent (uint32_t interface, Time beaconInterval)
{
  m_lastBeacon[interface] = Simulator::Now ();
  Simulator::Schedule (beaconInterval - TuToTime (m_maxBeaconShift + 1), &PeerManagementProtocol::CheckBeaconCollisions,this, interface);
  m_beaconInterval[interface] = beaconInterval;
}
PeerManagementProtocol::Statistics::Statistics (uint16_t t) :
  linksTotal (t), linksOpened (0), linksClosed (0)
{
}
void
PeerManagementProtocol::Statistics::Print (std::ostream & os) const
{
  os << "<Statistics "
  "linksTotal=\"" << linksTotal << "\" "
  "linksOpened=\"" << linksOpened << "\" "
  "linksClosed=\"" << linksClosed << "\"/>" << std::endl;
}
void
PeerManagementProtocol::Report (std::ostream & os) const
{
  os << "<PeerManagementProtocol>" << std::endl;
  m_stats.Print (os);
  for (PeerManagementProtocolMacMap::const_iterator plugins = m_plugins.begin (); plugins != m_plugins.end (); plugins++)
    {
      //Take statistics from plugin:
      plugins->second->Report (os);
      //Print all active peer links:
      PeerLinksMap::const_iterator iface = m_peerLinks.find (plugins->second->m_ifIndex);
      NS_ASSERT (iface != m_peerLinks.end ());
      for (PeerLinksOnInterface::const_iterator i = iface->second.begin (); i != iface->second.end (); i++)
        {
          (*i)->Report (os);
        }
    }
  os << "</PeerManagementProtocol>" << std::endl;
}
void
PeerManagementProtocol::ResetStats ()
{
  m_stats = Statistics (m_stats.linksTotal); // don't reset number of links
  for (PeerManagementProtocolMacMap::const_iterator plugins = m_plugins.begin (); plugins != m_plugins.end (); plugins++)
    {
      plugins->second->ResetStats ();
    }
}

int64_t
PeerManagementProtocol::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_beaconShift->SetStream (stream);
  return 1;
}

void
PeerManagementProtocol::DoInitialize ()
{
  // If beacon interval is equal to the neighbor's one and one o more beacons received
  // by my neighbor coincide with my beacon - apply random uniformly distributed shift from
  // [-m_maxBeaconShift, m_maxBeaconShift] except 0.
  m_beaconShift->SetAttribute ("Min", DoubleValue (-m_maxBeaconShift));
  m_beaconShift->SetAttribute ("Max", DoubleValue (m_maxBeaconShift));
}

void
PeerManagementProtocol::SetBeaconCollisionAvoidance (bool enable)
{
  m_enableBca = enable;
}
bool
PeerManagementProtocol::GetBeaconCollisionAvoidance () const
{
  return m_enableBca;
}
} // namespace dot11s
} // namespace ns3

