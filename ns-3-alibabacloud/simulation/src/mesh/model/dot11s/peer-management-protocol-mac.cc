/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ie-dot11s-configuration.h"
#include "ie-dot11s-peer-management.h"
#include "dot11s-mac-header.h"
#include "peer-management-protocol-mac.h"
#include "peer-management-protocol.h"
#include "peer-link-frame.h"
#include "ns3/wifi-mac-queue-item.h"
#include "ns3/mesh-wifi-interface-mac.h"
#include "ns3/simulator.h"
#include "ns3/mesh-information-element-vector.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PeerManagementProtocolMac");

namespace dot11s {
PeerManagementProtocolMac::PeerManagementProtocolMac (uint32_t interface,
                                                      Ptr<PeerManagementProtocol> protocol)
{
  m_ifIndex = interface;
  m_protocol = protocol;
}

PeerManagementProtocolMac::~PeerManagementProtocolMac ()
{
}

void
PeerManagementProtocolMac::SetParent (Ptr<MeshWifiInterfaceMac> parent)
{
  m_parent = parent;
  m_parent->TraceConnectWithoutContext ("DroppedMpdu", MakeCallback (&PeerManagementProtocolMac::TxError, this));
  m_parent->TraceConnectWithoutContext ("AckedMpdu",   MakeCallback (&PeerManagementProtocolMac::TxOk,    this));
}
void
PeerManagementProtocolMac::TxError (WifiMacDropReason reason, Ptr<const WifiMacQueueItem> mpdu)
{
  m_protocol->TransmissionFailure (m_ifIndex, mpdu->GetHeader ().GetAddr1 ());
}
void
PeerManagementProtocolMac::TxOk (Ptr <const WifiMacQueueItem> mpdu)
{
  m_protocol->TransmissionSuccess (m_ifIndex, mpdu->GetHeader ().GetAddr1 ());
}
bool
PeerManagementProtocolMac::Receive (Ptr<Packet> const_packet, const WifiMacHeader & header)
{
  NS_LOG_FUNCTION (this << const_packet << header);
  // First of all we copy a packet, because we need to remove some
  //headers
  Ptr<Packet> packet = const_packet->Copy ();
  if (header.IsBeacon ())
    {
      NS_LOG_DEBUG ("Is Beacon from " << header.GetAddr2 ());
      MgtBeaconHeader beacon_hdr;
      packet->RemoveHeader (beacon_hdr);
      MeshInformationElementVector elements;
      // To determine header size here, we can rely on the knowledge that 
      // this is the last header to remove.
      packet->RemoveHeader (elements, packet->GetSize ());
      Ptr<IeBeaconTiming> beaconTiming = DynamicCast<IeBeaconTiming> (elements.FindFirst (IE_BEACON_TIMING));
      Ptr<IeMeshId> meshId = DynamicCast<IeMeshId> (elements.FindFirst (IE_MESH_ID));

      if ((meshId != 0) && (m_protocol->GetMeshId ()->IsEqual (*meshId)))
        {
          m_protocol->ReceiveBeacon (m_ifIndex, header.GetAddr2 (), MicroSeconds (
                                       beacon_hdr.GetBeaconIntervalUs ()), beaconTiming);
        }
      else
        {
          NS_LOG_DEBUG ("MeshId mismatch " << m_protocol->GetMeshId ()->PeekString () << " " << (*meshId) << "; ignoring");
        }
      // Beacon shall not be dropped. May be needed to another plugins
      return true;
    }
  uint16_t aid = 0;  // applicable only in Confirm message
  IeConfiguration config;
  if (header.IsAction ())
    {
      NS_LOG_DEBUG ("Is action");
      WifiActionHeader actionHdr;
      packet->RemoveHeader (actionHdr);
      WifiActionHeader::ActionValue actionValue = actionHdr.GetAction ();
      // If can not handle - just return;
      if (actionHdr.GetCategory () != WifiActionHeader::SELF_PROTECTED)
        {
          NS_LOG_DEBUG ("Cannot handle non SELF PROTECTED");
          return m_protocol->IsActiveLink (m_ifIndex, header.GetAddr2 ());
        }
      m_stats.rxMgt++;
      m_stats.rxMgtBytes += packet->GetSize ();
      Mac48Address peerAddress = header.GetAddr2 ();
      Mac48Address peerMpAddress = header.GetAddr3 ();
      if (actionValue.selfProtectedAction == WifiActionHeader::PEER_LINK_OPEN)
        {
          NS_LOG_DEBUG ("Received PEER_LINK_OPEN");
          PeerLinkOpenStart::PlinkOpenStartFields fields;
          PeerLinkOpenStart peerFrame;
          packet->RemoveHeader (peerFrame);
          fields = peerFrame.GetFields ();
          if (!fields.meshId.IsEqual ( *(m_protocol->GetMeshId ())))
            {
              NS_LOG_DEBUG ("PEER_LINK_OPEN:  MeshId mismatch");
              m_protocol->ConfigurationMismatch (m_ifIndex, peerAddress);
              // Broken peer link frame - drop it
              m_stats.brokenMgt++;
              return false;
            }
          if (!(m_parent->CheckSupportedRates (fields.rates)))
            {
              NS_LOG_DEBUG ("PEER_LINK_OPEN:  configuration mismatch");
              m_protocol->ConfigurationMismatch (m_ifIndex, peerAddress);
              // Broken peer link frame - drop it
              m_stats.brokenMgt++;
              return false;
            }
          config = fields.config;
        }
      else if (actionValue.selfProtectedAction == WifiActionHeader::PEER_LINK_CONFIRM)
        {
          NS_LOG_DEBUG ("Received PEER_LINK_CONFIRM");
          PeerLinkConfirmStart::PlinkConfirmStartFields fields;
          PeerLinkConfirmStart peerFrame;
          packet->RemoveHeader (peerFrame);
          fields = peerFrame.GetFields ();
          if (!(m_parent->CheckSupportedRates (fields.rates)))
            {
              NS_LOG_DEBUG ("PEER_LINK_CONFIRM:  configuration mismatch");
              m_protocol->ConfigurationMismatch (m_ifIndex, peerAddress);
              // Broken peer link frame - drop it
              m_stats.brokenMgt++;
              return false;
            }
          aid = fields.aid;
          config = fields.config;
        }
      else if (actionValue.selfProtectedAction == WifiActionHeader::PEER_LINK_CLOSE)
        {
          NS_LOG_DEBUG ("Received PEER_LINK_CLOSE");
          PeerLinkCloseStart::PlinkCloseStartFields fields;
          PeerLinkCloseStart peerFrame;
          packet->RemoveHeader (peerFrame);
          fields = peerFrame.GetFields ();
          if (!fields.meshId.IsEqual ( *(m_protocol->GetMeshId ())))
            {
              NS_LOG_DEBUG ("PEER_LINK_CLOSE:  configuration mismatch");
              m_protocol->ConfigurationMismatch (m_ifIndex, peerAddress);
              // Broken peer link frame - drop it
              m_stats.brokenMgt++;
              return false;
            }
        }
      else
        {
          NS_FATAL_ERROR ("Unknown Self-protected Action type: " << actionValue.selfProtectedAction);
        }
      Ptr<IePeerManagement> peerElement;
      MeshInformationElementVector elements;
      // To determine header size here, we can rely on the knowledge that 
      // this is the last header to remove.
      packet->RemoveHeader (elements, packet->GetSize ());
      peerElement = DynamicCast<IePeerManagement>(elements.FindFirst (IE_MESH_PEERING_MANAGEMENT));

      NS_ASSERT (peerElement != 0);
      //Check that frame subtype corresponds to peer link subtype
      if (peerElement->SubtypeIsOpen ())
        {
          m_stats.rxOpen++;
          NS_ASSERT (actionValue.selfProtectedAction == WifiActionHeader::PEER_LINK_OPEN);
        }
      if (peerElement->SubtypeIsConfirm ())
        {
          m_stats.rxConfirm++;
          NS_ASSERT (actionValue.selfProtectedAction == WifiActionHeader::PEER_LINK_CONFIRM); 
        }
      if (peerElement->SubtypeIsClose ())
        {
          m_stats.rxClose++;
          NS_ASSERT (actionValue.selfProtectedAction == WifiActionHeader::PEER_LINK_CLOSE); 
        }
      //Deliver Peer link management frame to protocol:
      m_protocol->ReceivePeerLinkFrame (m_ifIndex, peerAddress, peerMpAddress, aid, *peerElement, config);
      // if we can handle a frame - drop it
      return false;
    }
  return m_protocol->IsActiveLink (m_ifIndex, header.GetAddr2 ());
}
bool
PeerManagementProtocolMac::UpdateOutcomingFrame (Ptr<Packet> packet, WifiMacHeader & header,
                                                 Mac48Address from, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << header << from << to);
  if (header.IsAction ())
    {
      WifiActionHeader actionHdr;
      packet->PeekHeader (actionHdr);
      if (actionHdr.GetCategory () == WifiActionHeader::SELF_PROTECTED) 
        {
          return true;
        }
    }
  if (header.GetAddr1 ().IsGroup ())
    {
      return true;
    }
  else
    {
      if (m_protocol->IsActiveLink (m_ifIndex, header.GetAddr1 ()))
        {
          return true;
        }
      else
        {
          m_stats.dropped++;
          return false;
        }
    }
}
void
PeerManagementProtocolMac::UpdateBeacon (MeshWifiBeacon & beacon) const
{
  if (m_protocol->GetBeaconCollisionAvoidance ())
    {
      Ptr<IeBeaconTiming> beaconTiming = m_protocol->GetBeaconTimingElement (m_ifIndex);
      beacon.AddInformationElement (beaconTiming);
    }
  beacon.AddInformationElement (m_protocol->GetMeshId ());
  m_protocol->NotifyBeaconSent (m_ifIndex, beacon.GetBeaconInterval ());
}

void
PeerManagementProtocolMac::SendPeerLinkManagementFrame (Mac48Address peerAddress, Mac48Address peerMpAddress,
                                                        uint16_t aid, IePeerManagement peerElement, IeConfiguration meshConfig)
{
  NS_LOG_FUNCTION (this << peerAddress << peerMpAddress);
  //Create a packet:
  meshConfig.SetNeighborCount (m_protocol->GetNumberOfLinks ());
  Ptr<Packet> packet = Create<Packet> ();
  MeshInformationElementVector elements;
  elements.AddInformationElement (Ptr<IePeerManagement> (&peerElement));
  packet->AddHeader (elements);
  //Create an 802.11 frame header:
  //Send management frame to MAC:
  if (peerElement.SubtypeIsOpen ())
    {
      PeerLinkOpenStart::PlinkOpenStartFields fields;
      fields.rates = m_parent->GetSupportedRates ();
      fields.capability = 0;
      fields.meshId = *(m_protocol->GetMeshId ());
      fields.config = meshConfig;
      PeerLinkOpenStart plinkOpen;
      WifiActionHeader actionHdr;
      m_stats.txOpen++;
      WifiActionHeader::ActionValue action;
      action.selfProtectedAction = WifiActionHeader::PEER_LINK_OPEN;
      actionHdr.SetAction (WifiActionHeader::SELF_PROTECTED, action); 
      plinkOpen.SetPlinkOpenStart (fields);
      packet->AddHeader (plinkOpen);
      packet->AddHeader (actionHdr);
    }
  if (peerElement.SubtypeIsConfirm ())
    {
      PeerLinkConfirmStart::PlinkConfirmStartFields fields;
      fields.rates = m_parent->GetSupportedRates ();
      fields.capability = 0;
      fields.config = meshConfig;
      PeerLinkConfirmStart plinkConfirm;
      WifiActionHeader actionHdr;
      m_stats.txConfirm++;
      WifiActionHeader::ActionValue action;
      action.selfProtectedAction = WifiActionHeader::PEER_LINK_CONFIRM; 
      fields.aid = aid;
      actionHdr.SetAction (WifiActionHeader::SELF_PROTECTED, action); 
      plinkConfirm.SetPlinkConfirmStart (fields);
      packet->AddHeader (plinkConfirm);
      packet->AddHeader (actionHdr);
    }
  if (peerElement.SubtypeIsClose ())
    {
      PeerLinkCloseStart::PlinkCloseStartFields fields;
      fields.meshId = *(m_protocol->GetMeshId ());
      PeerLinkCloseStart plinkClose;
      WifiActionHeader actionHdr;
      m_stats.txClose++;
      WifiActionHeader::ActionValue action;
      action.selfProtectedAction = WifiActionHeader::PEER_LINK_CLOSE; 
      actionHdr.SetAction (WifiActionHeader::SELF_PROTECTED, action); 
      plinkClose.SetPlinkCloseStart (fields);
      packet->AddHeader (plinkClose);
      packet->AddHeader (actionHdr);
    }
  m_stats.txMgt++;
  m_stats.txMgtBytes += packet->GetSize ();
  // Wifi Mac header:
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetAddr1 (peerAddress);
  hdr.SetAddr2 (m_parent->GetAddress ());
  //Addr is not used here, we use it as our MP address
  hdr.SetAddr3 (m_protocol->GetAddress ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  m_parent->SendManagementFrame (packet, hdr);
}

Mac48Address
PeerManagementProtocolMac::GetAddress () const
{
  if (m_parent != 0)
    {
      return m_parent->GetAddress ();
    }
  else
    {
      return Mac48Address ();
    }
}
void
PeerManagementProtocolMac::SetBeaconShift (Time shift)
{
  if (shift != Seconds (0))
    {
      m_stats.beaconShift++;
    }
  m_parent->ShiftTbtt (shift);
}
PeerManagementProtocolMac::Statistics::Statistics () :
  txOpen (0), txConfirm (0), txClose (0), rxOpen (0), rxConfirm (0), rxClose (0), dropped (0), brokenMgt (0),
  txMgt (0), txMgtBytes (0), rxMgt (0), rxMgtBytes (0), beaconShift (0)
{
}
void
PeerManagementProtocolMac::Statistics::Print (std::ostream & os) const
{
  os << "<Statistics "
  "txOpen=\"" << txOpen << "\"" << std::endl <<
  "txConfirm=\"" << txConfirm << "\"" << std::endl <<
  "txClose=\"" << txClose << "\"" << std::endl <<
  "rxOpen=\"" << rxOpen << "\"" << std::endl <<
  "rxConfirm=\"" << rxConfirm << "\"" << std::endl <<
  "rxClose=\"" << rxClose << "\"" << std::endl <<
  "dropped=\"" << dropped << "\"" << std::endl <<
  "brokenMgt=\"" << brokenMgt << "\"" << std::endl <<
  "txMgt=\"" << txMgt << "\"" << std::endl <<
  "txMgtBytes=\"" << txMgtBytes << "\"" << std::endl <<
  "rxMgt=\"" << rxMgt << "\"" << std::endl <<
  "rxMgtBytes=\"" << rxMgtBytes << "\"" << std::endl <<
  "beaconShift=\"" << beaconShift << "\"/>" << std::endl;
}
void
PeerManagementProtocolMac::Report (std::ostream & os) const
{
  os << "<PeerManagementProtocolMac "
  "address=\"" << m_parent->GetAddress () << "\">" << std::endl;
  m_stats.Print (os);
  os << "</PeerManagementProtocolMac>" << std::endl;
}
void
PeerManagementProtocolMac::ResetStats ()
{
  m_stats = Statistics ();
}
uint32_t
PeerManagementProtocolMac::GetLinkMetric (Mac48Address peerAddress)
{
  return m_parent->GetLinkMetric (peerAddress);
}
int64_t
PeerManagementProtocolMac::AssignStreams (int64_t stream)
{ 
  return m_protocol->AssignStreams (stream);
}

} // namespace dot11s
} // namespace ns3

