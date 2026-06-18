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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ns3/mesh-wifi-interface-mac.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/log.h"
#include "ns3/wifi-mac-header.h"
#include "dot11s-mac-header.h"
#include "hwmp-protocol-mac.h"
#include "hwmp-tag.h"
#include "ie-dot11s-preq.h"
#include "ie-dot11s-prep.h"
#include "ie-dot11s-rann.h"
#include "ie-dot11s-perr.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HwmpProtocolMac");
  
namespace dot11s {

HwmpProtocolMac::HwmpProtocolMac (uint32_t ifIndex, Ptr<HwmpProtocol> protocol) :
  m_ifIndex (ifIndex), m_protocol (protocol)
{
  NS_LOG_FUNCTION (this << ifIndex << protocol);
}
HwmpProtocolMac::~HwmpProtocolMac ()
{
}
void
HwmpProtocolMac::SetParent (Ptr<MeshWifiInterfaceMac> parent)
{
  NS_LOG_FUNCTION (this << parent);
  m_parent = parent;
}

bool
HwmpProtocolMac::ReceiveData (Ptr<Packet> packet, const WifiMacHeader & header)
{
  NS_LOG_FUNCTION (this << packet << header);
  NS_ASSERT (header.IsData ());

  MeshHeader meshHdr;
  HwmpTag tag;
  if (packet->PeekPacketTag (tag))
    {
      NS_FATAL_ERROR ("HWMP tag is not supposed to be received by network");
    }

  packet->RemoveHeader (meshHdr);
  m_stats.rxData++;
  m_stats.rxDataBytes += packet->GetSize ();

  /// \todo address extension
  Mac48Address destination;
  Mac48Address source;
  switch (meshHdr.GetAddressExt ())
    {
    case 0:
      source = header.GetAddr4 ();
      destination = header.GetAddr3 ();
      break;
    default:
      NS_FATAL_ERROR (
        "6-address scheme is not yet supported and 4-address extension is not supposed to be used for data frames.");
    }
  tag.SetSeqno (meshHdr.GetMeshSeqno ());
  tag.SetTtl (meshHdr.GetMeshTtl ());
  packet->AddPacketTag (tag);

  if ((destination == Mac48Address::GetBroadcast ()) && (m_protocol->DropDataFrame (meshHdr.GetMeshSeqno (),
                                                                                    source)))
    {
      NS_LOG_DEBUG ("Dropping frame; source " << source << " dest " << destination << " seqno " << meshHdr.GetMeshSeqno ()); 
      return false;
    }
  return true;
}

bool
HwmpProtocolMac::ReceiveAction (Ptr<Packet> packet, const WifiMacHeader & header)
{
  NS_LOG_FUNCTION (this << packet << header);
  m_stats.rxMgt++;
  m_stats.rxMgtBytes += packet->GetSize ();
  WifiActionHeader actionHdr;
  packet->RemoveHeader (actionHdr);
  if (actionHdr.GetCategory () != WifiActionHeader::MESH)
    {
      return true;
    }
  MeshInformationElementVector elements;
  // To determine header size here, we can rely on the knowledge that
  // this is the last header to remove.
  packet->RemoveHeader (elements, packet->GetSize ());
  std::vector<HwmpProtocol::FailedDestination> failedDestinations;
  for (MeshInformationElementVector::Iterator i = elements.Begin (); i != elements.End (); i++)
    {
      if ((*i)->ElementId () == IE_RANN)
        {
          NS_LOG_WARN ("RANN is not supported!");
        }
      if ((*i)->ElementId () == IE_PREQ)
        {
          Ptr<IePreq> preq = DynamicCast<IePreq> (*i);
          NS_ASSERT (preq != 0);
          m_stats.rxPreq++;
          if (preq->GetOriginatorAddress () == m_protocol->GetAddress ())
            {
              continue;
            }
          if (preq->GetTtl () == 0)
            {
              continue;
            }
          preq->DecrementTtl ();
          m_protocol->ReceivePreq (*preq, header.GetAddr2 (), m_ifIndex, header.GetAddr3 (),
                                   m_parent->GetLinkMetric (header.GetAddr2 ()));
        }
      if ((*i)->ElementId () == IE_PREP)
        {
          Ptr<IePrep> prep = DynamicCast<IePrep> (*i);
          NS_ASSERT (prep != 0);
          m_stats.rxPrep++;
          if (prep->GetTtl () == 0)
            {
              continue;
            }
          prep->DecrementTtl ();
          m_protocol->ReceivePrep (*prep, header.GetAddr2 (), m_ifIndex, header.GetAddr3 (),
                                   m_parent->GetLinkMetric (header.GetAddr2 ()));
        }
      if ((*i)->ElementId () == IE_PERR)
        {
          Ptr<IePerr> perr = DynamicCast<IePerr> (*i);
          NS_ASSERT (perr != 0);
          m_stats.rxPerr++;
          std::vector<HwmpProtocol::FailedDestination> destinations = perr->GetAddressUnitVector ();
          for (std::vector<HwmpProtocol::FailedDestination>::const_iterator i = destinations.begin (); i
               != destinations.end (); i++)
            {
              failedDestinations.push_back (*i);
            }
        }
    }
  if (failedDestinations.size () > 0)
    {
      m_protocol->ReceivePerr (failedDestinations, header.GetAddr2 (), m_ifIndex, header.GetAddr3 ());
    }
  NS_ASSERT (packet->GetSize () == 0);
  return false;
}

bool
HwmpProtocolMac::Receive (Ptr<Packet> packet, const WifiMacHeader & header)
{
  NS_LOG_FUNCTION (this << packet << header);
  if (header.IsData ())
    {
      return ReceiveData (packet, header);
    }
  else
    {
      if (header.IsAction ())
        {
          return ReceiveAction (packet, header);
        }
      else
        {
          return true; // don't care
        }
    }
}
bool
HwmpProtocolMac::UpdateOutcomingFrame (Ptr<Packet> packet, WifiMacHeader & header, Mac48Address from,
                                       Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << header << from << to);
  if (!header.IsData ())
    {
      return true;
    }
  HwmpTag tag;
  bool tagExists = packet->RemovePacketTag (tag);
  if (!tagExists)
    {
      NS_FATAL_ERROR ("HWMP tag must exist at this point");
    }
  m_stats.txData++;
  m_stats.txDataBytes += packet->GetSize ();
  MeshHeader meshHdr;
  meshHdr.SetMeshSeqno (tag.GetSeqno ());
  meshHdr.SetMeshTtl (tag.GetTtl ());
  packet->AddHeader (meshHdr);
  header.SetAddr1 (tag.GetAddress ());
  header.SetQosMeshControlPresent ();
  return true;
}
WifiActionHeader
HwmpProtocolMac::GetWifiActionHeader ()
{
  WifiActionHeader actionHdr;
  WifiActionHeader::ActionValue action;
  action.meshAction = WifiActionHeader::PATH_SELECTION;
  actionHdr.SetAction (WifiActionHeader::MESH, action); 
  return actionHdr;
}
void
HwmpProtocolMac::SendPreq (IePreq preq)
{
  NS_LOG_FUNCTION (this);
  std::vector<IePreq> preq_vector;
  preq_vector.push_back (preq);
  SendPreqVector (preq_vector);
}
void
HwmpProtocolMac::SendPreqVector (std::vector<IePreq> preq)
{
  NS_LOG_FUNCTION (this);
  Ptr<Packet> packet = Create<Packet> ();
  MeshInformationElementVector elements;
  for (std::vector<IePreq>::iterator i = preq.begin (); i != preq.end (); i++)
    {
      elements.AddInformationElement (Ptr<IePreq> (&(*i)));
    }
  packet->AddHeader (elements);
  packet->AddHeader (GetWifiActionHeader ());
  //create 802.11 header:
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  hdr.SetAddr2 (m_parent->GetAddress ());
  hdr.SetAddr3 (m_protocol->GetAddress ());
  //Send Management frame
  std::vector<Mac48Address> receivers = m_protocol->GetPreqReceivers (m_ifIndex);
  for (std::vector<Mac48Address>::const_iterator i = receivers.begin (); i != receivers.end (); i++)
    {
      hdr.SetAddr1 (*i);
      m_stats.txPreq++;
      m_stats.txMgt++;
      m_stats.txMgtBytes += packet->GetSize ();
      m_parent->SendManagementFrame (packet, hdr);
    }
}
void
HwmpProtocolMac::RequestDestination (Mac48Address dst, uint32_t originator_seqno, uint32_t dst_seqno)
{
  NS_LOG_FUNCTION (this << dst << originator_seqno << dst_seqno);
  for (std::vector<IePreq>::iterator i = m_myPreq.begin (); i != m_myPreq.end (); i++)
    {
      if (i->IsFull ())
        {
          continue;
        }
      NS_ASSERT (i->GetDestCount () > 0);
      i->AddDestinationAddressElement (m_protocol->GetDoFlag (), m_protocol->GetRfFlag (), dst, dst_seqno);
    }
  IePreq preq;
  preq.SetHopcount (0);
  preq.SetTTL (m_protocol->GetMaxTtl ());
  preq.SetPreqID (m_protocol->GetNextPreqId ());
  preq.SetOriginatorAddress (m_protocol->GetAddress ());
  preq.SetOriginatorSeqNumber (originator_seqno);
  preq.SetLifetime (m_protocol->GetActivePathLifetime ());
  preq.AddDestinationAddressElement (m_protocol->GetDoFlag (), m_protocol->GetRfFlag (), dst, dst_seqno);
  m_myPreq.push_back (preq);
  SendMyPreq ();
}
void
HwmpProtocolMac::SendMyPreq ()
{
  NS_LOG_FUNCTION (this);
  if (m_preqTimer.IsRunning ())
    {
      return;
    }
  if (m_myPreq.size () == 0)
    {
      return;
    }
  //reschedule sending PREQ
  NS_ASSERT (!m_preqTimer.IsRunning ());
  m_preqTimer = Simulator::Schedule (m_protocol->GetPreqMinInterval (), &HwmpProtocolMac::SendMyPreq, this);
  SendPreqVector (m_myPreq);
  m_myPreq.clear ();
}
void
HwmpProtocolMac::SendPrep (IePrep prep, Mac48Address receiver)
{
  NS_LOG_FUNCTION (this << receiver);
  //Create packet
  Ptr<Packet> packet = Create<Packet> ();
  MeshInformationElementVector elements;
  elements.AddInformationElement (Ptr<IePrep> (&prep));
  packet->AddHeader (elements);
  packet->AddHeader (GetWifiActionHeader ());
  //create 802.11 header:
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  hdr.SetAddr1 (receiver);
  hdr.SetAddr2 (m_parent->GetAddress ());
  hdr.SetAddr3 (m_protocol->GetAddress ());
  //Send Management frame
  m_stats.txPrep++;
  m_stats.txMgt++;
  m_stats.txMgtBytes += packet->GetSize ();
  m_parent->SendManagementFrame (packet, hdr);
}
void
HwmpProtocolMac::ForwardPerr (std::vector<HwmpProtocol::FailedDestination> failedDestinations, std::vector<
                                Mac48Address> receivers)
{
  NS_LOG_FUNCTION (this);
  Ptr<Packet> packet = Create<Packet> ();
  Ptr<IePerr> perr = Create <IePerr> ();
  MeshInformationElementVector elements;
  for (std::vector<HwmpProtocol::FailedDestination>::const_iterator i = failedDestinations.begin (); i
       != failedDestinations.end (); i++)
    {
      if (!perr->IsFull ())
        {
          perr->AddAddressUnit (*i);
        }
      else
        {
          elements.AddInformationElement (perr);
          perr->ResetPerr ();
        }
    }
  if (perr->GetNumOfDest () > 0)
    {
      elements.AddInformationElement (perr);
    }
  packet->AddHeader (elements);
  packet->AddHeader (GetWifiActionHeader ());
  //create 802.11 header:
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  hdr.SetAddr2 (m_parent->GetAddress ());
  hdr.SetAddr3 (m_protocol->GetAddress ());
  if (receivers.size () >= m_protocol->GetUnicastPerrThreshold ())
    {
      receivers.clear ();
      receivers.push_back (Mac48Address::GetBroadcast ());
    }
  //Send Management frame
  for (std::vector<Mac48Address>::const_iterator i = receivers.begin (); i != receivers.end (); i++)
    {
      //
      // 64-bit Intel valgrind complains about hdr.SetAddr1 (*i).  It likes this
      // just fine.
      //
      Mac48Address address = *i;
      hdr.SetAddr1 (address);
      m_stats.txPerr++;
      m_stats.txMgt++;
      m_stats.txMgtBytes += packet->GetSize ();
      m_parent->SendManagementFrame (packet, hdr);
    }
}
void
HwmpProtocolMac::InitiatePerr (std::vector<HwmpProtocol::FailedDestination> failedDestinations, std::vector<
                                 Mac48Address> receivers)
{
  NS_LOG_FUNCTION (this);
  //All duplicates in PERR are checked here, and there is no reason to
  //check it at any other place
  {
    std::vector<Mac48Address>::const_iterator end = receivers.end ();
    for (std::vector<Mac48Address>::const_iterator i = receivers.begin (); i != end; i++)
      {
        bool should_add = true;
        for (std::vector<Mac48Address>::const_iterator j = m_myPerr.receivers.begin (); j
             != m_myPerr.receivers.end (); j++)
          {
            if ((*i) == (*j))
              {
                should_add = false;
              }
          }
        if (should_add)
          {
            m_myPerr.receivers.push_back (*i);
            NS_LOG_DEBUG ("Initiate PERR:  Adding receiver: " << (*i));
          }
      }
  }
  {
    std::vector<HwmpProtocol::FailedDestination>::const_iterator end = failedDestinations.end ();
    for (std::vector<HwmpProtocol::FailedDestination>::const_iterator i = failedDestinations.begin (); i != end; i++)
      {
        bool should_add = true;
        for (std::vector<HwmpProtocol::FailedDestination>::const_iterator j = m_myPerr.destinations.begin (); j
             != m_myPerr.destinations.end (); j++)
          {
            if (((*i).destination == (*j).destination) && ((*j).seqnum > (*i).seqnum))
              {
                should_add = false;
              }
          }
        if (should_add)
          {
            m_myPerr.destinations.push_back (*i);
            NS_LOG_DEBUG ("Initiate PERR:  Adding failed destination: " << (*i).destination);
          }
      }
  }
  SendMyPerr ();
}
void
HwmpProtocolMac::SendMyPerr ()
{
  NS_LOG_FUNCTION (this);
  if (m_perrTimer.IsRunning ())
    {
      return;
    }
  m_perrTimer = Simulator::Schedule (m_protocol->GetPerrMinInterval (), &HwmpProtocolMac::SendMyPerr, this);
  ForwardPerr (m_myPerr.destinations, m_myPerr.receivers);
  m_myPerr.destinations.clear ();
  m_myPerr.receivers.clear ();
}
uint32_t
HwmpProtocolMac::GetLinkMetric (Mac48Address peerAddress) const
{
  return m_parent->GetLinkMetric (peerAddress);
}
uint16_t
HwmpProtocolMac::GetChannelId () const
{
  return m_parent->GetFrequencyChannel ();
}
HwmpProtocolMac::Statistics::Statistics () :
  txPreq (0), rxPreq (0), txPrep (0), rxPrep (0), txPerr (0), rxPerr (0), txMgt (0), txMgtBytes (0),
  rxMgt (0), rxMgtBytes (0), txData (0), txDataBytes (0), rxData (0), rxDataBytes (0)
{
}
void
HwmpProtocolMac::Statistics::Print (std::ostream & os) const
{
  os << "<Statistics "
  "txPreq= \"" << txPreq << "\"" << std::endl <<
  "txPrep=\"" << txPrep << "\"" << std::endl <<
  "txPerr=\"" << txPerr << "\"" << std::endl <<
  "rxPreq=\"" << rxPreq << "\"" << std::endl <<
  "rxPrep=\"" << rxPrep << "\"" << std::endl <<
  "rxPerr=\"" << rxPerr << "\"" << std::endl <<
  "txMgt=\"" << txMgt << "\"" << std::endl <<
  "txMgtBytes=\"" << txMgtBytes << "\"" << std::endl <<
  "rxMgt=\"" << rxMgt << "\"" << std::endl <<
  "rxMgtBytes=\"" << rxMgtBytes << "\"" << std::endl <<
  "txData=\"" << txData << "\"" << std::endl <<
  "txDataBytes=\"" << txDataBytes << "\"" << std::endl <<
  "rxData=\"" << rxData << "\"" << std::endl <<
  "rxDataBytes=\"" << rxDataBytes << "\"/>" << std::endl;
}
void
HwmpProtocolMac::Report (std::ostream & os) const
{
  os << "<HwmpProtocolMac" << std::endl <<
  "address =\"" << m_parent->GetAddress () << "\">" << std::endl;
  m_stats.Print (os);
  os << "</HwmpProtocolMac>" << std::endl;
}
void
HwmpProtocolMac::ResetStats ()
{
  NS_LOG_FUNCTION (this);
  m_stats = Statistics ();
}

int64_t
HwmpProtocolMac::AssignStreams (int64_t stream)
{
  return m_protocol->AssignStreams (stream);
}


} // namespace dot11s
} // namespace ns3
