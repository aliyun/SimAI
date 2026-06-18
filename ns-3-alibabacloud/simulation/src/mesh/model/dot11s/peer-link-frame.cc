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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#include "peer-link-frame.h"
#include "ie-dot11s-peer-management.h"
#include "ns3/mesh-wifi-interface-mac.h"
#include "ns3/test.h"
#include "ns3/packet.h"

namespace ns3 {
namespace dot11s {
NS_OBJECT_ENSURE_REGISTERED (PeerLinkOpenStart);

PeerLinkOpenStart::PeerLinkOpenStart () :
  m_capability (0), m_rates (SupportedRates ()), m_meshId (),
  m_config (IeConfiguration ())
{
}
void
PeerLinkOpenStart::SetPlinkOpenStart (PeerLinkOpenStart::PlinkOpenStartFields fields)
{
  m_capability = fields.capability;
  m_rates = fields.rates;
  m_meshId = fields.meshId;
  m_config = fields.config;
}
PeerLinkOpenStart::PlinkOpenStartFields
PeerLinkOpenStart::GetFields () const
{
  PlinkOpenStartFields retval;
  /// \todo protocol version:
  retval.capability = m_capability;
  retval.rates = m_rates;
  retval.meshId = m_meshId;
  retval.config = m_config;
  
  return retval;
}
TypeId
PeerLinkOpenStart::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dot11s::PeerLinkOpenStart")
    .SetParent<Header> ()
    .SetGroupName ("Mesh")
    .AddConstructor<PeerLinkOpenStart> ()
  ;
  return tid;
}
TypeId
PeerLinkOpenStart::GetInstanceTypeId () const
{
  return GetTypeId ();
}
void
PeerLinkOpenStart::Print (std::ostream &os) const
{
  os << "capability=" << m_capability << ", rates=" << m_rates;
  os << ", ";
  m_meshId.Print (os);
  os << ", ";
  m_config.Print (os);
}
uint32_t
PeerLinkOpenStart::GetSerializedSize () const
{
  uint32_t size =0; //Peering protocol
  size += 2; //capability
  size += m_rates.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_meshId.GetInformationFieldSize () + 2;
  size += m_config.GetInformationFieldSize () + 2;
  return size;
}
void
PeerLinkOpenStart::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  
  i.WriteHtolsbU16 (m_capability);
  i = m_rates.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_meshId.Serialize (i);
  i = m_config.Serialize (i);
}
uint32_t
PeerLinkOpenStart::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  
  m_capability = i.ReadLsbtohU16 ();
  i = m_rates.Deserialize (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  uint8_t id = i.ReadU8 ();
  uint8_t length = i.ReadU8 ();
  m_meshId.DeserializeInformationField (i, length);
  if ((m_meshId.ElementId () != (WifiInformationElementId) id) || (m_meshId.GetInformationFieldSize () != length))
    {
      NS_FATAL_ERROR ("Broken frame: Element ID does not match IE itself!");
    }
  i.Next (m_meshId.GetInformationFieldSize ());
  id = i.ReadU8 ();
  length = i.ReadU8 ();
  m_config.DeserializeInformationField (i, length);
  if ((m_config.ElementId () != (WifiInformationElementId) id) || (m_config.GetInformationFieldSize () != length))
    {
      NS_FATAL_ERROR ("Broken frame: Element ID does not match IE itself!");
    }
  i.Next (m_config.GetInformationFieldSize ());
 
  return i.GetDistanceFrom (start);
}
bool
operator== (const PeerLinkOpenStart & a, const PeerLinkOpenStart & b)
{
  return ((a.m_capability == b.m_capability)
          && (a.m_meshId.IsEqual (b.m_meshId)) && (a.m_config == b.m_config));
  
}
NS_OBJECT_ENSURE_REGISTERED (PeerLinkCloseStart);

PeerLinkCloseStart::PeerLinkCloseStart () :
  m_meshId ()
{
}
void
PeerLinkCloseStart::SetPlinkCloseStart (PeerLinkCloseStart::PlinkCloseStartFields fields)
{
  m_meshId = fields.meshId;
}
PeerLinkCloseStart::PlinkCloseStartFields
PeerLinkCloseStart::GetFields () const
{
  PlinkCloseStartFields retval;
  /// \todo protocol version:
  retval.meshId = m_meshId;
  
  return retval;
}
TypeId
PeerLinkCloseStart::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dot11s::PeerLinkCloseStart")
    .SetParent<Header> ()
    .SetGroupName ("Mesh")
    .AddConstructor<PeerLinkCloseStart> ()
  ;
  return tid;
}
TypeId
PeerLinkCloseStart::GetInstanceTypeId () const
{
  return GetTypeId ();
}
void
PeerLinkCloseStart::Print (std::ostream &os) const
{
  m_meshId.Print (os);
}
uint32_t
PeerLinkCloseStart::GetSerializedSize () const
{
  uint32_t size =0; //Peering protocol
  size += m_meshId.GetInformationFieldSize () + 2;
  return size;
}
void
PeerLinkCloseStart::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i = m_meshId.Serialize (i);
}
uint32_t
PeerLinkCloseStart::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  
  uint8_t id = i.ReadU8 ();
  uint8_t length = i.ReadU8 ();
  m_meshId.DeserializeInformationField (i, length);
  if ((m_meshId.ElementId () != (WifiInformationElementId) id) || (m_meshId.GetInformationFieldSize () != length))
    {
      NS_FATAL_ERROR ("Broken frame: Element ID does not match IE itself!");
    }
  i.Next (m_meshId.GetInformationFieldSize ());
  return i.GetDistanceFrom (start);
}
bool
operator== (const PeerLinkCloseStart & a, const PeerLinkCloseStart & b)
{
  return ((a.m_meshId.IsEqual (b.m_meshId)));
}
NS_OBJECT_ENSURE_REGISTERED (PeerLinkConfirmStart);

PeerLinkConfirmStart::PeerLinkConfirmStart () :
  m_capability (0), m_aid (0), m_rates (SupportedRates ()),
  m_config (IeConfiguration ())
{
}
void
PeerLinkConfirmStart::SetPlinkConfirmStart (PeerLinkConfirmStart::PlinkConfirmStartFields fields)
{
  m_capability = fields.capability;
  m_aid = fields.aid;
  m_rates = fields.rates;
  m_config = fields.config;
}
PeerLinkConfirmStart::PlinkConfirmStartFields
PeerLinkConfirmStart::GetFields () const
{
  PlinkConfirmStartFields retval;
  /// \todo protocol version:
  retval.capability = m_capability;
  retval.aid = m_aid;
  retval.rates = m_rates;
  retval.config = m_config;
  
  return retval;
}
TypeId
PeerLinkConfirmStart::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dot11s::PeerLinkConfirmStart")
    .SetParent<Header> ()
    .SetGroupName ("Mesh")
    .AddConstructor<PeerLinkConfirmStart> ()
  ;
  return tid;
}
TypeId
PeerLinkConfirmStart::GetInstanceTypeId () const
{
  return GetTypeId ();
}
void
PeerLinkConfirmStart::Print (std::ostream &os) const
{
  os << "capability=" << m_capability << ", rates=" << m_rates;
  os << ", ";
  m_config.Print (os);
}
uint32_t
PeerLinkConfirmStart::GetSerializedSize () const
{
  uint32_t size =0; //Peering protocol
  size += 2; //capability
  size += 2; //AID of remote peer
  size += m_rates.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_config.GetInformationFieldSize () + 2;
  return size;
}
void
PeerLinkConfirmStart::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  
  i.WriteHtolsbU16 (m_capability);
  i.WriteHtolsbU16 (m_aid);
  i = m_rates.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_config.Serialize (i);
}
uint32_t
PeerLinkConfirmStart::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  
  m_capability = i.ReadLsbtohU16 ();
  m_aid = i.ReadLsbtohU16 ();
  i = m_rates.Deserialize (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  uint8_t id = i.ReadU8 ();
  uint8_t length = i.ReadU8 ();
  m_config.DeserializeInformationField (i, length);
  if ((m_config.ElementId () != (WifiInformationElementId) id) || (m_config.GetInformationFieldSize () != length))
    {
      NS_FATAL_ERROR ("Broken frame: Element ID does not match IE itself!");
    }
  i.Next (m_config.GetInformationFieldSize ());
  return i.GetDistanceFrom (start);
}
bool
operator== (const PeerLinkConfirmStart & a, const PeerLinkConfirmStart & b)
{
  return ((a.m_capability == b.m_capability) && (a.m_aid == b.m_aid)
          && (a.m_config == b.m_config));
}
} // namespace dot11s
} // namespace ns3

