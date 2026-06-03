/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Ritsumeikan University, Shiga, Japan.
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
 *  Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */

#include "lr-wpan-mac-pl-headers.h"
#include <ns3/simulator.h>

namespace ns3 {

/***********************************************************
 *                Beacon MAC Payload
 ***********************************************************/

BeaconPayloadHeader::BeaconPayloadHeader ()
{
}

NS_OBJECT_ENSURE_REGISTERED (BeaconPayloadHeader);

TypeId
BeaconPayloadHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BeaconPayloadHeader")
    .SetParent<Header> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<BeaconPayloadHeader> ()
  ;
  return tid;
}

TypeId
BeaconPayloadHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
BeaconPayloadHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += m_superframeField.GetSerializedSize ();
  size += m_gtsFields.GetSerializedSize ();
  size += m_pndAddrFields.GetSerializedSize ();

  return size;
}

void
BeaconPayloadHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i = m_superframeField.Serialize (i);
  i = m_gtsFields.Serialize (i);
  i = m_pndAddrFields.Serialize (i);
}

uint32_t
BeaconPayloadHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  i = m_superframeField.Deserialize (i);
  i = m_gtsFields.Deserialize (i);
  i = m_pndAddrFields.Deserialize (i);

  return i.GetDistanceFrom (start);
}


void
BeaconPayloadHeader::Print (std::ostream &os) const
{
  os << "| Superframe Spec Field | = " << m_superframeField
     << "| GTS Spec Field | = " << m_gtsFields.GetGtsSpecField ()
     << "| Pending Spec Field| =" << m_pndAddrFields.GetPndAddrSpecField ();
}


void
BeaconPayloadHeader::SetSuperframeSpecField (SuperframeField sf)
{
  m_superframeField = sf;
}

void
BeaconPayloadHeader::SetGtsFields (GtsFields gtsFields)
{
  m_gtsFields = gtsFields;
}

void
BeaconPayloadHeader::SetPndAddrFields (PendingAddrFields pndAddrFields)
{
  m_pndAddrFields = pndAddrFields;
}

SuperframeField
BeaconPayloadHeader::GetSuperframeSpecField (void) const
{
  return m_superframeField;
}

GtsFields
BeaconPayloadHeader::GetGtsFields (void) const
{
  return m_gtsFields;
}

PendingAddrFields
BeaconPayloadHeader::GetPndAddrFields (void) const
{
  return m_pndAddrFields;
}


/***********************************************************
 *                Command MAC Payload
 ***********************************************************/

CommandPayloadHeader::CommandPayloadHeader ()
{
  SetCommandFrameType (CMD_RESERVED);
}


CommandPayloadHeader::CommandPayloadHeader (enum MacCommand macCmd)
{
  SetCommandFrameType (macCmd);
}

NS_OBJECT_ENSURE_REGISTERED (CommandPayloadHeader);

TypeId
CommandPayloadHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::CommandPayloadHeader")
    .SetParent<Header> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<CommandPayloadHeader> ()
  ;
  return tid;
}

TypeId
CommandPayloadHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
CommandPayloadHeader::GetSerializedSize (void) const
{
  uint32_t size = 1;

  return size;
}

void
CommandPayloadHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_cmdFrameId);
}

uint32_t
CommandPayloadHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_cmdFrameId = i.ReadU8 ();

  return i.GetDistanceFrom (start);
}

void
CommandPayloadHeader::Print (std::ostream &os) const
{
  os << "| MAC Command Frame ID | = " << (uint32_t) m_cmdFrameId;
}

void
CommandPayloadHeader::SetCommandFrameType (MacCommand macCommand)
{
  m_cmdFrameId = macCommand;
}


CommandPayloadHeader::MacCommand
CommandPayloadHeader::GetCommandFrameType (void) const
{
  switch (m_cmdFrameId)
    {
    case 0x01:
      return ASSOCIATION_REQ;
      break;
    case 0x02:
      return ASSOCIATION_RESP;
      break;
    case 0x03:
      return DISASSOCIATION_NOTIF;
      break;
    case 0x04:
      return DATA_REQ;
      break;
    case 0x05:
      return PANID_CONFLICT;
      break;
    case 0x06:
      return ORPHAN_NOTIF;
      break;
    case 0x07:
      return BEACON_REQ;
      break;
    case 0x08:
      return COOR_REALIGN;
      break;
    case 0x09:
      return GTS_REQ;
      break;
    default:
      return CMD_RESERVED;
    }
}


} // ns3 namespace
