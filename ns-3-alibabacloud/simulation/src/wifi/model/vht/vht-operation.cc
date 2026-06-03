/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "vht-operation.h"

namespace ns3 {

VhtOperation::VhtOperation ()
  : m_channelWidth (0),
    m_channelCenterFrequencySegment0 (0),
    m_channelCenterFrequencySegment1 (0),
    m_basicVhtMcsAndNssSet (0),
    m_vhtSupported (0)
{
}

WifiInformationElementId
VhtOperation::ElementId () const
{
  return IE_VHT_OPERATION;
}

void
VhtOperation::SetVhtSupported (uint8_t vhtSupported)
{
  m_vhtSupported = vhtSupported;
}

uint8_t
VhtOperation::GetInformationFieldSize () const
{
  //we should not be here if VHT is not supported
  NS_ASSERT (m_vhtSupported > 0);
  return 5;
}

void
VhtOperation::SetChannelWidth (uint8_t channelWidth)
{
  m_channelWidth = channelWidth;
}

void
VhtOperation::SetChannelCenterFrequencySegment0 (uint8_t channelCenterFrequencySegment0)
{
  m_channelCenterFrequencySegment0 = channelCenterFrequencySegment0;
}

void
VhtOperation::SetChannelCenterFrequencySegment1 (uint8_t channelCenterFrequencySegment1)
{
  m_channelCenterFrequencySegment1 = channelCenterFrequencySegment1;
}

void
VhtOperation::SetMaxVhtMcsPerNss (uint8_t nss, uint8_t maxVhtMcs)
{
  NS_ASSERT ((maxVhtMcs == 0 || (maxVhtMcs >= 7 && maxVhtMcs <= 9)) && (nss >= 1 && nss <= 8));
  if (maxVhtMcs != 0)
    {
      m_basicVhtMcsAndNssSet |= (((maxVhtMcs - 7) & 0x03) << ((nss - 1) * 2));
    }
  else
    {
      m_basicVhtMcsAndNssSet |= (3 << ((nss - 1) * 2));
    }
}

void
VhtOperation::SetBasicVhtMcsAndNssSet (uint16_t basicVhtMcsAndNssSet)
{
  m_basicVhtMcsAndNssSet = basicVhtMcsAndNssSet;
}

uint8_t
VhtOperation::GetChannelWidth (void) const
{
  return m_channelWidth;
}

uint8_t
VhtOperation::GetChannelCenterFrequencySegment0 (void) const
{
  return m_channelCenterFrequencySegment0;
}

uint8_t
VhtOperation::GetChannelCenterFrequencySegment1 (void) const
{
  return m_channelCenterFrequencySegment1;
}

uint16_t
VhtOperation::GetBasicVhtMcsAndNssSet (void) const
{
  return m_basicVhtMcsAndNssSet;
}

Buffer::Iterator
VhtOperation::Serialize (Buffer::Iterator i) const
{
  if (m_vhtSupported < 1)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
VhtOperation::GetSerializedSize () const
{
  if (m_vhtSupported < 1)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

void
VhtOperation::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_vhtSupported == 1)
    {
      //write the corresponding value for each bit
      start.WriteU8 (GetChannelWidth ());
      start.WriteU8 (GetChannelCenterFrequencySegment0 ());
      start.WriteU8 (GetChannelCenterFrequencySegment1 ());
      start.WriteU16 (GetBasicVhtMcsAndNssSet ());
    }
}

uint8_t
VhtOperation::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  uint8_t channelWidth = i.ReadU8 ();
  uint8_t channelCenterFrequencySegment0 = i.ReadU8 ();
  uint8_t channelCenterFrequencySegment1 = i.ReadU8 ();
  uint16_t basicVhtMcsAndNssSet = i.ReadU16 ();
  SetChannelWidth (channelWidth);
  SetChannelCenterFrequencySegment0 (channelCenterFrequencySegment0);
  SetChannelCenterFrequencySegment1 (channelCenterFrequencySegment1);
  SetBasicVhtMcsAndNssSet (basicVhtMcsAndNssSet);
  return length;
}

std::ostream &
operator << (std::ostream &os, const VhtOperation &VhtOperation)
{
  os << +VhtOperation.GetChannelWidth () << "|"
     << +VhtOperation.GetChannelCenterFrequencySegment0 () << "|"
     << +VhtOperation.GetChannelCenterFrequencySegment1 () << "|"
     << VhtOperation.GetBasicVhtMcsAndNssSet ();
  return os;
}

} //namespace ns3
