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

#include "ie-dot11s-beacon-timing.h"
#include "ns3/packet.h"
namespace ns3 {
namespace dot11s {
/*******************************************
 * IeBeaconTimingUnit
 *******************************************/
IeBeaconTimingUnit::IeBeaconTimingUnit () :
  m_aid (0), m_lastBeacon (0), m_beaconInterval (0)
{
}
void
IeBeaconTimingUnit::SetAid (uint8_t aid)
{
  m_aid = aid;
}
void
IeBeaconTimingUnit::SetLastBeacon (uint16_t lastBeacon)
{
  m_lastBeacon = lastBeacon;
}
void
IeBeaconTimingUnit::SetBeaconInterval (uint16_t beaconInterval)
{
  m_beaconInterval = beaconInterval;
}
uint8_t
IeBeaconTimingUnit::GetAid () const
{
  return m_aid;
}
uint16_t
IeBeaconTimingUnit::GetLastBeacon () const
{
  return m_lastBeacon;
}
uint16_t
IeBeaconTimingUnit::GetBeaconInterval () const
{
  return m_beaconInterval;
}
/*******************************************
 * IeBeaconTiming
 *******************************************/
WifiInformationElementId
IeBeaconTiming::ElementId () const
{
  return IE_BEACON_TIMING;
}
IeBeaconTiming::IeBeaconTiming () :
  m_numOfUnits (0)
{
}
IeBeaconTiming::NeighboursTimingUnitsList
IeBeaconTiming::GetNeighboursTimingElementsList ()
{
  return m_neighbours;
}
void
IeBeaconTiming::AddNeighboursTimingElementUnit (uint16_t aid, Time last_beacon, Time beacon_interval)
{
  if (m_numOfUnits == 50)
    {
      return;
    }
  //First we lookup if this element already exists
  for (NeighboursTimingUnitsList::const_iterator i = m_neighbours.begin (); i != m_neighbours.end (); i++)
    {
      if (((*i)->GetAid () == AidToU8 (aid)) && ((*i)->GetLastBeacon () == TimestampToU16 (last_beacon))
          && ((*i)->GetBeaconInterval () == BeaconIntervalToU16 (beacon_interval)))
        {
          return;
        }
    }
  Ptr<IeBeaconTimingUnit> new_element = Create<IeBeaconTimingUnit> ();
  new_element->SetAid (AidToU8 (aid));
  new_element->SetLastBeacon (TimestampToU16 (last_beacon));
  new_element->SetBeaconInterval (BeaconIntervalToU16 (beacon_interval));
  m_neighbours.push_back (new_element);
  m_numOfUnits++;
}
void
IeBeaconTiming::DelNeighboursTimingElementUnit (uint16_t aid, Time last_beacon, Time beacon_interval)
{
  for (NeighboursTimingUnitsList::iterator i = m_neighbours.begin (); i != m_neighbours.end (); i++)
    {
      if (((*i)->GetAid () == AidToU8 (aid)) && ((*i)->GetLastBeacon () == TimestampToU16 (last_beacon))
          && ((*i)->GetBeaconInterval () == BeaconIntervalToU16 (beacon_interval)))
        {
          m_neighbours.erase (i);
          m_numOfUnits--;
          break;
        }
    }
}
void
IeBeaconTiming::ClearTimingElement ()
{
  for (NeighboursTimingUnitsList::iterator j = m_neighbours.begin (); j != m_neighbours.end (); j++)
    {
      (*j) = 0;
    }
  m_neighbours.clear ();
}
uint8_t
IeBeaconTiming::GetInformationFieldSize () const
{
  return (5 * m_numOfUnits );
}
void
IeBeaconTiming::Print (std::ostream& os) const
{
  os << "BeaconTiming=(Number of units=" << (uint16_t) m_numOfUnits;
  for (NeighboursTimingUnitsList::const_iterator j = m_neighbours.begin (); j != m_neighbours.end (); j++)
    {
      os << "(AID=" << (uint16_t)(*j)->GetAid () << ", Last beacon at=" << (*j)->GetLastBeacon ()
         << ", with beacon interval=" << (*j)->GetBeaconInterval () << ")";
    }
  os << ")";
}
void
IeBeaconTiming::SerializeInformationField (Buffer::Iterator i) const
{
  for (NeighboursTimingUnitsList::const_iterator j = m_neighbours.begin (); j != m_neighbours.end (); j++)
    {
      i.WriteU8 ((*j)->GetAid ());
      i.WriteHtolsbU16 ((*j)->GetLastBeacon ());
      i.WriteHtolsbU16 ((*j)->GetBeaconInterval ());
    }
}
uint8_t
IeBeaconTiming::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  m_numOfUnits = length / 5;
  for (int j = 0; j < m_numOfUnits; j++)
    {
      Ptr<IeBeaconTimingUnit> new_element = Create<IeBeaconTimingUnit> ();
      new_element->SetAid (i.ReadU8 ());
      new_element->SetLastBeacon (i.ReadLsbtohU16 ());
      new_element->SetBeaconInterval (i.ReadLsbtohU16 ());
      m_neighbours.push_back (new_element);
    }
  return i.GetDistanceFrom (start);
}
;
uint16_t
IeBeaconTiming::TimestampToU16 (Time t)
{
  return ((uint16_t)((t.GetMicroSeconds () >> 8) & 0xffff));
}
;

uint16_t
IeBeaconTiming::BeaconIntervalToU16 (Time t)
{
  return ((uint16_t)(t.GetMicroSeconds () >> 10) & 0xffff);
}
uint8_t
IeBeaconTiming::AidToU8 (uint16_t x)
{
  return (uint8_t)(x & 0xff);
}
bool
operator== (const IeBeaconTimingUnit & a, const IeBeaconTimingUnit & b)
{
  return ((a.GetAid () == b.GetAid ()) && (a.GetLastBeacon () == b.GetLastBeacon ())
          && (a.GetBeaconInterval () == b.GetBeaconInterval ()));
}
bool
IeBeaconTiming::operator== (WifiInformationElement const & a) const
{
  try {
      IeBeaconTiming const & aa = dynamic_cast<IeBeaconTiming const &>(a);

      if (m_numOfUnits != aa.m_numOfUnits)
        {
          return false;
        }
      for (unsigned int i = 0; i < m_neighbours.size (); i++)
        {
          if (!(*PeekPointer (m_neighbours[i]) == *PeekPointer (aa.m_neighbours[i])))
            {
              return false;
            }
        }
      return true;
    }
  catch (std::bad_cast&)
    {
      return false;
    }
}
std::ostream &
operator << (std::ostream &os, const IeBeaconTiming &a)
{
  a.Print (os);
  return os;
}
} // namespace dot11s
} // namespace ns3

