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

#include "ie-dot11s-perr.h"
#include "ns3/address-utils.h"
#include "ns3/packet.h"
namespace ns3 {
namespace dot11s {
IePerr::IePerr ()
{
}
IePerr::~IePerr ()
{
}
WifiInformationElementId
IePerr::ElementId () const
{
  return IE_PERR;
}
void
IePerr::Print (std::ostream &os) const
{
  os << "PERR=(Number of failed destinations=" << m_addressUnits.size ();
  for (unsigned int j = 0; j < m_addressUnits.size (); j++)
    {
      os << "(Failed destination address=" << m_addressUnits[j].destination << ", sequence number = "
         << m_addressUnits[j].seqnum << ")";
    }
  os << ")";
}
uint8_t
IePerr::GetNumOfDest () const
{
  return m_addressUnits.size ();
}
void
IePerr::SerializeInformationField (Buffer::Iterator i) const
{
  i.WriteU8 (0);// TTL
  i.WriteU8 (m_addressUnits.size ()); // number of Destinations
  for (unsigned int j = 0; j < m_addressUnits.size (); j++)
    {
      i.WriteU8 (0); // not used // Bit 6: AE (Address Extension) subfield (1 = destination external address is present, 0 = otherwise).
      WriteTo (i, m_addressUnits[j].destination);
      i.WriteHtolsbU32 (m_addressUnits[j].seqnum);
      i.WriteU8 (0);
      i.WriteU8 (0);
    }
}
uint8_t
IePerr::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  i.Next (1); //TTL //Mode flags is not used now
  uint8_t numOfDest = i.ReadU8 ();
  NS_ABORT_UNLESS ((2 + 13 * numOfDest ) == length);

  for (unsigned int j = 0; j < numOfDest; j++)
    {
      i.Next (1); // flags is not used now
      HwmpProtocol::FailedDestination unit;
      ReadFrom (i, unit.destination);
      unit.seqnum = i.ReadLsbtohU32 ();
      m_addressUnits.push_back (unit);
      i.Next (2); // Reason
    }
  return i.GetDistanceFrom (start);
}

uint8_t
IePerr::GetInformationFieldSize () const
{
  uint8_t retval = 1 //TTL   //ModeFlags
    + 1   //NumOfDests
    + 1  * m_addressUnits.size () //ModeFlags
    + (6 + 4) * m_addressUnits.size ()
    + 2* m_addressUnits.size (); // Reason Code
  return retval;
}

void
IePerr::AddAddressUnit (HwmpProtocol::FailedDestination unit)
{
  for (unsigned int i = 0; i < m_addressUnits.size (); i++)
    {
      if (m_addressUnits[i].destination == unit.destination)
        {
          return;
        }
    }
 if ((m_addressUnits.size () + 1) * 13 + 2 > 255)
    {
      return;
    }
  m_addressUnits.push_back (unit);
}
bool
IePerr::IsFull () const
{
  // -fstrict-overflow sensitive, see bug 1868
  return (GetInformationFieldSize ()
          > 255
          -   2 /* ID + LENGTH*/
          -  13// 10 /* Size of Mac48Address + uint32_t (one unit)*/
          );
}
std::vector<HwmpProtocol::FailedDestination>
IePerr::GetAddressUnitVector () const
{
  return m_addressUnits;
}
void
IePerr::DeleteAddressUnit (Mac48Address address)
{
  for (std::vector<HwmpProtocol::FailedDestination>::iterator i = m_addressUnits.begin (); i
       != m_addressUnits.end (); i++)
    {
      if (i->destination == address)
        {
          m_addressUnits.erase (i);
          break;
        }
    }
}
void
IePerr::ResetPerr ()
{
  m_addressUnits.clear ();
}
bool
operator== (const IePerr & a, const IePerr & b)
{
  if (a.m_addressUnits.size () != b.m_addressUnits.size ())
    {
      return false;
    }
  for (unsigned int i = 0; i < a.m_addressUnits.size (); i++)
    {
      if (a.m_addressUnits[i].destination != b.m_addressUnits[i].destination)
        {
          return false;
        }
      if (a.m_addressUnits[i].seqnum != b.m_addressUnits[i].seqnum)
        {
          return false;
        }
    }
  return true;
}
std::ostream &
operator << (std::ostream & os, const IePerr & a)
{
  a.Print (os);
  return os;
}
} // namespace dot11s
} // namespace ns3


