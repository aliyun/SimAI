/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Rajib Bhattacharjea<raj.b@gatech.edu>
//


// This is a port of Data PDU Headers from:
// Georgia Tech Network Simulator
// George F. Riley.  Georgia Tech, Spring 2002

#include <iostream>
#include <algorithm>

#include <cstring>

#include "ns3/fatal-error.h"
#include "ns3/log.h"

#include "pending-data.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("PendingData");

PendingData::PendingData () : size (0), data (0),
                              msgSize (0), responseSize (0)
{
  NS_LOG_FUNCTION (this);
}

PendingData::PendingData (uint32_t s, uint8_t* d, uint32_t msg, uint32_t resp)
  : size (s), data (0), msgSize (msg), responseSize (resp)
{
  NS_LOG_FUNCTION (this << s);
  if (d)
    {
      data.push_back (Create<Packet> (d, size));
    }
}

PendingData::PendingData(const std::string& s) 
  : size (s.length () + 1), data (0),
    msgSize (0), responseSize (0)
{
  NS_LOG_FUNCTION (this << s.length () + 1);
  data.push_back (Create<Packet> ((uint8_t*)s.c_str (), size));
}

PendingData::PendingData(const PendingData& c)
  : size (c.Size ()), data (c.data),
    msgSize (c.msgSize), responseSize (c.responseSize)
{
  NS_LOG_FUNCTION (this << c.Size ());
}

PendingData::~PendingData()
{
  NS_LOG_FUNCTION (this);
}

PendingData* PendingData::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new PendingData (*this);
};

PendingData* PendingData::CopyS (uint32_t s)
{ // Copy, but with new size (assumes no associated data);
  NS_LOG_FUNCTION (this << s);
  return new PendingData (s, 0, msgSize, responseSize);
}

PendingData* PendingData::CopySD (uint32_t s, uint8_t* d)
{ // Copy, but with new size (assumes no associated data);
  NS_LOG_FUNCTION (this << s);
  return new PendingData (s, d, msgSize, responseSize);
}

void PendingData::Clear ()
{ // Remove all pending data
  NS_LOG_FUNCTION (this);
  data.clear ();
  size = 0;
}

void PendingData::Add (uint32_t s, const uint8_t* d)
{
  NS_LOG_FUNCTION (this << s);
  data.push_back (Create<Packet> (d,s));
  size += s;
}

void PendingData::Add (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this);
  data.push_back (p);
  size += p->GetSize ();
}

uint32_t PendingData::SizeFromSeq (const SequenceNumber32& seqFront, const SequenceNumber32& seqOffset)
{
  NS_LOG_FUNCTION (this << seqFront << seqOffset);
  uint32_t o1 = OffsetFromSeq (seqFront, seqOffset); // Offset to start of unused data
  return SizeFromOffset (o1);      // Amount of data after offset
}

uint32_t PendingData::SizeFromOffset (uint32_t offset)
{ // Find out how much data is available from offset
  NS_LOG_FUNCTION (this << offset);
  /// \todo should this return zero, or error out?
  if (offset > size) return 0;     // No data at requested offset
  return size - offset;            // Available data after offset
}

uint32_t PendingData::OffsetFromSeq (const SequenceNumber32& seqFront, const SequenceNumber32& seqOffset)
{ // f is the first sequence number in this data, o is offset sequence
  NS_LOG_FUNCTION (this << seqFront << seqOffset);
  if (seqOffset < seqFront) 
    {
      return 0; // HuH?  Shouldn't happen
    }
  return seqOffset - seqFront;
}

Ptr<Packet> PendingData::CopyFromOffset (uint32_t s, uint32_t o)
{ // Make a copy of data from starting position "o" for "s" bytes
  // Return NULL if results in zero length data
  NS_LOG_FUNCTION (this << s << o);
  uint32_t s1 = std::min (s, SizeFromOffset (o)); // Insure not beyond end of data
  if (s1 == 0)
    {
      return Create<Packet> ();   // No data requested
    }
  if (data.size () != 0)
    { // Actual data exists, make copy and return it
      uint32_t count = 0;
      std::vector<Ptr<Packet> >::size_type begin = 0;
      bool beginFound = false;
      std::vector<Ptr<Packet> >::size_type end = 0;
      Ptr<Packet> outPacket;
      Ptr<Packet> endFragment;
      for (std::vector<Ptr<Packet> >::size_type i=0; i<data.size (); ++i)
        {
          count+=data[i]->GetSize ();
          if (!beginFound)
            {
              if (count > o)
                {
                  if (count >= o + s1) //then just copy within this packet
                    {
                      Ptr<Packet> toFragment = data[i];
                      uint32_t packetStart = count - toFragment->GetSize ();
                      uint32_t packetOffset = o - packetStart;
                      outPacket = toFragment->CreateFragment (packetOffset, s1);
                      return outPacket;
                    }
                  begin = i;
                  beginFound = true;
                  Ptr<Packet> toFragment = data[begin];
                  uint32_t packetStart = count - toFragment->GetSize ();
                  uint32_t packetOffset = o - packetStart;
                  uint32_t fragmentLength = count - o;
                  outPacket = toFragment->CreateFragment (packetOffset, fragmentLength);
                }
            }
          else
            {
              if (count >= o + s1)
                {
                  end = i;
                  Ptr<Packet> toFragment = data[end];
                  uint32_t packetStart = count - toFragment->GetSize ();
                  uint32_t fragmentLength = o + s1 - packetStart;
                  endFragment = toFragment->CreateFragment (0, fragmentLength);
                  break;
                }
            }
        }
      for (std::vector<Ptr<Packet> >::size_type i=begin+1; i<end; ++i)
        {
          outPacket->AddAtEnd (data[i]);
        }
      if (endFragment)
        {
          outPacket->AddAtEnd (endFragment);
        }
      NS_ASSERT (outPacket->GetSize () == s1);
      return outPacket;
    }
  else
    { // No actual data, just return dummy-data packet of correct size
      return Create<Packet> (s1);
    }
}

Ptr<Packet> PendingData::CopyFromSeq (uint32_t s, const SequenceNumber32& f, const SequenceNumber32& o)
{
  NS_LOG_FUNCTION (this << s << f << o);
  return CopyFromOffset (s, OffsetFromSeq (f,o));
}

uint32_t
PendingData::RemoveToSeq (const SequenceNumber32& seqFront, const SequenceNumber32& seqOffset)
{
  NS_LOG_FUNCTION (this << seqFront << seqOffset);
  uint32_t count = OffsetFromSeq (seqFront, seqOffset);
  NS_ASSERT_MSG (count <= size, "Trying to remove more data than in the buffer"); 
  if (count == size)
    {
      Clear ();
      return size;
    }
  // Remove whole packets, if possible, from the front of the data
  // Do not perform buffer manipulations within packet; if a whole packet
  // cannot be removed, leave it alone
  std::vector<Ptr<Packet> >::iterator endI = data.begin ();
  uint32_t current = 0;
  // Any packet whose data has been completely acked can be removed
  for (std::vector<Ptr<Packet> >::iterator dataI = data.begin (); dataI < data.end (); dataI++)
    {
      if (current + (*dataI)->GetSize () > count)
        {
          break;
        }
      current += (*dataI)->GetSize ();
      ++endI;
    }
  data.erase (data.begin (), endI);
  size -= current;
  return current;
}

} //namespace ns3
