/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA, UDcast
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *         Mohamed Amine Ismail <Amine.Ismail@sophia.inria.fr>
 *                              <Amine.Ismail@UDcast.com>
 */

#include "wimax-mac-queue.h"
#include "ns3/packet.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WimaxMacQueue");

NS_OBJECT_ENSURE_REGISTERED (WimaxMacQueue);

WimaxMacQueue::QueueElement::QueueElement (void)
  : m_packet (Create<Packet> ()),
    m_hdrType (MacHeaderType ()),
    m_hdr (
      GenericMacHeader ()),
    m_timeStamp (Seconds (0)),
    m_fragmentation (false),
    m_fragmentNumber (0),
    m_fragmentOffset (0)
{
}

WimaxMacQueue::QueueElement::QueueElement (Ptr<Packet> packet,
                                           const MacHeaderType &hdrType,
                                           const GenericMacHeader &hdr, Time timeStamp)
  : m_packet (packet),
    m_hdrType (hdrType),
    m_hdr (hdr),
    m_timeStamp (timeStamp),
    m_fragmentation (false),
    m_fragmentNumber (0),
    m_fragmentOffset (0)
{
}

uint32_t
WimaxMacQueue::QueueElement::GetSize (void) const
{
  uint32_t size = m_packet->GetSize () + m_hdrType.GetSerializedSize ();

  /*check because may be it is a bandwidth request packet (in which case a Bandwidth Request Header
   has already been added to the packet) in which case Generic MAC Header will not be added to it.
   this will only happen in the case of SS as only SS sends the bandwidth request packet. */
  if (m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      size += m_hdr.GetSerializedSize ();
    }

  return size;
}

TypeId
WimaxMacQueue::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WimaxMacQueue")
    .SetParent<Object> ()
    .SetGroupName("Wimax")
    .AddAttribute (
      "MaxSize",
      "Maximum size",
      UintegerValue (1024),
      MakeUintegerAccessor (&WimaxMacQueue::GetMaxSize,
                            &WimaxMacQueue::SetMaxSize),
      MakeUintegerChecker<uint32_t> ())
    .AddTraceSource ("Enqueue",
                     "Enqueue trace",
                     MakeTraceSourceAccessor (&WimaxMacQueue::m_traceEnqueue),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("Dequeue",
                     "Dequeue trace",
                     MakeTraceSourceAccessor (&WimaxMacQueue::m_traceDequeue),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("Drop",
                     "Drop trace",
                     MakeTraceSourceAccessor (&WimaxMacQueue::m_traceDrop),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

WimaxMacQueue::WimaxMacQueue (void)
  : m_maxSize (0),
    m_bytes (0),
    m_nrDataPackets (0),
    m_nrRequestPackets (0)
{
}

WimaxMacQueue::WimaxMacQueue (uint32_t maxSize)
  : m_maxSize (maxSize),
    m_bytes (0),
    m_nrDataPackets (0),
    m_nrRequestPackets (0)
{
}

WimaxMacQueue::~WimaxMacQueue (void)
{
}

void
WimaxMacQueue::SetMaxSize (uint32_t maxSize)
{
  m_maxSize = maxSize;
}

uint32_t
WimaxMacQueue::GetMaxSize (void) const
{
  return m_maxSize;
}

bool
WimaxMacQueue::Enqueue (Ptr<Packet> packet, const MacHeaderType &hdrType,
                        const GenericMacHeader &hdr)
{

  if (m_queue.size () == m_maxSize)
    {

      m_traceDrop (packet);
      return false;
    }

  m_traceEnqueue (packet);
  QueueElement element (packet, hdrType, hdr, Simulator::Now ());
  m_queue.push_back (element);

  if (hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      m_nrDataPackets++;
    }
  else
    {
      m_nrRequestPackets++;
    }

  m_bytes += element.GetSize ();
  return true;
}

Ptr<Packet>
WimaxMacQueue::Dequeue (MacHeaderType::HeaderType packetType)
{
  if (!IsEmpty ())
    {
      QueueElement element = Front (packetType);
      Pop (packetType);

      if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
        {
          NS_LOG_INFO ("Enqueued Packet IS A data packet");
          NS_ASSERT_MSG (m_nrDataPackets >= 1,
                         "Can not enqueue more packets: no space left in the queue");
          m_nrDataPackets--;
        }
      else
        {
          NS_LOG_INFO ("Enqueued Packet IS A Request BW packet");
          NS_ASSERT_MSG (m_nrRequestPackets >= 1,
                         "Can not enqueue more packets: no space left in the queue");
          m_nrRequestPackets--;
        }

      Ptr<Packet> packet = element.m_packet;

      if (!element.m_fragmentation)
        {
          NS_LOG_INFO ("FRAG_DEBUG: Enqueued Packet IS NOT a fragment" << std::endl);
          /*check because may be it is a bandwidth request packet (in which case a Bandwidth Request Header
            has already been added to the packet) in which case Generic MAC Header will not be added to it.
            this will only happen in the case of SS as only SS sends the bandwidth request packet. */
          m_bytes -= element.GetSize ();
          if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
            {
              packet->AddHeader (element.m_hdr);
            }
          packet->AddHeader (element.m_hdrType);

          m_traceDequeue (packet);
          return packet;
        }
      else
        {
          /*
           The enqueued packet is a fragment (the latest fragment)
           We must modify type field of the m_hdr and add a fragmentation Subhdr
           */
          NS_LOG_INFO ("\t Enqueued Packet IS a fragment, add subhdr" << std::endl);

          // Create a fragment
          uint32_t fragmentOffset = element.m_fragmentOffset;
          uint32_t fragmentSize = element.m_packet->GetSize () - fragmentOffset;

          NS_LOG_INFO ("\t Create a fragment"
                       "\n\t\t fragmentOffset=" << fragmentOffset <<
                       "\n\t\t packetSize=" << element.m_packet->GetSize () <<
                       "\n\t\t fragmentSize=" << fragmentSize << std::endl);

          Ptr<Packet> fragment = packet->CreateFragment (fragmentOffset,fragmentSize);

          FragmentationSubheader fragmentSubhdr;
          NS_LOG_INFO ("\t Latest Fragment" << std::endl);
          fragmentSubhdr.SetFc (2);   // This is the latest fragment
          fragmentSubhdr.SetFsn (element.m_fragmentNumber);

          NS_LOG_INFO ("\t FragmentSize=" << fragment->GetSize () << std::endl);
          fragment->AddHeader (fragmentSubhdr);

          /*check because may be it is a bandwidth request packet (in which case a Bandwidth Request Header
          has already been added to the packet) in which case Generic MAC Header will not be added to it.
          this will only happen in the case of SS as only SS sends the bandwidth request packet. */
          if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
            {
              uint8_t tmpType = element.m_hdr.GetType ();
              tmpType |= 4;
              element.m_hdr.SetType (tmpType);

              uint32_t length = fragmentSize + element.m_hdr.GetSerializedSize ()
                + fragmentSubhdr.GetSerializedSize ();
              element.m_hdr.SetLen ((uint16_t)length);

              fragment->AddHeader (element.m_hdr);
            }
          fragment->AddHeader (element.m_hdrType);
          m_bytes -= fragmentSize;

          m_traceDequeue (fragment);
          return fragment;
        }
    }
  return 0;
}

Ptr<Packet>
WimaxMacQueue::Dequeue (MacHeaderType::HeaderType packetType, uint32_t availableByte)
{
  if (!IsEmpty ())
    {
      NS_LOG_INFO ("FRAG_DEBUG: Dequeue function" << std::endl);
      QueueElement element = Front (packetType);

      uint32_t headerSize = 2 + element.m_hdr.GetSerializedSize () +
        element.m_hdrType.GetSerializedSize ();

      // Create a fragment
      uint32_t maxFragmentSize = availableByte - headerSize;
      uint32_t fragmentOffset = element.m_fragmentOffset; // It is the latest byte sent.

      Ptr<Packet> packet = element.m_packet->Copy ();
      NS_LOG_INFO ("\t Create a fragment"
                   "\n\t\t availableByte=" << availableByte <<
                   "\n\t\t headerSize=" << headerSize <<
                   "\n\t\t maxFragmentSize=" << maxFragmentSize << ""
                   "\n\t\t fragmentOffset=" << fragmentOffset <<
                   "\n\t\t payloadSize=" << packet->GetSize ()
                                            << std::endl);
      Ptr<Packet> fragment = packet->CreateFragment (fragmentOffset,
                                                     maxFragmentSize);
      m_bytes -= maxFragmentSize;

      FragmentationSubheader fragmentSubhdr;
      if (!element.m_fragmentation)
        {
          NS_LOG_INFO ("\t First Fragment" << std::endl);
          SetFragmentation (packetType);
          fragmentSubhdr.SetFc (1);
        }
      else
        {
          NS_LOG_INFO ("\t Middle Fragment" << std::endl);
          fragmentSubhdr.SetFc (3);
        }
      fragmentSubhdr.SetFsn (element.m_fragmentNumber);
      NS_LOG_INFO ("\t FragmentSize=" << fragment->GetSize () << std::endl);
      fragment->AddHeader (fragmentSubhdr);

      SetFragmentNumber (packetType);
      SetFragmentOffset (packetType, maxFragmentSize);

      /*check because may be it is a bandwidth request packet (in which case a Bandwidth Request Header
        has already been added to the packet) in which case Generic MAC Header will not be added to it.
            this will only happen in the case of SS as only SS sends the bandwidth request packet. */
      if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
        {
          uint8_t tmpType = element.m_hdr.GetType ();
          tmpType |= 4;
          element.m_hdr.SetType (tmpType);

          uint32_t length = maxFragmentSize + element.m_hdr.GetSerializedSize ()
            + fragmentSubhdr.GetSerializedSize ();
          element.m_hdr.SetLen ((uint16_t)length);

          fragment->AddHeader (element.m_hdr);
        }
      fragment->AddHeader (element.m_hdrType);

      m_traceDequeue (fragment);
      return fragment;
    }
  return 0;
}

Ptr<Packet>
WimaxMacQueue::Peek (GenericMacHeader &hdr) const
{
  if (!IsEmpty ())
    {
      QueueElement element = m_queue.front ();
      hdr = element.m_hdr;
      Ptr<Packet> packet = element.m_packet->Copy ();

      // this function must not be used by SS as it may be then a bandwidth request header
      packet->AddHeader (element.m_hdr);
      return packet;
    }

  return 0;
}

Ptr<Packet>
WimaxMacQueue::Peek (GenericMacHeader &hdr, Time &timeStamp) const
{
  if (!IsEmpty ())
    {
      QueueElement element = m_queue.front ();
      hdr = element.m_hdr;
      timeStamp = element.m_timeStamp;
      Ptr<Packet> packet = element.m_packet->Copy ();

      // this function must not be used for by SS as it may be then a bandwidth request header
      packet->AddHeader (element.m_hdr);
      return packet;
    }

  return 0;
}

Ptr<Packet>
WimaxMacQueue::Peek (MacHeaderType::HeaderType packetType) const
{
  if (!IsEmpty ())
    {
      QueueElement element = Front (packetType);
      Ptr<Packet> packet = element.m_packet->Copy ();

      /*check because may be it is a bandwidth request packet (in which case a Bandwidth Request Header
       has already been added to the packet) in which case Generic MAC Header will not be added to it.
       this will only happen in the case of SS as only SS sends the bandwidth request packet. */
      if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
        {
          packet->AddHeader (element.m_hdr);
        }
      return packet;
    }

  return 0;
}

Ptr<Packet>
WimaxMacQueue::Peek (MacHeaderType::HeaderType packetType,
                     Time &timeStamp) const
{
  if (!IsEmpty ())
    {
      QueueElement element = Front (packetType);
      timeStamp = element.m_timeStamp;
      Ptr<Packet> packet = element.m_packet->Copy ();

      /*check because may be it is a bandwidth request packet (in which case a Bandwidth Request Header
       has already been added to the packet) in which case Generic MAC Header will not be added to it.
       this will only happen in the case of SS as only SS sends the bandwidth request packet. */
      if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
        {
          packet->AddHeader (element.m_hdr);
        }
      return packet;
    }

  return 0;
}

uint32_t
WimaxMacQueue::GetSize (void) const
{
  return m_queue.size ();
}

uint32_t
WimaxMacQueue::GetNBytes (void) const
{
  return m_bytes;
}

uint32_t WimaxMacQueue::GetQueueLengthWithMACOverhead (void)
{
  uint32_t queueSize = GetNBytes ();
  // Add MAC Overhead
  queueSize += GetSize () * 6;
  MacHeaderType::HeaderType packetType = MacHeaderType::HEADER_TYPE_GENERIC;
  if (CheckForFragmentation (packetType))
    {
      queueSize += 2;
    }
  return queueSize;
}

WimaxMacQueue::QueueElement
WimaxMacQueue::Front (MacHeaderType::HeaderType packetType) const
{
  QueueElement element;

  for (std::deque<QueueElement>::const_iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      element = *iter;
      if (element.m_hdrType.GetType () == packetType)
        {
          break;
        }
    }

  return element;
}

void
WimaxMacQueue::Pop (MacHeaderType::HeaderType packetType)
{
  QueueElement element;

  for (std::deque<QueueElement>::iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      element = *iter;
      if (element.m_hdrType.GetType () == packetType)
        {
          m_queue.erase (iter);
          break;
        }
    }
}

bool
WimaxMacQueue::IsEmpty (void) const
{
  return m_queue.empty ();
}

bool
WimaxMacQueue::IsEmpty (MacHeaderType::HeaderType packetType) const
{
  if (packetType == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      return m_nrDataPackets == 0;
    }
  else
    {
      return m_nrRequestPackets == 0;
    }

  return true;
}

const WimaxMacQueue::PacketQueue &
WimaxMacQueue::GetPacketQueue (void) const
{
  return m_queue;
}

bool
WimaxMacQueue::CheckForFragmentation (MacHeaderType::HeaderType packetType)
{
  QueueElement element;
  for (std::deque<QueueElement>::const_iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      element = *iter;
      if (element.m_hdrType.GetType () == packetType)
        {
          break;
        }
    }

  if (element.m_fragmentation)
    {
      NS_LOG_INFO ("FRAG_DEBUG: CheckForFragmentation"
                   "\n\t\t m_fragmentation is true " << std::endl);
    }

  return element.m_fragmentation;
}

uint32_t
WimaxMacQueue::GetFirstPacketHdrSize (MacHeaderType::HeaderType packetType)
{
  QueueElement element;
  for (std::deque<QueueElement>::const_iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      element = *iter;
      if (element.m_hdrType.GetType () == packetType)
        {
          break;
        }
    }

  NS_LOG_INFO ("\t\t GetFirstPacketHdrSize ()");

  uint32_t hdrSize = 0;
  if (element.m_hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      hdrSize += element.m_hdr.GetSerializedSize ();
      NS_LOG_INFO ("\t\t\t m_hdr.GetSerializedSize=" <<
                   element.m_hdr.GetSerializedSize ());
    }

  hdrSize += element.m_hdrType.GetSerializedSize ();
  NS_LOG_INFO ("\t\t\t m_hdrType.GetSerializedSize=" <<
               element.m_hdrType.GetSerializedSize ());

  if (CheckForFragmentation (packetType))
    {
      NS_LOG_INFO ("\t\t\t fragSubhdrSize=2");
      hdrSize += 2;
    }

  NS_LOG_INFO ("\t\t hdrSize=" << hdrSize);

  return hdrSize;
}

uint32_t
WimaxMacQueue::GetFirstPacketPayloadSize (MacHeaderType::HeaderType packetType)
{
  QueueElement element;
  for (std::deque<QueueElement>::const_iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      element = *iter;
      if (element.m_hdrType.GetType () == packetType)
        {
          break;
        }
    }

  NS_LOG_INFO ("\t\t GetFirstPacketPayloadSize ()");

  if (CheckForFragmentation (packetType))
    {
      NS_LOG_INFO ("\t\t\t fullPayloadSize=" << element.m_packet->GetSize ()
                                             << "\n\t\t\t fragmentOffset=" << element.m_fragmentOffset
                                             << "\n\t\t\t (fragment)payloadSize=" <<
                   element.m_packet->GetSize () - element.m_fragmentOffset);

      return element.m_packet->GetSize () - element.m_fragmentOffset;
    }
  NS_LOG_INFO ("\t\t payloadSize=" <<
               element.m_packet->GetSize ());

  return element.m_packet->GetSize ();
}

uint32_t
WimaxMacQueue::GetFirstPacketRequiredByte (MacHeaderType::HeaderType packetType)
{
  NS_LOG_INFO ("\t GetFirstPacketRequiredByte ()");

  uint32_t requiredByte = GetFirstPacketPayloadSize (packetType) +
    GetFirstPacketHdrSize (packetType);

  NS_LOG_INFO ("\t Required Bytes = " << requiredByte << std::endl);

  return requiredByte;
}

void
WimaxMacQueue::SetFragmentation (MacHeaderType::HeaderType packetType)
{
  for (std::deque<QueueElement>::iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      if (iter->m_hdrType.GetType () == packetType)
        {
          iter->SetFragmentation ();
          break;
        }
    }
}

void
WimaxMacQueue::SetFragmentNumber (MacHeaderType::HeaderType packetType)
{
  for (std::deque<QueueElement>::iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      if (iter->m_hdrType.GetType () == packetType)
        {
          iter->SetFragmentNumber ();
          break;
        }
    }
}

void
WimaxMacQueue::SetFragmentOffset (MacHeaderType::HeaderType packetType, uint32_t offset)
{
  for (std::deque<QueueElement>::iterator iter = m_queue.begin (); iter
       != m_queue.end (); ++iter)
    {
      if (iter->m_hdrType.GetType () == packetType)
        {
          iter->SetFragmentOffset (offset);
          break;
        }
    }
}

void
WimaxMacQueue::QueueElement::SetFragmentation ()
{
  m_fragmentation = true;
}

void
WimaxMacQueue::QueueElement::SetFragmentNumber ()
{
  m_fragmentNumber++;
}

void
WimaxMacQueue::QueueElement::SetFragmentOffset (uint32_t offset)
{
  m_fragmentOffset += offset;
}
} // namespace ns3
