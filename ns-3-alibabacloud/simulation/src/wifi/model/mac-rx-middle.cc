/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/log.h"
#include "ns3/sequence-number.h"
#include "ns3/packet.h"
#include "mac-rx-middle.h"
#include "wifi-mac-queue-item.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MacRxMiddle");

/**
 * A class to keep track of the packet originator status.
 * It recomposes the packet from multiple fragments.
 */
class OriginatorRxStatus
{
private:
  /**
   * typedef for a list of fragments (i.e. incomplete Packet).
   */
  typedef std::list<Ptr<const Packet> > Fragments;
  /**
   * typedef for a const iterator for Fragments
   */
  typedef std::list<Ptr<const Packet> >::const_iterator FragmentsCI;

  bool m_defragmenting; ///< flag to indicate whether we are defragmenting
  uint16_t m_lastSequenceControl; ///< last sequence control
  Fragments m_fragments; ///< fragments


public:
  OriginatorRxStatus ()
  {
    /* this is a magic value necessary. */
    m_lastSequenceControl = 0xffff;
    m_defragmenting = false;
  }
  ~OriginatorRxStatus ()
  {
    m_fragments.clear ();
  }
  /**
   * Check if we are de-fragmenting packets.
   *
   * \return true if we are de-fragmenting packets,
   *         false otherwise
   */
  bool IsDeFragmenting (void) const
  {
    return m_defragmenting;
  }
  /**
   * We have received a first fragmented packet.
   * We start the deframentation by saving the first fragment.
   *
   * \param packet the first fragmented packet
   */
  void AccumulateFirstFragment (Ptr<const Packet> packet)
  {
    NS_ASSERT (!m_defragmenting);
    m_defragmenting = true;
    m_fragments.push_back (packet);
  }
  /**
   * We have received a last fragment of the fragmented packets
   * (indicated by the no more fragment field).
   * We re-construct the packet from the fragments we saved
   * and return the full packet.
   *
   * \param packet the last fragment
   *
   * \return the fully reconstructed packet
   */
  Ptr<Packet> AccumulateLastFragment (Ptr<const Packet> packet)
  {
    NS_ASSERT (m_defragmenting);
    m_fragments.push_back (packet);
    m_defragmenting = false;
    Ptr<Packet> full = Create<Packet> ();
    for (FragmentsCI i = m_fragments.begin (); i != m_fragments.end (); i++)
      {
        full->AddAtEnd (*i);
      }
    m_fragments.erase (m_fragments.begin (), m_fragments.end ());
    return full;
  }
  /**
   * We received a fragmented packet (not first and not last).
   * We simply save it into our internal list.
   *
   * \param packet the received fragment
   */
  void AccumulateFragment (Ptr<const Packet> packet)
  {
    NS_ASSERT (m_defragmenting);
    m_fragments.push_back (packet);
  }
  /**
   * Check if the sequence control (i.e. fragment number) is
   * in order.
   *
   * \param sequenceControl the raw sequence control
   *
   * \return true if the sequence control is in order,
   *         false otherwise
   */
  bool IsNextFragment (uint16_t sequenceControl) const
  {
    if ((sequenceControl >> 4) == (m_lastSequenceControl >> 4)
        && (sequenceControl & 0x0f) == ((m_lastSequenceControl & 0x0f) + 1))
      {
        return true;
      }
    else
      {
        return false;
      }
  }
  /**
   * Return the last sequence control we received.
   *
   * \return the last sequence control
   */
  uint16_t GetLastSequenceControl (void) const
  {
    return m_lastSequenceControl;
  }
  /**
   * Set the last sequence control we received.
   *
   * \param sequenceControl the last sequence control we received
   */
  void SetSequenceControl (uint16_t sequenceControl)
  {
    m_lastSequenceControl = sequenceControl;
  }
};


MacRxMiddle::MacRxMiddle ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

MacRxMiddle::~MacRxMiddle ()
{
  NS_LOG_FUNCTION_NOARGS ();
  for (OriginatorsI i = m_originatorStatus.begin ();
       i != m_originatorStatus.end (); i++)
    {
      delete (*i).second;
    }
  m_originatorStatus.erase (m_originatorStatus.begin (),
                            m_originatorStatus.end ());
  for (QosOriginatorsI i = m_qosOriginatorStatus.begin ();
       i != m_qosOriginatorStatus.end (); i++)
    {
      delete (*i).second;
    }
  m_qosOriginatorStatus.erase (m_qosOriginatorStatus.begin (),
                               m_qosOriginatorStatus.end ());
}

void
MacRxMiddle::SetForwardCallback (ForwardUpCallback callback)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_callback = callback;
}

OriginatorRxStatus *
MacRxMiddle::Lookup (const WifiMacHeader *hdr)
{
  NS_LOG_FUNCTION (hdr);
  OriginatorRxStatus *originator;
  Mac48Address source = hdr->GetAddr2 ();
  if (hdr->IsQosData ()
      && !hdr->GetAddr2 ().IsGroup ())
    {
      /* only for QoS data non-broadcast frames */
      originator = m_qosOriginatorStatus[std::make_pair (source, hdr->GetQosTid ())];
      if (originator == 0)
        {
          originator = new OriginatorRxStatus ();
          m_qosOriginatorStatus[std::make_pair (source, hdr->GetQosTid ())] = originator;
        }
    }
  else
    {
      /* - management frames
       * - QoS data broadcast frames
       * - non-QoS data frames
       * see section 7.1.3.4.1
       */
      originator = m_originatorStatus[source];
      if (originator == 0)
        {
          originator = new OriginatorRxStatus ();
          m_originatorStatus[source] = originator;
        }
    }
  return originator;
}

bool
MacRxMiddle::IsDuplicate (const WifiMacHeader* hdr,
                          OriginatorRxStatus *originator) const
{
  NS_LOG_FUNCTION (hdr << originator);
  if (hdr->IsRetry ()
      && originator->GetLastSequenceControl () == hdr->GetSequenceControl ())
    {
      return true;
    }
  return false;
}

Ptr<const Packet>
MacRxMiddle::HandleFragments (Ptr<const Packet> packet, const WifiMacHeader *hdr,
                              OriginatorRxStatus *originator)
{
  NS_LOG_FUNCTION (packet << hdr << originator);
  if (originator->IsDeFragmenting ())
    {
      if (hdr->IsMoreFragments ())
        {
          if (originator->IsNextFragment (hdr->GetSequenceControl ()))
            {
              NS_LOG_DEBUG ("accumulate fragment seq=" << hdr->GetSequenceNumber () <<
                            ", frag=" << +hdr->GetFragmentNumber () <<
                            ", size=" << packet->GetSize ());
              originator->AccumulateFragment (packet);
              originator->SetSequenceControl (hdr->GetSequenceControl ());
            }
          else
            {
              NS_LOG_DEBUG ("non-ordered fragment");
            }
          return 0;
        }
      else
        {
          if (originator->IsNextFragment (hdr->GetSequenceControl ()))
            {
              NS_LOG_DEBUG ("accumulate last fragment seq=" << hdr->GetSequenceNumber () <<
                            ", frag=" << +hdr->GetFragmentNumber () <<
                            ", size=" << hdr->GetSize ());
              Ptr<Packet> p = originator->AccumulateLastFragment (packet);
              originator->SetSequenceControl (hdr->GetSequenceControl ());
              return p;
            }
          else
            {
              NS_LOG_DEBUG ("non-ordered fragment");
              return 0;
            }
        }
    }
  else
    {
      if (hdr->IsMoreFragments ())
        {
          NS_LOG_DEBUG ("accumulate first fragment seq=" << hdr->GetSequenceNumber () <<
                        ", frag=" << +hdr->GetFragmentNumber () <<
                        ", size=" << packet->GetSize ());
          originator->AccumulateFirstFragment (packet);
          originator->SetSequenceControl (hdr->GetSequenceControl ());
          return 0;
        }
      else
        {
          return packet;
        }
    }
}

void
MacRxMiddle::Receive (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (*mpdu);
  const WifiMacHeader* hdr = &mpdu->GetHeader ();
  NS_ASSERT (hdr->IsData () || hdr->IsMgt ());

  OriginatorRxStatus *originator = Lookup (hdr);
  /**
   * The check below is really unneeded because it can fail in a lot of
   * normal cases. Specifically, it is possible for sequence numbers to
   * loop back to zero once they reach 0xfff0 and to go up to 0xf7f0 in
   * which case the check below will report the two sequence numbers to
   * not have the correct order relationship.
   * So, this check cannot be used to discard old duplicate frames. It is
   * thus here only for documentation purposes.
   */
  if (!(SequenceNumber16 (originator->GetLastSequenceControl ()) < SequenceNumber16 (hdr->GetSequenceControl ())))
    {
      NS_LOG_DEBUG ("Sequence numbers have looped back. last recorded=" << originator->GetLastSequenceControl () <<
                    " currently seen=" << hdr->GetSequenceControl ());
    }
  //filter duplicates.
  if (IsDuplicate (hdr, originator))
    {
      NS_LOG_DEBUG ("duplicate from=" << hdr->GetAddr2 () <<
                    ", seq=" << hdr->GetSequenceNumber () <<
                    ", frag=" << +hdr->GetFragmentNumber ());
      return;
    }
  Ptr<const Packet> aggregate = HandleFragments (mpdu->GetPacket (), hdr, originator);
  if (aggregate == 0)
    {
      return;
    }
  NS_LOG_DEBUG ("forwarding data from=" << hdr->GetAddr2 () <<
                ", seq=" << hdr->GetSequenceNumber () <<
                ", frag=" << +hdr->GetFragmentNumber ());
  if (!hdr->GetAddr1 ().IsGroup ())
    {
      originator->SetSequenceControl (hdr->GetSequenceControl ());
    }
  if (aggregate == mpdu->GetPacket ())
    {
      m_callback (mpdu);
    }
  else
    {
      // We could do this in all cases, but passing the received mpdu in case of
      // A-MSDUs saves us the time to deaggregate the A-MSDU in MSDUs (which are
      // kept separate in the received mpdu) and allows us to pass the originally
      // transmitted packets (i.e., with the same UID) to the receiver.
      m_callback (Create<WifiMacQueueItem> (aggregate, *hdr));
    }
}

} //namespace ns3
