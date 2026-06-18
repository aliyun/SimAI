/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "delay-jitter-estimation.h"
#include "ns3/tag.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

namespace ns3 {

/**
 * Tag to perform Delay and Jitter estimations
 *
 * The tag holds the packet's creation timestamp
 */
class DelayJitterEstimationTimestampTag : public Tag
{
public:
  DelayJitterEstimationTimestampTag ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

  /**
   * \brief Get the Transmission time stored in the tag
   * \return the transmission time
   */
  Time GetTxTime (void) const;
private:
  Time m_creationTime; //!< The time stored in the tag
};

DelayJitterEstimationTimestampTag::DelayJitterEstimationTimestampTag ()
  : m_creationTime (Simulator::Now ())
{
}

TypeId
DelayJitterEstimationTimestampTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("anon::DelayJitterEstimationTimestampTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<DelayJitterEstimationTimestampTag> ()
    .AddAttribute ("CreationTime",
                   "The time at which the timestamp was created",
                   TimeValue (Time (0)),
                   MakeTimeAccessor (&DelayJitterEstimationTimestampTag::m_creationTime),
                   MakeTimeChecker ())
  ;
  return tid;
}
TypeId
DelayJitterEstimationTimestampTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
DelayJitterEstimationTimestampTag::GetSerializedSize (void) const
{
  return 8;
}
void
DelayJitterEstimationTimestampTag::Serialize (TagBuffer i) const
{
  i.WriteU64 (m_creationTime.GetTimeStep ());
}
void
DelayJitterEstimationTimestampTag::Deserialize (TagBuffer i)
{
  m_creationTime = TimeStep (i.ReadU64 ());
}
void
DelayJitterEstimationTimestampTag::Print (std::ostream &os) const
{
  os << "CreationTime=" << m_creationTime;
}
Time
DelayJitterEstimationTimestampTag::GetTxTime (void) const
{
  return m_creationTime;
}

DelayJitterEstimation::DelayJitterEstimation ()
  : m_jitter (Time (0)),
    m_transit (Time (0))
{
}
void
DelayJitterEstimation::PrepareTx (Ptr<const Packet> packet)
{
  DelayJitterEstimationTimestampTag tag;
  packet->AddByteTag (tag);
}
void
DelayJitterEstimation::RecordRx (Ptr<const Packet> packet)
{
  DelayJitterEstimationTimestampTag tag;
  bool found;
  found = packet->FindFirstMatchingByteTag (tag);
  if (!found)
    {
      return;
    }

  // Variable names from
  // RFC 1889 Appendix A.8 ,p. 71,
  // RFC 3550 Appendix A.8, p. 94
  
  Time r_ts = tag.GetTxTime ();
  Time arrival = Simulator::Now ();
  Time transit = arrival - r_ts;
  Time delta = transit - m_transit;
  m_transit = transit;

  // floating jitter version
  //  m_jitter += (Abs (delta) - m_jitter) / 16;
  
  // int variant
  m_jitter += Abs (delta) - ( (m_jitter + TimeStep (8)) / 16 );
}

Time 
DelayJitterEstimation::GetLastDelay (void) const
{
  return m_transit;
}
uint64_t
DelayJitterEstimation::GetLastJitter (void) const
{
  // floating jitter version
  // return m_jitter.GetTimeStep ();

  // int variant
  return (m_jitter / 16).GetTimeStep ();
}

} // namespace ns3
