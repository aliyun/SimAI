/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
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
 * Authors: Pasquale Imputato <p.imputato@gmail.com>
 *          Stefano Avallone <stefano.avallone@unina.it>
*/

#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/queue.h"
#include "fq-codel-queue-disc.h"
#include "codel-queue-disc.h"
#include "ns3/net-device-queue-interface.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FqCoDelQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (FqCoDelFlow);

TypeId FqCoDelFlow::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FqCoDelFlow")
    .SetParent<QueueDiscClass> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<FqCoDelFlow> ()
  ;
  return tid;
}

FqCoDelFlow::FqCoDelFlow ()
  : m_deficit (0),
    m_status (INACTIVE),
    m_index (0)
{
  NS_LOG_FUNCTION (this);
}

FqCoDelFlow::~FqCoDelFlow ()
{
  NS_LOG_FUNCTION (this);
}

void
FqCoDelFlow::SetDeficit (uint32_t deficit)
{
  NS_LOG_FUNCTION (this << deficit);
  m_deficit = deficit;
}

int32_t
FqCoDelFlow::GetDeficit (void) const
{
  NS_LOG_FUNCTION (this);
  return m_deficit;
}

void
FqCoDelFlow::IncreaseDeficit (int32_t deficit)
{
  NS_LOG_FUNCTION (this << deficit);
  m_deficit += deficit;
}

void
FqCoDelFlow::SetStatus (FlowStatus status)
{
  NS_LOG_FUNCTION (this);
  m_status = status;
}

FqCoDelFlow::FlowStatus
FqCoDelFlow::GetStatus (void) const
{
  NS_LOG_FUNCTION (this);
  return m_status;
}

void
FqCoDelFlow::SetIndex (uint32_t index)
{
  NS_LOG_FUNCTION (this);
  m_index = index;
}

uint32_t
FqCoDelFlow::GetIndex (void) const
{
  return m_index;
}


NS_OBJECT_ENSURE_REGISTERED (FqCoDelQueueDisc);

TypeId FqCoDelQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FqCoDelQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<FqCoDelQueueDisc> ()
    .AddAttribute ("UseEcn",
                   "True to use ECN (packets are marked instead of being dropped)",
                   BooleanValue (true),
                   MakeBooleanAccessor (&FqCoDelQueueDisc::m_useEcn),
                   MakeBooleanChecker ())
    .AddAttribute ("Interval",
                   "The CoDel algorithm interval for each FQCoDel queue",
                   StringValue ("100ms"),
                   MakeStringAccessor (&FqCoDelQueueDisc::m_interval),
                   MakeStringChecker ())
    .AddAttribute ("Target",
                   "The CoDel algorithm target queue delay for each FQCoDel queue",
                   StringValue ("5ms"),
                   MakeStringAccessor (&FqCoDelQueueDisc::m_target),
                   MakeStringChecker ())
    .AddAttribute ("MaxSize",
                   "The maximum number of packets accepted by this queue disc",
                   QueueSizeValue (QueueSize ("10240p")),
                   MakeQueueSizeAccessor (&QueueDisc::SetMaxSize,
                                          &QueueDisc::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("Flows",
                   "The number of queues into which the incoming packets are classified",
                   UintegerValue (1024),
                   MakeUintegerAccessor (&FqCoDelQueueDisc::m_flows),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("DropBatchSize",
                   "The maximum number of packets dropped from the fat flow",
                   UintegerValue (64),
                   MakeUintegerAccessor (&FqCoDelQueueDisc::m_dropBatchSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Perturbation",
                   "The salt used as an additional input to the hash function used to classify packets",
                   UintegerValue (0),
                   MakeUintegerAccessor (&FqCoDelQueueDisc::m_perturbation),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("CeThreshold",
                   "The FqCoDel CE threshold for marking packets",
                   TimeValue (Time::Max ()),
                   MakeTimeAccessor (&FqCoDelQueueDisc::m_ceThreshold),
                   MakeTimeChecker ())
    .AddAttribute ("EnableSetAssociativeHash",
                   "Enable/Disable Set Associative Hash",
                   BooleanValue (false),
                   MakeBooleanAccessor (&FqCoDelQueueDisc::m_enableSetAssociativeHash),
                   MakeBooleanChecker ())
    .AddAttribute ("SetWays",
                   "The size of a set of queues (used by set associative hash)",
                   UintegerValue (8),
                   MakeUintegerAccessor (&FqCoDelQueueDisc::m_setWays),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("UseL4s",
                   "True to use L4S (only ECT1 packets are marked at CE threshold)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&FqCoDelQueueDisc::m_useL4s),
                   MakeBooleanChecker ())
  ;
  return tid;
}

FqCoDelQueueDisc::FqCoDelQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::MULTIPLE_QUEUES, QueueSizeUnit::PACKETS),
    m_quantum (0)
{
  NS_LOG_FUNCTION (this);
}

FqCoDelQueueDisc::~FqCoDelQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

void
FqCoDelQueueDisc::SetQuantum (uint32_t quantum)
{
  NS_LOG_FUNCTION (this << quantum);
  m_quantum = quantum;
}

uint32_t
FqCoDelQueueDisc::GetQuantum (void) const
{
  return m_quantum;
}

uint32_t
FqCoDelQueueDisc::SetAssociativeHash (uint32_t flowHash)
{
  NS_LOG_FUNCTION (this << flowHash);

  uint32_t h = (flowHash % m_flows);
  uint32_t innerHash = h % m_setWays;
  uint32_t outerHash = h - innerHash;

  for (uint32_t i = outerHash; i < outerHash + m_setWays; i++)
    {
      auto it = m_flowsIndices.find (i);

      if (it == m_flowsIndices.end ()
          || (m_tags.find (i) != m_tags.end () && m_tags[i] == flowHash)
          || StaticCast<FqCoDelFlow> (GetQueueDiscClass (it->second))->GetStatus () == FqCoDelFlow::INACTIVE)
        {
          // this queue has not been created yet or is associated with this flow
          // or is inactive, hence we can use it
          m_tags[i] = flowHash;
          return i;
        }
    }

  // all the queues of the set are used. Use the first queue of the set
  m_tags[outerHash] = flowHash;
  return outerHash;
}

bool
FqCoDelQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  uint32_t flowHash, h;

  if (GetNPacketFilters () == 0)
    {
      flowHash = item->Hash (m_perturbation);
    }
  else
    {
      int32_t ret = Classify (item);

      if (ret != PacketFilter::PF_NO_MATCH)
        {
          flowHash = static_cast<uint32_t> (ret);
        }
      else
        {
          NS_LOG_ERROR ("No filter has been able to classify this packet, drop it.");
          DropBeforeEnqueue (item, UNCLASSIFIED_DROP);
          return false;
        }
    }

  if (m_enableSetAssociativeHash)
    {
      h = SetAssociativeHash (flowHash);
    }
  else
    {
      h = flowHash % m_flows;
    }

  Ptr<FqCoDelFlow> flow;
  if (m_flowsIndices.find (h) == m_flowsIndices.end ())
    {
      NS_LOG_DEBUG ("Creating a new flow queue with index " << h);
      flow = m_flowFactory.Create<FqCoDelFlow> ();
      Ptr<QueueDisc> qd = m_queueDiscFactory.Create<QueueDisc> ();
      // If CoDel, Set values of CoDelQueueDisc to match this QueueDisc
      Ptr<CoDelQueueDisc> codel = qd->GetObject<CoDelQueueDisc> ();
      if (codel)
        {
          codel->SetAttribute ("UseEcn", BooleanValue (m_useEcn));
          codel->SetAttribute ("CeThreshold", TimeValue (m_ceThreshold));
          codel->SetAttribute ("UseL4s", BooleanValue (m_useL4s));
        }
      qd->Initialize ();
      flow->SetQueueDisc (qd);
      flow->SetIndex (h);
      AddQueueDiscClass (flow);

      m_flowsIndices[h] = GetNQueueDiscClasses () - 1;
    }
  else
    {
      flow = StaticCast<FqCoDelFlow> (GetQueueDiscClass (m_flowsIndices[h]));
    }

  if (flow->GetStatus () == FqCoDelFlow::INACTIVE)
    {
      flow->SetStatus (FqCoDelFlow::NEW_FLOW);
      flow->SetDeficit (m_quantum);
      m_newFlows.push_back (flow);
    }

  flow->GetQueueDisc ()->Enqueue (item);

  NS_LOG_DEBUG ("Packet enqueued into flow " << h << "; flow index " << m_flowsIndices[h]);

  if (GetCurrentSize () > GetMaxSize ())
    {
      NS_LOG_DEBUG ("Overload; enter FqCodelDrop ()");
      FqCoDelDrop ();
    }

  return true;
}

Ptr<QueueDiscItem>
FqCoDelQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<FqCoDelFlow> flow;
  Ptr<QueueDiscItem> item;

  do
    {
      bool found = false;

      while (!found && !m_newFlows.empty ())
        {
          flow = m_newFlows.front ();

          if (flow->GetDeficit () <= 0)
            {
              NS_LOG_DEBUG ("Increase deficit for new flow index " << flow->GetIndex ());
              flow->IncreaseDeficit (m_quantum);
              flow->SetStatus (FqCoDelFlow::OLD_FLOW);
              m_oldFlows.splice (m_oldFlows.end (), m_newFlows, m_newFlows.begin ());
            }
          else
            {
              NS_LOG_DEBUG ("Found a new flow " << flow->GetIndex () << " with positive deficit");
              found = true;
            }
        }

      while (!found && !m_oldFlows.empty ())
        {
          flow = m_oldFlows.front ();

          if (flow->GetDeficit () <= 0)
            {
              NS_LOG_DEBUG ("Increase deficit for old flow index " << flow->GetIndex ());
              flow->IncreaseDeficit (m_quantum);
              m_oldFlows.splice (m_oldFlows.end (), m_oldFlows, m_oldFlows.begin ());
            }
          else
            {
              NS_LOG_DEBUG ("Found an old flow " << flow->GetIndex () << " with positive deficit");
              found = true;
            }
        }

      if (!found)
        {
          NS_LOG_DEBUG ("No flow found to dequeue a packet");
          return 0;
        }

      item = flow->GetQueueDisc ()->Dequeue ();

      if (!item)
        {
          NS_LOG_DEBUG ("Could not get a packet from the selected flow queue");
          if (!m_newFlows.empty ())
            {
              flow->SetStatus (FqCoDelFlow::OLD_FLOW);
              m_oldFlows.push_back (flow);
              m_newFlows.pop_front ();
            }
          else
            {
              flow->SetStatus (FqCoDelFlow::INACTIVE);
              m_oldFlows.pop_front ();
            }
        }
      else
        {
          NS_LOG_DEBUG ("Dequeued packet " << item->GetPacket ());
        }
    } while (item == 0);

  flow->IncreaseDeficit (item->GetSize () * -1);

  return item;
}

bool
FqCoDelQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNQueueDiscClasses () > 0)
    {
      NS_LOG_ERROR ("FqCoDelQueueDisc cannot have classes");
      return false;
    }

  if (GetNInternalQueues () > 0)
    {
      NS_LOG_ERROR ("FqCoDelQueueDisc cannot have internal queues");
      return false;
    }

  // we are at initialization time. If the user has not set a quantum value,
  // set the quantum to the MTU of the device (if any)
  if (!m_quantum)
    {
      Ptr<NetDeviceQueueInterface> ndqi = GetNetDeviceQueueInterface ();
      Ptr<NetDevice> dev;
      // if the NetDeviceQueueInterface object is aggregated to a
      // NetDevice, get the MTU of such NetDevice
      if (ndqi && (dev = ndqi->GetObject<NetDevice> ()))
        {
          m_quantum = dev->GetMtu ();
          NS_LOG_DEBUG ("Setting the quantum to the MTU of the device: " << m_quantum);
        }

      if (!m_quantum)
        {
          NS_LOG_ERROR ("The quantum parameter cannot be null");
          return false;
        }
    }

  if (m_enableSetAssociativeHash && (m_flows % m_setWays != 0))
    {
      NS_LOG_ERROR ("The number of queues must be an integer multiple of the size "
                    "of the set of queues used by set associative hash");
      return false;
    }

  if (m_useL4s)
    {
      NS_ABORT_MSG_IF (m_ceThreshold == Time::Max(), "CE threshold not set");
      if (m_useEcn == false)
        {
          NS_LOG_WARN ("Enabling ECN as L4S mode is enabled");
        }
    }
  return true;
}

void
FqCoDelQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);

  m_flowFactory.SetTypeId ("ns3::FqCoDelFlow");

  m_queueDiscFactory.SetTypeId ("ns3::CoDelQueueDisc");
  m_queueDiscFactory.Set ("MaxSize", QueueSizeValue (GetMaxSize ()));
  m_queueDiscFactory.Set ("Interval", StringValue (m_interval));
  m_queueDiscFactory.Set ("Target", StringValue (m_target));
}

uint32_t
FqCoDelQueueDisc::FqCoDelDrop (void)
{
  NS_LOG_FUNCTION (this);

  uint32_t maxBacklog = 0, index = 0;
  Ptr<QueueDisc> qd;

  /* Queue is full! Find the fat flow and drop packet(s) from it */
  for (uint32_t i = 0; i < GetNQueueDiscClasses (); i++)
    {
      qd = GetQueueDiscClass (i)->GetQueueDisc ();
      uint32_t bytes = qd->GetNBytes ();
      if (bytes > maxBacklog)
        {
          maxBacklog = bytes;
          index = i;
        }
    }

  /* Our goal is to drop half of this fat flow backlog */
  uint32_t len = 0, count = 0, threshold = maxBacklog >> 1;
  qd = GetQueueDiscClass (index)->GetQueueDisc ();
  Ptr<QueueDiscItem> item;

  do
    {
      NS_LOG_DEBUG ("Drop packet (overflow); count: " << count << " len: " << len << " threshold: " << threshold);
      item = qd->GetInternalQueue (0)->Dequeue ();
      DropAfterDequeue (item, OVERLIMIT_DROP);
      len += item->GetSize ();
    } while (++count < m_dropBatchSize && len < threshold);

  return index;
}

} // namespace ns3

