/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
 * Copyright (c) 2020 NITK Surathkal (adapted for COBALT)
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
 * Modified by: Bhaskar Kataria <bhaskar.k7920@gmail.com> (for COBALT)
 *          Tom Henderson <tomhend@u.washington.edu>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *          Vivek Jain <jain.vivek.anand@gmail.com>
 *          Ankit Deepak <adadeepak8@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/queue.h"
#include "fq-cobalt-queue-disc.h"
#include "cobalt-queue-disc.h"
#include "ns3/net-device-queue-interface.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FqCobaltQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (FqCobaltFlow);

TypeId FqCobaltFlow::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FqCobaltFlow")
    .SetParent<QueueDiscClass> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<FqCobaltFlow> ()
  ;
  return tid;
}

FqCobaltFlow::FqCobaltFlow ()
  : m_deficit (0),
    m_status (INACTIVE),
    m_index (0)
{
  NS_LOG_FUNCTION (this);
}

FqCobaltFlow::~FqCobaltFlow ()
{
  NS_LOG_FUNCTION (this);
}

void
FqCobaltFlow::SetDeficit (uint32_t deficit)
{
  NS_LOG_FUNCTION (this << deficit);
  m_deficit = deficit;
}

int32_t
FqCobaltFlow::GetDeficit (void) const
{
  NS_LOG_FUNCTION (this);
  return m_deficit;
}

void
FqCobaltFlow::IncreaseDeficit (int32_t deficit)
{
  NS_LOG_FUNCTION (this << deficit);
  m_deficit += deficit;
}

void
FqCobaltFlow::SetStatus (FlowStatus status)
{
  NS_LOG_FUNCTION (this);
  m_status = status;
}

FqCobaltFlow::FlowStatus
FqCobaltFlow::GetStatus (void) const
{
  NS_LOG_FUNCTION (this);
  return m_status;
}

void
FqCobaltFlow::SetIndex (uint32_t index)
{
  NS_LOG_FUNCTION (this);
  m_index = index;
}

uint32_t
FqCobaltFlow::GetIndex (void) const
{
  return m_index;
}


NS_OBJECT_ENSURE_REGISTERED (FqCobaltQueueDisc);

TypeId FqCobaltQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FqCobaltQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<FqCobaltQueueDisc> ()
    .AddAttribute ("UseEcn",
                   "True to use ECN (packets are marked instead of being dropped)",
                   BooleanValue (true),
                   MakeBooleanAccessor (&FqCobaltQueueDisc::m_useEcn),
                   MakeBooleanChecker ())
    .AddAttribute ("Interval",
                   "The CoDel algorithm interval for each FqCobalt queue",
                   StringValue ("100ms"),
                   MakeStringAccessor (&FqCobaltQueueDisc::m_interval),
                   MakeStringChecker ())
    .AddAttribute ("Target",
                   "The CoDel algorithm target queue delay for each FqCobalt queue",
                   StringValue ("5ms"),
                   MakeStringAccessor (&FqCobaltQueueDisc::m_target),
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
                   MakeUintegerAccessor (&FqCobaltQueueDisc::m_flows),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("DropBatchSize",
                   "The maximum number of packets dropped from the fat flow",
                   UintegerValue (64),
                   MakeUintegerAccessor (&FqCobaltQueueDisc::m_dropBatchSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Perturbation",
                   "The salt used as an additional input to the hash function used to classify packets",
                   UintegerValue (0),
                   MakeUintegerAccessor (&FqCobaltQueueDisc::m_perturbation),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("CeThreshold",
                   "The FqCobalt CE threshold for marking packets",
                   TimeValue (Time::Max ()),
                   MakeTimeAccessor (&FqCobaltQueueDisc::m_ceThreshold),
                   MakeTimeChecker ())
    .AddAttribute ("EnableSetAssociativeHash",
                   "Enable/Disable Set Associative Hash",
                   BooleanValue (false),
                   MakeBooleanAccessor (&FqCobaltQueueDisc::m_enableSetAssociativeHash),
                   MakeBooleanChecker ())
    .AddAttribute ("SetWays",
                   "The size of a set of queues (used by set associative hash)",
                   UintegerValue (8),
                   MakeUintegerAccessor (&FqCobaltQueueDisc::m_setWays),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("UseL4s",
                   "True to use L4S (only ECT1 packets are marked at CE threshold)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&FqCobaltQueueDisc::m_useL4s),
                   MakeBooleanChecker ())
    .AddAttribute ("Pdrop",
                   "Marking Probability",
                   DoubleValue (0),
                   MakeDoubleAccessor (&FqCobaltQueueDisc::m_Pdrop),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Increment",
                   "Pdrop increment value",
                   DoubleValue (1. / 256),
                   MakeDoubleAccessor (&FqCobaltQueueDisc::m_increment),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Decrement",
                   "Pdrop decrement Value",
                   DoubleValue (1. / 4096),
                   MakeDoubleAccessor (&FqCobaltQueueDisc::m_decrement),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("BlueThreshold",
                   "The Threshold after which Blue is enabled",
                   TimeValue (MilliSeconds (400)),
                   MakeTimeAccessor (&FqCobaltQueueDisc::m_blueThreshold),
                   MakeTimeChecker ())
  ;
  return tid;
}

FqCobaltQueueDisc::FqCobaltQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::MULTIPLE_QUEUES, QueueSizeUnit::PACKETS),
    m_quantum (0)
{
  NS_LOG_FUNCTION (this);
}

FqCobaltQueueDisc::~FqCobaltQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

void
FqCobaltQueueDisc::SetQuantum (uint32_t quantum)
{
  NS_LOG_FUNCTION (this << quantum);
  m_quantum = quantum;
}

uint32_t
FqCobaltQueueDisc::GetQuantum (void) const
{
  return m_quantum;
}

uint32_t
FqCobaltQueueDisc::SetAssociativeHash (uint32_t flowHash)
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
          || StaticCast<FqCobaltFlow> (GetQueueDiscClass (it->second))->GetStatus () == FqCobaltFlow::INACTIVE)
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
FqCobaltQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
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

  Ptr<FqCobaltFlow> flow;
  if (m_flowsIndices.find (h) == m_flowsIndices.end ())
    {
      NS_LOG_DEBUG ("Creating a new flow queue with index " << h);
      flow = m_flowFactory.Create<FqCobaltFlow> ();
      Ptr<QueueDisc> qd = m_queueDiscFactory.Create<QueueDisc> ();
      // If Cobalt, Set values of CobaltQueueDisc to match this QueueDisc
      Ptr<CobaltQueueDisc> cobalt = qd->GetObject<CobaltQueueDisc> ();
      if (cobalt)
        {
          cobalt->SetAttribute ("UseEcn", BooleanValue (m_useEcn));
          cobalt->SetAttribute ("CeThreshold", TimeValue (m_ceThreshold));
          cobalt->SetAttribute ("UseL4s", BooleanValue (m_useL4s));
          cobalt->SetAttribute ("BlueThreshold", TimeValue (m_blueThreshold));
        }
      qd->Initialize ();
      flow->SetQueueDisc (qd);
      flow->SetIndex (h);
      AddQueueDiscClass (flow);

      m_flowsIndices[h] = GetNQueueDiscClasses () - 1;
    }
  else
    {
      flow = StaticCast<FqCobaltFlow> (GetQueueDiscClass (m_flowsIndices[h]));
    }

  if (flow->GetStatus () == FqCobaltFlow::INACTIVE)
    {
      flow->SetStatus (FqCobaltFlow::NEW_FLOW);
      flow->SetDeficit (m_quantum);
      m_newFlows.push_back (flow);
    }

  flow->GetQueueDisc ()->Enqueue (item);

  NS_LOG_DEBUG ("Packet enqueued into flow " << h << "; flow index " << m_flowsIndices[h]);

  if (GetCurrentSize () > GetMaxSize ())
    {
      NS_LOG_DEBUG ("Overload; enter FqCobaltDrop ()");
      FqCobaltDrop ();
    }

  return true;
}

Ptr<QueueDiscItem>
FqCobaltQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<FqCobaltFlow> flow;
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
              flow->SetStatus (FqCobaltFlow::OLD_FLOW);
              m_oldFlows.push_back (flow);
              m_newFlows.pop_front ();
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
              m_oldFlows.push_back (flow);
              m_oldFlows.pop_front ();
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
              flow->SetStatus (FqCobaltFlow::OLD_FLOW);
              m_oldFlows.push_back (flow);
              m_newFlows.pop_front ();
            }
          else
            {
              flow->SetStatus (FqCobaltFlow::INACTIVE);
              m_oldFlows.pop_front ();
            }
        }
      else
        {
          NS_LOG_DEBUG ("Dequeued packet " << item->GetPacket ());
        }
    }
  while (item == 0);

  flow->IncreaseDeficit (item->GetSize () * -1);

  return item;
}

bool
FqCobaltQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNQueueDiscClasses () > 0)
    {
      NS_LOG_ERROR ("FqCobaltQueueDisc cannot have classes");
      return false;
    }

  if (GetNInternalQueues () > 0)
    {
      NS_LOG_ERROR ("FqCobaltQueueDisc cannot have internal queues");
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

  // If UseL4S attribute is enabled then CE threshold must be set.
  if (m_useL4s)
    {
      NS_ABORT_MSG_IF (m_ceThreshold == Time::Max (), "CE threshold not set");
      if (m_useEcn == false)
        {
          NS_LOG_WARN ("Enabling ECN as L4S mode is enabled");
        }
    }
  return true;
}

void
FqCobaltQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);

  m_flowFactory.SetTypeId ("ns3::FqCobaltFlow");

  m_queueDiscFactory.SetTypeId ("ns3::CobaltQueueDisc");
  m_queueDiscFactory.Set ("MaxSize", QueueSizeValue (GetMaxSize ()));
  m_queueDiscFactory.Set ("Interval", StringValue (m_interval));
  m_queueDiscFactory.Set ("Target", StringValue (m_target));
  m_queueDiscFactory.Set ("Pdrop", DoubleValue (m_Pdrop));
  m_queueDiscFactory.Set ("Increment", DoubleValue (m_increment));
  m_queueDiscFactory.Set ("Decrement", DoubleValue (m_decrement));
}

uint32_t
FqCobaltQueueDisc::FqCobaltDrop (void)
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
    }
  while (++count < m_dropBatchSize && len < threshold);

  return index;
}

} // namespace ns3

