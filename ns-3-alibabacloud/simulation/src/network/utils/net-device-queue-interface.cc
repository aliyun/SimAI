/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stefano.avallone@.unina.it>
 */

#include "ns3/abort.h"
#include "ns3/queue-limits.h"
#include "ns3/net-device-queue-interface.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"
#include "ns3/queue-item.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NetDeviceQueueInterface");

TypeId
NetDeviceQueue::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NetDeviceQueue")
    .SetParent<Object> ()
    .SetGroupName("Network")
    .AddConstructor<NetDeviceQueue> ()
  ;
  return tid;
}

NetDeviceQueue::NetDeviceQueue ()
  : m_stoppedByDevice (false),
    m_stoppedByQueueLimits (false),
    NS_LOG_TEMPLATE_DEFINE ("NetDeviceQueueInterface")
{
  NS_LOG_FUNCTION (this);
}

NetDeviceQueue::~NetDeviceQueue ()
{
  NS_LOG_FUNCTION (this);

  m_queueLimits = 0;
  m_wakeCallback.Nullify ();
  m_device = 0;
}

bool
NetDeviceQueue::IsStopped (void) const
{
  NS_LOG_FUNCTION (this);
  return m_stoppedByDevice || m_stoppedByQueueLimits;
}

void
NetDeviceQueue::Start (void)
{
  NS_LOG_FUNCTION (this);
  m_stoppedByDevice = false;
}

void
NetDeviceQueue::Stop (void)
{
  NS_LOG_FUNCTION (this);
  m_stoppedByDevice = true;
}

void
NetDeviceQueue::Wake (void)
{
  NS_LOG_FUNCTION (this);

  bool wasStoppedByDevice = m_stoppedByDevice;
  m_stoppedByDevice = false;

  // Request the queue disc to dequeue a packet
  if (wasStoppedByDevice && !m_wakeCallback.IsNull ())
    {
      Simulator::ScheduleNow (&NetDeviceQueue::m_wakeCallback, this);
    }
}

void
NetDeviceQueue::NotifyAggregatedObject (Ptr<NetDeviceQueueInterface> ndqi)
{
  NS_LOG_FUNCTION (this << ndqi);

  m_device = ndqi->GetObject<NetDevice> ();
  NS_ABORT_MSG_IF (!m_device, "No NetDevice object was aggregated to the NetDeviceQueueInterface");
}

void
NetDeviceQueue::SetWakeCallback (WakeCallback cb)
{
  m_wakeCallback = cb;
}

void
NetDeviceQueue::NotifyQueuedBytes (uint32_t bytes)
{
  NS_LOG_FUNCTION (this << bytes);
  if (!m_queueLimits)
    {
      return;
    }
  m_queueLimits->Queued (bytes);
  if (m_queueLimits->Available () >= 0)
    {
      return;
    }
  m_stoppedByQueueLimits = true;
}

void
NetDeviceQueue::NotifyTransmittedBytes (uint32_t bytes)
{
  NS_LOG_FUNCTION (this << bytes);
  if ((!m_queueLimits) || (!bytes))
    {
      return;
    }
  m_queueLimits->Completed (bytes);
  if (m_queueLimits->Available () < 0)
    {
      return;
    }
  bool wasStoppedByQueueLimits = m_stoppedByQueueLimits;
  m_stoppedByQueueLimits = false;
  // Request the queue disc to dequeue a packet
  if (wasStoppedByQueueLimits && !m_wakeCallback.IsNull ())
    {
      Simulator::ScheduleNow (&NetDeviceQueue::m_wakeCallback, this);
    }
}

void
NetDeviceQueue::ResetQueueLimits ()
{
  NS_LOG_FUNCTION (this);
  if (!m_queueLimits)
    {
      return;
    }
  m_queueLimits->Reset ();
}

void
NetDeviceQueue::SetQueueLimits (Ptr<QueueLimits> ql)
{
  NS_LOG_FUNCTION (this << ql);
  m_queueLimits = ql;
}

Ptr<QueueLimits>
NetDeviceQueue::GetQueueLimits ()
{
  NS_LOG_FUNCTION (this);
  return m_queueLimits;
}


NS_OBJECT_ENSURE_REGISTERED (NetDeviceQueueInterface);

TypeId
NetDeviceQueueInterface::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NetDeviceQueueInterface")
    .SetParent<Object> ()
    .SetGroupName("Network")
    .AddConstructor<NetDeviceQueueInterface> ()
    .AddAttribute ("TxQueuesType",
                   "The type of transmission queues to be used",
                   TypeId::ATTR_CONSTRUCT,
                   TypeIdValue (NetDeviceQueue::GetTypeId ()),
                   MakeTypeIdAccessor (&NetDeviceQueueInterface::SetTxQueuesType),
                   MakeTypeIdChecker ())
    .AddAttribute ("NTxQueues", "The number of device transmission queues",
                   TypeId::ATTR_GET | TypeId::ATTR_CONSTRUCT,
                   UintegerValue (1),
                   MakeUintegerAccessor (&NetDeviceQueueInterface::SetNTxQueues,
                                         &NetDeviceQueueInterface::GetNTxQueues),
                   MakeUintegerChecker<uint16_t> (1, 65535))
  ;
  return tid;
}

NetDeviceQueueInterface::NetDeviceQueueInterface ()
{
  NS_LOG_FUNCTION (this);

  // the default select queue callback returns 0
  m_selectQueueCallback = [] (Ptr<QueueItem> item) { return 0; };
}

NetDeviceQueueInterface::~NetDeviceQueueInterface ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<NetDeviceQueue>
NetDeviceQueueInterface::GetTxQueue (std::size_t i) const
{
  NS_ASSERT (i < m_txQueuesVector.size ());
  return m_txQueuesVector[i];
}

std::size_t
NetDeviceQueueInterface::GetNTxQueues (void) const
{
  return m_txQueuesVector.size ();
}

void
NetDeviceQueueInterface::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  m_txQueuesVector.clear ();
  Object::DoDispose ();
}

void
NetDeviceQueueInterface::NotifyNewAggregate (void)
{
  NS_LOG_FUNCTION (this);

  // Notify the NetDeviceQueue objects that an object was aggregated
  for (auto& tx : m_txQueuesVector)
    {
      tx->NotifyAggregatedObject (this);
    }
  Object::NotifyNewAggregate ();
}

void
NetDeviceQueueInterface::SetTxQueuesType (TypeId type)
{
  NS_LOG_FUNCTION (this << type);

  NS_ABORT_MSG_IF (!m_txQueuesVector.empty (), "Cannot call SetTxQueuesType after creating device queues");

  m_txQueues = ObjectFactory ();
  m_txQueues.SetTypeId (type);
}

void
NetDeviceQueueInterface::SetNTxQueues (std::size_t numTxQueues)
{
  NS_LOG_FUNCTION (this << numTxQueues);
  NS_ASSERT (numTxQueues > 0);

  NS_ABORT_MSG_IF (!m_txQueuesVector.empty (), "Cannot call SetNTxQueues after creating device queues");

  // create the netdevice queues
  for (std::size_t i = 0; i < numTxQueues; i++)
    {
      m_txQueuesVector.push_back (m_txQueues.Create ()->GetObject<NetDeviceQueue> ());
    }
}

void
NetDeviceQueueInterface::SetSelectQueueCallback (SelectQueueCallback cb)
{
  m_selectQueueCallback = cb;
}

NetDeviceQueueInterface::SelectQueueCallback
NetDeviceQueueInterface::GetSelectQueueCallback (void) const
{
  return m_selectQueueCallback;
}

} // namespace ns3
