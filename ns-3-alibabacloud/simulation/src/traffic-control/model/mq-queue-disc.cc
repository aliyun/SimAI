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
 *          Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "mq-queue-disc.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MqQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (MqQueueDisc);

TypeId MqQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MqQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<MqQueueDisc> ()
  ;
  return tid;
}

MqQueueDisc::MqQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::NO_LIMITS)
{
  NS_LOG_FUNCTION (this);
}

MqQueueDisc::~MqQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

MqQueueDisc::WakeMode
MqQueueDisc::GetWakeMode (void) const
{
  return WAKE_CHILD;
}

bool
MqQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_FATAL_ERROR ("MqQueueDisc: DoEnqueue should never be called");
}

Ptr<QueueDiscItem>
MqQueueDisc::DoDequeue (void)
{
  NS_FATAL_ERROR ("MqQueueDisc: DoDequeue should never be called");
}

Ptr<const QueueDiscItem>
MqQueueDisc::DoPeek (void)
{
  NS_FATAL_ERROR ("MqQueueDisc: DoPeek should never be called");
}

bool
MqQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);

  if (GetNPacketFilters () > 0)
    {
      NS_LOG_ERROR ("MqQueueDisc cannot have packet filters");
      return false;
    }

  if (GetNInternalQueues () > 0)
    {
      NS_LOG_ERROR ("MqQueueDisc cannot have internal queues");
      return false;
    }

  return true;
}

void
MqQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);
}

} // namespace ns3
