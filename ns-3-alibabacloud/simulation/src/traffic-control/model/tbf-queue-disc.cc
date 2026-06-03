/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Kungliga Tekniska Högskolan
 *               2017 Universita' degli Studi di Napoli Federico II
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
 * TBF, The Token Bucket Filter Queueing discipline
 *
 * This implementation is based on linux kernel code by
 * Authors:     Alexey Kuznetsov, <kuznet@ms2.inr.ac.ru>
 *              Dmitry Torokhov <dtor@mail.ru> - allow attaching inner qdiscs -
 *                                               original idea by Martin Devera
 *
 * Implemented in ns-3 by: Surya Seetharaman <suryaseetharaman.9@gmail.com>
 *                         Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "ns3/enum.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"
#include "ns3/attribute.h"
#include "ns3/object-factory.h"
#include "ns3/drop-tail-queue.h"
#include "ns3/net-device-queue-interface.h"
#include "tbf-queue-disc.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TbfQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (TbfQueueDisc);

TypeId TbfQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TbfQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<TbfQueueDisc> ()
    .AddAttribute ("MaxSize",
                   "The max queue size",
                   QueueSizeValue (QueueSize ("1000p")),
                   MakeQueueSizeAccessor (&QueueDisc::SetMaxSize,
                                          &QueueDisc::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("Burst",
                   "Size of the first bucket in bytes",
                   UintegerValue (125000),
                   MakeUintegerAccessor (&TbfQueueDisc::SetBurst),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Mtu",
                   "Size of the second bucket in bytes. If null, it is initialized"
                   " to the MTU of the receiving NetDevice (if any)",
                   UintegerValue (0),
                   MakeUintegerAccessor (&TbfQueueDisc::SetMtu),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Rate",
                   "Rate at which tokens enter the first bucket in bps or Bps.",
                   DataRateValue (DataRate ("125KB/s")),
                   MakeDataRateAccessor (&TbfQueueDisc::SetRate),
                   MakeDataRateChecker ())
    .AddAttribute ("PeakRate",
                   "Rate at which tokens enter the second bucket in bps or Bps."
                   "If null, there is no second bucket",
                   DataRateValue (DataRate ("0KB/s")),
                   MakeDataRateAccessor (&TbfQueueDisc::SetPeakRate),
                   MakeDataRateChecker ())
    .AddTraceSource ("TokensInFirstBucket",
                     "Number of First Bucket Tokens in bytes",
                     MakeTraceSourceAccessor (&TbfQueueDisc::m_btokens),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("TokensInSecondBucket",
                     "Number of Second Bucket Tokens in bytes",
                     MakeTraceSourceAccessor (&TbfQueueDisc::m_ptokens),
                     "ns3::TracedValueCallback::Uint32")
  ;

  return tid;
}

TbfQueueDisc::TbfQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::SINGLE_CHILD_QUEUE_DISC)
{
  NS_LOG_FUNCTION (this);
}

TbfQueueDisc::~TbfQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

void
TbfQueueDisc::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  QueueDisc::DoDispose ();
}

void
TbfQueueDisc::SetBurst (uint32_t burst)
{
  NS_LOG_FUNCTION (this << burst);
  m_burst = burst;
}

uint32_t
TbfQueueDisc::GetBurst (void) const
{
  NS_LOG_FUNCTION (this);
  return m_burst;
}

void
TbfQueueDisc::SetMtu (uint32_t mtu)
{
  NS_LOG_FUNCTION (this << mtu);
  m_mtu = mtu;
}

uint32_t
TbfQueueDisc::GetMtu (void) const
{
  NS_LOG_FUNCTION (this);
  return m_mtu;
}

void
TbfQueueDisc::SetRate (DataRate rate)
{
  NS_LOG_FUNCTION (this << rate);
  m_rate = rate;
}

DataRate
TbfQueueDisc::GetRate (void) const
{
  NS_LOG_FUNCTION (this);
  return m_rate;
}

void
TbfQueueDisc::SetPeakRate (DataRate peakRate)
{
  NS_LOG_FUNCTION (this << peakRate);
  m_peakRate = peakRate;
}

DataRate
TbfQueueDisc::GetPeakRate (void) const
{
  NS_LOG_FUNCTION (this);
  return m_peakRate;
}

uint32_t
TbfQueueDisc::GetFirstBucketTokens (void) const
{
  NS_LOG_FUNCTION (this);
  return m_btokens;
}

uint32_t
TbfQueueDisc::GetSecondBucketTokens (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ptokens;
}

bool
TbfQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  bool retval = GetQueueDiscClass (0)->GetQueueDisc ()->Enqueue (item);

  // If Queue::Enqueue fails, QueueDisc::Drop is called by the child queue
  // disc because QueueDisc::AddQueueDiscClass sets the drop callback

  NS_LOG_LOGIC ("Current queue size: " << GetNPackets () << " packets, " << GetNBytes () << " bytes");

  return retval;
}

Ptr<QueueDiscItem>
TbfQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<const QueueDiscItem> itemPeek = GetQueueDiscClass (0)->GetQueueDisc ()->Peek ();

  if (itemPeek)
    {
      uint32_t pktSize = itemPeek->GetSize ();
      NS_LOG_LOGIC ("Next packet size " << pktSize);

      int64_t btoks = 0;
      int64_t ptoks = 0;
      Time now = Simulator::Now ();

      double delta = (now  - m_timeCheckPoint).GetSeconds ();
      NS_LOG_LOGIC ("Time Difference delta " << delta);

      if (m_peakRate > DataRate ("0bps"))
        {
          ptoks =  m_ptokens + round (delta * (m_peakRate.GetBitRate () / 8));
          if (ptoks > m_mtu)
            {
              ptoks = m_mtu;
            }
          NS_LOG_LOGIC ("Number of ptokens we can consume " << ptoks);
          NS_LOG_LOGIC ("Required to dequeue next packet " << pktSize);
          ptoks -= pktSize;
        }

      btoks = m_btokens + round (delta * (m_rate.GetBitRate () / 8));

      if (btoks > m_burst)
        {
          btoks = m_burst;
        }

      NS_LOG_LOGIC ("Number of btokens we can consume " << btoks);
      NS_LOG_LOGIC ("Required to dequeue next packet " << pktSize);
      btoks -= pktSize;

      if ((btoks|ptoks) >= 0) // else packet blocked
        {
          Ptr<QueueDiscItem> item = GetQueueDiscClass (0)->GetQueueDisc ()->Dequeue ();
          if (!item)
            {
              NS_LOG_DEBUG ("That's odd! Expecting the peeked packet, we got no packet.");
              return item;
            }

          m_timeCheckPoint = now;
          m_btokens = btoks;
          m_ptokens = ptoks;

          NS_LOG_LOGIC (m_btokens << " btokens and " << m_ptokens << " ptokens after packet dequeue");
          NS_LOG_LOGIC ("Current queue size: " << GetNPackets () << " packets, " << GetNBytes () << " bytes");

          return item;
        }

      // the watchdog timer setup.
      // A packet gets blocked if the above if() condition is not satisfied:
      // either or both btoks and ptoks are negative.  In that case, we have
      // to schedule the waking of queue when enough tokens are available.
      if (m_id.IsExpired () == true)
        {
          NS_ASSERT_MSG (m_rate.GetBitRate () > 0, "Rate must be positive");
          Time requiredDelayTime;
          if (m_peakRate.GetBitRate () == 0)
            {
              NS_ASSERT_MSG (btoks < 0, "Logic error; btoks must be < 0 here");
              requiredDelayTime = m_rate.CalculateBytesTxTime (-btoks);
            }
          else
            {
              if (btoks < 0 && ptoks >= 0)
                {
                  requiredDelayTime = m_rate.CalculateBytesTxTime (-btoks);
                }
              else if (btoks >= 0 && ptoks < 0)
                {
                  requiredDelayTime = m_peakRate.CalculateBytesTxTime (-ptoks);
                }
              else
                {
                  requiredDelayTime = std::max (m_rate.CalculateBytesTxTime (-btoks),
                                                m_peakRate.CalculateBytesTxTime (-ptoks));
                }

            }
          NS_ASSERT_MSG (requiredDelayTime.GetSeconds () >= 0, "Negative time");
          m_id = Simulator::Schedule (requiredDelayTime, &QueueDisc::Run, this);
          NS_LOG_LOGIC("Waking Event Scheduled in " << requiredDelayTime.As (Time::S));
        }
    }
  return 0;
}

bool
TbfQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNInternalQueues () > 0)
    {
      NS_LOG_ERROR ("TbfQueueDisc cannot have internal queues");
      return false;
    }

  if (GetNPacketFilters () > 0)
    {
      NS_LOG_ERROR ("TbfQueueDisc cannot have packet filters");
      return false;
    }

  if (GetNQueueDiscClasses () == 0)
    {
      // create a FIFO queue disc
      ObjectFactory factory;
      factory.SetTypeId ("ns3::FifoQueueDisc");
      Ptr<QueueDisc> qd = factory.Create<QueueDisc> ();

      if (!qd->SetMaxSize (GetMaxSize ()))
        {
          NS_LOG_ERROR ("Cannot set the max size of the child queue disc to that of TbfQueueDisc");
          return false;
        }
      qd->Initialize ();

      Ptr<QueueDiscClass> c = CreateObject<QueueDiscClass> ();
      c->SetQueueDisc (qd);
      AddQueueDiscClass (c);
    }

  if (GetNQueueDiscClasses () != 1)
    {
      NS_LOG_ERROR ("TbfQueueDisc needs 1 child queue disc");
      return false;
    }

  // This type of variable initialization would normally be done in
  // InitializeParams (), but we want to use the value to subsequently
  // check configuration of peak rate, so we move it forward here.
  if (m_mtu == 0)
    {
      Ptr<NetDeviceQueueInterface> ndqi = GetNetDeviceQueueInterface ();
      Ptr<NetDevice> dev;
      // if the NetDeviceQueueInterface object is aggregated to a
      // NetDevice, get the MTU of such NetDevice
      if (ndqi && (dev = ndqi->GetObject<NetDevice> ()))
        {
          m_mtu = dev->GetMtu ();
        }
    }

  if (m_mtu == 0 && m_peakRate > DataRate ("0bps"))
    {
      NS_LOG_ERROR ("A non-null peak rate has been set, but the mtu is null. No packet will be dequeued");
      return false;
    }

  if (m_burst <= m_mtu)
    {
      NS_LOG_WARN ("The size of the first bucket (" << m_burst << ") should be "
                    << "greater than the size of the second bucket (" << m_mtu << ").");
    }

  if (m_peakRate > DataRate ("0bps") && m_peakRate <= m_rate)
    {
      NS_LOG_WARN ("The rate for the second bucket (" << m_peakRate << ") should be "
                    << "greater than the rate for the first bucket (" << m_rate << ").");
    }

  return true;
}

void
TbfQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);
  // Token Buckets are full at the beginning.
  m_btokens = m_burst;
  m_ptokens = m_mtu;
  // Initialising other variables to 0.
  m_timeCheckPoint = Seconds (0);
  m_id = EventId ();
}

} // namespace ns3
