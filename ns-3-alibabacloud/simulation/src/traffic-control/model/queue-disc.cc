/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007, 2014 University of Washington
 *               2015 Universita' degli Studi di Napoli Federico II
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
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/uinteger.h"
#include "ns3/pointer.h"
#include "ns3/object-vector.h"
#include "ns3/packet.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "queue-disc.h"
#include "ns3/net-device-queue-interface.h"
#include "ns3/queue.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("QueueDisc");


NS_OBJECT_ENSURE_REGISTERED (QueueDiscClass);

TypeId QueueDiscClass::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::QueueDiscClass")
    .SetParent<Object> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<QueueDiscClass> ()
    .AddAttribute ("QueueDisc", "The queue disc attached to the class",
                   PointerValue (),
                   MakePointerAccessor (&QueueDiscClass::m_queueDisc),
                   MakePointerChecker<QueueDisc> ())
  ;
  return tid;
}

QueueDiscClass::QueueDiscClass ()
{
  NS_LOG_FUNCTION (this);
}

QueueDiscClass::~QueueDiscClass ()
{
  NS_LOG_FUNCTION (this);
}

void
QueueDiscClass::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_queueDisc = 0;
  Object::DoDispose ();
}

Ptr<QueueDisc>
QueueDiscClass::GetQueueDisc (void) const
{
  NS_LOG_FUNCTION (this);
  return m_queueDisc;
}

void
QueueDiscClass::SetQueueDisc (Ptr<QueueDisc> qd)
{
  NS_LOG_FUNCTION (this);
  NS_ABORT_MSG_IF (m_queueDisc, "Cannot set the queue disc on a class already having an attached queue disc");
  m_queueDisc = qd;
}

QueueDisc::Stats::Stats ()
  : nTotalReceivedPackets (0),
    nTotalReceivedBytes (0),
    nTotalSentPackets (0),
    nTotalSentBytes (0),
    nTotalEnqueuedPackets (0),
    nTotalEnqueuedBytes (0),
    nTotalDequeuedPackets (0),
    nTotalDequeuedBytes (0),
    nTotalDroppedPackets (0),
    nTotalDroppedPacketsBeforeEnqueue (0),
    nTotalDroppedPacketsAfterDequeue (0),
    nTotalDroppedBytes (0),
    nTotalDroppedBytesBeforeEnqueue (0),
    nTotalDroppedBytesAfterDequeue (0),
    nTotalRequeuedPackets (0),
    nTotalRequeuedBytes (0),
    nTotalMarkedPackets (0),
    nTotalMarkedBytes (0)
{
}

uint32_t
QueueDisc::Stats::GetNDroppedPackets (std::string reason) const
{
  uint32_t count = 0;
  auto it = nDroppedPacketsBeforeEnqueue.find (reason);

  if (it != nDroppedPacketsBeforeEnqueue.end ())
    {
      count += it->second;
    }

  it = nDroppedPacketsAfterDequeue.find (reason);

  if (it != nDroppedPacketsAfterDequeue.end ())
    {
      count += it->second;
    }

  return count;
}

uint64_t
QueueDisc::Stats::GetNDroppedBytes (std::string reason) const
{
  uint64_t count = 0;
  auto it = nDroppedBytesBeforeEnqueue.find (reason);

  if (it != nDroppedBytesBeforeEnqueue.end ())
    {
      count += it->second;
    }

  it = nDroppedBytesAfterDequeue.find (reason);

  if (it != nDroppedBytesAfterDequeue.end ())
    {
      count += it->second;
    }

  return count;
}

uint32_t
QueueDisc::Stats::GetNMarkedPackets (std::string reason) const
{
  auto it = nMarkedPackets.find (reason);

  if (it != nMarkedPackets.end ())
    {
      return it->second;
    }

  return 0;
}

uint64_t
QueueDisc::Stats::GetNMarkedBytes (std::string reason) const
{
  auto it = nMarkedBytes.find (reason);

  if (it != nMarkedBytes.end ())
    {
      return it->second;
    }

  return 0;
}

void
QueueDisc::Stats::Print (std::ostream &os) const
{
  std::map<std::string, uint32_t>::const_iterator itp;
  std::map<std::string, uint64_t>::const_iterator itb;

  os << std::endl << "Packets/Bytes received: "
                  << nTotalReceivedPackets << " / "
                  << nTotalReceivedBytes
     << std::endl << "Packets/Bytes enqueued: "
                  << nTotalEnqueuedPackets << " / "
                  << nTotalEnqueuedBytes
     << std::endl << "Packets/Bytes dequeued: "
                  << nTotalDequeuedPackets << " / "
                  << nTotalDequeuedBytes
     << std::endl << "Packets/Bytes requeued: "
                  << nTotalRequeuedPackets << " / "
                  << nTotalRequeuedBytes
     << std::endl << "Packets/Bytes dropped: "
                  << nTotalDroppedPackets << " / "
                  << nTotalDroppedBytes
     << std::endl << "Packets/Bytes dropped before enqueue: "
                  << nTotalDroppedPacketsBeforeEnqueue << " / "
                  << nTotalDroppedBytesBeforeEnqueue;

  itp = nDroppedPacketsBeforeEnqueue.begin ();
  itb = nDroppedBytesBeforeEnqueue.begin ();

  while (itp != nDroppedPacketsBeforeEnqueue.end () &&
         itb != nDroppedBytesBeforeEnqueue.end ())
    {
      NS_ASSERT (itp->first.compare (itb->first) == 0);
      os << std::endl << "  " << itp->first << ": "
         << itp->second << " / " << itb->second;
      itp++;
      itb++;
    }

  os << std::endl << "Packets/Bytes dropped after dequeue: "
                  << nTotalDroppedPacketsAfterDequeue << " / "
                  << nTotalDroppedBytesAfterDequeue;

  itp = nDroppedPacketsAfterDequeue.begin ();
  itb = nDroppedBytesAfterDequeue.begin ();

  while (itp != nDroppedPacketsAfterDequeue.end () &&
         itb != nDroppedBytesAfterDequeue.end ())
    {
      NS_ASSERT (itp->first.compare (itb->first) == 0);
      os << std::endl << "  " << itp->first << ": "
         << itp->second << " / " << itb->second;
      itp++;
      itb++;
    }

  os << std::endl << "Packets/Bytes sent: "
                  << nTotalSentPackets << " / "
                  << nTotalSentBytes
     << std::endl << "Packets/Bytes marked: "
                  << nTotalMarkedPackets << " / "
                  << nTotalMarkedBytes;

  itp = nMarkedPackets.begin ();
  itb = nMarkedBytes.begin ();

  while (itp != nMarkedPackets.end () &&
         itb != nMarkedBytes.end ())
    {
      NS_ASSERT (itp->first.compare (itb->first) == 0);
      os << std::endl << "  " << itp->first << ": "
         << itp->second << " / " << itb->second;
      itp++;
      itb++;
    }

  os << std::endl;
}

std::ostream & operator << (std::ostream &os, const QueueDisc::Stats &stats)
{
  stats.Print (os);
  return os;
}

NS_OBJECT_ENSURE_REGISTERED (QueueDisc);

TypeId QueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::QueueDisc")
    .SetParent<Object> ()
    .SetGroupName ("TrafficControl")
    .AddAttribute ("Quota", "The maximum number of packets dequeued in a qdisc run",
                   UintegerValue (DEFAULT_QUOTA),
                   MakeUintegerAccessor (&QueueDisc::SetQuota,
                                         &QueueDisc::GetQuota),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("InternalQueueList", "The list of internal queues.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&QueueDisc::m_queues),
                   MakeObjectVectorChecker<InternalQueue> ())
    .AddAttribute ("PacketFilterList", "The list of packet filters.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&QueueDisc::m_filters),
                   MakeObjectVectorChecker<PacketFilter> ())
    .AddAttribute ("QueueDiscClassList", "The list of queue disc classes.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&QueueDisc::m_classes),
                   MakeObjectVectorChecker<QueueDiscClass> ())
    .AddTraceSource ("Enqueue", "Enqueue a packet in the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceEnqueue),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("Dequeue", "Dequeue a packet from the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceDequeue),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("Requeue", "Requeue a packet in the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceRequeue),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("Drop", "Drop a packet stored in the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceDrop),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("DropBeforeEnqueue", "Drop a packet before enqueue",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceDropBeforeEnqueue),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("DropAfterDequeue", "Drop a packet after dequeue",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceDropAfterDequeue),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("Mark", "Mark a packet stored in the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_traceMark),
                     "ns3::QueueDiscItem::TracedCallback")
    .AddTraceSource ("PacketsInQueue",
                     "Number of packets currently stored in the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_nPackets),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("BytesInQueue",
                     "Number of bytes currently stored in the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_nBytes),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("SojournTime",
                     "Sojourn time of the last packet dequeued from the queue disc",
                     MakeTraceSourceAccessor (&QueueDisc::m_sojourn),
                     "ns3::Time::TracedCallback")
  ;
  return tid;
}

QueueDisc::QueueDisc (QueueDiscSizePolicy policy)
  :  m_nPackets (0),
     m_nBytes (0),
     m_maxSize (QueueSize ("1p")),         // to avoid that setting the mode at construction time is ignored
     m_running (false),
     m_peeked (false),
     m_sizePolicy (policy),
     m_prohibitChangeMode (false)
{
  NS_LOG_FUNCTION (this << (uint16_t)policy);

  // These lambdas call the DropBeforeEnqueue or DropAfterDequeue methods of this
  // QueueDisc object. Given that a callback to the operator() of these lambdas
  // is connected to the DropBeforeEnqueue and DropAfterDequeue traces of the
  // internal queues, the INTERNAL_QUEUE_DROP constant is passed as the reason
  // why the packet is dropped.
  m_internalQueueDbeFunctor = [this] (Ptr<const QueueDiscItem> item)
    {
      return DropBeforeEnqueue (item, INTERNAL_QUEUE_DROP);
    };
  m_internalQueueDadFunctor = [this] (Ptr<const QueueDiscItem> item)
    {
      return DropAfterDequeue (item, INTERNAL_QUEUE_DROP);
    };

  // These lambdas call the DropBeforeEnqueue or DropAfterDequeue methods of this
  // QueueDisc object. Given that a callback to the operator() of these lambdas
  // is connected to the DropBeforeEnqueue and DropAfterDequeue traces of the
  // child queue discs, the concatenation of the CHILD_QUEUE_DISC_DROP constant
  // and the second argument provided by such traces is passed as the reason why
  // the packet is dropped.
  m_childQueueDiscDbeFunctor = [this] (Ptr<const QueueDiscItem> item, const char* r)
    {
      return DropBeforeEnqueue (item,
                                m_childQueueDiscDropMsg.assign (CHILD_QUEUE_DISC_DROP).append (r).data ());
    };
  m_childQueueDiscDadFunctor = [this] (Ptr<const QueueDiscItem> item, const char* r)
    {
      return DropAfterDequeue (item,
                               m_childQueueDiscDropMsg.assign (CHILD_QUEUE_DISC_DROP).append (r).data ());
    };
  m_childQueueDiscMarkFunctor = [this] (Ptr<const QueueDiscItem> item, const char* r)
    {
      return Mark (const_cast<QueueDiscItem *> (PeekPointer (item)),
                   m_childQueueDiscMarkMsg.assign (CHILD_QUEUE_DISC_MARK).append (r).data ());
    };
}

QueueDisc::QueueDisc (QueueDiscSizePolicy policy, QueueSizeUnit unit)
  : QueueDisc (policy)
{
  m_maxSize = QueueSize (unit, 0);
  m_prohibitChangeMode = true;
}

QueueDisc::~QueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

void
QueueDisc::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_queues.clear ();
  m_filters.clear ();
  m_classes.clear ();
  m_devQueueIface = 0;
  m_send = nullptr;
  m_requeued = 0;
  m_internalQueueDbeFunctor = nullptr;
  m_internalQueueDadFunctor = nullptr;
  m_childQueueDiscDbeFunctor = nullptr;
  m_childQueueDiscDadFunctor = nullptr;
  Object::DoDispose ();
}

void
QueueDisc::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);

  // Check the configuration and initialize the parameters of this queue disc
  [[maybe_unused]] bool ok = CheckConfig ();
  NS_ASSERT_MSG (ok, "The queue disc configuration is not correct");
  InitializeParams ();

  // Check the configuration and initialize the parameters of the child queue discs
  for (std::vector<Ptr<QueueDiscClass> >::iterator cl = m_classes.begin ();
       cl != m_classes.end (); cl++)
    {
      (*cl)->GetQueueDisc ()->Initialize ();
    }

  Object::DoInitialize ();
}

const QueueDisc::Stats&
QueueDisc::GetStats (void)
{
  NS_ASSERT (m_stats.nTotalDroppedPackets == m_stats.nTotalDroppedPacketsBeforeEnqueue
             + m_stats.nTotalDroppedPacketsAfterDequeue);
  NS_ASSERT (m_stats.nTotalDroppedBytes == m_stats.nTotalDroppedBytesBeforeEnqueue
             + m_stats.nTotalDroppedBytesAfterDequeue);

  // the total number of sent packets is only updated here to avoid to increase it
  // after a dequeue and then having to decrease it if the packet is dropped after
  // dequeue or requeued
  m_stats.nTotalSentPackets = m_stats.nTotalDequeuedPackets - (m_requeued ? 1 : 0)
                              - m_stats.nTotalDroppedPacketsAfterDequeue;
  m_stats.nTotalSentBytes = m_stats.nTotalDequeuedBytes - (m_requeued ? m_requeued->GetSize () : 0)
                            - m_stats.nTotalDroppedBytesAfterDequeue;

  return m_stats;
}

uint32_t
QueueDisc::GetNPackets () const
{
  NS_LOG_FUNCTION (this);
  return m_nPackets;
}

uint32_t
QueueDisc::GetNBytes (void) const
{
  NS_LOG_FUNCTION (this);
  return m_nBytes;
}

QueueSize
QueueDisc::GetMaxSize (void) const
{
  NS_LOG_FUNCTION (this);

  switch (m_sizePolicy)
    {
    case QueueDiscSizePolicy::NO_LIMITS:
      NS_FATAL_ERROR ("The size of this queue disc is not limited");

    case QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE:
      if (GetNInternalQueues ())
        {
          return GetInternalQueue (0)->GetMaxSize ();
        }

    case QueueDiscSizePolicy::SINGLE_CHILD_QUEUE_DISC:
      if (GetNQueueDiscClasses ())
        {
          return GetQueueDiscClass (0)->GetQueueDisc ()->GetMaxSize ();
        }

    case QueueDiscSizePolicy::MULTIPLE_QUEUES:
    default:
      return m_maxSize;
    }
}

bool
QueueDisc::SetMaxSize (QueueSize size)
{
  NS_LOG_FUNCTION (this << size);

  // do nothing if the limit is null
  if (!size.GetValue ())
    {
      return false;
    }

  if (m_prohibitChangeMode && size.GetUnit () != m_maxSize.GetUnit ())
    {
      NS_LOG_DEBUG ("Changing the mode of this queue disc is prohibited");
      return false;
    }

  switch (m_sizePolicy)
    {
    case QueueDiscSizePolicy::NO_LIMITS:
      NS_FATAL_ERROR ("The size of this queue disc is not limited");

    case QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE:
      if (GetNInternalQueues ())
        {
          GetInternalQueue (0)->SetMaxSize (size);
        }

    case QueueDiscSizePolicy::SINGLE_CHILD_QUEUE_DISC:
      if (GetNQueueDiscClasses ())
        {
          GetQueueDiscClass (0)->GetQueueDisc ()->SetMaxSize (size);
        }

    case QueueDiscSizePolicy::MULTIPLE_QUEUES:
    default:
      m_maxSize = size;
    }
  return true;
}

QueueSize
QueueDisc::GetCurrentSize (void)
{
  NS_LOG_FUNCTION (this);

  if (GetMaxSize ().GetUnit () == QueueSizeUnit::PACKETS)
    {
      return QueueSize (QueueSizeUnit::PACKETS, m_nPackets);
    }
  if (GetMaxSize ().GetUnit () == QueueSizeUnit::BYTES)
    {
      return QueueSize (QueueSizeUnit::BYTES, m_nBytes);
    }
  NS_ABORT_MSG ("Unknown queue size unit");
}

void
QueueDisc::SetNetDeviceQueueInterface (Ptr<NetDeviceQueueInterface> ndqi)
{
  NS_LOG_FUNCTION (this << ndqi);
  m_devQueueIface = ndqi;
}

Ptr<NetDeviceQueueInterface>
QueueDisc::GetNetDeviceQueueInterface (void) const
{
  NS_LOG_FUNCTION (this);
  return m_devQueueIface;
}

void
QueueDisc::SetSendCallback (SendCallback func)
{
  NS_LOG_FUNCTION (this);
  m_send = func;
}

QueueDisc::SendCallback
QueueDisc::GetSendCallback (void) const
{
  NS_LOG_FUNCTION (this);
  return m_send;
}

void
QueueDisc::SetQuota (const uint32_t quota)
{
  NS_LOG_FUNCTION (this << quota);
  m_quota = quota;
}

uint32_t
QueueDisc::GetQuota (void) const
{
  NS_LOG_FUNCTION (this);
  return m_quota;
}

void
QueueDisc::AddInternalQueue (Ptr<InternalQueue> queue)
{
  NS_LOG_FUNCTION (this);

  // set various callbacks on the internal queue, so that the queue disc is
  // notified of packets enqueued, dequeued or dropped by the internal queue
  queue->TraceConnectWithoutContext ("Enqueue",
                                     MakeCallback (&QueueDisc::PacketEnqueued, this));
  queue->TraceConnectWithoutContext ("Dequeue",
                                     MakeCallback (&QueueDisc::PacketDequeued, this));
  queue->TraceConnectWithoutContext ("DropBeforeEnqueue",
                                     MakeCallback (&InternalQueueDropFunctor::operator(),
                                                   &m_internalQueueDbeFunctor));
  queue->TraceConnectWithoutContext ("DropAfterDequeue",
                                     MakeCallback (&InternalQueueDropFunctor::operator(),
                                                   &m_internalQueueDadFunctor));
  m_queues.push_back (queue);
}

Ptr<QueueDisc::InternalQueue>
QueueDisc::GetInternalQueue (std::size_t i) const
{
  NS_ASSERT (i < m_queues.size ());
  return m_queues[i];
}

std::size_t
QueueDisc::GetNInternalQueues (void) const
{
  return m_queues.size ();
}

void
QueueDisc::AddPacketFilter (Ptr<PacketFilter> filter)
{
  NS_LOG_FUNCTION (this);
  m_filters.push_back (filter);
}

Ptr<PacketFilter>
QueueDisc::GetPacketFilter (std::size_t i) const
{
  NS_ASSERT (i < m_filters.size ());
  return m_filters[i];
}

std::size_t
QueueDisc::GetNPacketFilters (void) const
{
  return m_filters.size ();
}

void
QueueDisc::AddQueueDiscClass (Ptr<QueueDiscClass> qdClass)
{
  NS_LOG_FUNCTION (this);
  NS_ABORT_MSG_IF (qdClass->GetQueueDisc () == 0, "Cannot add a class with no attached queue disc");
  // the child queue disc cannot be one with wake mode equal to WAKE_CHILD because
  // such queue discs do not implement the enqueue/dequeue methods
  NS_ABORT_MSG_IF (qdClass->GetQueueDisc ()->GetWakeMode () == WAKE_CHILD,
                   "A queue disc with WAKE_CHILD as wake mode can only be a root queue disc");

  // set the parent callbacks on the child queue disc, so that it can notify
  // the parent queue disc of packets enqueued, dequeued, dropped, or marked
  qdClass->GetQueueDisc ()->TraceConnectWithoutContext ("Enqueue",
                                     MakeCallback (&QueueDisc::PacketEnqueued, this));
  qdClass->GetQueueDisc ()->TraceConnectWithoutContext ("Dequeue",
                                     MakeCallback (&QueueDisc::PacketDequeued, this));
  qdClass->GetQueueDisc ()->TraceConnectWithoutContext ("DropBeforeEnqueue",
                                     MakeCallback (&ChildQueueDiscDropFunctor::operator(),
                                                   &m_childQueueDiscDbeFunctor));
  qdClass->GetQueueDisc ()->TraceConnectWithoutContext ("DropAfterDequeue",
                                     MakeCallback (&ChildQueueDiscDropFunctor::operator(),
                                                   &m_childQueueDiscDadFunctor));
  qdClass->GetQueueDisc ()->TraceConnectWithoutContext ("Mark",
                                     MakeCallback (&ChildQueueDiscMarkFunctor::operator(),
                                                   &m_childQueueDiscMarkFunctor));
  m_classes.push_back (qdClass);
}

Ptr<QueueDiscClass>
QueueDisc::GetQueueDiscClass (std::size_t i) const
{
  NS_ASSERT (i < m_classes.size ());
  return m_classes[i];
}

std::size_t
QueueDisc::GetNQueueDiscClasses (void) const
{
  return m_classes.size ();
}

int32_t
QueueDisc::Classify (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  int32_t ret = PacketFilter::PF_NO_MATCH;
  for (std::vector<Ptr<PacketFilter> >::iterator f = m_filters.begin ();
       f != m_filters.end () && ret == PacketFilter::PF_NO_MATCH; f++)
    {
      ret = (*f)->Classify (item);
    }
  return ret;
}

QueueDisc::WakeMode
QueueDisc::GetWakeMode (void) const
{
  return WAKE_ROOT;
}

void
QueueDisc::PacketEnqueued (Ptr<const QueueDiscItem> item)
{
  m_nPackets++;
  m_nBytes += item->GetSize ();
  m_stats.nTotalEnqueuedPackets++;
  m_stats.nTotalEnqueuedBytes += item->GetSize ();

  NS_LOG_LOGIC ("m_traceEnqueue (p)");
  m_traceEnqueue (item);
}

void
QueueDisc::PacketDequeued (Ptr<const QueueDiscItem> item)
{
  // If the queue disc asked the internal queue or the child queue disc to
  // dequeue a packet because a peek operation was requested, the packet is
  // still held by the queue disc, hence we do not need to update statistics
  // and fire the dequeue trace. This function will be explicitly called when
  // the packet will be actually dequeued.
  if (!m_peeked)
    {
      m_nPackets--;
      m_nBytes -= item->GetSize ();
      m_stats.nTotalDequeuedPackets++;
      m_stats.nTotalDequeuedBytes += item->GetSize ();

      m_sojourn (Simulator::Now () - item->GetTimeStamp ());

      NS_LOG_LOGIC ("m_traceDequeue (p)");
      m_traceDequeue (item);
    }
}

void
QueueDisc::DropBeforeEnqueue (Ptr<const QueueDiscItem> item, const char* reason)
{
  NS_LOG_FUNCTION (this << item << reason);

  m_stats.nTotalDroppedPackets++;
  m_stats.nTotalDroppedBytes += item->GetSize ();
  m_stats.nTotalDroppedPacketsBeforeEnqueue++;
  m_stats.nTotalDroppedBytesBeforeEnqueue += item->GetSize ();

  // update the number of packets dropped for the given reason
  std::map<std::string, uint32_t>::iterator itp = m_stats.nDroppedPacketsBeforeEnqueue.find (reason);
  if (itp != m_stats.nDroppedPacketsBeforeEnqueue.end ())
    {
      itp->second++;
    }
  else
    {
      m_stats.nDroppedPacketsBeforeEnqueue[reason] = 1;
    }
  // update the amount of bytes dropped for the given reason
  std::map<std::string, uint64_t>::iterator itb = m_stats.nDroppedBytesBeforeEnqueue.find (reason);
  if (itb != m_stats.nDroppedBytesBeforeEnqueue.end ())
    {
      itb->second += item->GetSize ();
    }
  else
    {
      m_stats.nDroppedBytesBeforeEnqueue[reason] = item->GetSize ();
    }

  NS_LOG_DEBUG ("Total packets/bytes dropped before enqueue: "
                << m_stats.nTotalDroppedPacketsBeforeEnqueue << " / "
                << m_stats.nTotalDroppedBytesBeforeEnqueue);
  NS_LOG_LOGIC ("m_traceDropBeforeEnqueue (p)");
  m_traceDrop (item);
  m_traceDropBeforeEnqueue (item, reason);
}

void
QueueDisc::DropAfterDequeue (Ptr<const QueueDiscItem> item, const char* reason)
{
  NS_LOG_FUNCTION (this << item << reason);

  m_stats.nTotalDroppedPackets++;
  m_stats.nTotalDroppedBytes += item->GetSize ();
  m_stats.nTotalDroppedPacketsAfterDequeue++;
  m_stats.nTotalDroppedBytesAfterDequeue += item->GetSize ();

  // update the number of packets dropped for the given reason
  std::map<std::string, uint32_t>::iterator itp = m_stats.nDroppedPacketsAfterDequeue.find (reason);
  if (itp != m_stats.nDroppedPacketsAfterDequeue.end ())
    {
      itp->second++;
    }
  else
    {
      m_stats.nDroppedPacketsAfterDequeue[reason] = 1;
    }
  // update the amount of bytes dropped for the given reason
  std::map<std::string, uint64_t>::iterator itb = m_stats.nDroppedBytesAfterDequeue.find (reason);
  if (itb != m_stats.nDroppedBytesAfterDequeue.end ())
    {
      itb->second += item->GetSize ();
    }
  else
    {
      m_stats.nDroppedBytesAfterDequeue[reason] = item->GetSize ();
    }

  // if in the context of a peek request a dequeued packet is dropped, we need
  // to update the statistics and fire the dequeue trace before firing the drop
  // after dequeue trace
  if (m_peeked)
    {
      // temporarily set m_peeked to false, otherwise PacketDequeued does nothing
      m_peeked = false;
      PacketDequeued (item);
      m_peeked = true;
    }

  NS_LOG_DEBUG ("Total packets/bytes dropped after dequeue: "
                << m_stats.nTotalDroppedPacketsAfterDequeue << " / "
                << m_stats.nTotalDroppedBytesAfterDequeue);
  NS_LOG_LOGIC ("m_traceDropAfterDequeue (p)");
  m_traceDrop (item);
  m_traceDropAfterDequeue (item, reason);
}

bool
QueueDisc::Mark (Ptr<QueueDiscItem> item, const char* reason)
{
  NS_LOG_FUNCTION (this << item << reason);

  bool retval = item->Mark ();

  if (!retval)
    {
      return false;
    }

  m_stats.nTotalMarkedPackets++;
  m_stats.nTotalMarkedBytes += item->GetSize ();

  // update the number of packets marked for the given reason
  std::map<std::string, uint32_t>::iterator itp = m_stats.nMarkedPackets.find (reason);
  if (itp != m_stats.nMarkedPackets.end ())
    {
      itp->second++;
    }
  else
    {
      m_stats.nMarkedPackets[reason] = 1;
    }
  // update the amount of bytes marked for the given reason
  std::map<std::string, uint64_t>::iterator itb = m_stats.nMarkedBytes.find (reason);
  if (itb != m_stats.nMarkedBytes.end ())
    {
      itb->second += item->GetSize ();
    }
  else
    {
      m_stats.nMarkedBytes[reason] = item->GetSize ();
    }

  NS_LOG_DEBUG ("Total packets/bytes marked: "
                << m_stats.nTotalMarkedPackets << " / "
                << m_stats.nTotalMarkedBytes);
  m_traceMark (item, reason);
  return true;
}

bool
QueueDisc::Enqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  m_stats.nTotalReceivedPackets++;
  m_stats.nTotalReceivedBytes += item->GetSize ();

  bool retval = DoEnqueue (item);

  if (retval)
    {
      item->SetTimeStamp (Simulator::Now ());
    }

  // DoEnqueue may return false because:
  // 1) the internal queue is full
  //    -> the DropBeforeEnqueue method of this queue disc is automatically called
  //       because QueueDisc::AddInternalQueue sets the trace callback
  // 2) the child queue disc dropped the packet
  //    -> the DropBeforeEnqueue method of this queue disc is automatically called
  //       because QueueDisc::AddQueueDiscClass sets the trace callback
  // 3) it dropped the packet
  //    -> DoEnqueue has to explicitly call DropBeforeEnqueue
  // Thus, we do not have to call DropBeforeEnqueue here.

  // check that the received packet was either enqueued or dropped
  NS_ASSERT (m_stats.nTotalReceivedPackets == m_stats.nTotalDroppedPacketsBeforeEnqueue +
             m_stats.nTotalEnqueuedPackets);
  NS_ASSERT (m_stats.nTotalReceivedBytes == m_stats.nTotalDroppedBytesBeforeEnqueue +
             m_stats.nTotalEnqueuedBytes);

  return retval;
}

Ptr<QueueDiscItem>
QueueDisc::Dequeue (void)
{
  NS_LOG_FUNCTION (this);

  // The QueueDisc::DoPeek method dequeues a packet and keeps it as a requeued
  // packet. Thus, first check whether a peeked packet exists. Otherwise, call
  // the private DoDequeue method.
  Ptr<QueueDiscItem> item = m_requeued;

  if (item)
    {
      m_requeued = 0;
      if (m_peeked)
        {
          // If the packet was requeued because a peek operation was requested
          // (which is the case here because DequeuePacket calls Dequeue only
          // when m_requeued is null), we need to explicitly call PacketDequeued
          // to update statistics about dequeued packets and fire the dequeue trace.
          m_peeked = false;
          PacketDequeued (item);
        }
    }
  else
    {
      item = DoDequeue ();
    }

  NS_ASSERT (m_nPackets == m_stats.nTotalEnqueuedPackets - m_stats.nTotalDequeuedPackets);
  NS_ASSERT (m_nBytes == m_stats.nTotalEnqueuedBytes - m_stats.nTotalDequeuedBytes);

  return item;
}

Ptr<const QueueDiscItem>
QueueDisc::Peek (void)
{
  NS_LOG_FUNCTION (this);
  return DoPeek ();
}

Ptr<const QueueDiscItem>
QueueDisc::DoPeek (void)
{
  NS_LOG_FUNCTION (this);

  if (!m_requeued)
    {
      m_peeked = true;
      m_requeued = Dequeue ();
      // if no packet is returned, reset the m_peeked flag
      if (!m_requeued)
        {
          m_peeked = false;
        }
    }
  return m_requeued;
}

void
QueueDisc::Run (void)
{
  NS_LOG_FUNCTION (this);

  if (RunBegin ())
    {
      uint32_t quota = m_quota;
      while (Restart ())
        {
          quota -= 1;
          if (quota <= 0)
            {
              /// \todo netif_schedule (q);
              break;
            }
        }
      RunEnd ();
    }
}

bool
QueueDisc::RunBegin (void)
{
  NS_LOG_FUNCTION (this);
  if (m_running)
    {
      return false;
    }

  m_running = true;
  return true;
}

void
QueueDisc::RunEnd (void)
{
  NS_LOG_FUNCTION (this);
  m_running = false;
}

bool
QueueDisc::Restart (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<QueueDiscItem> item = DequeuePacket();
  if (item == 0)
    {
      NS_LOG_LOGIC ("No packet to send");
      return false;
    }

  return Transmit (item);
}

Ptr<QueueDiscItem>
QueueDisc::DequeuePacket ()
{
  NS_LOG_FUNCTION (this);

  Ptr<QueueDiscItem> item;

  // First check if there is a requeued packet
  if (m_requeued != 0)
    {
        // If the queue where the requeued packet is destined to is not stopped, return
        // the requeued packet; otherwise, return an empty packet.
        // If the device does not support flow control, the device queue is never stopped
        if (!m_devQueueIface || !m_devQueueIface->GetTxQueue (m_requeued->GetTxQueueIndex ())->IsStopped ())
          {
            item = m_requeued;
            m_requeued = 0;
            if (m_peeked)
              {
                // If the packet was requeued because a peek operation was requested
                // we need to explicitly call PacketDequeued to update statistics
                // about dequeued packets and fire the dequeue trace.
                m_peeked = false;
                PacketDequeued (item);
              }
          }
    }
  else
    {
      // If the device is multi-queue (actually, Linux checks if the queue disc has
      // multiple queues), ask the queue disc to dequeue a packet (a multi-queue aware
      // queue disc should try not to dequeue a packet destined to a stopped queue).
      // Otherwise, ask the queue disc to dequeue a packet only if the (unique) queue
      // is not stopped.
      if (!m_devQueueIface ||
          m_devQueueIface->GetNTxQueues ()>1 || !m_devQueueIface->GetTxQueue (0)->IsStopped ())
        {
          item = Dequeue ();
          // If the item is not null, add the header to the packet.
          if (item != 0)
            {
              item->AddHeader ();
            }
          // Here, Linux tries bulk dequeues
        }
    }
  return item;
}

void
QueueDisc::Requeue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);
  m_requeued = item;
  /// \todo netif_schedule (q);

  m_stats.nTotalRequeuedPackets++;
  m_stats.nTotalRequeuedBytes += item->GetSize ();

  NS_LOG_LOGIC ("m_traceRequeue (p)");
  m_traceRequeue (item);
}

bool
QueueDisc::Transmit (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  // if the device queue is stopped, requeue the packet and return false.
  // Note that if the underlying device is tc-unaware, packets are never
  // requeued because the queues of tc-unaware devices are never stopped
  if (m_devQueueIface && m_devQueueIface->GetTxQueue (item->GetTxQueueIndex ())->IsStopped ())
    {
      Requeue (item);
      return false;
    }

  // a single queue device makes no use of the priority tag
  // a device that does not install a device queue interface likely makes no use of it as well
  if (!m_devQueueIface || m_devQueueIface->GetNTxQueues () == 1)
    {
      SocketPriorityTag priorityTag;
      item->GetPacket ()->RemovePacketTag (priorityTag);
    }
  NS_ASSERT_MSG (m_send, "Send callback not set");
  m_send (item);

  // the behavior here slightly diverges from Linux. In Linux, it is advised that
  // the function called when a packet needs to be transmitted (ndo_start_xmit)
  // should always return NETDEV_TX_OK, which means that the packet is consumed by
  // the device driver and thus is not requeued. However, the ndo_start_xmit function
  // of the device driver is allowed to return NETDEV_TX_BUSY (and hence the packet
  // is requeued) when there is no room for the received packet in the device queue,
  // despite the queue is not stopped. This case is considered as a corner case or
  // an hard error, and should be avoided.
  // Here, we do not handle such corner case and always assume that the packet is
  // consumed by the netdevice. Thus, we ignore the value returned by Send and a
  // packet sent to a netdevice is never requeued. The reason is that the semantics
  // of the value returned by NetDevice::Send does not match that of the value
  // returned by ndo_start_xmit.

  // if the queue disc is empty or the device queue is now stopped, return false so
  // that the Run method does not attempt to dequeue other packets and exits
  if (GetNPackets () == 0 ||
      (m_devQueueIface && m_devQueueIface->GetTxQueue (item->GetTxQueueIndex ())->IsStopped ()))
    {
      return false;
    }

  return true;
}

} // namespace ns3
