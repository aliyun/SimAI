/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 NITK Surathkal
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
 * Cobalt, the CoDel - BLUE - Alternate Queueing discipline
 * Based on linux code.
 *
 * Ported to ns-3 by: Vignesh Kannan <vignesh2496@gmail.com>
 *                    Harsh Lara <harshapplefan@gmail.com>
 *                    Jendaipou Palmei <jendaipoupalmei@gmail.com>
 *                    Shefali Gupta <shefaligups11@gmail.com>
 *                    Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

#include "ns3/log.h"
#include "ns3/enum.h"
#include "ns3/uinteger.h"
#include "ns3/abort.h"
#include "cobalt-queue-disc.h"
#include "ns3/object-factory.h"
#include "ns3/drop-tail-queue.h"
#include "ns3/net-device-queue-interface.h"
#include <climits>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("CobaltQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (CobaltQueueDisc);

TypeId CobaltQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::CobaltQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<CobaltQueueDisc> ()
    .AddAttribute ("MaxSize",
                   "The maximum number of packets/bytes accepted by this queue disc.",
                   QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, 1500 * DEFAULT_COBALT_LIMIT)),
                   MakeQueueSizeAccessor (&QueueDisc::SetMaxSize,
                                          &QueueDisc::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("Interval",
                   "The Cobalt algorithm interval",
                   StringValue ("100ms"),
                   MakeTimeAccessor (&CobaltQueueDisc::m_interval),
                   MakeTimeChecker ())
    .AddAttribute ("Target",
                   "The Cobalt algorithm target queue delay",
                   StringValue ("5ms"),
                   MakeTimeAccessor (&CobaltQueueDisc::m_target),
                   MakeTimeChecker ())
    .AddAttribute ("UseEcn",
                   "True to use ECN (packets are marked instead of being dropped)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&CobaltQueueDisc::m_useEcn),
                   MakeBooleanChecker ())
    .AddAttribute ("Pdrop",
                   "Marking Probability",
                   DoubleValue (0),
                   MakeDoubleAccessor (&CobaltQueueDisc::m_pDrop),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Increment",
                   "Pdrop increment value",
                   DoubleValue (1. / 256),
                   MakeDoubleAccessor (&CobaltQueueDisc::m_increment),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Decrement",
                   "Pdrop decrement Value",
                   DoubleValue (1. / 4096),
                   MakeDoubleAccessor (&CobaltQueueDisc::m_decrement),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("CeThreshold",
                   "The CoDel CE threshold for marking packets",
                   TimeValue (Time::Max ()),
                   MakeTimeAccessor (&CobaltQueueDisc::m_ceThreshold),
                   MakeTimeChecker ())
    .AddAttribute ("UseL4s",
                   "True to use L4S (only ECT1 packets are marked at CE threshold)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&CobaltQueueDisc::m_useL4s),
                   MakeBooleanChecker ())
    .AddAttribute ("BlueThreshold",
                   "The Threshold after which Blue is enabled",
                   TimeValue (MilliSeconds (400)),
                   MakeTimeAccessor (&CobaltQueueDisc::m_blueThreshold),
                   MakeTimeChecker ())
    .AddTraceSource ("Count",
                     "Cobalt count",
                     MakeTraceSourceAccessor (&CobaltQueueDisc::m_count),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("DropState",
                     "Dropping state",
                     MakeTraceSourceAccessor (&CobaltQueueDisc::m_dropping),
                     "ns3::TracedValueCallback::Bool")
    .AddTraceSource ("DropNext",
                     "Time until next packet drop",
                     MakeTraceSourceAccessor (&CobaltQueueDisc::m_dropNext),
                     "ns3::TracedValueCallback::Uint32")
  ;

  return tid;
}

/**
 * Performs a reciprocal divide, similar to the
 * Linux kernel reciprocal_divide function
 * \param A numerator
 * \param R reciprocal of the denominator B
 * \return the value of A/B
 */
/* borrowed from the linux kernel */
static inline uint32_t ReciprocalDivide (uint32_t A, uint32_t R)
{
  return (uint32_t)(((uint64_t)A * R) >> 32);
}

/**
 * Returns the current time translated in CoDel time representation
 * \return the current time
 */
static int64_t CoDelGetTime (void)
{
  Time time = Simulator::Now ();
  int64_t ns = time.GetNanoSeconds ();

  return ns;
}

CobaltQueueDisc::CobaltQueueDisc ()
  : QueueDisc ()
{
  NS_LOG_FUNCTION (this);
  InitializeParams ();
  m_uv = CreateObject<UniformRandomVariable> ();
}

double CobaltQueueDisc::GetPdrop () const
{
  return m_pDrop;
}

CobaltQueueDisc::~CobaltQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

int64_t
CobaltQueueDisc::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_uv->SetStream (stream);
  return 1;
}

void
CobaltQueueDisc::InitializeParams (void)
{
  // Cobalt parameters
  NS_LOG_FUNCTION (this);
  m_recInvSqrtCache[0] = ~0;
  CacheInit ();
  m_count = 0;
  m_dropping = false;
  m_recInvSqrt = ~0U;
  m_lastUpdateTimeBlue = 0;
  m_dropNext = 0;
}

bool
CobaltQueueDisc::CoDelTimeAfter (int64_t a, int64_t b)
{
  return  ((int64_t)(a) - (int64_t)(b) > 0);
}

bool
CobaltQueueDisc::CoDelTimeAfterEq (int64_t a, int64_t b)
{
  return ((int64_t)(a) - (int64_t)(b) >= 0);
}

int64_t
CobaltQueueDisc::Time2CoDel (Time t) const
{
  return (t.GetNanoSeconds ());
}

Time
CobaltQueueDisc::GetTarget (void) const
{
  return m_target;
}

Time
CobaltQueueDisc::GetInterval (void) const
{
  return m_interval;
}

int64_t
CobaltQueueDisc::GetDropNext (void) const
{
  return m_dropNext;
}

void
CobaltQueueDisc::NewtonStep (void)
{
  NS_LOG_FUNCTION (this);
  uint32_t invsqrt = ((uint32_t) m_recInvSqrt);
  uint32_t invsqrt2 = ((uint64_t) invsqrt * invsqrt) >> 32;
  uint64_t val = (3ll << 32) - ((uint64_t) m_count * invsqrt2);

  val >>= 2; /* avoid overflow */
  val = (val * invsqrt) >> (32 - 2 + 1);
  m_recInvSqrt = val;
}

void
CobaltQueueDisc::CacheInit (void)
{
  m_recInvSqrt = ~0U;
  m_recInvSqrtCache[0] = m_recInvSqrt;

  for (m_count = 1; m_count < (uint32_t)(REC_INV_SQRT_CACHE); m_count++)
    {
      NewtonStep ();
      NewtonStep ();
      NewtonStep ();
      NewtonStep ();
      m_recInvSqrtCache[m_count] = m_recInvSqrt;
    }
}

void
CobaltQueueDisc::InvSqrt (void)
{
  if (m_count < (uint32_t)REC_INV_SQRT_CACHE)
    {
      m_recInvSqrt = m_recInvSqrtCache[m_count];
    }
  else
    {
      NewtonStep ();
    }
}

int64_t
CobaltQueueDisc::ControlLaw (int64_t t)
{
  NS_LOG_FUNCTION (this);
  return t + ReciprocalDivide (Time2CoDel (m_interval), m_recInvSqrt);
}

void
CobaltQueueDisc::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_uv = 0;
  QueueDisc::DoDispose ();
}

Ptr<const QueueDiscItem>
CobaltQueueDisc::DoPeek (void)
{
  NS_LOG_FUNCTION (this);
  if (GetInternalQueue (0)->IsEmpty ())
    {
      NS_LOG_LOGIC ("Queue empty");
      return 0;
    }

  Ptr<const QueueDiscItem> item = GetInternalQueue (0)->Peek ();

  NS_LOG_LOGIC ("Number packets " << GetInternalQueue (0)->GetNPackets ());
  NS_LOG_LOGIC ("Number bytes " << GetInternalQueue (0)->GetNBytes ());

  return item;
}

bool
CobaltQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNQueueDiscClasses () > 0)
    {
      NS_LOG_ERROR ("CobaltQueueDisc cannot have classes");
      return false;
    }

  if (GetNPacketFilters () > 0)
    {
      NS_LOG_ERROR ("CobaltQueueDisc cannot have packet filters");
      return false;
    }

  if (GetNInternalQueues () == 0)
    {

      AddInternalQueue (CreateObjectWithAttributes<DropTailQueue<QueueDiscItem> >
                          ("MaxSize", QueueSizeValue (GetMaxSize ())));
    }


  if (GetNInternalQueues () != 1)
    {
      NS_LOG_ERROR ("CobaltQueueDisc needs 1 internal queue");
      return false;
    }
  return true;
}

bool
CobaltQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);
  Ptr<Packet> p = item->GetPacket ();
  if (GetCurrentSize () + item > GetMaxSize ())
    {
      NS_LOG_LOGIC ("Queue full -- dropping pkt");
      int64_t now = CoDelGetTime ();
      // Call this to update Blue's drop probability
      CobaltQueueFull (now);
      DropBeforeEnqueue (item, OVERLIMIT_DROP);
      return false;
    }

  bool retval = GetInternalQueue (0)->Enqueue (item);

  // If Queue::Enqueue fails, QueueDisc::Drop is called by the internal queue
  // because QueueDisc::AddInternalQueue sets the drop callback

  NS_LOG_LOGIC ("Number packets " << GetInternalQueue (0)->GetNPackets ());
  NS_LOG_LOGIC ("Number bytes " << GetInternalQueue (0)->GetNBytes ());

  return retval;
}

Ptr<QueueDiscItem>
CobaltQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);

  while (1)
    {
      Ptr<QueueDiscItem> item = GetInternalQueue (0)->Dequeue ();
      if (!item)
        {
          // Leave dropping state when queue is empty (derived from Codel)
          m_dropping = false;
          NS_LOG_LOGIC ("Queue empty");
          int64_t now = CoDelGetTime ();
          // Call this to update Blue's drop probability
          CobaltQueueEmpty (now);
          return 0;
        }

      int64_t now = CoDelGetTime ();

      NS_LOG_LOGIC ("Popped " << item);
      NS_LOG_LOGIC ("Number packets remaining " << GetInternalQueue (0)->GetNPackets ());
      NS_LOG_LOGIC ("Number bytes remaining " << GetInternalQueue (0)->GetNBytes ());

      // Determine if item should be dropped
      // ECN marking happens inside this function, so it need not be done here
      bool drop = CobaltShouldDrop (item, now);

      if (drop)
        {
          DropAfterDequeue (item, TARGET_EXCEEDED_DROP);
        }
      else
        {
          return item;
        }
    }
}

// Call this when a packet had to be dropped due to queue overflow.
void CobaltQueueDisc::CobaltQueueFull (int64_t now)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC ("Outside IF block");
  if (CoDelTimeAfter ((now - m_lastUpdateTimeBlue), Time2CoDel (m_target)))
    {
      NS_LOG_LOGIC ("inside IF block");
      m_pDrop = std::min (m_pDrop + m_increment, (double)1.0);
      m_lastUpdateTimeBlue = now;
    }
  m_dropping = true;
  m_dropNext = now;
  if (!m_count)
    {
      m_count = 1;
    }
}

// Call this when the queue was serviced but turned out to be empty.
void CobaltQueueDisc::CobaltQueueEmpty (int64_t now)
{
  NS_LOG_FUNCTION (this);
  if (m_pDrop && CoDelTimeAfter ((now - m_lastUpdateTimeBlue), Time2CoDel (m_target)))
    {
      m_pDrop = std::max (m_pDrop - m_decrement, (double)0.0);
      m_lastUpdateTimeBlue = now;
    }
  m_dropping = false;

  if (m_count && CoDelTimeAfterEq ((now - m_dropNext), 0))
    {
      m_count--;
      InvSqrt ();
      m_dropNext = ControlLaw (m_dropNext);
    }
}

// Determines if Cobalt should drop the packet
bool CobaltQueueDisc::CobaltShouldDrop (Ptr<QueueDiscItem> item, int64_t now)
{
  NS_LOG_FUNCTION (this);
  bool drop = false;

  /* Simplified Codel implementation */
  Time delta = Simulator::Now () - item->GetTimeStamp ();
  NS_LOG_INFO ("Sojourn time " << delta.As (Time::S));
  int64_t sojournTime = Time2CoDel (delta);
  int64_t schedule = now - m_dropNext;
  bool over_target = CoDelTimeAfter (sojournTime, Time2CoDel (m_target));
  bool next_due = m_count && schedule >= 0;
  bool isMarked = false;

  // If L4S mode is enabled then check if the packet is ECT1 or CE and
  // if sojourn time is greater than CE threshold then the packet is marked.
  // If packet is marked succesfully then the CoDel steps can be skipped.
  if (item && m_useL4s)
    {
      uint8_t tosByte = 0;
      if (item->GetUint8Value (QueueItem::IP_DSFIELD, tosByte) && (((tosByte & 0x3) == 1) || (tosByte & 0x3) == 3))
        {
          if ((tosByte & 0x3) == 1)
            {
              NS_LOG_DEBUG ("ECT1 packet " << static_cast<uint16_t> (tosByte & 0x3));
            }
          else
            {
              NS_LOG_DEBUG ("CE packet " << static_cast<uint16_t> (tosByte & 0x3));
            }
          if (CoDelTimeAfter (sojournTime, Time2CoDel (m_ceThreshold)) && Mark (item, CE_THRESHOLD_EXCEEDED_MARK))
            {
              NS_LOG_LOGIC ("Marking due to CeThreshold " << m_ceThreshold.GetSeconds ());
            }
          return false;
        }
    }

  if (over_target)
    {
      if (!m_dropping)
        {
          m_dropping = true;
          m_dropNext = ControlLaw (now);
        }
      if (!m_count)
        {
          m_count = 1;
        }
    }
  else if (m_dropping)
    {
      m_dropping = false;
    }

  if (next_due && m_dropping)
    {
      /* Check for marking possibility only if BLUE decides NOT to drop. */
      /* Check if router and packet, both have ECN enabled. Only if this is true, mark the packet. */
      isMarked = (m_useEcn && Mark (item, FORCED_MARK));
      drop = !isMarked;

      m_count = std::max (m_count, m_count + 1);

      InvSqrt ();
      m_dropNext = ControlLaw (m_dropNext);
      schedule = now - m_dropNext;
    }
  else
    {
      while (next_due)
        {
          m_count--;
          InvSqrt ();
          m_dropNext = ControlLaw (m_dropNext);
          schedule = now - m_dropNext;
          next_due = m_count && schedule >= 0;
        }
    }

  // If CE threshold is enabled then isMarked flag is used to determine whether
  // packet is marked and if the packet is marked then a second attempt at marking should be suppressed.
  // If UseL4S attribute is enabled then ECT0 packets should not be marked.
  if (!isMarked && !m_useL4s && m_useEcn && CoDelTimeAfter (sojournTime, Time2CoDel (m_ceThreshold)) && Mark (item, CE_THRESHOLD_EXCEEDED_MARK))
    {
      NS_LOG_LOGIC ("Marking due to CeThreshold " << m_ceThreshold.GetSeconds ());
    }

  // Enable Blue Enhancement if sojourn time is greater than blueThreshold and its been m_target time until the last time blue was updated
  if (CoDelTimeAfter (sojournTime, Time2CoDel (m_blueThreshold)) && CoDelTimeAfter ((now - m_lastUpdateTimeBlue), Time2CoDel (m_target)))
    {
      m_pDrop = std::min (m_pDrop + m_increment, (double)1.0);
      m_lastUpdateTimeBlue = now;
    }

  /* Simple BLUE implementation. Lack of ECN is deliberate. */
  if (m_pDrop)
    {
      double u = m_uv->GetValue ();
      drop = drop | (u < m_pDrop);
    }

  /* Overload the drop_next field as an activity timeout */
  if (!m_count)
    {
      m_dropNext = now + Time2CoDel (m_interval);
    }
  else if (schedule > 0 && !drop)
    {
      m_dropNext = now;
    }

  return drop;
}

} // namespace ns3
