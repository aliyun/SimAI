/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */


#include "uan-header-rc.h"
#include "ns3/mac8-address.h"

#include <set>

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (UanHeaderRcData);
NS_OBJECT_ENSURE_REGISTERED (UanHeaderRcRts);
NS_OBJECT_ENSURE_REGISTERED (UanHeaderRcCtsGlobal);
NS_OBJECT_ENSURE_REGISTERED (UanHeaderRcCts);
NS_OBJECT_ENSURE_REGISTERED (UanHeaderRcAck);

UanHeaderRcData::UanHeaderRcData ()
  : Header (),
    m_frameNo (0),
    m_propDelay (Seconds (0))
{
}

UanHeaderRcData::UanHeaderRcData (uint8_t frameNo, Time propDelay)
  : Header (),
    m_frameNo (frameNo),
    m_propDelay (propDelay)
{

}

UanHeaderRcData::~UanHeaderRcData ()
{
}

TypeId
UanHeaderRcData::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::UanHeaderRcData")
    .SetParent<Header> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanHeaderRcData> ()
  ;
  return tid;
}

void
UanHeaderRcData::SetFrameNo (uint8_t no)
{
  m_frameNo = no;
}

void
UanHeaderRcData::SetPropDelay (Time propDelay)
{
  m_propDelay = propDelay;
}

uint8_t
UanHeaderRcData::GetFrameNo (void) const
{
  return m_frameNo;
}

Time
UanHeaderRcData::GetPropDelay (void) const
{
  return m_propDelay;
}

uint32_t
UanHeaderRcData::GetSerializedSize (void) const
{
  return 1 + 2;
}

void
UanHeaderRcData::Serialize (Buffer::Iterator start) const
{
  start.WriteU8 (m_frameNo);
  start.WriteU16 ( (uint16_t) m_propDelay.RoundTo (Time::MS).GetMilliSeconds ());
}
uint32_t
UanHeaderRcData::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator rbuf = start;

  m_frameNo = start.ReadU8 ();
  m_propDelay = Seconds ( ((double) start.ReadU16 ()) / 1000.0 );

  return rbuf.GetDistanceFrom (start);
}

void
UanHeaderRcData::Print (std::ostream &os, Time::Unit unit) const
{
  os << "Frame No=" << (uint32_t) m_frameNo << " Prop Delay=" << m_propDelay.As (unit);
}

void
UanHeaderRcData::Print (std::ostream &os) const
{
  Print (os, Time::S);
}

TypeId
UanHeaderRcData::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}


UanHeaderRcRts::UanHeaderRcRts ()
  : Header (),
    m_frameNo (0),
    m_noFrames (0),
    m_length (0),
    m_timeStamp (Seconds (0)),
    m_retryNo (0)
{

}

UanHeaderRcRts::UanHeaderRcRts (uint8_t frameNo, uint8_t retryNo, uint8_t noFrames, uint16_t length, Time timeStamp)
  : Header (),
    m_frameNo (frameNo),
    m_noFrames (noFrames),
    m_length (length),
    m_timeStamp (timeStamp),
    m_retryNo (retryNo)
{

}

UanHeaderRcRts::~UanHeaderRcRts ()
{

}

TypeId
UanHeaderRcRts::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::UanHeaderRcRts")
    .SetParent<Header> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanHeaderRcRts> ()
  ;
  return tid;

}

void
UanHeaderRcRts::SetFrameNo (uint8_t no)
{
  m_frameNo = no;
}

void
UanHeaderRcRts::SetNoFrames (uint8_t no)
{
  m_noFrames = no;
}

void
UanHeaderRcRts::SetLength (uint16_t length)
{
  m_length = length;
}
void
UanHeaderRcRts::SetTimeStamp (Time timeStamp)
{
  m_timeStamp = timeStamp;
}

void
UanHeaderRcRts::SetRetryNo (uint8_t no)
{
  m_retryNo = no;
}
uint8_t
UanHeaderRcRts::GetNoFrames () const
{
  return m_noFrames;
}

uint16_t
UanHeaderRcRts::GetLength () const
{
  return m_length;
}

Time
UanHeaderRcRts::GetTimeStamp (void) const
{
  return m_timeStamp;
}

uint8_t
UanHeaderRcRts::GetRetryNo (void) const
{
  return m_retryNo;
}

uint8_t
UanHeaderRcRts::GetFrameNo (void) const
{
  return m_frameNo;
}

uint32_t
UanHeaderRcRts::GetSerializedSize (void) const
{
  return 1 + 1 + 1 + 4 + 2;
}

void
UanHeaderRcRts::Serialize (Buffer::Iterator start) const
{
  start.WriteU8 (m_frameNo);
  start.WriteU8 (m_retryNo);
  start.WriteU8 (m_noFrames);
  start.WriteU16 (m_length);
  start.WriteU32 ((uint32_t) (m_timeStamp.RoundTo (Time::MS).GetMilliSeconds ()));
}

uint32_t
UanHeaderRcRts::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator rbuf = start;
  m_frameNo = rbuf.ReadU8 ();
  m_retryNo = rbuf.ReadU8 ();
  m_noFrames = rbuf.ReadU8 ();
  m_length = rbuf.ReadU16 ();
  m_timeStamp = Seconds ( ((double) rbuf.ReadU32 ()) / 1000.0 );
  // m_timeStamp = Seconds ( rbuf.ReadU16 ()/1000 );
  return rbuf.GetDistanceFrom (start);
}

void
UanHeaderRcRts::Print (std::ostream &os, Time::Unit unit) const
{
  os << "Frame #=" << (uint32_t) m_frameNo << " Retry #=" << (uint32_t) m_retryNo << " Num Frames=" << (uint32_t) m_noFrames << "Length=" << m_length << " Time Stamp=" << m_timeStamp.As (unit);
}

void
UanHeaderRcRts::Print (std::ostream &os) const
{
  Print(os, Time::S);
}

TypeId
UanHeaderRcRts::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}




UanHeaderRcCtsGlobal::UanHeaderRcCtsGlobal ()
  : Header (),
    m_retryRate (0),
    m_rateNum (0)
{

}

UanHeaderRcCtsGlobal::UanHeaderRcCtsGlobal (Time wt, Time ts, uint16_t rate, uint16_t retryRate)
  : Header (),
    m_timeStampTx (ts),
    m_winTime (wt),
    m_retryRate (retryRate),
    m_rateNum (rate)
{

}

UanHeaderRcCtsGlobal::~UanHeaderRcCtsGlobal ()
{

}

TypeId
UanHeaderRcCtsGlobal::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::UanHeaderRcCtsGlobal")
    .SetParent<Header> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanHeaderRcCtsGlobal> ()
  ;
  return tid;

}


void
UanHeaderRcCtsGlobal::SetRateNum (uint16_t rate)
{
  m_rateNum = rate;
}

void
UanHeaderRcCtsGlobal::SetRetryRate (uint16_t rate)
{
  m_retryRate = rate;
}

void
UanHeaderRcCtsGlobal::SetWindowTime (Time t)
{
  m_winTime = t;
}

void
UanHeaderRcCtsGlobal::SetTxTimeStamp (Time t)
{
  m_timeStampTx = t;
}

Time
UanHeaderRcCtsGlobal::GetWindowTime (void) const
{
  return m_winTime;
}

Time
UanHeaderRcCtsGlobal::GetTxTimeStamp (void) const
{
  return m_timeStampTx;
}

uint16_t
UanHeaderRcCtsGlobal::GetRetryRate (void) const
{
  return m_retryRate;
}

uint16_t
UanHeaderRcCtsGlobal::GetRateNum (void) const
{
  return m_rateNum;
}
uint32_t
UanHeaderRcCtsGlobal::GetSerializedSize (void) const
{
  return 4 + 4 + 2 + 2;
}

void
UanHeaderRcCtsGlobal::Serialize (Buffer::Iterator start) const
{
  start.WriteU16 (m_rateNum);
  start.WriteU16 (m_retryRate);
  start.WriteU32 ( (uint32_t) (m_timeStampTx.RoundTo (Time::MS).GetMilliSeconds ()));
  start.WriteU32 ( (uint32_t) (m_winTime.RoundTo (Time::MS).GetMilliSeconds ()));
}

uint32_t
UanHeaderRcCtsGlobal::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator rbuf = start;
  m_rateNum = rbuf.ReadU16 ();
  m_retryRate = rbuf.ReadU16 ();
  m_timeStampTx = Seconds ( ( (double) rbuf.ReadU32 ()) / 1000.0 );
  m_winTime = Seconds ( ( (double) rbuf.ReadU32 ()) / 1000.0 );
  return rbuf.GetDistanceFrom (start);

}

void
UanHeaderRcCtsGlobal::Print (std::ostream &os, Time::Unit unit) const
{
  os << "CTS Global (Rate #=" << m_rateNum << ", Retry Rate=" << m_retryRate << ", TX Time=" << m_timeStampTx.As (Time::S) << ", Win Time=" << m_winTime.As (Time::S) << ")";
}

void
UanHeaderRcCtsGlobal::Print (std::ostream &os) const
{
  Print(os, Time::S);
}

TypeId
UanHeaderRcCtsGlobal::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

UanHeaderRcCts::UanHeaderRcCts ()
  : Header (),
    m_frameNo (0),
    m_timeStampRts (Seconds (0)),
    m_retryNo (0),
    m_delay (Seconds (0)),
    m_address (Mac8Address::GetBroadcast ())
{

}

UanHeaderRcCts::UanHeaderRcCts (uint8_t frameNo, uint8_t retryNo, Time ts, Time delay, Mac8Address addr)
  : Header (),
    m_frameNo (frameNo),
    m_timeStampRts (ts),
    m_retryNo (retryNo),
    m_delay (delay),
    m_address (addr)
{

}

UanHeaderRcCts::~UanHeaderRcCts ()
{

}

TypeId
UanHeaderRcCts::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::UanHeaderRcCts")
    .SetParent<Header> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanHeaderRcCts> ()
  ;
  return tid;

}

void
UanHeaderRcCts::SetFrameNo (uint8_t frameNo)
{
  m_frameNo = frameNo;
}

void
UanHeaderRcCts::SetRtsTimeStamp (Time timeStamp)
{
  m_timeStampRts = timeStamp;
}


void
UanHeaderRcCts::SetDelayToTx (Time delay)
{
  m_delay = delay;
}

void
UanHeaderRcCts::SetRetryNo (uint8_t no)
{
  m_retryNo = no;
}

void
UanHeaderRcCts::SetAddress (Mac8Address addr)
{
  m_address = addr;
}
uint8_t
UanHeaderRcCts::GetFrameNo () const
{
  return m_frameNo;
}

Time
UanHeaderRcCts::GetRtsTimeStamp (void) const
{
  return m_timeStampRts;
}

Time
UanHeaderRcCts::GetDelayToTx (void) const
{
  return m_delay;
}

uint8_t
UanHeaderRcCts::GetRetryNo () const
{
  return m_retryNo;
}

Mac8Address
UanHeaderRcCts::GetAddress () const
{
  return m_address;
}

uint32_t
UanHeaderRcCts::GetSerializedSize (void) const
{
  return 1 + 1 + 1 + 4 + 4;
}


void
UanHeaderRcCts::Serialize (Buffer::Iterator start) const
{
  uint8_t address = 0;
  m_address.CopyTo (&address);
  start.WriteU8 (address);
  start.WriteU8 (m_frameNo);
  start.WriteU8 (m_retryNo);
  start.WriteU32 ((uint32_t) (m_timeStampRts.RoundTo (Time::MS).GetMilliSeconds ()));
  start.WriteU32 ((uint32_t) (m_delay.RoundTo (Time::MS).GetMilliSeconds ()));
}

uint32_t
UanHeaderRcCts::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator rbuf = start;
  m_address = Mac8Address (rbuf.ReadU8 ());
  m_frameNo = rbuf.ReadU8 ();
  m_retryNo = rbuf.ReadU8 ();
  m_timeStampRts = Seconds ( ( (double) rbuf.ReadU32 ()) / 1000.0 );
  m_delay = Seconds ( ( (double) rbuf.ReadU32 ()) / 1000.0 );

  return rbuf.GetDistanceFrom (start);
}

void
UanHeaderRcCts::Print (std::ostream &os, Time::Unit unit) const
{
  os << "CTS (Addr=" << m_address << " Frame #=" << (uint32_t) m_frameNo << " Retry #=" << (uint32_t) m_retryNo << " RTS Rx Timestamp=" << m_timeStampRts.As (unit) << " Delay until TX=" << m_delay.As (unit) << ")";
}

void
UanHeaderRcCts::Print (std::ostream &os) const
{
  Print (os, Time::S);
}

TypeId
UanHeaderRcCts::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

UanHeaderRcAck::UanHeaderRcAck ()
  : m_frameNo (0)
{
}

UanHeaderRcAck::~UanHeaderRcAck ()
{
  m_nackedFrames.clear ();
}

TypeId
UanHeaderRcAck::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::UanHeaderRcAck")
    .SetParent<Header> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanHeaderRcAck> ()
  ;
  return tid;
}

void
UanHeaderRcAck::SetFrameNo (uint8_t noFrames)
{
  m_frameNo = noFrames;
}

void
UanHeaderRcAck::AddNackedFrame (uint8_t frame)
{
  m_nackedFrames.insert (frame);
}

const std::set<uint8_t> &
UanHeaderRcAck::GetNackedFrames (void) const
{
  return m_nackedFrames;
}

uint8_t
UanHeaderRcAck::GetFrameNo (void) const
{
  return m_frameNo;
}

uint8_t
UanHeaderRcAck::GetNoNacks (void) const
{
  return static_cast<uint8_t> (m_nackedFrames.size ());
}

uint32_t
UanHeaderRcAck::GetSerializedSize (void) const
{
  return 1 + 1 + GetNoNacks ();
}

void
UanHeaderRcAck::Serialize (Buffer::Iterator start) const
{
  start.WriteU8 (m_frameNo);
  start.WriteU8 (GetNoNacks ());
  std::set<uint8_t>::iterator it = m_nackedFrames.begin ();
  for (; it != m_nackedFrames.end (); it++)
    {
      start.WriteU8 (*it);
    }
}

uint32_t
UanHeaderRcAck::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator rbuf = start;
  m_frameNo = rbuf.ReadU8 ();
  uint8_t noAcks = rbuf.ReadU8 ();
  m_nackedFrames.clear ();
  for (uint32_t i = 0; i < noAcks; i++)
    {
      m_nackedFrames.insert (rbuf.ReadU8 ());
    }
  return rbuf.GetDistanceFrom (start);
}

void
UanHeaderRcAck::Print (std::ostream &os) const
{
  os << "# Frames=" << (uint32_t) m_frameNo << " # nacked=" << (uint32_t) GetNoNacks () << " Nacked: ";
  if (GetNoNacks () > 0)
    {
      std::set<uint8_t>::iterator it = m_nackedFrames.begin ();
      os << (uint32_t) *it;
      it++;
      for (; it != m_nackedFrames.end (); it++)
        {
          os << ", " << (uint32_t) *it;
        }
    }
}

TypeId
UanHeaderRcAck::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

} // namespace ns3
