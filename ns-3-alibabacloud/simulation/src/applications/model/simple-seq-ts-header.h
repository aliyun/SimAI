/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA
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

#ifndef SIMPLE_SEQ_TS_HEADER_H
#define SIMPLE_SEQ_TS_HEADER_H

#include "ns3/header.h"
#include "ns3/nstime.h"
#include "ns3/int-header.h"

namespace ns3 {
/**
 * \ingroup udpclientserver
 * \class SimpleSeqTsHeader
 * \brief Packet header for Udp client/server application
 * The header is made of a 32bits sequence number followed by
 * a 64bits time stamp.
 */
class SimpleSeqTsHeader : public Header
{
public:
  SimpleSeqTsHeader ();

  /**
   * \param seq the sequence number
   */
  void SetSeq (uint64_t seq);
  /**
   * \return the sequence number
   */
  uint64_t GetSeq (void) const;
  /**
   * \return the time stamp
   */
  Time GetTs (void) const;

  void SetPG (uint16_t pg);
  uint16_t GetPG () const;

  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  static uint32_t GetHeaderSize(void);
private:
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  uint64_t m_seq;
  uint16_t m_pg;
public:
  IntHeader ih;
};

} // namespace ns3

#endif /* SEQ_TS_HEADER_H */
