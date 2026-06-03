/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
 * Author: Jaume Nin <jaume.nin@cttc.es>
 */

#ifndef RLC_TAG_H
#define RLC_TAG_H

#include "ns3/packet.h"
#include "ns3/nstime.h"


namespace ns3 {

class Tag;

/**
 * Tag to calculate the per-PDU delay from eNb RLC to UE RLC
 */

class RlcTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId  GetTypeId (void);
  virtual TypeId  GetInstanceTypeId (void) const;

  /**
   * Create an empty RLC tag
   */
  RlcTag ();
  /**
   * Create an RLC tag with the given senderTimestamp 
   * \param senderTimestamp the time
   */
  RlcTag (Time senderTimestamp);

  virtual void  Serialize (TagBuffer i) const;
  virtual void  Deserialize (TagBuffer i);
  virtual uint32_t  GetSerializedSize () const;
  virtual void Print (std::ostream &os) const;

  /**
   * Get the instant when the RLC delivers the PDU to the MAC SAP provider
   * @return the sender timestamp
   */
  Time  GetSenderTimestamp (void) const
  {
    return m_senderTimestamp;
  }

  /**
   * Set the sender timestamp
   * @param senderTimestamp time stamp of the instant when the RLC delivers the PDU to the MAC SAP provider
   */
  void  SetSenderTimestamp (Time senderTimestamp)
  {
    this->m_senderTimestamp = senderTimestamp;
  }

private:
  Time m_senderTimestamp; ///< sender timestamp

};

} //namespace ns3

#endif /* RLC_TAG_H */
