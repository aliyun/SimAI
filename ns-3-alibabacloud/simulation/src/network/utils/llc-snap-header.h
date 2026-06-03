/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#ifndef LLC_SNAP_HEADER_H
#define LLC_SNAP_HEADER_H

#include <stdint.h>
#include <string>
#include "ns3/header.h"

namespace ns3 {

/** 
 * The length in octects of the LLC/SNAP header
 */
static const uint16_t LLC_SNAP_HEADER_LENGTH = 8;

/**
 * \ingroup network
 *
 * \brief Header for the LLC/SNAP encapsulation
 *
 * For a list of EtherTypes, see http://www.iana.org/assignments/ieee-802-numbers/ieee-802-numbers.xhtml
 */
class LlcSnapHeader : public Header 
{
public:
  LlcSnapHeader ();

  /**
   * \brief Set the Ethertype.
   * \param type the Ethertype
   */
  void SetType (uint16_t type);
  /**
   * \brief Return the Ethertype.
   * \return Ethertype
   */
  uint16_t GetType (void);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
private:
  uint16_t m_etherType; //!< the Ethertype
};

} // namespace ns3

#endif /* LLC_SNAP_HEADER_H */
