/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef MESH_PEERING_PROTOCOL_H
#define MESH_PEERING_PROTOCOL_H

#include "ns3/mesh-information-element-vector.h"

namespace ns3
{
namespace dot11s
{

/**
 * \ingroup dot11s
 *
 * \brief Mesh Peering Protocol Identifier information element
 * Note that it does not permit to set any value besides zero
 * (corresponding to mesh peering management protocol)
 */
class IePeeringProtocol : public WifiInformationElement
{
public:
  IePeeringProtocol ();

  // Inherited from WifiInformationElement
  virtual WifiInformationElementId ElementId () const;
  virtual uint8_t GetInformationFieldSize () const;
  virtual void SerializeInformationField (Buffer::Iterator i) const;
  virtual uint8_t DeserializeInformationField (Buffer::Iterator i, uint8_t length);
  virtual void Print (std::ostream& os) const;

private:
  uint8_t m_protocol; ///< the protocol
};
} // namespace dot11s
} // namespace ns3
#endif
