/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@cutebugs.net>
 */

#include "chunk.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (Chunk);

TypeId 
Chunk::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Chunk")
    .SetParent<ObjectBase> ()
    .SetGroupName("Network")
  ;
  return tid;
}

// This default implementation is provided for backward compatibility
// reasons.  Subclasses should implement this method themselves.
uint32_t
Chunk::Deserialize (Buffer::Iterator start, Buffer::Iterator end)
{
  return Deserialize (start);
}

} // namespace ns3
