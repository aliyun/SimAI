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

#include "ns3/flame-installer.h"
#include "ns3/flame-protocol.h"
#include "ns3/mesh-wifi-interface-mac.h"

namespace ns3
{
using namespace flame;
NS_OBJECT_ENSURE_REGISTERED (FlameStack);
  
TypeId
FlameStack::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::FlameStack")
    .SetParent<MeshStack> ()
    .SetGroupName ("Mesh")
    .AddConstructor<FlameStack> ();
  return tid;
}
FlameStack::FlameStack ()
{
}
FlameStack::~FlameStack ()
{
}
void
FlameStack::DoDispose ()
{
}
bool
FlameStack::InstallStack (Ptr<MeshPointDevice> mp)
{
  Ptr<FlameProtocol> flame = CreateObject<FlameProtocol> ();
  return flame->Install (mp);
}
void
FlameStack::Report (const Ptr<MeshPointDevice> mp, std::ostream& os)
{
  mp->Report (os);
  /// \todo report flame counters
  Ptr<FlameProtocol> flame = mp->GetObject<FlameProtocol> ();
  NS_ASSERT (flame != 0);
  flame->Report (os);
}
void
FlameStack::ResetStats (const Ptr<MeshPointDevice> mp)
{
  mp->ResetStats ();
  /// \todo reset flame counters
  Ptr<FlameProtocol> flame = mp->GetObject<FlameProtocol> ();
  NS_ASSERT (flame != 0);

  flame->ResetStats ();
}
} // namespace ns3
