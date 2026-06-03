/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: David Gross <gdavid.devel@gmail.com>
 */

#include <sstream>
#include "ns3/node.h"
#include "ns3/ptr.h"
#include "ns3/object-vector.h"
#include "ipv6-extension-demux.h"
#include "ipv6-extension.h"

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionDemux);

TypeId Ipv6ExtensionDemux::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionDemux")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddAttribute ("Extensions", "The set of IPv6 extensions registered with this demux.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&Ipv6ExtensionDemux::m_extensions),
                   MakeObjectVectorChecker<Ipv6Extension> ())
  ;
  return tid;
}

Ipv6ExtensionDemux::Ipv6ExtensionDemux ()
{
}

Ipv6ExtensionDemux::~Ipv6ExtensionDemux ()
{
}

void Ipv6ExtensionDemux::DoDispose ()
{
  for (Ipv6ExtensionList_t::iterator it = m_extensions.begin (); it != m_extensions.end (); it++)
    {
      (*it)->Dispose ();
      *it = 0;
    }
  m_extensions.clear ();
  m_node = 0;
  Object::DoDispose ();
}

void Ipv6ExtensionDemux::SetNode (Ptr<Node> node)
{
  m_node = node;
}

void Ipv6ExtensionDemux::Insert (Ptr<Ipv6Extension> extension)
{
  m_extensions.push_back (extension);
}

Ptr<Ipv6Extension> Ipv6ExtensionDemux::GetExtension (uint8_t extensionNumber)
{
  for (Ipv6ExtensionList_t::iterator i = m_extensions.begin (); i != m_extensions.end (); ++i)
    {
      if ((*i)->GetExtensionNumber () == extensionNumber)
        {
          return *i;
        }
    }
  return 0;
}

void Ipv6ExtensionDemux::Remove (Ptr<Ipv6Extension> extension)
{
  m_extensions.remove (extension);
}

} /* namespace ns3 */

