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
#include "ipv6-option-demux.h"
#include "ipv6-option.h"

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED (Ipv6OptionDemux);

TypeId Ipv6OptionDemux::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6OptionDemux")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddAttribute ("Options", "The set of IPv6 options registered with this demux.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&Ipv6OptionDemux::m_options),
                   MakeObjectVectorChecker<Ipv6Option> ())
  ;
  return tid;
}

Ipv6OptionDemux::Ipv6OptionDemux ()
{
}

Ipv6OptionDemux::~Ipv6OptionDemux ()
{
}

void Ipv6OptionDemux::DoDispose ()
{
  for (Ipv6OptionList_t::iterator it = m_options.begin (); it != m_options.end (); it++)
    {
      (*it)->Dispose ();
      *it = 0;
    }
  m_options.clear ();
  m_node = 0;
  Object::DoDispose ();
}

void Ipv6OptionDemux::SetNode (Ptr<Node> node)
{
  m_node = node;
}

void Ipv6OptionDemux::Insert (Ptr<Ipv6Option> option)
{
  m_options.push_back (option);
}

Ptr<Ipv6Option> Ipv6OptionDemux::GetOption (int optionNumber)
{
  for (Ipv6OptionList_t::iterator i = m_options.begin (); i != m_options.end (); ++i)
    {
      if ((*i)->GetOptionNumber () == optionNumber)
        {
          return *i;
        }
    }
  return 0;
}

void Ipv6OptionDemux::Remove (Ptr<Ipv6Option> option)
{
  m_options.remove (option);
}

} /* namespace ns3 */

