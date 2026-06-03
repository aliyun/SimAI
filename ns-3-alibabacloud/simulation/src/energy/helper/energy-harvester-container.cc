/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
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
 * Author: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#include "energy-harvester-container.h"

#include "ns3/names.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EnergyHarvesterContainer");

NS_OBJECT_ENSURE_REGISTERED (EnergyHarvesterContainer);

TypeId
EnergyHarvesterContainer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EnergyHarvesterContainer")
    .SetParent<Object> ()
    .SetGroupName ("Energy")
    .AddConstructor<EnergyHarvesterContainer> ()
  ;
  return tid;
}

EnergyHarvesterContainer::EnergyHarvesterContainer ()
{
  NS_LOG_FUNCTION (this);
}

EnergyHarvesterContainer::~EnergyHarvesterContainer ()
{
  NS_LOG_FUNCTION (this);
}

EnergyHarvesterContainer::EnergyHarvesterContainer (Ptr<EnergyHarvester> harvester)
{
  NS_LOG_FUNCTION (this << harvester);
  NS_ASSERT (harvester != 0);
  m_harvesters.push_back (harvester);
}

EnergyHarvesterContainer::EnergyHarvesterContainer (std::string harvesterName)
{
  NS_LOG_FUNCTION (this << harvesterName);
  Ptr<EnergyHarvester> harvester = Names::Find<EnergyHarvester> (harvesterName);
  NS_ASSERT (harvester != 0);
  m_harvesters.push_back (harvester);
}

EnergyHarvesterContainer::EnergyHarvesterContainer (const EnergyHarvesterContainer &a,
                                                    const EnergyHarvesterContainer &b)
{
  NS_LOG_FUNCTION (this << &a << &b);
  *this = a;
  Add (b);
}

EnergyHarvesterContainer::Iterator
EnergyHarvesterContainer::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_harvesters.begin ();
}

EnergyHarvesterContainer::Iterator
EnergyHarvesterContainer::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_harvesters.end ();
}

uint32_t
EnergyHarvesterContainer::GetN (void) const
{
  NS_LOG_FUNCTION (this);
  return m_harvesters.size ();
}

Ptr<EnergyHarvester>
EnergyHarvesterContainer::Get (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return m_harvesters[i];
}

void
EnergyHarvesterContainer::Add (EnergyHarvesterContainer container)
{
  NS_LOG_FUNCTION (this << &container);
  for (Iterator i = container.Begin (); i != container.End (); i++)
  {
      m_harvesters.push_back (*i);
  }
}

void
EnergyHarvesterContainer::Add (Ptr<EnergyHarvester> harvester)
{
  NS_LOG_FUNCTION (this << harvester);
  NS_ASSERT (harvester != 0);
  m_harvesters.push_back (harvester);
}

void
EnergyHarvesterContainer::Add (std::string harvesterName)
{
  NS_LOG_FUNCTION (this << harvesterName);
  Ptr<EnergyHarvester> harvester = Names::Find<EnergyHarvester> (harvesterName);
  NS_ASSERT (harvester != 0);
  m_harvesters.push_back (harvester);
}

void
EnergyHarvesterContainer::Clear (void)
{
  NS_LOG_FUNCTION (this);
  m_harvesters.clear ();
}


/*
 * Private functions start here.
 */

void
EnergyHarvesterContainer::DoDispose (void)
{
  // call Object::Dispose for all EnergyHarvester objects
  for (std::vector< Ptr<EnergyHarvester> >::iterator i = m_harvesters.begin ();
       i != m_harvesters.end (); i++)
  {
      (*i)->Dispose ();
  }
  m_harvesters.clear ();
}

void
EnergyHarvesterContainer::DoInitialize (void)
{
  // call Object::Initialize for all EnergyHarvester objects
  for (std::vector< Ptr<EnergyHarvester> >::iterator i = m_harvesters.begin ();
       i != m_harvesters.end (); i++)
  {
      (*i)->Initialize ();
  }
}

} // namespace ns3
