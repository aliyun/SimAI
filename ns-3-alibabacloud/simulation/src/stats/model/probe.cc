/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Authors: L. Felipe Perrone (perrone@bucknell.edu)
 *          Tiago G. Rodrigues (tgr002@bucknell.edu)
 */

#include "ns3/probe.h"
#include "ns3/object.h"
#include "ns3/simulator.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Probe");

NS_OBJECT_ENSURE_REGISTERED (Probe);

TypeId
Probe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Probe")
    .SetParent<DataCollectionObject> ()
    .SetGroupName ("Stats")
    .AddAttribute ("Start",
                   "Time data collection starts",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&Probe::m_start),
                   MakeTimeChecker ())
    .AddAttribute ("Stop",
                   "Time when data collection stops.  The special time value of 0 disables this attribute",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&Probe::m_stop),
                   MakeTimeChecker ())
  ;
  return tid;
}

Probe::Probe ()
{
  NS_LOG_FUNCTION (this);
}

Probe::~Probe ()
{
  NS_LOG_FUNCTION (this);
}

bool
Probe::IsEnabled (void) const
{
  return (DataCollectionObject::IsEnabled ()
          && Simulator::Now () >= m_start
          && (m_stop == Seconds (0) || Simulator::Now () < m_stop));

}

} // namespace ns3
