/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
#include "watchdog.h"
#include "log.h"


/**
 * \file
 * \ingroup timer
 * ns3::Watchdog timer class implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Watchdog");

Watchdog::Watchdog ()
  : m_impl (0),
    m_event (),
    m_end (MicroSeconds (0))
{
  NS_LOG_FUNCTION_NOARGS ();
}

Watchdog::~Watchdog ()
{
  NS_LOG_FUNCTION (this);
  m_event.Cancel ();
  delete m_impl;
}

void
Watchdog::Ping (Time delay)
{
  NS_LOG_FUNCTION (this << delay);
  Time end = Simulator::Now () + delay;
  m_end = std::max (m_end, end);
  if (m_event.IsRunning ())
    {
      return;
    }
  m_event = Simulator::Schedule (m_end - Now (), &Watchdog::Expire, this);
}

void
Watchdog::Expire (void)
{
  NS_LOG_FUNCTION (this);
  if (m_end == Simulator::Now ())
    {
      m_impl->Invoke ();
    }
  else
    {
      m_event = Simulator::Schedule (m_end - Now (), &Watchdog::Expire, this);
    }
}

} // namespace ns3

