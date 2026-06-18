/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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

#include "scheduler.h"
#include "assert.h"
#include "log.h"

/**
 * \file
 * \ingroup scheduler
 * ns3::Scheduler implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Scheduler");

NS_OBJECT_ENSURE_REGISTERED (Scheduler);

Scheduler::~Scheduler ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
Scheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Scheduler")
    .SetParent<Object> ()
    .SetGroupName ("Core")
  ;
  return tid;
}

} // namespace ns3
