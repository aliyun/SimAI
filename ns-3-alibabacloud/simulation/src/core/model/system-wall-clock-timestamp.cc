/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "system-wall-clock-timestamp.h"

/**
 * \file
 * \ingroup system
 * ns3::SystemWallClockTimestamp implementation.
 */

namespace ns3 {

SystemWallClockTimestamp::SystemWallClockTimestamp (void)
  : m_last (0),
    m_diff (0)
{
  Stamp ();
}

void  
SystemWallClockTimestamp::Stamp (void)
{
  std::time_t seconds  = std::time (NULL);
  m_diff = seconds - m_last;
  m_last = seconds;
}
  
std::string
SystemWallClockTimestamp::ToString (void) const
{
  std::string now = std::ctime ( &m_last );
  now.resize (now.length () - 1);  // trim trailing newline
  return now;
}

std::time_t
SystemWallClockTimestamp::GetLast (void) const
{
  return m_last;
}

std::time_t
SystemWallClockTimestamp::GetInterval (void) const
{
  return m_diff;
}

  

}  // namespace ns3

