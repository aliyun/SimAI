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

#ifndef SYSTEM_WALL_CLOCK_TIMESTAMP_H
#define SYSTEM_WALL_CLOCK_TIMESTAMP_H

/**
 * \file
 * \ingroup system
 * ns3::SystemWallClockTimestamp declaration.
 */

#include <string>
#include <ctime>  // ctime, time_t


namespace ns3 {
  
/**
 * Utility class to record the difference between two wall-clock times.
 */
class SystemWallClockTimestamp
{
public:
  /** Constructor */
  SystemWallClockTimestamp (void);

  /** Record the current wall-clock time and delta since the last stamp(). */
  void Stamp (void);

  /**
   * Get the last time stamp as a string.
   * \return The last time stamp.
   */
  std::string ToString (void) const;

  /**
   * Get the last recorded raw value.
   * \returns The last time stamp recorded.
   */
  std::time_t GetLast (void) const;

  /**
   * Get the last recorded interval.
   * \returns The interval between the last two time stamps.
   */
  std::time_t GetInterval (void) const;
  
private:

  /** The last time stamp. */
  std::time_t m_last;
  
  /** Difference between the two previous time stamps. */
  std::time_t m_diff;    

};  // class SystemWallClockTimestamp

}  // namespace ns3

#endif  /* SYSTEM_WALL_CLOCK_TIMESTAMP_H */
