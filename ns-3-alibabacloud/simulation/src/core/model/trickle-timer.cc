/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "trickle-timer.h"
#include "log.h"
#include <limits>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TrickleTimer");

TrickleTimer::TrickleTimer ()
  : m_impl (0),
    m_timerExpiration (),
    m_intervalExpiration (),
    m_currentInterval (Time(0)),
    m_counter (0),
    m_uniRand (CreateObject<UniformRandomVariable> ())
{
  NS_LOG_FUNCTION (this);

  m_minInterval = Time (0);
  m_ticks = 0;
  m_maxInterval = Time (0);
  m_redundancy = 0;
}

TrickleTimer::TrickleTimer (Time minInterval, uint8_t doublings, uint16_t redundancy)
  : m_impl (0),
    m_timerExpiration (),
    m_intervalExpiration (),
    m_currentInterval (Time(0)),
    m_counter (0),
    m_uniRand (CreateObject<UniformRandomVariable> ())
{
  NS_LOG_FUNCTION (this << minInterval << doublings << redundancy);
  NS_ASSERT_MSG (doublings < std::numeric_limits<decltype(m_ticks)>::digits, "Doublings value is too large");

  m_minInterval = minInterval;
  m_ticks = 1;
  m_ticks <<= doublings;
  m_maxInterval = m_ticks * minInterval;
  m_redundancy = redundancy;
}

TrickleTimer::~TrickleTimer ()
{
  NS_LOG_FUNCTION (this);
  m_timerExpiration.Cancel ();
  m_intervalExpiration.Cancel ();
  delete m_impl;
}

int64_t
TrickleTimer::AssignStreams (int64_t streamNum)
{
  m_uniRand->SetStream (streamNum);
  return 1;
}

void
TrickleTimer::SetParameters (Time minInterval, uint8_t doublings, uint16_t redundancy)
{
  NS_LOG_FUNCTION (this << minInterval << doublings << redundancy);
  NS_ASSERT_MSG (doublings < std::numeric_limits<decltype(m_ticks)>::digits, "Doublings value is too large");

  m_minInterval = minInterval;
  m_ticks = 1;
  m_ticks <<= doublings;
  m_maxInterval = m_ticks * minInterval;
  m_redundancy = redundancy;
}

Time
TrickleTimer::GetMinInterval (void) const
{
  NS_LOG_FUNCTION (this);
  return m_minInterval;
}

Time
TrickleTimer::GetMaxInterval (void) const
{
  NS_LOG_FUNCTION (this);
  return m_maxInterval;
}

uint8_t
TrickleTimer::GetDoublings (void) const
{
  NS_LOG_FUNCTION (this);

  if (m_ticks == 0)
    {
      return 0;
    }

  // Here we assume that m_ticks is a power of 2.
  // This could have been way more elegant by using
  // std::countl_zero() defined in the <bit> header
  // which is c++20 - so not yet widely available.

  uint64_t ticks = m_ticks;
  uint8_t doublings = 0;
  while (ticks != 1)
    {
      ticks >>= 1;
      doublings ++;
    }

  return doublings;
}

uint16_t
TrickleTimer::GetRedundancy (void) const
{
  NS_LOG_FUNCTION (this);
  return m_redundancy;
}

Time
TrickleTimer::GetDelayLeft (void) const
{
  NS_LOG_FUNCTION (this);

  if (m_timerExpiration.IsRunning ())
    {
      return Simulator::GetDelayLeft (m_timerExpiration);
    }

  return TimeStep (0);
}

Time
TrickleTimer::GetIntervalLeft (void) const
{
  NS_LOG_FUNCTION (this);

  if (m_intervalExpiration.IsRunning ())
    {
      return Simulator::GetDelayLeft (m_intervalExpiration);
    }

  return TimeStep (0);
}


void
TrickleTimer::Enable ()
{
  NS_LOG_FUNCTION (this);

  uint64_t randomInt;
  double random;

  NS_ASSERT_MSG (m_minInterval != Time (0), "Timer not initialized");

  randomInt = m_uniRand->GetInteger (1, m_ticks);
  random = randomInt;
  if (randomInt < m_ticks)
    {
      random += m_uniRand->GetValue (0, 1);
    }

  m_currentInterval = m_minInterval * random;
  m_intervalExpiration = Simulator::Schedule (m_currentInterval, &TrickleTimer::IntervalExpire, this);

  m_counter = 0;

  Time timerExpitation = m_uniRand->GetValue (0.5, 1) * m_currentInterval;
  m_timerExpiration = Simulator::Schedule (timerExpitation, &TrickleTimer::TimerExpire, this);

  return;
}

void
TrickleTimer::ConsistentEvent ()
{
  NS_LOG_FUNCTION (this);
  m_counter ++;
}

void
TrickleTimer::InconsistentEvent ()
{
  NS_LOG_FUNCTION (this);
  if (m_currentInterval > m_minInterval)
    {
      Reset ();
    }
}

void
TrickleTimer::Reset ()
{
  NS_LOG_FUNCTION (this);

  m_currentInterval = m_minInterval;
  m_intervalExpiration.Cancel ();
  m_timerExpiration.Cancel ();

  m_intervalExpiration = Simulator::Schedule (m_currentInterval, &TrickleTimer::IntervalExpire, this);

  m_counter = 0;

  Time timerExpitation = m_uniRand->GetValue (0.5, 1) * m_currentInterval;
  m_timerExpiration = Simulator::Schedule (timerExpitation, &TrickleTimer::TimerExpire, this);

  return;
}

void
TrickleTimer::Stop ()
{
  NS_LOG_FUNCTION (this);

  m_currentInterval = m_minInterval;
  m_intervalExpiration.Cancel ();
  m_timerExpiration.Cancel ();
  m_counter = 0;
}

void
TrickleTimer::TimerExpire(void)
{
  NS_LOG_FUNCTION (this);

  if (m_counter < m_redundancy || m_redundancy == 0)
    {
      m_impl->Invoke ();
    }
}

void
TrickleTimer::IntervalExpire(void)
{
  NS_LOG_FUNCTION (this);

  m_currentInterval = m_currentInterval * 2;
  if (m_currentInterval > m_maxInterval)
    {
      m_currentInterval = m_maxInterval;
    }

  m_intervalExpiration = Simulator::Schedule (m_currentInterval, &TrickleTimer::IntervalExpire, this);

  m_counter = 0;

  Time timerExpitation = m_uniRand->GetValue (0.5, 1) * m_currentInterval;
  m_timerExpiration = Simulator::Schedule (timerExpitation, &TrickleTimer::TimerExpire, this);
}



} // namespace ns3

