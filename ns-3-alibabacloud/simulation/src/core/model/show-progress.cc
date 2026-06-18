/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Lawrence Livermore National Laboratory
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
 * Author: Gustavo Carneiro <gjc@inescporto.pt>
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

/**
 * \file
 * \ingroup core
 * ns3::ShowProgress implementation.
 */

#include "show-progress.h"
#include "event-id.h"
#include "log.h"
#include "nstime.h"
#include "simulator.h"

#include <iomanip>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ShowProgress");

/* static */
const int64x64_t ShowProgress::HYSTERESIS = 1.414;
/* static */
const int64x64_t ShowProgress::MAXGAIN = 2.0;
  
ShowProgress::ShowProgress (const Time interval /* = Seconds (1.0) */,
                            std::ostream & os /* = std::cout */)
  : m_timer (),
    m_stamp (),
    m_elapsed (),
    m_interval (interval),
    m_vtime (Time (1)),
    m_event (),
    m_eventCount (0),
    m_printer (DefaultTimePrinter),
    m_os (&os),
    m_verbose (false),
    m_repCount (0)
{
  NS_LOG_FUNCTION (this << interval);
  ScheduleCheckProgress ();
  Start ();
}


ShowProgress::~ShowProgress (void)
{
  Stop ();
}


void
ShowProgress::SetInterval (const Time interval)
{
  NS_LOG_FUNCTION (this << interval);
  const int64x64_t ratio = interval  / m_interval;
  m_interval = interval;
  // If we aren't at the initial value assume we have a reasonable
  // update time m_vtime, so we should rescale it
  if (m_vtime > Time (1))
    {
      m_vtime = m_vtime * ratio;
    }
  Simulator::Cancel (m_event);
  Start ();

}  // ShowProgress::SetInterval

void
ShowProgress::SetTimePrinter (TimePrinter lp)
{
  NS_LOG_FUNCTION (this << lp);
  m_printer = lp;
}

void
ShowProgress::SetVerbose (bool verbose)
{
  NS_LOG_FUNCTION (this << verbose);
  m_verbose = verbose;
}

void
ShowProgress::SetStream (std::ostream & os)
{
  m_os = &os;
}

void
ShowProgress::ScheduleCheckProgress (void)
{
  NS_LOG_FUNCTION (this);
  m_event = Simulator::Schedule (m_vtime, &ShowProgress::CheckProgress, this);
  m_timer.Start ();

}  // ShowProgress::ScheduleCheckProgress

void
ShowProgress::GiveFeedback (uint64_t nEvents, int64x64_t ratio, int64x64_t speed)
{
  // Save stream state
  auto precision = m_os->precision ();
  auto flags = m_os->flags ();

  m_os->setf (std::ios::fixed, std:: ios::floatfield);

  if (m_verbose)
    {
      (*m_os) << std::right << std::setw (5) << m_repCount << std::left
              << (ratio > (1.0 / HYSTERESIS) ? "-->" : "   ")
              << std::setprecision (9) 
              << " [del: " << m_elapsed.As (Time::S)
              << "/ int: " << m_interval.As (Time::S)
              << " = rat: " << ratio
              << (ratio > HYSTERESIS ? " dn" :
          (ratio < 1.0 / HYSTERESIS ? " up" : " --"))
              << ", vt: " << m_vtime.As (Time::S) << "] ";
    }

  // Print the current time
  (*m_printer)(*m_os);

  (*m_os) << " ("
          << std::setprecision (3) << std::setw (8) << speed.GetDouble () << "x real time) "
          << nEvents << " events processed"
          << std::endl
          << std::flush;

  // Restore stream state
  m_os->precision (precision);
  m_os->flags (flags);

}  // ShowProgress::GiveFeedback

void
ShowProgress::CheckProgress (void)
{
  // Get elapsed wall clock time
  m_elapsed += MilliSeconds (m_timer.End ());
  NS_LOG_FUNCTION (this << m_elapsed);

  // Don't do anything unless the elapsed time is positive.
  if (m_elapsed <= Time (0))
    {
      m_vtime = m_vtime * MAXGAIN;
      ++m_repCount;
      ScheduleCheckProgress ();
      return;
    }

  // Speed: how fast are we compared to real time
  const int64x64_t speed = m_vtime / m_elapsed;

  // Ratio: how much real time did we use,
  // compared to reporting interval target
  const int64x64_t ratio = m_elapsed / m_interval;

  // Elapsed event count
  uint64_t events = Simulator::GetEventCount ();
  uint64_t nEvents = events - m_eventCount;
  /**
   * \internal Update algorithm
   *
   * We steer \c m_vtime to obtain updates approximately every
   * \c m_interval in wall clock time.  To smooth things out a little
   * we impose a hysteresis band around \c m_interval where we
   * don't change \c m_vtime.  To avoid too rapid movements
   * chasing spikes or dips in execution rate, we bound the
   * change in \c m_vtime to a maximum factor.
   *
   * In mathematical terms, we compute the ratio of elapsed wall clock time
   * compared to the target reporting interval:
   * \f[ ratio = \frac{elapsed}{target interval)} \f]
   *
   * Graphically, the windows in ratio value and the corresponding
   * updates to \c m_vtime are sketched in this figure:
   * \verbatim
                      ^
                      |
              ratio   |   vtime update
                      |
                      |
                      |   /= MAXGAIN
                      |
            MAXGAIN  -|--------------    /= min (ratio, MAXGAIN)
                      |
                      |   /= ratio
                      |
         HYSTERESIS  -|=============================================
                      |
                      |
                      |
                   1 -|   No change
                      |
                      |
                      |
      1/ HYSTERESIS  -|==============================================
                      |
                      |   *= 1 / ratio
                      |
         1/ MAXGAIN  -|---------------   *=  min (1 / ratio, MAXGAIN)
                      |
                      |   *= MAXGAIN
                      |
     \endverbatim
   *
   * As indicated, when ratio is outside the hysteresis band
   * it amounts to multiplying \c m_vtime by the min/max of the ratio
   * with the appropriate MAXGAIN factor.
   *
   * Finally, some experimentation suggests we further dampen
   * movement between HYSTERESIS and MAXGAIN, so we only apply
   * half the ratio.  This reduces "hunting" for a stable update
   * period.
   *
   * \todo Evaluate if simple exponential averaging would be
   * more effective, simpler.
   */
  if (ratio > HYSTERESIS)
    {
      int64x64_t f = 1 + (ratio - 1) / 2;
      if (ratio > MAXGAIN)
        {
          f = MAXGAIN;
        }

      m_vtime = m_vtime / f;
    }
  else if (ratio < 1.0 / HYSTERESIS)
    {
      int64x64_t f = 1 + (1 / ratio - 1)/2;
      if (1 / ratio > MAXGAIN)
        {
          f = MAXGAIN;
        }
      m_vtime = m_vtime * f;
    }

  // Only give feedback if ratio is at least as big as 1/HYSTERESIS
  if (ratio > (1.0 / HYSTERESIS))
    {
      GiveFeedback (nEvents, ratio, speed);
      m_elapsed = Time (0);
      m_eventCount = events;
    }
  else
    {
      NS_LOG_LOGIC ("skipping update: " << ratio);
      // enable this line for debugging, with --verbose
      // GiveFeedback (nEvents, ratio, speed);
    }
  ++m_repCount;
  
  // And do it again
  ScheduleCheckProgress ();

}  // ShowProgress::CheckProgress

void
ShowProgress::Start (void)
{
  m_stamp.Stamp ();
  (*m_os) << "Start wall clock: " << m_stamp.ToString ()
          << std::endl;
}  // ShowProgress::Start

void
ShowProgress::Stop (void)
{
  m_stamp.Stamp ();
  (*m_os) << "End wall clock:  " << m_stamp.ToString ()
          << "\nElapsed wall clock: " << m_stamp.GetInterval () << "s"
          << std::endl;
}  // ShowProgress::Stop

}  // namespace ns3
