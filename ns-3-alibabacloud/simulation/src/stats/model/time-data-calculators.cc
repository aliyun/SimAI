/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Drexel University
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
 * Author: Joe Kopena (tjkopena@cs.drexel.edu)
 */

#include "ns3/log.h"
#include "ns3/nstime.h"

#include "time-data-calculators.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TimeDataCalculators");

//--------------------------------------------------------------
//----------------------------------------------
TimeMinMaxAvgTotalCalculator::TimeMinMaxAvgTotalCalculator()
{
  NS_LOG_FUNCTION (this);

  m_count = 0;
}
TimeMinMaxAvgTotalCalculator::~TimeMinMaxAvgTotalCalculator()
{
  NS_LOG_FUNCTION (this);
}
/* static */
TypeId
TimeMinMaxAvgTotalCalculator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TimeMinMaxAvgTotalCalculator")
    .SetParent<DataCalculator> ()
    .SetGroupName ("Stats")
    .AddConstructor<TimeMinMaxAvgTotalCalculator> ();
  return tid;
}
  
void
TimeMinMaxAvgTotalCalculator::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  DataCalculator::DoDispose ();
  // TimeMinMaxAvgTotalCalculator::DoDispose
}

void
TimeMinMaxAvgTotalCalculator::Update (const Time i)
{
  NS_LOG_FUNCTION (this << i);

  if (m_enabled) {
      if (m_count) {
          m_total += i;

          if (i < m_min)
            m_min = i;

          if (i > m_max)
            m_max = i;

        } else {
          m_min = i;
          m_max = i;
          m_total = i;
        }
      m_count++;

    }
  // end TimeMinMaxAvgTotalCalculator::Update
}
void
TimeMinMaxAvgTotalCalculator::Output (DataOutputCallback &callback) const
{
  NS_LOG_FUNCTION (this << &callback);

  callback.OutputSingleton (m_context, m_key + "-count", m_count);
  if (m_count > 0) {
      callback.OutputSingleton (m_context, m_key + "-total", m_total);
      callback.OutputSingleton (m_context, m_key + "-average", Time (m_total / m_count));
      callback.OutputSingleton (m_context, m_key + "-max", m_max);
      callback.OutputSingleton (m_context, m_key + "-min", m_min);
    }
  // end TimeMinMaxAvgTotalCalculator::Output
}
