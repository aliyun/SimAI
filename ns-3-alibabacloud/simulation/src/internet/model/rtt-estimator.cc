/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Rajib Bhattacharjea<raj.b@gatech.edu>
//

// Ported from:
// Georgia Tech Network Simulator - Round Trip Time Estimator Class
// George F. Riley.  Georgia Tech, Spring 2002

// Base class allows variations of round trip time estimators to be
// implemented

#include <iostream>
#include <cmath>

#include "rtt-estimator.h"
#include "ns3/double.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RttEstimator");

NS_OBJECT_ENSURE_REGISTERED (RttEstimator);

/// Tolerance used to check reciprocal of two numbers.
static const double TOLERANCE = 1e-6;

TypeId 
RttEstimator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RttEstimator")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddAttribute ("InitialEstimation", 
                   "Initial RTT estimate",
                   TimeValue (Seconds (1.0)),
                   MakeTimeAccessor (&RttEstimator::m_initialEstimatedRtt),
                   MakeTimeChecker ())
  ;
  return tid;
}

Time
RttEstimator::GetEstimate (void) const
{
  return m_estimatedRtt;
}

Time 
RttEstimator::GetVariation (void) const
{
  return m_estimatedVariation;
}


// Base class methods

RttEstimator::RttEstimator ()
  : m_nSamples (0)
{ 
  NS_LOG_FUNCTION (this);
  
  // We need attributes initialized here, not later, so use the 
  // ConstructSelf() technique documented in the manual
  ObjectBase::ConstructSelf (AttributeConstructionList ());
  m_estimatedRtt = m_initialEstimatedRtt;
  m_estimatedVariation = Time (0);
  NS_LOG_DEBUG ("Initialize m_estimatedRtt to " << m_estimatedRtt.GetSeconds () << " sec.");
}

RttEstimator::RttEstimator (const RttEstimator& c)
  : Object (c),
    m_initialEstimatedRtt (c.m_initialEstimatedRtt),
    m_estimatedRtt (c.m_estimatedRtt),
    m_estimatedVariation (c.m_estimatedVariation),
    m_nSamples (c.m_nSamples)
{
  NS_LOG_FUNCTION (this);
}

RttEstimator::~RttEstimator ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
RttEstimator::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void RttEstimator::Reset ()
{ 
  NS_LOG_FUNCTION (this);
  // Reset to initial state
  m_estimatedRtt = m_initialEstimatedRtt;
  m_estimatedVariation = Time (0);
  m_nSamples = 0;
}

uint32_t 
RttEstimator::GetNSamples (void) const
{
  return m_nSamples;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Mean-Deviation Estimator

NS_OBJECT_ENSURE_REGISTERED (RttMeanDeviation);

TypeId 
RttMeanDeviation::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RttMeanDeviation")
    .SetParent<RttEstimator> ()
    .SetGroupName ("Internet")
    .AddConstructor<RttMeanDeviation> ()
    .AddAttribute ("Alpha",
                   "Gain used in estimating the RTT, must be 0 <= alpha <= 1",
                   DoubleValue (0.125),
                   MakeDoubleAccessor (&RttMeanDeviation::m_alpha),
                   MakeDoubleChecker<double> (0, 1))
    .AddAttribute ("Beta",
                   "Gain used in estimating the RTT variation, must be 0 <= beta <= 1",
                   DoubleValue (0.25),
                   MakeDoubleAccessor (&RttMeanDeviation::m_beta),
                   MakeDoubleChecker<double> (0, 1))
  ;
  return tid;
}

RttMeanDeviation::RttMeanDeviation()
{
  NS_LOG_FUNCTION (this);
}

RttMeanDeviation::RttMeanDeviation (const RttMeanDeviation& c)
  : RttEstimator (c), m_alpha (c.m_alpha), m_beta (c.m_beta)
{
  NS_LOG_FUNCTION (this);
}

TypeId
RttMeanDeviation::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
RttMeanDeviation::CheckForReciprocalPowerOfTwo (double val) const
{
  NS_LOG_FUNCTION (this << val);
  if (val < TOLERANCE)
    {
      return 0;
    }
  // supports 1/32, 1/16, 1/8, 1/4, 1/2
  if (std::abs (1/val - 8) < TOLERANCE)
    {
      return 3;
    }
  if (std::abs (1/val - 4) < TOLERANCE)
    {
      return 2;
    }
  if (std::abs (1/val - 32) < TOLERANCE)
    {
      return 5;
    }
  if (std::abs (1/val - 16) < TOLERANCE)
    {
      return 4;
    }
  if (std::abs (1/val - 2) < TOLERANCE)
    {
      return 1;
    }
  return 0;
}

void
RttMeanDeviation::FloatingPointUpdate (Time m)
{
  NS_LOG_FUNCTION (this << m);

  // EWMA formulas are implemented as suggested in
  // Jacobson/Karels paper appendix A.2

  // SRTT <- (1 - alpha) * SRTT + alpha *  R'
  Time err (m - m_estimatedRtt);
  double gErr = err.ToDouble (Time::S) * m_alpha;
  m_estimatedRtt += Time::FromDouble (gErr, Time::S);

  // RTTVAR <- (1 - beta) * RTTVAR + beta * |SRTT - R'|
  Time difference = Abs (err) - m_estimatedVariation;
  m_estimatedVariation += difference * m_beta;
  return;
}

void
RttMeanDeviation::IntegerUpdate (Time m, uint32_t rttShift, uint32_t variationShift)
{
  NS_LOG_FUNCTION (this << m << rttShift << variationShift);
  // Jacobson/Karels paper appendix A.2
  int64_t meas = m.GetInteger ();
  int64_t delta = meas - m_estimatedRtt.GetInteger ();
  int64_t srtt = (m_estimatedRtt.GetInteger () << rttShift) + delta;
  m_estimatedRtt = Time::From (srtt >> rttShift);
  if (delta < 0)
    {
      delta = -delta;
    }
  delta -= m_estimatedVariation.GetInteger ();
  int64_t rttvar = m_estimatedVariation.GetInteger () << variationShift;
  rttvar += delta;
  m_estimatedVariation = Time::From (rttvar >> variationShift);
  return;
}

void 
RttMeanDeviation::Measurement (Time m)
{
  NS_LOG_FUNCTION (this << m);
  if (m_nSamples)
    { 
      // If both alpha and beta are reciprocal powers of two, updating can
      // be done with integer arithmetic according to Jacobson/Karels paper.
      // If not, since class Time only supports integer multiplication,
      // must convert Time to floating point and back again
      uint32_t rttShift = CheckForReciprocalPowerOfTwo (m_alpha);
      uint32_t variationShift = CheckForReciprocalPowerOfTwo (m_beta);
      if (rttShift && variationShift)
        {
          IntegerUpdate (m, rttShift, variationShift);
        }
      else
        {
          FloatingPointUpdate (m);
        }
    }
  else
    { // First sample
      m_estimatedRtt = m;               // Set estimate to current
      m_estimatedVariation = m / 2;  // And variation to current / 2
      NS_LOG_DEBUG ("(first sample) m_estimatedVariation += " << m);
    }
  m_nSamples++;
}

Ptr<RttEstimator> 
RttMeanDeviation::Copy () const
{
  NS_LOG_FUNCTION (this);
  return CopyObject<RttMeanDeviation> (this);
}

void 
RttMeanDeviation::Reset ()
{ 
  NS_LOG_FUNCTION (this);
  RttEstimator::Reset ();
}

} //namespace ns3
