/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Magister Solutions
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
 * Author: Budiarto Herman <budiarto.herman@magister.fi>
 *
 */

#include "three-gpp-http-variables.h"

#include <ns3/log.h>
#include <ns3/uinteger.h>
#include <ns3/double.h>
#include <math.h>


NS_LOG_COMPONENT_DEFINE ("ThreeGppHttpVariables");

namespace ns3 {


NS_OBJECT_ENSURE_REGISTERED (ThreeGppHttpVariables);


ThreeGppHttpVariables::ThreeGppHttpVariables ()
{
  NS_LOG_FUNCTION (this);
  m_mtuSizeRng = CreateObject<UniformRandomVariable> ();
  m_requestSizeRng = CreateObject<ConstantRandomVariable> ();
  m_mainObjectGenerationDelayRng = CreateObject<ConstantRandomVariable> ();
  m_mainObjectSizeRng = CreateObject<LogNormalRandomVariable> ();
  m_embeddedObjectGenerationDelayRng = CreateObject<ConstantRandomVariable> ();
  m_embeddedObjectSizeRng = CreateObject<LogNormalRandomVariable> ();
  m_numOfEmbeddedObjectsRng = CreateObject<ParetoRandomVariable> ();
  m_readingTimeRng = CreateObject<ExponentialRandomVariable> ();
  m_parsingTimeRng = CreateObject<ExponentialRandomVariable> ();
}


// static
TypeId
ThreeGppHttpVariables::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::ThreeGppHttpVariables")
    .SetParent<Object> ()
    .AddConstructor<ThreeGppHttpVariables> ()

    // REQUEST SIZE
    .AddAttribute ("RequestSize",
                   "The constant size of HTTP request packet (in bytes).",
                   UintegerValue (328),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetRequestSize),
                   MakeUintegerChecker<uint32_t> ())

    // MAIN OBJECT GENERATION DELAY
    .AddAttribute ("MainObjectGenerationDelay",
                   "The constant time needed by HTTP server "
                   "to generate a main object as a response.",
                   TimeValue (MilliSeconds (0)),
                   MakeTimeAccessor (&ThreeGppHttpVariables::SetMainObjectGenerationDelay),
                   MakeTimeChecker ())

    // MAIN OBJECT SIZE
    .AddAttribute ("MainObjectSizeMean",
                   "The mean of main object sizes (in bytes).",
                   UintegerValue (10710),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetMainObjectSizeMean),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MainObjectSizeStdDev",
                   "The standard deviation of main object sizes (in bytes).",
                   UintegerValue (25032),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetMainObjectSizeStdDev),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MainObjectSizeMin",
                   "The minimum value of main object sizes (in bytes).",
                   UintegerValue (100),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::m_mainObjectSizeMin),
                   MakeUintegerChecker<uint32_t> (22))
    .AddAttribute ("MainObjectSizeMax",
                   "The maximum value of main object sizes (in bytes).",
                   UintegerValue (2000000), // 2 MB
                   MakeUintegerAccessor (&ThreeGppHttpVariables::m_mainObjectSizeMax),
                   MakeUintegerChecker<uint32_t> ())

    // EMBEDDED OBJECT GENERATION DELAY
    .AddAttribute ("EmbeddedObjectGenerationDelay",
                   "The constant time needed by HTTP server "
                   "to generate an embedded object as a response.",
                   TimeValue (MilliSeconds (0)),
                   MakeTimeAccessor (&ThreeGppHttpVariables::SetEmbeddedObjectGenerationDelay),
                   MakeTimeChecker ())

    // EMBEDDED OBJECT SIZE
    .AddAttribute ("EmbeddedObjectSizeMean",
                   "The mean of embedded object sizes (in bytes).",
                   UintegerValue (7758),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetEmbeddedObjectSizeMean),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("EmbeddedObjectSizeStdDev",
                   "The standard deviation of embedded object sizes (in bytes).",
                   UintegerValue (126168),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetEmbeddedObjectSizeStdDev),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("EmbeddedObjectSizeMin",
                   "The minimum value of embedded object sizes (in bytes).",
                   UintegerValue (50),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::m_embeddedObjectSizeMin),
                   MakeUintegerChecker<uint32_t> (22))
    .AddAttribute ("EmbeddedObjectSizeMax",
                   "The maximum value of embedded object sizes (in bytes).",
                   UintegerValue (2000000), // 2 MB
                   MakeUintegerAccessor (&ThreeGppHttpVariables::m_embeddedObjectSizeMax),
                   MakeUintegerChecker<uint32_t> ())

    // NUMBER OF EMBEDDED OBJECTS PER PAGE
    .AddAttribute ("NumOfEmbeddedObjectsMax",
                   "The upper bound parameter of Pareto distribution for "
                   "the number of embedded objects per web page. The actual "
                   "maximum value is this value subtracted by the scale parameter.",
                   UintegerValue (55),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetNumOfEmbeddedObjectsMax),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("NumOfEmbeddedObjectsShape",
                   "The shape parameter of Pareto distribution for "
                   "the number of embedded objects per web page.",
                   DoubleValue (1.1),
                   MakeDoubleAccessor (&ThreeGppHttpVariables::SetNumOfEmbeddedObjectsShape),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("NumOfEmbeddedObjectsScale",
                   "The scale parameter of Pareto distribution for "
                   "the number of embedded objects per web page.",
                   UintegerValue (2),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::SetNumOfEmbeddedObjectsScale),
                   MakeUintegerChecker<uint32_t> ())

    // READING TIME
    .AddAttribute ("ReadingTimeMean",
                   "The mean of reading time.",
                   TimeValue (Seconds (30)),
                   MakeTimeAccessor (&ThreeGppHttpVariables::SetReadingTimeMean),
                   MakeTimeChecker ())

    // PARSING TIME
    .AddAttribute ("ParsingTimeMean",
                   "The mean of parsing time.",
                   TimeValue (MilliSeconds (130)),
                   MakeTimeAccessor (&ThreeGppHttpVariables::SetParsingTimeMean),
                   MakeTimeChecker ())

    // MTU SIZE
    .AddAttribute ("LowMtuSize",
                   "The lower MTU size.",
                   UintegerValue (536),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::m_lowMtu),
                   MakeUintegerChecker<uint32_t> (0))
    .AddAttribute ("HighMtuSize",
                   "The higher MTU size.",
                   UintegerValue (1460),
                   MakeUintegerAccessor (&ThreeGppHttpVariables::m_highMtu),
                   MakeUintegerChecker<uint32_t> (0))
    .AddAttribute ("HighMtuProbability",
                   "The probability that higher MTU size is used.",
                   DoubleValue (0.76),
                   MakeDoubleAccessor (&ThreeGppHttpVariables::m_highMtuProbability),
                   MakeDoubleChecker<double> (0, 1))
  ;
  return tid;

}


uint32_t
ThreeGppHttpVariables::GetMtuSize ()
{
  const double r = m_mtuSizeRng->GetValue ();
  NS_ASSERT (r >= 0.0);
  NS_ASSERT (r < 1.0);
  if (r < m_highMtuProbability)
    {
      return m_highMtu; // 1500 bytes if including TCP header.
    }
  else
    {
      return m_lowMtu; // 576 bytes if including TCP header.
    }
}


uint32_t
ThreeGppHttpVariables::GetRequestSize ()
{
  return m_requestSizeRng->GetInteger ();
}


Time
ThreeGppHttpVariables::GetMainObjectGenerationDelay ()
{
  return Seconds (m_mainObjectGenerationDelayRng->GetValue ());
}


uint32_t
ThreeGppHttpVariables::GetMainObjectSize ()
{
  // Validate parameters.
  if (m_mainObjectSizeMax <= m_mainObjectSizeMin)
    {
      NS_FATAL_ERROR ("`MainObjectSizeMax` attribute "
                      << " must be greater than"
                      << " the `MainObjectSizeMin` attribute.");
    }

  /*
   * Repeatedly draw one new random value until it falls in the interval
   * [min, max). The previous validation ensures this process does not loop
   * indefinitely.
   */
  uint32_t value;
  do
    {
      value = m_mainObjectSizeRng->GetInteger ();
    }
  while ((value < m_mainObjectSizeMin) || (value >= m_mainObjectSizeMax));

  return value;
}


Time
ThreeGppHttpVariables::GetEmbeddedObjectGenerationDelay ()
{
  return Seconds (m_embeddedObjectGenerationDelayRng->GetValue ());
}


uint32_t
ThreeGppHttpVariables::GetEmbeddedObjectSize ()
{
  // Validate parameters.
  if (m_embeddedObjectSizeMax <= m_embeddedObjectSizeMin)
    {
      NS_FATAL_ERROR ("`EmbeddedObjectSizeMax` attribute "
                      << " must be greater than"
                      << " the `EmbeddedObjectSizeMin` attribute.");
    }

  /*
   * Repeatedly draw one new random value until it falls in the interval
   * [min, max). The previous validation ensures this process does not loop
   * indefinitely.
   */
  uint32_t value;
  do
    {
      value = m_embeddedObjectSizeRng->GetInteger ();
    }
  while ((value < m_embeddedObjectSizeMin) || (value >= m_embeddedObjectSizeMax));

  return value;
}


uint32_t
ThreeGppHttpVariables::GetNumOfEmbeddedObjects ()
{
  // Validate parameters.
  const uint32_t upperBound =
    static_cast<uint32_t> (m_numOfEmbeddedObjectsRng->GetBound ());
  if (upperBound <= m_numOfEmbeddedObjectsScale)
    {
      NS_FATAL_ERROR ("`NumOfEmbeddedObjectsMax` attribute "
                      << " must be greater than"
                      << " the `NumOfEmbeddedObjectsScale` attribute.");
    }

  /*
   * Repeatedly draw one new random value until it falls in the interval
   * [scale, upperBound). The previous validation ensures this process does
   * not loop indefinitely.
   */
  uint32_t value;
  do
    {
      value = m_numOfEmbeddedObjectsRng->GetInteger ();
    }
  while ((value < m_numOfEmbeddedObjectsScale) || (value >= upperBound));

  /*
   * Normalize the random value with the scale parameter. The returned value
   * shall now be within the interval [0, (upperBound - scale)).
   */
  return (value - m_numOfEmbeddedObjectsScale);
}


Time
ThreeGppHttpVariables::GetReadingTime ()
{
  return Seconds (m_readingTimeRng->GetValue ());
}


Time
ThreeGppHttpVariables::GetParsingTime ()
{
  return Seconds (m_parsingTimeRng->GetValue ());
}


int64_t
ThreeGppHttpVariables::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);

  m_mtuSizeRng->SetStream (stream);
  m_requestSizeRng->SetStream (stream + 1);
  m_mainObjectGenerationDelayRng->SetStream (stream + 2);
  m_mainObjectSizeRng->SetStream (stream + 3);
  m_embeddedObjectGenerationDelayRng->SetStream (stream + 4);
  m_embeddedObjectSizeRng->SetStream (stream + 5);
  m_numOfEmbeddedObjectsRng->SetStream (stream + 6);
  m_readingTimeRng->SetStream (stream + 7);
  m_parsingTimeRng->SetStream (stream + 8);

  return 9;
}

void
ThreeGppHttpVariables::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  UpdateMainObjectMuAndSigma ();
  UpdateEmbeddedObjectMuAndSigma ();
}

// SETTER METHODS /////////////////////////////////////////////////////////////


void
ThreeGppHttpVariables::SetRequestSize (uint32_t constant)
{
  NS_LOG_FUNCTION (this << constant);
  m_requestSizeRng->SetAttribute ("Constant",
                                  DoubleValue (static_cast<double> (constant)));
}


void
ThreeGppHttpVariables::SetMainObjectGenerationDelay (Time constant)
{
  NS_LOG_FUNCTION (this << constant.As (Time::S));
  m_mainObjectGenerationDelayRng->SetAttribute ("Constant",
                                                DoubleValue (constant.GetSeconds ()));
}

void
ThreeGppHttpVariables::UpdateMainObjectMuAndSigma (void)
{
  NS_LOG_FUNCTION (this);
  const double a1 = std::pow (m_mainObjectSizeStdDev, 2.0);
  const double a2 = std::pow (m_mainObjectSizeMean, 2.0);
  const double a = std::log (1.0 + (a1 / a2));
  const double mu = std::log (m_mainObjectSizeMean) - (0.5 * a);
  const double sigma = std::sqrt (a);
  NS_LOG_DEBUG (this << " Mu= " << mu << " Sigma= " << sigma << ".");
  m_mainObjectSizeRng->SetAttribute ("Mu", DoubleValue (mu));
  m_mainObjectSizeRng->SetAttribute ("Sigma", DoubleValue (sigma));
}

void
ThreeGppHttpVariables::UpdateEmbeddedObjectMuAndSigma (void)
{
  NS_LOG_FUNCTION (this);
  const double a1 = std::pow (m_embeddedObjectSizeStdDev, 2.0);
  const double a2 = std::pow (m_embeddedObjectSizeMean, 2.0);
  const double a = std::log (1.0 + (a1 / a2));
  const double mu = std::log (m_embeddedObjectSizeMean) - (0.5 * a);
  const double sigma = std::sqrt (a);
  NS_LOG_DEBUG (this << " Mu= " << mu << " Sigma= " << sigma << ".");
  m_embeddedObjectSizeRng->SetAttribute ("Mu", DoubleValue (mu));
  m_embeddedObjectSizeRng->SetAttribute ("Sigma", DoubleValue (sigma));
}

void
ThreeGppHttpVariables::SetMainObjectSizeMean (uint32_t mean)
{
  NS_LOG_FUNCTION (this << mean);
  NS_ASSERT_MSG (mean > 0, "Mean must be greater than zero.");
  m_mainObjectSizeMean = mean;

  if (IsInitialized ())
    {
      UpdateMainObjectMuAndSigma ();
    }
}


void
ThreeGppHttpVariables::SetMainObjectSizeStdDev (uint32_t stdDev)
{
  NS_LOG_FUNCTION (this << stdDev);
  m_mainObjectSizeStdDev = stdDev;
  
  if (IsInitialized ())
    {
      UpdateMainObjectMuAndSigma ();
    }
}


void
ThreeGppHttpVariables::SetEmbeddedObjectGenerationDelay (Time constant)
{
  NS_LOG_FUNCTION (this << constant.As (Time::S));
  m_embeddedObjectGenerationDelayRng->SetAttribute ("Constant",
                                                    DoubleValue (constant.GetSeconds ()));
}


void
ThreeGppHttpVariables::SetEmbeddedObjectSizeMean (uint32_t mean)
{
  NS_LOG_FUNCTION (this << mean);
  NS_ASSERT_MSG (mean > 0, "Mean must be greater than zero.");
  m_embeddedObjectSizeMean = mean;

  if (IsInitialized ())
    {
      UpdateEmbeddedObjectMuAndSigma ();
    }
}


void
ThreeGppHttpVariables::SetEmbeddedObjectSizeStdDev (uint32_t stdDev)
{
  NS_LOG_FUNCTION (this << stdDev);
  m_embeddedObjectSizeStdDev = stdDev;

  if (IsInitialized ())
    {
      UpdateEmbeddedObjectMuAndSigma ();
    }
}


void
ThreeGppHttpVariables::SetNumOfEmbeddedObjectsMax (uint32_t max)
{
  NS_LOG_FUNCTION (this << max);
  m_numOfEmbeddedObjectsRng->SetAttribute ("Bound",
                                           DoubleValue (static_cast<double> (max)));
}


void
ThreeGppHttpVariables::SetNumOfEmbeddedObjectsShape (double shape)
{
  NS_LOG_FUNCTION (this << shape);
  NS_ASSERT_MSG (std::fabs (shape - 1.0) > 0.000001,
                 "Shape parameter must not equal to 1.0.");
  m_numOfEmbeddedObjectsRng->SetAttribute ("Shape", DoubleValue (shape));
}


void
ThreeGppHttpVariables::SetNumOfEmbeddedObjectsScale (uint32_t scale)
{
  NS_LOG_FUNCTION (this << scale);
  NS_ASSERT_MSG (scale > 0, "Scale parameter must be greater than zero.");
  m_numOfEmbeddedObjectsScale = scale;
  m_numOfEmbeddedObjectsRng->SetAttribute ("Scale", DoubleValue (scale));
}


void
ThreeGppHttpVariables::SetReadingTimeMean (Time mean)
{
  NS_LOG_FUNCTION (this << mean.As (Time::S));
  m_readingTimeRng->SetAttribute ("Mean", DoubleValue (mean.GetSeconds ()));
}


void
ThreeGppHttpVariables::SetParsingTimeMean (Time mean)
{
  NS_LOG_FUNCTION (this << mean.As (Time::S));
  m_parsingTimeRng->SetAttribute ("Mean", DoubleValue (mean.GetSeconds ()));
}


} // end of `namespace ns3`
