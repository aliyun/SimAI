/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Telum (www.telum.ru)
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
 * Author: Kirill Andreev <andreev@telum.ru>, Alexander Sofronov <sofronov@telum.ru>
 */

#include "jakes-process.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "propagation-loss-model.h"
#include "jakes-propagation-loss-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("JakesProcess");

/// Represents a single oscillator
JakesProcess::Oscillator::Oscillator (std::complex<double> amplitude, double initialPhase, double omega) :
  m_amplitude (amplitude),
  m_phase (initialPhase),
  m_omega (omega)
{}

std::complex<double>
JakesProcess::Oscillator::GetValueAt (Time at) const
{
  return (m_amplitude * std::cos (at.GetSeconds () * m_omega + m_phase));
}

NS_OBJECT_ENSURE_REGISTERED (JakesProcess);

TypeId
JakesProcess::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::JakesProcess")
    .SetParent<Object> ()
    .SetGroupName ("Propagation")
    .AddConstructor<JakesProcess> ()
    .AddAttribute ("DopplerFrequencyHz", "Corresponding doppler frequency[Hz]",
                   DoubleValue (80),
                   MakeDoubleAccessor (&JakesProcess::SetDopplerFrequencyHz),
                   MakeDoubleChecker<double> (0.0, 1e4))
    .AddAttribute ("NumberOfOscillators", "The number of oscillators",
                   UintegerValue (20),
                   MakeUintegerAccessor (&JakesProcess::SetNOscillators),
                   MakeUintegerChecker<unsigned int> (4, 1000))
  ;
  return tid;
}

void
JakesProcess::SetPropagationLossModel (Ptr<const PropagationLossModel> propagationModel)
{
  Ptr<const JakesPropagationLossModel> jakes = propagationModel->GetObject<JakesPropagationLossModel> ();
  NS_ASSERT_MSG (jakes != 0, "Jakes Process can work only with JakesPropagationLossModel!");
  m_jakes = jakes;
  
  NS_ASSERT (m_nOscillators != 0);
  NS_ASSERT (m_omegaDopplerMax != 0);
  
  ConstructOscillators ();
}

void
JakesProcess::SetNOscillators (unsigned int nOscillators)
{
  m_nOscillators = nOscillators;
}

void
JakesProcess::SetDopplerFrequencyHz (double dopplerFrequencyHz)
{
  m_omegaDopplerMax = 2 * dopplerFrequencyHz * M_PI;
}

void
JakesProcess::ConstructOscillators ()
{
  NS_ASSERT (m_jakes);
  // Initial phase is common for all oscillators:
  double phi = m_jakes->GetUniformRandomVariable ()->GetValue ();
  // Theta is common for all oscillators:
  double theta = m_jakes->GetUniformRandomVariable ()->GetValue ();
  for (unsigned int i = 0; i < m_nOscillators; i++)
    {
      unsigned int n = i + 1;
      /// 1. Rotation speed
      /// 1a. Initiate \f[ \alpha_n = \frac{2\pi n - \pi + \theta}{4M},  n=1,2, \ldots,M\f], n is oscillatorNumber, M is m_nOscillators
      double alpha = (2.0 * M_PI * n - M_PI + theta) / (4.0 * m_nOscillators);
      /// 1b. Initiate rotation speed:
      double omega = m_omegaDopplerMax * std::cos (alpha);
      /// 2. Initiate complex amplitude:
      double psi = m_jakes->GetUniformRandomVariable ()->GetValue ();
      std::complex<double> amplitude = std::complex<double> (std::cos (psi), std::sin (psi)) * 2.0 / std::sqrt (m_nOscillators);
      /// 3. Construct oscillator:
      m_oscillators.push_back (Oscillator (amplitude, phi, omega)); 
    }
}

JakesProcess::JakesProcess () :
  m_omegaDopplerMax (0),
  m_nOscillators (0)
{
}

JakesProcess::~JakesProcess()
{
  m_oscillators.clear ();
}

void
JakesProcess::DoDispose ()
{
  m_jakes = 0;
}

std::complex<double>
JakesProcess::GetComplexGain () const
{
  std::complex<double> sumAplitude = std::complex<double> (0, 0);
  for (unsigned int i = 0; i < m_oscillators.size (); i++)
    {
      sumAplitude += m_oscillators[i].GetValueAt (Now ());
    }
  return sumAplitude;
}

double
JakesProcess::GetChannelGainDb () const
{
  std::complex<double> complexGain = GetComplexGain ();
  return (10 * std::log10 ((std::pow (complexGain.real (), 2) + std::pow (complexGain.imag (), 2)) / 2));
}

} // namespace ns3
