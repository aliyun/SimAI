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
#ifndef DOPPLER_PROCESS_H
#define DOPPLER_PROCESS_H

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"
#include <complex>

namespace ns3
{
class PropagationLossModel;
class JakesPropagationLossModel;
/**
 * \ingroup fading
 *
 * \brief Implementation for a single path Stationary Jakes propagation loss model.
 *
 * The Jakes propagation loss model implemented here is
 * described in [1].
 *
 * We consider one transmitter - receiver pair and calculate
 * the complex coefficients for this case as follow:
 * \f[ X(t)=X_c(t) + j X_s(t)\f]
 * \f[ X_c(t) = \frac{2}{\sqrt{M}}\sum_{n=1}^{M}\cos(\psi_n)\cos(\omega_d t\cos(\alpha_n)+\phi_n)\f]
 * \f[ X_s(t) = \frac{2}{\sqrt{M}}\sum_{n=1}^{M}\sin(\psi_n)\cos(\omega_d t\cos(\alpha_n)+\phi_n)\f]
 * with
 * \f[ \alpha_n = \frac{2\pi n - \pi + \theta}{4M},  n=1,2, \ldots,M\f]
 * where
 *\f$\theta\f$, \f$\phi\f$, and \f$\psi_n\f$ are statically independent and uniformly distributed over \f$[-\pi, \pi)\f$ for all \f$n\f$.
 *
 *
 * [1] Y. R. Zheng and C. Xiao, "Simulation Models With Correct
 * Statistical Properties for Rayleigh Fading Channel", IEEE
 * Trans. on Communications, Vol. 51, pp 920-928, June 2003
 */
class JakesProcess : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  JakesProcess ();
  virtual ~JakesProcess();
  virtual void DoDispose ();

  /**
   * Get the channel complex gain
   * \return the channel complex gain
   */
  std::complex<double> GetComplexGain () const;
  /**
   * Get the channel gain in dB
   * \return the channel gain [dB]
   */
  double GetChannelGainDb () const;

  /**
   * Set the propagation model using this class
   * \param model the propagation model using this class
   */
  void SetPropagationLossModel (Ptr<const PropagationLossModel> model);
private:
  /**
   * This class Represents a single oscillator
   */
  struct Oscillator
  {
    /**
     * Initiate oscillator with complex amplitude, initial phase and rotation speed
     * \param amplitude initial complex amplitude
     * \param initialPhase initial phase
     * \param omega rotation speed
     */
    Oscillator (std::complex<double> amplitude, double initialPhase, double omega);
    /**
     * Get the complex amplitude at a given moment
     * \param t time instant
     * \returns the complex amplitude
     */
    std::complex<double> GetValueAt (Time t) const;

    std::complex<double> m_amplitude; //!< Complex number \f$Re=\cos(\psi_n), Im = i\sin(\psi_n)]\f$
    double m_phase; //!< Phase \f$\phi_n\f$ of the oscillator
    double m_omega; //!< Rotation speed of the oscillator \f$\omega_d \cos(\alpha_n)\f$
  };
private:

  /**
   * Set the number of Oscillators to use
   * @param nOscillators the number of oscillators
   */
  void SetNOscillators (unsigned int nOscillators);

  /**
   * Set the Doppler frequency
   * @param dopplerFrequencyHz the Doppler frequency [Hz]
   */
  void SetDopplerFrequencyHz (double dopplerFrequencyHz);

  /**
   *
   */
  void ConstructOscillators ();
private:
  std::vector<Oscillator> m_oscillators; //!< Vector of oscillators
  double m_omegaDopplerMax; //!< max rotation speed Doppler frequency
  unsigned int m_nOscillators;  //!< number of oscillators
  Ptr<UniformRandomVariable> m_uniformVariable; //!< random stream
  Ptr<const JakesPropagationLossModel> m_jakes; //!< pointer to the propagation loss model
};
} // namespace ns3
#endif // DOPPLER_PROCESS_H
