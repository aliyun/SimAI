/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef FRIIS_SPECTRUM_PROPAGATION_LOSS_H
#define FRIIS_SPECTRUM_PROPAGATION_LOSS_H


#include <ns3/spectrum-propagation-loss-model.h>


namespace ns3 {

class MobilityModel;


/**
 * \ingroup spectrum
 * \brief Friis spectrum propagation loss model
 *
 * The propagation loss is calculated according to a simplified version of Friis'
 * formula in which antenna gains are unitary:
 *
 * \f$ L = \frac{4 \pi * d * f}{C^2}\f$
 *
 * where C = 3e8 m/s is the light speed in the vacuum. The intended
 * use is to calculate Prx = Ptx * G
 */
class FriisSpectrumPropagationLossModel : public SpectrumPropagationLossModel
{

public:
  FriisSpectrumPropagationLossModel ();
  ~FriisSpectrumPropagationLossModel ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();


  virtual Ptr<SpectrumValue> DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                           Ptr<const MobilityModel> a,
                                                           Ptr<const MobilityModel> b) const;


  /**
   * Return the propagation loss L according to a simplified version of Friis'
   * formula in which antenna gains are unitary
   *
   * @param f frequency in Hz
   * @param d distance in m
   *
   * @return if Prx < Ptx then return Prx; else return Ptx
   */
  double CalculateLoss (double f, double d) const;
};






} // namespace ns3

#endif /* FRIIS_SPECTRUM_PROPAGATION_LOSS_MODEL_H */
