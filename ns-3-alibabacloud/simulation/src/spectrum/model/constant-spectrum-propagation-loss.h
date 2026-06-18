/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#ifndef CONSTANT_SPECTRUM_PROPAGATION_LOSS_H
#define CONSTANT_SPECTRUM_PROPAGATION_LOSS_H

#include "ns3/spectrum-propagation-loss-model.h"

namespace ns3 {

/**
 * \ingroup spectrum
 *
 * A Constant (fixed) propagation loss. The loss is not dependent on the distance.
 */
class ConstantSpectrumPropagationLossModel : public SpectrumPropagationLossModel
{
public:
  ConstantSpectrumPropagationLossModel ();
  ~ConstantSpectrumPropagationLossModel ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  virtual Ptr<SpectrumValue> DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                           Ptr<const MobilityModel> a,
                                                           Ptr<const MobilityModel> b) const;
  /**
   * Set the propagation loss
   * \param lossDb the propagation loss [dB]
   */
  void SetLossDb (double lossDb);
  /**
   * Get the propagation loss
   * \returns the propagation loss [dB]
   */
  double GetLossDb () const;

protected:
  double m_lossDb;      //!< Propagation loss [dB]
  double m_lossLinear;  //!< Propagation loss (linear)
private:
};


} // namespace ns3

#endif /* CONSTANT_SPECTRUM_PROPAGATION_LOSS_MODEL_H */
