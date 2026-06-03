/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 CTTC
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
 */

#ifndef PHASED_ARRAY_SPECTRUM_PROPAGATION_LOSS_MODEL_H
#define PHASED_ARRAY_SPECTRUM_PROPAGATION_LOSS_MODEL_H


#include <ns3/object.h>
#include <ns3/mobility-model.h>
#include <ns3/spectrum-value.h>
#include <ns3/phased-array-model.h>

namespace ns3 {


/**
 * \ingroup spectrum
 *
 * \brief spectrum-aware propagation loss model that is
 * compatible with PhasedArrayModel type of ns-3 antenna
 *
 * Interface for propagation loss models to be adopted when
 * transmissions are modeled with a power spectral density by means of
 * the SpectrumValue class, and when PhasedArrayModel type of atenna
 * is being used for TX and RX.
 *
 */
class PhasedArraySpectrumPropagationLossModel : public Object
{
public:
  PhasedArraySpectrumPropagationLossModel ();
  virtual ~PhasedArraySpectrumPropagationLossModel ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * Used to chain various instances of PhasedArraySpectrumPropagationLossModel
   *
   * @param next
   */
  void SetNext (Ptr<PhasedArraySpectrumPropagationLossModel> next);

  /**
   * This method is to be called to calculate
   *
   * @param txPsd the SpectrumValue representing the power spectral
   * density of the transmission. Watt units are to be used for radio
   * communications, and Pascal units for acoustic communications
   * (e.g., underwater).
   *
   * @param a sender mobility
   * @param b receiver mobility
   * @param aPhasedArrayModel the instance of the phased antenna array of the sender
   * @param bPhasedArrayModel the instance of the phased antenna array of the receiver
   *
   * @return set of values Vs frequency representing the received
   * power in the same units used for the txPower parameter.
   */
  Ptr<SpectrumValue> CalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                 Ptr<const MobilityModel> a,
                                                 Ptr<const MobilityModel> b,
                                                 Ptr<const PhasedArrayModel> aPhasedArrayModel,
                                                 Ptr<const PhasedArrayModel> bPhasedArrayModel) const;

protected:
  virtual void DoDispose ();


private:
  /**
   *
   * @param txPsd set of values Vs frequency representing the
   * transmission power. See SpectrumChannel for details.
   * @param a sender mobility
   * @param b receiver mobility
   * @param aPhasedArrayModel the instance of the phased antenna array of the sender
   * @param bPhasedArrayModel the instance of the phased antenna array of the receiver
   *
   * @return set of values Vs frequency representing the received
   * power in the same units used for the txPower parameter.
   */
  virtual Ptr<SpectrumValue> DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                           Ptr<const MobilityModel> a,
                                                           Ptr<const MobilityModel> b,
                                                           Ptr<const PhasedArrayModel> aPhasedArrayModel,
                                                           Ptr<const PhasedArrayModel> bPhasedArrayModel) const = 0;

  Ptr<PhasedArraySpectrumPropagationLossModel> m_next; //!< PhasedArraySpectrumPropagationLossModel chained to this one.
};






} // namespace ns3

#endif /* PHASED_ARRAY_SPECTRUM_PROPAGATION_LOSS_MODEL_H */


