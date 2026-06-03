/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 * 
 */

#ifndef ITU_R_1411_LOS_PROPAGATION_LOSS_MODEL_H
#define ITU_R_1411_LOS_PROPAGATION_LOSS_MODEL_H

#include "ns3/propagation-loss-model.h"

namespace ns3 {


/**
 * \ingroup propagation
 *
 * \brief the ITU-R 1411 LOS propagation model
 * 
 * This class implements the ITU-R 1411 LOS propagation model for
 * Line-of-Sight (LoS) short range outdoor communication in the
 * frequency range 300 MHz to 100 GHz.  
 * For more information about the model, please see
 * the propagation module documentation in .rst format.
 */
class ItuR1411LosPropagationLossModel : public PropagationLossModel
{

public:

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  ItuR1411LosPropagationLossModel ();
  virtual ~ItuR1411LosPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  ItuR1411LosPropagationLossModel (const ItuR1411LosPropagationLossModel &) = delete;
  ItuR1411LosPropagationLossModel & operator = (const ItuR1411LosPropagationLossModel &) = delete;

  /** 
   * Set the operating frequency
   * 
   * \param freq the frequency in Hz
   */
  void SetFrequency (double freq);

  /** 
   * 
   * 
   * \param a the first mobility model
   * \param b the second mobility model
   * 
   * \return the loss in dBm for the propagation between
   * the two given mobility models
   */
  double GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  double m_lambda; //!< wavelength
};

} // namespace ns3


#endif // ITU_R_1411_LOS_PROPAGATION_LOSS_MODEL_H

