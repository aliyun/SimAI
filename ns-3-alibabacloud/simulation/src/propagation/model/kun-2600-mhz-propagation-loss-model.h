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

#ifndef KUN_2600MHZ_PROPAGATION_LOSS_MODEL_H
#define KUN_2600MHZ_PROPAGATION_LOSS_MODEL_H

#include <ns3/propagation-loss-model.h>

namespace ns3 {


/**
 * \ingroup propagation
 *
 * \brief Empirical propagation model for the 2.6 GHz frequency
 * 
 * This class implements the empirical model for 2.6 GHz taken from this paper:
 * Sun Kun, Wang Ping, Li Yingze 
 * "Path Loss Models for Suburban Scenario at 2.3GHz, 2.6GHz and 3.5GHz"
 * 8th International Symposium on Antennas, Propagation and EM Theory (ISAPE), 2008.
 */
class Kun2600MhzPropagationLossModel : public PropagationLossModel
{

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  Kun2600MhzPropagationLossModel ();
  virtual ~Kun2600MhzPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  Kun2600MhzPropagationLossModel (const Kun2600MhzPropagationLossModel &) = delete;
  Kun2600MhzPropagationLossModel & operator = (const Kun2600MhzPropagationLossModel &) = delete;

  /** 
   * \param a the first mobility model
   * \param b the second mobility model
   * 
   * \return the loss in dBm for the propagation between
   * the two given mobility models
   */
  double GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;

private:
  // inherited from PropagationLossModel
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;
  int64_t DoAssignStreams (int64_t stream) override;
};

} // namespace ns3


#endif // KUN_2600MHZ_PROPAGATION_LOSS_MODEL_H

