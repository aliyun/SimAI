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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 * 
 */

#ifndef OH_BUILDINGS_PROPAGATION_LOSS_MODEL_H_
#define OH_BUILDINGS_PROPAGATION_LOSS_MODEL_H_

#include <ns3/buildings-propagation-loss-model.h>

namespace ns3 {

class OkumuraHataPropagationLossModel;

/**
 * \ingroup propagation
 *
 *  this model combines the OkumuraHata model with the BuildingsPropagationLossModel
 * 
 *  \warning This model works with MobilityBuildingInfo only
 *
 */
class OhBuildingsPropagationLossModel : public BuildingsPropagationLossModel
{

public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  OhBuildingsPropagationLossModel ();
  ~OhBuildingsPropagationLossModel ();

  /**
   * \param a the mobility model of the source
   * \param b the mobility model of the destination
   * \returns the propagation loss (in dBm)
   */
  virtual double GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;
  
private:

  Ptr<OkumuraHataPropagationLossModel> m_okumuraHata; //!< OkumuraHata Propagation Loss Model

};

}

#endif /* OH_BUILDINGS_PROPAGATION_LOSS_MODEL_H_ */
