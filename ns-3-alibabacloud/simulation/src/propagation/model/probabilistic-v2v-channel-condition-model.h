/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
 */

#ifndef PROBABILISTIC_V2V_CHANNEL_CONDITION_MODEL_H
#define PROBABILISTIC_V2V_CHANNEL_CONDITION_MODEL_H

#include "ns3/channel-condition-model.h"

namespace ns3 {

class MobilityModel;

/**
 * \ingroup propagation
 *
 * \brief Computes the channel condition for the V2V Urban scenario
 *
 * Computes the channel condition following the probabilistic model described in 
 * M. Boban,  X.Gong, and  W. Xu, “Modeling the evolution of line-of-sight 
 * blockage for V2V channels,” in IEEE 84th Vehicular Technology 
 * Conference (VTC-Fall), 2016.
 */
class ProbabilisticV2vUrbanChannelConditionModel : public ThreeGppChannelConditionModel
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor for the ProbabilisticV2vUrbanChannelConditionModel class
   */
  ProbabilisticV2vUrbanChannelConditionModel ();

  /**
   * Destructor for the ProbabilisticV2vUrbanChannelConditionModel class
   */
  virtual ~ProbabilisticV2vUrbanChannelConditionModel () override;

private:
  /**
   * Compute the LOS probability 
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the LOS probability
   */
  virtual double ComputePlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;
  
  /**
   * Compute the NLOS probability 
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the NLOS probability
   */
  virtual double ComputePnlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;
  
  enum VehicleDensity m_densityUrban {VehicleDensity::INVALID}; //!< vehicle density
};

/**
 * \ingroup propagation
 *
 * \brief Computes the channel condition for the V2V Highway scenario
 *
 * Computes the channel condition following the probabilistic model described in 
 * M. Boban,  X.Gong, and  W. Xu, “Modeling the evolution of line-of-sight 
 * blockage for V2V channels,” in IEEE 84th Vehicular Technology 
 * Conference (VTC-Fall), 2016.
 */
class ProbabilisticV2vHighwayChannelConditionModel : public ThreeGppChannelConditionModel
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor for the ProbabilisticV2vHighwayChannelConditionModel class
   */
  ProbabilisticV2vHighwayChannelConditionModel ();

  /**
   * Destructor for the ProbabilisticV2vHighwayChannelConditionModel class
   */
  virtual ~ProbabilisticV2vHighwayChannelConditionModel () override;

private:
  /**
   * Compute the LOS probability
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the LOS probability
   */
  virtual double ComputePlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;
  
  /**
   * Compute the NLOS probability 
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the NLOS probability
   */
  virtual double ComputePnlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;
  
  enum VehicleDensity m_densityHighway {VehicleDensity::INVALID}; //!< vehicle density
};

} // end ns3 namespace

#endif /* PROBABILISTIC_V2V_CHANNEL_CONDITION_MODEL_H */
