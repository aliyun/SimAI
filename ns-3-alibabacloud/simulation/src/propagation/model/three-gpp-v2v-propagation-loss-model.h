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

#ifndef THREE_GPP_V2V_PROPAGATION_LOSS_MODEL_H
#define THREE_GPP_V2V_PROPAGATION_LOSS_MODEL_H

#include "three-gpp-propagation-loss-model.h"

namespace ns3 {

/**
 * \ingroup propagation
 *
 * \brief Implements the pathloss model defined in 3GPP TR 37.885, Table 6.2.1-1
 *        for the Urban scenario.
 */
class ThreeGppV2vUrbanPropagationLossModel : public ThreeGppPropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor
   */
  ThreeGppV2vUrbanPropagationLossModel ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vUrbanPropagationLossModel () override;

  /**
   * \brief Copy constructor
   *
   * Defined and unimplemented to avoid misuse
   */
  ThreeGppV2vUrbanPropagationLossModel (const ThreeGppV2vUrbanPropagationLossModel &) = delete;

  /**
   * \brief Copy constructor
   *
   * Defined and unimplemented to avoid misuse
   * \returns the ThreeGppRmaPropagationLossModel instance
   */
  ThreeGppV2vUrbanPropagationLossModel & operator = (const ThreeGppV2vUrbanPropagationLossModel &) = delete;
  
private:
  /**
   * \brief Computes the pathloss between a and b considering that the line of
   *        sight is not obstructed
   * \param distance2D the 2D distance between tx and rx in meters
   * \param distance3D the 3D distance between tx and rx in meters
   * \param hUt the height of the UT in meters
   * \param hBs the height of the BS in meters
   * \return pathloss value in dB
   */
  virtual double GetLossLos (double distance2D, double distance3D, double hUt, double hBs) const override;

  /**
   * \brief Computes the pathloss between a and b considering that the line of
   *        sight is obstructed by a vehicle
   * \param distance2D the 2D distance between tx and rx in meters
   * \param distance3D the 3D distance between tx and rx in meters
   * \param hUt the height of the UT in meters
   * \param hBs the height of the BS in meters
   * \return pathloss value in dB
   */
  virtual double GetLossNlosv (double distance2D, double distance3D, double hUt, double hBs) const override;
  
  /**
   * \brief Computes the pathloss between a and b considering that the line of
   *        sight is obstructed by a building
   * \param distance2D the 2D distance between tx and rx in meters
   * \param distance3D the 3D distance between tx and rx in meters
   * \param hUt the height of the UT in meters
   * \param hBs the height of the BS in meters
   * \return pathloss value in dB
   */
  virtual double GetLossNlos (double distance2D, double distance3D, double hUt, double hBs) const override;
  
  /**
   * \brief Computes the additional loss due to an obstruction caused by a vehicle
   * \param distance3D the 3D distance between tx and rx in meters
   * \param hUt the height of the UT in meters
   * \param hBs the height of the BS in meters
   * \return pathloss value in dB
   */
  double GetAdditionalNlosvLoss (double distance3D, double hUt, double hBs) const;

  /**
   * \brief Returns the shadow fading standard deviation
   * \param a tx mobility model
   * \param b rx mobility model
   * \param cond the LOS/NLOS channel condition
   * \return shadowing std in dB
   */
  virtual double GetShadowingStd (Ptr<MobilityModel> a, Ptr<MobilityModel> b, ChannelCondition::LosConditionValue cond) const override;

  /**
   * \brief Returns the shadow fading correlation distance
   * \param cond the LOS/NLOS channel condition
   * \return shadowing correlation distance in meters
   */
  virtual double GetShadowingCorrelationDistance (ChannelCondition::LosConditionValue cond) const override;
  
  double m_percType3Vehicles; //!< percentage of Type 3 vehicles in the scenario (i.e., trucks)
  Ptr<UniformRandomVariable> m_uniformVar; //!< uniform random variable
  Ptr<LogNormalRandomVariable> m_logNorVar; //!< log normal random variable
};

/**
 * \ingroup propagation
 *
 * \brief Implements the pathloss model defined in 3GPP TR 37.885, Table 6.2.1-1
 *        for the Highway scenario.
 */
class ThreeGppV2vHighwayPropagationLossModel : public ThreeGppV2vUrbanPropagationLossModel
{ 
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor
   */
  ThreeGppV2vHighwayPropagationLossModel ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vHighwayPropagationLossModel () override;

private:
  /**
   * \brief Computes the pathloss between a and b considering that the line of
   *        sight is not obstructed
   * \param distance2D the 2D distance between tx and rx in meters
   * \param distance3D the 3D distance between tx and rx in meters
   * \param hUt the height of the UT in meters
   * \param hBs the height of the BS in meters
   * \return pathloss value in dB
   */
  virtual double GetLossLos (double distance2D, double distance3D, double hUt, double hBs) const override;
};

} // namespace ns3

#endif /* THREE_GPP_V2V_PROPAGATION_LOSS_MODEL_H */
