/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 SIGNET Lab, Department of Information Engineering,
 * University of Padova
 * Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#ifndef THREE_GPP_V2V_CHANNEL_CONDITION_MODEL
#define THREE_GPP_V2V_CHANNEL_CONDITION_MODEL

#include "ns3/channel-condition-model.h"
#include "buildings-channel-condition-model.h"
#include <functional>

namespace ns3 {

class MobilityModel;

/**
 * \ingroup buildings
 *
 * \brief Computes the channel condition for the V2V Urban scenario
 *
 * Computes the channel condition following the specifications for the
 * V2V Urban scenario reported in Table 6.2-1 of 3GPP TR 37.885.
 *
 * 3GPP TR 37.885 defines 3 different channel states for vehicular environments:
 * LOS, NLOS and NLOSv, the latter representing the case in which the LOS path is
 * blocked by other vehicles in the scenario. The document defines a probabilistic
 * model to determine if the channel state is LOS or NLOSv, while the NLOS state
 * is determined in a deterministic way based on the buildings deployed in the
 * scenario. For this reason, this class makes use of an instance of
 * BuildingsChannelConditionModel to determine if the LOS is obstructed by
 * buildings or not.
 */
class ThreeGppV2vUrbanChannelConditionModel : public ThreeGppChannelConditionModel
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor for the ThreeGppV2vUrbanChannelConditionModel class
   */
  ThreeGppV2vUrbanChannelConditionModel ();

  /**
   * Destructor for the ThreeGppV2vUrbanChannelConditionModel class
   */
  virtual ~ThreeGppV2vUrbanChannelConditionModel () override;

private:
  /**
   * Compute the LOS probability as specified in Table Table 6.2-1 of 3GPP TR 37.885
   * for the V2V Urban scenario.
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the LOS probability
   */
  virtual double ComputePlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;

  /**
   * Compute the NLOS probability. It determines the presence of obstructions
   * between the tx and the rx based on the buildings deployed in the scenario.
   * It returns 1 if the LOS path is obstructed, 0 otherwise.
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the NLOS probability
   */
  virtual double ComputePnlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;

  Ptr<BuildingsChannelConditionModel> m_buildingsCcm; //!< used to determine the obstructions due to buildings
};

/**
 * \ingroup buildings
 *
 * \brief Computes the channel condition for the V2V Highway scenario
 *
 * Computes the channel condition following the specifications for the
 * V2V Highway scenario reported in Table 6.2-1 of 3GPP TR 37.885.
 *
 * 3GPP TR 37.885 defines 3 different channel states for vehicular environments:
 * LOS, NLOS and NLOSv, the latter representing the case in which the LOS path is
 * blocked by other vehicles in the scenario. The document defines a probabilistic
 * model to determine if the channel state is LOS or NLOSv, while the NLOS state
 * is determined in a deterministic way based on the buildings deployed in the
 * scenario. For this reason, this class makes use of an instance of
 * BuildingsChannelConditionModel to determine if the LOS is obstructed by
 * buildings or not.
 */
class ThreeGppV2vHighwayChannelConditionModel : public ThreeGppChannelConditionModel
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor for the ThreeGppV2vHighwayChannelConditionModel class
   */
  ThreeGppV2vHighwayChannelConditionModel ();

  /**
   * Destructor for the ThreeGppV2vHighwayChannelConditionModel class
   */
  virtual ~ThreeGppV2vHighwayChannelConditionModel () override;

private:
  /**
   * Compute the LOS probability as specified in Table Table 6.2-1 of 3GPP TR 37.885
   * for the V2V Highway scenario.
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the LOS probability
   */
  virtual double ComputePlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;

  /**
   * Compute the NLOS probability. It determines the presence of obstructions
   * between the tx and the rx based on the buildings deployed in the scenario.
   * It returns 1 if the LOS path is obstructed, 0 otherwise.
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the NLOS probability
   */
  virtual double ComputePnlos (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const override;

  /**
   * \brief The callback which is hooked to a method to compute channel condition.
   *
   * This callback is implemented to make this model robust against the
   * presence and absence of buildings in a highway scenario. If there are
   * buildings in a scenario, this model will use
   * \link BuildingsChannelConditionModel \endlink, which requires
   * \link MobilityBuildingInfo \endlink aggregated to the nodes to compute
   * LOS and NLOS. Otherwise, the callback is hooked to a local method
   * \link GetChCondWithNoBuildings \endlink
   * , which construct the ChannelCondtion object and set the condition to
   * outdoor to outdoor with LOS.
   */
  std::function <Ptr<ChannelCondition> (Ptr<const MobilityModel>, Ptr<const MobilityModel>) > ComputeChCond;

  /**
   * \brief Get the channel condition and redirect the callback
   * \link ComputeChCond \endlink to \link GetChCondWithBuildings \endlink
   * or to \link GetChCondWithNoBuildings \endlink depending on if there are
   * buildings in the scenario or not.
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the the condition of the channel between \p a and \p b
   */
  Ptr<ChannelCondition> GetChCondAndFixCallback (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b);

  /**
   * \brief Get the channel condition between \p a and \p b
   *        using BuildingsChannelConditionModel.
   *
   * This method will be called for the scenarios with buildings
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the condition of the channel between \p a and \p b
   */
  Ptr<ChannelCondition> GetChCondWithBuildings (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const;

  /**
   * \brief Get the channel condition between \p a and \p b
   *
   * This method will be called for the scenarios without buildings
   *
   * \param a tx mobility model
   * \param b rx mobility model
   * \return the condition of the channel between \p a and \p b
   */
  Ptr<ChannelCondition> GetChCondWithNoBuildings (Ptr<const MobilityModel> a, Ptr<const MobilityModel> b) const;

  Ptr<BuildingsChannelConditionModel> m_buildingsCcm; //!< used to determine the obstructions due to buildings
};

} // end ns3 namespace

#endif /* THREE_GPP_V2V_CHANNEL_CONDITION_MODEL */
