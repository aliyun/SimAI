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

#ifndef BUILDINGS_PROPAGATION_LOSS_MODEL_H_
#define BUILDINGS_PROPAGATION_LOSS_MODEL_H_

#include "ns3/nstime.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/random-variable-stream.h"
#include <ns3/building.h>
#include <ns3/mobility-building-info.h>



namespace ns3 {

class ShadowingLossModel;
class JakesFadingLossModel;

/**
 * \ingroup propagation
 *
 *  This model provides means for simulating the following propagation
 *  phenomena in the presence of buildings: 
 *
 *   - shadowing (indoor, outdoor)
 *   - external wall penetration loss
 *   - internal wall penetration loss
 *  
 *  The distance-dependent component of propagation loss is deferred
 *  to derived classes which are expected to implement the GetLoss method.
 *  
 *  \warning This model works only when MobilityBuildingInfo is aggreegated
 *  to the mobility model
 *
 */

class BuildingsPropagationLossModel : public PropagationLossModel
{

public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  BuildingsPropagationLossModel ();
  /**
   * \param a the mobility model of the source
   * \param b the mobility model of the destination
   * \returns the propagation loss (in dBm)
   */
  virtual double GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const = 0;

  // inherited from PropagationLossModel
  virtual double DoCalcRxPower (double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;

protected:
  /**
   * Calculate the external wall loss
   * \param a Building data
   * \returns the propagation loss (in dBm)
   */
  double ExternalWallLoss (Ptr<MobilityBuildingInfo> a) const;
  /**
   * Calculate the height loss
   * \param n Building data
   * \returns the propagation loss (in dBm)
   */
  double HeightLoss (Ptr<MobilityBuildingInfo> n) const;
  /**
   * Calculate the internal wall loss
   * \param a Room A data
   * \param b Room B data
   * \returns the propagation loss (in dBm)
   */
  double InternalWallsLoss (Ptr<MobilityBuildingInfo> a, Ptr<MobilityBuildingInfo> b) const;
  
  /**
   * Calculate the shadowing loss
   * \param a Room A data
   * \param b Room B data
   * \returns the propagation loss (in dBm)
   */
  double GetShadowing (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;

  double m_lossInternalWall; //!< loss from internal walls (in dBm)

  /**
   * \ingroup propagation
   * 
   * This model allows the computation of shadowing loss
   */
  class ShadowingLoss
  {
  public:
    ShadowingLoss ();
    /**
     * Constructor
     * \param shadowingValue Value for shadowing
     * \param receiver Receiver position
     */
    ShadowingLoss (double shadowingValue, Ptr<MobilityModel> receiver);
    /**
     * \returns the loss (in dBm)
     */
    double GetLoss () const;
    /**
     * \returns the receiver mobility model
     */
    Ptr<MobilityModel> GetReceiver (void) const;
  protected:
    double m_shadowingValue; //!< Shadowing value
    Ptr<MobilityModel> m_receiver; //!< The receiver mobility model
  };

  /// Map of the shadowng loss
  mutable std::map<Ptr<MobilityModel>,  std::map<Ptr<MobilityModel>, ShadowingLoss> > m_shadowingLossMap;
  /**
   * Calculate the Standard deviation of the normal distribution used to calculate the shadowing
   * \param a Room A data
   * \param b Room B data
   * \return the Standard deviation of the normal distribution
   */
  double EvaluateSigma (Ptr<MobilityBuildingInfo> a, Ptr<MobilityBuildingInfo> b) const;

  /// Standard deviation of the normal distribution used to calculate the shadowing due to ext walls
  double m_shadowingSigmaExtWalls;
  /// Standard deviation of the normal distribution used to calculate the shadowing for outdoor nodes
  double m_shadowingSigmaOutdoor;
  /// Standard deviation of the normal distribution used to calculate the shadowing for indoor nodes
  double m_shadowingSigmaIndoor;
  Ptr<NormalRandomVariable> m_randVariable; //!< Random variable

  virtual int64_t DoAssignStreams (int64_t stream);
};

}

#endif /* BUILDINGS_PROPAGATION_LOSS_MODEL_H_ */
