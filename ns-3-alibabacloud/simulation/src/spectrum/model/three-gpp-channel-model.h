/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015, NYU WIRELESS, Tandon School of Engineering,
 * New York University
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
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

#ifndef THREE_GPP_CHANNEL_H
#define THREE_GPP_CHANNEL_H

#include  <complex.h>
#include "ns3/angles.h"
#include <ns3/object.h>
#include <ns3/nstime.h>
#include <ns3/random-variable-stream.h>
#include <ns3/boolean.h>
#include <unordered_map>
#include <ns3/channel-condition-model.h>
#include <ns3/matrix-based-channel-model.h>

namespace ns3 {

class MobilityModel;

/**
 * \ingroup spectrum
 * \brief Channel Matrix Generation following 3GPP TR 38.901
 *
 * The class implements the channel matrix generation procedure
 * described in 3GPP TR 38.901.
 *
 * \see GetChannel
 */
class ThreeGppChannelModel : public MatrixBasedChannelModel
{
public:
  /**
   * Constructor
   */
  ThreeGppChannelModel ();

  /**
   * Destructor
   */
  ~ThreeGppChannelModel ();

  void DoDispose () override;

  /**
   * Get the type ID
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * Set the channel condition model
   * \param model a pointer to the ChannelConditionModel object
   */
  void SetChannelConditionModel (Ptr<ChannelConditionModel> model);

  /**
   * Get the associated channel condition model
   * \return a pointer to the ChannelConditionModel object
   */
  Ptr<ChannelConditionModel> GetChannelConditionModel () const;

  /**
   * Sets the center frequency of the model
   * \param f the center frequency in Hz
   */
  void SetFrequency (double f);

  /**
   * Returns the center frequency
   * \return the center frequency in Hz
   */
  double GetFrequency (void) const;

  /**
   * Sets the propagation scenario
   * \param scenario the propagation scenario
   */
  void SetScenario (const std::string &scenario);

  /**
   * Returns the propagation scenario
   * \return the propagation scenario
   */
  std::string GetScenario (void) const;

  /**
   * Looks for the channel matrix associated to the aMob and bMob pair in m_channelMatrixMap.
   * If found, it checks if it has to be updated. If not found or if it has to
   * be updated, it generates a new uncorrelated channel matrix using the
   * method GetNewChannel and updates m_channelMap.
   *
   * \param aMob mobility model of the a device
   * \param bMob mobility model of the b device
   * \param aAntenna antenna of the a device
   * \param bAntenna antenna of the b device
   * \return the channel matrix
   */
  Ptr<const ChannelMatrix> GetChannel (Ptr<const MobilityModel> aMob,
                                       Ptr<const MobilityModel> bMob,
                                       Ptr<const PhasedArrayModel> aAntenna,
                                       Ptr<const PhasedArrayModel> bAntenna) override;


  /**
   * Looks for the channel params associated to the aMob and bMob pair in
   * m_channelParamsMap. If not found it will return a nullptr.
   *
   * \param aMob mobility model of the a device
   * \param bMob mobility model of the b device
   * \return the channel params
   */
  Ptr<const ChannelParams> GetParams (Ptr<const MobilityModel> aMob,
                                      Ptr<const MobilityModel> bMob) const override;
  /**
   * \brief Assign a fixed random variable stream number to the random variables
   * used by this model.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);
  
protected:
  /**
   * Wrap an (azimuth, inclination) angle pair in a valid range.
   * Specifically, inclination must be in [0, M_PI] and azimuth in [0, 2*M_PI).
   * If the inclination angle is outside of its range, the azimuth angle is
   * rotated by M_PI.
   * This methods aims specifically at solving the problem of generating angles at
   * the boundaries of the angle domain, specifically, generating angle distributions
   * close to inclinationRad=0 and inclinationRad=M_PI.
   *
   * \param azimuthRad the azimuth angle in radians
   * \param inclinationRad the inclination angle in radians
   * \return the wrapped (azimuth, inclination) angle pair in radians
   */
  static std::pair<double, double> WrapAngles (double azimuthRad, double inclinationRad);

  /**
   * \brief Shuffle the elements of a simple sequence container of type double
   * \param first Pointer to the first element among the elements to be shuffled
   * \param last Pointer to the last element among the elements to be shuffled
   */
  void Shuffle (double * first, double * last) const;

  /**
   * Extends the struct ChannelParams by including information that is used
   * within the ThreeGppChannelModel class
   */
  struct ThreeGppChannelParams : public MatrixBasedChannelModel::ChannelParams
  {
    ChannelCondition:: LosConditionValue m_losCondition; //!< contains the information about the LOS state of the channel
    ChannelCondition::O2iConditionValue m_o2iCondition; //!< contains the information about the O2I state of the channel
    // TODO these are not currently used, they have to be correctly set when including the spatial consistent update procedure
    /*The following parameters are stored for spatial consistent updating. The notation is
    that of 3GPP technical reports, but it can apply also to other channel realizations*/
    MatrixBasedChannelModel::Double2DVector m_nonSelfBlocking; //!< store the blockages
    Vector m_preLocUT; //!< location of UT when generating the previous channel
    Vector m_locUT; //!< location of UT
    MatrixBasedChannelModel::Double2DVector m_norRvAngles; //!< stores the normal variable for random angles angle[cluster][id] generated for equation (7.6-11)-(7.6-14), where id = 0(aoa),1(zoa),2(aod),3(zod)
    double m_DS; //!< delay spread
    double m_K_factor; //!< K factor
    uint8_t m_reducedClusterNumber; //!< reduced cluster number;
    MatrixBasedChannelModel::Double2DVector m_rayAodRadian; //!< the vector containing AOD angles
    MatrixBasedChannelModel::Double2DVector m_rayAoaRadian; //!< the vector containing AOA angles
    MatrixBasedChannelModel::Double2DVector m_rayZodRadian; //!< the vector containing ZOD angles
    MatrixBasedChannelModel::Double2DVector m_rayZoaRadian; //!< the vector containing ZOA angles
    MatrixBasedChannelModel::Double3DVector m_clusterPhase; //!< the initial random phases
    MatrixBasedChannelModel::Double2DVector m_crossPolarizationPowerRatios;//!< cross polarization power ratios
    Vector m_speed; //!< velocity
    double m_dis2D; //!< 2D distance between tx and rx
    double m_dis3D; //!< 3D distance between tx and rx
    DoubleVector m_clusterPower; //!< cluster powers
    DoubleVector m_attenuation_dB; //!< vector that stores the attenuation of the blockage
    uint8_t m_cluster1st; //!< index of the first strongest cluster
    uint8_t m_cluster2nd; //!< index of the second strongest cluster

  };

  /**
   * Data structure that stores the parameters of 3GPP TR 38.901, Table 7.5-6,
   * for a certain scenario
   */
  struct ParamsTable : public SimpleRefCount<ParamsTable>
  {
    uint8_t m_numOfCluster = 0; //!< Number of clusters
    uint8_t m_raysPerCluster = 0; //!< Number of rays per cluster
    double m_uLgDS = 0; //!< Mean value of 10-base logarithm of delay spread
    double m_sigLgDS = 0; //!< Standard deviation value of 10-base logarithm of delay spread
    double m_uLgASD = 0; //!< Mean value of 10-base logarithm of azimuth angle spread of departure
    double m_sigLgASD = 0; //!< Standard deviation of 10-base logarithm of azimuth angle spread of departure
    double m_uLgASA = 0; //!< Mean value of 10-base logarithm of azimuth angle spread of arrival
    double m_sigLgASA = 0; //!< Standard deviation of 10-base logarithm of azimuth angle spread of arrival
    double m_uLgZSA = 0; //!< Mean value of 10-base logarithm of zenith angle spread of arrival
    double m_sigLgZSA = 0; //!< Standard deviation of 10-base logarithm of zenith angle spread of arrival
    double m_uLgZSD = 0; //!< Mean value of 10-base logarithm of zenith angle spread of departure
    double m_sigLgZSD = 0; //!< Standard deviation of 10-base logarithm of zenith angle spread of departure
    double m_offsetZOD = 0; //!< Offset of zenith angle of departure
    double m_cDS = 0; //!< Cluster DS
    double m_cASD = 0; //!< Cluster ASD (Azimuth angle Spread of Departure)
    double m_cASA = 0; //!< Cluster ASA (Azimuth angle Spread of Arrival)
    double m_cZSA = 0; //!< Cluster ZSA (Zenith angle Spread of Arrival)
    double m_uK = 0; //!< Mean of K-factor
    double m_sigK = 0; //!< Standard deviation of K-factor
    double m_rTau = 0; //!< Delay scaling parameter
    double m_uXpr = 0; //!< Mean of Cross-Polarization Ratio
    double m_sigXpr = 0; //!< Standard deviation of Cross-Polarization Ratio
    double m_perClusterShadowingStd = 0; //!< Per cluster shadowing standard deviation 
    double m_sqrtC[7][7]; //!< The square root matrix and follows the order of [SF, K, DS, ASD, ASA, ZSD, ZSA]
  };

  /**
   * Get the parameters needed to apply the channel generation procedure
   * \param channelCondition the channel condition
   * \param hBS the height of the BS
   * \param hUT the height of the UT
   * \param distance2D the 2D distance between tx and rx
   * \return the parameters table
   */
  virtual Ptr<const ParamsTable> GetThreeGppTable (Ptr<const ChannelCondition> channelCondition, double hBS, double hUT, double distance2D) const;

  /**
   * Prepare 3gpp channel parameters among the nodes a and b.
   * The function does the followin steps described in 3GPP 38.901:
   *
   * Step 4: Generate large scale parameters. All LSPS are uncorrelated.
   * Step 5: Generate Delays.
   * Step 6: Generate cluster powers.
   * Step 7: Generate arrival and departure angles for both azimuth and elevation.
   * Step 8: Coupling of rays within a cluster for both azimuth and elevation
   * shuffle all the arrays to perform random coupling
   * Step 9: Generate the cross polarization power ratios
   * Step 10: Draw initial phases
   *
   * All relevant generated parameters are added then to ThreeGppChannelParams
   * which is the return value of this function.
   * \param channelCondition the channel condition
   * \param table3gpp the 3gpp parameters from the table
   * \param aMob the a node mobility model
   * \param bMob the b node mobility model
   * \return ThreeGppChannelParams structure with all the channel parameters generated according 38.901 steps from 4 to 10.
   */
  Ptr<ThreeGppChannelParams> GenerateChannelParameters (const Ptr<const ChannelCondition> channelCondition,
                                                        const Ptr<const ParamsTable> table3gpp,
                                                        const Ptr<const MobilityModel> aMob,
                                                        const Ptr<const MobilityModel> bMob) const;

  /**
   * Compute the channel matrix between two nodes a and b, and their
   * antenna arrays aAntenna and bAntenna using the procedure
   * described in 3GPP TR 38.901
   * \param channelParams the channel parameters previously generated for the pair of nodes a and b
   * \param table3gpp the 3gpp parameters table
   * \param sMob the mobility model of node s
   * \param uMob the mobility model of node u
   * \param sAntenna the antenna array of node s
   * \param uAntenna the antenna array of node u
   * \return the channel realization
   */

  virtual Ptr<ChannelMatrix> GetNewChannel (Ptr<const ThreeGppChannelParams> channelParams,
                                    Ptr<const ParamsTable> table3gpp,
                                    const Ptr<const MobilityModel> sMob,
                                    const Ptr<const MobilityModel> uMob,
                                    Ptr<const PhasedArrayModel> sAntenna,
                                    Ptr<const PhasedArrayModel> uAntenna) const;
  /**
   * Applies the blockage model A described in 3GPP TR 38.901
   * \param channelParams the channel parameters structure
   * \param clusterAOA vector containing the azimuth angle of arrival for each cluster
   * \param clusterZOA vector containing the zenith angle of arrival for each cluster
   * \return vector containing the power attenuation for each cluster
   */
  DoubleVector CalcAttenuationOfBlockage (const Ptr<ThreeGppChannelModel::ThreeGppChannelParams> channelParams,
                                          const DoubleVector &clusterAOA,
                                          const DoubleVector &clusterZOA) const;

  /**
   * Check if the channel params has to be updated
   * \param channelParams channel params
   * \param channelCondition the channel condition
   * \return true if the channel params has to be updated, false otherwise
   */
  bool ChannelParamsNeedsUpdate (Ptr<const ThreeGppChannelParams> channelParams,
                                 Ptr<const ChannelCondition> channelCondition) const;

  /**
   * Check if the channel matrix has to be updated (it needs update when the channel params generation
   * time is more recent than channel matrix generation time
   * \param channelParams channel params structure
   * \param channelMatrix channel matrix structure
   * \return true if the channel matrix has to be updated, false otherwise
   */
  bool ChannelMatrixNeedsUpdate (Ptr<const ThreeGppChannelParams> channelParams, Ptr<const ChannelMatrix> channelMatrix);

  std::unordered_map<uint64_t, Ptr<ChannelMatrix> > m_channelMatrixMap; //!< map containing the channel realizations per pair of PhasedAntennaArray instances, the key of this map is reciprocal uniquely identifies a pair of PhasedAntennaArrays
  std::unordered_map<uint64_t, Ptr<ThreeGppChannelParams> > m_channelParamsMap; //!< map containing the common channel parameters per pair of nodes, the key of this map is reciprocal and uniquely identifies a pair of nodes
  Time m_updatePeriod; //!< the channel update period
  double m_frequency; //!< the operating frequency
  std::string m_scenario; //!< the 3GPP scenario
  Ptr<ChannelConditionModel> m_channelConditionModel; //!< the channel condition model
  Ptr<UniformRandomVariable> m_uniformRv; //!< uniform random variable
  Ptr<NormalRandomVariable> m_normalRv; //!< normal random variable
  Ptr<UniformRandomVariable> m_uniformRvShuffle; //!< uniform random variable used to shuffle array in GetNewChannel

  // Variable used to compute the additional Doppler contribution for the delayed
  // (reflected) paths, as described in 3GPP TR 37.885 v15.3.0, Sec. 6.2.3.
  double m_vScatt; //!< value used to compute the additional Doppler contribution for the delayed paths
  Ptr<UniformRandomVariable> m_uniformRvDoppler; //!< uniform random variable, used to compute the additional Doppler contribution

  // parameters for the blockage model
  bool m_blockage; //!< enables the blockage model A
  uint16_t m_numNonSelfBlocking; //!< number of non-self-blocking regions
  bool m_portraitMode; //!< true if potrait mode, false if landscape
  double m_blockerSpeed; //!< the blocker speed

  static const uint8_t PHI_INDEX = 0; //!< index of the PHI value in the m_nonSelfBlocking array
  static const uint8_t X_INDEX = 1; //!< index of the X value in the m_nonSelfBlocking array
  static const uint8_t THETA_INDEX = 2; //!< index of the THETA value in the m_nonSelfBlocking array
  static const uint8_t Y_INDEX = 3; //!< index of the Y value in the m_nonSelfBlocking array
  static const uint8_t R_INDEX = 4; //!< index of the R value in the m_nonSelfBlocking array
};
} // namespace ns3

#endif /* THREE_GPP_CHANNEL_H */
