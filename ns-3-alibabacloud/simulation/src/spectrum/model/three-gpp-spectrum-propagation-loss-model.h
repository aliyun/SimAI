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

#ifndef THREE_GPP_SPECTRUM_PROPAGATION_LOSS_H
#define THREE_GPP_SPECTRUM_PROPAGATION_LOSS_H

#include <complex.h>
#include <map>
#include <unordered_map>
#include "ns3/matrix-based-channel-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/phased-array-spectrum-propagation-loss-model.h"

namespace ns3 {

class NetDevice;

/**
 * \ingroup spectrum
 * \brief 3GPP Spectrum Propagation Loss Model
 *
 * This class models the frequency dependent propagation phenomena in the way
 * described by 3GPP TR 38.901 document. The main method is DoCalcRxPowerSpectralDensity,
 * which takes as input the power spectral density (PSD) of the transmitted signal,
 * the mobility models of the transmitting node and receiving node, and
 * returns the PSD of the received signal.
 *
 * \see MatrixBasedChannelModel
 * \see PhasedArrayModel
 * \see ChannelCondition
 */
class ThreeGppSpectrumPropagationLossModel : public PhasedArraySpectrumPropagationLossModel
{
public:
  /**
   * Constructor
   */
  ThreeGppSpectrumPropagationLossModel ();

  /**
   * Destructor
   */
  ~ThreeGppSpectrumPropagationLossModel ();

  void DoDispose () override;

  /**
   * Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * Set the channel model object
   * \param channel a pointer to an object implementing the MatrixBasedChannelModel interface
   */
  void SetChannelModel (Ptr<MatrixBasedChannelModel> channel);

  /**
   * Get the channel model object
   * \return a pointer to the object implementing the MatrixBasedChannelModel interface
   */
  Ptr<MatrixBasedChannelModel> GetChannelModel () const;

  /**
   * Sets the value of an attribute belonging to the associated
   * MatrixBasedChannelModel instance
   * \param name name of the attribute
   * \param value the attribute value
   */
  void SetChannelModelAttribute (const std::string &name, const AttributeValue &value);

  /**
   * Returns the value of an attribute belonging to the associated
   * MatrixBasedChannelModel instance
   * \param name name of the attribute
   * \param value where the result should be stored
   */
  void GetChannelModelAttribute (const std::string &name, AttributeValue &value) const;

  /**
   * \brief Computes the received PSD.
   *
   * This function computes the received PSD by applying the 3GPP fast fading
   * model and the beamforming gain.
   * In particular, it retrieves the matrix representing the channel between
   * node a and node b, computes the corresponding long term component, i.e.,
   * the product between the cluster matrices and the TX and RX beamforming
   * vectors (w_rx^T H^n_ab w_tx), and accounts for the Doppler component and
   * the propagation delay.
   * To reduce the computational load, the long term component associated with
   * a certain channel is cached and recomputed only when the channel realization
   * is updated, or when the beamforming vectors change.
   *
   * \param txPsd tx PSD
   * \param a first node mobility model
   * \param b second node mobility model
   * \param aPhasedArrayModel the antenna array of the first node
   * \param bPhasedArrayModel the antenna array of the second node
   * \return the received PSD
   */
  Ptr<SpectrumValue> DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                   Ptr<const MobilityModel> a,
                                                   Ptr<const MobilityModel> b,
                                                   Ptr<const PhasedArrayModel> aPhasedArrayModel,
                                                   Ptr<const PhasedArrayModel> bPhasedArrayModel) const override;

private:
  /**
   * Data structure that stores the long term component for a tx-rx pair
   */
  struct LongTerm : public SimpleRefCount<LongTerm>
  {
    PhasedArrayModel::ComplexVector m_longTerm; //!< vector containing the long term component for each cluster
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> m_channel; //!< pointer to the channel matrix used to compute the long term
    PhasedArrayModel::ComplexVector m_sW; //!< the beamforming vector for the node s used to compute the long term
    PhasedArrayModel::ComplexVector m_uW; //!< the beamforming vector for the node u used to compute the long term
  };

  /**
   * Get the operating frequency
   * \return the operating frequency in Hz
  */
  double GetFrequency () const;

  /**
   * Looks for the long term component in m_longTermMap. If found, checks
   * whether it has to be updated. If not found or if it has to be updated,
   * calls the method CalcLongTerm to compute it.
   * \param channelMatrix the channel matrix
   * \param aPhasedArrayModel the antenna array of the tx device
   * \param bPhasedArrayModel the antenna array of the rx device
   * \return vector containing the long term component for each cluster
   */
  PhasedArrayModel::ComplexVector GetLongTerm (Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
                                               Ptr<const PhasedArrayModel> aPhasedArrayModel,
                                               Ptr<const PhasedArrayModel> bPhasedArrayModel) const;
  /**
   * Computes the long term component
   * \param channelMatrix the channel matrix H
   * \param sW the beamforming vector of the s device
   * \param uW the beamforming vector of the u device
   * \return the long term component
   */
  PhasedArrayModel::ComplexVector CalcLongTerm (Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
                                                const PhasedArrayModel::ComplexVector &sW,
                                                const PhasedArrayModel::ComplexVector &uW) const;

  /**
   * Computes the beamforming gain and applies it to the tx PSD
   * \param txPsd the tx PSD
   * \param longTerm the long term component
   * \param channelMatrix The channel matrix structure
   * \param channelParams The channel params structure
   * \param sSpeed speed of the first node
   * \param uSpeed speed of the second node
   * \return the rx PSD
   */
  Ptr<SpectrumValue> CalcBeamformingGain (Ptr<SpectrumValue> txPsd,
                                          PhasedArrayModel::ComplexVector longTerm,
                                          Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
                                          Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams,
                                          const Vector &sSpeed, const Vector &uSpeed) const;

  mutable std::unordered_map < uint64_t, Ptr<const LongTerm> > m_longTermMap; //!< map containing the long term components
  Ptr<MatrixBasedChannelModel> m_channelModel; //!< the model to generate the channel matrix
};
} // namespace ns3

#endif /* THREE_GPP_SPECTRUM_PROPAGATION_LOSS_H */
