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
 *
 */

#include "ns3/log.h"
#include "three-gpp-channel-model.h"
#include "three-gpp-spectrum-propagation-loss-model.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/simulator.h"
#include "ns3/pointer.h"
#include <map>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ThreeGppSpectrumPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (ThreeGppSpectrumPropagationLossModel);

ThreeGppSpectrumPropagationLossModel::ThreeGppSpectrumPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

ThreeGppSpectrumPropagationLossModel::~ThreeGppSpectrumPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

void
ThreeGppSpectrumPropagationLossModel::DoDispose ()
{
  m_longTermMap.clear ();
  m_channelModel->Dispose ();
  m_channelModel = nullptr;
}

TypeId
ThreeGppSpectrumPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppSpectrumPropagationLossModel")
    .SetParent<PhasedArraySpectrumPropagationLossModel> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<ThreeGppSpectrumPropagationLossModel> ()
    .AddAttribute("ChannelModel", 
                  "The channel model. It needs to implement the MatrixBasedChannelModel interface",
                  StringValue("ns3::ThreeGppChannelModel"),
                  MakePointerAccessor (&ThreeGppSpectrumPropagationLossModel::SetChannelModel,
                                       &ThreeGppSpectrumPropagationLossModel::GetChannelModel),
                                       MakePointerChecker<MatrixBasedChannelModel> ())
    ;
  return tid;
}

void
ThreeGppSpectrumPropagationLossModel::SetChannelModel (Ptr<MatrixBasedChannelModel> channel)
{
  m_channelModel = channel;
}

Ptr<MatrixBasedChannelModel>
ThreeGppSpectrumPropagationLossModel::GetChannelModel () const
{
  return m_channelModel;
}

double
ThreeGppSpectrumPropagationLossModel::GetFrequency () const
{
  DoubleValue freq;
  m_channelModel->GetAttribute ("Frequency", freq);
  return freq.Get ();
}

void
ThreeGppSpectrumPropagationLossModel::SetChannelModelAttribute (const std::string &name, const AttributeValue &value)
{
  m_channelModel->SetAttribute (name, value);
}

void
ThreeGppSpectrumPropagationLossModel::GetChannelModelAttribute (const std::string &name, AttributeValue &value) const
{
  m_channelModel->GetAttribute (name, value);
}

PhasedArrayModel::ComplexVector
ThreeGppSpectrumPropagationLossModel::CalcLongTerm (Ptr<const MatrixBasedChannelModel::ChannelMatrix> params,
                                                    const PhasedArrayModel::ComplexVector &sW,
                                                    const PhasedArrayModel::ComplexVector &uW) const
{
  NS_LOG_FUNCTION (this);

  uint16_t sAntenna = static_cast<uint16_t> (sW.size ());
  uint16_t uAntenna = static_cast<uint16_t> (uW.size ());

  NS_ASSERT (uAntenna == params->m_channel.size ());
  NS_ASSERT (sAntenna == params->m_channel.at (0).size());

  NS_LOG_DEBUG ("CalcLongTerm with sAntenna " << sAntenna << " uAntenna " << uAntenna);
  //store the long term part to reduce computation load
  //only the small scale fading needs to be updated if the large scale parameters and antenna weights remain unchanged.
  PhasedArrayModel::ComplexVector longTerm;
  uint8_t numCluster = static_cast<uint8_t> (params->m_channel[0][0].size ());

  NS_ASSERT (uAntenna == params->m_channel.size ());
  NS_ASSERT (sAntenna == params->m_channel.at (0).size());

  for (uint8_t cIndex = 0; cIndex < numCluster; cIndex++)
    {
      std::complex<double> txSum (0, 0);
      for (uint16_t sIndex = 0; sIndex < sAntenna; sIndex++)
        {
          std::complex<double> rxSum (0, 0);
          for (uint16_t uIndex = 0; uIndex < uAntenna; uIndex++)
            {
              rxSum = rxSum + uW[uIndex] * params->m_channel[uIndex][sIndex][cIndex];
            }
          txSum = txSum + sW[sIndex] * rxSum;
        }
      longTerm.push_back (txSum);
    }
  return longTerm;
}

Ptr<SpectrumValue>
ThreeGppSpectrumPropagationLossModel::CalcBeamformingGain (Ptr<SpectrumValue> txPsd,
                                                           PhasedArrayModel::ComplexVector longTerm,
                                                           Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
                                                           Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams,
                                                           const ns3::Vector &sSpeed, const ns3::Vector &uSpeed) const
{
  NS_LOG_FUNCTION (this);

  Ptr<SpectrumValue> tempPsd = Copy<SpectrumValue> (txPsd);

  //channel[rx][tx][cluster]
  uint8_t numCluster = static_cast<uint8_t> (channelMatrix->m_channel[0][0].size ());

  // compute the doppler term
  // NOTE the update of Doppler is simplified by only taking the center angle of
  // each cluster in to consideration.
  double slotTime = Simulator::Now ().GetSeconds ();
  double factor = 2 * M_PI * slotTime * GetFrequency () / 3e8;
  PhasedArrayModel::ComplexVector doppler;

  // The following asserts might seem paranoic, but it is important to
  // make sure that all the structures that are passed to this function
  // are of the correct dimensions before using the operator [].
  // If you dont understand the comment read about the difference of .at()
  // and [] operators, ...
  NS_ASSERT (numCluster <= channelParams->m_alpha.size ());
  NS_ASSERT (numCluster <= channelParams->m_D.size());
  NS_ASSERT (numCluster <= channelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX].size());
  NS_ASSERT (numCluster <= channelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX].size());
  NS_ASSERT (numCluster <= channelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX].size());
  NS_ASSERT (numCluster <= channelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX].size());
  NS_ASSERT (numCluster <= longTerm.size());

  // check if channelParams structure is generated in direction s-to-u or u-to-s
  bool isSameDirection = (channelParams->m_nodeIds == channelMatrix->m_nodeIds);

  MatrixBasedChannelModel::DoubleVector zoa;
  MatrixBasedChannelModel::DoubleVector zod;
  MatrixBasedChannelModel::DoubleVector aoa;
  MatrixBasedChannelModel::DoubleVector aod;

  // if channel params is generated in the same direction in which we
  // generate the channel matrix, angles and zenit od departure and arrival are ok,
  // just set them to corresponding variable that will be used for the generation
  // of channel matrix, otherwise we need to flip angles and zenits of departure and arrival
  if (isSameDirection)
    {
      zoa = channelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX];
      zod = channelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX];
      aoa = channelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX];
      aod = channelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX];
    }
  else
    {
      zod = channelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX];
      zoa = channelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX];
      aod = channelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX];
      aoa = channelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX];
    }

  for (uint8_t cIndex = 0; cIndex < numCluster; cIndex++)
    {
      // Compute alpha and D as described in 3GPP TR 37.885 v15.3.0, Sec. 6.2.3
      // These terms account for an additional Doppler contribution due to the
      // presence of moving objects in the sorrounding environment, such as in
      // vehicular scenarios.
      // This contribution is applied only to the delayed (reflected) paths and
      // must be properly configured by setting the value of
      // m_vScatt, which is defined as "maximum speed of the vehicle in the
      // layout".
      // By default, m_vScatt is set to 0, so there is no additional Doppler
      // contribution.

      double alpha = channelParams->m_alpha [cIndex];
      double D = channelParams->m_D [cIndex];

      //cluster angle angle[direction][n], where direction = 0(aoa), 1(zoa).
      double tempDoppler = factor * ((sin (zoa [cIndex] * M_PI / 180) * cos (aoa [cIndex] * M_PI / 180) * uSpeed.x
                                       + sin (zoa [cIndex] * M_PI / 180) * sin (aoa [cIndex] * M_PI / 180) * uSpeed.y
                                       + cos (zoa [cIndex] * M_PI / 180) * uSpeed.z)
                                       + (sin (zod [cIndex] * M_PI / 180) * cos (aod [cIndex] * M_PI / 180) * sSpeed.x
                                       + sin (zod [cIndex] * M_PI / 180) * sin (aod [cIndex] * M_PI / 180) * sSpeed.y
                                       + cos (zod [cIndex] * M_PI / 180) * sSpeed.z) + 2 * alpha * D);
      doppler.push_back (std::complex<double> (cos (tempDoppler), sin (tempDoppler)));
    }

  NS_ASSERT (numCluster <= doppler.size());

  // apply the doppler term and the propagation delay to the long term component
  // to obtain the beamforming gain
  auto vit = tempPsd->ValuesBegin (); // psd iterator
  auto sbit = tempPsd->ConstBandsBegin (); // band iterator
  while (vit != tempPsd->ValuesEnd ())
    {
      if ((*vit) != 0.00)
        {
          std::complex<double> subsbandGain (0.0, 0.0);
          double fsb = (*sbit).fc; // center frequency of the sub-band
          for (uint8_t cIndex = 0; cIndex < numCluster; cIndex++)
            {
              double delay = -2 * M_PI * fsb * (channelParams->m_delay[cIndex]);
              subsbandGain = subsbandGain + longTerm[cIndex] * doppler[cIndex] * std::complex<double> (cos (delay), sin (delay));
            }
          *vit = (*vit) * (norm (subsbandGain));
        }
      vit++;
      sbit++;
    }
  return tempPsd;
}

PhasedArrayModel::ComplexVector
ThreeGppSpectrumPropagationLossModel::GetLongTerm (Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
                                                   Ptr<const PhasedArrayModel> aPhasedArrayModel,
                                                   Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
  PhasedArrayModel::ComplexVector longTerm; // vector containing the long term component for each cluster

  // check if the channel matrix was generated considering a as the s-node and
  // b as the u-node or viceversa
  PhasedArrayModel::ComplexVector sW, uW;
  if (!channelMatrix->IsReverse (aPhasedArrayModel->GetId (), bPhasedArrayModel->GetId ()))
  {
    sW = aPhasedArrayModel->GetBeamformingVector ();
    uW = bPhasedArrayModel->GetBeamformingVector ();
  }
  else
  {
    sW = bPhasedArrayModel->GetBeamformingVector ();
    uW = aPhasedArrayModel->GetBeamformingVector ();
  }

  bool update = false; // indicates whether the long term has to be updated
  bool notFound = false; // indicates if the long term has not been computed yet

  // compute the long term key, the key is unique for each tx-rx pair
  uint64_t longTermId = MatrixBasedChannelModel::GetKey (aPhasedArrayModel->GetId (), bPhasedArrayModel->GetId ());

  // look for the long term in the map and check if it is valid
  if (m_longTermMap.find (longTermId) != m_longTermMap.end ())
    {
      NS_LOG_DEBUG ("found the long term component in the map");
      longTerm = m_longTermMap[longTermId]->m_longTerm;

      // check if the channel matrix has been updated
      // or the s beam has been changed
      // or the u beam has been changed
      update = (m_longTermMap[longTermId]->m_channel->m_generatedTime != channelMatrix->m_generatedTime
                || m_longTermMap[longTermId]->m_sW != sW
                || m_longTermMap[longTermId]->m_uW != uW);

    }
  else
    {
      NS_LOG_DEBUG ("long term component NOT found");
      notFound = true;
    }

  if (update || notFound)
    {
      NS_LOG_DEBUG ("compute the long term");
      // compute the long term component
      longTerm = CalcLongTerm (channelMatrix, sW, uW);

      // store the long term
      Ptr<LongTerm> longTermItem = Create<LongTerm> ();
      longTermItem->m_longTerm = longTerm;
      longTermItem->m_channel = channelMatrix;
      longTermItem->m_sW = sW;
      longTermItem->m_uW = uW;

      m_longTermMap[longTermId] = longTermItem;
    }

  return longTerm;
}

Ptr<SpectrumValue>
ThreeGppSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                                    Ptr<const MobilityModel> a,
                                                                    Ptr<const MobilityModel> b,
                                                                    Ptr<const PhasedArrayModel> aPhasedArrayModel,
                                                                    Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
  NS_LOG_FUNCTION (this);
  uint32_t aId = a->GetObject<Node> ()->GetId (); // id of the node a
  uint32_t bId = b->GetObject<Node> ()->GetId (); // id of the node b

  NS_ASSERT (aId != bId);
  NS_ASSERT_MSG (a->GetDistanceFrom (b) > 0.0, "The position of a and b devices cannot be the same");

  Ptr<SpectrumValue> rxPsd = Copy<SpectrumValue> (txPsd);

  // retrieve the antenna of device a
  NS_ASSERT_MSG (aPhasedArrayModel, "Antenna not found for node " << aId);
  NS_LOG_DEBUG ("a node " << a->GetObject<Node> () << " antenna " << aPhasedArrayModel);

  // retrieve the antenna of the device b
  NS_ASSERT_MSG (bPhasedArrayModel, "Antenna not found for device " << bId);
  NS_LOG_DEBUG ("b node " << bId << " antenna " << bPhasedArrayModel);

  Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix = m_channelModel->GetChannel (a, b, aPhasedArrayModel, bPhasedArrayModel);
  Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams = m_channelModel->GetParams (a, b);

  // retrieve the long term component
  PhasedArrayModel::ComplexVector longTerm = GetLongTerm (channelMatrix, aPhasedArrayModel, bPhasedArrayModel);

  // apply the beamforming gain
  rxPsd = CalcBeamformingGain (rxPsd, longTerm, channelMatrix, channelParams, a->GetVelocity (), b->GetVelocity ());

  return rxPsd;
}


}  // namespace ns3
