/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include <ns3/object.h>
#include <ns3/simulator.h>
#include <ns3/log.h>
#include <ns3/packet.h>
#include <ns3/packet-burst.h>
#include <ns3/net-device.h>
#include <ns3/node.h>
#include <ns3/double.h>
#include <ns3/mobility-model.h>
#include <ns3/spectrum-phy.h>
#include <ns3/spectrum-propagation-loss-model.h>
#include <ns3/propagation-loss-model.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/antenna-model.h>
#include <ns3/angles.h>

#include <algorithm>

#include "single-model-spectrum-channel.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SingleModelSpectrumChannel");

NS_OBJECT_ENSURE_REGISTERED (SingleModelSpectrumChannel);

SingleModelSpectrumChannel::SingleModelSpectrumChannel ()
{
  NS_LOG_FUNCTION (this);
}

void
SingleModelSpectrumChannel::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_phyList.clear ();
  m_spectrumModel = 0;
  SpectrumChannel::DoDispose ();
}

TypeId
SingleModelSpectrumChannel::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static TypeId tid = TypeId ("ns3::SingleModelSpectrumChannel")
    .SetParent<SpectrumChannel> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<SingleModelSpectrumChannel> ()
  ;
  return tid;
}


void
SingleModelSpectrumChannel::RemoveRx (Ptr<SpectrumPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  auto it = std::find (begin (m_phyList), end (m_phyList), phy);
  if (it != std::end (m_phyList))
    {
      m_phyList.erase (it);
    }
}

void
SingleModelSpectrumChannel::AddRx (Ptr<SpectrumPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_phyList.push_back (phy);
}


void
SingleModelSpectrumChannel::StartTx (Ptr<SpectrumSignalParameters> txParams)
{
  NS_LOG_FUNCTION (this << txParams->psd << txParams->duration << txParams->txPhy);
  NS_ASSERT_MSG (txParams->psd, "NULL txPsd");
  NS_ASSERT_MSG (txParams->txPhy, "NULL txPhy");

  Ptr<SpectrumSignalParameters> txParamsTrace = txParams->Copy (); // copy it since traced value cannot be const (because of potential underlying DynamicCasts)
  m_txSigParamsTrace (txParamsTrace);

  // just a sanity check routine. We might want to remove it to save some computational load -- one "if" statement  ;-)
  if (m_spectrumModel == 0)
    {
      // first pak, record SpectrumModel
      m_spectrumModel = txParams->psd->GetSpectrumModel ();
    }
  else
    {
      // all attached SpectrumPhy instances must use the same SpectrumModel
      NS_ASSERT (*(txParams->psd->GetSpectrumModel ()) == *m_spectrumModel);
    }




  Ptr<MobilityModel> senderMobility = txParams->txPhy->GetMobility ();

  for (PhyList::const_iterator rxPhyIterator = m_phyList.begin ();
       rxPhyIterator != m_phyList.end ();
       ++rxPhyIterator)
    {
      Ptr<NetDevice> rxNetDevice = (*rxPhyIterator)->GetDevice ();
      Ptr<NetDevice> txNetDevice = txParams->txPhy->GetDevice ();

      if (rxNetDevice && txNetDevice)
        {
          // we assume that devices are attached to a node
          if (rxNetDevice->GetNode()->GetId() == txNetDevice->GetNode()->GetId())
            {
              NS_LOG_DEBUG ("Skipping the pathloss calculation among different antennas of the same node, not supported yet by any pathloss model in ns-3.");
              continue;
            }
        }

      if ((*rxPhyIterator) != txParams->txPhy)
        {
          Time delay  = MicroSeconds (0);

          Ptr<MobilityModel> receiverMobility = (*rxPhyIterator)->GetMobility ();
          NS_LOG_LOGIC ("copying signal parameters " << txParams);
          Ptr<SpectrumSignalParameters> rxParams = txParams->Copy ();

          if (senderMobility && receiverMobility)
            {
              double txAntennaGain = 0;
              double rxAntennaGain = 0;
              double propagationGainDb = 0;
              double pathLossDb = 0;
              if (rxParams->txAntenna != 0)
                {
                  Angles txAngles (receiverMobility->GetPosition (), senderMobility->GetPosition ());
                  txAntennaGain = rxParams->txAntenna->GetGainDb (txAngles);
                  NS_LOG_LOGIC ("txAntennaGain = " << txAntennaGain << " dB");
                  pathLossDb -= txAntennaGain;
                }
              Ptr<AntennaModel> rxAntenna = DynamicCast<AntennaModel>((*rxPhyIterator)->GetAntenna ());
              if (rxAntenna != 0)
                {
                  Angles rxAngles (senderMobility->GetPosition (), receiverMobility->GetPosition ());
                  rxAntennaGain = rxAntenna->GetGainDb (rxAngles);
                  NS_LOG_LOGIC ("rxAntennaGain = " << rxAntennaGain << " dB");
                  pathLossDb -= rxAntennaGain;
                }
              if (m_propagationLoss)
                {
                  propagationGainDb = m_propagationLoss->CalcRxPower (0, senderMobility, receiverMobility);
                  NS_LOG_LOGIC ("propagationGainDb = " << propagationGainDb << " dB");
                  pathLossDb -= propagationGainDb;
                }                    
              NS_LOG_LOGIC ("total pathLoss = " << pathLossDb << " dB");
              // Gain trace
              m_gainTrace (senderMobility, receiverMobility, txAntennaGain, rxAntennaGain, propagationGainDb, pathLossDb);
              // Pathloss trace
              m_pathLossTrace (txParams->txPhy, *rxPhyIterator, pathLossDb);
              if ( pathLossDb > m_maxLossDb)
                {
                  // beyond range
                  continue;
                }
              double pathGainLinear = std::pow (10.0, (-pathLossDb) / 10.0);
              *(rxParams->psd) *= pathGainLinear;              

              if (m_spectrumPropagationLoss)
                {
                  rxParams->psd = m_spectrumPropagationLoss->CalcRxPowerSpectralDensity (rxParams->psd, senderMobility, receiverMobility);
                }

              if (m_propagationDelay)
                {
                  delay = m_propagationDelay->GetDelay (senderMobility, receiverMobility);
                }
            }


          if (rxNetDevice)
            {
              // the receiver has a NetDevice, so we expect that it is attached to a Node
              uint32_t dstNode =  rxNetDevice->GetNode ()->GetId ();
              Simulator::ScheduleWithContext (dstNode, delay, &SingleModelSpectrumChannel::StartRx, this, rxParams, *rxPhyIterator);
            }
          else
            {
              // the receiver is not attached to a NetDevice, so we cannot assume that it is attached to a node
              Simulator::Schedule (delay, &SingleModelSpectrumChannel::StartRx, this,
                                   rxParams, *rxPhyIterator);
            }
        }
    }
}

void
SingleModelSpectrumChannel::StartRx (Ptr<SpectrumSignalParameters> params, Ptr<SpectrumPhy> receiver)
{
  NS_LOG_FUNCTION (this << params);
  receiver->StartRx (params);
}

std::size_t
SingleModelSpectrumChannel::GetNDevices (void) const
{
  NS_LOG_FUNCTION (this);
  return m_phyList.size ();
}

Ptr<NetDevice>
SingleModelSpectrumChannel::GetDevice (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return m_phyList.at (i)->GetDevice ()->GetObject<NetDevice> ();
}


} // namespace ns3
