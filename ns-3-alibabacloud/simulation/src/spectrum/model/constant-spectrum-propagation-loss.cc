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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include <cmath>

#include "ns3/log.h"

#include "ns3/constant-spectrum-propagation-loss.h"
#include "ns3/double.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ConstantSpectrumPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (ConstantSpectrumPropagationLossModel);

ConstantSpectrumPropagationLossModel::ConstantSpectrumPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

ConstantSpectrumPropagationLossModel::~ConstantSpectrumPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
ConstantSpectrumPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ConstantSpectrumPropagationLossModel")
    .SetParent<SpectrumPropagationLossModel> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<ConstantSpectrumPropagationLossModel> ()
    .AddAttribute ("Loss",
                   "Path loss (dB) between transmitter and receiver",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&ConstantSpectrumPropagationLossModel::SetLossDb,
                                       &ConstantSpectrumPropagationLossModel::GetLossDb),
                   MakeDoubleChecker<double> ())
    ;
  return tid;
}


void
ConstantSpectrumPropagationLossModel::SetLossDb (double lossDb)
{
  NS_LOG_FUNCTION (this);
  m_lossDb = lossDb;
  m_lossLinear = std::pow (10, m_lossDb / 10);
}


double
ConstantSpectrumPropagationLossModel::GetLossDb () const
{
  NS_LOG_FUNCTION (this);
  return m_lossDb;
}


Ptr<SpectrumValue>
ConstantSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                                    Ptr<const MobilityModel> a,
                                                                    Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);

  Ptr<SpectrumValue> rxPsd = Copy<SpectrumValue> (txPsd);
  Values::iterator vit = rxPsd->ValuesBegin ();
  Bands::const_iterator fit = rxPsd->ConstBandsBegin ();

  while (vit != rxPsd->ValuesEnd ())
    {
      NS_ASSERT (fit != rxPsd->ConstBandsEnd ());
      NS_LOG_LOGIC ("Ptx = " << *vit);
      *vit /= m_lossLinear; // Prx = Ptx / loss
      NS_LOG_LOGIC ("Prx = " << *vit);
      ++vit;
      ++fit;
    }
  return rxPsd;
}


}  // namespace ns3
