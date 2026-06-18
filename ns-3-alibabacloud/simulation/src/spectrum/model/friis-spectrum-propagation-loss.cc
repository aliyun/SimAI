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

#include <ns3/mobility-model.h>
#include <ns3/friis-spectrum-propagation-loss.h>
#include <cmath> // for M_PI


namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (FriisSpectrumPropagationLossModel);


FriisSpectrumPropagationLossModel::FriisSpectrumPropagationLossModel ()
{
}

FriisSpectrumPropagationLossModel::~FriisSpectrumPropagationLossModel ()
{
}


TypeId
FriisSpectrumPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FriisSpectrumPropagationLossModel")
    .SetParent<SpectrumPropagationLossModel> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<FriisSpectrumPropagationLossModel> ()
  ;
  return tid;
}


Ptr<SpectrumValue>
FriisSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity (Ptr<const SpectrumValue> txPsd,
                                                                 Ptr<const MobilityModel> a,
                                                                 Ptr<const MobilityModel> b) const
{
  Ptr<SpectrumValue> rxPsd = Copy<SpectrumValue> (txPsd);
  Values::iterator vit = rxPsd->ValuesBegin ();
  Bands::const_iterator fit = rxPsd->ConstBandsBegin ();

  NS_ASSERT (a);
  NS_ASSERT (b);

  double d = a->GetDistanceFrom (b);

  while (vit != rxPsd->ValuesEnd ())
    {
      NS_ASSERT (fit != rxPsd->ConstBandsEnd ());
      *vit /= CalculateLoss (fit->fc, d); // Prx = Ptx / loss
      ++vit;
      ++fit;
    }
  return rxPsd;
}


double
FriisSpectrumPropagationLossModel::CalculateLoss (double f, double d) const
{
  NS_ASSERT (d >= 0);

  if (d == 0)
    {
      return 1;
    }

  NS_ASSERT (f > 0);
  double loss_sqrt = (4 * M_PI * f * d) / 3e8;
  double loss = loss_sqrt * loss_sqrt;

  if (loss < 1)
    {
      loss = 1;
    }
  return loss;
}











}  // namespace ns3
