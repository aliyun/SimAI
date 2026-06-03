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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>,
 *         Nicola Baldo <nbaldo@cttc.es>
 * 
 */
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/mobility-model.h"
#include <cmath>

#include "itu-r-1411-los-propagation-loss-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ItuR1411LosPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (ItuR1411LosPropagationLossModel);

TypeId
ItuR1411LosPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ItuR1411LosPropagationLossModel")
    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ItuR1411LosPropagationLossModel> ()
    .AddAttribute ("Frequency",
                   "The propagation frequency in Hz",
                   DoubleValue (2160e6),
                   MakeDoubleAccessor (&ItuR1411LosPropagationLossModel::SetFrequency),
                   MakeDoubleChecker<double> ());

  return tid;
}

ItuR1411LosPropagationLossModel::ItuR1411LosPropagationLossModel ()
  : PropagationLossModel ()
{
}

ItuR1411LosPropagationLossModel::~ItuR1411LosPropagationLossModel ()
{
}

double
ItuR1411LosPropagationLossModel::GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);
  double dist = a->GetDistanceFrom (b);
  double lossLow = 0.0;
  double lossUp = 0.0;
  NS_ASSERT_MSG (a->GetPosition ().z > 0 && b->GetPosition ().z > 0, "nodes' height must be greater than 0");
  double Lbp = std::fabs (20 * std::log10 ((m_lambda * m_lambda) / (8 * M_PI * a->GetPosition ().z * b->GetPosition ().z)));
  double Rbp = (4 * a->GetPosition ().z * b->GetPosition ().z) / m_lambda;
  NS_LOG_LOGIC (this << " Lbp " << Lbp << " Rbp " << Rbp << " lambda " << m_lambda);
  if (dist <= Rbp)
    {
      lossLow = Lbp + 20 * std::log10 (dist / Rbp);
      lossUp = Lbp + 20 + 25 * std::log10 (dist / Rbp);
    }
  else
    {
      lossLow = Lbp + 40 * std::log10 (dist / Rbp);
      lossUp = Lbp + 20 + 40 * std::log10 (dist / Rbp);
    }

  double loss = (lossUp + lossLow) / 2;

  return loss;
}


void
ItuR1411LosPropagationLossModel::SetFrequency (double freq)
{
  NS_ASSERT (freq > 0.0);
  m_lambda = 299792458.0 / freq;
}


double 
ItuR1411LosPropagationLossModel::DoCalcRxPower (double txPowerDbm,
						Ptr<MobilityModel> a,
						Ptr<MobilityModel> b) const
{
  return (txPowerDbm - GetLoss (a, b));
}

int64_t
ItuR1411LosPropagationLossModel::DoAssignStreams (int64_t stream)
{
  return 0;
}
} // namespace ns3
