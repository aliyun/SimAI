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

#include "okumura-hata-propagation-loss-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OkumuraHataPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (OkumuraHataPropagationLossModel);


TypeId
OkumuraHataPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OkumuraHataPropagationLossModel")
    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<OkumuraHataPropagationLossModel> ()
    .AddAttribute ("Frequency",
                   "The propagation frequency in Hz",
                   DoubleValue (2160e6),
                   MakeDoubleAccessor (&OkumuraHataPropagationLossModel::m_frequency),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Environment",
                   "Environment Scenario",
                   EnumValue (UrbanEnvironment),
                   MakeEnumAccessor (&OkumuraHataPropagationLossModel::m_environment),
                   MakeEnumChecker (UrbanEnvironment, "Urban",
                                    SubUrbanEnvironment, "SubUrban",
                                    OpenAreasEnvironment, "OpenAreas"))
    .AddAttribute ("CitySize",
                   "Dimension of the city",
                   EnumValue (LargeCity),
                   MakeEnumAccessor (&OkumuraHataPropagationLossModel::m_citySize),
                   MakeEnumChecker (SmallCity, "Small",
                                    MediumCity, "Medium",
                                    LargeCity, "Large"));
  return tid;
}

OkumuraHataPropagationLossModel::OkumuraHataPropagationLossModel ()
  : PropagationLossModel ()
{
}

OkumuraHataPropagationLossModel::~OkumuraHataPropagationLossModel ()
{
}

double
OkumuraHataPropagationLossModel::GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  double loss = 0.0;
  double fmhz = m_frequency / 1e6;
  double dist = a->GetDistanceFrom (b) / 1000.0; 
  if (m_frequency <= 1.500e9)
    {
      // standard Okumura Hata 
      // see eq. (4.4.1) in the COST 231 final report
      double log_f = std::log10 (fmhz);
      double hb = (a->GetPosition ().z > b->GetPosition ().z ? a->GetPosition ().z : b->GetPosition ().z);
      double hm = (a->GetPosition ().z < b->GetPosition ().z ? a->GetPosition ().z : b->GetPosition ().z);
      NS_ASSERT_MSG (hb > 0 && hm > 0, "nodes' height must be greater then 0");
      double log_aHeight = 13.82 * std::log10 (hb);
      double log_bHeight = 0.0;
      if (m_citySize == LargeCity)
        {
          if (fmhz < 200)
            {
              log_bHeight = 8.29 * std::pow (log10 (1.54 * hm), 2) -  1.1;
            }
          else
            {
              log_bHeight = 3.2 * std::pow (log10 (11.75 * hm), 2) - 4.97;
            }
        }
      else
        {
          log_bHeight = 0.8 + (1.1 * log_f - 0.7) * hm - 1.56 * log_f;
        }

      NS_LOG_INFO (this << " logf " << 26.16 * log_f << " loga " << log_aHeight << " X " << (((44.9 - (6.55 * std::log10 (hb)) )) * std::log10 (a->GetDistanceFrom (b))) << " logb " << log_bHeight);
      loss = 69.55 + (26.16 * log_f) - log_aHeight + (((44.9 - (6.55 * std::log10 (hb)) )) * std::log10 (dist)) - log_bHeight;
      if (m_environment == SubUrbanEnvironment)
        {
          loss += -2 * (std::pow (std::log10 (fmhz / 28), 2)) - 5.4;
        }
      else if (m_environment == OpenAreasEnvironment)
        {
          loss += -4.70 * std::pow (std::log10 (fmhz),2) + 18.33 * std::log10 (fmhz) - 40.94;
        }

    }
  else
    {
      // COST 231 Okumura model
      // see eq. (4.4.3) in the COST 231 final report

      double log_f = std::log10 (fmhz);
      double hb = (a->GetPosition ().z > b->GetPosition ().z ? a->GetPosition ().z : b->GetPosition ().z);
      double hm = (a->GetPosition ().z < b->GetPosition ().z ? a->GetPosition ().z : b->GetPosition ().z);
      NS_ASSERT_MSG (hb > 0 && hm > 0, "nodes' height must be greater then 0");
      double log_aHeight = 13.82 * std::log10 (hb);
      double log_bHeight = 0.0;
      double C = 0.0;

      if (m_citySize == LargeCity)
        {
          log_bHeight = 3.2 * std::pow ((std::log10 (11.75 * hm)), 2);
          C = 3;
        }
      else
        {
          log_bHeight = (1.1 * log_f - 0.7) * hm - (1.56 * log_f - 0.8);
        }

      loss = 46.3 + (33.9 * log_f) - log_aHeight + (((44.9 - (6.55 * std::log10 (hb)) )) * std::log10 (dist)) - log_bHeight + C;
    }
  return loss;
}

double 
OkumuraHataPropagationLossModel::DoCalcRxPower (double txPowerDbm,
						Ptr<MobilityModel> a,
						Ptr<MobilityModel> b) const
{
  return (txPowerDbm - GetLoss (a, b));
}

int64_t
OkumuraHataPropagationLossModel::DoAssignStreams (int64_t stream)
{
  return 0;
}


} // namespace ns3
