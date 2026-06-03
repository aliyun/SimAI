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

#include "itu-r-1411-nlos-over-rooftop-propagation-loss-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ItuR1411NlosOverRooftopPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (ItuR1411NlosOverRooftopPropagationLossModel);


TypeId
ItuR1411NlosOverRooftopPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ItuR1411NlosOverRooftopPropagationLossModel")
    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ItuR1411NlosOverRooftopPropagationLossModel> ()
    .AddAttribute ("Frequency",
                   "The Frequency  (default is 2.106 GHz).",
                   DoubleValue (2160e6),
                   MakeDoubleAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::SetFrequency),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Environment",
                   "Environment Scenario",
                   EnumValue (UrbanEnvironment),
                   MakeEnumAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_environment),
                   MakeEnumChecker (UrbanEnvironment, "Urban",
                                    SubUrbanEnvironment, "SubUrban",
                                    OpenAreasEnvironment, "OpenAreas"))
    .AddAttribute ("CitySize",
                   "Dimension of the city",
                   EnumValue (LargeCity),
                   MakeEnumAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_citySize),
                   MakeEnumChecker (SmallCity, "Small",
                                    MediumCity, "Medium",
                                    LargeCity, "Large"))
    .AddAttribute ("RooftopLevel",
                   "The height of the rooftop level in meters",
                   DoubleValue (20.0),
                   MakeDoubleAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_rooftopHeight),
                   MakeDoubleChecker<double> (0.0, 90.0))
    .AddAttribute ("StreetsOrientation",
                   "The orientation of streets in degrees [0,90] with respect to the direction of propagation",
                   DoubleValue (45.0),
                   MakeDoubleAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_streetsOrientation),
                   MakeDoubleChecker<double> (0.0, 90.0))
    .AddAttribute ("StreetsWidth",
                   "The width of streets",
                   DoubleValue (20.0),
                   MakeDoubleAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_streetsWidth),
                   MakeDoubleChecker<double> (0.0, 1000.0))
    .AddAttribute ("BuildingsExtend",
                   "The distance over which the buildings extend",
                   DoubleValue (80.0),
                   MakeDoubleAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_buildingsExtend),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("BuildingSeparation",
                   "The separation between buildings",
                   DoubleValue (50.0),
                   MakeDoubleAccessor (&ItuR1411NlosOverRooftopPropagationLossModel::m_buildingSeparation),
                   MakeDoubleChecker<double> ());

  return tid;
}

ItuR1411NlosOverRooftopPropagationLossModel::ItuR1411NlosOverRooftopPropagationLossModel ()
  : PropagationLossModel ()
{
}

ItuR1411NlosOverRooftopPropagationLossModel::~ItuR1411NlosOverRooftopPropagationLossModel ()
{
}

double
ItuR1411NlosOverRooftopPropagationLossModel::GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  NS_LOG_FUNCTION (this << a << b);
  double Lori = 0.0;
  double fmhz = m_frequency / 1e6;

  NS_ASSERT_MSG (((m_streetsOrientation >= 0) && (m_streetsOrientation <= 90)),
                 " Street Orientation must be in [0,90]");
  if (m_streetsOrientation < 35)
    {
      Lori = -10.0 + 0.354 * m_streetsOrientation;
    }
  else if ((m_streetsOrientation >= 35)&&(m_streetsOrientation < 55))
    {
      Lori = 2.5 + 0.075 * (m_streetsOrientation - 35);
    }
  else // m_streetsOrientation >= 55
    {
      Lori = 2.5 + 0.075 * (m_streetsOrientation - 55);
    }

  double distance = a->GetDistanceFrom (b);
  double hb = (a->GetPosition ().z > b->GetPosition ().z ? a->GetPosition ().z : b->GetPosition ().z);
  double hm = (a->GetPosition ().z < b->GetPosition ().z ? a->GetPosition ().z : b->GetPosition ().z);
  NS_ASSERT_MSG (hm > 0 && hb > 0, "nodes' height must be greater then 0");
  double Dhb = hb - m_rooftopHeight;
  double ds = (m_lambda * distance * distance) / (Dhb * Dhb);
  double Lmsd = 0.0;
  NS_LOG_LOGIC (this << " build " << m_buildingsExtend << " ds " << ds << " roof " << m_rooftopHeight << " hb " << hb << " lambda " << m_lambda);
  if (ds < m_buildingsExtend)
    {
      double Lbsh = 0.0;
      double ka = 0.0;
      double kd = 0.0;
      double kf = 0.0;
      if (hb > m_rooftopHeight)
        {
          Lbsh = -18 * std::log10 (1 + Dhb);
          ka = (fmhz > 2000 ? 71.4 : 54.0);
          kd = 18.0;
        }
      else 
        {
          Lbsh = 0;
          kd = 18.0 - 15 * Dhb / a->GetPosition ().z;
          if (distance < 500)
            {
              ka = 54.0 - 1.6 * Dhb * distance / 1000;
            }
          else
            {
              ka = 54.0 - 0.8 * Dhb;
            }
        }
      if (fmhz > 2000)
        {
          kf = -8;
        }
      else if ((m_environment == UrbanEnvironment)&&(m_citySize == LargeCity))
        {
          kf = -4 + 0.7 * (fmhz / 925.0 - 1);
        }
      else
        {
          kf = -4 + 1.5 * (fmhz / 925.0 - 1);
        }

      Lmsd = Lbsh + ka + kd * std::log10 (distance / 1000.0) + kf * std::log10 (fmhz) - 9.0 * std::log10 (m_buildingSeparation);
    }
  else
    {
      double theta = std::atan (Dhb / m_buildingSeparation);
      double rho = std::sqrt (Dhb * Dhb + m_buildingSeparation * m_buildingSeparation);
      double Qm = 0.0;
      if ((hb > m_rooftopHeight - 1.0) && (hb < m_rooftopHeight + 1.0))
        {
          Qm = m_buildingSeparation / distance;
        }
      else if (hb > m_rooftopHeight)
        {
          Qm = 2.35 * pow (Dhb / distance * std::sqrt (m_buildingSeparation / m_lambda), 0.9);
        }
      else
        {
          Qm = m_buildingSeparation / (2 * M_PI * distance) * std::sqrt (m_lambda / rho) * (1 / theta - (1 / (2 * M_PI + theta)));
        }
      Lmsd = -10 * std::log10 (Qm * Qm);
    }
  double Lbf = 32.4 + 20 * std::log10 (distance / 1000) + 20 * std::log10 (fmhz);
  double Dhm = m_rooftopHeight - hm;
  double Lrts = -8.2 - 10 * std::log10 (m_streetsWidth) + 10 * std::log10 (fmhz) + 20 * std::log10 (Dhm) + Lori;
  NS_LOG_LOGIC (this << " Lbf " << Lbf << " Lrts " << Lrts << " Dhm" << Dhm << " Lmsd "  << Lmsd);
  double loss = 0.0;
  if (Lrts + Lmsd > 0)
    {
      loss = Lbf + Lrts + Lmsd;
    }
  else
    {
      loss = Lbf;
    }
  return loss;
}


void
ItuR1411NlosOverRooftopPropagationLossModel::SetFrequency (double freq)
{
  m_frequency = freq;
  m_lambda = 299792458.0 / freq;
}


double 
ItuR1411NlosOverRooftopPropagationLossModel::DoCalcRxPower (double txPowerDbm,
						Ptr<MobilityModel> a,
						Ptr<MobilityModel> b) const
{
  return (txPowerDbm - GetLoss (a, b));
}

int64_t
ItuR1411NlosOverRooftopPropagationLossModel::DoAssignStreams (int64_t stream)
{
  return 0;
}


} // namespace ns3
