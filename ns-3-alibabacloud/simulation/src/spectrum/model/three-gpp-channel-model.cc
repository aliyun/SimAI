/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
 * University of Padova
 * Copyright (c) 2015, NYU WIRELESS, Tandon School of Engineering,
 * New York University
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

#include "three-gpp-channel-model.h"
#include "ns3/log.h"
#include "ns3/phased-array-model.h"
#include "ns3/node.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/integer.h"
#include <algorithm>
#include <random>
#include "ns3/log.h"
#include <ns3/simulator.h>
#include "ns3/mobility-model.h"
#include "ns3/pointer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ThreeGppChannelModel");

NS_OBJECT_ENSURE_REGISTERED (ThreeGppChannelModel);

/// The ray offset angles within a cluster, given for rms angle spread normalized to 1. (Table 7.5-3)
static const double offSetAlpha[20] = {
  0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
  0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551};

/**
 * The square root matrix for <em>RMa LOS</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 2 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 *
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_RMa_LOS[7][7] = {
  {1, 0, 0, 0, 0, 0, 0},
  {0, 1, 0, 0, 0, 0, 0},
  {-0.5, 0, 0.866025, 0, 0, 0, 0},
  {0, 0, 0, 1, 0, 0, 0},
  {0, 0, 0, 0, 1, 0, 0},
  {0.01, 0, -0.0519615, 0.73, -0.2, 0.651383, 0},
  {-0.17, -0.02, 0.21362, -0.14, 0.24, 0.142773, 0.909661},
};

/**
 * The square root matrix for <em>RMa NLOS</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 2 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * The parameter K is ignored.
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_RMa_NLOS[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {-0.5, 0.866025, 0, 0, 0, 0},
  {0.6, -0.11547, 0.791623, 0, 0, 0},
  {0, 0, 0, 1, 0, 0},
  {-0.04, -0.138564, 0.540662, -0.18, 0.809003, 0},
  {-0.25, -0.606218, -0.240013, 0.26, -0.231685, 0.625392},
};

/**
 * The square root matrix for <em>RMa O2I</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 2 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_RMa_O2I[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {0, 1, 0, 0, 0, 0},
  {0, 0, 1, 0, 0, 0},
  {0, 0, -0.7, 0.714143, 0, 0},
  {0, 0, 0.66, -0.123225, 0.741091, 0},
  {0, 0, 0.47, 0.152631, -0.393194, 0.775373},
};

/**
 * The square root matrix for <em>UMa LOS</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 1 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 *
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_UMa_LOS[7][7] = {
  {1, 0, 0, 0, 0, 0, 0},
  {0, 1, 0, 0, 0, 0, 0},
  {-0.4, -0.4, 0.824621, 0, 0, 0, 0},
  {-0.5, 0, 0.242536, 0.83137, 0, 0, 0},
  {-0.5, -0.2, 0.630593, -0.484671, 0.278293, 0, 0},
  {0, 0, -0.242536, 0.672172, 0.642214, 0.27735, 0},
  {-0.8, 0, -0.388057, -0.367926, 0.238537, -3.58949e-15, 0.130931},
};

/**
 * The square root matrix for <em>UMa NLOS</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 1 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * The parameter K is ignored.
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_UMa_NLOS[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {-0.4, 0.916515, 0, 0, 0, 0},
  {-0.6, 0.174574, 0.78072, 0, 0, 0},
  {0, 0.654654, 0.365963, 0.661438, 0, 0},
  {0, -0.545545, 0.762422, 0.118114, 0.327327, 0},
  {-0.4, -0.174574, -0.396459, 0.392138, 0.49099, 0.507445},
};

/**
 * The square root matrix for <em>UMa O2I</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 1 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_UMa_O2I[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {-0.5, 0.866025, 0, 0, 0, 0},
  {0.2, 0.57735, 0.791623, 0, 0, 0},
  {0, 0.46188, -0.336861, 0.820482, 0, 0},
  {0, -0.69282, 0.252646, 0.493742, 0.460857, 0},
  {0, -0.23094, 0.16843, 0.808554, -0.220827, 0.464515},

};

/**
 * The square root matrix for <em>UMi LOS</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 1 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 *
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_UMi_LOS[7][7] = {
  {1, 0, 0, 0, 0, 0, 0},
  {0.5, 0.866025, 0, 0, 0, 0, 0},
  {-0.4, -0.57735, 0.711805, 0, 0, 0, 0},
  {-0.5, 0.057735, 0.468293, 0.726201, 0, 0, 0},
  {-0.4, -0.11547, 0.805464, -0.23482, 0.350363, 0, 0},
  {0, 0, 0, 0.688514, 0.461454, 0.559471, 0},
  {0, 0, 0.280976, 0.231921, -0.490509, 0.11916, 0.782603},
};

/**
 * The square root matrix for <em>UMi NLOS</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 1 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * The parameter K is ignored.
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_UMi_NLOS[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {-0.7, 0.714143, 0, 0, 0, 0},
  {0, 0, 1, 0, 0, 0},
  {-0.4, 0.168034, 0, 0.90098, 0, 0},
  {0, -0.70014, 0.5, 0.130577, 0.4927, 0},
  {0, 0, 0.5, 0.221981, -0.566238, 0.616522},
};

/**
 * The square root matrix for <em>UMi O2I</em>, which is generated using the
 * Cholesky decomposition according to table 7.5-6 Part 1 and follows the order
 * of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_UMi_O2I[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {-0.5, 0.866025, 0, 0, 0, 0},
  {0.2, 0.57735, 0.791623, 0, 0, 0},
  {0, 0.46188, -0.336861, 0.820482, 0, 0},
  {0, -0.69282, 0.252646, 0.493742, 0.460857, 0},
  {0, -0.23094, 0.16843, 0.808554, -0.220827, 0.464515},
};

/**
 * The square root matrix for <em>Indoor-Office LOS</em>, which is generated
 * using the Cholesky decomposition according to table 7.5-6 Part 2 and follows
 * the order of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 *
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_office_LOS[7][7] = {
  {1, 0, 0, 0, 0, 0, 0},
  {0.5, 0.866025, 0, 0, 0, 0, 0},
  {-0.8, -0.11547, 0.588784, 0, 0, 0, 0},
  {-0.4, 0.23094, 0.520847, 0.717903, 0, 0, 0},
  {-0.5, 0.288675, 0.73598, -0.348236, 0.0610847, 0, 0},
  {0.2, -0.11547, 0.418943, 0.541106, 0.219905, 0.655744, 0},
  {0.3, -0.057735, 0.73598, -0.348236, 0.0610847, -0.304997, 0.383375},
};

/**
 * The square root matrix for <em>Indoor-Office NLOS</em>, which is generated
 * using the Cholesky decomposition according to table 7.5-6 Part 2 and follows
 * the order of [SF, K, DS, ASD, ASA, ZSD, ZSA].
 * The parameter K is ignored.
 * 
 * The Matlab file to generate the matrices can be found in
 * https://github.com/nyuwireless-unipd/ns3-mmwave/blob/master/src/mmwave/model/BeamFormingMatrix/SqrtMatrix.m
 */
static const double sqrtC_office_NLOS[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {-0.5, 0.866025, 0, 0, 0, 0},
  {0, 0.46188, 0.886942, 0, 0, 0},
  {-0.4, -0.23094, 0.120263, 0.878751, 0, 0},
  {0, -0.311769, 0.55697, -0.249198, 0.728344, 0},
  {0, -0.069282, 0.295397, 0.430696, 0.468462, 0.709214},
};

ThreeGppChannelModel::ThreeGppChannelModel ()
{
  NS_LOG_FUNCTION (this);
  m_uniformRv = CreateObject<UniformRandomVariable> ();
  m_uniformRvShuffle = CreateObject<UniformRandomVariable> ();
  m_uniformRvDoppler = CreateObject<UniformRandomVariable> ();

  m_normalRv = CreateObject<NormalRandomVariable> ();
  m_normalRv->SetAttribute ("Mean", DoubleValue (0.0));
  m_normalRv->SetAttribute ("Variance", DoubleValue (1.0));
}

ThreeGppChannelModel::~ThreeGppChannelModel ()
{
  NS_LOG_FUNCTION (this);
}

void
ThreeGppChannelModel::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  if (m_channelConditionModel)
    {
      m_channelConditionModel->Dispose ();
    }
  m_channelMatrixMap.clear ();
  m_channelParamsMap.clear ();
  m_channelConditionModel = nullptr;
}

TypeId
ThreeGppChannelModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppChannelModel")
    .SetGroupName ("Spectrum")
    .SetParent<MatrixBasedChannelModel> ()
    .AddConstructor<ThreeGppChannelModel> ()
    .AddAttribute ("Frequency",
                   "The operating Frequency in Hz",
                   DoubleValue (500.0e6),
                   MakeDoubleAccessor (&ThreeGppChannelModel::SetFrequency,
                                       &ThreeGppChannelModel::GetFrequency),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Scenario",
                   "The 3GPP scenario (RMa, UMa, UMi-StreetCanyon, InH-OfficeOpen, InH-OfficeMixed)",
                   StringValue ("UMa"),
                   MakeStringAccessor (&ThreeGppChannelModel::SetScenario,
                                       &ThreeGppChannelModel::GetScenario),
                   MakeStringChecker ())
    .AddAttribute ("ChannelConditionModel",
                   "Pointer to the channel condition model",
                   PointerValue (),
                   MakePointerAccessor (&ThreeGppChannelModel::SetChannelConditionModel,
                                        &ThreeGppChannelModel::GetChannelConditionModel),
                   MakePointerChecker<ChannelConditionModel> ())
    .AddAttribute ("UpdatePeriod",
                   "Specify the channel coherence time",
                   TimeValue (MilliSeconds (0)),
                   MakeTimeAccessor (&ThreeGppChannelModel::m_updatePeriod),
                   MakeTimeChecker ())
    // attributes for the blockage model
    .AddAttribute ("Blockage",
                   "Enable blockage model A (sec 7.6.4.1)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&ThreeGppChannelModel::m_blockage),
                   MakeBooleanChecker ())
    .AddAttribute ("NumNonselfBlocking",
                   "number of non-self-blocking regions",
                   IntegerValue (4),
                   MakeIntegerAccessor (&ThreeGppChannelModel::m_numNonSelfBlocking),
                   MakeIntegerChecker<uint16_t> ())
    .AddAttribute ("PortraitMode",
                   "true for portrait mode, false for landscape mode",
                   BooleanValue (true),
                   MakeBooleanAccessor (&ThreeGppChannelModel::m_portraitMode),
                   MakeBooleanChecker ())
    .AddAttribute ("BlockerSpeed",
                   "The speed of moving blockers, the unit is m/s",
                   DoubleValue (1),
                   MakeDoubleAccessor (&ThreeGppChannelModel::m_blockerSpeed),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("vScatt",
                   "Maximum speed of the vehicle in the layout (see 3GPP TR 37.885 v15.3.0, Sec. 6.2.3)."
                   "Used to compute the additional contribution for the Doppler of"
                   "delayed (reflected) paths",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&ThreeGppChannelModel::m_vScatt),
                   MakeDoubleChecker<double> (0.0))

  ;
  return tid;
}

void
ThreeGppChannelModel::SetChannelConditionModel (Ptr<ChannelConditionModel> model)
{
  NS_LOG_FUNCTION (this);
  m_channelConditionModel = model;
}

Ptr<ChannelConditionModel>
ThreeGppChannelModel::GetChannelConditionModel () const
{
  NS_LOG_FUNCTION (this);
  return m_channelConditionModel;
}

void
ThreeGppChannelModel::SetFrequency (double f)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (f >= 500.0e6 && f <= 100.0e9, "Frequency should be between 0.5 and 100 GHz but is " << f);
  m_frequency = f;
}

double
ThreeGppChannelModel::GetFrequency () const
{
  NS_LOG_FUNCTION (this);
  return m_frequency;
}

void
ThreeGppChannelModel::SetScenario (const std::string &scenario)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (scenario == "RMa" || scenario == "UMa" || scenario == "UMi-StreetCanyon"
                 || scenario == "InH-OfficeOpen" || scenario == "InH-OfficeMixed"
                 || scenario == "V2V-Urban" || scenario == "V2V-Highway",
                 "Unknown scenario, choose between: RMa, UMa, UMi-StreetCanyon, "
                 "InH-OfficeOpen, InH-OfficeMixed, V2V-Urban or V2V-Highway");
  m_scenario = scenario;
}

std::string
ThreeGppChannelModel::GetScenario () const
{
  NS_LOG_FUNCTION (this);
  return m_scenario;
}

Ptr<const ThreeGppChannelModel::ParamsTable>
ThreeGppChannelModel::GetThreeGppTable (Ptr<const ChannelCondition> channelCondition, double hBS,
                                        double hUT, double distance2D) const
{
  NS_LOG_FUNCTION (this);

  double fcGHz = m_frequency / 1e9;
  Ptr<ParamsTable> table3gpp = Create<ParamsTable> ();
  // table3gpp includes the following parameters:
  // numOfCluster, raysPerCluster, uLgDS, sigLgDS, uLgASD, sigLgASD,
  // uLgASA, sigLgASA, uLgZSA, sigLgZSA, uLgZSD, sigLgZSD, offsetZOD,
  // cDS, cASD, cASA, cZSA, uK, sigK, rTau, uXpr, sigXpr, shadowingStd

  bool los = channelCondition->IsLos ();
  bool o2i = channelCondition->IsO2i ();

  // In NLOS case, parameter uK and sigK are not used and they are set to 0
  if (m_scenario == "RMa")
    {
      if (los && !o2i)
        {
          // 3GPP mentioned that 3.91 ns should be used when the Cluster DS (cDS)
          // entry is N/A.
          table3gpp->m_numOfCluster = 11;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -7.49;
          table3gpp->m_sigLgDS = 0.55;
          table3gpp->m_uLgASD = 0.90;
          table3gpp->m_sigLgASD = 0.38;
          table3gpp->m_uLgASA = 1.52;
          table3gpp->m_sigLgASA = 0.24;
          table3gpp->m_uLgZSA = 0.47;
          table3gpp->m_sigLgZSA = 0.40;
          table3gpp->m_uLgZSD = 0.34;
          table3gpp->m_sigLgZSD = std::max (-1.0, -0.17 * (distance2D / 1000) - 0.01 * (hUT - 1.5) + 0.22);
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 3.91e-9;
          table3gpp->m_cASD = 2;
          table3gpp->m_cASA = 3;
          table3gpp->m_cZSA = 3;
          table3gpp->m_uK = 7;
          table3gpp->m_sigK = 4;
          table3gpp->m_rTau = 3.8;
          table3gpp->m_uXpr = 12;
          table3gpp->m_sigXpr = 4;
          table3gpp->m_perClusterShadowingStd = 3;

          for (uint8_t row = 0; row < 7; row++)
            {
              for (uint8_t column = 0; column < 7; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_RMa_LOS[row][column];
                }
            }
        }
      else if (!los && !o2i)
        {
          table3gpp->m_numOfCluster = 10;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -7.43;
          table3gpp->m_sigLgDS = 0.48;
          table3gpp->m_uLgASD = 0.95;
          table3gpp->m_sigLgASD = 0.45;
          table3gpp->m_uLgASA = 1.52;
          table3gpp->m_sigLgASA = 0.13;
          table3gpp->m_uLgZSA = 0.58;
          table3gpp->m_sigLgZSA = 0.37;
          table3gpp->m_uLgZSD = std::max (-1.0, -0.19 * (distance2D / 1000) - 0.01 * (hUT - 1.5) + 0.28);
          table3gpp->m_sigLgZSD = 0.30;
          table3gpp->m_offsetZOD = atan ((35 - 3.5) / distance2D) - atan ((35 - 1.5) / distance2D);
          table3gpp->m_cDS = 3.91e-9;
          table3gpp->m_cASD = 2;
          table3gpp->m_cASA = 3;
          table3gpp->m_cZSA = 3;
          table3gpp->m_uK = 0;
          table3gpp->m_sigK = 0;
          table3gpp->m_rTau = 1.7;
          table3gpp->m_uXpr = 7;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 3;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_RMa_NLOS[row][column];
                }
            }
        }
      else // o2i
        {
          table3gpp->m_numOfCluster = 10;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -7.47;
          table3gpp->m_sigLgDS = 0.24;
          table3gpp->m_uLgASD = 0.67;
          table3gpp->m_sigLgASD = 0.18;
          table3gpp->m_uLgASA = 1.66;
          table3gpp->m_sigLgASA = 0.21;
          table3gpp->m_uLgZSA = 0.93;
          table3gpp->m_sigLgZSA = 0.22;
          table3gpp->m_uLgZSD = std::max (-1.0, -0.19 * (distance2D / 1000) - 0.01 * (hUT - 1.5) + 0.28);
          table3gpp->m_sigLgZSD = 0.30;
          table3gpp->m_offsetZOD = atan ((35 - 3.5) / distance2D) - atan ((35 - 1.5) / distance2D);
          table3gpp->m_cDS = 3.91e-9;
          table3gpp->m_cASD = 2;
          table3gpp->m_cASA = 3;
          table3gpp->m_cZSA = 3;
          table3gpp->m_uK = 0;
          table3gpp->m_sigK = 0;
          table3gpp->m_rTau = 1.7;
          table3gpp->m_uXpr = 7;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 3;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_RMa_O2I[row][column];
                }
            }
        }
    }
  else if (m_scenario == "UMa")
    {
      if (los && !o2i)
        {
          table3gpp->m_numOfCluster = 12;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -6.955 - 0.0963 * log10 (fcGHz);
          table3gpp->m_sigLgDS = 0.66;
          table3gpp->m_uLgASD = 1.06 + 0.1114 * log10 (fcGHz);
          table3gpp->m_sigLgASD = 0.28;
          table3gpp->m_uLgASA = 1.81;
          table3gpp->m_sigLgASA = 0.20;
          table3gpp->m_uLgZSA = 0.95;
          table3gpp->m_sigLgZSA = 0.16;
          table3gpp->m_uLgZSD = std::max (-0.5, -2.1 * distance2D / 1000 - 0.01 * (hUT - 1.5) + 0.75);
          table3gpp->m_sigLgZSD = 0.40;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = std::max (0.25, -3.4084 * log10 (fcGHz) + 6.5622) * 1e-9;
          table3gpp->m_cASD = 5;
          table3gpp->m_cASA = 11;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 9;
          table3gpp->m_sigK = 3.5;
          table3gpp->m_rTau = 2.5;
          table3gpp->m_uXpr = 8;
          table3gpp->m_sigXpr = 4;
          table3gpp->m_perClusterShadowingStd = 3;

          for (uint8_t row = 0; row < 7; row++)
            {
              for (uint8_t column = 0; column < 7; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMa_LOS[row][column];
                }
            }
        }
      else
        {
          double uLgZSD = std::max (-0.5, -2.1 * distance2D / 1000 - 0.01 * (hUT - 1.5) + 0.9);

          double afc = 0.208 * log10 (fcGHz) - 0.782;
          double bfc = 25;
          double cfc = -0.13 * log10 (fcGHz) + 2.03;
          double efc = 7.66 * log10 (fcGHz) - 5.96;

          double offsetZOD = efc - std::pow (10, afc * log10 (std::max (bfc, distance2D)) + cfc);

          if (!los && !o2i)
            {
              table3gpp->m_numOfCluster = 20;
              table3gpp->m_raysPerCluster = 20;
              table3gpp->m_uLgDS = -6.28 - 0.204 * log10 (fcGHz);
              table3gpp->m_sigLgDS = 0.39;
              table3gpp->m_uLgASD = 1.5 - 0.1144 * log10 (fcGHz);
              table3gpp->m_sigLgASD = 0.28;
              table3gpp->m_uLgASA = 2.08 - 0.27 * log10 (fcGHz);
              table3gpp->m_sigLgASA = 0.11;
              table3gpp->m_uLgZSA = -0.3236 * log10 (fcGHz) + 1.512;
              table3gpp->m_sigLgZSA = 0.16;
              table3gpp->m_uLgZSD = uLgZSD;
              table3gpp->m_sigLgZSD = 0.49;
              table3gpp->m_offsetZOD = offsetZOD;
              table3gpp->m_cDS = std::max (0.25, -3.4084 * log10 (fcGHz) + 6.5622) * 1e-9;
              table3gpp->m_cASD = 2;
              table3gpp->m_cASA = 15;
              table3gpp->m_cZSA = 7;
              table3gpp->m_uK = 0;
              table3gpp->m_sigK = 0;
              table3gpp->m_rTau = 2.3;
              table3gpp->m_uXpr = 7;
              table3gpp->m_sigXpr = 3;
              table3gpp->m_perClusterShadowingStd = 3;

              for (uint8_t row = 0; row < 6; row++)
                {
                  for (uint8_t column = 0; column < 6; column++)
                    {
                      table3gpp->m_sqrtC[row][column] = sqrtC_UMa_NLOS[row][column];
                    }
                }
            }
          else //(o2i)
            {
              table3gpp->m_numOfCluster = 12;
              table3gpp->m_raysPerCluster = 20;
              table3gpp->m_uLgDS = -6.62;
              table3gpp->m_sigLgDS = 0.32;
              table3gpp->m_uLgASD = 1.25;
              table3gpp->m_sigLgASD = 0.42;
              table3gpp->m_uLgASA = 1.76;
              table3gpp->m_sigLgASA = 0.16;
              table3gpp->m_uLgZSA = 1.01;
              table3gpp->m_sigLgZSA = 0.43;
              table3gpp->m_uLgZSD = uLgZSD;
              table3gpp->m_sigLgZSD = 0.49;
              table3gpp->m_offsetZOD = offsetZOD;
              table3gpp->m_cDS = 11e-9;
              table3gpp->m_cASD = 5;
              table3gpp->m_cASA = 8;
              table3gpp->m_cZSA = 3;
              table3gpp->m_uK = 0;
              table3gpp->m_sigK = 0;
              table3gpp->m_rTau = 2.2;
              table3gpp->m_uXpr = 9;
              table3gpp->m_sigXpr = 5;
              table3gpp->m_perClusterShadowingStd = 4;

              for (uint8_t row = 0; row < 6; row++)
                {
                  for (uint8_t column = 0; column < 6; column++)
                    {
                      table3gpp->m_sqrtC[row][column] = sqrtC_UMa_O2I[row][column];
                    }
                }

            }

        }

    }
  else if (m_scenario == "UMi-StreetCanyon")
    {
      if (los && !o2i)
        {
          table3gpp->m_numOfCluster = 12;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.24 * log10 (1 + fcGHz) - 7.14;
          table3gpp->m_sigLgDS = 0.38;
          table3gpp->m_uLgASD = -0.05 * log10 (1 + fcGHz) + 1.21;
          table3gpp->m_sigLgASD = 0.41;
          table3gpp->m_uLgASA = -0.08 * log10 (1 + fcGHz) + 1.73;
          table3gpp->m_sigLgASA = 0.014 * log10 (1 + fcGHz) + 0.28;
          table3gpp->m_uLgZSA = -0.1 * log10 (1 + fcGHz) + 0.73;
          table3gpp->m_sigLgZSA = -0.04 * log10 (1 + fcGHz) + 0.34;
          table3gpp->m_uLgZSD = std::max (-0.21, -14.8 * distance2D / 1000 + 0.01 * std::abs (hUT - hBS) + 0.83);
          table3gpp->m_sigLgZSD = 0.35;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 5e-9;
          table3gpp->m_cASD = 3;
          table3gpp->m_cASA = 17;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 9;
          table3gpp->m_sigK = 5;
          table3gpp->m_rTau = 3;
          table3gpp->m_uXpr = 9;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 3;

          for (uint8_t row = 0; row < 7; row++)
            {
              for (uint8_t column = 0; column < 7; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_LOS[row][column];
                }
            }
        }
      else
        {
          double uLgZSD = std::max (-0.5, -3.1 * distance2D / 1000 + 0.01 * std::max (hUT - hBS, 0.0) + 0.2);
          double offsetZOD = -1 * std::pow (10, -1.5 * log10 (std::max (10.0, distance2D)) + 3.3);
          if (!los && !o2i)
            {
              table3gpp->m_numOfCluster = 19;
              table3gpp->m_raysPerCluster = 20;
              table3gpp->m_uLgDS = -0.24 * log10 (1 + fcGHz) - 6.83;
              table3gpp->m_sigLgDS = 0.16 * log10 (1 + fcGHz) + 0.28;
              table3gpp->m_uLgASD = -0.23 * log10 (1 + fcGHz) + 1.53;
              table3gpp->m_sigLgASD = 0.11 * log10 (1 + fcGHz) + 0.33;
              table3gpp->m_uLgASA = -0.08 * log10 (1 + fcGHz) + 1.81;
              table3gpp->m_sigLgASA = 0.05 * log10 (1 + fcGHz) + 0.3;
              table3gpp->m_uLgZSA = -0.04 * log10 (1 + fcGHz) + 0.92;
              table3gpp->m_sigLgZSA = -0.07 * log10 (1 + fcGHz) + 0.41;
              table3gpp->m_uLgZSD = uLgZSD;
              table3gpp->m_sigLgZSD = 0.35;
              table3gpp->m_offsetZOD = offsetZOD;
              table3gpp->m_cDS = 11e-9;
              table3gpp->m_cASD = 10;
              table3gpp->m_cASA = 22;
              table3gpp->m_cZSA = 7;
              table3gpp->m_uK = 0;
              table3gpp->m_sigK = 0;
              table3gpp->m_rTau = 2.1;
              table3gpp->m_uXpr = 8;
              table3gpp->m_sigXpr = 3;
              table3gpp->m_perClusterShadowingStd = 3;

              for (uint8_t row = 0; row < 6; row++)
                {
                  for (uint8_t column = 0; column < 6; column++)
                    {
                      table3gpp->m_sqrtC[row][column] = sqrtC_UMi_NLOS[row][column];
                    }
                }
            }
          else //(o2i)
            {
              table3gpp->m_numOfCluster = 12;
              table3gpp->m_raysPerCluster = 20;
              table3gpp->m_uLgDS = -6.62;
              table3gpp->m_sigLgDS = 0.32;
              table3gpp->m_uLgASD = 1.25;
              table3gpp->m_sigLgASD = 0.42;
              table3gpp->m_uLgASA = 1.76;
              table3gpp->m_sigLgASA = 0.16;
              table3gpp->m_uLgZSA = 1.01;
              table3gpp->m_sigLgZSA = 0.43;
              table3gpp->m_uLgZSD = uLgZSD;
              table3gpp->m_sigLgZSD = 0.35;
              table3gpp->m_offsetZOD = offsetZOD;
              table3gpp->m_cDS = 11e-9;
              table3gpp->m_cASD = 5;
              table3gpp->m_cASA = 8;
              table3gpp->m_cZSA = 3;
              table3gpp->m_uK = 0;
              table3gpp->m_sigK = 0;
              table3gpp->m_rTau = 2.2;
              table3gpp->m_uXpr = 9;
              table3gpp->m_sigXpr = 5;
              table3gpp->m_perClusterShadowingStd = 4;

              for (uint8_t row = 0; row < 6; row++)
                {
                  for (uint8_t column = 0; column < 6; column++)
                    {
                      table3gpp->m_sqrtC[row][column] = sqrtC_UMi_O2I[row][column];
                    }
                }
            }
        }
    }
  else if (m_scenario == "InH-OfficeMixed" || m_scenario == "InH-OfficeOpen")
    {
      NS_ASSERT_MSG (!o2i, "The indoor scenario does out support outdoor to indoor");
      if (los)
        {
          table3gpp->m_numOfCluster = 15;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.01 * log10 (1 + fcGHz) - 7.692;
          table3gpp->m_sigLgDS = 0.18;
          table3gpp->m_uLgASD = 1.60;
          table3gpp->m_sigLgASD = 0.18;
          table3gpp->m_uLgASA = -0.19 * log10 (1 + fcGHz) + 1.781;
          table3gpp->m_sigLgASA = 0.12 * log10 (1 + fcGHz) + 0.119;
          table3gpp->m_uLgZSA = -0.26 * log10 (1 + fcGHz) + 1.44;
          table3gpp->m_sigLgZSA = -0.04 * log10 (1 + fcGHz) + 0.264;
          table3gpp->m_uLgZSD = -1.43 * log10 (1 + fcGHz) + 2.228;
          table3gpp->m_sigLgZSD = 0.13 * log10 (1 + fcGHz) + 0.30;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 3.91e-9;
          table3gpp->m_cASD = 5;
          table3gpp->m_cASA = 8;
          table3gpp->m_cZSA = 9;
          table3gpp->m_uK = 7;
          table3gpp->m_sigK = 4;
          table3gpp->m_rTau = 3.6;
          table3gpp->m_uXpr = 11;
          table3gpp->m_sigXpr = 4;
          table3gpp->m_perClusterShadowingStd = 6;

          for (uint8_t row = 0; row < 7; row++)
            {
              for (uint8_t column = 0; column < 7; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_office_LOS[row][column];
                }
            }
        }
      else
        {
          table3gpp->m_numOfCluster = 19;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.28 * log10 (1 + fcGHz) - 7.173;
          table3gpp->m_sigLgDS = 0.1 * log10 (1 + fcGHz) + 0.055;
          table3gpp->m_uLgASD = 1.62;
          table3gpp->m_sigLgASD = 0.25;
          table3gpp->m_uLgASA = -0.11 * log10 (1 + fcGHz) + 1.863;
          table3gpp->m_sigLgASA = 0.12 * log10 (1 + fcGHz) + 0.059;
          table3gpp->m_uLgZSA = -0.15 * log10 (1 + fcGHz) + 1.387;
          table3gpp->m_sigLgZSA = -0.09 * log10 (1 + fcGHz) + 0.746;
          table3gpp->m_uLgZSD = 1.08;
          table3gpp->m_sigLgZSD = 0.36;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 3.91e-9;
          table3gpp->m_cASD = 5;
          table3gpp->m_cASA = 11;
          table3gpp->m_cZSA = 9;
          table3gpp->m_uK = 0;
          table3gpp->m_sigK = 0;
          table3gpp->m_rTau = 3;
          table3gpp->m_uXpr = 10;
          table3gpp->m_sigXpr = 4;
          table3gpp->m_perClusterShadowingStd = 3;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_office_NLOS[row][column];
                }
            }
        }
    }
  else if (m_scenario == "V2V-Urban")
    {
      if (channelCondition->IsLos ())
        {
          // 3GPP mentioned that 3.91 ns should be used when the Cluster DS (cDS)
          // entry is N/A.
          table3gpp->m_numOfCluster = 12;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.2 * log10 (1 + fcGHz) - 7.5;
          table3gpp->m_sigLgDS = 0.1;
          table3gpp->m_uLgASD = -0.1 * log10 (1 + fcGHz) + 1.6;
          table3gpp->m_sigLgASD = 0.1;
          table3gpp->m_uLgASA = -0.1 * log10 (1 + fcGHz) + 1.6;
          table3gpp->m_sigLgASA = 0.1;
          table3gpp->m_uLgZSA = -0.1 * log10 (1 + fcGHz) + 0.73;
          table3gpp->m_sigLgZSA = -0.04 * log10 (1 + fcGHz) + 0.34;
          table3gpp->m_uLgZSD = -0.1 * log10 (1 + fcGHz) + 0.73;
          table3gpp->m_sigLgZSD = -0.04 * log10 (1 + fcGHz) + 0.34;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 5;
          table3gpp->m_cASD = 17;
          table3gpp->m_cASA = 17;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 3.48;
          table3gpp->m_sigK = 2;
          table3gpp->m_rTau = 3;
          table3gpp->m_uXpr = 9;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 4;

          for (uint8_t row = 0; row < 7; row++)
            {
              for (uint8_t column = 0; column < 7; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_LOS[row][column];
                }
            }
        }
      else if (channelCondition->IsNlos ())
        {
          table3gpp->m_numOfCluster = 19;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.3 * log10 (1 + fcGHz) - 7;
          table3gpp->m_sigLgDS = 0.28;
          table3gpp->m_uLgASD = -0.08 * log10 (1 + fcGHz) + 1.81;
          table3gpp->m_sigLgASD = 0.05 * log10 (1 + fcGHz) + 0.3;
          table3gpp->m_uLgASA = -0.08 * log10 (1 + fcGHz) + 1.81;
          table3gpp->m_sigLgASA = 0.05 * log10 (1 + fcGHz) + 0.3;
          table3gpp->m_uLgZSA = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSA = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_uLgZSD = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSD = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 11;
          table3gpp->m_cASD = 22;
          table3gpp->m_cASA = 22;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 0; // N/A
          table3gpp->m_sigK = 0; // N/A
          table3gpp->m_rTau = 2.1;
          table3gpp->m_uXpr = 8;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 4;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_NLOS[row][column];
                }
            }
        }
      else if (channelCondition->IsNlosv ())
        {
          table3gpp->m_numOfCluster = 19;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.4 * log10 (1 + fcGHz) - 7;
          table3gpp->m_sigLgDS = 0.1;
          table3gpp->m_uLgASD = -0.1 * log10 (1 + fcGHz) + 1.7;
          table3gpp->m_sigLgASD = 0.1;
          table3gpp->m_uLgASA = -0.1 * log10 (1 + fcGHz) + 1.7;
          table3gpp->m_sigLgASA = 0.1;
          table3gpp->m_uLgZSA = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSA = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_uLgZSD = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSD = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 11;
          table3gpp->m_cASD = 22;
          table3gpp->m_cASA = 22;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 0;
          table3gpp->m_sigK = 4.5;
          table3gpp->m_rTau = 2.1;
          table3gpp->m_uXpr = 8;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 4;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_LOS[row][column];
                }
            }
        }
      else
        {
          NS_FATAL_ERROR ("Unknown channel condition");
        }
    }
  else if (m_scenario == "V2V-Highway")
    {
      if (channelCondition->IsLos ())
        {
          table3gpp->m_numOfCluster = 12;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -8.3;
          table3gpp->m_sigLgDS = 0.2;
          table3gpp->m_uLgASD = 1.4;
          table3gpp->m_sigLgASD = 0.1;
          table3gpp->m_uLgASA = 1.4;
          table3gpp->m_sigLgASA = 0.1;
          table3gpp->m_uLgZSA = -0.1 * log10 (1 + fcGHz) + 0.73;
          table3gpp->m_sigLgZSA = -0.04 * log10 (1 + fcGHz) + 0.34;
          table3gpp->m_uLgZSD = -0.1 * log10 (1 + fcGHz) + 0.73;
          table3gpp->m_sigLgZSD = -0.04 * log10 (1 + fcGHz) + 0.34;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 5;
          table3gpp->m_cASD = 17;
          table3gpp->m_cASA = 17;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 9;
          table3gpp->m_sigK = 3.5;
          table3gpp->m_rTau = 3;
          table3gpp->m_uXpr = 9;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 4;

          for (uint8_t row = 0; row < 7; row++)
            {
              for (uint8_t column = 0; column < 7; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_LOS[row][column];
                }
            }
        }
      else if (channelCondition->IsNlosv ())
        {
          table3gpp->m_numOfCluster = 19;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -8.3;
          table3gpp->m_sigLgDS = 0.3;
          table3gpp->m_uLgASD = 1.5;
          table3gpp->m_sigLgASD = 0.1;
          table3gpp->m_uLgASA = 1.5;
          table3gpp->m_sigLgASA = 0.1;
          table3gpp->m_uLgZSA = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSA = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_uLgZSD = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSD = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 11;
          table3gpp->m_cASD = 22;
          table3gpp->m_cASA = 22;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 0;
          table3gpp->m_sigK = 4.5;
          table3gpp->m_rTau = 2.1;
          table3gpp->m_uXpr = 8.0;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 4;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_LOS[row][column];
                }
            }
        }
      else if (channelCondition->IsNlos ())
        {
          NS_LOG_WARN ("The fast fading parameters for the NLOS condition in the Highway scenario are not defined in TR 37.885, use the ones defined in TDoc R1-1803671 instead");

          table3gpp->m_numOfCluster = 19;
          table3gpp->m_raysPerCluster = 20;
          table3gpp->m_uLgDS = -0.3 * log10 (1 + fcGHz) - 7;
          table3gpp->m_sigLgDS = 0.28;
          table3gpp->m_uLgASD = -0.08 * log10 (1 + fcGHz) + 1.81;
          table3gpp->m_sigLgASD = 0.05 * log10 (1 + fcGHz) + 0.3;
          table3gpp->m_uLgASA = -0.08 * log10 (1 + fcGHz) + 1.81;
          table3gpp->m_sigLgASA = 0.05 * log10 (1 + fcGHz) + 0.3;
          table3gpp->m_uLgZSA = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSA = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_uLgZSD = -0.04 * log10 (1 + fcGHz) + 0.92;
          table3gpp->m_sigLgZSD = -0.07 * log10 (1 + fcGHz) + 0.41;
          table3gpp->m_offsetZOD = 0;
          table3gpp->m_cDS = 11;
          table3gpp->m_cASD = 22;
          table3gpp->m_cASA = 22;
          table3gpp->m_cZSA = 7;
          table3gpp->m_uK = 0; // N/A
          table3gpp->m_sigK = 0; // N/A
          table3gpp->m_rTau = 2.1;
          table3gpp->m_uXpr = 8;
          table3gpp->m_sigXpr = 3;
          table3gpp->m_perClusterShadowingStd = 4;

          for (uint8_t row = 0; row < 6; row++)
            {
              for (uint8_t column = 0; column < 6; column++)
                {
                  table3gpp->m_sqrtC[row][column] = sqrtC_UMi_NLOS[row][column];
                }
            }
        }
      else
        {
          NS_FATAL_ERROR ("Unknown channel condition");
        }
    }
  else
    {
      NS_FATAL_ERROR ("unkonw scenarios");
    }

  return table3gpp;
}

bool
ThreeGppChannelModel::ChannelParamsNeedsUpdate (Ptr<const ThreeGppChannelParams> channelParams,
                                                Ptr<const ChannelCondition> channelCondition) const
{
  NS_LOG_FUNCTION (this);

  bool update = false;

  // if the channel condition is different the channel has to be updated
  if (!channelCondition->IsEqual (channelParams->m_losCondition, channelParams->m_o2iCondition))
    {
      NS_LOG_DEBUG ("Update the channel condition");
      update = true;
    }

  // if the coherence time is over the channel has to be updated
  if (!m_updatePeriod.IsZero () && Simulator::Now () - channelParams->m_generatedTime > m_updatePeriod)
    {
      NS_LOG_DEBUG ("Generation time " << channelParams->m_generatedTime.As (Time::NS) << " now " << Now ().As (Time::NS));
      update = true;
    }

  return update;
}


bool
ThreeGppChannelModel::ChannelMatrixNeedsUpdate (Ptr<const ThreeGppChannelParams> channelParams, Ptr<const ChannelMatrix> channelMatrix)
{

  if (channelParams->m_generatedTime > channelMatrix->m_generatedTime)
    {
      return true;
    }
  else
    {
      return false;
    }
}

Ptr<const MatrixBasedChannelModel::ChannelMatrix>
ThreeGppChannelModel::GetChannel (Ptr<const MobilityModel> aMob,
                                  Ptr<const MobilityModel> bMob,
                                  Ptr<const PhasedArrayModel> aAntenna,
                                  Ptr<const PhasedArrayModel> bAntenna)
{
  NS_LOG_FUNCTION (this);

  // Compute the channel params key. The key is reciprocal, i.e., key (a, b) = key (b, a)
  uint64_t channelParamsKey = GetKey (aMob->GetObject<Node> ()->GetId (), bMob->GetObject<Node> ()->GetId ());
  // Compute the channel matrix key. The key is reciprocal, i.e., key (a, b) = key (b, a)
  uint64_t channelMatrixKey = GetKey (aAntenna->GetId (), bAntenna->GetId ());

  // retrieve the channel condition
  Ptr<const ChannelCondition> condition = m_channelConditionModel->GetChannelCondition (aMob, bMob);

  // Check if the channel is present in the map and return it, otherwise
  // generate a new channel
  bool updateParams = false;
  bool updateMatrix = false;
  bool notFoundParams = false;
  bool notFoundMatrix = false;
  Ptr<ChannelMatrix> channelMatrix;
  Ptr<ThreeGppChannelParams> channelParams;


  if (m_channelParamsMap.find (channelParamsKey) != m_channelParamsMap.end ())
    {
      channelParams = m_channelParamsMap[channelParamsKey];
      // check if it has to be updated
      updateParams = ChannelParamsNeedsUpdate (channelParams, condition);
    }
  else
    {
      NS_LOG_DEBUG ("channel params not found");
      notFoundParams = true;
    }

  double x = aMob->GetPosition ().x - bMob->GetPosition ().x;
  double y = aMob->GetPosition ().y - bMob->GetPosition ().y;
  double distance2D = sqrt (x * x + y * y);

  // NOTE we assume hUT = min (height(a), height(b)) and
  // hBS = max (height (a), height (b))
  double hUt = std::min (aMob->GetPosition ().z, bMob->GetPosition ().z);
  double hBs = std::max (aMob->GetPosition ().z, bMob->GetPosition ().z);

  // get the 3GPP parameters
  Ptr<const ParamsTable> table3gpp = GetThreeGppTable (condition, hBs, hUt, distance2D);

  if (notFoundParams || updateParams)
    {
      //Step 4: Generate large scale parameters. All LSPS are uncorrelated.
      //Step 5: Generate Delays.
      //Step 6: Generate cluster powers.
      //Step 7: Generate arrival and departure angles for both azimuth and elevation.
      //Step 8: Coupling of rays within a cluster for both azimuth and elevation
      //shuffle all the arrays to perform random coupling
      //Step 9: Generate the cross polarization power ratios
      //Step 10: Draw initial phases
      channelParams = GenerateChannelParameters (condition, table3gpp, aMob, bMob);
      // store or replace the channel parameters
      m_channelParamsMap[channelParamsKey] = channelParams;
    }

  if (m_channelMatrixMap.find (channelMatrixKey) != m_channelMatrixMap.end ())
    {
      // channel matrix present in the map
      NS_LOG_DEBUG ("channel matrix present in the map");
      channelMatrix = m_channelMatrixMap[channelMatrixKey];
      updateMatrix = ChannelMatrixNeedsUpdate (channelParams, channelMatrix);
    }
  else
    {
      NS_LOG_DEBUG ("channel matrix not found");
      notFoundMatrix = true;
    }

  // If the channel is not present in the map or if it has to be updated
  // generate a new realization
  if (notFoundMatrix || updateMatrix)
    {
      // channel matrix not found or has to be updated, generate a new one
      channelMatrix = GetNewChannel (channelParams, table3gpp, aMob, bMob, aAntenna, bAntenna);
      channelMatrix->m_antennaPair = std::make_pair (aAntenna->GetId (), bAntenna->GetId ()); // save antenna pair, with the exact order of s and u antennas at the moment of the channel generation

      // store or replace the channel matrix in the channel map
      m_channelMatrixMap[channelMatrixKey] = channelMatrix;
    }

  return channelMatrix;
}

Ptr<const MatrixBasedChannelModel::ChannelParams>
ThreeGppChannelModel::GetParams (Ptr<const MobilityModel> aMob,
                                 Ptr<const MobilityModel> bMob) const
{
  NS_LOG_FUNCTION (this);

  // Compute the channel key. The key is reciprocal, i.e., key (a, b) = key (b, a)
  uint64_t channelParamsKey = GetKey (aMob->GetObject<Node> ()->GetId (), bMob->GetObject<Node> ()->GetId ());

  if (m_channelParamsMap.find (channelParamsKey) != m_channelParamsMap.end ())
    {
      return m_channelParamsMap.find (channelParamsKey)->second;
    }
  else
    {
      NS_LOG_WARN ("Channel params map not found. Returning a nullptr.");
      return nullptr;
    }
}


Ptr<ThreeGppChannelModel::ThreeGppChannelParams>
ThreeGppChannelModel::GenerateChannelParameters (const Ptr<const ChannelCondition> channelCondition,
                                                 const Ptr<const ParamsTable> table3gpp,
                                                 const Ptr<const MobilityModel> aMob,
                                                 const Ptr<const MobilityModel> bMob) const
{

  NS_LOG_FUNCTION (this);
  // create a channel matrix instance
  Ptr<ThreeGppChannelParams> channelParams = Create<ThreeGppChannelParams> ();
  channelParams->m_generatedTime = Simulator::Now ();
  channelParams->m_nodeIds = std::make_pair (aMob->GetObject<Node> ()->GetId (), bMob->GetObject<Node> ()->GetId ());
  channelParams->m_losCondition = channelCondition->GetLosCondition ();
  channelParams->m_o2iCondition = channelCondition->GetO2iCondition ();

  //Step 4: Generate large scale parameters. All LSPS are uncorrelated.
  DoubleVector LSPsIndep, LSPs;
  uint8_t paramNum = 6;
  if (channelParams->m_losCondition == ChannelCondition::LOS)
    {
      paramNum = 7;
    }

  //Generate paramNum independent LSPs.
  for (uint8_t iter = 0; iter < paramNum; iter++)
    {
      LSPsIndep.push_back (m_normalRv->GetValue ());
    }
  for (uint8_t row = 0; row < paramNum; row++)
    {
      double temp = 0;
      for (uint8_t column = 0; column < paramNum; column++)
        {
          temp += table3gpp->m_sqrtC[row][column] * LSPsIndep[column];
        }
      LSPs.push_back (temp);
    }

  // NOTE the shadowing is generated in the propagation loss model
  double DS, ASD, ASA, ZSA, ZSD, kFactor = 0;
  if (channelParams->m_losCondition == ChannelCondition::LOS)
    {
      kFactor = LSPs[1] * table3gpp->m_sigK + table3gpp->m_uK;
      DS = pow (10, LSPs[2] * table3gpp->m_sigLgDS + table3gpp->m_uLgDS);
      ASD = pow (10, LSPs[3] * table3gpp->m_sigLgASD + table3gpp->m_uLgASD);
      ASA = pow (10, LSPs[4] * table3gpp->m_sigLgASA + table3gpp->m_uLgASA);
      ZSD = pow (10, LSPs[5] * table3gpp->m_sigLgZSD + table3gpp->m_uLgZSD);
      ZSA = pow (10, LSPs[6] * table3gpp->m_sigLgZSA + table3gpp->m_uLgZSA);
    }
  else
    {
      DS = pow (10, LSPs[1] * table3gpp->m_sigLgDS + table3gpp->m_uLgDS);
      ASD = pow (10, LSPs[2] * table3gpp->m_sigLgASD + table3gpp->m_uLgASD);
      ASA = pow (10, LSPs[3] * table3gpp->m_sigLgASA + table3gpp->m_uLgASA);
      ZSD = pow (10, LSPs[4] * table3gpp->m_sigLgZSD + table3gpp->m_uLgZSD);
      ZSA = pow (10, LSPs[5] * table3gpp->m_sigLgZSA + table3gpp->m_uLgZSA);

    }
  ASD = std::min (ASD, 104.0);
  ASA = std::min (ASA, 104.0);
  ZSD = std::min (ZSD, 52.0);
  ZSA = std::min (ZSA, 52.0);

  // save DS and K_factor parameters in the structure
  channelParams->m_DS = DS;
  channelParams->m_K_factor = kFactor;

  NS_LOG_INFO ("K-factor=" << kFactor << ", DS=" << DS << ", ASD=" << ASD << ", ASA=" << ASA
                           << ", ZSD=" << ZSD << ", ZSA=" << ZSA);

  //Step 5: Generate Delays.
  DoubleVector clusterDelay;
  double minTau = 100.0;
  for (uint8_t cIndex = 0; cIndex < table3gpp->m_numOfCluster; cIndex++)
    {
      double tau = -1 * table3gpp->m_rTau * DS * log (m_uniformRv->GetValue (0, 1)); //(7.5-1)
      if (minTau > tau)
        {
          minTau = tau;
        }
      clusterDelay.push_back (tau);
    }

  for (uint8_t cIndex = 0; cIndex < table3gpp->m_numOfCluster; cIndex++)
    {
      clusterDelay[cIndex] -= minTau;
    }
  std::sort (clusterDelay.begin (), clusterDelay.end ()); //(7.5-2)

  /* since the scaled Los delays are not to be used in cluster power generation,
   * we will generate cluster power first and resume to compute Los cluster delay later.*/

  //Step 6: Generate cluster powers.
  DoubleVector clusterPower;
  double powerSum = 0;
  for (uint8_t cIndex = 0; cIndex < table3gpp->m_numOfCluster; cIndex++)
    {
      double power = exp (-1 * clusterDelay[cIndex] * (table3gpp->m_rTau - 1) / table3gpp->m_rTau / DS) *
        pow (10, -1 * m_normalRv->GetValue () * table3gpp->m_perClusterShadowingStd / 10);                       //(7.5-5)
      powerSum += power;
      clusterPower.push_back (power);
    }
  channelParams->m_clusterPower = clusterPower;

  double powerMax = 0;

  for (uint8_t cIndex = 0; cIndex < table3gpp->m_numOfCluster; cIndex++)
    {
      channelParams->m_clusterPower[cIndex] = channelParams->m_clusterPower[cIndex] / powerSum; //(7.5-6)
    }

  DoubleVector clusterPowerForAngles; // this power is only for equation (7.5-9) and (7.5-14), not for (7.5-22)
  if (channelParams->m_losCondition == ChannelCondition::LOS)
    {
      double kLinear = pow (10, kFactor / 10);

      for (uint8_t cIndex = 0; cIndex < table3gpp->m_numOfCluster; cIndex++)
        {
          if (cIndex == 0)
            {
              clusterPowerForAngles.push_back (channelParams->m_clusterPower[cIndex] / (1 + kLinear) +
                                               kLinear / (1 + kLinear));                  //(7.5-8)
            }
          else
            {
              clusterPowerForAngles.push_back (channelParams->m_clusterPower[cIndex] / (1 + kLinear)); //(7.5-8)
            }
          if (powerMax < clusterPowerForAngles[cIndex])
            {
              powerMax = clusterPowerForAngles[cIndex];
            }
        }
    }
  else
    {
      for (uint8_t cIndex = 0; cIndex < table3gpp->m_numOfCluster; cIndex++)
        {
          clusterPowerForAngles.push_back (channelParams->m_clusterPower[cIndex]); //(7.5-6)
          if (powerMax < clusterPowerForAngles[cIndex])
            {
              powerMax = clusterPowerForAngles[cIndex];
            }
        }
    }

  //remove clusters with less than -25 dB power compared to the maxim cluster power;
  //double thresh = pow(10, -2.5);
  double thresh = 0.0032;
  for (uint8_t cIndex = table3gpp->m_numOfCluster; cIndex > 0; cIndex--)
    {
      if (clusterPowerForAngles[cIndex - 1] < thresh * powerMax )
        {
          clusterPowerForAngles.erase (clusterPowerForAngles.begin () + cIndex - 1);
          channelParams->m_clusterPower.erase (channelParams->m_clusterPower.begin () + cIndex - 1);
          clusterDelay.erase (clusterDelay.begin () + cIndex - 1);
        }
    }

  NS_ASSERT (channelParams->m_clusterPower.size () < UINT8_MAX);
  channelParams->m_reducedClusterNumber = channelParams->m_clusterPower.size ();
  // Resume step 5 to compute the delay for LoS condition.
  if (channelParams->m_losCondition == ChannelCondition::LOS)
    {
      double cTau = 0.7705 - 0.0433 * kFactor + 2e-4 * pow (kFactor,2) + 17e-6 * pow (kFactor,3);         //(7.5-3)
      for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
        {
          clusterDelay[cIndex] = clusterDelay[cIndex] / cTau; //(7.5-4)
        }
    }

  //Step 7: Generate arrival and departure angles for both azimuth and elevation.

  double cNlos;
  //According to table 7.5-6, only cluster number equals to 8, 10, 11, 12, 19 and 20 is valid.
  //Not sure why the other cases are in Table 7.5-2.
  switch (table3gpp->m_numOfCluster) // Table 7.5-2
    {
      case 4:
        cNlos = 0.779;
        break;
      case 5:
        cNlos = 0.860;
        break;
      case 8:
        cNlos = 1.018;
        break;
      case 10:
        cNlos = 1.090;
        break;
      case 11:
        cNlos = 1.123;
        break;
      case 12:
        cNlos = 1.146;
        break;
      case 14:
        cNlos = 1.190;
        break;
      case 15:
        cNlos = 1.221;
        break;
      case 16:
        cNlos = 1.226;
        break;
      case 19:
        cNlos = 1.273;
        break;
      case 20:
        cNlos = 1.289;
        break;
      default:
        NS_FATAL_ERROR ("Invalid cluster number");
    }

  double cPhi = cNlos;

  if (channelParams->m_losCondition == ChannelCondition::LOS)
    {
      cPhi *= (1.1035 - 0.028 * kFactor - 2e-3 * pow (kFactor, 2) + 1e-4 * pow (kFactor, 3)); //(7.5-10))
    }

  switch (table3gpp->m_numOfCluster) //Table 7.5-4
    {
      case 8:
        cNlos = 0.889;
        break;
      case 10:
        cNlos = 0.957;
        break;
      case 11:
        cNlos = 1.031;
        break;
      case 12:
        cNlos = 1.104;
        break;
      case 15:
        cNlos = 1.1088;
        break;
      case 19:
        cNlos = 1.184;
        break;
      case 20:
        cNlos = 1.178;
        break;
      default:
        NS_FATAL_ERROR ("Invalid cluster number");
    }

  double cTheta = cNlos;
  if (channelCondition->IsLos())
    {
      cTheta *= (1.3086 + 0.0339 * kFactor - 0.0077 * pow (kFactor, 2) + 2e-4 * pow (kFactor, 3)); //(7.5-15)
    }

  DoubleVector clusterAoa, clusterAod, clusterZoa, clusterZod;
  for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
    {
      double logCalc = -1 * log (clusterPowerForAngles[cIndex] / powerMax);
      double angle = 2 * sqrt (logCalc) / 1.4 / cPhi; //(7.5-9)
      clusterAoa.push_back (ASA * angle);
      clusterAod.push_back (ASD * angle);
      angle = logCalc / cTheta; //(7.5-14)
      clusterZoa.push_back (ZSA * angle);
      clusterZod.push_back (ZSD * angle);
    }

  Angles sAngle (bMob->GetPosition (), aMob->GetPosition ());
  Angles uAngle (aMob->GetPosition (), bMob->GetPosition ());

  for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
    {
      int Xn = 1;
      if (m_uniformRv->GetValue (0, 1) < 0.5)
        {
          Xn = -1;
        }
      clusterAoa[cIndex] = clusterAoa[cIndex] * Xn + (m_normalRv->GetValue () * ASA / 7) + RadiansToDegrees (uAngle.GetAzimuth ());        //(7.5-11)
      clusterAod[cIndex] = clusterAod[cIndex] * Xn + (m_normalRv->GetValue () * ASD / 7) + RadiansToDegrees (sAngle.GetAzimuth ());
      if (channelCondition->IsO2i ())
        {
          clusterZoa[cIndex] = clusterZoa[cIndex] * Xn + (m_normalRv->GetValue () * ZSA / 7) + 90;            //(7.5-16)
        }
      else
        {
          clusterZoa[cIndex] = clusterZoa[cIndex] * Xn + (m_normalRv->GetValue () * ZSA / 7) + RadiansToDegrees (uAngle.GetInclination ());            //(7.5-16)
        }
      clusterZod[cIndex] = clusterZod[cIndex] * Xn + (m_normalRv->GetValue () * ZSD / 7) + RadiansToDegrees (sAngle.GetInclination ()) + table3gpp->m_offsetZOD;        //(7.5-19)
    }

  if (channelParams->m_losCondition == ChannelCondition::LOS)
    {
      // The 7.5-12 can be rewrite as Theta_n,ZOA = Theta_n,ZOA - (Theta_1,ZOA - Theta_LOS,ZOA) = Theta_n,ZOA - diffZOA,
      // Similar as AOD, ZSA and ZSD.
      double diffAoa = clusterAoa[0] - RadiansToDegrees (uAngle.GetAzimuth ());
      double diffAod = clusterAod[0] - RadiansToDegrees (sAngle.GetAzimuth ());
      double diffZsa = clusterZoa[0] - RadiansToDegrees (uAngle.GetInclination ());
      double diffZsd = clusterZod[0] - RadiansToDegrees (sAngle.GetInclination ());

      for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
        {
          clusterAoa[cIndex] -= diffAoa; //(7.5-12)
          clusterAod[cIndex] -= diffAod;
          clusterZoa[cIndex] -= diffZsa; //(7.5-17)
          clusterZod[cIndex] -= diffZsd;
        }
    }

  double sizeTemp = clusterZoa.size ();
  for (uint8_t ind = 0; ind < 4; ind++)
    {
      DoubleVector angleDegree;
      switch (ind)
        {
          case 0:
            angleDegree = clusterAoa;
            break;
          case 1:
            angleDegree = clusterZoa;
            break;
          case 2:
            angleDegree = clusterAod;
            break;
          case 3:
            angleDegree = clusterZod;
            break;
          default:
            NS_FATAL_ERROR ("Programming Error");
        }
      for (uint8_t nIndex = 0; nIndex < sizeTemp; nIndex++)
        {
          while (angleDegree[nIndex] > 360)
            {
              angleDegree[nIndex] -= 360;
            }

          while (angleDegree[nIndex] < 0)
            {
              angleDegree[nIndex] += 360;
            }

          if (ind == 1 || ind == 3)
            {
              if (angleDegree[nIndex] > 180)
                {
                  angleDegree[nIndex] = 360 - angleDegree[nIndex];
                }
            }
        }
      switch (ind)
        {
          case 0:
            clusterAoa = angleDegree;
            break;
          case 1:
            clusterZoa = angleDegree;
            break;
          case 2:
            clusterAod = angleDegree;
            break;
          case 3:
            clusterZod = angleDegree;
            break;
          default:
            NS_FATAL_ERROR ("Programming Error");
        }
    }

  DoubleVector attenuationDb;
  if (m_blockage)
    {
      attenuationDb = CalcAttenuationOfBlockage (channelParams, clusterAoa, clusterZoa);
      for (uint8_t cInd = 0; cInd < channelParams->m_reducedClusterNumber; cInd++)
        {
          channelParams->m_clusterPower[cInd] = channelParams->m_clusterPower[cInd] / pow (10,attenuationDb[cInd] / 10);
        }
    }
  else
    {
      attenuationDb.push_back (0);
    }

  // store attenuation
  channelParams->m_attenuation_dB = attenuationDb;

  //Step 8: Coupling of rays within a cluster for both azimuth and elevation
  //shuffle all the arrays to perform random coupling
  MatrixBasedChannelModel::Double2DVector rayAoaRadian (channelParams->m_reducedClusterNumber, DoubleVector (table3gpp->m_raysPerCluster, 0)); //rayAoaRadian[n][m], where n is cluster index, m is ray index
  MatrixBasedChannelModel::Double2DVector rayAodRadian (channelParams->m_reducedClusterNumber, DoubleVector (table3gpp->m_raysPerCluster, 0)); //rayAodRadian[n][m], where n is cluster index, m is ray index
  MatrixBasedChannelModel::Double2DVector rayZoaRadian (channelParams->m_reducedClusterNumber, DoubleVector (table3gpp->m_raysPerCluster, 0)); //rayZoaRadian[n][m], where n is cluster index, m is ray index
  MatrixBasedChannelModel::Double2DVector rayZodRadian (channelParams->m_reducedClusterNumber, DoubleVector (table3gpp->m_raysPerCluster, 0)); //rayZodRadian[n][m], where n is cluster index, m is ray index

  for (uint8_t nInd = 0; nInd < channelParams->m_reducedClusterNumber; nInd++)
    {
      for (uint8_t mInd = 0; mInd < table3gpp->m_raysPerCluster; mInd++)
        {
          double tempAoa = clusterAoa[nInd] + table3gpp->m_cASA * offSetAlpha[mInd]; //(7.5-13)
          double tempZoa = clusterZoa[nInd] + table3gpp->m_cZSA * offSetAlpha[mInd]; //(7.5-18)
          std::tie (rayAoaRadian[nInd][mInd], rayZoaRadian[nInd][mInd]) = WrapAngles (DegreesToRadians (tempAoa), DegreesToRadians (tempZoa));

          double tempAod = clusterAod[nInd] + table3gpp->m_cASD * offSetAlpha[mInd]; //(7.5-13)
          double tempZod = clusterZod[nInd] + 0.375 * pow (10,table3gpp->m_uLgZSD) * offSetAlpha[mInd]; //(7.5-20)
          std::tie (rayAodRadian[nInd][mInd], rayZodRadian[nInd][mInd]) = WrapAngles (DegreesToRadians (tempAod), DegreesToRadians (tempZod));
        }
    }

  for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
    {
      Shuffle (&rayAodRadian[cIndex][0], &rayAodRadian[cIndex][table3gpp->m_raysPerCluster]);
      Shuffle (&rayAoaRadian[cIndex][0], &rayAoaRadian[cIndex][table3gpp->m_raysPerCluster]);
      Shuffle (&rayZodRadian[cIndex][0], &rayZodRadian[cIndex][table3gpp->m_raysPerCluster]);
      Shuffle (&rayZoaRadian[cIndex][0], &rayZoaRadian[cIndex][table3gpp->m_raysPerCluster]);
    }

  // store values
  channelParams->m_rayAodRadian = rayAodRadian;
  channelParams->m_rayAoaRadian = rayAoaRadian;
  channelParams->m_rayZodRadian = rayZodRadian;
  channelParams->m_rayZoaRadian = rayZoaRadian;

  //Step 9: Generate the cross polarization power ratios
  //Step 10: Draw initial phases
  Double2DVector crossPolarizationPowerRatios; // vector containing the cross polarization power ratios, as defined by 7.5-21
  Double3DVector clusterPhase; //rayAoaRadian[n][m], where n is cluster index, m is ray index
  for (uint8_t nInd = 0; nInd < channelParams->m_reducedClusterNumber; nInd++)
    {
      DoubleVector temp; // used to store the XPR values
      Double2DVector temp2; // used to store the PHI values for all the possible combination of polarization
      for (uint8_t mInd = 0; mInd < table3gpp->m_raysPerCluster; mInd++)
        {
          double uXprLinear = pow (10, table3gpp->m_uXpr / 10); // convert to linear
          double sigXprLinear = pow (10, table3gpp->m_sigXpr / 10); // convert to linear

          temp.push_back (std::pow (10, (m_normalRv->GetValue () * sigXprLinear + uXprLinear) / 10));
          DoubleVector temp3; // used to store the PHI valuse
          for (uint8_t pInd = 0; pInd < 4; pInd++)
            {
              temp3.push_back (m_uniformRv->GetValue (-1 * M_PI, M_PI));
            }
          temp2.push_back (temp3);
        }
      crossPolarizationPowerRatios.push_back (temp);
      clusterPhase.push_back (temp2);
    }
  // store the cluster phase
  channelParams->m_clusterPhase = clusterPhase;
  channelParams->m_crossPolarizationPowerRatios = crossPolarizationPowerRatios;

  uint8_t cluster1st = 0, cluster2nd = 0; // first and second strongest cluster;
  double maxPower = 0;
  for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
    {
      if (maxPower < channelParams->m_clusterPower[cIndex])
        {
          maxPower = channelParams->m_clusterPower[cIndex];
          cluster1st = cIndex;
        }
    }
  channelParams->m_cluster1st = cluster1st;
  maxPower = 0;
  for (uint8_t cIndex = 0; cIndex < channelParams->m_reducedClusterNumber; cIndex++)
    {
      if (maxPower < channelParams->m_clusterPower[cIndex] && cluster1st != cIndex)
        {
          maxPower = channelParams->m_clusterPower[cIndex];
          cluster2nd = cIndex;
        }
    }
  channelParams->m_cluster2nd = cluster2nd;

  NS_LOG_INFO ("1st strongest cluster:" << +cluster1st
                                        << ", 2nd strongest cluster:" << +cluster2nd);

  // store the delays and the angles for the subclusters
  if (cluster1st == cluster2nd)
    {
      clusterDelay.push_back (clusterDelay[cluster1st] + 1.28 * table3gpp->m_cDS);
      clusterDelay.push_back (clusterDelay[cluster1st] + 2.56 * table3gpp->m_cDS);

      clusterAoa.push_back (clusterAoa[cluster1st]);
      clusterAoa.push_back (clusterAoa[cluster1st]);

      clusterZoa.push_back (clusterZoa[cluster1st]);
      clusterZoa.push_back (clusterZoa[cluster1st]);

      clusterAod.push_back (clusterAod[cluster1st]);
      clusterAod.push_back (clusterAod[cluster1st]);

      clusterZod.push_back (clusterZod[cluster1st]);
      clusterZod.push_back (clusterZod[cluster1st]);
    }
  else
    {
      double min, max;
      if (cluster1st < cluster2nd)
        {
          min = cluster1st;
          max = cluster2nd;
        }
      else
        {
          min = cluster2nd;
          max = cluster1st;
        }
      clusterDelay.push_back (clusterDelay[min] + 1.28 * table3gpp->m_cDS);
      clusterDelay.push_back (clusterDelay[min] + 2.56 * table3gpp->m_cDS);
      clusterDelay.push_back (clusterDelay[max] + 1.28 * table3gpp->m_cDS);
      clusterDelay.push_back (clusterDelay[max] + 2.56 * table3gpp->m_cDS);

      clusterAoa.push_back (clusterAoa[min]);
      clusterAoa.push_back (clusterAoa[min]);
      clusterAoa.push_back (clusterAoa[max]);
      clusterAoa.push_back (clusterAoa[max]);

      clusterZoa.push_back (clusterZoa[min]);
      clusterZoa.push_back (clusterZoa[min]);
      clusterZoa.push_back (clusterZoa[max]);
      clusterZoa.push_back (clusterZoa[max]);

      clusterAod.push_back (clusterAod[min]);
      clusterAod.push_back (clusterAod[min]);
      clusterAod.push_back (clusterAod[max]);
      clusterAod.push_back (clusterAod[max]);

      clusterZod.push_back (clusterZod[min]);
      clusterZod.push_back (clusterZod[min]);
      clusterZod.push_back (clusterZod[max]);
      clusterZod.push_back (clusterZod[max]);


    }

  channelParams->m_delay = clusterDelay;
  channelParams->m_angle.clear ();
  channelParams->m_angle.push_back (clusterAoa);
  channelParams->m_angle.push_back (clusterZoa);
  channelParams->m_angle.push_back (clusterAod);
  channelParams->m_angle.push_back (clusterZod);

  // Compute alpha and D as described in 3GPP TR 37.885 v15.3.0, Sec. 6.2.3
  // These terms account for an additional Doppler contribution due to the
  // presence of moving objects in the surrounding environment, such as in
  // vehicular scenarios.
  // This contribution is applied only to the delayed (reflected) paths and
  // must be properly configured by setting the value of
  // m_vScatt, which is defined as "maximum speed of the vehicle in the
  // layout".
  // By default, m_vScatt is set to 0, so there is no additional Doppler
  // contribution.

  DoubleVector dopplerTermAlpha, dopplerTermD;

  // 2 or 4 is added to account for additional subrays for the 1st and 2nd clusters, if there is only one cluster then would be added 2 more subrays (see creation of Husn channel matrix)
  uint8_t updatedClusterNumber = (channelParams->m_reducedClusterNumber == 1) ? channelParams->m_reducedClusterNumber + 2 : channelParams->m_reducedClusterNumber + 4;

  for (uint8_t cIndex = 0; cIndex < updatedClusterNumber ; cIndex++)
    {
      double alpha = 0;
      double D = 0;
      if (cIndex != 0)
      {
        alpha = m_uniformRvDoppler->GetValue (-1, 1);
        D = m_uniformRvDoppler->GetValue (-m_vScatt, m_vScatt);
      }
      dopplerTermAlpha.push_back (alpha);
      dopplerTermD.push_back (D);
    }
  channelParams->m_alpha = dopplerTermAlpha;
  channelParams->m_D = dopplerTermD;

  return channelParams;
}

Ptr<MatrixBasedChannelModel::ChannelMatrix>
ThreeGppChannelModel::GetNewChannel (Ptr<const ThreeGppChannelParams> channelParams,
                                     Ptr<const ParamsTable> table3gpp,
                                     const Ptr<const MobilityModel> sMob,
                                     const Ptr<const MobilityModel> uMob,
                                     Ptr<const PhasedArrayModel> sAntenna,
                                     Ptr<const PhasedArrayModel> uAntenna
                                     ) const
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (m_frequency > 0.0, "Set the operating frequency first!");

  // create a channel matrix instance
  Ptr<ChannelMatrix> channelMatrix = Create<ChannelMatrix> ();
  channelMatrix->m_generatedTime = Simulator::Now ();
  // save in which order is generated this matrix
  channelMatrix->m_nodeIds = std::make_pair (sMob->GetObject<Node> ()->GetId (), uMob->GetObject<Node> ()->GetId ());
  // check if channelParams structure is generated in direction s-to-u or u-to-s
  bool isSameDirection = (channelParams->m_nodeIds == channelMatrix->m_nodeIds);

  MatrixBasedChannelModel::Double2DVector rayAodRadian;
  MatrixBasedChannelModel::Double2DVector rayAoaRadian;
  MatrixBasedChannelModel::Double2DVector rayZodRadian;
  MatrixBasedChannelModel::Double2DVector rayZoaRadian;

  // if channel params is generated in the same direction in which we
  // generate the channel matrix, angles and zenit od departure and arrival are ok,
  // just set them to corresponding variable that will be used for the generation
  // of channel matrix, otherwise we need to flip angles and zenits of departure and arrival
  if (isSameDirection)
    {
      rayAodRadian = channelParams->m_rayAodRadian;
      rayAoaRadian = channelParams->m_rayAoaRadian;
      rayZodRadian = channelParams->m_rayZodRadian;
      rayZoaRadian = channelParams->m_rayZoaRadian;
    }
  else
    {
      rayAodRadian = channelParams->m_rayAoaRadian;
      rayAoaRadian = channelParams->m_rayAodRadian;
      rayZodRadian = channelParams->m_rayZoaRadian;
      rayZoaRadian = channelParams->m_rayZodRadian;
    }

  //Step 11: Generate channel coefficients for each cluster n and each receiver
  // and transmitter element pair u,s.
  Complex3DVector hUsn;  //channel coffecient hUsn[u][s][n];
  // where u and s are receive and transmit antenna element, n is cluster index.
  // NOTE Since each of the strongest 2 clusters are divided into 3 sub-clusters,
  // the total cluster will be numReducedCLuster + 4.
  uint64_t uSize = uAntenna->GetNumberOfElements ();
  uint64_t sSize = sAntenna->GetNumberOfElements ();

  hUsn.resize (uSize);
  for (uint64_t uIndex = 0; uIndex < uSize; uIndex++)
    {
      hUsn[uIndex].resize (sSize);
      for (uint64_t sIndex = 0; sIndex < sSize; sIndex++)
        {
          hUsn[uIndex][sIndex].resize (channelParams->m_reducedClusterNumber);
        }
    }

  NS_ASSERT (channelParams->m_reducedClusterNumber <= channelParams->m_clusterPhase.size ());
  NS_ASSERT (channelParams->m_reducedClusterNumber <= channelParams->m_clusterPower.size ());
  NS_ASSERT (channelParams->m_reducedClusterNumber <= channelParams->m_crossPolarizationPowerRatios.size ());
  NS_ASSERT (channelParams->m_reducedClusterNumber <= rayZoaRadian.size ());
  NS_ASSERT (channelParams->m_reducedClusterNumber <= rayZodRadian.size ());
  NS_ASSERT (channelParams->m_reducedClusterNumber <= rayAoaRadian.size ());
  NS_ASSERT (channelParams->m_reducedClusterNumber <= rayAodRadian.size ());
  NS_ASSERT (table3gpp->m_raysPerCluster <= channelParams->m_clusterPhase[0].size ());
  NS_ASSERT (table3gpp->m_raysPerCluster <= channelParams->m_crossPolarizationPowerRatios[0].size ());
  NS_ASSERT (table3gpp->m_raysPerCluster <= rayZoaRadian[0].size ());
  NS_ASSERT (table3gpp->m_raysPerCluster <= rayZodRadian[0].size ());
  NS_ASSERT (table3gpp->m_raysPerCluster <= rayAoaRadian[0].size ());
  NS_ASSERT (table3gpp->m_raysPerCluster <= rayAodRadian[0].size ());


  double x = sMob->GetPosition ().x - uMob->GetPosition ().x;
  double y = sMob->GetPosition ().y - uMob->GetPosition ().y;
  double distance2D = sqrt (x * x + y * y);
  // NOTE we assume hUT = min (height(a), height(b)) and
  // hBS = max (height (a), height (b))
  double hUt = std::min (sMob->GetPosition ().z, uMob->GetPosition ().z);
  double hBs = std::max (sMob->GetPosition ().z, uMob->GetPosition ().z);
  // compute the 3D distance using eq. 7.4-1
  double distance3D = std::sqrt (distance2D * distance2D + (hBs - hUt) * (hBs - hUt));

  Angles sAngle (uMob->GetPosition (), sMob->GetPosition ());
  Angles uAngle (sMob->GetPosition (), uMob->GetPosition ());


  // The following for loops computes the channel coefficients
  for (uint64_t uIndex = 0; uIndex < uSize; uIndex++)
    {
      Vector uLoc = uAntenna->GetElementLocation (uIndex);

      for (uint64_t sIndex = 0; sIndex < sSize; sIndex++)
        {
          Vector sLoc = sAntenna->GetElementLocation (sIndex);

          for (uint8_t nIndex = 0; nIndex < channelParams->m_reducedClusterNumber; nIndex++)
            {
              //Compute the N-2 weakest cluster, assuming 0 slant angle and a
              //polarization slant angle configured in the array (7.5-22)
              if (nIndex != channelParams->m_cluster1st && nIndex != channelParams->m_cluster2nd)
                {
                  std::complex<double> rays (0,0);
                  for (uint8_t mIndex = 0; mIndex < table3gpp->m_raysPerCluster; mIndex++)
                    {
                      DoubleVector initialPhase = channelParams->m_clusterPhase[nIndex][mIndex];
                      double k = channelParams->m_crossPolarizationPowerRatios[nIndex][mIndex];
                      //lambda_0 is accounted in the antenna spacing uLoc and sLoc.
                      double rxPhaseDiff = 2 * M_PI * (sin (rayZoaRadian[nIndex][mIndex]) * cos (rayAoaRadian[nIndex][mIndex]) * uLoc.x
                                                       + sin (rayZoaRadian[nIndex][mIndex]) * sin (rayAoaRadian[nIndex][mIndex]) * uLoc.y
                                                       + cos (rayZoaRadian[nIndex][mIndex]) * uLoc.z);

                      double txPhaseDiff = 2 * M_PI * (sin (rayZodRadian[nIndex][mIndex]) * cos (rayAodRadian[nIndex][mIndex]) * sLoc.x
                                                       + sin (rayZodRadian[nIndex][mIndex]) * sin (rayAodRadian[nIndex][mIndex]) * sLoc.y
                                                       + cos (rayZodRadian[nIndex][mIndex]) * sLoc.z);
                      // NOTE Doppler is computed in the CalcBeamformingGain function and is simplified to only account for the center angle of each cluster.

                      double rxFieldPatternPhi, rxFieldPatternTheta, txFieldPatternPhi, txFieldPatternTheta;
                      std::tie (rxFieldPatternPhi, rxFieldPatternTheta) = uAntenna->GetElementFieldPattern (Angles (channelParams->m_rayAoaRadian[nIndex][mIndex], channelParams->m_rayZoaRadian[nIndex][mIndex]));
                      std::tie (txFieldPatternPhi, txFieldPatternTheta) = sAntenna->GetElementFieldPattern (Angles (channelParams->m_rayAodRadian[nIndex][mIndex], channelParams->m_rayZodRadian[nIndex][mIndex]));
                      NS_ASSERT (4 <= initialPhase.size ());
                      rays += (std::complex<double> (cos (initialPhase[0]), sin (initialPhase[0])) * rxFieldPatternTheta * txFieldPatternTheta +
                               std::complex<double> (cos (initialPhase[1]), sin (initialPhase[1])) * std::sqrt (1 / k) * rxFieldPatternTheta * txFieldPatternPhi +
                               std::complex<double> (cos (initialPhase[2]), sin (initialPhase[2])) * std::sqrt (1 / k) * rxFieldPatternPhi * txFieldPatternTheta +
                               std::complex<double> (cos (initialPhase[3]), sin (initialPhase[3])) * rxFieldPatternPhi * txFieldPatternPhi)
                        * std::complex<double> (cos (rxPhaseDiff), sin (rxPhaseDiff))
                        * std::complex<double> (cos (txPhaseDiff), sin (txPhaseDiff));
                    }
                  rays *= sqrt (channelParams->m_clusterPower[nIndex] / table3gpp->m_raysPerCluster);
                  hUsn[uIndex][sIndex][nIndex] = rays;
                }
              else  //(7.5-28)
                {
                  std::complex<double> raysSub1 (0, 0);
                  std::complex<double> raysSub2 (0, 0);
                  std::complex<double> raysSub3 (0, 0);

                  for (uint8_t mIndex = 0; mIndex < table3gpp->m_raysPerCluster; mIndex++)
                    {
                      double k = channelParams->m_crossPolarizationPowerRatios[nIndex][mIndex];

                      //ZML:Just remind me that the angle offsets for the 3 subclusters were not generated correctly.
                      DoubleVector initialPhase = channelParams->m_clusterPhase[nIndex][mIndex];
                      NS_ASSERT (4 <= initialPhase.size ());

                      double rxPhaseDiff = 2 * M_PI * (sin (rayZoaRadian[nIndex][mIndex]) * cos (rayAoaRadian[nIndex][mIndex]) * uLoc.x
                                                       + sin (rayZoaRadian[nIndex][mIndex]) * sin (rayAoaRadian[nIndex][mIndex]) * uLoc.y
                                                       + cos (rayZoaRadian[nIndex][mIndex]) * uLoc.z);
                      double txPhaseDiff = 2 * M_PI * (sin (rayZodRadian[nIndex][mIndex]) * cos (rayAodRadian[nIndex][mIndex]) * sLoc.x
                                                       + sin (rayZodRadian[nIndex][mIndex]) * sin (rayAodRadian[nIndex][mIndex]) * sLoc.y
                                                       + cos (rayZodRadian[nIndex][mIndex]) * sLoc.z);

                      double rxFieldPatternPhi, rxFieldPatternTheta, txFieldPatternPhi, txFieldPatternTheta;
                      std::tie (rxFieldPatternPhi, rxFieldPatternTheta) = uAntenna->GetElementFieldPattern (Angles (rayAoaRadian[nIndex][mIndex], rayZoaRadian[nIndex][mIndex]));
                      std::tie (txFieldPatternPhi, txFieldPatternTheta) = sAntenna->GetElementFieldPattern (Angles (rayAodRadian[nIndex][mIndex], rayZodRadian[nIndex][mIndex]));

                      std::complex<double> raySub = (std::complex<double> (cos (initialPhase[0]), sin (initialPhase[0])) * rxFieldPatternTheta * txFieldPatternTheta +
                                                     std::complex<double> (cos (initialPhase[1]), sin (initialPhase[1])) * sqrt (1 / k) * rxFieldPatternTheta * txFieldPatternPhi +
                                                     std::complex<double> (cos (initialPhase[2]), sin (initialPhase[2])) * sqrt (1 / k) * rxFieldPatternPhi * txFieldPatternTheta +
                                                     std::complex<double> (cos (initialPhase[3]), sin (initialPhase[3])) * rxFieldPatternPhi * txFieldPatternPhi)
                        * std::complex<double> (cos (rxPhaseDiff), sin (rxPhaseDiff))
                        * std::complex<double> (cos (txPhaseDiff), sin (txPhaseDiff));

                      switch (mIndex)
                        {
                          case 9:
                          case 10:
                          case 11:
                          case 12:
                          case 17:
                          case 18:
                            raysSub2 += raySub;
                            break;
                          case 13:
                          case 14:
                          case 15:
                          case 16:
                            raysSub3 += raySub;
                            break;
                          default:                      //case 1,2,3,4,5,6,7,8,19,20
                            raysSub1 += raySub;
                            break;
                        }
                    }
                  raysSub1 *= sqrt (channelParams->m_clusterPower[nIndex] / table3gpp->m_raysPerCluster);
                  raysSub2 *= sqrt (channelParams->m_clusterPower[nIndex] / table3gpp->m_raysPerCluster);
                  raysSub3 *= sqrt (channelParams->m_clusterPower[nIndex] / table3gpp->m_raysPerCluster);
                  hUsn[uIndex][sIndex][nIndex] = raysSub1;
                  hUsn[uIndex][sIndex].push_back (raysSub2);
                  hUsn[uIndex][sIndex].push_back (raysSub3);
                }
            }

          if (channelParams->m_losCondition == ChannelCondition::LOS) //(7.5-29) && (7.5-30)
            {
              std::complex<double> ray (0, 0);
              double rxPhaseDiff = 2 * M_PI * (sin (uAngle.GetInclination ()) * cos (uAngle.GetAzimuth ()) * uLoc.x
                                               + sin (uAngle.GetInclination ()) * sin (uAngle.GetAzimuth ()) * uLoc.y
                                               + cos (uAngle.GetInclination ()) * uLoc.z);
              double txPhaseDiff = 2 * M_PI * (sin (sAngle.GetInclination ()) * cos (sAngle.GetAzimuth ()) * sLoc.x
                                               + sin (sAngle.GetInclination ()) * sin (sAngle.GetAzimuth ()) * sLoc.y
                                               + cos (sAngle.GetInclination ()) * sLoc.z);

              double rxFieldPatternPhi, rxFieldPatternTheta, txFieldPatternPhi, txFieldPatternTheta;
              std::tie (rxFieldPatternPhi, rxFieldPatternTheta) = uAntenna->GetElementFieldPattern (Angles (uAngle.GetAzimuth (), uAngle.GetInclination ()));
              std::tie (txFieldPatternPhi, txFieldPatternTheta) = sAntenna->GetElementFieldPattern (Angles (sAngle.GetAzimuth (), sAngle.GetInclination ()));

              double lambda = 3e8 / m_frequency; // the wavelength of the carrier frequency

              ray = (rxFieldPatternTheta * txFieldPatternTheta - rxFieldPatternPhi * txFieldPatternPhi)
                * std::complex<double> (cos (-2 * M_PI * distance3D / lambda), sin (-2 * M_PI * distance3D / lambda))
                * std::complex<double> (cos (rxPhaseDiff), sin (rxPhaseDiff))
                * std::complex<double> (cos (txPhaseDiff), sin (txPhaseDiff));

              double kLinear = pow (10, channelParams->m_K_factor / 10);
              // the LOS path should be attenuated if blockage is enabled.
              hUsn[uIndex][sIndex][0] = sqrt (1 / (kLinear + 1)) * hUsn[uIndex][sIndex][0] + sqrt (kLinear / (1 + kLinear)) * ray / pow (10, channelParams->m_attenuation_dB[0] / 10);           //(7.5-30) for tau = tau1
              double tempSize = hUsn[uIndex][sIndex].size ();
              for (uint8_t nIndex = 1; nIndex < tempSize; nIndex++)
                {
                  hUsn[uIndex][sIndex][nIndex] *= sqrt (1 / (kLinear + 1)); //(7.5-30) for tau = tau2...taunN
                }
            }
        }
    }

  NS_LOG_DEBUG ("Husn (sAntenna, uAntenna):" << sAntenna->GetId () << ", " << uAntenna->GetId ());
  for (auto& i:hUsn)
    {
      for (auto& j:i)
        {
          for (auto& k:j)
            {
              NS_LOG_DEBUG (" " << k << ",");
            }
        }
    }
  NS_LOG_INFO ("size of coefficient matrix =[" << hUsn.size () << "][" << hUsn[0].size () << "][" << hUsn[0][0].size () << "]");
  channelMatrix->m_channel = hUsn;
  return channelMatrix;
}

std::pair<double, double>
ThreeGppChannelModel::WrapAngles (double azimuthRad, double inclinationRad)
{
  inclinationRad = WrapTo2Pi (inclinationRad);
  if (inclinationRad > M_PI)
    {
      // inclination must be in [0, M_PI]
      inclinationRad -= M_PI;
      azimuthRad += M_PI;
    }

  azimuthRad = WrapTo2Pi (azimuthRad);

  NS_ASSERT_MSG (0 <= inclinationRad && inclinationRad <= M_PI,
                 "inclinationRad=" << inclinationRad << " not valid, should be in [0, pi]");
  NS_ASSERT_MSG (0 <= azimuthRad && azimuthRad <= 2 * M_PI,
                 "azimuthRad=" << azimuthRad << " not valid, should be in [0, 2*pi]");

  return std::make_pair (azimuthRad, inclinationRad);
}

MatrixBasedChannelModel::DoubleVector
ThreeGppChannelModel::CalcAttenuationOfBlockage (const Ptr<ThreeGppChannelModel::ThreeGppChannelParams> channelParams,
                                                 const DoubleVector &clusterAOA,
                                                 const DoubleVector &clusterZOA) const
{
  NS_LOG_FUNCTION (this);

  DoubleVector powerAttenuation;
  uint8_t clusterNum = clusterAOA.size ();
  for (uint8_t cInd = 0; cInd < clusterNum; cInd++)
    {
      powerAttenuation.push_back (0); //Initial power attenuation for all clusters to be 0 dB;
    }
  //step a: the number of non-self blocking blockers is stored in m_numNonSelfBlocking.

  //step b:Generate the size and location of each blocker
  //generate self blocking (i.e., for blockage from the human body)
  //table 7.6.4.1-1 Self-blocking region parameters.
  // Defaults: landscape mode
  double phiSb = 40;
  double xSb = 160;
  double thetaSb = 110;
  double ySb = 75;
  if (m_portraitMode)
    {
      phiSb = 260;
      xSb = 120;
      thetaSb = 100;
      ySb = 80;
    }

  //generate or update non-self blocking
  if (channelParams->m_nonSelfBlocking.size () == 0) //generate new blocking regions
    {
      for (uint16_t blockInd = 0; blockInd < m_numNonSelfBlocking; blockInd++)
        {
          //draw value from table 7.6.4.1-2 Blocking region parameters
          DoubleVector table;
          table.push_back (m_normalRv->GetValue ()); //phi_k: store the normal RV that will be mapped to uniform (0,360) later.
          if (m_scenario == "InH-OfficeMixed" || m_scenario == "InH-OfficeOpen")
            {
              table.push_back (m_uniformRv->GetValue (15, 45)); //x_k
              table.push_back (90);  //Theta_k
              table.push_back (m_uniformRv->GetValue (5, 15)); //y_k
              table.push_back (2);  //r
            }
          else
            {
              table.push_back (m_uniformRv->GetValue (5, 15)); //x_k
              table.push_back (90);  //Theta_k
              table.push_back (5);  //y_k
              table.push_back (10);  //r
            }
          channelParams->m_nonSelfBlocking.push_back (table);
        }
    }
  else
    {
      double deltaX = sqrt (pow (channelParams->m_preLocUT.x - channelParams->m_locUT.x, 2) + pow (channelParams->m_preLocUT.y - channelParams->m_locUT.y, 2));
      //if deltaX and speed are both 0, the autocorrelation is 1, skip updating
      if (deltaX > 1e-6 || m_blockerSpeed > 1e-6)
        {
          double corrDis;
          //draw value from table 7.6.4.1-4: Spatial correlation distance for different m_scenarios.
          if (m_scenario == "InH-OfficeMixed" || m_scenario == "InH-OfficeOpen")
            {
              //InH, correlation distance = 5;
              corrDis = 5;
            }
          else
            {
              if (channelParams->m_o2iCondition == ChannelCondition::O2I) // outdoor to indoor
                {
                  corrDis = 5;
                }
              else  //LOS or NLOS
                {
                  corrDis = 10;
                }
            }
          double R;
          if (m_blockerSpeed > 1e-6) // speed not equal to 0
            {
              double corrT = corrDis / m_blockerSpeed;
              R = exp (-1 * (deltaX / corrDis + (Now ().GetSeconds () - channelParams->m_generatedTime.GetSeconds ()) / corrT));
            }
          else
            {
              R = exp (-1 * (deltaX / corrDis));
            }

          NS_LOG_INFO ("Distance change:" << deltaX << " Speed:" << m_blockerSpeed
                                          << " Time difference:" << Now ().GetSeconds () - channelParams->m_generatedTime.GetSeconds ()
                                          << " correlation:" << R);

          //In order to generate correlated uniform random variables, we first generate correlated normal random variables and map the normal RV to uniform RV.
          //Notice the correlation will change if the RV is transformed from normal to uniform.
          //To compensate the distortion, the correlation of the normal RV is computed
          //such that the uniform RV would have the desired correlation when transformed from normal RV.

          //The following formula was obtained from MATLAB numerical simulation.

          if (R * R * (-0.069) + R * 1.074 - 0.002 < 1) //transform only when the correlation of normal RV is smaller than 1
            {
              R = R * R * (-0.069) + R * 1.074 - 0.002;
            }
          for (uint16_t blockInd = 0; blockInd < m_numNonSelfBlocking; blockInd++)
            {

              //Generate a new correlated normal RV with the following formula
              channelParams->m_nonSelfBlocking[blockInd][PHI_INDEX] =
                R * channelParams->m_nonSelfBlocking[blockInd][PHI_INDEX] + sqrt (1 - R * R) * m_normalRv->GetValue ();
            }
        }

    }

  //step c: Determine the attenuation of each blocker due to blockers
  for (uint8_t cInd = 0; cInd < clusterNum; cInd++)
    {
      NS_ASSERT_MSG (clusterAOA[cInd] >= 0 && clusterAOA[cInd] <= 360, "the AOA should be the range of [0,360]");
      NS_ASSERT_MSG (clusterZOA[cInd] >= 0 && clusterZOA[cInd] <= 180, "the ZOA should be the range of [0,180]");

      //check self blocking
      NS_LOG_INFO ("AOA=" << clusterAOA[cInd] << " Block Region[" << phiSb - xSb / 2 << ","
                          << phiSb + xSb / 2 << "]");
      NS_LOG_INFO ("ZOA=" << clusterZOA[cInd] << " Block Region[" << thetaSb - ySb / 2 << ","
                          << thetaSb + ySb / 2 << "]");
      if (std::abs (clusterAOA[cInd] - phiSb) < (xSb / 2)
          && std::abs (clusterZOA[cInd] - thetaSb) < (ySb / 2))
        {
          powerAttenuation[cInd] += 30;               //anttenuate by 30 dB.
          NS_LOG_INFO ("Cluster[" << +cInd << "] is blocked by self blocking region and reduce 30 dB power,"
                       "the attenuation is [" << powerAttenuation[cInd] << " dB]");
        }

      //check non-self blocking
      for (uint16_t blockInd = 0; blockInd < m_numNonSelfBlocking; blockInd++)
        {
          //The normal RV is transformed to uniform RV with the desired correlation.
          double phiK = (0.5 * erfc (-1 * channelParams->m_nonSelfBlocking[blockInd][PHI_INDEX] / sqrt (2))) * 360;
          while (phiK > 360)
            {
              phiK -= 360;
            }

          while (phiK < 0)
            {
              phiK += 360;
            }

          double xK = channelParams->m_nonSelfBlocking[blockInd][X_INDEX];
          double thetaK = channelParams->m_nonSelfBlocking[blockInd][THETA_INDEX];
          double yK = channelParams->m_nonSelfBlocking[blockInd][Y_INDEX];

          NS_LOG_INFO ("AOA=" << clusterAOA[cInd] << " Block Region[" << phiK - xK << "," << phiK + xK << "]");
          NS_LOG_INFO ("ZOA=" << clusterZOA[cInd] << " Block Region[" << thetaK - yK << "," << thetaK + yK << "]");

          if ( std::abs (clusterAOA[cInd] - phiK) < (xK)
               && std::abs (clusterZOA[cInd] - thetaK) < (yK))
            {
              double A1 = clusterAOA[cInd] - (phiK + xK / 2); //(7.6-24)
              double A2 = clusterAOA[cInd] - (phiK - xK / 2); //(7.6-25)
              double Z1 = clusterZOA[cInd] - (thetaK + yK / 2); //(7.6-26)
              double Z2 = clusterZOA[cInd] - (thetaK - yK / 2); //(7.6-27)
              int signA1, signA2, signZ1, signZ2;
              //draw sign for the above parameters according to table 7.6.4.1-3 Description of signs
              if (xK / 2 < clusterAOA[cInd] - phiK && clusterAOA[cInd] - phiK <= xK)
                {
                  signA1 = -1;
                }
              else
                {
                  signA1 = 1;
                }
              if (-1 * xK < clusterAOA[cInd] - phiK && clusterAOA[cInd] - phiK <= -1 * xK / 2)
                {
                  signA2 = -1;
                }
              else
                {
                  signA2 = 1;
                }

              if (yK / 2 < clusterZOA[cInd] - thetaK && clusterZOA[cInd] - thetaK <= yK)
                {
                  signZ1 = -1;
                }
              else
                {
                  signZ1 = 1;
                }
              if (-1 * yK < clusterZOA[cInd] - thetaK && clusterZOA[cInd] - thetaK <= -1 * yK / 2)
                {
                  signZ2 = -1;
                }
              else
                {
                  signZ2 = 1;
                }
              double lambda = 3e8 / m_frequency;
              double fA1 = atan (signA1 * M_PI / 2 * sqrt (M_PI / lambda *
                                                           channelParams->m_nonSelfBlocking[blockInd][R_INDEX] * (1 / cos (DegreesToRadians (A1)) - 1))) / M_PI; //(7.6-23)
              double fA2 = atan (signA2 * M_PI / 2 * sqrt (M_PI / lambda *
                                                           channelParams->m_nonSelfBlocking[blockInd][R_INDEX] * (1 / cos (DegreesToRadians (A2)) - 1))) / M_PI;
              double fZ1 = atan (signZ1 * M_PI / 2 * sqrt (M_PI / lambda *
                                                           channelParams->m_nonSelfBlocking[blockInd][R_INDEX] * (1 / cos (DegreesToRadians (Z1)) - 1))) / M_PI;
              double fZ2 = atan (signZ2 * M_PI / 2 * sqrt (M_PI / lambda *
                                                           channelParams->m_nonSelfBlocking[blockInd][R_INDEX] * (1 / cos (DegreesToRadians (Z2)) - 1))) / M_PI;
              double lDb = -20 * log10 (1 - (fA1 + fA2) * (fZ1 + fZ2));                  //(7.6-22)
              powerAttenuation[cInd] += lDb;
              NS_LOG_INFO ("Cluster[" << +cInd << "] is blocked by no-self blocking, the loss is ["
                                      << lDb << "] dB");
            }
        }
    }
  return powerAttenuation;
}

void
ThreeGppChannelModel::Shuffle (double * first, double * last) const
{
  for (auto i = (last - first) - 1; i > 0; --i)
    {
      std::swap (first[i], first[m_uniformRvShuffle->GetInteger (0, i)]);
    }
}

int64_t
ThreeGppChannelModel::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_normalRv->SetStream (stream);
  m_uniformRv->SetStream (stream + 1);
  m_uniformRvShuffle->SetStream (stream + 2);
  m_uniformRvDoppler->SetStream (stream + 3);
  return 4;
}

}  // namespace ns3
