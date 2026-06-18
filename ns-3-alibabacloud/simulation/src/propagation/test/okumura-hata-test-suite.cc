/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011,2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */


#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/okumura-hata-propagation-loss-model.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/enum.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OkumuraHataPropagationLossModelTest");

/**
 * \ingroup propagation-tests
 *
 * \brief OkumuraHataPropagationLossModel Test Case
 */
class OkumuraHataPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   * 
   * \param freq carrier frequency in Hz
   * \param dist 2D distance between UT and BS in meters
   * \param hb height of BS in meters
   * \param hm height of UT in meters
   * \param env environment type
   * \param city city type
   * \param refValue reference loss value
   * \param name TestCase name
   */ 
  OkumuraHataPropagationLossModelTestCase (double freq, double dist, double hb, double hm, EnvironmentType env, CitySize city, double refValue, std::string name);
  virtual ~OkumuraHataPropagationLossModelTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Create a MobilityModel
   * \param index mobility model index
   * \return a new MobilityModel
   */
  Ptr<MobilityModel> CreateMobilityModel (uint16_t index);

  double m_freq;    //!< carrier frequency in Hz
  double m_dist;    //!< 2D distance between UT and BS in meters
  double m_hb;      //!< height of BS in meters
  double m_hm;      //!< height of UT in meters
  EnvironmentType m_env;  //!< environment type
  CitySize m_city;        //!< city type
  double m_lossRef; //!< reference loss
};

OkumuraHataPropagationLossModelTestCase::OkumuraHataPropagationLossModelTestCase (double freq, double dist, double hb, double hm, EnvironmentType env, CitySize city, double refValue, std::string name)
  : TestCase (name),
    m_freq (freq),
    m_dist (dist),
    m_hb (hb),
    m_hm (hm),
    m_env (env),
    m_city (city),
    m_lossRef (refValue)
{
}

OkumuraHataPropagationLossModelTestCase::~OkumuraHataPropagationLossModelTestCase ()
{
}



void
OkumuraHataPropagationLossModelTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);


  Ptr<MobilityModel> mma = CreateObject<ConstantPositionMobilityModel> ();
  mma->SetPosition (Vector (0.0, 0.0, m_hb));

  Ptr<MobilityModel> mmb = CreateObject<ConstantPositionMobilityModel> ();
  mmb->SetPosition (Vector (m_dist, 0.0, m_hm));

  Ptr<OkumuraHataPropagationLossModel> propagationLossModel = CreateObject<OkumuraHataPropagationLossModel> ();
  propagationLossModel->SetAttribute ("Frequency", DoubleValue (m_freq));
  propagationLossModel->SetAttribute ("Environment", EnumValue (m_env));
  propagationLossModel->SetAttribute ("CitySize", EnumValue (m_city));

  double loss = propagationLossModel->GetLoss (mma, mmb);

  NS_LOG_INFO ("Calculated loss: " << loss);
  NS_LOG_INFO ("Theoretical loss: " << m_lossRef);
 
  NS_TEST_ASSERT_MSG_EQ_TOL (loss, m_lossRef, 0.1, "Wrong loss!");

}


/**
 * \ingroup propagation-tests
 *
 * \brief OkumuraHataPropagationLossModel TestSuite
 * 
 * This TestSuite tests the following cases:
 *   - UrbanEnvironment - Large City (original OH and COST231 OH)
 *   - UrbanEnvironment - Small City (original OH and COST231 OH)
 *   - SubUrbanEnvironment (original OH only)
 *   - OpenAreasEnvironment (original OH only)
 */
class OkumuraHataPropagationLossModelTestSuite : public TestSuite
{
public:
  OkumuraHataPropagationLossModelTestSuite ();
};

OkumuraHataPropagationLossModelTestSuite::OkumuraHataPropagationLossModelTestSuite ()
  : TestSuite ("okumura-hata", SYSTEM)
{
  LogComponentEnable ("OkumuraHataPropagationLossModelTest", LOG_LEVEL_ALL);
  // reference values obtained with the octave scripts in src/propagation/test/reference/

  double freq = 869e6; // this will use the original OH model
  AddTestCase (new OkumuraHataPropagationLossModelTestCase (freq, 2000, 30, 1, UrbanEnvironment, LargeCity, 137.93, "original OH Urban Large city"), TestCase::QUICK);
  AddTestCase (new OkumuraHataPropagationLossModelTestCase (freq, 2000, 30, 1, UrbanEnvironment, SmallCity, 137.88, "original OH Urban small city"), TestCase::QUICK);
  AddTestCase (new OkumuraHataPropagationLossModelTestCase (freq, 2000, 30, 1, SubUrbanEnvironment, LargeCity, 128.03, "original OH SubUrban"), TestCase::QUICK);
  AddTestCase (new OkumuraHataPropagationLossModelTestCase (freq, 2000, 30, 1, OpenAreasEnvironment, LargeCity, 110.21, "original OH OpenAreas"), TestCase::QUICK);

  freq = 2.1140e9;  // this will use the extended COST231 OH model
  AddTestCase (new OkumuraHataPropagationLossModelTestCase (freq, 2000, 30, 1, UrbanEnvironment, LargeCity, 148.55, "COST231 OH Urban Large city"), TestCase::QUICK);
  AddTestCase (new OkumuraHataPropagationLossModelTestCase (freq, 2000, 30, 1, UrbanEnvironment, SmallCity, 150.64, "COST231 OH Urban small city and suburban"), TestCase::QUICK);
}


/// Static variable for test initialization
static OkumuraHataPropagationLossModelTestSuite g_okumuraHataTestSuite;
