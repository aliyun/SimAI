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
#include <ns3/kun-2600-mhz-propagation-loss-model.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/enum.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Kun2600MhzPropagationLossModelTest");

/**
 * \ingroup propagation-tests
 *
 * \brief Kun2600MhzPropagationLossModel Test Case
 * 
 */
class Kun2600MhzPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   * 
   * \param dist 2D distance between UT and BS in meters
   * \param hb height of BS in meters
   * \param hm height of UT in meters
   * \param refValue reference loss value
   * \param name TestCase name
   */ 
  Kun2600MhzPropagationLossModelTestCase (double dist, double hb, double hm, double refValue, std::string name);
  virtual ~Kun2600MhzPropagationLossModelTestCase ();

private:
  virtual void DoRun (void);
  
  /**
   * Create a MobilityModel
   * \param index mobility model index
   * \return a new MobilityModel
   */
  Ptr<MobilityModel> CreateMobilityModel (uint16_t index);

  double m_dist;    //!< 2D distance between UT and BS in meters
  double m_hb;      //!< height of BS in meters
  double m_hm;      //!< height of UT in meters
  double m_lossRef; //!< reference loss
};

Kun2600MhzPropagationLossModelTestCase::Kun2600MhzPropagationLossModelTestCase (double dist, double hb, double hm, double refValue, std::string name)
  : TestCase (name),
    m_dist (dist),
    m_hb (hb),
    m_hm (hm),
    m_lossRef (refValue)
{
}

Kun2600MhzPropagationLossModelTestCase::~Kun2600MhzPropagationLossModelTestCase ()
{
}



void
Kun2600MhzPropagationLossModelTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);


  Ptr<MobilityModel> mma = CreateObject<ConstantPositionMobilityModel> ();
  mma->SetPosition (Vector (0.0, 0.0, m_hb));

  Ptr<MobilityModel> mmb = CreateObject<ConstantPositionMobilityModel> ();
  mmb->SetPosition (Vector (m_dist, 0.0, m_hm));

  Ptr<Kun2600MhzPropagationLossModel> propagationLossModel = CreateObject<Kun2600MhzPropagationLossModel> ();

  double loss = propagationLossModel->GetLoss (mma, mmb);

  NS_LOG_INFO ("Calculated loss: " << loss);
  NS_LOG_INFO ("Theoretical loss: " << m_lossRef);
 
  NS_TEST_ASSERT_MSG_EQ_TOL (loss, m_lossRef, 0.1, "Wrong loss!");

}



/**
 * \ingroup propagation-tests
 *
 * \brief Kun2600MhzPropagationLossModel TestSuite
 * 
 */
class Kun2600MhzPropagationLossModelTestSuite : public TestSuite
{
public:
  Kun2600MhzPropagationLossModelTestSuite ();
};



Kun2600MhzPropagationLossModelTestSuite::Kun2600MhzPropagationLossModelTestSuite ()
  : TestSuite ("kun-2600-mhz", SYSTEM)
{
  LogComponentEnable ("Kun2600MhzPropagationLossModelTest", LOG_LEVEL_ALL);

  AddTestCase (new Kun2600MhzPropagationLossModelTestCase (2000, 30, 1,  121.83, "dist=2000m"), TestCase::QUICK);
}

/// Static variable for test initialization
static Kun2600MhzPropagationLossModelTestSuite g_kun2600MhzTestSuite;
