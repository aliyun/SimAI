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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>,
 *         Nicola Baldo <nbaldo@cttc.es>
 *         Dizhi Zhou <dizhi.zhou@gmail.com>
 */

#ifndef LENA_TEST_TDMT_FF_MAC_SCHEDULER_H
#define LENA_TEST_TDMT_FF_MAC_SCHEDULER_H

#include "ns3/simulator.h"
#include "ns3/test.h"


using namespace ns3;


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This system test program creates different test cases with a single eNB and 
 * several UEs, all having the same Radio Bearer specification. In each test 
 * case, the UEs see the same SINR from the eNB; different test cases are 
 * implemented obtained by using different SINR values and different numbers of 
 * UEs. The test consists on checking that the obtained throughput performance 
 * is consistent with the definition of maximum throughput
 * scheduling
 */
class LenaTdMtFfMacSchedulerTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nUser the number of UE nodes
   * \param dist the distance between UE nodes and eNodeB
   * \param thrRefDl the DL throughput reference
   * \param thrRefUl the UL throughput reference
   * \param errorModelEnabled if true the error model is enabled
   */
  LenaTdMtFfMacSchedulerTestCase (uint16_t nUser, double dist, double thrRefDl, double thrRefUl,bool errorModelEnabled);
  virtual ~LenaTdMtFfMacSchedulerTestCase ();

private:
  /**
   * Builds the test name string based on provided parameter values
   * \param nUser the number of UE nodes
   * \param dist the distnace between nodes
   * \returns the name string
   */
  static std::string BuildNameString (uint16_t nUser, double dist);
  virtual void DoRun (void);
  uint16_t m_nUser; ///< number of UE nodes
  double m_dist; ///< the distance between nodes
  double m_thrRefDl; ///< the DL throughput reference
  double m_thrRefUl; ///< the UL throughput reference
  bool m_errorModelEnabled; ///< whether the error model is enabled
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for TdMtFfMacScheduler tests.
 */
class LenaTestTdMtFfMacSchedulerSuite : public TestSuite
{
public:
  LenaTestTdMtFfMacSchedulerSuite ();
};




#endif /* LENA_TEST_TDMT_FF_MAC_SCHEDULER_H */
