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

#ifndef LENA_TEST_FDBET_FF_MAC_SCHEDULER_H
#define LENA_TEST_FDBET_FF_MAC_SCHEDULER_H

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
 * is equal among users is consistent with the definition of blind equal throughput
 * scheduling
 */
class LenaFdBetFfMacSchedulerTestCase1 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nUser number of UE nodes
   * \param dist distance between nodes
   * \param thrRefDl DL throughput reference
   * \param thrRefUl UL throughput reference
   * \param errorModelEnabled error model enabled?
   */
  LenaFdBetFfMacSchedulerTestCase1 (uint16_t nUser, double dist, double thrRefDl, double thrRefUl,bool errorModelEnabled);
  virtual ~LenaFdBetFfMacSchedulerTestCase1 ();

private:
  /**
   *  Builds the test name string based on provided parameter values
   *
   * \param nUser number of UE nodes
   * \param dist distance between nodes
   * \returns name string
   */
  static std::string BuildNameString (uint16_t nUser, double dist);
  virtual void DoRun (void);
  uint16_t m_nUser; ///< number of UE nodes
  double m_dist; ///< distance between the nodes
  double m_thrRefDl; ///< DL throughput reference
  double m_thrRefUl; ///< UL throughput reference
  bool m_errorModelEnabled; ///< error model enabled?
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test case is simillar to the one defined in 
 * LenaFdBetFfMacSchedulerTestCase1, with the difference that UEs are 
 * placed in such a way to experience different SINRs from eNodeB.
 */
class LenaFdBetFfMacSchedulerTestCase2 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param dist distance between nodes
   * \param achievableRateDl DL achievable rate
   * \param estThrFdBetUl UL estimated throughput bet UL
   * \param errorModelEnabled error model enabled?
   */
  LenaFdBetFfMacSchedulerTestCase2 (std::vector<double> dist, std::vector<uint32_t> achievableRateDl, std::vector<uint32_t> estThrFdBetUl, bool errorModelEnabled);
  virtual ~LenaFdBetFfMacSchedulerTestCase2 ();

private:
  /**
   *  Builds the test name string based on provided parameter values
   *
   * \param nUser number of UE nodes
   * \param dist distance between nodes
   * \returns name string
   */
  static std::string BuildNameString (uint16_t nUser, std::vector<double> dist);
  virtual void DoRun (void);
  uint16_t m_nUser; ///< number of UE nodes
  std::vector<double> m_dist; ///< distance between the nodes
  std::vector<uint32_t> m_achievableRateDl; ///< DL achievable rate
  std::vector<uint32_t> m_estThrFdBetUl; ///< estimated throughput FDBET UL
  bool m_errorModelEnabled; ///< specifies whether the error model is enabled
};



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for FDBetFfMacScheduler test case.
 */

class LenaTestFdBetFfMacSchedulerSuite : public TestSuite
{
public:
  LenaTestFdBetFfMacSchedulerSuite ();
};

#endif /* LENA_TEST_FDBET_FF_MAC_SCHEDULER_H */
