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
 */

#ifndef LENA_TEST_PF_FF_MAC_SCHEDULER_H
#define LENA_TEST_PF_FF_MAC_SCHEDULER_H

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
 * is equal among users is consistent with the definition of proportional 
 * fair scheduling
 */
class LenaPfFfMacSchedulerTestCase1 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nUser the number of UE nodes
   * \param dist the distance between nodes
   * \param thrRefDl the DL throughput reference
   * \param thrRefUl the UL throughput reference
   * \param errorModelEnabled if true the error model is enabled
   */
  LenaPfFfMacSchedulerTestCase1 (uint16_t nUser, double dist, double thrRefDl, double thrRefUl, bool errorModelEnabled);
  virtual ~LenaPfFfMacSchedulerTestCase1 ();

private:
  /**
   * Build name string
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
  bool m_errorModelEnabled; ///< whether error model is enabled
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lena PfFf Mac Scheduler Test Case 2
 */
class LenaPfFfMacSchedulerTestCase2 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param dist the distance between nodes
   * \param estThrPfDl the estimated DL throughput PF
   * \param estThrPfUl the estimated UL throughput PF
   * \param errorModelEnabled if true the error model is enabled
   */
  LenaPfFfMacSchedulerTestCase2 (std::vector<double> dist, std::vector<uint32_t> estThrPfDl, std::vector<uint32_t> estThrPfUl, bool errorModelEnabled);
  virtual ~LenaPfFfMacSchedulerTestCase2 ();

private:
  /**
   * Builds the test name string based on provided parameter values
   * \param nUser the number of UE nodes
   * \param dist the distnace between nodes
   * \returns the name string
   */
  static std::string BuildNameString (uint16_t nUser, std::vector<double> dist);
  virtual void DoRun (void);
  uint16_t m_nUser; ///< number of UE nodes
  std::vector<double> m_dist; ///< the distance between nodes
  std::vector<uint32_t> m_estThrPfDl; ///< the estimated DL throughput 
  std::vector<uint32_t> m_estThrPfUl; ///< the estimated UL throughput
  bool m_errorModelEnabled; ///< indicates whether the error model is enabled
};



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief PfFfMacScheduler test suite
 */

class LenaTestPfFfMacSchedulerSuite : public TestSuite
{
public:
  LenaTestPfFfMacSchedulerSuite ();
};

#endif /* LENA_TEST_PF_FF_MAC_SCHEDULER_H */
