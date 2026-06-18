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
 * Author:  Biljana Bojovic<bbojovic@cttc.es>
 *          Dizhi Zhou <dizhi.zhou@gmail.com>
 *          Marco Miozzo <marco.miozzo@cttc.es>,
 *          Nicola Baldo <nbaldo@cttc.es>
 *
 */

#ifndef LENA_TEST_CQA_FF_MAC_SCHEDULER_H
#define LENA_TEST_CQA_FF_MAC_SCHEDULER_H

#include "ns3/simulator.h"
#include "ns3/test.h"


using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This is a system test program. The test is based on a scenario with single eNB and several UEs.
 *  The goal of the test is validating if the obtained throughput performance is consistent with the definition of CQA scheduler.
 */

class LenaCqaFfMacSchedulerTestCase1 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nUser number of UE nodes
   * \param dist distance between nodes
   * \param thrRefDl DL throughput reference
   * \param thrRefUl UL throughput reference
   * \param packetSize packet size
   * \param interval time interval
   * \param errorModelEnabled error model enabled?
   */
  LenaCqaFfMacSchedulerTestCase1 (uint16_t nUser, double dist, double thrRefDl, double thrRefUl, uint16_t packetSize, uint16_t interval, bool  errorModelEnabled);
  virtual ~LenaCqaFfMacSchedulerTestCase1 ();

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
  uint16_t m_packetSize;  ///< packet size in bytes
  uint16_t m_interval;    ///< interval time in ms
  double m_thrRefDl; ///< estimated downlink throughput
  double m_thrRefUl; ///< estimated uplink throughput
  bool m_errorModelEnabled; ///< whether error model is enabled
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This is a system test program. The test is based on a scenario with single eNB and several UEs.
 *  The goal of the test is validating if the obtained throughput performance is consistent with the definition of 
*   CQA scheduler when the UEs with different SINRs.
*/

class LenaCqaFfMacSchedulerTestCase2 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param dist distance between nodes
   * \param estThrCqaDl estimated CQA throughput in the downlink
   * \param packetSize packet size
   * \param interval the UDP packet time interval
   * \param errorModelEnabled whether the error model enabled is enabled in the test case
   */
  LenaCqaFfMacSchedulerTestCase2 (std::vector<double> dist, std::vector<uint32_t> estThrCqaDl, std::vector<uint16_t> packetSize, uint16_t interval, bool  errorModelEnabled);
  virtual ~LenaCqaFfMacSchedulerTestCase2 ();

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
  std::vector<uint16_t> m_packetSize;  ///< packet size in bytes
  uint16_t m_interval;    ///< UDP interval time in ms
  std::vector<uint32_t> m_estThrCqaDl; ///< estimated throughput CQA DL
  bool m_errorModelEnabled; ///< whether the error model is enabled
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief The test suite for testing CQA scheduler functionality
 */

class LenaTestCqaFfMacSchedulerSuite : public TestSuite
{
public:
  LenaTestCqaFfMacSchedulerSuite ();
};

#endif /* LENA_TEST_CQA_FF_MAC_SCHEDULER_H */
