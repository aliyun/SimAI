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

#ifndef LENA_TEST_FDTBFQ_FF_MAC_SCHEDULER_H
#define LENA_TEST_FDTBFQ_FF_MAC_SCHEDULER_H

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
 * is equal among users is consistent with the definition of token bank fair  
 * queue scheduling
 */
class LenaFdTbfqFfMacSchedulerTestCase1 : public TestCase
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
  LenaFdTbfqFfMacSchedulerTestCase1 (uint16_t nUser, double dist, double thrRefDl, double thrRefUl, uint16_t packetSize, uint16_t interval, bool  errorModelEnabled);
  virtual ~LenaFdTbfqFfMacSchedulerTestCase1 ();

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
  double m_thrRefDl; ///< DL throughput reference
  double m_thrRefUl; ///< UL throughput reference
  bool m_errorModelEnabled; ///< whether the error model is enabled
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test case is simillar to the one defined in 
 * LenaFdTbfqFfMacSchedulerTestCase1, with the difference that UEs are 
 * placed in such a way to experience different SINRs from eNodeB.
 */
class LenaFdTbfqFfMacSchedulerTestCase2 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param dist distance between nodes
   * \param estThrFdTbfqDl estimated DL throughput
   * \param packetSize packet size
   * \param interval time interval
   * \param errorModelEnabled indicates whether the error model is enabled
   */
  LenaFdTbfqFfMacSchedulerTestCase2 (std::vector<double> dist, std::vector<uint32_t> estThrFdTbfqDl, std::vector<uint16_t> packetSize, uint16_t interval, bool  errorModelEnabled);
  virtual ~LenaFdTbfqFfMacSchedulerTestCase2 ();

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
  uint16_t m_interval;    ///< interval time in ms
  std::vector<uint32_t> m_estThrFdTbfqDl; ///< estimated thrpughput FDTBFQ DL
  bool m_errorModelEnabled; ///< error model enabled?
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suit for FdTbfqFfMacScheduler test.
 */
class LenaTestFdTbfqFfMacSchedulerSuite : public TestSuite
{
public:
  LenaTestFdTbfqFfMacSchedulerSuite ();
};



#endif /* LENA_TEST_FDTBFQ_FF_MAC_SCHEDULER_H */
