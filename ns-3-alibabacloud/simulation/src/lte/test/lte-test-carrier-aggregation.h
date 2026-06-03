/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Biljana Bojovic <biljana.bojovic@cttc.es>
 *
 */

#ifndef TEST_CARRIER_AGGREGATION_H
#define TEST_CARRIER_AGGREGATION_H

#include "ns3/simulator.h"
#include "ns3/test.h"
#include "fcntl.h"


using namespace ns3;

/**
 * \ingroup wifi-test
 * \ingroup tests wifi-test wifi module tests
 *
 * \brief This system test program creates different test cases with a single eNB and 
 * several UEs, all having the same Radio Bearer specification. In each test 
 * case, the UEs see the same SINR from the eNB; different test cases are 
 * implemented obtained by using different SINR values and different numbers of 
 * UEs. eNb and UEs are configured to use the secondary carrier and the component
 * carrier manager is configured to split the data equally between primary and
 * secondary carrier. The test consists of checking that the throughput
 * obtained over different carriers are equal within a given tolerance.
 */
class CarrierAggregationTestCase : public TestCase
{
public:

  static bool s_writeResults; ///< write results flag, determines whether to write results to outoput files

  /**
   * Constructor of test case
   *
   * \param nUser number of users
   * \param dist the distance
   * \param dlbandwidth the DL bandwidth
   * \param ulBandwidth the UL badnwidth
   * \param numberOfComponentCarriers number of component carriers to be used in test configuration
   */
  CarrierAggregationTestCase (uint16_t nUser, uint16_t dist, uint32_t dlbandwidth, uint32_t ulBandwidth, uint32_t numberOfComponentCarriers);
  virtual ~CarrierAggregationTestCase ();
  /**
   * DL Scheduling function that is used in this test as callback function of DL scheduling trace
   * \param dlInfo the DL scheduling callback info
   */
  void DlScheduling (DlSchedulingCallbackInfo dlInfo);
  /**
   * UL Scheduling function that is used in this test as callback function of UL sceduling trace
   * \param frameNo the frame number
   * \param subframeNo the subframe number
   * \param rnti the RNTI
   * \param mcs the MCS
   * \param sizeTb
   * \param componentCarrierId the component carrier ID
   */
  void UlScheduling (uint32_t frameNo, uint32_t subframeNo, uint16_t rnti, uint8_t mcs, uint16_t sizeTb, uint8_t componentCarrierId);
  /// Write result to file function
  void WriteResultToFile ();


private:

  virtual void DoRun (void);
  /**
   * Builds the test name string based on provided parameter values
   * \param nUser number of users
   * \param dist the distance
   * \param dlBandwidth the DL bandwidth
   * \param ulBandwidth the UL badnwidth
   * \param numberOfComponentCarriers number of component carriers
   * \returns the test name
   */   
  static std::string BuildNameString  (uint16_t nUser, uint16_t dist, uint32_t dlBandwidth, uint32_t ulBandwidth, uint32_t numberOfComponentCarriers);

  uint16_t m_nUser; ///< the number of users
  uint16_t m_dist; ///< the distance
  uint16_t m_dlBandwidth; ///< DL bandwidth
  uint16_t m_ulBandwidth; ///< UL bandwidth
  uint32_t m_numberOfComponentCarriers; ///< number of component carriers

  std::map <uint8_t, uint32_t> m_ccDownlinkTraffic; ///< CC DL traffic
  std::map <uint8_t, uint32_t> m_ccUplinkTraffic; ///< CC UL traffic
  uint64_t m_dlThroughput; ///< DL throughput
  uint64_t m_ulThroughput; ///< UL throughput
  double m_statsDuration; ///< stats duration
};



/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test Carrier Aggregation Suite
 */
class TestCarrierAggregationSuite : public TestSuite
{
public:
  TestCarrierAggregationSuite ();
};

#endif /* TEST_CARRIER_AGGREGATION_H */
