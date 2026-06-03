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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
 */

#ifndef LENA_TEST_PHY_ERROR_MODEL_H
#define LENA_TEST_PHY_ERROR_MODEL_H

#include <ns3/simulator.h>
#include <ns3/test.h>
#include <ns3/nstime.h>


using namespace ns3;


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This system test program creates different test cases with a single eNB and 
 * several UEs, all having the same Radio Bearer specification. In each test 
 * case, the UEs see the same SINR from the eNB; different test cases are 
 * implemented obtained by using different SINR values and different numbers of 
 * UEs. The test consists on ...
 */
class LenaDataPhyErrorModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nUser the number of UE nodes
   * \param dist the distance between nodes
   * \param blerRef expected BLER
   * \param toleranceRxPackets receive packet loss tolerance
   * \param statsStartTime the start time for collecting statistics
   * \param rngRun rng run
   */
  LenaDataPhyErrorModelTestCase (uint16_t nUser, uint16_t dist,
                                 double blerRef, uint16_t toleranceRxPackets,
                                 Time statsStartTime, uint32_t rngRun);
  virtual ~LenaDataPhyErrorModelTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Builds the test name string based on provided parameter values
   * \param nUser the number of UE nodes
   * \param dist the distance between nodes
   * \param rngRun the rng run
   * \returns the name string
   */
  static std::string BuildNameString (uint16_t nUser, uint16_t dist, uint32_t rngRun);
  uint16_t m_nUser; ///< number of UE nodes
  double m_dist; ///< the distance between nodes
  double m_blerRef; ///< the expected BLER
  uint16_t m_toleranceRxPackets; ///< receive packet tolerance loss
  Time m_statsStartTime; ///< Extra time in the beginning of simulation to allow RRC connection establishment + SRS
  uint32_t m_rngRun; ///< the rng run

};



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lena Dl Ctrl Phy Error Model Test Case
 */
class LenaDlCtrlPhyErrorModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nEnb the number of ENB nodes
   * \param dist the distance between nodes
   * \param blerRef expected BLER
   * \param toleranceRxPackets receive packet loss tolerance
   * \param statsStartTime the start time for collecting statistics
   * \param rngRun rng number
   */
  LenaDlCtrlPhyErrorModelTestCase (uint16_t nEnb, uint16_t dist,
                                   double blerRef, uint16_t toleranceRxPackets,
                                   Time statsStartTime, uint32_t rngRun);
  virtual ~LenaDlCtrlPhyErrorModelTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Build name string
   * \param nUser the number of UE nodes
   * \param dist the distnace between nodes
   * \param rngRun the rng run
   * \returns the name string
   */
  static std::string BuildNameString (uint16_t nUser, uint16_t dist, uint32_t rngRun);
  uint16_t m_nEnb; ///< the number of ENB nodes
  double m_dist; ///< the distance between nodes
  double m_blerRef; ///< the expected BLER
  uint16_t m_toleranceRxPackets; ///< receive packet tolerance loss
  Time m_statsStartTime; ///< Extra time in the beginning of simulation to allow RRC connection establishment + SRS
  uint32_t m_rngRun; ///< the rng run number

};



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lena Test Phy Error Model Suite
 */
class LenaTestPhyErrorModelSuite : public TestSuite
{
public:
  LenaTestPhyErrorModelSuite ();
};

#endif /* LENA_TEST_PHY_ERROR_MODEL_H */
