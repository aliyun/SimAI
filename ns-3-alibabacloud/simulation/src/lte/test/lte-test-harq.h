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

#ifndef LENA_TEST_HARQ_H
#define LENA_TEST_HARQ_H

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
 * UEs. The test consists on ...
 */
class LenaHarqTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param nUser the number of UE nodes
   * \param dist the distance between n odes
   * \param tbSize
   * \param amcBer the AMC bit error rate
   * \param thrRef the throughput reference
   */
  LenaHarqTestCase (uint16_t nUser, uint16_t dist, uint16_t tbSize, double amcBer, double thrRef);
  virtual ~LenaHarqTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Build name string function
   *
   * \param nUser number of UE nodes
   * \param dist distance between nodes
   * \param tbSize
   * \returns name string
   */
  static std::string BuildNameString (uint16_t nUser, uint16_t dist, uint16_t tbSize);
  uint16_t m_nUser; ///< number of UE nodes
  uint16_t m_dist; ///< distance between nodes
  double m_amcBer; ///< AMC bit error rate
  double m_throughputRef; ///< throughput reference

};




/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for harq test.
 */

class LenaTestHarqSuite : public TestSuite
{
public:
  LenaTestHarqSuite ();
};

#endif /* LENA_TEST_HARQ_H */
