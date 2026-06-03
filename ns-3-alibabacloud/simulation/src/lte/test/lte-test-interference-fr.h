/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on lte-test-interference.{h,cc} by Manuel Requena <manuel.requena@cttc.es>
 *                                          Nicola Baldo <nbaldo@cttc.es>
 *
 */

#ifndef LTE_TEST_INTERFERENCE_FR_H
#define LTE_TEST_INTERFERENCE_FR_H

#include "ns3/test.h"

using namespace ns3;


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for the interference test when using different 
 * frequency reuse algorithms.Check if the interfence values correspond to 
 * theoretical values.
 */
class LteInterferenceFrTestSuite : public TestSuite
{
public:
  LteInterferenceFrTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte interference test when using hard frequency reuse algorithm. Check 
 * if the interfence values correspond to theoretical values.
 */
class LteInterferenceHardFrTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param d1 distance between ENB and UE
   * \param d2 distnace between ENB and other UE
   * \param dlSinr the DL SINR
   * \param ulSinr the UL SINR
   */
  LteInterferenceHardFrTestCase (std::string name, double d1, double d2, double dlSinr, double ulSinr);
  virtual ~LteInterferenceHardFrTestCase ();

private:
  virtual void DoRun (void);

  double m_d1; ///< distance between UE and ENB
  double m_d2; ///< distance between UE and other ENB
  double m_expectedDlSinrDb; ///< expected DL SINR in dB
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte interference test when using strict frequency reuse algorithm.
 */
class LteInterferenceStrictFrTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param d1 distance between ENB and UE
   * \param d2 distnace between ENB and other UE
   * \param commonDlSinr the DL SINR
   * \param commonUlSinr the UL SINR
   * \param edgeDlSinr the DL SINR
   * \param edgeUlSinr the UL SINR
   * \param rspqThreshold RSPQ threshold
   */
  LteInterferenceStrictFrTestCase (std::string name, double d1, double d2,
                                   double commonDlSinr, double commonUlSinr, double edgeDlSinr, double edgeUlSinr,
                                   uint32_t rspqThreshold);
  virtual ~LteInterferenceStrictFrTestCase ();

private:
  virtual void DoRun (void);

  double m_d1; ///< distance between UE and ENB
  double m_d2; ///< distance between UE and other ENB
  double m_commonDlSinrDb; ///< expected common DL SINR in dB
  double m_edgeDlSinrDb; ///< expected edge DL SINR in dB

  uint32_t m_rspqThreshold; ///< RSPQ threshold
};

#endif /* LTE_TEST_INTERFERENCE_FR_H */
