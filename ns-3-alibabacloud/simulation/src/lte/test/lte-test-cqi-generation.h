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
 *
 */

#ifndef LTE_TEST_CQI_GENERATION_H
#define LTE_TEST_CQI_GENERATION_H

#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Cqi Generation Test Suite
 */

class LteCqiGenerationTestSuite : public TestSuite
{
public:
  LteCqiGenerationTestSuite ();
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This is the test case for testing different configuration of CQI generation.
 *  The topology consists of the two UEs and two eNbs. UEs have the same position,
 *  while eNodeBs are at the same distance from both UEs. The checking whether CQI is 
 *  generated properly for two different cases: when PDCCH is used for the CQI 
 *  estimation and when PDSCH is used for CQI estimation.
 */

class LteCqiGenerationTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name reference name
   * \param usePdcchForCqiGeneration use PDCCH for CQI generation
   * \param dlMcs DL MCS
   * \param ulMcs UL MCS
   */
  LteCqiGenerationTestCase (std::string name, bool usePdcchForCqiGeneration,
                            uint16_t dlMcs, uint16_t ulMcs);
  virtual ~LteCqiGenerationTestCase ();

  /**
   * \brief DL Scheduling function
   * \param dlInfo DL info
   */
  void DlScheduling (DlSchedulingCallbackInfo dlInfo);

  /**
   * \brief UL Scheduling function
   * \param frameNo frame number
   * \param subframeNo subframe number
   * \param rnti the RNTI
   * \param mcs the MCS
   * \param sizeTb size
   */
  void UlScheduling (uint32_t frameNo, uint32_t subframeNo, uint16_t rnti,
                     uint8_t mcs, uint16_t sizeTb);

private:
  virtual void DoRun (void);

  bool m_usePdschForCqiGeneration; ///< use PDCCH for CQI generation
  uint16_t m_dlMcs; ///< the DL MCS
  uint16_t m_ulMcs; ///< the UL MCS

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This test is very similar to LteCqiGenerationTestCase. The difference is that in this
 * test is enabled the downlink power control.
 */

class LteCqiGenerationDlPowerControlTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name reference name
   * \param cell0Pa cell # 0 PA
   * \param cell1Pa cell # 1 PA
   * \param dlMcs DL MCS
   * \param ulMcs UL MCS
   */
  LteCqiGenerationDlPowerControlTestCase (std::string name, uint8_t cell0Pa, uint8_t cell1Pa,
                                          uint16_t dlMcs, uint16_t ulMcs);
  virtual ~LteCqiGenerationDlPowerControlTestCase ();

  /**
   * \brief DL Scheduling function
   * \param dlInfo DL info
   */
  void DlScheduling (DlSchedulingCallbackInfo dlInfo);

  /**
   * \brief UL Scheduling function
   * \param frameNo frame number
   * \param subframeNo subframe number
   * \param rnti the RNTI
   * \param mcs the MCS
   * \param sizeTb size
   */
  void UlScheduling (uint32_t frameNo, uint32_t subframeNo, uint16_t rnti,
                     uint8_t mcs, uint16_t sizeTb);

private:
  virtual void DoRun (void);

  uint8_t m_cell0Pa; ///< cell #0 PA
  uint8_t m_cell1Pa; ///< cell #1 PA

  uint16_t m_dlMcs; ///< the DL MCS
  uint16_t m_ulMcs; ///< the UL MCS

};

#endif /* LTE_TEST_CQI_GENERATION_H */
