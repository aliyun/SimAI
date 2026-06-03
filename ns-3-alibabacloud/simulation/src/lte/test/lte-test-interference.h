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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef LTE_TEST_INTERFERENCE_H
#define LTE_TEST_INTERFERENCE_H

#include "ns3/test.h"


using namespace ns3;


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for interference test.
 */
class LteInterferenceTestSuite : public TestSuite
{
public:
  LteInterferenceTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test that SINR calculation and MCS selection works fine in a 
 * multi-cell interference scenario.
 */
class LteInterferenceTestCase : public TestCase
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
   * \param dlSe the DL se
   * \param ulSe the UL se
   * \param dlMcs the DL MCS
   * \param ulMcs the UL MCS
   */
  LteInterferenceTestCase (std::string name, double d1, double d2, double dlSinr, double ulSinr, double dlSe, double ulSe, uint16_t dlMcs, uint16_t ulMcs);
  virtual ~LteInterferenceTestCase ();

  /**
   * DL scheduling function
   * \param dlInfo the DL info
   */
  void DlScheduling (DlSchedulingCallbackInfo dlInfo);

  /**
   * UL scheduling function
   * \param frameNo the frame number
   * \param subframeNo the subframe number
   * \param rnti the RNTI
   * \param mcs the MCS
   * \param sizeTb
   */
  void UlScheduling (uint32_t frameNo, uint32_t subframeNo, uint16_t rnti,
                     uint8_t mcs, uint16_t sizeTb);

private:
  virtual void DoRun (void);


  double m_d1; ///< distance between UE and ENB
  double m_d2; ///< distance between UE and other ENB
  double m_expectedDlSinrDb; ///< expected DL SINR in dB
  double m_expectedUlSinrDb; ///< expected UL SINR in dB
  uint16_t m_dlMcs; ///< the DL MCS
  uint16_t m_ulMcs; ///< the UL MCS
};

#endif /* LTE_TEST_INTERFERENCE_H */
