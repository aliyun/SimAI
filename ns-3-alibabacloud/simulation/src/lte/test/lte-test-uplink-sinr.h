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
 * Modified by Marco Miozzo <mmiozzo@ctt.es>
 *     Extend to Data and SRS frames
 */

#ifndef LTE_TEST_UPLINK_SINR_H
#define LTE_TEST_UPLINK_SINR_H

#include "ns3/spectrum-value.h"

#include "ns3/test.h"


using namespace ns3;


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 1.2 SINR calculation in uplink
 */
class LteUplinkSinrTestSuite : public TestSuite
{
public:
  LteUplinkSinrTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test generation of SINR in the uplink. Test schedules the signal and the interference 
 * signals and it evaluates if the obtained value for SINR corresponds to the theoretical value 
 * in given conditions. 
 */
class LteUplinkDataSinrTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param sv1 spectrum value 1
   * \param sv2 spectrum value 2
   * \param sinr the SINR
   * \param name the reference name
   */
  LteUplinkDataSinrTestCase (Ptr<SpectrumValue> sv1, Ptr<SpectrumValue> sv2, Ptr<SpectrumValue> sinr, std::string name);
  virtual ~LteUplinkDataSinrTestCase ();

private:
  virtual void DoRun (void);

  Ptr<SpectrumValue> m_sv1; ///< the spectrum value #1
  Ptr<SpectrumValue> m_sv2; ///< the spectrum value #2
  Ptr<const SpectrumModel> m_sm; ///< the spectrum model
  Ptr<SpectrumValue> m_expectedSinr; ///< the expected SINR
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief  In this test one signal will be of interest, i.e., the
  *  SRS of the first signal will have the same CellId of the 
  *  receiving PHY; the others will have a different 
  *  CellId and hence will be the interfering signals. The test 
  *  checks whether the SINR of the signal correspond to the 
  *  theoretical value.
 */
class LteUplinkSrsSinrTestCase : public TestCase
{
  public:
  /**
   * Constructor
   *
   * \param sv1 spectrum value 1
   * \param sv2 spectrum value 2
   * \param sinr the SINR
   * \param name the reference name
   */
    LteUplinkSrsSinrTestCase (Ptr<SpectrumValue> sv1, Ptr<SpectrumValue> sv2, Ptr<SpectrumValue> sinr, std::string name);
    virtual ~LteUplinkSrsSinrTestCase ();
    
  /** 
   * Callback to be connected to an LteChunkProcessor to collect the reported SINR
   * 
   * \param sinr 
   */
  void ReportSinr (const SpectrumValue& sinr);

private:
  virtual void DoRun (void);
  
  Ptr<SpectrumValue> m_sv1; ///< the spectrum value #1
  Ptr<SpectrumValue> m_sv2; ///< the spectrum value #2
  Ptr<const SpectrumModel> m_sm; ///< the spectrum model

  Ptr<SpectrumValue> m_expectedSinr; ///< the expected SINR
  Ptr<SpectrumValue> m_actualSinr; ///< the actual SINR
};


#endif /* LTE_TEST_UPLINK_SINR_H */
