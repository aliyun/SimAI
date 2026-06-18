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
 *     Extend to Data and Ctrl frames
 */

#ifndef LTE_TEST_DOWNLINK_SINR_H
#define LTE_TEST_DOWNLINK_SINR_H

#include "ns3/spectrum-value.h"

#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 1.1 SINR calculation in downlink
 */
class LteDownlinkSinrTestSuite : public TestSuite
{
public:
  LteDownlinkSinrTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief  Test compares if the generated SINR is equal to estimated SINR 
 *  value. Test generates several calls to LteSpectrumPhy::StartRx corresponding 
 *  to several signals. One will be the signal of interest, i.e., the
 *  LteSpectrumSignalParametersDlCtrlFrame of the first signal will have the
 *  same CellId of the receiving PHY; the others will have a different 
 *  CellId and hence will be the interfering signals.
 */
class LteDownlinkDataSinrTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param sv the spectrum value
   * \param sinr the SINR
   * \param name the name of the test
   */
  LteDownlinkDataSinrTestCase (Ptr<SpectrumValue> sv, Ptr<SpectrumValue> sinr, std::string name);
  virtual ~LteDownlinkDataSinrTestCase ();

private:
  virtual void DoRun (void);

  Ptr<SpectrumValue> m_sv; ///< the spectrum value
  Ptr<const SpectrumModel> m_sm; ///< the spectrum model
  Ptr<SpectrumValue> m_expectedSinr; ///< the expected SINR
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief  
 *  Test compares if the generated SINR is equal to estimated SINR value. 
 *  Test generates several calls to LteSpectrumPhy::StartRx corresponding 
 *  to several signals. One will be the signal of interest, i.e., the
 *  LteSpectrumSignalParametersDlCtrlFrame of the first signal will have the
 *  same CellId of the receiving PHY; the others will have a different 
 *  CellId and hence will be the interfering signals.
 */
class LteDownlinkCtrlSinrTestCase : public TestCase
{
  public:
  /**
   * Constructor
   *
   * \param sv the spectrum value
   * \param sinr the SINR
   * \param name the name of the test
   */
    LteDownlinkCtrlSinrTestCase (Ptr<SpectrumValue> sv, Ptr<SpectrumValue> sinr, std::string name);
    virtual ~LteDownlinkCtrlSinrTestCase ();
  
private:
  virtual void DoRun (void);
  
  Ptr<SpectrumValue> m_sv; ///< the spectrum value
  Ptr<const SpectrumModel> m_sm; ///< the spectrum model
  Ptr<SpectrumValue> m_expectedSinr; ///< the expected SINR
};

#endif /* LTE_TEST_DOWNLINK_SINR_H */
