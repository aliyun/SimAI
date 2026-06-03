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
 */

#ifndef LTE_TEST_RLC_UM_TRANSMITTER_H
#define LTE_TEST_RLC_UM_TRANSMITTER_H

#include "ns3/test.h"

namespace ns3 {

class LteTestRrc;
class LteTestMac;
class LteTestPdcp;

}

using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief TestSuite 4.1.1 for RLC UM: Only transmitter part.
 */
class LteRlcUmTransmitterTestSuite : public TestSuite
{
  public:
    LteRlcUmTransmitterTestSuite ();
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test case used by LteRlcUmTransmitterOneSduTestCase to create topology 
 * and to implement functionalities and check if data received corresponds to 
 * data sent. 
 */
class LteRlcUmTransmitterTestCase : public TestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the test name
     */
    LteRlcUmTransmitterTestCase (std::string name);
    LteRlcUmTransmitterTestCase ();
    virtual ~LteRlcUmTransmitterTestCase ();

    /**
     * Check data received function
     * \param time the time to check
     * \param shouldReceived shoul dhave received indicator
     * \param assertMsg the assert message
     */
    void CheckDataReceived (Time time, std::string shouldReceived, std::string assertMsg);

  protected:
    virtual void DoRun (void);

    Ptr<LteTestPdcp> txPdcp; ///< the transmit PDCP
    Ptr<LteRlc> txRlc; ///< the RLC
    Ptr<LteTestMac> txMac; ///< the MAC

  private:
    /**
     * Check data received function
     * \param shouldReceived should have received indicator
     * \param assertMsg the assert message
     */
    void DoCheckDataReceived (std::string shouldReceived, std::string assertMsg);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.1 One SDU, One PDU
 */
class LteRlcUmTransmitterOneSduTestCase : public LteRlcUmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the test name
     */
    LteRlcUmTransmitterOneSduTestCase (std::string name);
    LteRlcUmTransmitterOneSduTestCase ();
    virtual ~LteRlcUmTransmitterOneSduTestCase ();

  private:
    virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.2 Segmentation (One SDU => n PDUs)
 */
class LteRlcUmTransmitterSegmentationTestCase : public LteRlcUmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcUmTransmitterSegmentationTestCase (std::string name);
    LteRlcUmTransmitterSegmentationTestCase ();
    virtual ~LteRlcUmTransmitterSegmentationTestCase ();

  private:
    virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.3 Concatenation (n SDUs => One PDU)
 */
class LteRlcUmTransmitterConcatenationTestCase : public LteRlcUmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcUmTransmitterConcatenationTestCase (std::string name);
    LteRlcUmTransmitterConcatenationTestCase ();
    virtual ~LteRlcUmTransmitterConcatenationTestCase ();

  private:
    virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.4 Report Buffer Status (test primitive parameters)
 */
class LteRlcUmTransmitterReportBufferStatusTestCase : public LteRlcUmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcUmTransmitterReportBufferStatusTestCase (std::string name);
    LteRlcUmTransmitterReportBufferStatusTestCase ();
    virtual ~LteRlcUmTransmitterReportBufferStatusTestCase ();

  private:
    virtual void DoRun (void);

};

#endif /* LTE_TEST_RLC_UM_TRANSMITTER_H */
