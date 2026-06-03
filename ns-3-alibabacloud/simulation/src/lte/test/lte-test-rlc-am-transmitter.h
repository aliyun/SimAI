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

#ifndef LTE_TEST_RLC_AM_TRANSMITTER_H
#define LTE_TEST_RLC_AM_TRANSMITTER_H

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
 * \brief TestSuite 4.1.1 RLC AM: Only transmitter functionality.
 */
class LteRlcAmTransmitterTestSuite : public TestSuite
{
  public:
    LteRlcAmTransmitterTestSuite ();
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test case used by LteRlcAmTransmitterOneSduTestCase to create topology 
 * and to implement functionalities and check if data received corresponds to 
 * data sent. 
 */
class LteRlcAmTransmitterTestCase : public TestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcAmTransmitterTestCase (std::string name);
    LteRlcAmTransmitterTestCase ();
    virtual ~LteRlcAmTransmitterTestCase ();

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
     * \param shouldReceived shoul dhave received indicator
     * \param assertMsg the assert message
     */
    void DoCheckDataReceived (std::string shouldReceived, std::string assertMsg);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.1 Test that SDU transmitted at PDCP corresponds to PDU 
 * received by MAC.
 */
class LteRlcAmTransmitterOneSduTestCase : public LteRlcAmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcAmTransmitterOneSduTestCase (std::string name);
    LteRlcAmTransmitterOneSduTestCase ();
    virtual ~LteRlcAmTransmitterOneSduTestCase ();

  private:
    virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.2 Test the correct functionality of the Segmentation. 
 * Test check that single SDU is properly segmented to n PDUs.
 */
class LteRlcAmTransmitterSegmentationTestCase : public LteRlcAmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcAmTransmitterSegmentationTestCase (std::string name);
    LteRlcAmTransmitterSegmentationTestCase ();
    virtual ~LteRlcAmTransmitterSegmentationTestCase ();

  private:
    virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.3 Test that concatenation functionality works properly.
 * Test check if n SDUs are correctly contactenate to single PDU.
 */
class LteRlcAmTransmitterConcatenationTestCase : public LteRlcAmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcAmTransmitterConcatenationTestCase (std::string name);
    LteRlcAmTransmitterConcatenationTestCase ();
    virtual ~LteRlcAmTransmitterConcatenationTestCase ();

  private:
    virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 4.1.1.4 Test checks functionality of Report Buffer Status by 
 * testing primitive parameters.
 */
class LteRlcAmTransmitterReportBufferStatusTestCase : public LteRlcAmTransmitterTestCase
{
  public:
    /**
     * Constructor
     *
     * \param name the reference name
     */
    LteRlcAmTransmitterReportBufferStatusTestCase (std::string name);
    LteRlcAmTransmitterReportBufferStatusTestCase ();
    virtual ~LteRlcAmTransmitterReportBufferStatusTestCase ();

  private:
    virtual void DoRun (void);

};

#endif // LTE_TEST_RLC_AM_TRANSMITTER_H
