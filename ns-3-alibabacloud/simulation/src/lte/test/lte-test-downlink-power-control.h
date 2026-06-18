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

#ifndef LTE_TEST_DOWNLINK_POWER_CONTROL_H
#define LTE_TEST_DOWNLINK_POWER_CONTROL_H

#include "ns3/spectrum-value.h"
#include <ns3/lte-rrc-sap.h>

#include "ns3/spectrum-test.h"
#include "ns3/lte-spectrum-value-helper.h"

#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for the LteDownlinkPowerControlSpectrumValueTestCase.
 *
 */
class LteDownlinkPowerControlTestSuite : public TestSuite
{
public:
  LteDownlinkPowerControlTestSuite ();

/**
 * \brief Calculate RB transmit power function
 * \param txPower the transmit power
 * \param pa the PA
 * \returns RB transmit power
 */
  double CalculateRbTxPower (double txPower, uint8_t pa);
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test SINR calculation in the downlink when power control is used. 
 * Test if the difference in power levels are corresponding to the estamated values.
 */
class LteDownlinkPowerControlSpectrumValueTestCase : public TestCase
{
public:
/**
 * \brief Constructor
 *
 * \param name the reference name
 * \param earfcn the EARFCN
 * \param bw the bandwidth
 * \param powerTx 
 * \param powerTxMap
 * \param activeRbs
 * \param expected the expected Tx Power Spectral Density
 */
  LteDownlinkPowerControlSpectrumValueTestCase (std::string name,
                                                uint16_t earfcn, uint16_t bw, double powerTx,
                                                std::map<int, double> powerTxMap, std::vector <int> activeRbs,
                                                SpectrumValue& expected);
  virtual ~LteDownlinkPowerControlSpectrumValueTestCase ();

private:
  virtual void DoRun (void);
  Ptr<SpectrumValue> m_actual; ///< actual Tx Power Spectral Density
  Ptr<SpectrumValue> m_expected; ///< expected Tx Power Spectral Density

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test SINR calculation in the downlink when the power control is used.
 * Test the power control by comparing the downlink data and ctrl power 
 * difference with the estimated value based on the specified change in power.
 */
class LteDownlinkPowerControlTestCase : public TestCase
{
public:
/**
 * \brief Constructor
 *
 * \param changePower
 * \param pa
 * \param name the reference name
 */
  LteDownlinkPowerControlTestCase (bool changePower, uint8_t pa, std::string name);
  virtual ~LteDownlinkPowerControlTestCase ();

private:
  virtual void DoRun (void);

  bool m_changePdschConfigDedicated; ///< PDSCH config dedicated change
  LteRrcSap::PdschConfigDedicated m_pdschConfigDedicated; ///< PDSCH config dedicated
  double m_expectedPowerDiff; ///< expected power difference
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test if RRC connection reconfiguration messages are properly 
 * generated upon the change in the downlink power.
 * 
 */
class LteDownlinkPowerControlRrcConnectionReconfigurationTestCase : public TestCase
{
public:
  /**
   * \brief Constructor
   *
   * \param useIdealRrc if true use ideal RRC
   * \param name the reference name
   */
  LteDownlinkPowerControlRrcConnectionReconfigurationTestCase (bool useIdealRrc, std::string name);
  virtual ~LteDownlinkPowerControlRrcConnectionReconfigurationTestCase ();

  /**
   * \brief Connection Reconfiguration ENB
   *
   * \param context the context name
   * \param imsi the IMSI
   * \param cellid the cell ID
   * \param rnti the RNTI
   */
  void ConnectionReconfigurationEnb (std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti);

  /**
   * \brief Connection Reconfiguration UE
   *
   * \param context the context name
   * \param imsi the IMSI
   * \param cellid the cell ID
   * \param rnti the RNTI
   */
  void ConnectionReconfigurationUe (std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti);

  /**
   * \brief Change PDSCH config dedicated
   *
   * \param rnti the RNTI
   * \param pa the PA
   */
  void ChangePdschConfigDedicated (uint16_t rnti, uint8_t pa);
private:
  virtual void DoRun (void);
  bool m_useIdealRrc; ///< use ideal RRC?

  bool m_changePdschConfigDedicatedTriggered; ///< change PDSCH config dedicated triggered?
  bool m_connectionReconfigurationUeReceived; ///< connection reconfiguration UE received?
  bool m_connectionReconfigurationEnbCompleted; ///< connection reconfiguration ENB completed?
};

#endif /* LTE_TEST_DOWNLINK_POWER_CONTROL_H */
