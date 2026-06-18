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

#ifndef LTE_TEST_UPLINK_POWER_CONTROL_H
#define LTE_TEST_UPLINK_POWER_CONTROL_H

#include "ns3/spectrum-value.h"
#include "ns3/spectrum-test.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test 1.1 Uplink Power Control
 */
class LteUplinkPowerControlTestSuite : public TestSuite
{
public:
  LteUplinkPowerControlTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Uplink Power Control Test Case
 */
class LteUplinkPowerControlTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   */
  LteUplinkPowerControlTestCase (std::string name);
  virtual ~LteUplinkPowerControlTestCase ();

  /**
   * Teleport UE funcction
   *
   * \param x the X position
   * \param y the Y position
   * \param expectedPuschTxPower the expected PUSCH transmit power
   * \param expectedPucchTxPower the expected PUCCH transmit power
   * \param expectedSrsTxPower the expected SRS transmit power
   */
  void TeleportUe (uint32_t x, uint32_t y,
                   double expectedPuschTxPower, double expectedPucchTxPower, double expectedSrsTxPower);

  /**
   * Set TPC configuration funcction
   *
   * \param tpc the TPC
   * \param tpcNum the TPC number
   * \param expectedPuschTxPower the expected PUSCH transmit power
   * \param expectedPucchTxPower the expected PUCCH transmit power
   * \param expectedSrsTxPower the expected SRS transmit power
   */
  void SetTpcConfiguration (uint32_t tpc, uint32_t tpcNum,
                            double expectedPuschTxPower, double expectedPucchTxPower, double expectedSrsTxPower);

  /**
   * PUSCH transmit power trace funcction
   *
   * \param cellId the cell ID
   * \param rnti the RNTI
   * \param txPower the transmit power
   */
  void PuschTxPowerTrace (uint16_t cellId, uint16_t rnti, double txPower);
  /**
   * PUCCH transmit power trace funcction
   *
   * \param cellId the cell ID
   * \param rnti the RNTI
   * \param txPower the transmit power
   */
  void PucchTxPowerTrace (uint16_t cellId, uint16_t rnti, double txPower);
  /**
   * SRS transmit power trace funcction
   *
   * \param cellId the cell ID
   * \param rnti the RNTI
   * \param txPower the transmit power
   */
  void SrsTxPowerTrace (uint16_t cellId, uint16_t rnti, double txPower);

protected:
  virtual void DoRun (void);

  Ptr<MobilityModel> m_ueMobility; ///< UE mobility model
  Time m_teleportTime; ///< teleport time

  double m_expectedPuschTxPower; ///< expected PUSCH transmit power
  double m_expectedPucchTxPower; ///< expected PUCCH transmit power
  double m_expectedSrsTxPower; ///< expected SRS transmit power

  Ptr<LteUePowerControl> m_ueUpc; ///< UE power control
  bool m_accumulatedMode; ///< indicates whether accumulated mode is being used

  Ptr<LteFfrSimple> m_ffrSimple; ///< FFR simple
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Uplink Open Loop Power Control Test Case
 */
class LteUplinkOpenLoopPowerControlTestCase : public LteUplinkPowerControlTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   */
  LteUplinkOpenLoopPowerControlTestCase (std::string name);
  virtual ~LteUplinkOpenLoopPowerControlTestCase ();

private:
  virtual void DoRun (void);
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Uplink Closed Loop Power Control Absolute Mode Test Case
 */
class LteUplinkClosedLoopPowerControlAbsoluteModeTestCase : public LteUplinkPowerControlTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   */
  LteUplinkClosedLoopPowerControlAbsoluteModeTestCase (std::string name);
  virtual ~LteUplinkClosedLoopPowerControlAbsoluteModeTestCase ();

private:
  virtual void DoRun (void);
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Uplink Closed Loop Power Control Accumulated Mode Test Case
 */
class LteUplinkClosedLoopPowerControlAccumulatedModeTestCase : public LteUplinkPowerControlTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   */
  LteUplinkClosedLoopPowerControlAccumulatedModeTestCase (std::string name);
  virtual ~LteUplinkClosedLoopPowerControlAccumulatedModeTestCase ();

private:
  virtual void DoRun (void);
};

#endif /* LTE_TEST_UPLINK_POWER_CONTROL_H */
