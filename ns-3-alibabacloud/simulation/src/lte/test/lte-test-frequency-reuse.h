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

#ifndef LTE_TEST_DOWNLINK_FR_H
#define LTE_TEST_DOWNLINK_FR_H

#include "ns3/test.h"
#include "ns3/spectrum-value.h"
#include <ns3/lte-rrc-sap.h>

#include "ns3/spectrum-test.h"
#include "ns3/lte-spectrum-value-helper.h"

using namespace ns3;

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test the fractional frequency reuse algorithms.
 */
class LteFrequencyReuseTestSuite : public TestSuite
{
public:
  LteFrequencyReuseTestSuite ();
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test frequency reuse algorithm. Test fails if the muted frequencies 
 *  are being used.
 */
class LteFrTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param userNum the number of UE nodes
   * \param dlBandwidth the DL bandwidth
   * \param ulBandwidth the UL bandwidth
   * \param availableDlRb the available DL per RB
   * \param availableUlRb the available UL per RB
   */
  LteFrTestCase (std::string name,
                 uint32_t userNum,uint16_t dlBandwidth,uint16_t ulBandwidth,
                 std::vector<bool> availableDlRb, std::vector<bool> availableUlRb);
  virtual ~LteFrTestCase ();

  /**
   * DL data receive start function
   * \param spectrumValue the DL data reeive spectrum value
   */
  void DlDataRxStart (Ptr<const SpectrumValue> spectrumValue);
  /**
   * UL data receive start function
   * \param spectrumValue the UL data reeive spectrum value
   */
  void UlDataRxStart (Ptr<const SpectrumValue> spectrumValue);

protected:
  virtual void DoRun (void);

  uint32_t m_userNum; ///< the number of UE nodes
  uint16_t m_dlBandwidth; ///< the DL bandwidth
  uint16_t m_ulBandwidth; ///< the UL bandwidth

  std::vector<bool> m_availableDlRb; ///< the available DL for each RB
  bool m_usedMutedDlRbg; ///< used muted DL RBG?

  std::vector<bool> m_availableUlRb; ///< the available UL for each RB
  bool m_usedMutedUlRbg; ///< used muted UL RBG?
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test hard frequency reuse algorithm. Test fails if the muted 
 *  frequencies are being used.
 */
class LteHardFrTestCase : public LteFrTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param userNum the number of UE nodes
   * \param schedulerType the scheduler type
   * \param dlBandwidth the DL bandwidth
   * \param ulBandwidth the UL bandwidth
   * \param dlSubBandOffset DL subband offset
   * \param dlSubBandwidth DL subbandwidth
   * \param ulSubBandOffset UL subband offset
   * \param ulSubBandwidth UL subbandwidth
   * \param availableDlRb the available DL per RB
   * \param availableUlRb the avaialbel UL per RB
   */
  LteHardFrTestCase (std::string name, uint32_t userNum,
                     std::string schedulerType,
                     uint16_t dlBandwidth, uint16_t ulBandwidth,
                     uint8_t dlSubBandOffset, uint16_t dlSubBandwidth,
                     uint8_t ulSubBandOffset, uint16_t ulSubBandwidth,
                     std::vector<bool> availableDlRb, std::vector<bool> availableUlRb);
  virtual ~LteHardFrTestCase ();

private:
  virtual void DoRun (void);

  std::string m_schedulerType; ///< the scheduler type

  uint8_t m_dlSubBandOffset; ///< the DL subband offset
  uint8_t m_dlSubBandwidth; ///< the DL subband width

  uint8_t m_ulSubBandOffset; ///< UL subband offset
  uint8_t m_ulSubBandwidth; ///< UL subband offset
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test stric frequency reuse algorithm. Test fails if the muted frequencies 
 *  are being used.
 */
class LteStrictFrTestCase : public LteFrTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param userNum the number of UE nodes
   * \param schedulerType the scheduler type
   * \param dlBandwidth the DL bandwidth
   * \param ulBandwidth the UL bandwidth
   * \param dlCommonSubBandwidth DL common bandwidth
   * \param dlEdgeSubBandOffset DL edge subband offset
   * \param dlEdgeSubBandwidth DL edge subbandwidth
   * \param ulCommonSubBandwidth UL common subbandwidth
   * \param ulEdgeSubBandOffset UL subband offset
   * \param ulEdgeSubBandwidth UL subbandwidth
   * \param availableDlRb the available DL per RB
   * \param availableUlRb the avaialbel UL per RB
   */
  LteStrictFrTestCase (std::string name, uint32_t userNum,
                       std::string schedulerType,
                       uint16_t dlBandwidth, uint16_t ulBandwidth,
                       uint16_t dlCommonSubBandwidth, uint8_t dlEdgeSubBandOffset, uint16_t dlEdgeSubBandwidth,
                       uint16_t ulCommonSubBandwidth, uint8_t ulEdgeSubBandOffset, uint16_t ulEdgeSubBandwidth,
                       std::vector<bool> availableDlRb, std::vector<bool> availableUlRb);
  virtual ~LteStrictFrTestCase ();

private:
  virtual void DoRun (void);

  std::string m_schedulerType; ///< scheduler type

  uint16_t m_dlCommonSubBandwidth; ///< DL common subbandwidth
  uint8_t m_dlEdgeSubBandOffset; ///< DL edge subband offset
  uint16_t m_dlEdgeSubBandwidth; ///< DL edge subbandwidth

  uint16_t m_ulCommonSubBandwidth; ///< UL common subbandwidth
  uint8_t m_ulEdgeSubBandOffset; ///< UL edge subband offset
  uint16_t m_ulEdgeSubBandwidth; ///< UL edge subbandwidth
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test frequency reuse algorithm by teleporing UEs to different 
 * parts of area and checking if the frequency is used according to the 
 * frequency pattern for different parts of area. Test fails if the muted 
 * frequencies for a given part of area are being used by UE.
 */
class LteFrAreaTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param schedulerType the scheduler type
   */
  LteFrAreaTestCase (std::string name, std::string schedulerType);
  virtual ~LteFrAreaTestCase ();

  /**
   * DL data receive start function
   * \param spectrumValue the DL receive specturm value
   */
  void DlDataRxStart (Ptr<const SpectrumValue> spectrumValue);
  /**
   * UL data receive start function
   * \param spectrumValue the UL receive specturm value
   */
  void UlDataRxStart (Ptr<const SpectrumValue> spectrumValue);

  /**
   * Simple teleport UE function
   * \param x the X position
   * \param y the Y position
   */
  void SimpleTeleportUe (uint32_t x, uint32_t y);
  /**
   * Teleport UE function
   * \param x the X position
   * \param y the Y position
   * \param expectedPower the expected power
   * \param expectedDlRb the expected DL pwr RB
   */
  void TeleportUe (uint32_t x, uint32_t y, double expectedPower, std::vector<bool> expectedDlRb);
  /**
   * Teleport UE 2 function
   * \param ueNode the UE node
   * \param x the X position
   * \param y the Y position
   * \param expectedPower the expected power
   * \param expectedDlRb the expected DL pwr RB
   */
  void TeleportUe2 (Ptr<Node> ueNode, uint32_t x, uint32_t y, double expectedPower,
                    std::vector<bool> expectedDlRb);

  /**
   * Set DL expected values function
   * \param expectedPower the expected power
   * \param expectedDlRb the expected DL pwr RB
   */
  void SetDlExpectedValues (double expectedPower, std::vector<bool> expectedDlRb);
  /**
   * Set UL expected values function
   * \param expectedPower the expected power
   * \param expectedDlRb the expected DL pwr RB
   */
  void SetUlExpectedValues (double expectedPower, std::vector<bool> expectedDlRb);

protected:
  virtual void DoRun (void);

  std::string m_schedulerType; ///< the scheduler type
 
  uint16_t m_dlBandwidth; ///< the DL bandwidth
  uint16_t m_ulBandwidth; ///< the UL bandwidth

  Time m_teleportTime; ///< the telport time
  Ptr<MobilityModel> m_ueMobility; ///< the UE mobility model

  double m_expectedDlPower; ///< the expected DL power
  std::vector<bool> m_expectedDlRb; ///< the expected DL per RB
  bool m_usedWrongDlRbg; ///< used wrong DL RBG?
  bool m_usedWrongDlPower; ///< used wrong DL power?

  double m_expectedUlPower; ///< expected UL power
  std::vector<bool> m_expectedUlRb; ///< expected UL per RB
  bool m_usedWrongUlRbg; ///< used wrong UL RBG?
  bool m_usedWrongUlPower; ///< used wrong UL power?

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Fr Area Test Case
 */
class LteStrictFrAreaTestCase : public LteFrAreaTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param schedulerType the scheduler type
   */
  LteStrictFrAreaTestCase (std::string name, std::string schedulerType);
  virtual ~LteStrictFrAreaTestCase ();

private:
  virtual void DoRun (void);
};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Soft Fr Area Test Case
 */
class LteSoftFrAreaTestCase : public LteFrAreaTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param schedulerType the scheduler type
   */
  LteSoftFrAreaTestCase (std::string name, std::string schedulerType);
  virtual ~LteSoftFrAreaTestCase ();

private:
  virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Soft Ffr Area Test Case
 */
class LteSoftFfrAreaTestCase : public LteFrAreaTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param schedulerType the scheduler type
   */
  LteSoftFfrAreaTestCase (std::string name, std::string schedulerType);
  virtual ~LteSoftFfrAreaTestCase ();

private:
  virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Enhanced Ffr Area Test Case
 */
class LteEnhancedFfrAreaTestCase : public LteFrAreaTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param schedulerType the scheduler type
   */
  LteEnhancedFfrAreaTestCase (std::string name, std::string schedulerType);
  virtual ~LteEnhancedFfrAreaTestCase ();

private:
  virtual void DoRun (void);

};

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Distributed Ffr Area Test Case
 */
class LteDistributedFfrAreaTestCase : public LteFrAreaTestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param schedulerType the scheduler type
   */
  LteDistributedFfrAreaTestCase (std::string name, std::string schedulerType);
  virtual ~LteDistributedFfrAreaTestCase ();

private:
  virtual void DoRun (void);

};

#endif /* LTE_TEST_DOWNLINK_FR_H */
