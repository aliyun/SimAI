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

#include <ns3/simulator.h>
#include <ns3/log.h>
#include <ns3/callback.h>
#include <ns3/config.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include <ns3/enum.h>
#include <ns3/boolean.h>
#include <ns3/pointer.h>

#include "ns3/mobility-helper.h"
#include "ns3/lte-helper.h"

#include <ns3/ff-mac-scheduler.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/lte-enb-rrc.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-ue-phy.h>
#include <ns3/lte-ue-rrc.h>

#include "lte-ffr-simple.h"
#include <ns3/lte-common.h>


#include <ns3/lte-chunk-processor.h>

#include "lte-test-downlink-power-control.h"
#include <ns3/lte-rrc-sap.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteDownlinkPowerControlTest");

/**
 * TestSuite
 */

LteDownlinkPowerControlTestSuite::LteDownlinkPowerControlTestSuite ()
  : TestSuite ("lte-downlink-power-control", SYSTEM)
{
//  LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_DEBUG);
//  LogComponentEnable ("LteDownlinkPowerControlTest", logLevel);
  NS_LOG_INFO ("Creating LteDownlinkPowerControlTestSuite");

  // Spectrum helper test
  {
    std::vector<int> activeRbs_txpowdB30nrb6run1earfcn500 (4);
    activeRbs_txpowdB30nrb6run1earfcn500[0] = 0;
    activeRbs_txpowdB30nrb6run1earfcn500[1] = 2;
    activeRbs_txpowdB30nrb6run1earfcn500[2] = 3;
    activeRbs_txpowdB30nrb6run1earfcn500[3] = 5;

    uint16_t earfcn = 500;
    uint8_t bw = 6;
    double powerTx = 30;

    std::map<int, double> powerTxMap;
    double rbTxPower = powerTx;

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_6);
    powerTxMap.insert (std::pair<int,double>(0, rbTxPower));

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(2, rbTxPower));

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB0);
    powerTxMap.insert (std::pair<int,double>(3, rbTxPower));

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(5, rbTxPower));

    SpectrumValue spectrumValue_txpowdB30nrb6run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
    spectrumValue_txpowdB30nrb6run1earfcn500[0] = 2.325820769916e-07;
    spectrumValue_txpowdB30nrb6run1earfcn500[1] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb6run1earfcn500[2] = 4.640622533585e-07;
    spectrumValue_txpowdB30nrb6run1earfcn500[3] = 9.259259259259e-07;
    spectrumValue_txpowdB30nrb6run1earfcn500[4] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb6run1earfcn500[5] = 1.467493696723e-06;

    AddTestCase (new LteDownlinkPowerControlSpectrumValueTestCase ("txpowdB30nrb6run1earfcn500_allRbPower",
                                                                   earfcn, bw, powerTx,
                                                                   powerTxMap, activeRbs_txpowdB30nrb6run1earfcn500,
                                                                   spectrumValue_txpowdB30nrb6run1earfcn500), TestCase::QUICK);
  }
  {
    //if power for RB not present, but RB is active, use powerTx
    std::vector<int> activeRbs_txpowdB30nrb6run1earfcn500 (4);
    activeRbs_txpowdB30nrb6run1earfcn500[0] = 0;
    activeRbs_txpowdB30nrb6run1earfcn500[1] = 2;
    activeRbs_txpowdB30nrb6run1earfcn500[2] = 3;
    activeRbs_txpowdB30nrb6run1earfcn500[3] = 5;

    uint16_t earfcn = 500;
    uint8_t bw = 6;
    double powerTx = 30;

    std::map<int, double> powerTxMap;
    double rbTxPower = powerTx;

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(2, rbTxPower));

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB0);
    powerTxMap.insert (std::pair<int,double>(3, rbTxPower));

    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(5, rbTxPower));

    SpectrumValue spectrumValue_txpowdB30nrb6run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
    spectrumValue_txpowdB30nrb6run1earfcn500[0] = 9.259259259259e-07;
    spectrumValue_txpowdB30nrb6run1earfcn500[1] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb6run1earfcn500[2] = 4.640622533585e-07;
    spectrumValue_txpowdB30nrb6run1earfcn500[3] = 9.259259259259e-07;
    spectrumValue_txpowdB30nrb6run1earfcn500[4] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb6run1earfcn500[5] = 1.467493696723e-06;

    AddTestCase (new LteDownlinkPowerControlSpectrumValueTestCase ("txpowdB30nrb6run1earfcn500",
                                                                   earfcn, bw, powerTx,
                                                                   powerTxMap, activeRbs_txpowdB30nrb6run1earfcn500,
                                                                   spectrumValue_txpowdB30nrb6run1earfcn500), TestCase::QUICK);
  }
  {
    std::vector<int> activeRbs_txpowdB30nrb25run1earfcn500 (15);
    activeRbs_txpowdB30nrb25run1earfcn500[0] = 0;
    activeRbs_txpowdB30nrb25run1earfcn500[1] = 1;
    activeRbs_txpowdB30nrb25run1earfcn500[2] = 2;
    activeRbs_txpowdB30nrb25run1earfcn500[3] = 3;
    activeRbs_txpowdB30nrb25run1earfcn500[4] = 7;
    activeRbs_txpowdB30nrb25run1earfcn500[5] = 8;
    activeRbs_txpowdB30nrb25run1earfcn500[6] = 9;
    activeRbs_txpowdB30nrb25run1earfcn500[7] = 11;
    activeRbs_txpowdB30nrb25run1earfcn500[8] = 13;
    activeRbs_txpowdB30nrb25run1earfcn500[9] = 14;
    activeRbs_txpowdB30nrb25run1earfcn500[10] = 18;
    activeRbs_txpowdB30nrb25run1earfcn500[11] = 19;
    activeRbs_txpowdB30nrb25run1earfcn500[12] = 20;
    activeRbs_txpowdB30nrb25run1earfcn500[13] = 21;
    activeRbs_txpowdB30nrb25run1earfcn500[14] = 24;

    uint16_t earfcn = 500;
    uint8_t bw = 25;
    double powerTx = 30;

    std::map<int, double> powerTxMap;
    double rbTxPower = powerTx;
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(0, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_4dot77);
    powerTxMap.insert (std::pair<int,double>(1, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_6);
    powerTxMap.insert (std::pair<int,double>(2, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB1);
    powerTxMap.insert (std::pair<int,double>(3, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(7, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(8, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(9, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_1dot77);
    powerTxMap.insert (std::pair<int,double>(11, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(13, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(14, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB0);
    powerTxMap.insert (std::pair<int,double>(18, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(19, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(20, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_4dot77);
    powerTxMap.insert (std::pair<int,double>(21, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB3);
    powerTxMap.insert (std::pair<int,double>(24, rbTxPower));


    SpectrumValue spectrumValue_txpowdB30nrb25run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, bw));
    spectrumValue_txpowdB30nrb25run1earfcn500[0] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[1] = 7.409475839182e-08;
    spectrumValue_txpowdB30nrb25run1earfcn500[2] = 5.581969847799e-08;
    spectrumValue_txpowdB30nrb25run1earfcn500[3] = 2.797612026209e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[4] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[5] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[6] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[7] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[8] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[9] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[10] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[11] = 1.478384791559e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[12] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[13] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[14] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[15] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[16] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[17] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[18] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[19] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[20] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[21] = 7.409475839182e-08;
    spectrumValue_txpowdB30nrb25run1earfcn500[22] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[23] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[24] = 4.433916255486e-07;

    AddTestCase (new LteDownlinkPowerControlSpectrumValueTestCase ("txpowdB30nrb25run1earfcn500_allRbPower",
                                                                   earfcn, bw, powerTx,
                                                                   powerTxMap, activeRbs_txpowdB30nrb25run1earfcn500,
                                                                   spectrumValue_txpowdB30nrb25run1earfcn500), TestCase::QUICK);
  }
  {
    //if power for RB not present, but RB is active, use powerTx
    std::vector<int> activeRbs_txpowdB30nrb25run1earfcn500 (15);
    activeRbs_txpowdB30nrb25run1earfcn500[0] = 0;
    activeRbs_txpowdB30nrb25run1earfcn500[1] = 1;
    activeRbs_txpowdB30nrb25run1earfcn500[2] = 2;
    activeRbs_txpowdB30nrb25run1earfcn500[3] = 3;
    activeRbs_txpowdB30nrb25run1earfcn500[4] = 7;
    activeRbs_txpowdB30nrb25run1earfcn500[5] = 8;
    activeRbs_txpowdB30nrb25run1earfcn500[6] = 9;
    activeRbs_txpowdB30nrb25run1earfcn500[7] = 11;
    activeRbs_txpowdB30nrb25run1earfcn500[8] = 13;
    activeRbs_txpowdB30nrb25run1earfcn500[9] = 14;
    activeRbs_txpowdB30nrb25run1earfcn500[10] = 18;
    activeRbs_txpowdB30nrb25run1earfcn500[11] = 19;
    activeRbs_txpowdB30nrb25run1earfcn500[12] = 20;
    activeRbs_txpowdB30nrb25run1earfcn500[13] = 21;
    activeRbs_txpowdB30nrb25run1earfcn500[14] = 24;

    uint16_t earfcn = 500;
    uint8_t bw = 25;
    double powerTx = 30;

    std::map<int, double> powerTxMap;
    double rbTxPower = powerTx;
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(0, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_6);
    powerTxMap.insert (std::pair<int,double>(2, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB1);
    powerTxMap.insert (std::pair<int,double>(3, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(7, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(8, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB2);
    powerTxMap.insert (std::pair<int,double>(13, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(14, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB0);
    powerTxMap.insert (std::pair<int,double>(18, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB_3);
    powerTxMap.insert (std::pair<int,double>(19, rbTxPower));
    rbTxPower = CalculateRbTxPower (powerTx, LteRrcSap::PdschConfigDedicated::dB3);
    powerTxMap.insert (std::pair<int,double>(24, rbTxPower));


    SpectrumValue spectrumValue_txpowdB30nrb25run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, bw));
    spectrumValue_txpowdB30nrb25run1earfcn500[0] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[1] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[2] = 5.581969847799e-08;
    spectrumValue_txpowdB30nrb25run1earfcn500[3] = 2.797612026209e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[4] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[5] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[6] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[7] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[8] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[9] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[10] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[11] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[12] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[13] = 3.521984872135e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[14] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[15] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[16] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[17] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[18] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[19] = 1.113749408060e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[20] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[21] = 2.222222222222e-07;
    spectrumValue_txpowdB30nrb25run1earfcn500[22] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[23] = 0.000000000000e+00;
    spectrumValue_txpowdB30nrb25run1earfcn500[24] = 4.433916255486e-07;

    AddTestCase (new LteDownlinkPowerControlSpectrumValueTestCase ("txpowdB30nrb25run1earfcn500",
                                                                   earfcn, bw, powerTx,
                                                                   powerTxMap, activeRbs_txpowdB30nrb25run1earfcn500,
                                                                   spectrumValue_txpowdB30nrb25run1earfcn500), TestCase::QUICK);
  }


  // Downlink DATA and CONTROL channels power comparison
  AddTestCase (new LteDownlinkPowerControlTestCase (false, LteRrcSap::PdschConfigDedicated::dB0,
                                                    "DataCtrlPowerDifference_noChange"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB_6,
                                                    "DataCtrlPowerDifference_dB_6"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB_4dot77,
                                                    "DataCtrlPowerDifference_dB_4dot77"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB_3,
                                                    "DataCtrlPowerDifference_dB_3"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB_1dot77,
                                                    "DataCtrlPowerDifference_dB_1dot77"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB0,
                                                    "DataCtrlPowerDifference_dB0"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB1,
                                                    "DataCtrlPowerDifference_dB1"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB2,
                                                    "DataCtrlPowerDifference_dB2"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlTestCase (true, LteRrcSap::PdschConfigDedicated::dB3,
                                                    "DataCtrlPowerDifference_dB3"), TestCase::QUICK);


  // RrcConnectionReconfiguration test
  AddTestCase (new LteDownlinkPowerControlRrcConnectionReconfigurationTestCase (false, "RrcConnReconf-IdealRrc"), TestCase::QUICK);
  AddTestCase (new LteDownlinkPowerControlRrcConnectionReconfigurationTestCase (true, "RrcConnReconf-RealRrc"), TestCase::QUICK);


}

double
LteDownlinkPowerControlTestSuite::CalculateRbTxPower (double txPower, uint8_t pa)
{
  LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
  pdschConfigDedicated.pa = pa;
  double rbTxpower = txPower + LteRrcSap::ConvertPdschConfigDedicated2Double (pdschConfigDedicated);

  return rbTxpower;
}

static LteDownlinkPowerControlTestSuite lteDownlinkPowerControlTestSuite;


/**
 * TestCase Data
 */

LteDownlinkPowerControlSpectrumValueTestCase::LteDownlinkPowerControlSpectrumValueTestCase (std::string name,
                                                                                            uint16_t earfcn, uint16_t bw, double powerTx,
                                                                                            std::map<int, double> powerTxMap, std::vector <int> activeRbs,
                                                                                            SpectrumValue& expected)
  : TestCase ("Downlink Power Control: " + name),
    m_actual (LteSpectrumValueHelper::CreateTxPowerSpectralDensity (earfcn, bw, powerTx, powerTxMap, activeRbs)),
    m_expected (Create<SpectrumValue> (expected))
{
  NS_LOG_INFO ("Creating LteDownlinkPowerControlTestCase");
}

LteDownlinkPowerControlSpectrumValueTestCase::~LteDownlinkPowerControlSpectrumValueTestCase ()
{
}

void
LteDownlinkPowerControlSpectrumValueTestCase::DoRun (void)
{
  NS_LOG_INFO ("Creating LteDownlinkPowerControlSpectrumValueTestCase");
  NS_TEST_ASSERT_MSG_EQ (m_actual->GetSpectrumModelUid (), m_expected->GetSpectrumModelUid (), "SpectrumModel UID mismatch");
  NS_TEST_ASSERT_MSG_SPECTRUM_VALUE_EQ_TOL ((*m_actual), (*m_expected), 0.0000001, "SpectrumValues not equal");

}

LteDownlinkPowerControlTestCase::LteDownlinkPowerControlTestCase (bool changePower, uint8_t pa, std::string name)
  : TestCase ("Downlink Power Control: " + name)
{
  m_changePdschConfigDedicated = changePower;
  m_pdschConfigDedicated.pa = pa;
  m_expectedPowerDiff = LteRrcSap::ConvertPdschConfigDedicated2Double (m_pdschConfigDedicated);
  NS_LOG_INFO ("Creating LteDownlinkPowerControlTestCase");
}

LteDownlinkPowerControlTestCase::~LteDownlinkPowerControlTestCase ()
{
}

void
LteDownlinkPowerControlTestCase::DoRun (void)
{
  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (false));

  /**
    * Simulation Topology
    */

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetFfrAlgorithmType ("ns3::LteFfrSimple");

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes );

  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (allNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  PointerValue tmp;
  enbDevs.Get (0)->GetAttribute ("LteFfrAlgorithm", tmp);
  Ptr<LteFfrSimple> simpleFfrAlgorithm = DynamicCast<LteFfrSimple>(tmp.GetObject ());
  simpleFfrAlgorithm->ChangePdschConfigDedicated (m_changePdschConfigDedicated);
  simpleFfrAlgorithm->SetPdschConfigDedicated (m_pdschConfigDedicated);

  // Activate the default EPS bearer
  enum EpsBearer::Qci q = EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  // Use testing chunk processor in the PHY layer
  // It will be used to calculate data and ctrl power in downlink

  Ptr<LtePhy> ue1Phy = ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();

  Ptr<LteChunkProcessor> testDlDataPowerReceived = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlDataPowerReceivedCatcher;
  testDlDataPowerReceived->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlDataPowerReceivedCatcher));
  ue1Phy->GetDownlinkSpectrumPhy ()->AddDataPowerChunkProcessor (testDlDataPowerReceived);

  Ptr<LteChunkProcessor> testDlCtrlPowerReceived = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlCtrlPowerReceivedCatcher;
  testDlCtrlPowerReceived->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlCtrlPowerReceivedCatcher));
  ue1Phy->GetDownlinkSpectrumPhy ()->AddRsPowerChunkProcessor (testDlCtrlPowerReceived);

  Simulator::Stop (Seconds (0.400));
  Simulator::Run ();

  double dataPower = 10.0 * std::log10 (dlDataPowerReceivedCatcher.GetValue ()->operator[] (0));
  double ctrlPower = 10.0 * std::log10 (dlCtrlPowerReceivedCatcher.GetValue ()->operator[] (0));
  double powerDiff = (-1.0)*ctrlPower + dataPower;

  NS_LOG_DEBUG ("DataPower: " << dataPower);
  NS_LOG_DEBUG ("CtrlPower: " << ctrlPower);
  NS_LOG_DEBUG ("PowerDifference: " << powerDiff);
  NS_LOG_DEBUG ("ExpectedPowerDifference: " << m_expectedPowerDiff);

  NS_TEST_ASSERT_MSG_EQ_TOL (powerDiff, m_expectedPowerDiff, 0.001,
                             "Downlink Data and Ctrl Power Difference are not equal within tolerance");

  Simulator::Destroy ();
}

void
NotifyConnectionReconfigurationUe (LteDownlinkPowerControlRrcConnectionReconfigurationTestCase *testcase,
                                   std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  testcase->ConnectionReconfigurationUe (context, imsi, cellId, rnti);
}

void
NotifyConnectionReconfigurationEnb (LteDownlinkPowerControlRrcConnectionReconfigurationTestCase *testcase,
                                    std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  testcase->ConnectionReconfigurationEnb (context, imsi, cellId, rnti);
}

void
ChangePdschConfigDedicatedCallback (LteDownlinkPowerControlRrcConnectionReconfigurationTestCase *testcase,
                                    std::string context, uint16_t rnti, uint8_t pa)
{
  testcase->ChangePdschConfigDedicated (rnti, pa);
}


LteDownlinkPowerControlRrcConnectionReconfigurationTestCase::LteDownlinkPowerControlRrcConnectionReconfigurationTestCase (bool useIdealRrc, std::string name)
  : TestCase ("Downlink Power Control: " + name),
    m_useIdealRrc (useIdealRrc),
    m_changePdschConfigDedicatedTriggered (false),
    m_connectionReconfigurationUeReceived (false),
    m_connectionReconfigurationEnbCompleted (false)
{

}

LteDownlinkPowerControlRrcConnectionReconfigurationTestCase::~LteDownlinkPowerControlRrcConnectionReconfigurationTestCase ()
{
}

void
LteDownlinkPowerControlRrcConnectionReconfigurationTestCase::ConnectionReconfigurationEnb (std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
  if (Simulator::Now () > MilliSeconds (100))
    {
      NS_LOG_DEBUG ("eNb ConnectionReconfigurationCompleted");
      m_connectionReconfigurationEnbCompleted = true;
    }
}

void
LteDownlinkPowerControlRrcConnectionReconfigurationTestCase::ConnectionReconfigurationUe (std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
  if (Simulator::Now () > MilliSeconds (100))
    {
      NS_LOG_DEBUG ("Ue ConnectionReconfiguration");
      m_connectionReconfigurationUeReceived = true;
    }
}

void
LteDownlinkPowerControlRrcConnectionReconfigurationTestCase::ChangePdschConfigDedicated (uint16_t rnti, uint8_t pa)
{
  NS_LOG_DEBUG ("FFR Algorithm ChangePdschConfigDedicated");
  m_changePdschConfigDedicatedTriggered = true;
}

void
LteDownlinkPowerControlRrcConnectionReconfigurationTestCase::DoRun (void)
{
  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (m_useIdealRrc));

  /**
  * Simulation Topology
  */

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetFfrAlgorithmType ("ns3::LteFfrSimple");

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes );

  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (allNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  PointerValue tmp;
  enbDevs.Get (0)->GetAttribute ("LteFfrAlgorithm", tmp);
  Ptr<LteFfrSimple> simpleFfrAlgorithm = DynamicCast<LteFfrSimple>(tmp.GetObject ());
  LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
  pdschConfigDedicated.pa = LteRrcSap::PdschConfigDedicated::dB_6;
  simpleFfrAlgorithm->ChangePdschConfigDedicated (true);
  simpleFfrAlgorithm->SetPdschConfigDedicated (pdschConfigDedicated);

  // Activate the default EPS bearer
  enum EpsBearer::Qci q = EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  Config::Connect ("/NodeList/0/DeviceList/0/LteFfrAlgorithm/ChangePdschConfigDedicated",
                   MakeBoundCallback (&ChangePdschConfigDedicatedCallback, this));
  Config::Connect ("/NodeList/1/DeviceList/0/LteUeRrc/ConnectionReconfiguration",
                   MakeBoundCallback (&NotifyConnectionReconfigurationUe, this));
  Config::Connect ("/NodeList/0/DeviceList/0/LteEnbRrc/ConnectionReconfiguration",
                   MakeBoundCallback (&NotifyConnectionReconfigurationEnb, this));

  Simulator::Stop (Seconds (0.400));
  Simulator::Run ();

  NS_TEST_ASSERT_MSG_EQ (m_changePdschConfigDedicatedTriggered, true,
                         "Ffr have not changed PdschConfigDedicated for UE");
  NS_TEST_ASSERT_MSG_EQ (m_connectionReconfigurationUeReceived, true,
                         "Ue have not received RrcConnectionReconfiguration");
  NS_TEST_ASSERT_MSG_EQ (m_connectionReconfigurationEnbCompleted, true,
                         "Enb have not received RrcConnectionReconfigurationCompleted");

  Simulator::Destroy ();
}

