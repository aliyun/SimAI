/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Lluis Parcerisa <lparcerisa@cttc.cat>
 */

#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/boolean.h"
#include "ns3/test.h"
#include "ns3/ptr.h"
#include "ns3/packet.h"

#include "ns3/lte-rrc-header.h"
#include "ns3/lte-rrc-sap.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Asn1EncodingTest");

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Contains ASN encoding test utility functions.
 */
class TestUtils
{
public:
  /**
   * Function to convert packet contents in hex format
   * \param pkt the packet
   * \returns the text string
   */
  static std::string sprintPacketContentsHex (Ptr<Packet> pkt)
  {
    uint32_t psize = pkt->GetSize ();
    uint8_t buffer[psize];
    char sbuffer[psize * 3];
    pkt->CopyData (buffer, psize);
    for (uint32_t i = 0; i < psize; i++)
      {
        sprintf (&sbuffer[i * 3],"%02x ",buffer[i]);
      }
    return std::string (sbuffer);
  }

  /**
   * Function to convert packet contents in binary format
   * \param pkt the packet
   * \returns the text string
   */
  static std::string sprintPacketContentsBin (Ptr<Packet> pkt)
  {
    uint32_t psize = pkt->GetSize ();
    uint8_t buffer[psize];
    std::ostringstream oss (std::ostringstream::out);
    pkt->CopyData (buffer, psize);
    for (uint32_t i = 0; i < psize; i++)
      {
        oss << (std::bitset<8> (buffer[i]));
      }
    return std::string (oss.str () + "\n");
  }

  /**
   * Function to log packet contents
   * \param pkt the packet
   */
  static void LogPacketContents (Ptr<Packet> pkt)
  {
    NS_LOG_DEBUG ("---- SERIALIZED PACKET CONTENTS (HEX): -------");
    NS_LOG_DEBUG ("Hex: " << TestUtils::sprintPacketContentsHex (pkt));
    NS_LOG_DEBUG ("Bin: " << TestUtils::sprintPacketContentsBin (pkt));
  }

  /**
   * Function to log packet info
   * \param source T
   * \param s the string
   */
  template <class T>
  static void LogPacketInfo (T source,std::string s)
  {
    NS_LOG_DEBUG ("--------- " << s.data () << " INFO: -------");
    std::ostringstream oss (std::ostringstream::out);
    source.Print (oss);
    NS_LOG_DEBUG (oss.str ());
  }
};

// --------------------------- CLASS RrcHeaderTestCase -----------------------------
/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This class provides common functions to be inherited
 * by the children TestCases
 */
class RrcHeaderTestCase : public TestCase
{
public:
  /**
   * Constructor
   * \param s the reference name
   */
  RrcHeaderTestCase (std::string s);
  virtual void DoRun (void) = 0;
  /**
   * \brief Create radio resource config dedicated
   * \returns LteRrcSap::RadioResourceConfigDedicated
   */
  LteRrcSap::RadioResourceConfigDedicated CreateRadioResourceConfigDedicated ();
  /**
   * \brief Assert equal radio resource config dedicated
   * \param rrcd1 LteRrcSap::RadioResourceConfigDedicated # 1
   * \param rrcd2 LteRrcSap::RadioResourceConfigDedicated # 2
   */
  void AssertEqualRadioResourceConfigDedicated (LteRrcSap::RadioResourceConfigDedicated rrcd1, LteRrcSap::RadioResourceConfigDedicated rrcd2);

protected:
  Ptr<Packet> packet; ///< the packet
};

RrcHeaderTestCase::RrcHeaderTestCase (std::string s) : TestCase (s)
{
}

LteRrcSap::RadioResourceConfigDedicated
RrcHeaderTestCase::CreateRadioResourceConfigDedicated ()
{
  LteRrcSap::RadioResourceConfigDedicated rrd;

  rrd.drbToReleaseList = std::list<uint8_t> (4,2);

  LteRrcSap::SrbToAddMod srbToAddMod;
  srbToAddMod.srbIdentity = 2;

  LteRrcSap::LogicalChannelConfig logicalChannelConfig;
  logicalChannelConfig.priority = 9;
  logicalChannelConfig.prioritizedBitRateKbps = 128;
  logicalChannelConfig.bucketSizeDurationMs = 100;
  logicalChannelConfig.logicalChannelGroup = 3;
  srbToAddMod.logicalChannelConfig = logicalChannelConfig;

  rrd.srbToAddModList.insert (rrd.srbToAddModList.begin (),srbToAddMod);

  LteRrcSap::DrbToAddMod drbToAddMod;
  drbToAddMod.epsBearerIdentity = 1;
  drbToAddMod.drbIdentity = 1;
  drbToAddMod.logicalChannelIdentity = 5;
  LteRrcSap::RlcConfig rlcConfig;
  rlcConfig.choice = LteRrcSap::RlcConfig::UM_BI_DIRECTIONAL;
  drbToAddMod.rlcConfig = rlcConfig;

  LteRrcSap::LogicalChannelConfig logicalChannelConfig2;
  logicalChannelConfig2.priority = 7;
  logicalChannelConfig2.prioritizedBitRateKbps = 256;
  logicalChannelConfig2.bucketSizeDurationMs = 50;
  logicalChannelConfig2.logicalChannelGroup = 2;
  drbToAddMod.logicalChannelConfig = logicalChannelConfig2;

  rrd.drbToAddModList.insert (rrd.drbToAddModList.begin (),drbToAddMod);

  rrd.havePhysicalConfigDedicated = true;
  LteRrcSap::PhysicalConfigDedicated physicalConfigDedicated;
  physicalConfigDedicated.haveSoundingRsUlConfigDedicated = true;
  physicalConfigDedicated.soundingRsUlConfigDedicated.type = LteRrcSap::SoundingRsUlConfigDedicated::SETUP;
  physicalConfigDedicated.soundingRsUlConfigDedicated.srsBandwidth = 2;
  physicalConfigDedicated.soundingRsUlConfigDedicated.srsConfigIndex = 12;

  physicalConfigDedicated.haveAntennaInfoDedicated = true;
  physicalConfigDedicated.antennaInfo.transmissionMode = 2;

  physicalConfigDedicated.havePdschConfigDedicated = true;
  physicalConfigDedicated.pdschConfigDedicated.pa = LteRrcSap::PdschConfigDedicated::dB0;

  rrd.physicalConfigDedicated = physicalConfigDedicated;

  return rrd;
}

void
RrcHeaderTestCase::AssertEqualRadioResourceConfigDedicated (LteRrcSap::RadioResourceConfigDedicated rrcd1, LteRrcSap::RadioResourceConfigDedicated rrcd2)
{
  NS_TEST_ASSERT_MSG_EQ (rrcd1.srbToAddModList.size (), rrcd2.srbToAddModList.size (),"SrbToAddModList different sizes");

  std::list<LteRrcSap::SrbToAddMod> srcSrbToAddModList = rrcd1.srbToAddModList;
  std::list<LteRrcSap::SrbToAddMod>::iterator it1 = srcSrbToAddModList.begin ();
  std::list<LteRrcSap::SrbToAddMod> dstSrbToAddModList = rrcd2.srbToAddModList;
  std::list<LteRrcSap::SrbToAddMod>::iterator it2 = dstSrbToAddModList.begin ();

  for (; it1 != srcSrbToAddModList.end (); it1++, it2++)
    {
      NS_TEST_ASSERT_MSG_EQ (it1->srbIdentity,it2->srbIdentity, "srbIdentity");
      NS_TEST_ASSERT_MSG_EQ (it1->logicalChannelConfig.priority,it2->logicalChannelConfig.priority, "logicalChannelConfig.priority");
      NS_TEST_ASSERT_MSG_EQ (it1->logicalChannelConfig.prioritizedBitRateKbps,it2->logicalChannelConfig.prioritizedBitRateKbps, "logicalChannelConfig.prioritizedBitRateKbps");
      NS_TEST_ASSERT_MSG_EQ (it1->logicalChannelConfig.bucketSizeDurationMs,it2->logicalChannelConfig.bucketSizeDurationMs, "logicalChannelConfig.bucketSizeDurationMs");
      NS_TEST_ASSERT_MSG_EQ (it1->logicalChannelConfig.logicalChannelGroup,it2->logicalChannelConfig.logicalChannelGroup, "logicalChannelConfig.logicalChannelGroup");
    }

  NS_TEST_ASSERT_MSG_EQ (rrcd1.drbToAddModList.size (), rrcd2.drbToAddModList.size (),"DrbToAddModList different sizes");

  std::list<LteRrcSap::DrbToAddMod> srcDrbToAddModList = rrcd1.drbToAddModList;
  std::list<LteRrcSap::DrbToAddMod>::iterator it3 = srcDrbToAddModList.begin ();
  std::list<LteRrcSap::DrbToAddMod> dstDrbToAddModList = rrcd2.drbToAddModList;
  std::list<LteRrcSap::DrbToAddMod>::iterator it4 = dstDrbToAddModList.begin ();

  for (; it3 != srcDrbToAddModList.end (); it3++, it4++)
    {
      NS_TEST_ASSERT_MSG_EQ (it3->epsBearerIdentity,it4->epsBearerIdentity, "epsBearerIdentity");
      NS_TEST_ASSERT_MSG_EQ (it3->drbIdentity,it4->drbIdentity, "drbIdentity");
      NS_TEST_ASSERT_MSG_EQ (it3->rlcConfig.choice,it4->rlcConfig.choice, "rlcConfig.choice");
      NS_TEST_ASSERT_MSG_EQ (it3->logicalChannelIdentity,it4->logicalChannelIdentity, "logicalChannelIdentity");
      NS_TEST_ASSERT_MSG_EQ (it3->epsBearerIdentity,it4->epsBearerIdentity, "epsBearerIdentity");

      NS_TEST_ASSERT_MSG_EQ (it3->logicalChannelConfig.priority,it4->logicalChannelConfig.priority, "logicalChannelConfig.priority");
      NS_TEST_ASSERT_MSG_EQ (it3->logicalChannelConfig.prioritizedBitRateKbps,it4->logicalChannelConfig.prioritizedBitRateKbps, "logicalChannelConfig.prioritizedBitRateKbps");
      NS_TEST_ASSERT_MSG_EQ (it3->logicalChannelConfig.bucketSizeDurationMs,it4->logicalChannelConfig.bucketSizeDurationMs, "logicalChannelConfig.bucketSizeDurationMs");
      NS_TEST_ASSERT_MSG_EQ (it3->logicalChannelConfig.logicalChannelGroup,it4->logicalChannelConfig.logicalChannelGroup, "logicalChannelConfig.logicalChannelGroup");
    }

  NS_TEST_ASSERT_MSG_EQ (rrcd1.drbToReleaseList.size (), rrcd2.drbToReleaseList.size (),"DrbToReleaseList different sizes");

  std::list<uint8_t> srcDrbToReleaseList = rrcd1.drbToReleaseList;
  std::list<uint8_t> dstDrbToReleaseList = rrcd2.drbToReleaseList;
  std::list<uint8_t>::iterator it5 = srcDrbToReleaseList.begin ();
  std::list<uint8_t>::iterator it6 = dstDrbToReleaseList.begin ();

  for (; it5 != srcDrbToReleaseList.end (); it5++, it6++)
    {
      NS_TEST_ASSERT_MSG_EQ (*it5, *it6,"element != in DrbToReleaseList");
    }

  NS_TEST_ASSERT_MSG_EQ (rrcd1.havePhysicalConfigDedicated,rrcd2.havePhysicalConfigDedicated, "HavePhysicalConfigDedicated");

  if (rrcd1.havePhysicalConfigDedicated)
    {
      NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.haveSoundingRsUlConfigDedicated,
                             rrcd2.physicalConfigDedicated.haveSoundingRsUlConfigDedicated,
                             "haveSoundingRsUlConfigDedicated");

      NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.soundingRsUlConfigDedicated.type,
                             rrcd2.physicalConfigDedicated.soundingRsUlConfigDedicated.type,
                             "soundingRsUlConfigDedicated.type");
      NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.soundingRsUlConfigDedicated.srsBandwidth,
                             rrcd2.physicalConfigDedicated.soundingRsUlConfigDedicated.srsBandwidth,
                             "soundingRsUlConfigDedicated.srsBandwidth");

      NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.soundingRsUlConfigDedicated.srsConfigIndex,
                             rrcd2.physicalConfigDedicated.soundingRsUlConfigDedicated.srsConfigIndex,
                             "soundingRsUlConfigDedicated.srsConfigIndex");

      NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.haveAntennaInfoDedicated,
                             rrcd2.physicalConfigDedicated.haveAntennaInfoDedicated,
                             "haveAntennaInfoDedicated");

      if (rrcd1.physicalConfigDedicated.haveAntennaInfoDedicated)
        {
          NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.antennaInfo.transmissionMode,
                                 rrcd2.physicalConfigDedicated.antennaInfo.transmissionMode,
                                 "antennaInfo.transmissionMode");
        }

      NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.havePdschConfigDedicated,
                             rrcd2.physicalConfigDedicated.havePdschConfigDedicated,
                             "havePdschConfigDedicated");

      if (rrcd1.physicalConfigDedicated.havePdschConfigDedicated)
        {
          NS_TEST_ASSERT_MSG_EQ (rrcd1.physicalConfigDedicated.pdschConfigDedicated.pa,
                                 rrcd2.physicalConfigDedicated.pdschConfigDedicated.pa,
                                 "pdschConfigDedicated.pa");
        }
    }
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Request Test Case
 */
class RrcConnectionRequestTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionRequestTestCase ();
  virtual void DoRun (void);
};

RrcConnectionRequestTestCase::RrcConnectionRequestTestCase () : RrcHeaderTestCase ("Testing RrcConnectionRequest")
{
}

void
RrcConnectionRequestTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionRequestTestCase ===========");

  LteRrcSap::RrcConnectionRequest msg;
  msg.ueIdentity = 0x83fecafecaULL;

  RrcConnectionRequestHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionRequestHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // Remove header
  RrcConnectionRequestHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionRequestHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetMmec (),destination.GetMmec (), "Different m_mmec!");
  NS_TEST_ASSERT_MSG_EQ (source.GetMtmsi (),destination.GetMtmsi (), "Different m_mTmsi!");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Setup Test Case
 */
class RrcConnectionSetupTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionSetupTestCase ();
  virtual void DoRun (void);
};

RrcConnectionSetupTestCase::RrcConnectionSetupTestCase () : RrcHeaderTestCase ("Testing RrcConnectionSetupTestCase")
{
}

void
RrcConnectionSetupTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionSetupTestCase ===========");

  LteRrcSap::RrcConnectionSetup msg;
  msg.rrcTransactionIdentifier = 3;
  msg.radioResourceConfigDedicated = CreateRadioResourceConfigDedicated ();

  RrcConnectionSetupHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionSetupHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionSetupHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionSetupHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetRrcTransactionIdentifier (),destination.GetRrcTransactionIdentifier (), "RrcTransactionIdentifier");

  AssertEqualRadioResourceConfigDedicated (source.GetRadioResourceConfigDedicated (),destination.GetRadioResourceConfigDedicated ());

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Setup Complete Test Case
 */
class RrcConnectionSetupCompleteTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionSetupCompleteTestCase ();
  virtual void DoRun (void);
};

RrcConnectionSetupCompleteTestCase::RrcConnectionSetupCompleteTestCase () : RrcHeaderTestCase ("Testing RrcConnectionSetupCompleteTestCase")
{
}

void
RrcConnectionSetupCompleteTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionSetupCompleteTestCase ===========");

  LteRrcSap::RrcConnectionSetupCompleted msg;
  msg.rrcTransactionIdentifier = 3;

  RrcConnectionSetupCompleteHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionSetupCompleteHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // Remove header
  RrcConnectionSetupCompleteHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionSetupCompleteHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetRrcTransactionIdentifier (),destination.GetRrcTransactionIdentifier (), "RrcTransactionIdentifier");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Reconfiguration Complete Test Case
 */
class RrcConnectionReconfigurationCompleteTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionReconfigurationCompleteTestCase ();
  virtual void DoRun (void);
};

RrcConnectionReconfigurationCompleteTestCase::RrcConnectionReconfigurationCompleteTestCase ()
  : RrcHeaderTestCase ("Testing RrcConnectionReconfigurationCompleteTestCase")
{
}

void
RrcConnectionReconfigurationCompleteTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionReconfigurationCompleteTestCase ===========");

  LteRrcSap::RrcConnectionReconfigurationCompleted msg;
  msg.rrcTransactionIdentifier = 2;

  RrcConnectionReconfigurationCompleteHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionReconfigurationCompleteHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionReconfigurationCompleteHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionReconfigurationCompleteHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetRrcTransactionIdentifier (),destination.GetRrcTransactionIdentifier (), "RrcTransactionIdentifier");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Reconfiguration Test Case
 */
class RrcConnectionReconfigurationTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionReconfigurationTestCase ();
  virtual void DoRun (void);
};

RrcConnectionReconfigurationTestCase::RrcConnectionReconfigurationTestCase ()
  : RrcHeaderTestCase ("Testing RrcConnectionReconfigurationTestCase")
{
}

void
RrcConnectionReconfigurationTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionReconfigurationTestCase ===========");

  LteRrcSap::RrcConnectionReconfiguration msg;
  msg.rrcTransactionIdentifier = 2;

  msg.haveMeasConfig = true;

  msg.measConfig.haveQuantityConfig = true;
  msg.measConfig.quantityConfig.filterCoefficientRSRP = 8;
  msg.measConfig.quantityConfig.filterCoefficientRSRQ = 7;

  msg.measConfig.haveMeasGapConfig = true;
  msg.measConfig.measGapConfig.type = LteRrcSap::MeasGapConfig::SETUP;
  msg.measConfig.measGapConfig.gapOffsetChoice = LteRrcSap::MeasGapConfig::GP0;
  msg.measConfig.measGapConfig.gapOffsetValue = 21;

  msg.measConfig.haveSmeasure = true;
  msg.measConfig.sMeasure = 57;

  msg.measConfig.haveSpeedStatePars = true;
  msg.measConfig.speedStatePars.type = LteRrcSap::SpeedStatePars::SETUP;
  msg.measConfig.speedStatePars.mobilityStateParameters.tEvaluation = 240;
  msg.measConfig.speedStatePars.mobilityStateParameters.tHystNormal = 60;
  msg.measConfig.speedStatePars.mobilityStateParameters.nCellChangeMedium = 5;
  msg.measConfig.speedStatePars.mobilityStateParameters.nCellChangeHigh = 13;
  msg.measConfig.speedStatePars.timeToTriggerSf.sfMedium = 25;
  msg.measConfig.speedStatePars.timeToTriggerSf.sfHigh = 75;

  msg.measConfig.measObjectToRemoveList.push_back (23);
  msg.measConfig.measObjectToRemoveList.push_back (13);

  msg.measConfig.reportConfigToRemoveList.push_back (7);
  msg.measConfig.reportConfigToRemoveList.push_back (16);

  msg.measConfig.measIdToRemoveList.push_back (4);
  msg.measConfig.measIdToRemoveList.push_back (18);

  // Set measObjectToAddModList
  LteRrcSap::MeasObjectToAddMod measObjectToAddMod;
  measObjectToAddMod.measObjectId = 3;
  measObjectToAddMod.measObjectEutra.carrierFreq = 21;
  measObjectToAddMod.measObjectEutra.allowedMeasBandwidth = 15;
  measObjectToAddMod.measObjectEutra.presenceAntennaPort1 = true;
  measObjectToAddMod.measObjectEutra.neighCellConfig = 3;
  measObjectToAddMod.measObjectEutra.offsetFreq = -12;
  measObjectToAddMod.measObjectEutra.cellsToRemoveList.push_back (5);
  measObjectToAddMod.measObjectEutra.cellsToRemoveList.push_back (2);
  measObjectToAddMod.measObjectEutra.blackCellsToRemoveList.push_back (1);
  measObjectToAddMod.measObjectEutra.haveCellForWhichToReportCGI = true;
  measObjectToAddMod.measObjectEutra.cellForWhichToReportCGI = 250;
  LteRrcSap::CellsToAddMod cellsToAddMod;
  cellsToAddMod.cellIndex = 20;
  cellsToAddMod.physCellId = 14;
  cellsToAddMod.cellIndividualOffset = 22;
  measObjectToAddMod.measObjectEutra.cellsToAddModList.push_back (cellsToAddMod);
  LteRrcSap::BlackCellsToAddMod blackCellsToAddMod;
  blackCellsToAddMod.cellIndex = 18;
  blackCellsToAddMod.physCellIdRange.start = 128;
  blackCellsToAddMod.physCellIdRange.haveRange = true;
  blackCellsToAddMod.physCellIdRange.range = 128;
  measObjectToAddMod.measObjectEutra.blackCellsToAddModList.push_back (blackCellsToAddMod);
  msg.measConfig.measObjectToAddModList.push_back (measObjectToAddMod);

  // Set reportConfigToAddModList
  LteRrcSap::ReportConfigToAddMod reportConfigToAddMod;
  reportConfigToAddMod.reportConfigId = 22;
  reportConfigToAddMod.reportConfigEutra.triggerType = LteRrcSap::ReportConfigEutra::EVENT;
  reportConfigToAddMod.reportConfigEutra.eventId = LteRrcSap::ReportConfigEutra::EVENT_A2;
  reportConfigToAddMod.reportConfigEutra.threshold1.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRP;
  reportConfigToAddMod.reportConfigEutra.threshold1.range = 15;
  reportConfigToAddMod.reportConfigEutra.threshold2.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRQ;
  reportConfigToAddMod.reportConfigEutra.threshold2.range = 10;
  reportConfigToAddMod.reportConfigEutra.reportOnLeave = true;
  reportConfigToAddMod.reportConfigEutra.a3Offset = -25;
  reportConfigToAddMod.reportConfigEutra.hysteresis = 18;
  reportConfigToAddMod.reportConfigEutra.timeToTrigger = 100;
  reportConfigToAddMod.reportConfigEutra.purpose = LteRrcSap::ReportConfigEutra::REPORT_STRONGEST_CELLS;
  reportConfigToAddMod.reportConfigEutra.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRQ;
  reportConfigToAddMod.reportConfigEutra.reportQuantity = LteRrcSap::ReportConfigEutra::SAME_AS_TRIGGER_QUANTITY;
  reportConfigToAddMod.reportConfigEutra.maxReportCells = 5;
  reportConfigToAddMod.reportConfigEutra.reportInterval = LteRrcSap::ReportConfigEutra::MIN60;
  reportConfigToAddMod.reportConfigEutra.reportAmount = 16; 
  msg.measConfig.reportConfigToAddModList.push_back (reportConfigToAddMod);

  // Set measIdToAddModList
  LteRrcSap::MeasIdToAddMod measIdToAddMod,measIdToAddMod2;
  measIdToAddMod.measId = 7;
  measIdToAddMod.measObjectId = 6;
  measIdToAddMod.reportConfigId = 5;
  measIdToAddMod2.measId = 4;
  measIdToAddMod2.measObjectId = 8;
  measIdToAddMod2.reportConfigId = 12;
  msg.measConfig.measIdToAddModList.push_back (measIdToAddMod);
  msg.measConfig.measIdToAddModList.push_back (measIdToAddMod2);

  msg.haveMobilityControlInfo = true;
  msg.mobilityControlInfo.targetPhysCellId = 4;
  msg.mobilityControlInfo.haveCarrierFreq = true;
  msg.mobilityControlInfo.carrierFreq.dlCarrierFreq = 3;
  msg.mobilityControlInfo.carrierFreq.ulCarrierFreq = 5;
  msg.mobilityControlInfo.haveCarrierBandwidth = true;
  msg.mobilityControlInfo.carrierBandwidth.dlBandwidth = 50;
  msg.mobilityControlInfo.carrierBandwidth.ulBandwidth = 25;
  msg.mobilityControlInfo.newUeIdentity = 11;
  msg.mobilityControlInfo.haveRachConfigDedicated = true;
  msg.mobilityControlInfo.rachConfigDedicated.raPreambleIndex = 2;
  msg.mobilityControlInfo.rachConfigDedicated.raPrachMaskIndex = 2;
  msg.mobilityControlInfo.radioResourceConfigCommon.rachConfigCommon.preambleInfo.numberOfRaPreambles = 4;
  msg.mobilityControlInfo.radioResourceConfigCommon.rachConfigCommon.raSupervisionInfo.preambleTransMax = 3;
  msg.mobilityControlInfo.radioResourceConfigCommon.rachConfigCommon.raSupervisionInfo.raResponseWindowSize = 6;

  msg.haveRadioResourceConfigDedicated = true;

  msg.radioResourceConfigDedicated = CreateRadioResourceConfigDedicated ();

  msg.haveNonCriticalExtension = false; //Danilo
  RrcConnectionReconfigurationHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionReconfigurationHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionReconfigurationHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionReconfigurationHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetRrcTransactionIdentifier (),destination.GetRrcTransactionIdentifier (), "RrcTransactionIdentifier");
  NS_TEST_ASSERT_MSG_EQ (source.GetHaveMeasConfig (),destination.GetHaveMeasConfig (), "GetHaveMeasConfig");
  NS_TEST_ASSERT_MSG_EQ (source.GetHaveMobilityControlInfo (),destination.GetHaveMobilityControlInfo (), "GetHaveMobilityControlInfo");
  NS_TEST_ASSERT_MSG_EQ (source.GetHaveRadioResourceConfigDedicated (),destination.GetHaveRadioResourceConfigDedicated (), "GetHaveRadioResourceConfigDedicated");

  if ( source.GetHaveMobilityControlInfo () )
    {
      NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().targetPhysCellId,destination.GetMobilityControlInfo ().targetPhysCellId, "GetMobilityControlInfo().targetPhysCellId");
      NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().haveCarrierFreq,destination.GetMobilityControlInfo ().haveCarrierFreq, "GetMobilityControlInfo().haveCarrierFreq");
      NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().haveCarrierBandwidth,destination.GetMobilityControlInfo ().haveCarrierBandwidth, "GetMobilityControlInfo().haveCarrierBandwidth");
      NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().newUeIdentity,destination.GetMobilityControlInfo ().newUeIdentity, "GetMobilityControlInfo().newUeIdentity");
      NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().haveRachConfigDedicated,destination.GetMobilityControlInfo ().haveRachConfigDedicated, "GetMobilityControlInfo().haveRachConfigDedicated");

      if (source.GetMobilityControlInfo ().haveCarrierFreq)
        {
          NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().carrierFreq.dlCarrierFreq,
                                 destination.GetMobilityControlInfo ().carrierFreq.dlCarrierFreq,
                                 "GetMobilityControlInfo().carrierFreq.dlCarrierFreq");
          NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().carrierFreq.ulCarrierFreq,
                                 destination.GetMobilityControlInfo ().carrierFreq.ulCarrierFreq,
                                 "GetMobilityControlInfo().carrierFreq.ulCarrierFreq");
        }

      if (source.GetMobilityControlInfo ().haveCarrierBandwidth)
        {
          NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().carrierBandwidth.dlBandwidth,
                                 destination.GetMobilityControlInfo ().carrierBandwidth.dlBandwidth,
                                 "GetMobilityControlInfo().carrierBandwidth.dlBandwidth");
          NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().carrierBandwidth.ulBandwidth,
                                 destination.GetMobilityControlInfo ().carrierBandwidth.ulBandwidth,
                                 "GetMobilityControlInfo().carrierBandwidth.ulBandwidth");
        }

      if (source.GetMobilityControlInfo ().haveRachConfigDedicated)
        {
          NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().rachConfigDedicated.raPreambleIndex,
                                 destination.GetMobilityControlInfo ().rachConfigDedicated.raPreambleIndex,
                                 "GetMobilityControlInfo().rachConfigDedicated.raPreambleIndex");
          NS_TEST_ASSERT_MSG_EQ (source.GetMobilityControlInfo ().rachConfigDedicated.raPrachMaskIndex,
                                 destination.GetMobilityControlInfo ().rachConfigDedicated.raPrachMaskIndex,
                                 "GetMobilityControlInfo().rachConfigDedicated.raPrachMaskIndex");
        }
    }

  if (source.GetHaveRadioResourceConfigDedicated ())
    {
      AssertEqualRadioResourceConfigDedicated (source.GetRadioResourceConfigDedicated (), destination.GetRadioResourceConfigDedicated ());
    }

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Handover Preparation Info Test Case
 */
class HandoverPreparationInfoTestCase : public RrcHeaderTestCase
{
public:
  HandoverPreparationInfoTestCase ();
  virtual void DoRun (void);
};

HandoverPreparationInfoTestCase::HandoverPreparationInfoTestCase () : RrcHeaderTestCase ("Testing HandoverPreparationInfoTestCase")
{
}

void
HandoverPreparationInfoTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= HandoverPreparationInfoTestCase ===========");

  LteRrcSap::HandoverPreparationInfo msg;
  msg.asConfig.sourceDlCarrierFreq = 3;
  msg.asConfig.sourceUeIdentity = 11;
  msg.asConfig.sourceRadioResourceConfig = CreateRadioResourceConfigDedicated ();
  msg.asConfig.sourceMasterInformationBlock.dlBandwidth = 50;
  msg.asConfig.sourceMasterInformationBlock.systemFrameNumber = 1;

  msg.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIndication = true;
  msg.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.cellIdentity = 5;
  msg.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIdentity = 4;
  msg.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.plmnIdentityInfo.plmnIdentity = 123;

  msg.asConfig.sourceSystemInformationBlockType2.freqInfo.ulBandwidth = 100;
  msg.asConfig.sourceSystemInformationBlockType2.freqInfo.ulCarrierFreq = 10;
  msg.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon.rachConfigCommon.preambleInfo.numberOfRaPreambles = 4;
  msg.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon.rachConfigCommon.raSupervisionInfo.preambleTransMax = 3;
  msg.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon.rachConfigCommon.raSupervisionInfo.raResponseWindowSize = 6;

  msg.asConfig.sourceMeasConfig.haveQuantityConfig = false;
  msg.asConfig.sourceMeasConfig.haveMeasGapConfig = false;
  msg.asConfig.sourceMeasConfig.haveSmeasure = false;
  msg.asConfig.sourceMeasConfig.haveSpeedStatePars = false;

  HandoverPreparationInfoHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<HandoverPreparationInfoHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  HandoverPreparationInfoHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<HandoverPreparationInfoHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  AssertEqualRadioResourceConfigDedicated (source.GetAsConfig ().sourceRadioResourceConfig, destination.GetAsConfig ().sourceRadioResourceConfig);
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceUeIdentity, destination.GetAsConfig ().sourceUeIdentity, "sourceUeIdentity");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceMasterInformationBlock.dlBandwidth,destination.GetAsConfig ().sourceMasterInformationBlock.dlBandwidth, "dlBandwidth");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceMasterInformationBlock.systemFrameNumber, destination.GetAsConfig ().sourceMasterInformationBlock.systemFrameNumber, "systemFrameNumber");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.plmnIdentityInfo.plmnIdentity, destination.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.plmnIdentityInfo.plmnIdentity, "plmnIdentity");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIndication, destination.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIndication, "csgIndication");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.cellIdentity, destination.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.cellIdentity, "cellIdentity");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIdentity, destination.GetAsConfig ().sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIdentity, "csgIdentity");
  NS_TEST_ASSERT_MSG_EQ (source.GetAsConfig ().sourceDlCarrierFreq, destination.GetAsConfig ().sourceDlCarrierFreq, "sourceDlCarrierFreq");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Reestablishment Request Test Case
 */
class RrcConnectionReestablishmentRequestTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionReestablishmentRequestTestCase ();
  virtual void DoRun (void);
};

RrcConnectionReestablishmentRequestTestCase::RrcConnectionReestablishmentRequestTestCase () : RrcHeaderTestCase ("Testing RrcConnectionReestablishmentRequestTestCase")
{
}

void
RrcConnectionReestablishmentRequestTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionReestablishmentRequestTestCase ===========");

  LteRrcSap::RrcConnectionReestablishmentRequest msg;
  msg.ueIdentity.cRnti = 12;
  msg.ueIdentity.physCellId = 21;
  msg.reestablishmentCause = LteRrcSap::HANDOVER_FAILURE;

  RrcConnectionReestablishmentRequestHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionReestablishmentRequestHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionReestablishmentRequestHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionReestablishmentRequestHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetUeIdentity ().cRnti, destination.GetUeIdentity ().cRnti, "cRnti");
  NS_TEST_ASSERT_MSG_EQ (source.GetUeIdentity ().physCellId, destination.GetUeIdentity ().physCellId, "physCellId");
  NS_TEST_ASSERT_MSG_EQ (source.GetReestablishmentCause (),destination.GetReestablishmentCause (), "ReestablishmentCause");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Reestablishment Test Case
 */
class RrcConnectionReestablishmentTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionReestablishmentTestCase ();
  virtual void DoRun (void);
};

RrcConnectionReestablishmentTestCase::RrcConnectionReestablishmentTestCase () : RrcHeaderTestCase ("Testing RrcConnectionReestablishmentTestCase")
{
}

void
RrcConnectionReestablishmentTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionReestablishmentTestCase ===========");

  LteRrcSap::RrcConnectionReestablishment msg;
  msg.rrcTransactionIdentifier = 2;
  msg.radioResourceConfigDedicated = CreateRadioResourceConfigDedicated ();

  RrcConnectionReestablishmentHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionReestablishmentHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionReestablishmentHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionReestablishmentHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetRrcTransactionIdentifier (), destination.GetRrcTransactionIdentifier (), "rrcTransactionIdentifier");
  AssertEqualRadioResourceConfigDedicated (source.GetRadioResourceConfigDedicated (),destination.GetRadioResourceConfigDedicated ());

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Reestablishment Complete Test Case
 */
class RrcConnectionReestablishmentCompleteTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionReestablishmentCompleteTestCase ();
  virtual void DoRun (void);
};

RrcConnectionReestablishmentCompleteTestCase::RrcConnectionReestablishmentCompleteTestCase () : RrcHeaderTestCase ("Testing RrcConnectionReestablishmentCompleteTestCase")
{
}

void
RrcConnectionReestablishmentCompleteTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionReestablishmentCompleteTestCase ===========");

  LteRrcSap::RrcConnectionReestablishmentComplete msg;
  msg.rrcTransactionIdentifier = 3;

  RrcConnectionReestablishmentCompleteHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionReestablishmentCompleteHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionReestablishmentCompleteHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionReestablishmentCompleteHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetRrcTransactionIdentifier (), destination.GetRrcTransactionIdentifier (), "rrcTransactionIdentifier");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Rrc Connection Reject Test Case
 */
class RrcConnectionRejectTestCase : public RrcHeaderTestCase
{
public:
  RrcConnectionRejectTestCase ();
  virtual void DoRun (void);
};

RrcConnectionRejectTestCase::RrcConnectionRejectTestCase () : RrcHeaderTestCase ("Testing RrcConnectionRejectTestCase")
{
}

void
RrcConnectionRejectTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= RrcConnectionRejectTestCase ===========");

  LteRrcSap::RrcConnectionReject msg;
  msg.waitTime = 2;

  RrcConnectionRejectHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<RrcConnectionRejectHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  RrcConnectionRejectHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<RrcConnectionRejectHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  NS_TEST_ASSERT_MSG_EQ (source.GetMessage ().waitTime, destination.GetMessage ().waitTime, "Different waitTime!");

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Measurement Report Test Case
 */
class MeasurementReportTestCase : public RrcHeaderTestCase
{
public:
  MeasurementReportTestCase ();
  virtual void DoRun (void);
};

MeasurementReportTestCase::MeasurementReportTestCase () : RrcHeaderTestCase ("Testing MeasurementReportTestCase")
{
}

void
MeasurementReportTestCase::DoRun (void)
{
  packet = Create<Packet> ();
  NS_LOG_DEBUG ("============= MeasurementReportTestCase ===========");

  LteRrcSap::MeasurementReport msg;
  msg.measResults.measId = 5;
  msg.measResults.measResultPCell.rsrpResult = 18;
  msg.measResults.measResultPCell.rsrqResult = 21;
  msg.measResults.haveMeasResultNeighCells = true;

  LteRrcSap::MeasResultEutra mResEutra;
  mResEutra.physCellId = 9;
  mResEutra.haveRsrpResult = true;
  mResEutra.rsrpResult = 33;
  mResEutra.haveRsrqResult = true;
  mResEutra.rsrqResult = 22;
  mResEutra.haveCgiInfo = true;
  mResEutra.cgiInfo.plmnIdentity = 7;
  mResEutra.cgiInfo.cellIdentity = 6;
  mResEutra.cgiInfo.trackingAreaCode = 5;
  msg.measResults.measResultListEutra.push_back (mResEutra);

  msg.measResults.haveMeasResultServFreqList = false;

  MeasurementReportHeader source;
  source.SetMessage (msg);

  // Log source info
  TestUtils::LogPacketInfo<MeasurementReportHeader> (source,"SOURCE");

  // Add header
  packet->AddHeader (source);

  // Log serialized packet contents
  TestUtils::LogPacketContents (packet);

  // remove header
  MeasurementReportHeader destination;
  packet->RemoveHeader (destination);

  // Log destination info
  TestUtils::LogPacketInfo<MeasurementReportHeader> (destination,"DESTINATION");

  // Check that the destination and source headers contain the same values
  LteRrcSap::MeasResults srcMeas = source.GetMessage ().measResults;
  LteRrcSap::MeasResults dstMeas = destination.GetMessage ().measResults;

  NS_TEST_ASSERT_MSG_EQ (srcMeas.measId, dstMeas.measId, "Different measId!");
  NS_TEST_ASSERT_MSG_EQ (srcMeas.measResultPCell.rsrpResult, dstMeas.measResultPCell.rsrpResult, "Different rsrpResult!");
  NS_TEST_ASSERT_MSG_EQ (srcMeas.measResultPCell.rsrqResult, dstMeas.measResultPCell.rsrqResult, "Different rsrqResult!");
  NS_TEST_ASSERT_MSG_EQ (srcMeas.haveMeasResultNeighCells, dstMeas.haveMeasResultNeighCells, "Different haveMeasResultNeighCells!");

  if (srcMeas.haveMeasResultNeighCells)
    {
      std::list<LteRrcSap::MeasResultEutra>::iterator itsrc = srcMeas.measResultListEutra.begin ();
      std::list<LteRrcSap::MeasResultEutra>::iterator itdst = dstMeas.measResultListEutra.begin ();
      for (; itsrc != srcMeas.measResultListEutra.end (); itsrc++, itdst++)
        {
          NS_TEST_ASSERT_MSG_EQ (itsrc->physCellId, itdst->physCellId, "Different physCellId!");

          NS_TEST_ASSERT_MSG_EQ (itsrc->haveCgiInfo, itdst->haveCgiInfo, "Different haveCgiInfo!");
          if (itsrc->haveCgiInfo)
            {
              NS_TEST_ASSERT_MSG_EQ (itsrc->cgiInfo.plmnIdentity, itdst->cgiInfo.plmnIdentity, "Different cgiInfo.plmnIdentity!");
              NS_TEST_ASSERT_MSG_EQ (itsrc->cgiInfo.cellIdentity, itdst->cgiInfo.cellIdentity, "Different cgiInfo.cellIdentity!");
              NS_TEST_ASSERT_MSG_EQ (itsrc->cgiInfo.trackingAreaCode, itdst->cgiInfo.trackingAreaCode, "Different cgiInfo.trackingAreaCode!");
              NS_TEST_ASSERT_MSG_EQ (itsrc->cgiInfo.plmnIdentityList.size (), itdst->cgiInfo.plmnIdentityList.size (), "Different cgiInfo.plmnIdentityList.size()!");

              if (!itsrc->cgiInfo.plmnIdentityList.empty ())
                {
                  std::list<uint32_t>::iterator itsrc2 = itsrc->cgiInfo.plmnIdentityList.begin ();
                  std::list<uint32_t>::iterator itdst2 = itdst->cgiInfo.plmnIdentityList.begin ();
                  for (; itsrc2 != itsrc->cgiInfo.plmnIdentityList.begin (); itsrc2++, itdst2++)
                    {
                      NS_TEST_ASSERT_MSG_EQ (*itsrc2, *itdst2, "Different plmnId elements!");
                    }
                }
            }

          NS_TEST_ASSERT_MSG_EQ (itsrc->haveRsrpResult, itdst->haveRsrpResult, "Different haveRsrpResult!");
          if (itsrc->haveRsrpResult)
            {
              NS_TEST_ASSERT_MSG_EQ (itsrc->rsrpResult, itdst->rsrpResult, "Different rsrpResult!");
            }

          NS_TEST_ASSERT_MSG_EQ (itsrc->haveRsrqResult, itdst->haveRsrqResult, "Different haveRsrqResult!");
          if (itsrc->haveRsrqResult)
            {
              NS_TEST_ASSERT_MSG_EQ (itsrc->rsrqResult, itdst->rsrqResult, "Different rsrqResult!");
            }

        }
    }

  packet = 0;
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Asn1Encoding Test Suite
 */
class Asn1EncodingSuite : public TestSuite
{
public:
  Asn1EncodingSuite ();
};

Asn1EncodingSuite::Asn1EncodingSuite ()
  : TestSuite ("test-asn1-encoding", UNIT)
{
  NS_LOG_FUNCTION (this);
  AddTestCase (new RrcConnectionRequestTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionSetupTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionSetupCompleteTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionReconfigurationCompleteTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionReconfigurationTestCase (), TestCase::QUICK);
  AddTestCase (new HandoverPreparationInfoTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionReestablishmentRequestTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionReestablishmentTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionReestablishmentCompleteTestCase (), TestCase::QUICK);
  AddTestCase (new RrcConnectionRejectTestCase (), TestCase::QUICK);
  AddTestCase (new MeasurementReportTestCase (), TestCase::QUICK);
}

Asn1EncodingSuite asn1EncodingSuite;

