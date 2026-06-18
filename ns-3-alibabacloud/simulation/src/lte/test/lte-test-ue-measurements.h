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
 *         Marco Miozzo <mmiozzo@cttc.es>
 *              adapt lte-test-interference.cc to lte-ue-measurements.cc
 *         Budiarto Herman <budiarto.herman@magister.fi>
 */

#ifndef LTE_TEST_UE_MEASUREMENTS_H
#define LTE_TEST_UE_MEASUREMENTS_H

#include <ns3/test.h>
#include <ns3/lte-rrc-sap.h>
#include <ns3/nstime.h>
#include <list>
#include <set>
#include <vector>

namespace ns3 {

class MobilityModel;

}

using namespace ns3;


// ===== LTE-UE-MEASUREMENTS TEST SUITE ==================================== //


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test that UE Measurements (see 36.214) calculation works fine in a
 * multi-cell interference scenario.
 */
class LteUeMeasurementsTestSuite : public TestSuite
{
public:
  LteUeMeasurementsTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test that UE measurements calculation works properly in a scenario 
 * with 2 eNodeBs and 2UEs. Test that RSRP and RSRQ of the serving cell and of 
 * the neighbor cell correspond to the reference values.
 */
class LteUeMeasurementsTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param d1 distance between UE and ENB node pair
   * \param d2 distance between UE and other ENB node
   * \param rsrpDbmUe1 RSRP in dBm UE 1
   * \param rsrpDbmUe2 RSRP in dBm UE 2
   * \param rsrqDbUe1 RSRQ in dBm UE 1
   * \param rsrqDbUe2 RSRQ in dBm UE 2
   */
  LteUeMeasurementsTestCase (std::string name, double d1, double d2, double rsrpDbmUe1, double rsrpDbmUe2, double rsrqDbUe1, double rsrqDbUe2);
  virtual ~LteUeMeasurementsTestCase ();

  /**
   * Report UE measurements function
   * \param rnti the RNTI
   * \param cellId the cell ID
   * \param rsrp the RSRP
   * \param rsrq the RSRQ
   * \param servingCell the serving cell
   */
  void ReportUeMeasurements (uint16_t rnti, uint16_t cellId, double rsrp, double rsrq, bool servingCell);

  /**
   * Reeive measurement report function
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   * \param meas LteRrcSap::MeasurementReport
   */
  void RecvMeasurementReport (uint64_t imsi, uint16_t cellId, uint16_t rnti, LteRrcSap::MeasurementReport meas);

private:
  virtual void DoRun (void);

  double m_d1; ///< distance between UE and ENB node pair
  double m_d2; ///< distance between UE and other ENB node
  double m_rsrpDbmUeServingCell; ///< RSRP in dBm UE 1
  double m_rsrpDbmUeNeighborCell; ///< RSRP in dBm UE 2
  double m_rsrqDbUeServingCell; ///< RSRQ in dBm UE 1
  double m_rsrqDbUeNeighborCell; ///< RSRQ in dBm UE 2

};



// ===== LTE-UE-MEASUREMENTS-PIECEWISE-1 TEST SUITE ======================== //


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for generating calls to UE measurements test case
 *        ns3::LteUeMeasurementsPiecewiseTestCase1.
 */
class LteUeMeasurementsPiecewiseTestSuite1 : public TestSuite
{
public:
  LteUeMeasurementsPiecewiseTestSuite1 ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Testing UE measurements in LTE with simulation of 1 eNodeB and 1 UE in
 *        piecewise configuration and 120 ms report interval. During the simulation 
 *        the placement of UE is being changed several times. Four different 
 *        cases are considered: UE is very near to eNodeB, UE is near to eNodeB, 
 *        UE is far from eNodeB and UE is very far from eNodeB. Test checks 
 *        if the measurments correspond to the real conditions of the UE, i.e. 
 *        wnen the signal from serving cell becomes weak, the measurements should 
 *        also start to correspond to the new channel conditions. Additionally, it 
 *        is checked if UE detects some neighboring signal, if it does the test 
 *        fails because there is no neighbor in this configuration. Also, test 
 *        checks if the reporting occurs at intervals that it should according to 
 *        the measurements configuration. If it occurs in some different time 
 *        stamp the test will fail.
 */
class LteUeMeasurementsPiecewiseTestCase1 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param config LteRrcSap::ReportConfigEutra
   * \param expectedTime the expected time
   * \param expectedRsrp the expected RSRP
   */
  LteUeMeasurementsPiecewiseTestCase1 (std::string name,
                                       LteRrcSap::ReportConfigEutra config,
                                       std::vector<Time> expectedTime,
                                       std::vector<uint8_t> expectedRsrp);

  virtual ~LteUeMeasurementsPiecewiseTestCase1 ();

  /**
   * \brief Triggers when eNodeB receives measurement report from UE, then
   *        perform verification on it.
   *
   * The trigger is set up beforehand by connecting to the
   * `LteUeRrc::RecvMeasurementReport` trace source.
   *
   * Verification consists of checking whether the report carries the right
   * value of RSRP or not, and whether it occurs at the expected time or not.
   *
   * \param context the context
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti  the RNTI
   * \param report LteRrcSap::MeasurementReport
   */
  void RecvMeasurementReportCallback (std::string context, uint64_t imsi,
                                      uint16_t cellId, uint16_t rnti,
                                      LteRrcSap::MeasurementReport report);

private:
  /**
   * \brief Setup the simulation with the intended UE measurement reporting
   *        configuration, run it, and connect the
   *        `RecvMeasurementReportCallback` function to the
   *        `LteUeRrc::RecvMeasurementReport` trace source.
   */
  virtual void DoRun ();

  /**
   * \brief Runs at the end of the simulation, verifying that all expected
   *        measurement reports have been examined.
   */
  virtual void DoTeardown ();

  /// Teleport very near function
  void TeleportVeryNear ();
  /// Teleport near function
  void TeleportNear ();
  /// Teleport far function
  void TeleportFar ();
  /// Teleport far function
  void TeleportVeryFar ();

  /**
   * \brief The active report triggering configuration.
   */
  LteRrcSap::ReportConfigEutra m_config;

  /**
   * \brief The list of expected time when measurement reports are received by
   *        eNodeB.
   */
  std::vector<Time> m_expectedTime;

  /**
   * \brief The list of expected values of RSRP (in 3GPP range unit) from the
   *        measurement reports received.
   */
  std::vector<uint8_t> m_expectedRsrp;

  /**
   * \brief Pointer to the element of `m_expectedTime` which is expected to
   *        occur next in the simulation.
   */
  std::vector<Time>::iterator m_itExpectedTime;

  /**
   * \brief Pointer to the element of `m_expectedRsrp` which is expected to
   *        occur next in the simulation.
   */
  std::vector<uint8_t>::iterator m_itExpectedRsrp;

  /**
   * \brief The measurement identity being tested. Measurement reports with
   *        different measurement identity (e.g. from handover algorithm) will
   *        be ignored.
   */
  uint8_t m_expectedMeasId;

  Ptr<MobilityModel> m_ueMobility; ///< the mobility model

}; // end of class LteUeMeasurementsPiecewiseTestCase1



// ===== LTE-UE-MEASUREMENTS-PIECEWISE-2 TEST SUITE ======================== //


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for generating calls to UE measurements test case
 *        ns3::LteUeMeasurementsPiecewiseTestCase2.
 */
class LteUeMeasurementsPiecewiseTestSuite2 : public TestSuite
{
public:
  LteUeMeasurementsPiecewiseTestSuite2 ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Testing UE measurements in LTE with simulation of 2 eNodeB and 1 UE in
 *        piecewise configuration and 240 ms report interval.
 *        Here is intenisvely test events A1 and A2. A1 event should be triggered
 *        when the serving cell becomes better than a given threshold. A2 shall 
 *        be triggered when the serving cell becomes worse than threshold. A3 
 *        event is triggered when the neighbour becomes offset better than the PCell.
 *        In this test UE is being transported several times during the simulation 
 *        duration. Then the test checks if measurement report contains correct 
 *        RSRP and RSRQ result, than if CQI info is correctly generated, if the 
 *        reporting occurs at the correct time stamp in the simulation and according 
 *        to the measurements configuration. It also checks if the reported values 
 *        of RSRP and RSRQ are equal to expected reference values.
 *        
 */
class LteUeMeasurementsPiecewiseTestCase2 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param config LteRrcSap::ReportConfigEutra
   * \param expectedTime the expected time
   * \param expectedRsrp the expected RSRP
   */
  LteUeMeasurementsPiecewiseTestCase2 (std::string name,
                                       LteRrcSap::ReportConfigEutra config,
                                       std::vector<Time> expectedTime,
                                       std::vector<uint8_t> expectedRsrp);

  virtual ~LteUeMeasurementsPiecewiseTestCase2 ();

  /**
   * \brief Triggers when eNodeB receives measurement report from UE, then
   *        perform verification on it.
   *
   * The trigger is set up beforehand by connecting to the
   * `LteUeRrc::RecvMeasurementReport` trace source.
   *
   * Verification consists of checking whether the report carries the right
   * value of RSRP or not, and whether it occurs at the expected time or not.
   *
   * \param context the context
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti  the RNTI
   * \param report LteRrcSap::MeasurementReport
   */
  void RecvMeasurementReportCallback (std::string context, uint64_t imsi,
                                      uint16_t cellId, uint16_t rnti,
                                      LteRrcSap::MeasurementReport report);

private:
  /**
   * \brief Setup the simulation with the intended UE measurement reporting
   *        configuration, run it, and connect the
   *        `RecvMeasurementReportCallback` function to the
   *        `LteUeRrc::RecvMeasurementReport` trace source.
   */
  virtual void DoRun ();

  /**
   * \brief Runs at the end of the simulation, verifying that all expected
   *        measurement reports have been examined.
   */
  virtual void DoTeardown ();

  /// Teleport very near function
  void TeleportVeryNear ();
  /// Teleport near function
  void TeleportNear ();
  /// Teleport far function
  void TeleportFar ();
  /// Teleport very far function
  void TeleportVeryFar ();

  /**
   * \brief The active report triggering configuration.
   */
  LteRrcSap::ReportConfigEutra m_config;

  /**
   * \brief The list of expected time when measurement reports are received by
   *        eNodeB.
   */
  std::vector<Time> m_expectedTime;

  /**
   * \brief The list of expected values of RSRP (in 3GPP range unit) from the
   *        measurement reports received.
   */
  std::vector<uint8_t> m_expectedRsrp;

  /**
   * \brief Pointer to the element of `m_expectedTime` which is expected to
   *        occur next in the simulation.
   */
  std::vector<Time>::iterator m_itExpectedTime;

  /**
   * \brief Pointer to the element of `m_expectedRsrp` which is expected to
   *        occur next in the simulation.
   */
  std::vector<uint8_t>::iterator m_itExpectedRsrp;

  /**
   * \brief The measurement identity being tested. Measurement reports with
   *        different measurement identity (e.g. from handover algorithm) will
   *        be ignored.
   */
  uint8_t m_expectedMeasId;

  Ptr<MobilityModel> m_ueMobility; ///< the mobility model

}; // end of class LteUeMeasurementsPiecewiseTestCase2

// ===== LTE-UE-MEASUREMENTS-PIECEWISE-3 TEST SUITE ======================== //


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for generating calls to UE measurements test case
 *        ns3::LteUeMeasurementsPiecewiseTestCase3.
 */
class LteUeMeasurementsPiecewiseTestSuite3 : public TestSuite
{
public:
  LteUeMeasurementsPiecewiseTestSuite3 ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Testing UE measurements in LTE with simulation of 3 eNodeB and 1 UE in
 *        piecewise configuration and 240 ms report interval.
 *        This test is to cover a corner case using event A4, which is not
 *        covered by LteUeMeasurementsPiecewiseTestCase1 and
 *        LteUeMeasurementsPiecewiseTestCase2. In this case, we test that the UE
 *        measurements at eNB are arriving 240 ms apart. Please note, the
 *        scenario simulated is engineered to specifically test the corner case
 *        in which whenever a new neighbour fulfils the entry condition
 *        for event A4, the UE RRC calls VarMeasReportListAdd method to include
 *        the new cell id in cellsTriggeredList, and then it schedules the
 *        SendMeasurementReport for periodic reporting. However, if the UE has
 *        already started the periodic reporting, scheduling the
 *        SendMeasurementReport method again causes following buggy behaviors:
 *
 *        1. It generates an intermediate measurement event, which then leads
 *        to parallel intermediate measurement reports from a UE to its eNB.
 *
 *        2. The old EvenId is overwritten by the new EventId stored in
 *        VarMeasReportList. This makes us lose control over the old EventId
 *        and it is impossible to cancel its events later on.
 *
 *        These buggy behaviors generated an issue reported in
 *        https://gitlab.com/nsnam/ns-3-dev/-/issues/224, where a UE try to
 *        send measurement reports after the RLF, even though all the measurement
 *        events are properly cancelled upon detecting RLF.
 *
 *        The correct behaviour should be that if a UE has already started the
 *        periodic reporting, and once a new neighbour fulfils the entry
 *        condition, we just need to add its cell id in cellsTriggeredList,
 *        without scheduling a new periodic event.
 *
 */
class LteUeMeasurementsPiecewiseTestCase3 : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param config LteRrcSap::ReportConfigEutra
   * \param expectedTime the expected time
   */
  LteUeMeasurementsPiecewiseTestCase3 (std::string name,
                                       LteRrcSap::ReportConfigEutra config,
                                       std::vector<Time> expectedTime);

  virtual ~LteUeMeasurementsPiecewiseTestCase3 ();

  /**
   * \brief Triggers when eNodeB receives measurement report from UE, then
   *        perform verification on it.
   *
   * The trigger is set up beforehand by connecting to the
   * `LteUeRrc::RecvMeasurementReport` trace source.
   *
   * Verification consists of checking whether the report carries the right
   * value of RSRP or not, and whether it occurs at the expected time or not.
   *
   * \param context the context
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti  the RNTI
   * \param report LteRrcSap::MeasurementReport
   */
  void RecvMeasurementReportCallback (std::string context, uint64_t imsi,
                                      uint16_t cellId, uint16_t rnti,
                                      LteRrcSap::MeasurementReport report);

private:
  /**
   * \brief Setup the simulation with the intended UE measurement reporting
   *        configuration, run it, and connect the
   *        `RecvMeasurementReportCallback` function to the
   *        `LteUeRrc::RecvMeasurementReport` trace source.
   */
  virtual void DoRun ();

  /**
   * \brief Runs at the end of the simulation, verifying that all expected
   *        measurement reports have been examined.
   */
  virtual void DoTeardown ();


  /// Teleport the eNb near function
  void TeleportEnbNear ();

  /**
   * \brief The active report triggering configuration.
   */
  LteRrcSap::ReportConfigEutra m_config;

  /**
   * \brief The list of expected time when measurement reports are received by
   *        eNodeB.
   */
  std::vector<Time> m_expectedTime;

  /**
   * \brief Pointer to the element of `m_expectedTime` which is expected to
   *        occur next in the simulation.
   */
  std::vector<Time>::iterator m_itExpectedTime;

  /**
   * \brief The measurement identity being tested. Measurement reports with
   *        different measurement identity (e.g. from handover algorithm) will
   *        be ignored.
   */
  uint8_t m_expectedMeasId;

  Ptr<MobilityModel> m_enbMobility; ///< the mobility model

}; // end of class LteUeMeasurementsPiecewiseTestCase3




// ===== LTE-UE-MEASUREMENTS-HANDOVER TEST SUITE =========================== //


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for generating calls to UE measurements test case
 *        ns3::LteUeMeasurementsHandoverTestCase.
 */
class LteUeMeasurementsHandoverTestSuite : public TestSuite
{
public:
  LteUeMeasurementsHandoverTestSuite ();
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Testing UE measurements in LTE with simulation of 2 eNodeB and 1 UE in
 *        a handover configuration.
 *
 * The simulation will run for the specified duration, while the handover
 * command will be issued exactly at the middle of simulation. Handover 
 * test covers four different scenarios: switch from event A1 to event A2, 
 * from event A2 to event A1, from event A3 to event A4, from event A4 to event 
 * A3, from event A2 to event A3, from event A3 to event A2, from event A4 to 
 * event A5, from event A5 to event A4. Additionally, it tests different 
 * handover threshold configurations, A2 threshold difference, A3 offset difference,
 * A4 and A5 threshold difference. Finally, different handover is tested for 
 * different time-to-trigger (TTT) configurations.
 * 
 */
class LteUeMeasurementsHandoverTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param sourceConfigList std::list<LteRrcSap::ReportConfigEutra>
   * \param targetConfigList std::list<LteRrcSap::ReportConfigEutra>
   * \param expectedTime the expected time
   * \param expectedRsrp the expected RSRP
   * \param duration the duration
   */
  LteUeMeasurementsHandoverTestCase (std::string name,
                                     std::list<LteRrcSap::ReportConfigEutra> sourceConfigList,
                                     std::list<LteRrcSap::ReportConfigEutra> targetConfigList,
                                     std::vector<Time> expectedTime,
                                     std::vector<uint8_t> expectedRsrp,
                                     Time duration);

  virtual ~LteUeMeasurementsHandoverTestCase ();

  /**
   * \brief Triggers when either one of the eNodeBs receives measurement report
   *        from UE, then perform verification on it.
   *
   * The trigger is set up beforehand by connecting to the
   * `LteUeRrc::RecvMeasurementReport` trace source.
   *
   * Verification consists of checking whether the report carries the right
   * value of RSRP or not, and whether it occurs at the expected time or not.
   *
   * \param context the context
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti  the RNTI
   * \param report LteRrcSap::MeasurementReport
   */
  void RecvMeasurementReportCallback (std::string context, uint64_t imsi,
                                      uint16_t cellId, uint16_t rnti,
                                      LteRrcSap::MeasurementReport report);

private:
  /**
   * \brief Setup the simulation with the intended UE measurement reporting
   *        configuration, run it, and connect the
   *        `RecvMeasurementReportCallback` function to the
   *        `LteUeRrc::RecvMeasurementReport` trace source.
   */
  virtual void DoRun ();

  /**
   * \brief Runs at the end of the simulation, verifying that all expected
   *        measurement reports have been examined.
   */
  virtual void DoTeardown ();

  /**
   * \brief The list of active report triggering configuration for the source
   *        eNodeB.
   */
  std::list<LteRrcSap::ReportConfigEutra> m_sourceConfigList;

  /**
   * \brief The list of active report triggering configuration for the target
   *        eNodeB.
   */
  std::list<LteRrcSap::ReportConfigEutra> m_targetConfigList;

  /**
   * \brief The list of expected time when measurement reports are received by
   *        eNodeB.
   */
  std::vector<Time> m_expectedTime;

  /**
   * \brief The list of expected values of RSRP (in 3GPP range unit) from the
   *        measurement reports received.
   */
  std::vector<uint8_t> m_expectedRsrp;

  /**
   * \brief Pointer to the element of `m_expectedTime` which is expected to
   *        occur next in the simulation.
   */
  std::vector<Time>::iterator m_itExpectedTime;

  /**
   * \brief Pointer to the element of `m_expectedRsrp` which is expected to
   *        occur next in the simulation.
   */
  std::vector<uint8_t>::iterator m_itExpectedRsrp;

  /**
   * \brief Duration of simulation.
   */
  Time m_duration;

  /**
   * \brief The list of measurement identities being tested in the source cell.
   *        Measurement reports with different measurement identity (e.g. from
   *        handover algorithm and ANR) will be ignored.
   */
  std::set<uint8_t> m_expectedSourceCellMeasId;

  /**
   * \brief The list of measurement identities being tested in the target cell.
   *        Measurement reports with different measurement identity (e.g. from
   *        handover algorithm and ANR) will be ignored.
   */
  std::set<uint8_t> m_expectedTargetCellMeasId;

}; // end of class LteUeMeasurementsHandoverTestCase

#endif /* LTE_TEST_UE_MEASUREMENTS_H */
