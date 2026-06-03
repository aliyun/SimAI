/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Alexander Krotov
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 *
 */

#ifndef LTE_TEST_PRIMARY_CELL_CHANGE_H
#define LTE_TEST_PRIMARY_CELL_CHANGE_H

#include <ns3/test.h>
#include <ns3/nstime.h>
#include <ns3/node-container.h>
#include <ns3/vector.h>
#include <ns3/lte-ue-rrc.h>
#include <vector>

using namespace ns3;

/**
 * \brief Test suite for executing the primary cell change test cases.
 *
 * \sa ns3::LtePrimaryCellChangeTestCase
 */
class LtePrimaryCellChangeTestSuite : public TestSuite
{
public:
  LtePrimaryCellChangeTestSuite ();
};




/**
 * \ingroup lte
 *
 * \brief Testing the handover procedure with multiple component carriers.
 */
class LtePrimaryCellChangeTestCase : public TestCase
{
public:
  /**
   * \brief Creates an instance of the initial cell selection test case.
   * \param name name of this test
   * \param isIdealRrc if true, simulation uses Ideal RRC protocol, otherwise
   *                   simulation uses Real RRC protocol
   * \param rngRun the number of run to be used by the random number generator
   * \param numberOfComponentCarriers number of component carriers
   * \param sourceComponentCarrier source component carrier
   * \param targetComponentCarrier target component carrier
   *
   * If sourceComponentCarrier or targetComponentCarrier is greater than
   * number of component carriers, it identifies component carrier on second eNB.
   */
  LtePrimaryCellChangeTestCase (std::string name, bool isIdealRrc, int64_t rngRun,
                                uint8_t numberOfComponentCarriers,
                                uint8_t sourceComponentCarrier,
                                uint8_t targetComponentCarrier);

  virtual ~LtePrimaryCellChangeTestCase ();

private:
  /**
   * \brief Setup the simulation according to the configuration set by the
   *        class constructor, run it, and verify the result.
   */
  virtual void DoRun ();

  /**
   * \brief State transition callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   * \param oldState the old state
   * \param newState the new state
   */
  void StateTransitionCallback (std::string context, uint64_t imsi,
                                uint16_t cellId, uint16_t rnti,
                                LteUeRrc::State oldState, LteUeRrc::State newState);
  /**
   * \brief Initial cell selection end ok callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   */
  void InitialPrimaryCellChangeEndOkCallback (std::string context, uint64_t imsi,
                                          uint16_t cellId);
  /**
   * \brief Connection established callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   */
  void ConnectionEstablishedCallback (std::string context, uint64_t imsi,
                                      uint16_t cellId, uint16_t rnti);

  bool m_isIdealRrc; ///< whether the LTE is configured to use ideal RRC
  int64_t m_rngRun; ///< rng run
  uint8_t m_numberOfComponentCarriers; ///< number of component carriers
  uint8_t m_sourceComponentCarrier; ///< source primary component carrier
  uint8_t m_targetComponentCarrier; ///< target primary component carrier

  /// The current UE RRC state.
  std::map<uint64_t, LteUeRrc::State> m_lastState;

}; // end of class LtePrimaryCellChangeTestCase

#endif /* LTE_TEST_PRIMARY_CELL_CHANGE_H */
