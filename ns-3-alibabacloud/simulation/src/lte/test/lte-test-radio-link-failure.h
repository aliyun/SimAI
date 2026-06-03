/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Fraunhofer ESK
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
 * Author: Vignesh Babu <ns3-dev@esk.fraunhofer.de>
 */

#ifndef LTE_TEST_RADIO_LINK_FAILURE_H
#define LTE_TEST_RADIO_LINK_FAILURE_H


#include <ns3/test.h>
#include <ns3/nstime.h>
#include <ns3/node-container.h>
#include <ns3/net-device-container.h>
#include <ns3/vector.h>
#include <ns3/lte-ue-rrc.h>
#include <vector>


namespace ns3 {

class LteUeNetDevice;

}

using namespace ns3;


/**
 * \brief Test suite for
 *
 * \sa ns3::LteRadioLinkFailureTestCase
 */
class LteRadioLinkFailureTestSuite : public TestSuite
{
public:
  LteRadioLinkFailureTestSuite ();
};

/**
 * \ingroup lte
 *
 * \brief Testing the cell reselection procedure by UE at IDLE state
 */
class LteRadioLinkFailureTestCase : public TestCase
{
public:
  /**
   * \brief Creates an instance of the radio link failure test case.
   *
   * \param numEnbs number of eNodeBs
   * \param numUes number of UEs
   * \param simTime the simulation time
   * \param isIdealRrc if true, simulation uses Ideal RRC protocol, otherwise
   *                   simulation uses Real RRC protocol
   * \param uePositionList Position of the UEs
   * \param enbPositionList Position of the eNodeBs
   * \param ueJumpAwayPosition Vector holding the UE jump away coordinates
   * \param checkConnectedList the time at which UEs should have an active RRC connection
   */
  LteRadioLinkFailureTestCase (uint32_t numEnbs, uint32_t numUes, Time simTime, bool isIdealRrc,
                               std::vector<Vector> uePositionList, std::vector<Vector> enbPositionList,
                               Vector ueJumpAwayPosition, std::vector<Time> checkConnectedList);

  virtual ~LteRadioLinkFailureTestCase ();

private:
  /**
   * Builds the test name string based on provided parameter values
   * \param numEnbs the number of eNB nodes
   * \param numUes the number of UE nodes
   * \param isIdealRrc True if the Ideal RRC protocol is used
   * \returns the name string
   */
  std::string BuildNameString (uint32_t numEnbs, uint32_t numUes, bool isIdealRrc);
  /**
   * \brief Setup the simulation according to the configuration set by the
   *        class constructor, run it, and verify the result.
   */
  virtual void DoRun ();

  /**
   * Check connected function
   * \param ueDevice the UE device
   * \param enbDevices the ENB devices
   */
  void CheckConnected (Ptr<NetDevice> ueDevice, NetDeviceContainer enbDevices);

  /**
   * Check if the UE is in idle state
   * \param ueDevice the UE device
   * \param enbDevices the ENB devices
   */
  void CheckIdle (Ptr<NetDevice> ueDevice, NetDeviceContainer enbDevices);

  /**
   * \brief Check if the UE exist at the eNB
   * \param rnti the RNTI of the UE
   * \param enbDevice the eNB device
   * \return true if the UE exist at the eNB, otherwise false
   */
  bool CheckUeExistAtEnb (uint16_t rnti, Ptr<NetDevice> enbDevice);

  /**
   * \brief State transition callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   * \param oldState the old state
   * \param newState the new state
   */
  void UeStateTransitionCallback (std::string context, uint64_t imsi,
                                  uint16_t cellId, uint16_t rnti,
                                  LteUeRrc::State oldState, LteUeRrc::State newState);

  /**
   * \brief Connection established at UE callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   */
  void ConnectionEstablishedUeCallback (std::string context, uint64_t imsi,
                                        uint16_t cellId, uint16_t rnti);

  /**
   * \brief Connection established at eNodeB callback function
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   */
  void ConnectionEstablishedEnbCallback (std::string context, uint64_t imsi,
                                         uint16_t cellId, uint16_t rnti);

  /**
   * \brief This callback function is executed when UE context is removed at eNodeB
   * \param context the context string
   * \param imsi the IMSI
   * \param cellId the cell ID
   * \param rnti the RNTI
   */
  void ConnectionReleaseAtEnbCallback (std::string context, uint64_t imsi,
                                       uint16_t cellId, uint16_t rnti);

  /**
   * \brief This callback function is executed when UE RRC receives an in-sync or out-of-sync indication
   * \param context the context string
   * \param imsi the IMSI
   * \param rnti the RNTI
   * \param cellId the cell ID
   * \param type in-sync or out-of-sync indication
   * \param count the number of in-sync or out-of-sync indications
   */
  void PhySyncDetectionCallback (std::string context, uint64_t imsi, uint16_t rnti,
                                 uint16_t cellId, std::string type, uint8_t count);

  /**
   * \brief This callback function is executed when radio link failure is detected
   * \param context the context string
   * \param imsi the IMSI
   * \param rnti the RNTI
   * \param cellId the cell ID
   */
  void RadioLinkFailureCallback (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti);

  /**
   * \brief Jump away function
   *
   * \param UeJumpAwayPositionList A list of positions where UE would jump
   */
  void JumpAway (Vector UeJumpAwayPositionList);

  uint32_t m_numEnbs; ///<number of eNodeBs
  uint32_t m_numUes; ///<number of UEs
  Time m_simTime; ///< simulation time
  bool m_isIdealRrc; ///< whether the LTE is configured to use ideal RRC
  std::vector<Vector> m_uePositionList; ///< Position of the UEs
  std::vector<Vector> m_enbPositionList; ///< Position of the eNodeBs
  std::vector<Time> m_checkConnectedList; ///<the time at which UEs should have an active RRC connection
  Vector m_ueJumpAwayPosition; ///< Position where the UE(s) would jump

  /// The current UE RRC state.
  LteUeRrc::State m_lastState;

  bool m_radioLinkFailureDetected;  ///< true if radio link fails
  uint32_t m_numOfInSyncIndications; ///< number of in-sync indications detected
  uint32_t m_numOfOutOfSyncIndications; ///< number of out-of-sync indications detected
  Ptr<MobilityModel> m_ueMobility; ///< UE mobility model

}; // end of class LteRadioLinkFailureTestCase

#endif /* LTE_TEST_RADIO_LINK_FAILURE_H */
