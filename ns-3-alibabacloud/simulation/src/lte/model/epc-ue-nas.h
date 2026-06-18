/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef EPC_UE_NAS_H
#define EPC_UE_NAS_H


#include <ns3/object.h>
#include <ns3/lte-as-sap.h>
#include <ns3/epc-tft-classifier.h>

namespace ns3 {


class EpcHelper;

class EpcUeNas : public Object
{
  /// allow MemberLteAsSapUser<EpcUeNas> class friend access
  friend class MemberLteAsSapUser<EpcUeNas>;
public:

  /** 
   * Constructor
   */
  EpcUeNas ();

  /** 
   * Destructor
   */
  virtual ~EpcUeNas ();

  // inherited from Object
  virtual void DoDispose (void);
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);


  /** 
   * 
   * \param dev the UE NetDevice
   */
  void SetDevice (Ptr<NetDevice> dev);

  /** 
   * 
   * 
   * \param imsi the unique UE identifier
   */
  void SetImsi (uint64_t imsi);

  /**
   *
   * \param csgId Closed Subscriber Group identity
   */
  void SetCsgId (uint32_t csgId);

  /**
   *
   * \return csgId Closed Subscriber Group identity
   */
  uint32_t GetCsgId () const;

  /**
   * Set the AS SAP provider to interact with the NAS entity
   *
   * \param s the AS SAP provider
   */
  void SetAsSapProvider (LteAsSapProvider* s);

  /**
   *
   *
   * \return the AS SAP user exported by this RRC
   */
  LteAsSapUser* GetAsSapUser ();

  /**
   * set the callback used to forward data packets up the stack
   *
   * \param cb the callback
   */
  void SetForwardUpCallback (Callback <void, Ptr<Packet> > cb);

  /**
   * \brief Causes NAS to tell AS to find a suitable cell and camp to it.
   *
   * \param dlEarfcn the DL frequency of the eNB
   */
  void StartCellSelection (uint32_t dlEarfcn);

  /**
   * \brief Causes NAS to tell AS to go to ACTIVE state.
   * 
   * The end result is equivalent with EMM Registered + ECM Connected states.
   */
  void Connect ();

  /**
   * \brief Causes NAS to tell AS to camp to a specific cell and go to ACTIVE
   *        state.
   * \param cellId the id of the eNB to camp on
   * \param dlEarfcn the DL frequency of the eNB
   *
   * The end result is equivalent with EMM Registered + ECM Connected states.
   * Since RRC Idle Mode cell selection is not supported yet, we force the UE
   * RRC to be camped on a specific eNB.
   */
  void Connect (uint16_t cellId, uint32_t dlEarfcn);
 
  /** 
   * instruct the NAS to disconnect
   * 
   */
  void Disconnect ();


  /** 
   * Activate an EPS bearer
   * 
   * \param bearer the characteristics of the bearer to be created
   * \param tft the TFT identifying the traffic that will go on this bearer
   */
  void ActivateEpsBearer (EpsBearer bearer, Ptr<EpcTft> tft);

  /** 
   * Enqueue an IP packet on the proper bearer for uplink transmission
   * 
   * \param p the packet
   * \param protocolNumber the protocol number of the packet
   * 
   * \return true if successful, false if an error occurred
   */
  bool Send (Ptr<Packet> p, uint16_t protocolNumber);


  /**
   * Definition of NAS states as per "LTE - From theory to practice",
   * Section 3.2.3.2 "Connection Establishment and Release"
   * 
   */
  enum State 
  {
    OFF = 0,
    ATTACHING,
    IDLE_REGISTERED,
    CONNECTING_TO_EPC,
    ACTIVE,
    NUM_STATES
  };

  /**
   * \return The current state
   */
  State GetState () const;

  /**
   * TracedCallback signature for state change events.
   *
   * \param [in] oldState The old State.
   * \param [in] newState the new State.
   */
  typedef void (* StateTracedCallback)
    (const State oldState, const State newState);
 
private:

  // LTE AS SAP methods
  /// Notify successful connection
  void DoNotifyConnectionSuccessful ();
  /// Notify connection failed
  void DoNotifyConnectionFailed ();
  /// Notify connection released
  void DoNotifyConnectionReleased ();
  /**
   * Receive data
   * \param packet the packet
   */
  void DoRecvData (Ptr<Packet> packet);

  // internal methods
  /**
   * Activate EPS Bearer
   * \param bearer the EPS bearer
   * \param tft the EPC TFT
   */
  void DoActivateEpsBearer (EpsBearer bearer, Ptr<EpcTft> tft);
  /**
   * Switch the UE RRC to the given state.
   * \param s the destination state
   */
  void SwitchToState (State s);

  /// The current UE NAS state.
  State m_state;

  /**
   * The `StateTransition` trace source. Fired upon every UE NAS state
   * transition. Exporting old state and new state.
   * \todo This should be a TracedValue
   */
  TracedCallback<State, State> m_stateTransitionCallback;

  /// The UE NetDevice.
  Ptr<NetDevice> m_device;

  /// The unique UE identifier.
  uint64_t m_imsi;

  /// Closed Subscriber Group identity.
  uint32_t m_csgId;

  /// LTE SAP provider
  LteAsSapProvider* m_asSapProvider;
  /// LTE SAP user
  LteAsSapUser* m_asSapUser;

  uint8_t m_bidCounter; ///< bid counter
  EpcTftClassifier m_tftClassifier; ///< tft classifier

  Callback <void, Ptr<Packet> > m_forwardUpCallback; ///< upward callback

  /// BearerToBeActivated structure
  struct BearerToBeActivated
  {
    EpsBearer bearer; ///< EPS bearer
    Ptr<EpcTft> tft; ///< TFT
  };

  std::list<BearerToBeActivated> m_bearersToBeActivatedList; ///< bearers to be activated list

  /**
   * bearers to be activated list maintained and to be used for reconnecting
   * an out-of-sync UE
   *
   */
  std::list<BearerToBeActivated> m_bearersToBeActivatedListForReconnection;

};


} // namespace ns3

#endif // EPC_UE_NAS_H
