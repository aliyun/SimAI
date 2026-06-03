/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 *
 */

#ifndef LTE_ENB_COMPONENT_CARRIER_MANAGER_H
#define LTE_ENB_COMPONENT_CARRIER_MANAGER_H

#include <ns3/object.h>
#include <ns3/lte-enb-rrc.h>
#include <ns3/lte-rrc-sap.h>
#include <ns3/lte-ccm-rrc-sap.h>
#include <ns3/lte-mac-sap.h>
#include <ns3/lte-enb-cmac-sap.h>
#include <ns3/lte-ccm-mac-sap.h>
#include <map>
#include <vector>

namespace ns3 {


class LteCcmRrcSapUser;
class LteCcmRrcSapProvider;
class LteMacSapUser;
class LteMacSapProvider;
class LteEnbCmacSapProvider;
class LteCcmMacSapProvider;

/**
 * \brief The class implements Component Carrier Manager (CCM) that operates
 * using the Component Carrier Manager SAP interfaces.
 *
 * CCM receives measurement reports from an eNode RRC instance and is forwarding
 * calls from RLC to MAC layer, and from MAC to RLC.
 *
 * This class is an abstract class intended to be inherited by subclasses that
 * will implement its virtual methods. The subclasses are compatible with the
 * LteEnbNetDevice class, and are accessible using namespace-based access
 * through ns-3 Config subsystem, and can be installed and configured by
 * LteHelper class.
 *
 * The communication with the eNodeB RRC instance is done through the *Component
 * Carrier Manager SAP* interface. The LteEnbComponentCarrierManager instance
 * corresponds to the "provider" part of this interface, while the eNodeB RRC
 * instance takes the role of the "user" part. The following code skeleton
 * establishes the connection between both instances:
 *
 * Ptr<LteEnbRrc> rrc = ...;
 * Ptr<LteComponentCarrierManager> ccmEnb = ...;
 * rrc->SetLteCcmRrcSapProvider (ccmEnb->GetLteCcmRrcSapProvider ());
 * ccmEnb->SetLteCcmRrcSapUser (rrc->GetLteCcmRrcSapUser ())
 *
 * Similarly, LteEnbComponentCarrierManager instance communicates with MAC, and
 * it takes the role of the "user".
 *
 * However, user rarely needs to use the above code, since it has already been
 * taken care by LteHelper::InstallEnbDevice.
 *
 * \sa LteCcmRrcSapUser, LteCcmRrcSapProvider, LteCcmMacSapUser, LteCcmMacSapProvider
 */

class LteEnbComponentCarrierManager : public Object
{
public:

  LteEnbComponentCarrierManager ();
  virtual ~LteEnbComponentCarrierManager ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * \brief Set the "user" part of the ComponentCarrier Management SAP interface that
   *        this ComponentCarrier algorithm instance will interact with.
   * \param s a reference to the "user" part of the interface, typically a
   *          member of an LteEnbRrc instance
   */
  virtual void SetLteCcmRrcSapUser (LteCcmRrcSapUser* s);

  /**
   * \brief Export the "provider" part of the ComponentCarrier Management SAP interface.
   * \return the reference to the "provider" part of the interface, typically to
   *         be kept by an LteEnbRlc instance
   */
  virtual LteCcmRrcSapProvider* GetLteCcmRrcSapProvider ();

  /**
   * \brief This function returns a pointer to the LteCcmMacSapUser interface, which
   * is used by MAC to communicate to CCM when e.g. UL buffer status report is
   * received, or to notify CCM about PRB occupancy, and similar. Functions that are
   * specific for the communication between MAC and CCM.
   *
   * \returns LteCcmMacSapUser* 
   */
  virtual LteCcmMacSapUser* GetLteCcmMacSapUser ();

  /**
   * \brief Returns the pointer to the LteMacSapProvider interface, the
   * provider of MAC, which is this new architecture served by
   * LteEnbComponentCarrierManager object which will behave as a
   * proxy, and will forward calls between to MAC objects of
   * component carriers based on the logic implemented in the
   * specific component carrier manager.
   *
   * \returns LteMacSapProvider* 
   */
  virtual LteMacSapProvider* GetLteMacSapProvider ();

  /**
   * \brief Set LteMacSapProvider interface for the MAC object of
   * the specified component carrier.
   *
   * \param componentCarrierId component carrier ID
   * \param sap the MAC SAP provider
   * \returns true if successful
   */
  virtual bool SetMacSapProvider (uint8_t componentCarrierId, LteMacSapProvider* sap);

  /**
   * \brief Set LteCcmMacSapProvider interface for the MAC object of
   * the specified component carrier. Through this interface CCM communicates with
   * MAC, e.g. it notifies MAC of the specific carrier when to scheduler UL BSR.
   *
   * \param componentCarrierId component carrier ID
   * \param sap the MAC SAP provider
   * \returns true if successful
   */
  virtual bool SetCcmMacSapProviders (uint8_t componentCarrierId, LteCcmMacSapProvider* sap);

  /**
   * \brief Sets the total number of component carriers.
   * \param noOfComponentCarriers number of component carriers
   */
  virtual void SetNumberOfComponentCarriers (uint16_t noOfComponentCarriers);


protected:

  // inherited from Object
  virtual void DoDispose ();

  /**
   * \brief Implementation of ReportUeMeas.
   * \param rnti Radio Network Temporary Identity, an integer identifying the UE
   *             where the report originates from
   * \param measResults a single report of one measurement identity
   */
  virtual void DoReportUeMeas (uint16_t rnti, LteRrcSap::MeasResults measResults) = 0;

  /**
   * \brief Structure to represent UE info
   */
  struct UeInfo {
    std::map<uint8_t, LteMacSapUser*> m_ueAttached; //!< Map from LCID to SAP of the RLC instance.
    std::map<uint8_t, LteEnbCmacSapProvider::LcInfo> m_rlcLcInstantiated; //!< Logical channel configuration per flow Id (rnti, lcid).
    uint8_t m_enabledComponentCarrier; //!< The number of enabled component carriers.
    uint8_t m_ueState; //!< RRC states of UE, e.g. CONNECTED_NORMALLY
  };

  std::map <uint16_t, UeInfo> m_ueInfo; //!< The map from RNTI to UE information.
  uint16_t m_noOfComponentCarriers; //!< The number component of carriers that are supported by this eNb.
  // pointer to RRC object for direct function calls, e.g. when CCM needs to obtain
  // a pointer to RLC object of a specific flow
  Ptr<LteEnbRrc> m_rrc; //!< A pointer to the RRC instance of this eNb.

  /*
   * This interface is used to receive API calls from the RLC instance that through
   * LteMacSapProvider interface. The component carrier manager acts a proxy. This means that all
   * RLC instances will see as in previous architecture the LteMacSapProvider interface, but the
   * actual provider in new architecture will be some of child classes of LteEnbComponentCarrierManager.
   * So, LteEnbComponentCarrierManager class will receive function calls that are meant for MAC, and
   * will forward them to the MAC of the component carriers based on the logic implemented
   * in LteComponentCarrierManager. This attribute will be initialized by using class that implements
   * LteMacSapProvider interface and class that implements LteEnbComponentCarrierManager base class
   * e.g.:EnbMacMemberLteMacSapProvider <LteEnbComponentCarrierManagerImpl>
   */
  LteMacSapProvider* m_macSapProvider;//!< A pointer to main SAP interface of the MAC instance, which is in this case handled by CCM.
  // This map is initialized in LteHelper when the Component Carrier Manager is initialized, contains
  // component carrier id and a pointer to the corresponding LteMacSapProvider interface of the
  // MAC instance
  std::map <uint8_t, LteMacSapProvider*> m_macSapProvidersMap; //!< A map of pointers to real SAP interfaces of MAC instances.
  // This map contains pointers to LteCcmMacSapProvider interfaces of the
  // MAC instances. LteCcmMacSapProvider is new interface added for the
  // communication between component carrier manager and MAC instance,
  // to allow CCM to control UL buffer status reporting, and forwarding to
  // schedulers. Before adding carrier aggregation to LTE module, MAC was
  // directly forwarding UL buffer status report to scheduler. Now is this
  // done through CCM, which decides to which MAC scheduler to forward UL BSR.
  std::map< uint8_t, LteCcmMacSapProvider*> m_ccmMacSapProviderMap; //!< A map of pointers to the SAP interfaces of CCM instance that provides the  CCM specific functionalities to MAC, i.e. ReportMacCeToScheduler.
  LteCcmMacSapUser* m_ccmMacSapUser; //!< LteCcmMacSapUser is extended version of LteMacSapUser interface. Contains functions that allow reporting of UL BSR from MAC to CCM.
  LteCcmRrcSapUser* m_ccmRrcSapUser; //!< A pointer to SAP interface of RRC instance, i.e. to configure measurements reporting for CCM.
  LteCcmRrcSapProvider* m_ccmRrcSapProvider; //!< A pointer to the SAP interface of the CCM instance to receive API calls from the eNodeB RRC instance.

}; // end of class LteEnbComponentCarrierManager


} // end of namespace ns3


#endif /* LTE_ENB_COMPONENT_CARRIER_MANAGER_H */
