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
 * Author: Lluis Parcerisa <lparcerisa@cttc.cat>
 * Modified by:
 *          Danilo Abrignani <danilo.abrignani@unibo.it> (Carrier Aggregation - GSoC 2015)
 *          Biljana Bojovic <biljana.bojovic@cttc.es> (Carrier Aggregation)
 */

#ifndef RRC_HEADER_H
#define RRC_HEADER_H

#include "ns3/header.h"

#include <bitset>
#include <string>

#include "ns3/lte-rrc-sap.h"
#include "ns3/lte-asn1-header.h"

namespace ns3 {

/**
 * \ingroup lte
 */

/**
 * This class extends Asn1Header functions, adding serialization/deserialization
 * of some Information elements defined in 3GPP TS 36.331
 */
class RrcAsn1Header : public Asn1Header
{
public:
  RrcAsn1Header ();
  /**
   * Get message type
   *
   * \returns the message type
   */
  int GetMessageType ();

protected:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  // Inherited from Asn1Header
  virtual TypeId GetInstanceTypeId (void) const;
  uint32_t Deserialize (Buffer::Iterator bIterator) = 0;
  virtual void PreSerialize (void) const = 0;


  // Auxiliary functions
  /**
   * Convert from bandwidth (in RBs) to ENUMERATED value
   *
   * \param bandwidth Bandwidth in RBs: 6, 15, 25, 50, 75, 100
   * \returns ENUMERATED value: 0, 1, 2, 3, 4, 5
   */
  int BandwidthToEnum (uint16_t bandwidth) const;
  /**
   * Convert from ENUMERATED value to bandwidth (in RBs)
   *
   * \param n ENUMERATED value: 0, 1, 2, 3, 4, 5
   * \returns bandwidth Bandwidth in RBs: 6, 15, 25, 50, 75, 100
   */
  uint16_t EnumToBandwidth (int n) const;


  // Serialization functions
  /**
   * Serialize SRB to add mod list function
   *
   * \param srbToAddModList std::list<LteRrcSap::SrbToAddMod>
   */
  void SerializeSrbToAddModList (std::list<LteRrcSap::SrbToAddMod> srbToAddModList) const;
  /**
   * Serialize DRB to add mod list function
   *
   * \param drbToAddModList std::list<LteRrcSap::SrbToAddMod>
   */
  void SerializeDrbToAddModList (std::list<LteRrcSap::DrbToAddMod> drbToAddModList) const;
  /**
   * Serialize logicala channel config function
   *
   * \param logicalChannelConfig LteRrcSap::LogicalChannelConfig
   */
  void SerializeLogicalChannelConfig (LteRrcSap::LogicalChannelConfig logicalChannelConfig) const;
  /**
   * Serialize radio resource config function
   *
   * \param radioResourceConfigDedicated LteRrcSap::RadioResourceConfigDedicated
   */
  void SerializeRadioResourceConfigDedicated (LteRrcSap::RadioResourceConfigDedicated radioResourceConfigDedicated) const;
  /**
   * Serialize physical config dedicated function
   *
   * \param physicalConfigDedicated LteRrcSap::PhysicalConfigDedicated
   */
  void SerializePhysicalConfigDedicated (LteRrcSap::PhysicalConfigDedicated physicalConfigDedicated) const;
  /**
   * Serialize physical config dedicated function
   *
   * \param pcdsc LteRrcSap::PhysicalConfigDedicatedSCell
   */
  void SerializePhysicalConfigDedicatedSCell (LteRrcSap::PhysicalConfigDedicatedSCell pcdsc) const;
  /**
   * Serialize system information block type 1 function
   *
   * \param systemInformationBlockType1 LteRrcSap::SystemInformationBlockType1
   */
  void SerializeSystemInformationBlockType1 (LteRrcSap::SystemInformationBlockType1 systemInformationBlockType1) const;
  /**
   * Serialize system information block type 2 function
   *
   * \param systemInformationBlockType2 LteRrcSap::SystemInformationBlockType2
   */
  void SerializeSystemInformationBlockType2 (LteRrcSap::SystemInformationBlockType2 systemInformationBlockType2) const;
  /**
   * Serialize system information block type 2 function
   *
   * \param radioResourceConfigCommon LteRrcSap::RadioResourceConfigCommon
   */
  void SerializeRadioResourceConfigCommon (LteRrcSap::RadioResourceConfigCommon radioResourceConfigCommon) const;
  /**
   * Serialize radio resource config common SIB function
   *
   * \param radioResourceConfigCommonSib LteRrcSap::RadioResourceConfigCommonSib
   */
  void SerializeRadioResourceConfigCommonSib (LteRrcSap::RadioResourceConfigCommonSib radioResourceConfigCommonSib) const;
  /**
   * Serialize measure results function
   *
   * \param measResults LteRrcSap::MeasResults
   */
  void SerializeMeasResults (LteRrcSap::MeasResults measResults) const;
  /**
   * Serialize PLMN identity function
   *
   * \param plmnId the PLMN ID
   */
  void SerializePlmnIdentity (uint32_t plmnId) const;
  /**
   * Serialize RACH config common function
   *
   * \param rachConfigCommon LteRrcSap::RachConfigCommon
   */
  void SerializeRachConfigCommon (LteRrcSap::RachConfigCommon rachConfigCommon) const;
  /**
   * Serialize measure config function
   *
   * \param measConfig LteRrcSap::MeasConfig
   */
  void SerializeMeasConfig (LteRrcSap::MeasConfig measConfig) const;
  /**
   * Serialize non critical extension config function
   *
   * \param nonCriticalExtensionConfiguration LteRrcSap::NonCriticalExtensionConfiguration
   */
  void SerializeNonCriticalExtensionConfiguration (LteRrcSap::NonCriticalExtensionConfiguration nonCriticalExtensionConfiguration) const;
  /**
   * Serialize radio resource config common SCell function
   *
   * \param rrccsc LteRrcSap::RadioResourceConfigCommonSCell
   */
  void SerializeRadioResourceConfigCommonSCell (LteRrcSap::RadioResourceConfigCommonSCell rrccsc) const;
  /**
   * Serialize radio resource dedicated SCell function
   *
   * \param rrcdsc LteRrcSap::RadioResourceConfigDedicatedSCell
   */
  void SerializeRadioResourceDedicatedSCell (LteRrcSap::RadioResourceConfigDedicatedSCell rrcdsc) const;
  /**
   * Serialize Q offset range function
   *
   * \param qOffsetRange q offset range
   */
  void SerializeQoffsetRange (int8_t qOffsetRange) const;
  /**
   * Serialize threshold eutra function
   *
   * \param thresholdEutra LteRrcSap::ThresholdEutra
   */
  void SerializeThresholdEutra (LteRrcSap::ThresholdEutra thresholdEutra) const;
  
  // Deserialization functions
  /**
   * Deserialize DRB to add mod list function
   *
   * \param drbToAddModLis std::list<LteRrcSap::DrbToAddMod> *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeDrbToAddModList (std::list<LteRrcSap::DrbToAddMod> *drbToAddModLis, Buffer::Iterator bIterator);
  /**
   * Deserialize SRB to add mod list function
   *
   * \param srbToAddModList std::list<LteRrcSap::SrbToAddMod> *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeSrbToAddModList (std::list<LteRrcSap::SrbToAddMod> *srbToAddModList, Buffer::Iterator bIterator);
  /**
   * Deserialize logical channel config function
   *
   * \param logicalChannelConfig LteRrcSap::LogicalChannelConfig *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeLogicalChannelConfig (LteRrcSap::LogicalChannelConfig *logicalChannelConfig, Buffer::Iterator bIterator);
  /**
   * Deserialize radio resource config dedicated function
   *
   * \param radioResourceConfigDedicated LteRrcSap::RadioResourceConfigDedicated *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeRadioResourceConfigDedicated (LteRrcSap::RadioResourceConfigDedicated *radioResourceConfigDedicated, Buffer::Iterator bIterator);
  /**
   * Deserialize physical config dedicated function
   *
   * \param physicalConfigDedicated LteRrcSap::PhysicalConfigDedicated *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializePhysicalConfigDedicated (LteRrcSap::PhysicalConfigDedicated *physicalConfigDedicated, Buffer::Iterator bIterator);
  /**
   * Deserialize system information block type 1 function
   *
   * \param systemInformationBlockType1 LteRrcSap::SystemInformationBlockType1 *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeSystemInformationBlockType1 (LteRrcSap::SystemInformationBlockType1 *systemInformationBlockType1, Buffer::Iterator bIterator);
  /**
   * Deserialize system information block type 2 function
   *
   * \param systemInformationBlockType2 LteRrcSap::SystemInformationBlockType2 *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeSystemInformationBlockType2 (LteRrcSap::SystemInformationBlockType2 *systemInformationBlockType2, Buffer::Iterator bIterator);
  /**
   * Deserialize radio resource config common function
   *
   * \param radioResourceConfigCommon LteRrcSap::RadioResourceConfigCommon *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeRadioResourceConfigCommon (LteRrcSap::RadioResourceConfigCommon *radioResourceConfigCommon, Buffer::Iterator bIterator);
  /**
   * Deserialize radio resource config common SIB function
   *
   * \param radioResourceConfigCommonSib LteRrcSap::RadioResourceConfigCommonSib *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeRadioResourceConfigCommonSib (LteRrcSap::RadioResourceConfigCommonSib *radioResourceConfigCommonSib, Buffer::Iterator bIterator);
  /**
   * Deserialize measure results function
   *
   * \param measResults LteRrcSap::MeasResults *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeMeasResults (LteRrcSap::MeasResults *measResults, Buffer::Iterator bIterator);
  /**
   * Deserialize PLMN identity function
   *
   * \param plmnId the PLMN ID
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializePlmnIdentity (uint32_t *plmnId, Buffer::Iterator bIterator);
  /**
   * Deserialize RACH config common function
   *
   * \param rachConfigCommon LteRrcSap::RachConfigCommon *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeRachConfigCommon (LteRrcSap::RachConfigCommon * rachConfigCommon, Buffer::Iterator bIterator);
  /**
   * Deserialize measure config function
   *
   * \param measConfig LteRrcSap::MeasConfig *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeMeasConfig (LteRrcSap::MeasConfig * measConfig, Buffer::Iterator bIterator);
  /**
   * Deserialize Qoffset range function
   *
   * \param qOffsetRange Qoffset range
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeQoffsetRange (int8_t * qOffsetRange, Buffer::Iterator bIterator);
  /**
   * Deserialize threshold eutra function
   *
   * \param thresholdEutra LteRrcSap::ThresholdEutra *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeThresholdEutra (LteRrcSap::ThresholdEutra * thresholdEutra, Buffer::Iterator bIterator);
  /**
   * Deserialize non critical extension config function
   *
   * \param nonCriticalExtension LteRrcSap::NonCriticalExtensionConfiguration *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeNonCriticalExtensionConfig (LteRrcSap::NonCriticalExtensionConfiguration * nonCriticalExtension, Buffer::Iterator bIterator);
  /**
   * Deserialize cell identification function
   *
   * \param ci LteRrcSap::CellIdentification *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeCellIdentification (LteRrcSap::CellIdentification * ci, Buffer::Iterator bIterator);
  /**
   * Deserialize radio resource config common SCell function
   *
   * \param rrccsc LteRrcSap::RadioResourceConfigCommonSCell *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeRadioResourceConfigCommonSCell (LteRrcSap::RadioResourceConfigCommonSCell * rrccsc, Buffer::Iterator bIterator);
  /**
   * Deserialize radio resource config dedicated SCell function
   *
   * \param rrcdsc LteRrcSap::RadioResourceConfigDedicatedSCell *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeRadioResourceConfigDedicatedSCell (LteRrcSap::RadioResourceConfigDedicatedSCell * rrcdsc, Buffer::Iterator bIterator);
  /**
   * Deserialize physical config dedicated SCell function
   *
   * \param pcdsc LteRrcSap::PhysicalConfigDedicatedSCell *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializePhysicalConfigDedicatedSCell (LteRrcSap::PhysicalConfigDedicatedSCell *pcdsc, Buffer::Iterator bIterator);

  /**
   * This function prints the object, for debugging purposes.
   * @param os The output stream to use (i.e. std::cout)
   */
  void Print (std::ostream &os) const;
  /**
   * This function prints RadioResourceConfigDedicated IE, for debugging purposes.
   * @param os The output stream to use (i.e. std::cout)
   * @param radioResourceConfigDedicated The information element to be printed
   */
  void Print (std::ostream &os, LteRrcSap::RadioResourceConfigDedicated radioResourceConfigDedicated) const;

  /// Stores RRC message type, according to 3GPP TS 36.331
  int m_messageType;
};


/**
 * This class only serves to discriminate which message type has been received
 * in uplink (ue to eNb) for channel DCCH
 */
class RrcUlDcchMessage : public RrcAsn1Header
{
public:
  RrcUlDcchMessage ();
  ~RrcUlDcchMessage ();

  // Inherited from RrcAsn1Header
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;
  void PreSerialize () const;

protected:
  /**
   * Serialize UL DCCH message function
   *
   * \param msgType message type
   */
  void SerializeUlDcchMessage (int msgType) const;
  /**
   * Deserialize UL DCCH message function
   *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeUlDcchMessage (Buffer::Iterator bIterator);
};

/**
 * This class only serves to discriminate which message type has been received
 * in downlink (eNb to ue) for channel DCCH
 */
class RrcDlDcchMessage : public RrcAsn1Header
{
public:
  RrcDlDcchMessage ();
  ~RrcDlDcchMessage ();

  // Inherited from RrcAsn1Header
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;
  void PreSerialize () const;

protected:
  /**
   * Serialize DL DCCH message function
   *
   * \param msgType message type
   */
  void SerializeDlDcchMessage (int msgType) const;
  /**
   * Deserialize DL DCCH message function
   *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeDlDcchMessage (Buffer::Iterator bIterator);
};

/**
 * This class only serves to discriminate which message type has been received
 * in uplink (ue to eNb) for channel CCCH
 */
class RrcUlCcchMessage : public RrcAsn1Header
{
public:
  RrcUlCcchMessage ();
  ~RrcUlCcchMessage ();

  // Inherited from RrcAsn1Header 
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;
  void PreSerialize () const;

protected:
  /**
   * Serialize UL CCCH message function
   *
   * \param msgType message type
   */
  void SerializeUlCcchMessage (int msgType) const;
  /**
   * Deserialize DL CCCH message function
   *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeUlCcchMessage (Buffer::Iterator bIterator);
};

/**
 * This class only serves to discriminate which message type has been received
 * in downlink (eNb to ue) for channel CCCH
 */
class RrcDlCcchMessage : public RrcAsn1Header
{
public:
  RrcDlCcchMessage ();
  ~RrcDlCcchMessage ();

  // Inherited from RrcAsn1Header 
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;
  void PreSerialize () const;

protected:
  /**
   * Serialize DL CCCH message function
   *
   * \param msgType message type
   */
  void SerializeDlCcchMessage (int msgType) const;
  /**
   * Deserialize DL CCCH message function
   *
   * \param bIterator buffer iterator
   * \returns buffer iterator
   */
  Buffer::Iterator DeserializeDlCcchMessage (Buffer::Iterator bIterator);
};

/**
* This class manages the serialization/deserialization of RrcConnectionRequest IE
*/
class RrcConnectionRequestHeader : public RrcUlCcchMessage
{
public:
  RrcConnectionRequestHeader ();
  ~RrcConnectionRequestHeader ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
   * Receives a RrcConnectionRequest IE and stores the contents into the class attributes
   * @param msg The information element to parse
   */
  void SetMessage (LteRrcSap::RrcConnectionRequest msg);

  /**
   * Returns a RrcConnectionRequest IE from the values in the class attributes
   * @return A RrcConnectionRequest, as defined in LteRrcSap
   */
  LteRrcSap::RrcConnectionRequest GetMessage () const;

  /**
   * Get MMEC attribute
   * @return m_mmec attribute
   */
  std::bitset<8> GetMmec () const;

  /**
   * Get M-TMSI attribute
   * @return m_tmsi attribute
   */
  std::bitset<32> GetMtmsi () const;

private:
  std::bitset<8> m_mmec; ///< MMEC
  std::bitset<32> m_mTmsi; ///< TMSI
  /// EstablishmentCause enumeration
  enum
  {
    EMERGENCY = 0, HIGHPRIORITYACCESS, MT_ACCESS,
    MO_SIGNALLING, MO_DATA, SPARE3, SPARE2, SPARE1
  } m_establishmentCause; ///< the establishent cause
  std::bitset<1> m_spare; ///< spare bit
};

/**
* This class manages the serialization/deserialization of RrcConnectionSetup IE
*/
class RrcConnectionSetupHeader : public RrcDlCcchMessage
{
public:
  RrcConnectionSetupHeader ();
  ~RrcConnectionSetupHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionSetup IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */
  void SetMessage (LteRrcSap::RrcConnectionSetup msg);

  /**
  * Returns a RrcConnectionSetup IE from the values in the class attributes
  * @return A RrcConnectionSetup, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionSetup GetMessage () const;

  /**
  * Getter for m_rrcTransactionIdentifier
  * @return m_rrcTransactionIdentifier
  */
  uint8_t GetRrcTransactionIdentifier () const;

  /**
  * Getter for m_radioResourceConfigDedicated
  * @return m_radioResourceConfigDedicated
  */
  LteRrcSap::RadioResourceConfigDedicated GetRadioResourceConfigDedicated () const; 

  /**
  * Gets m_radioResourceConfigDedicated.havePhysicalConfigDedicated
  * @return m_radioResourceConfigDedicated.havePhysicalConfigDedicated
  */
  bool HavePhysicalConfigDedicated () const;

  /**
  * Gets m_radioResourceConfigDedicated.physicalConfigDedicated
  * @return m_radioResourceConfigDedicated.physicalConfigDedicated
  */
  LteRrcSap::PhysicalConfigDedicated GetPhysicalConfigDedicated () const;

  /**
  * Gets m_radioResourceConfigDedicated.srbToAddModList
  * @return m_radioResourceConfigDedicated.srbToAddModList
  */
  std::list<LteRrcSap::SrbToAddMod> GetSrbToAddModList () const;

  /**
  * Gets m_radioResourceConfigDedicated.drbToAddModList
  * @return m_radioResourceConfigDedicated.drbToAddModList
  */
  std::list<LteRrcSap::DrbToAddMod> GetDrbToAddModList () const;

  /**
  * Gets m_radioResourceConfigDedicated.drbToReleaseList
  * @return m_radioResourceConfigDedicated.drbToReleaseList
  */
  std::list<uint8_t> GetDrbToReleaseList () const;

private:
  uint8_t m_rrcTransactionIdentifier; ///< RRC transaction identifier
  mutable LteRrcSap::RadioResourceConfigDedicated m_radioResourceConfigDedicated; ///< radio resource config dedicated
};

/**
* This class manages the serialization/deserialization of RrcConnectionSetupComplete IE
*/
class RrcConnectionSetupCompleteHeader : public RrcUlDcchMessage
{
public:
  RrcConnectionSetupCompleteHeader ();
  ~RrcConnectionSetupCompleteHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionSetupCompleted IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */
  void SetMessage (LteRrcSap::RrcConnectionSetupCompleted msg);

  /**
  * Returns a RrcConnectionSetupCompleted IE from the values in the class attributes
  * @return A RrcConnectionSetupCompleted, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionSetupCompleted GetMessage () const;

  /**
  * Getter for m_rrcTransactionIdentifier
  * @return m_rrcTransactionIdentifier
  */
  uint8_t GetRrcTransactionIdentifier () const;

private:
  uint8_t m_rrcTransactionIdentifier; ///< RRC transaction identifier

};

/**
* This class manages the serialization/deserialization of RrcConnectionSetupComplete IE
*/
class RrcConnectionReconfigurationCompleteHeader : public RrcUlDcchMessage
{
public:
  RrcConnectionReconfigurationCompleteHeader ();
  ~RrcConnectionReconfigurationCompleteHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReconfigurationCompleted IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */
  void SetMessage (LteRrcSap::RrcConnectionReconfigurationCompleted msg);

  /**
  * Returns a RrcConnectionReconfigurationCompleted IE from the values in the class attributes
  * @return A RrcConnectionReconfigurationCompleted, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReconfigurationCompleted GetMessage () const;

  /**
  * Getter for m_rrcTransactionIdentifier
  * @return m_rrcTransactionIdentifier
  */ 
  uint8_t GetRrcTransactionIdentifier () const;

private:
  uint8_t m_rrcTransactionIdentifier; ///< RRC transaction identifier

};

/**
* This class manages the serialization/deserialization of RrcConnectionReconfiguration IE
*/
class RrcConnectionReconfigurationHeader : public RrcDlDcchMessage
{
public:
  RrcConnectionReconfigurationHeader ();
  ~RrcConnectionReconfigurationHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReconfiguration IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionReconfiguration msg);

  /**
  * Returns a RrcConnectionReconfiguration IE from the values in the class attributes
  * @return A RrcConnectionReconfiguration, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReconfiguration GetMessage () const; 

  /**
  * Getter for m_haveMeasConfig
  * @return m_haveMeasConfig
  */
  bool GetHaveMeasConfig ();

  /**
  * Getter for m_measConfig
  * @return m_measConfig
  */
  LteRrcSap::MeasConfig GetMeasConfig ();

  /**
  * Getter for m_haveMobilityControlInfo
  * @return m_haveMobilityControlInfo
  */
  bool GetHaveMobilityControlInfo ();

  /**
  * Getter for m_mobilityControlInfo
  * @return m_mobilityControlInfo
  */
  LteRrcSap::MobilityControlInfo GetMobilityControlInfo ();

  /**
  * Getter for m_haveRadioResourceConfigDedicated
  * @return m_haveRadioResourceConfigDedicated
  */
  bool GetHaveRadioResourceConfigDedicated ();

  /**
  * Getter for m_radioResourceConfigDedicated
  * @return m_radioResourceConfigDedicated
  */
  LteRrcSap::RadioResourceConfigDedicated GetRadioResourceConfigDedicated ();

  /**
  * Getter for m_rrcTransactionIdentifier
  * @return m_rrcTransactionIdentifier
  */
  uint8_t GetRrcTransactionIdentifier () const;

  /**
  * Getter for m_radioResourceConfigDedicated
  * @return m_radioResourceConfigDedicated
  */
  LteRrcSap::RadioResourceConfigDedicated GetRadioResourceConfigDedicated () const; 

  /**
  * Getter for m_haveNonCriticalExtension
  * @return m_haveNonCriticalExtension
  */
  bool GetHaveNonCriticalExtensionConfig ();

  /**
  * Getter for m_nonCriticalExtension
  * @return m_nonCriticalExtension
  */
  LteRrcSap::NonCriticalExtensionConfiguration GetNonCriticalExtensionConfig ();  

  /**
  * Gets m_radioResourceConfigDedicated.havePhysicalConfigDedicated
  * @return m_radioResourceConfigDedicated.havePhysicalConfigDedicated
  */
  bool HavePhysicalConfigDedicated () const;

  /**
  * Gets m_radioResourceConfigDedicated.physicalConfigDedicated
  * @return m_radioResourceConfigDedicated.physicalConfigDedicated
  */
  LteRrcSap::PhysicalConfigDedicated GetPhysicalConfigDedicated () const;

  /**
  * Gets m_radioResourceConfigDedicated.srbToAddModList
  * @return m_radioResourceConfigDedicated.srbToAddModList
  */
  std::list<LteRrcSap::SrbToAddMod> GetSrbToAddModList () const;

  /**
  * Gets m_radioResourceConfigDedicated.drbToAddModList
  * @return m_radioResourceConfigDedicated.drbToAddModList
  */
  std::list<LteRrcSap::DrbToAddMod> GetDrbToAddModList () const;

  /**
  * Gets m_radioResourceConfigDedicated.drbToReleaseList
  * @return m_radioResourceConfigDedicated.drbToReleaseList
  */
  std::list<uint8_t> GetDrbToReleaseList () const;

private:
  uint8_t m_rrcTransactionIdentifier; ///< RRC transaction identifier
  bool m_haveMeasConfig; ///< have measure config?
  LteRrcSap::MeasConfig m_measConfig; ///< the measure config
  bool m_haveMobilityControlInfo; ///< have mobility control info?
  LteRrcSap::MobilityControlInfo m_mobilityControlInfo; ///< the modility control info
  bool m_haveRadioResourceConfigDedicated; ///< have radio resource config dedicated?
  LteRrcSap::RadioResourceConfigDedicated m_radioResourceConfigDedicated; ///< the radio resource config dedicated
  bool m_haveNonCriticalExtension; ///< Have non-critical extension
  LteRrcSap::NonCriticalExtensionConfiguration m_nonCriticalExtension; ///< the non-critical extension
};

/**
* This class manages the serialization/deserialization of HandoverPreparationInfo IE
*/
class HandoverPreparationInfoHeader : public RrcAsn1Header
{
public:
  HandoverPreparationInfoHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a HandoverPreparationInfo IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::HandoverPreparationInfo msg);

  /**
  * Returns a HandoverPreparationInfo IE from the values in the class attributes
  * @return A HandoverPreparationInfo, as defined in LteRrcSap
  */
  LteRrcSap::HandoverPreparationInfo GetMessage () const;

  /**
  * Getter for m_asConfig
  * @return m_asConfig
  */ 
  LteRrcSap::AsConfig GetAsConfig () const;

private:
  LteRrcSap::AsConfig m_asConfig; ///< AS config
};

/**
* This class manages the serialization/deserialization of RRCConnectionReestablishmentRequest IE
*/
class RrcConnectionReestablishmentRequestHeader : public RrcUlCcchMessage
{
public:
  RrcConnectionReestablishmentRequestHeader ();
  ~RrcConnectionReestablishmentRequestHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReestablishmentRequest IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionReestablishmentRequest msg);

  /**
  * Returns a RrcConnectionReestablishmentRequest IE from the values in the class attributes
  * @return A RrcConnectionReestablishmentRequest, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReestablishmentRequest GetMessage () const;

  /**
  * Getter for m_ueIdentity
  * @return m_ueIdentity
  */
  LteRrcSap::ReestabUeIdentity GetUeIdentity () const;

  /**
  * Getter for m_reestablishmentCause
  * @return m_reestablishmentCause
  */
  LteRrcSap::ReestablishmentCause GetReestablishmentCause () const;

private:
  LteRrcSap::ReestabUeIdentity m_ueIdentity; ///< UE identity
  LteRrcSap::ReestablishmentCause m_reestablishmentCause; ///< reestablishment cause
};

/**
* This class manages the serialization/deserialization of RrcConnectionReestablishment IE
*/
class RrcConnectionReestablishmentHeader : public RrcDlCcchMessage
{
public:
  RrcConnectionReestablishmentHeader ();
  ~RrcConnectionReestablishmentHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReestablishment IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionReestablishment msg);

  /**
  * Returns a RrcConnectionReestablishment IE from the values in the class attributes
  * @return A RrcConnectionReestablishment, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReestablishment GetMessage () const;

  /**
  * Getter for m_rrcTransactionIdentifier attribute
  * @return m_rrcTransactionIdentifier
  */
  uint8_t GetRrcTransactionIdentifier () const;

  /**
  * Getter for m_radioResourceConfigDedicated attribute
  * @return m_radioResourceConfigDedicated
  */
  LteRrcSap::RadioResourceConfigDedicated GetRadioResourceConfigDedicated () const;

private:
  uint8_t m_rrcTransactionIdentifier; ///< RRC transaction identifier
  LteRrcSap::RadioResourceConfigDedicated m_radioResourceConfigDedicated; ///< radio resource config dedicated
};

/**
* This class manages the serialization/deserialization of RrcConnectionReestablishmentComplete IE
*/
class RrcConnectionReestablishmentCompleteHeader : public RrcUlDcchMessage
{
public:
  RrcConnectionReestablishmentCompleteHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReestablishmentComplete IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionReestablishmentComplete msg);

  /**
  * Returns a RrcConnectionReestablishmentComplete IE from the values in the class attributes
  * @return A RrcConnectionReestablishmentComplete, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReestablishmentComplete GetMessage () const;

  /**
  * Getter for m_rrcTransactionIdentifier attribute
  * @return m_rrcTransactionIdentifier
  */
  uint8_t GetRrcTransactionIdentifier () const;

private:
  uint8_t m_rrcTransactionIdentifier; ///< RRC transaction identifier
};

/**
* This class manages the serialization/deserialization of RrcConnectionReestablishmentReject IE
*/
class RrcConnectionReestablishmentRejectHeader : public RrcDlCcchMessage
{
public:
  RrcConnectionReestablishmentRejectHeader ();
  ~RrcConnectionReestablishmentRejectHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReestablishmentReject IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionReestablishmentReject msg);

  /**
  * Returns a RrcConnectionReestablishmentReject IE from the values in the class attributes
  * @return A RrcConnectionReestablishmentReject, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReestablishmentReject GetMessage () const;

private:
  LteRrcSap::RrcConnectionReestablishmentReject m_rrcConnectionReestablishmentReject; ///< RRC connection reestablishmnet reject
};

/**
* This class manages the serialization/deserialization of RrcConnectionRelease IE
*/
class RrcConnectionReleaseHeader : public RrcDlDcchMessage
{
public:
  RrcConnectionReleaseHeader ();
  ~RrcConnectionReleaseHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionRelease IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionRelease msg);

  /**
  * Returns a RrcConnectionRelease IE from the values in the class attributes
  * @return A RrcConnectionRelease, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionRelease GetMessage () const;

private:
  LteRrcSap::RrcConnectionRelease m_rrcConnectionRelease; ///< RRC connection release
};

/**
* This class manages the serialization/deserialization of RrcConnectionReject IE
*/
class RrcConnectionRejectHeader : public RrcDlCcchMessage
{
public:
  RrcConnectionRejectHeader ();
  ~RrcConnectionRejectHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a RrcConnectionReject IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */ 
  void SetMessage (LteRrcSap::RrcConnectionReject msg);

  /**
  * Returns a RrcConnectionReject IE from the values in the class attributes
  * @return A RrcConnectionReject, as defined in LteRrcSap
  */
  LteRrcSap::RrcConnectionReject GetMessage () const;

private:
  LteRrcSap::RrcConnectionReject m_rrcConnectionReject; ///< RRC connection reject
};

/**
* This class manages the serialization/deserialization of MeasurementReport IE
*/
class MeasurementReportHeader : public RrcUlDcchMessage
{
public:
  MeasurementReportHeader ();
  ~MeasurementReportHeader ();

  // Inherited from RrcAsn1Header 
  void PreSerialize () const;
  uint32_t Deserialize (Buffer::Iterator bIterator);
  void Print (std::ostream &os) const;

  /**
  * Receives a MeasurementReport IE and stores the contents into the class attributes
  * @param msg The information element to parse
  */
  void SetMessage (LteRrcSap::MeasurementReport msg);

  /**
  * Returns a MeasurementReport IE from the values in the class attributes
  * @return A MeasurementReport, as defined in LteRrcSap
  */
  LteRrcSap::MeasurementReport GetMessage () const;

private:
  LteRrcSap::MeasurementReport m_measurementReport; ///< measurement report

};

} // namespace ns3

#endif // RRC_HEADER_H

