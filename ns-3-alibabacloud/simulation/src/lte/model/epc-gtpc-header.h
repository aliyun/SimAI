/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 */

#ifndef EPC_GTPC_HEADER_H
#define EPC_GTPC_HEADER_H

#include "ns3/header.h"
#include "ns3/epc-tft.h"
#include "ns3/eps-bearer.h"

namespace ns3 {

/**
 * \ingroup lte
 *
 * \brief Header of the GTPv2-C protocol
 *
 * Implementation of the GPRS Tunnelling Protocol for Control Plane (GTPv2-C) header
 * according to the 3GPP TS 29.274 document
 */
class GtpcHeader : public Header
{
public:
  GtpcHeader ();
  virtual ~GtpcHeader ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;

  /**
   * Get the message size.
   * 
   * Subclasses are supposed to have a message size greater than zero.
   * 
   * \returns the message size
   */
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get message type
   * \returns the message type
   */
  uint8_t GetMessageType () const;
  /**
   * Get message length
   * \returns the message length
   */
  uint16_t GetMessageLength () const;
  /**
   * Get TEID
   * \returns the TEID
   */
  uint32_t GetTeid () const;
  /**
   * Get sequence number
   * \returns the sequence number
   */
  uint32_t GetSequenceNumber () const;

  /**
   * Set message type
   * \param messageType the message type
   */
  void SetMessageType (uint8_t messageType);
  /**
   * Set message length
   * \param messageLength the message length
   */
  void SetMessageLength (uint16_t messageLength);
  /**
   * Set TEID
   * \param teid the TEID
   */
  void SetTeid (uint32_t teid);
  /**
   * Set sequence number
   * \param sequenceNumber the sequence number
   */
  void SetSequenceNumber (uint32_t sequenceNumber);
  /**
   * Set IEs length. It is used to compute the message length
   * \param iesLength the IEs length
   */
  void SetIesLength (uint16_t iesLength);

  /**
   * Compute the message length according to the message type
   */
  void ComputeMessageLength (void);

  /// Interface Type enumeration
  enum InterfaceType_t
  {
    S1U_ENB_GTPU  = 0,
    S5_SGW_GTPU   = 4,
    S5_PGW_GTPU   = 5,
    S5_SGW_GTPC   = 6,
    S5_PGW_GTPC   = 7,
    S11_MME_GTPC  = 10,
  };

  /// FTEID structure
  struct Fteid_t
  {
    InterfaceType_t interfaceType; //!< Interface type
    Ipv4Address addr; //!< IPv4 address
    uint32_t teid; //!< TEID
  };

  /// Message Type enumeration
  enum MessageType_t
  {
    Reserved                  = 0,
    CreateSessionRequest      = 32,
    CreateSessionResponse     = 33,
    ModifyBearerRequest       = 34,
    ModifyBearerResponse      = 35,
    DeleteSessionRequest      = 36,
    DeleteSessionResponse     = 37,
    DeleteBearerCommand       = 66,
    DeleteBearerRequest       = 99,
    DeleteBearerResponse      = 100,
  };


private:
  /**
   * TEID flag.
   * This flag indicates if TEID field is present or not
   */
  bool m_teidFlag;
  /**
   * Message type field.
   * It can be one of the values of MessageType_t
   */
  uint8_t m_messageType;
  /**
   * Message length field.
   * This field indicates the length of the message in octets excluding
   * the mandatory part of the GTP-C header (the first 4 octets)
   */
  uint16_t m_messageLength;
  /**
   * Tunnel Endpoint Identifier (TEID) field
   */
  uint32_t m_teid;
  /**
   * GTP Sequence number field
   */
  uint32_t m_sequenceNumber;


protected:
  /**
   * Serialize the GTP-C header in the GTP-C messages
   * \param i the buffer iterator
   */
  void PreSerialize (Buffer::Iterator &i) const;
  /**
   * Deserialize the GTP-C header in the GTP-C messages
   * \param i the buffer iterator
   * \return number of bytes deserialized
   */
  uint32_t PreDeserialize (Buffer::Iterator &i);

};

/**
 * \ingroup lte
 * GTP-C Information Elemements
 */
class GtpcIes
{
public:
  /**
   * Cause
   */
  enum Cause_t
  {
    RESERVED          = 0,
    REQUEST_ACCEPTED  = 16,
  };

  const uint32_t serializedSizeImsi = 12; //!< IMSI serialized size
  const uint32_t serializedSizeCause = 6; //!< Cause serialized size
  const uint32_t serializedSizeEbi = 5;   //!< EBI serialized size
  const uint32_t serializedSizeBearerQos = 26;  //!< Bearer QoS serialized size
  const uint32_t serializedSizePacketFilter = 3 + 9 + 9 + 5 + 5 + 3; //!< Packet filter serialized size
  /**
   * \return the BearerTft serialized size
   * \param packetFilters The packet filter
   */
  uint32_t GetSerializedSizeBearerTft (std::list<EpcTft::PacketFilter> packetFilters) const;
  const uint32_t serializedSizeUliEcgi = 12; //!< UliEcgi serialized size
  const uint32_t serializedSizeFteid = 13;   //!< Fteid serialized size
  const uint32_t serializedSizeBearerContextHeader = 4;  //!< Fteid serialized size

  /**
   * Serialize the IMSI
   * \param i Buffer iterator 
   * \param imsi The IMSI
   */
  void SerializeImsi (Buffer::Iterator &i, uint64_t imsi) const;
  /**
   * Deserialize the IMSI
   * \param i Buffer iterator 
   * \param [out] imsi The IMSI
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeImsi (Buffer::Iterator &i, uint64_t &imsi);

  /**
   * Serialize the Cause
   * \param i Buffer iterator 
   * \param cause The Cause
   */
  void SerializeCause (Buffer::Iterator &i, Cause_t cause) const;
  /**
   * Deserialize the Cause
   * \param i Buffer iterator 
   * \param [out] cause The cause
   * \return the number of deserialized bytes
   */
   uint32_t DeserializeCause (Buffer::Iterator &i, Cause_t &cause);

  /**
   * Serialize the eps Bearer Id
   * \param i Buffer iterator 
   * \param  epsBearerId The eps Bearer Id
   */
  void SerializeEbi (Buffer::Iterator &i, uint8_t epsBearerId) const;
  /**
   * Deserialize the eps Bearer Id
   * \param i Buffer iterator 
   * \param [out] epsBearerId The eps Bearer Id
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeEbi (Buffer::Iterator &i, uint8_t &epsBearerId);

  /**
   * \param i Buffer iterator 
   * \param data data to write in buffer
   *
   * Write the data in buffer and advance the iterator position
   * by five bytes. The data is written in network order and the
   * input data is expected to be in host order.
   */
  void WriteHtonU40 (Buffer::Iterator &i, uint64_t data) const;
  /**
   * \param i Buffer iterator 
   * \return the five bytes read in the buffer.
   *
   * Read data and advance the Iterator by the number of bytes
   * read.
   * The data is read in network format and returned in host format.
   */
  uint64_t ReadNtohU40 (Buffer::Iterator &i);

  /**
   * Serialize the eps Bearer QoS
   * \param i Buffer iterator 
   * \param bearerQos The Bearer QoS
   */
  void SerializeBearerQos (Buffer::Iterator &i, EpsBearer bearerQos) const;
  /**
   * Deserialize the eps Bearer QoS
   * \param i Buffer iterator 
   * \param [out] bearerQos The Bearer QoS
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeBearerQos (Buffer::Iterator &i, EpsBearer &bearerQos);

  /**
   * Serialize the Bearer TFT
   * \param i Buffer iterator 
   * \param packetFilters The Packet filters
   */
  void SerializeBearerTft (Buffer::Iterator &i, std::list<EpcTft::PacketFilter> packetFilters) const;
  /**
   * Deserialize the Bearer TFT
   * \param i Buffer iterator 
   * \param [out] epcTft The Bearer TFT
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeBearerTft (Buffer::Iterator &i, Ptr<EpcTft> epcTft);

  /**
   * Serialize the UliEcgi
   * \param i Buffer iterator 
   * \param uliEcgi The UliEcgi
   */
  void SerializeUliEcgi (Buffer::Iterator &i, uint32_t uliEcgi) const;
  /**
   * Deserialize the UliEcgi
   * \param i Buffer iterator 
   * \param [out] uliEcgi UliEcgi
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeUliEcgi (Buffer::Iterator &i, uint32_t &uliEcgi);

  /**
   * Serialize the Fteid_t
   * \param i Buffer iterator 
   * \param fteid The Fteid_t
   */
  void SerializeFteid (Buffer::Iterator &i, GtpcHeader::Fteid_t fteid) const;
  /**
   * Deserialize the Fteid
   * \param i Buffer iterator 
   * \param [out] fteid Fteid
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeFteid (Buffer::Iterator &i, GtpcHeader::Fteid_t &fteid);

  /**
   * Serialize the Bearer Context Header
   * \param i Buffer iterator 
   * \param length The length
   */
  void SerializeBearerContextHeader (Buffer::Iterator &i, uint16_t length) const;
  /**
   * Deserialize the Bearer Context Header
   * \param i Buffer iterator 
   * \param [out] length length
   * \return the number of deserialized bytes
   */
  uint32_t DeserializeBearerContextHeader (Buffer::Iterator &i, uint16_t &length);
};

/**
 * \ingroup lte
 * GTP-C Create Session Request Message
 */
class GtpcCreateSessionRequestMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcCreateSessionRequestMessage ();
  virtual ~GtpcCreateSessionRequestMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get the IMSI
   * \return IMSI
   */
  uint64_t GetImsi () const;
  /**
   * Set the IMSI
   * \param imsi IMSI
   */
  void SetImsi (uint64_t imsi);

  /**
   * Get the UliEcgi
   * \return UliEcgi
   */
  uint32_t GetUliEcgi () const;
  /**
   * Set the UliEcgi
   * \param uliEcgi UliEcgi
   */
  void SetUliEcgi (uint32_t uliEcgi);

  /**
   * Get the Sender CpFteid
   * \return Sender CpFteid
   */
  GtpcHeader::Fteid_t GetSenderCpFteid () const;
  /**
   * Set the Sender CpFteid
   * \param fteid Sender CpFteid
   */
  void SetSenderCpFteid (GtpcHeader::Fteid_t fteid);

  /**
   * Bearer Context structure
   */
  struct BearerContextToBeCreated
  {
    GtpcHeader::Fteid_t sgwS5uFteid; ///< FTEID
    uint8_t epsBearerId; ///< EPS bearer ID
    Ptr<EpcTft> tft; ///< traffic flow template
    EpsBearer bearerLevelQos; ///< bearer QOS level
  };

  /**
   * Get the Bearer Contexts
   * \return the Bearer Context list
   */
  std::list<BearerContextToBeCreated> GetBearerContextsToBeCreated () const;
  /**
   * Set the Bearer Contexts
   * \param bearerContexts the Bearer Context list
   */
  void SetBearerContextsToBeCreated (std::list<BearerContextToBeCreated> bearerContexts);

private:
  uint64_t m_imsi; //!< IMSI
  uint32_t m_uliEcgi; //!< UliEcgi
  GtpcHeader::Fteid_t m_senderCpFteid; //!< Sender CpFteid

  /// Bearer Context list
  std::list<BearerContextToBeCreated> m_bearerContextsToBeCreated;
};

/**
 * \ingroup lte
 * GTP-C Create Session Response Message
 */
class GtpcCreateSessionResponseMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcCreateSessionResponseMessage ();
  virtual ~GtpcCreateSessionResponseMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get the Cause
   * \return the Cause
   */
  Cause_t GetCause () const;
  /**
   * Set the Cause
   * \param cause The cause
   */
  void SetCause (Cause_t cause);

  /**
   * Get the Sender CpFteid
   * \return the Sender CpFteid
   */
  GtpcHeader::Fteid_t GetSenderCpFteid () const;
  /**
   * Set the Sender CpFteid
   * \param fteid the Sender CpFteid
   */
  void SetSenderCpFteid (GtpcHeader::Fteid_t fteid);

  /**
   * Bearer Context structure
   */
  struct BearerContextCreated
  {
    uint8_t epsBearerId; ///< EPS bearer ID
    uint8_t cause; ///< Cause
    Ptr<EpcTft> tft; ///< Bearer traffic flow template
    GtpcHeader::Fteid_t fteid; ///< FTEID
    EpsBearer bearerLevelQos; ///< Bearer QOS level
  };

  /**
   * Get the Container of Bearer Contexts
   * \return a list of Bearer Contexts
   */
  std::list<BearerContextCreated> GetBearerContextsCreated () const;
  /**
   * Set the Bearer Contexts
   * \param bearerContexts a list of Bearer Contexts
   */
  void SetBearerContextsCreated (std::list<BearerContextCreated> bearerContexts);

private:
  Cause_t m_cause; //!< Cause
  GtpcHeader::Fteid_t m_senderCpFteid; //!< Sender CpFteid
  /// Container of Bearer Contexts
  std::list<BearerContextCreated> m_bearerContextsCreated;
};


/**
 * \ingroup lte
 * GTP-C Modify Bearer Request Message
 */
class GtpcModifyBearerRequestMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcModifyBearerRequestMessage ();
  virtual ~GtpcModifyBearerRequestMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get the IMSI
   * \return IMSI
   */
  uint64_t GetImsi () const;
  /**
   * Set the IMSI
   * \param imsi IMSI
   */
  void SetImsi (uint64_t imsi);

  /**
   * Get the UliEcgi
   * \return UliEcgi
   */
  uint32_t GetUliEcgi () const;
  /**
   * Set the UliEcgi
   * \param uliEcgi UliEcgi
   */
  void SetUliEcgi (uint32_t uliEcgi);

  /**
   * Bearer Context structure
   */
  struct BearerContextToBeModified
  {
    uint8_t epsBearerId; ///< EPS bearer ID
    GtpcHeader::Fteid_t fteid; ///< FTEID
  };

  /**
   * Get the Bearer Contexts
   * \return the Bearer Context list
   */
  std::list<BearerContextToBeModified> GetBearerContextsToBeModified () const;
  /**
   * Set the Bearer Contexts
   * \param bearerContexts the Bearer Context list
   */
  void SetBearerContextsToBeModified (std::list<BearerContextToBeModified> bearerContexts);

private:
  uint64_t m_imsi;     //!< IMSI
  uint32_t m_uliEcgi;  //!< UliEcgi

  /// Bearer Context list
  std::list<BearerContextToBeModified> m_bearerContextsToBeModified;
};

/**
 * \ingroup lte
 * GTP-C Modify Bearer Response Message
 */
class GtpcModifyBearerResponseMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcModifyBearerResponseMessage ();
  virtual ~GtpcModifyBearerResponseMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get the Cause
   * \return the Cause
   */
  Cause_t GetCause () const;
  /**
   * Set the Cause
   * \param cause The cause
   */
  void SetCause (Cause_t cause);

private:
  Cause_t m_cause; //!< Cause
};

/**
 * \ingroup lte
 * GTP-C Delete Bearer Command Message
 */
class GtpcDeleteBearerCommandMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcDeleteBearerCommandMessage ();
  virtual ~GtpcDeleteBearerCommandMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /// Bearer context
  struct BearerContext
  {
    uint8_t m_epsBearerId; ///< EPS bearer ID
  };

  /**
   * Get the Bearer contexts
   * \return container of beraer contexts
   */
  std::list<BearerContext> GetBearerContexts () const;
  /**
   * Set the Bearer contexts
   * \param bearerContexts container of beraer contexts
   */
  void SetBearerContexts (std::list<BearerContext> bearerContexts);

private:
  std::list<BearerContext> m_bearerContexts; //!< Container of Bearer Contexts
};

/**
 * \ingroup lte
 * GTP-C Delete Bearer Request Message
 */
class GtpcDeleteBearerRequestMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcDeleteBearerRequestMessage ();
  virtual ~GtpcDeleteBearerRequestMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get the Bearers IDs
   * \return a container of Bearers IDs
   */
  std::list<uint8_t> GetEpsBearerIds () const;
  /**
   * Set the Bearers IDs
   * \param epsBearerIds The container of Bearers IDs
   */
  void SetEpsBearerIds (std::list<uint8_t> epsBearerIds);

private:
  std::list<uint8_t> m_epsBearerIds; //!< Container of Bearers IDs
};

/**
 * \ingroup lte
 * GTP-C Delete Bearer Response Message
 */
class GtpcDeleteBearerResponseMessage : public GtpcHeader, public GtpcIes
{
public:
  GtpcDeleteBearerResponseMessage ();
  virtual ~GtpcDeleteBearerResponseMessage ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetMessageSize (void) const;

  /**
   * Get the Cause
   * \return the Cause
   */
  Cause_t GetCause () const;
  /**
   * Set the Cause
   * \param cause The cause
   */
  void SetCause (Cause_t cause);

  /**
   * Get the Bearers IDs
   * \return a container of Bearers IDs
   */
  std::list<uint8_t> GetEpsBearerIds () const;
  /**
   * Set the Bearers IDs
   * \param epsBearerIds The container of Bearers IDs
   */
  void SetEpsBearerIds (std::list<uint8_t> epsBearerIds);

private:
  Cause_t m_cause; //!< Cause
  std::list<uint8_t> m_epsBearerIds; //!< Container of Bearers IDs
};

} // namespace ns3

#endif // EPC_GTPC_HEADER_H
