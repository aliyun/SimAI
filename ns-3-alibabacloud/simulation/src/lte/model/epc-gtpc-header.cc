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

#include "ns3/epc-gtpc-header.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GtpcHeader");

NS_OBJECT_ENSURE_REGISTERED (GtpcHeader);

/// GTPv2-C protocol version number
static const uint8_t VERSION = 2;

TypeId
GtpcHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcHeader")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcHeader> ();
  return tid;
}

GtpcHeader::GtpcHeader ()
  : m_teidFlag (false),
    m_messageType (0),
    m_messageLength (4),
    m_teid (0),
    m_sequenceNumber (0)
{
}

GtpcHeader::~GtpcHeader ()
{
}

TypeId
GtpcHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcHeader::GetSerializedSize (void) const
{
  return m_teidFlag ? 12 : 8;
}

void
GtpcHeader::Serialize (Buffer::Iterator start) const
{
  NS_FATAL_ERROR ("Serialize GTP-C header is forbidden");
}

void
GtpcHeader::PreSerialize (Buffer::Iterator &i) const
{
  i.WriteU8 ((VERSION << 5) | (1 << 3));
  i.WriteU8 (m_messageType);
  i.WriteHtonU16 (m_messageLength);
  i.WriteHtonU32 (m_teid);
  i.WriteU8 ((m_sequenceNumber & 0x00ff0000) >> 16);
  i.WriteU8 ((m_sequenceNumber & 0x0000ff00) >> 8);
  i.WriteU8 (m_sequenceNumber & 0x000000ff);
  i.WriteU8 (0);
}

uint32_t
GtpcHeader::Deserialize (Buffer::Iterator start)
{
  return PreDeserialize (start);
}

uint32_t
GtpcHeader::PreDeserialize (Buffer::Iterator &i)
{
  uint8_t firstByte = i.ReadU8 ();
  uint8_t version = (firstByte >> 5) & 0x07;
  if (version != 2)
    {
      NS_FATAL_ERROR ("GTP-C version not supported");
      return 0;
    }

  m_teidFlag = ((firstByte >> 3) & 0x01) == 1;
  if (!m_teidFlag)
    {
      NS_FATAL_ERROR ("TEID is missing");
      return 0;
    }

  m_messageType = i.ReadU8 ();
  m_messageLength = i.ReadNtohU16 ();
  if (m_teidFlag)
    {
      m_teid = i.ReadNtohU32 ();
    }
  m_sequenceNumber = i.ReadU8 () << 16 | i.ReadU8 () << 8 | i.ReadU8 ();
  i.ReadU8();

  return GtpcHeader::GetSerializedSize ();
}

void
GtpcHeader::Print (std::ostream &os) const
{
  os << " messageType " << (uint32_t) m_messageType << " messageLength " << m_messageLength;
  os << " TEID " << m_teid << " sequenceNumber " << m_sequenceNumber;
}

uint32_t
GtpcHeader::GetMessageSize (void) const
{
  return 0;
}

uint8_t
GtpcHeader::GetMessageType () const
{
  return m_messageType;
}

uint16_t
GtpcHeader::GetMessageLength () const
{
  return m_messageLength;
}

uint32_t
GtpcHeader::GetTeid () const
{
  return m_teid;
}

uint32_t
GtpcHeader::GetSequenceNumber () const
{
  return m_sequenceNumber;
}

void
GtpcHeader::SetMessageType (uint8_t messageType)
{
  m_messageType = messageType;
}

void
GtpcHeader::SetMessageLength (uint16_t messageLength)
{
  m_messageLength = messageLength;
}

void
GtpcHeader::SetTeid (uint32_t teid)
{
  m_teidFlag = true;
  m_teid = teid;
  m_messageLength = m_teidFlag ? 8 : 4;
}

void
GtpcHeader::SetSequenceNumber (uint32_t sequenceNumber)
{
  m_sequenceNumber = sequenceNumber;
}

void
GtpcHeader::SetIesLength (uint16_t iesLength)
{
  m_messageLength = iesLength;
  m_messageLength += (m_teidFlag) ? 8 : 4;
}

void
GtpcHeader::ComputeMessageLength (void)
{
  SetIesLength (GetMessageSize ()); 
}
/////////////////////////////////////////////////////////////////////

void
GtpcIes::SerializeImsi (Buffer::Iterator &i, uint64_t imsi) const
{
  i.WriteU8 (1);            // IE Type = IMSI
  i.WriteHtonU16 (8);       // Length
  i.WriteU8 (0);            // Spare + Instance
  i.WriteHtonU64 (imsi);
}

uint32_t
GtpcIes::DeserializeImsi (Buffer::Iterator &i, uint64_t &imsi)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 1, "Wrong IMSI IE type = " << (uint16_t) type);
  uint16_t length = i.ReadNtohU16 ();
  NS_ASSERT_MSG (length == 8, "Wrong IMSI IE length");
  uint8_t instance = i.ReadU8 () & 0x0f;
  NS_ASSERT_MSG (instance == 0, "Wrong IMSI IE instance");
  imsi = i.ReadNtohU64 ();

  return serializedSizeImsi;
}

void
GtpcIes::SerializeCause (Buffer::Iterator &i, Cause_t cause) const
{
  i.WriteU8 (2);            // IE Type = Cause
  i.WriteHtonU16 (2);       // Length
  i.WriteU8 (0);            // Spare + Instance
  i.WriteU8 (cause);        // Cause value
  i.WriteU8 (0);            // Spare + CS
}

uint32_t
GtpcIes::DeserializeCause (Buffer::Iterator &i, Cause_t &cause)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 2, "Wrong Cause IE type = " << (uint16_t) type);
  uint16_t length = i.ReadNtohU16 ();
  NS_ASSERT_MSG (length == 2, "Wrong Cause IE length");
  uint8_t instance = i.ReadU8 () & 0x0f;
  NS_ASSERT_MSG (instance == 0, "Wrong Cause IE instance");
  cause = Cause_t (i.ReadU8 ());
  i.ReadU8 ();

  return serializedSizeCause;
}

void
GtpcIes::SerializeEbi (Buffer::Iterator &i, uint8_t epsBearerId) const
{
  i.WriteU8 (73);         // IE Type = EPS Bearer ID (EBI)
  i.WriteHtonU16 (1);     // Length
  i.WriteU8 (0);          // Spare + Instance
  i.WriteU8 (epsBearerId & 0x0f);
}

uint32_t
GtpcIes::DeserializeEbi (Buffer::Iterator &i, uint8_t &epsBearerId)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 73, "Wrong EBI IE type = " << (uint16_t) type);
  uint16_t length = i.ReadNtohU16 ();
  NS_ASSERT_MSG (length == 1, "Wrong EBI IE length");
  uint8_t instance = i.ReadU8 ();
  NS_ASSERT_MSG (instance == 0, "Wrong EBI IE instance");
  epsBearerId = i.ReadU8 () & 0x0f;

  return serializedSizeEbi;
}

void
GtpcIes::WriteHtonU40 (Buffer::Iterator &i, uint64_t data) const
{
  i.WriteU8 ((data >> 32) & 0xff);
  i.WriteU8 ((data >> 24) & 0xff);
  i.WriteU8 ((data >> 16) & 0xff);
  i.WriteU8 ((data >> 8) & 0xff);
  i.WriteU8 ((data >> 0) & 0xff);
}

uint64_t
GtpcIes::ReadNtohU40 (Buffer::Iterator &i)
{
  uint64_t retval = 0;
  retval |= i.ReadU8 ();
  retval <<= 8;
  retval |= i.ReadU8 ();
  retval <<= 8;
  retval |= i.ReadU8 ();
  retval <<= 8;
  retval |= i.ReadU8 ();
  retval <<= 8;
  retval |= i.ReadU8 ();
  return retval;
}

void
GtpcIes::SerializeBearerQos (Buffer::Iterator &i, EpsBearer bearerQos) const
{
  i.WriteU8 (80);         // IE Type = Bearer QoS
  i.WriteHtonU16 (22);    // Length
  i.WriteU8 (0);          // Spare + Instance
  i.WriteU8 (0); // MRE TODO: bearerQos.arp
  i.WriteU8 (bearerQos.qci);
  WriteHtonU40 (i, bearerQos.gbrQosInfo.mbrUl);
  WriteHtonU40 (i, bearerQos.gbrQosInfo.mbrDl);
  WriteHtonU40 (i, bearerQos.gbrQosInfo.gbrUl);
  WriteHtonU40 (i, bearerQos.gbrQosInfo.gbrDl);
}

uint32_t
GtpcIes::DeserializeBearerQos (Buffer::Iterator &i, EpsBearer &bearerQos)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 80, "Wrong Bearer QoS IE type = " << (uint16_t) type);
  uint16_t length = i.ReadNtohU16 ();
  NS_ASSERT_MSG (length == 22, "Wrong Bearer QoS IE length");
  uint8_t instance = i.ReadU8 ();
  NS_ASSERT_MSG (instance == 0, "Wrong Bearer QoS IE instance");
  i.ReadU8 ();
  bearerQos.qci = EpsBearer::Qci (i.ReadU8 ());
  bearerQos.gbrQosInfo.mbrUl = ReadNtohU40 (i);
  bearerQos.gbrQosInfo.mbrDl = ReadNtohU40 (i);
  bearerQos.gbrQosInfo.gbrUl = ReadNtohU40 (i);
  bearerQos.gbrQosInfo.gbrDl = ReadNtohU40 (i);
  return serializedSizeBearerQos;
}

void
GtpcIes::SerializeBearerTft (Buffer::Iterator &i, std::list<EpcTft::PacketFilter> packetFilters) const
{
  i.WriteU8 (84);   // IE Type = EPS Bearer Level Fraffic Flow Template (Bearer TFT)
  i.WriteHtonU16 (1 + packetFilters.size () * serializedSizePacketFilter);
  i.WriteU8 (0);    // Spare + Instance
  i.WriteU8 (0x20 + (packetFilters.size () & 0x0f)); // Create new TFT + Number of packet filters

  for (auto &pf : packetFilters)
    {
      i.WriteU8 ((pf.direction << 4) & 0x30);
      i.WriteU8 (pf.precedence);
      i.WriteU8 (serializedSizePacketFilter - 3); // Length of Packet filter contents

      i.WriteU8 (0x10);   // IPv4 remote address type
      i.WriteHtonU32 (pf.remoteAddress.Get ());
      i.WriteHtonU32 (pf.remoteMask.Get ());
      i.WriteU8 (0x11);   // IPv4 local address type
      i.WriteHtonU32 (pf.localAddress.Get ());
      i.WriteHtonU32 (pf.localMask.Get ());
      i.WriteU8 (0x41);   // Local port range type
      i.WriteHtonU16 (pf.localPortStart);
      i.WriteHtonU16 (pf.localPortEnd);
      i.WriteU8 (0x51);   // Remote port range type
      i.WriteHtonU16 (pf.remotePortStart);
      i.WriteHtonU16 (pf.remotePortEnd);
      i.WriteU8 (0x70);   // Type of service
      i.WriteU8 (pf.typeOfService);
      i.WriteU8 (pf.typeOfServiceMask);
    }
}

uint32_t
GtpcIes::DeserializeBearerTft (Buffer::Iterator &i, Ptr<EpcTft> epcTft)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 84, "Wrong Bearer TFT IE type = " << (uint16_t) type);
  i.ReadNtohU16 ();
  i.ReadU8 ();
  uint8_t numberOfPacketFilters = i.ReadU8 () & 0x0f;

  for (uint8_t pf = 0; pf < numberOfPacketFilters; ++pf)
    {
      EpcTft::PacketFilter packetFilter;
      packetFilter.direction = EpcTft::Direction ((i.ReadU8 () & 0x30) >> 4);
      packetFilter.precedence = i.ReadU8 ();
      i.ReadU8 (); // Length of Packet filter contents
      i.ReadU8 ();
      packetFilter.remoteAddress = Ipv4Address (i.ReadNtohU32 ());
      packetFilter.remoteMask = Ipv4Mask (i.ReadNtohU32 ());
      i.ReadU8 ();
      packetFilter.localAddress = Ipv4Address (i.ReadNtohU32 ());
      packetFilter.localMask = Ipv4Mask (i.ReadNtohU32 ());
      i.ReadU8 ();
      packetFilter.localPortStart = i.ReadNtohU16 ();
      packetFilter.localPortEnd = i.ReadNtohU16 ();
      i.ReadU8 ();
      packetFilter.remotePortStart = i.ReadNtohU16 ();
      packetFilter.remotePortEnd = i.ReadNtohU16 ();
      i.ReadU8 ();
      packetFilter.typeOfService = i.ReadU8 ();
      packetFilter.typeOfServiceMask = i.ReadU8 ();
      epcTft->Add (packetFilter);
    }

  return GetSerializedSizeBearerTft (epcTft->GetPacketFilters ());
}

uint32_t
GtpcIes::GetSerializedSizeBearerTft (std::list<EpcTft::PacketFilter> packetFilters) const
{
  return (5 + packetFilters.size () * serializedSizePacketFilter);
}

void
GtpcIes::SerializeUliEcgi (Buffer::Iterator &i, uint32_t uliEcgi) const
{
  i.WriteU8 (86);       // IE Type = ULI (ECGI)
  i.WriteHtonU16 (8);   // Length
  i.WriteU8 (0);        // Spare + Instance
  i.WriteU8 (0x10);     // ECGI flag
  i.WriteU8 (0);        // Dummy MCC and MNC
  i.WriteU8 (0);        // Dummy MCC and MNC
  i.WriteU8 (0);        // Dummy MCC and MNC
  i.WriteHtonU32 (uliEcgi);
}

uint32_t
GtpcIes::DeserializeUliEcgi (Buffer::Iterator &i, uint32_t &uliEcgi)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 86, "Wrong ULI ECGI IE type = " << (uint16_t) type);
  uint16_t length = i.ReadNtohU16 ();
  NS_ASSERT_MSG (length == 8, "Wrong ULI ECGI IE length");
  uint8_t instance = i.ReadU8 () & 0x0f;
  NS_ASSERT_MSG (instance == 0, "Wrong ULI ECGI IE instance");
  i.Next (4);
  uliEcgi = i.ReadNtohU32 () & 0x0fffffff;

  return serializedSizeUliEcgi;
}

void
GtpcIes::SerializeFteid (Buffer::Iterator &i, GtpcHeader::Fteid_t fteid) const
{
  i.WriteU8 (87);       // IE Type = Fully Qualified TEID (F-TEID)
  i.WriteHtonU16 (9);   // Length
  i.WriteU8 (0);        // Spare + Instance
  i.WriteU8 (0x80 | ((uint8_t) fteid.interfaceType & 0x1f)); // IP version flag + Iface type
  i.WriteHtonU32 (fteid.teid); // TEID
  i.WriteHtonU32 (fteid.addr.Get ()); // IPv4 address
}

uint32_t
GtpcIes::DeserializeFteid (Buffer::Iterator &i, GtpcHeader::Fteid_t &fteid)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 87, "Wrong FTEID IE type = " << (uint16_t) type);
  uint16_t length = i.ReadNtohU16 ();
  NS_ASSERT_MSG (length == 9, "Wrong FTEID IE length");
  uint8_t instance = i.ReadU8 () & 0x0f;
  NS_ASSERT_MSG (instance == 0, "Wrong FTEID IE instance");
  uint8_t flags = i.ReadU8 (); // IP version flag + Iface type
  fteid.interfaceType = GtpcHeader::InterfaceType_t (flags & 0x1f);
  fteid.teid = i.ReadNtohU32 (); // TEID
  fteid.addr.Set (i.ReadNtohU32 ()); // IPv4 address

  return serializedSizeFteid;
}

void
GtpcIes::SerializeBearerContextHeader (Buffer::Iterator &i, uint16_t length) const
{
  i.WriteU8 (93);   // IE Type = Bearer Context
  i.WriteU16 (length);
  i.WriteU8 (0);    // Spare + Instance
}

uint32_t
GtpcIes::DeserializeBearerContextHeader (Buffer::Iterator &i, uint16_t &length)
{
  uint8_t type = i.ReadU8 ();
  NS_ASSERT_MSG (type == 93, "Wrong Bearer Context IE type = " << (uint16_t) type);
  length = i.ReadNtohU16 ();
  uint8_t instance = i.ReadU8 () & 0x0f;
  NS_ASSERT_MSG (instance == 0, "Wrong Bearer Context IE instance");

  return serializedSizeBearerContextHeader;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcCreateSessionRequestMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcCreateSessionRequestMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcCreateSessionRequestMessage> ();
  return tid;
}

GtpcCreateSessionRequestMessage::GtpcCreateSessionRequestMessage ()
{
  SetMessageType (GtpcHeader::CreateSessionRequest);
  SetSequenceNumber (0);
  m_imsi = 0;
  m_uliEcgi = 0;
}

GtpcCreateSessionRequestMessage::~GtpcCreateSessionRequestMessage ()
{
}

TypeId
GtpcCreateSessionRequestMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcCreateSessionRequestMessage::GetMessageSize (void) const
{
  uint32_t serializedSize = serializedSizeImsi + serializedSizeUliEcgi + serializedSizeFteid;
  for (auto &bc : m_bearerContextsToBeCreated)
    {
      serializedSize += serializedSizeBearerContextHeader + serializedSizeEbi
                     + GetSerializedSizeBearerTft (bc.tft->GetPacketFilters ())
                     + serializedSizeFteid + serializedSizeBearerQos;
    }

  return serializedSize;
}

uint32_t
GtpcCreateSessionRequestMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcCreateSessionRequestMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  SerializeImsi (i, m_imsi);
  SerializeUliEcgi (i, m_uliEcgi);
  SerializeFteid (i, m_senderCpFteid);

  for (auto &bc : m_bearerContextsToBeCreated)
    {
      std::list<EpcTft::PacketFilter> packetFilters = bc.tft->GetPacketFilters ();

      SerializeBearerContextHeader (i, serializedSizeEbi + GetSerializedSizeBearerTft (packetFilters) + serializedSizeFteid + serializedSizeBearerQos);

      SerializeEbi (i, bc.epsBearerId);
      SerializeBearerTft (i, packetFilters);
      SerializeFteid (i, bc.sgwS5uFteid);
      SerializeBearerQos (i, bc.bearerLevelQos);
    }
}

uint32_t
GtpcCreateSessionRequestMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  DeserializeImsi (i, m_imsi);
  DeserializeUliEcgi (i, m_uliEcgi);
  DeserializeFteid (i, m_senderCpFteid);

  m_bearerContextsToBeCreated.clear ();
  while (i.GetRemainingSize () > 0)
    {
      uint16_t length;
      DeserializeBearerContextHeader (i, length);

      BearerContextToBeCreated bearerContext;
      DeserializeEbi (i, bearerContext.epsBearerId);

      Ptr<EpcTft> epcTft = Create<EpcTft> ();
      DeserializeBearerTft (i, epcTft);
      bearerContext.tft = epcTft;

      DeserializeFteid (i, bearerContext.sgwS5uFteid);
      DeserializeBearerQos (i, bearerContext.bearerLevelQos);

      m_bearerContextsToBeCreated.push_back (bearerContext);
    }

  return GetSerializedSize ();
}

void
GtpcCreateSessionRequestMessage::Print (std::ostream &os) const
{
  os << " imsi " << m_imsi << " uliEcgi " << m_uliEcgi;
}

uint64_t
GtpcCreateSessionRequestMessage::GetImsi () const
{
  return m_imsi;
}

void
GtpcCreateSessionRequestMessage::SetImsi (uint64_t imsi)
{
  m_imsi = imsi;
}

uint32_t
GtpcCreateSessionRequestMessage::GetUliEcgi () const
{
  return m_uliEcgi;
}

void
GtpcCreateSessionRequestMessage::SetUliEcgi (uint32_t uliEcgi)
{
  m_uliEcgi = uliEcgi;
}

GtpcHeader::Fteid_t
GtpcCreateSessionRequestMessage::GetSenderCpFteid () const
{
  return m_senderCpFteid;
}

void
GtpcCreateSessionRequestMessage::SetSenderCpFteid (GtpcHeader::Fteid_t fteid)
{
  m_senderCpFteid = fteid;
}

std::list<GtpcCreateSessionRequestMessage::BearerContextToBeCreated>
GtpcCreateSessionRequestMessage::GetBearerContextsToBeCreated () const
{
  return m_bearerContextsToBeCreated;
}

void
GtpcCreateSessionRequestMessage::SetBearerContextsToBeCreated (std::list<GtpcCreateSessionRequestMessage::BearerContextToBeCreated> bearerContexts)
{
  m_bearerContextsToBeCreated = bearerContexts;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcCreateSessionResponseMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcCreateSessionResponseMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcCreateSessionResponseMessage> ();
  return tid;
}

GtpcCreateSessionResponseMessage::GtpcCreateSessionResponseMessage ()
{
  SetMessageType (GtpcHeader::CreateSessionResponse);
  SetSequenceNumber (0);
  m_cause = Cause_t::RESERVED;
}

GtpcCreateSessionResponseMessage::~GtpcCreateSessionResponseMessage ()
{
}

TypeId
GtpcCreateSessionResponseMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcCreateSessionResponseMessage::GetMessageSize (void) const
{
  uint32_t serializedSize = serializedSizeCause + serializedSizeFteid;
  for (auto &bc : m_bearerContextsCreated)
    {
      serializedSize += serializedSizeBearerContextHeader + serializedSizeEbi
                     + GetSerializedSizeBearerTft (bc.tft->GetPacketFilters ())
                     + serializedSizeFteid + serializedSizeBearerQos;
    }

  return serializedSize;
}

uint32_t
GtpcCreateSessionResponseMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcCreateSessionResponseMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  SerializeCause (i, m_cause);
  SerializeFteid (i, m_senderCpFteid);

  for (auto &bc : m_bearerContextsCreated)
    {
      std::list<EpcTft::PacketFilter> packetFilters = bc.tft->GetPacketFilters ();

      SerializeBearerContextHeader (i, serializedSizeEbi + GetSerializedSizeBearerTft (packetFilters) + serializedSizeFteid + serializedSizeBearerQos);

      SerializeEbi (i, bc.epsBearerId);
      SerializeBearerTft (i, packetFilters);
      SerializeFteid (i, bc.fteid);
      SerializeBearerQos (i, bc.bearerLevelQos);
    }
}

uint32_t
GtpcCreateSessionResponseMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  DeserializeCause (i, m_cause);
  DeserializeFteid (i, m_senderCpFteid);

  m_bearerContextsCreated.clear ();
  while (i.GetRemainingSize () > 0)
    {
      BearerContextCreated bearerContext;
      uint16_t length;

      DeserializeBearerContextHeader (i, length);
      DeserializeEbi (i, bearerContext.epsBearerId);

      Ptr<EpcTft> epcTft = Create<EpcTft> ();
      DeserializeBearerTft (i, epcTft);
      bearerContext.tft = epcTft;

      DeserializeFteid (i, bearerContext.fteid);
      DeserializeBearerQos (i, bearerContext.bearerLevelQos);

      m_bearerContextsCreated.push_back (bearerContext);
    }

  return GetSerializedSize ();
}

void
GtpcCreateSessionResponseMessage::Print (std::ostream &os) const
{
  os << " cause " << m_cause << " FTEID " << m_senderCpFteid.addr << "," << m_senderCpFteid.teid ;
}

GtpcCreateSessionResponseMessage::Cause_t
GtpcCreateSessionResponseMessage::GetCause () const
{
  return m_cause;
}

void
GtpcCreateSessionResponseMessage::SetCause (GtpcCreateSessionResponseMessage::Cause_t cause)
{
  m_cause = cause;
}

GtpcHeader::Fteid_t
GtpcCreateSessionResponseMessage::GetSenderCpFteid () const
{
  return m_senderCpFteid;
}

void
GtpcCreateSessionResponseMessage::SetSenderCpFteid (GtpcHeader::Fteid_t fteid)
{
  m_senderCpFteid = fteid;
}

std::list<GtpcCreateSessionResponseMessage::BearerContextCreated>
GtpcCreateSessionResponseMessage::GetBearerContextsCreated () const
{
  return m_bearerContextsCreated;
}

void
GtpcCreateSessionResponseMessage::SetBearerContextsCreated (std::list<GtpcCreateSessionResponseMessage::BearerContextCreated> bearerContexts)
{
  m_bearerContextsCreated = bearerContexts;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcModifyBearerRequestMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcModifyBearerRequestMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcModifyBearerRequestMessage> ();
  return tid;
}

GtpcModifyBearerRequestMessage::GtpcModifyBearerRequestMessage ()
{
  SetMessageType (GtpcHeader::ModifyBearerRequest);
  SetSequenceNumber (0);
  m_imsi = 0;
  m_uliEcgi = 0;
}

GtpcModifyBearerRequestMessage::~GtpcModifyBearerRequestMessage ()
{
}

TypeId
GtpcModifyBearerRequestMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcModifyBearerRequestMessage::GetMessageSize (void) const
{
  uint32_t serializedSize = serializedSizeImsi + serializedSizeUliEcgi
                          + m_bearerContextsToBeModified.size ()
                            * (serializedSizeBearerContextHeader
                               + serializedSizeEbi + serializedSizeFteid);
  return serializedSize;
}

uint32_t
GtpcModifyBearerRequestMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcModifyBearerRequestMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  SerializeImsi (i, m_imsi);
  SerializeUliEcgi (i, m_uliEcgi);

  for (auto &bc : m_bearerContextsToBeModified)
    {
      SerializeBearerContextHeader (i, serializedSizeEbi + serializedSizeFteid);

      SerializeEbi (i, bc.epsBearerId);
      SerializeFteid (i, bc.fteid);
    }
}

uint32_t
GtpcModifyBearerRequestMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  DeserializeImsi (i, m_imsi);
  DeserializeUliEcgi (i, m_uliEcgi);

  while (i.GetRemainingSize () > 0)
    {
      BearerContextToBeModified bearerContext;
      uint16_t length;

      DeserializeBearerContextHeader (i, length);

      DeserializeEbi (i, bearerContext.epsBearerId);
      DeserializeFteid (i, bearerContext.fteid);

      m_bearerContextsToBeModified.push_back (bearerContext);
    }

  return GetSerializedSize ();
}

void
GtpcModifyBearerRequestMessage::Print (std::ostream &os) const
{
  os << " imsi " << m_imsi << " uliEcgi " << m_uliEcgi;
}

uint64_t
GtpcModifyBearerRequestMessage::GetImsi () const
{
  return m_imsi;
}

void
GtpcModifyBearerRequestMessage::SetImsi (uint64_t imsi)
{
  m_imsi = imsi;
}

uint32_t
GtpcModifyBearerRequestMessage::GetUliEcgi () const
{
  return m_uliEcgi;
}

void
GtpcModifyBearerRequestMessage::SetUliEcgi (uint32_t uliEcgi)
{
  m_uliEcgi = uliEcgi;
}

std::list<GtpcModifyBearerRequestMessage::BearerContextToBeModified>
GtpcModifyBearerRequestMessage::GetBearerContextsToBeModified () const
{
  return m_bearerContextsToBeModified;
}

void
GtpcModifyBearerRequestMessage::SetBearerContextsToBeModified (std::list<GtpcModifyBearerRequestMessage::BearerContextToBeModified> bearerContexts)
{
  m_bearerContextsToBeModified = bearerContexts;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcModifyBearerResponseMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcModifyBearerResponseMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcModifyBearerResponseMessage> ();
  return tid;
}

GtpcModifyBearerResponseMessage::GtpcModifyBearerResponseMessage ()
{
  SetMessageType (GtpcHeader::ModifyBearerResponse);
  SetSequenceNumber (0);
  m_cause = Cause_t::RESERVED;
}

GtpcModifyBearerResponseMessage::~GtpcModifyBearerResponseMessage ()
{
}

TypeId
GtpcModifyBearerResponseMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcModifyBearerResponseMessage::GetMessageSize (void) const
{
  return serializedSizeCause;
}

uint32_t
GtpcModifyBearerResponseMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcModifyBearerResponseMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  SerializeCause (i, m_cause);
}

uint32_t
GtpcModifyBearerResponseMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  DeserializeCause (i, m_cause);

  return GetSerializedSize ();
}

void
GtpcModifyBearerResponseMessage::Print (std::ostream &os) const
{
   os << " cause " << (uint16_t)m_cause;
}

GtpcModifyBearerResponseMessage::Cause_t
GtpcModifyBearerResponseMessage::GetCause () const
{
  return m_cause;
}

void
GtpcModifyBearerResponseMessage::SetCause (GtpcModifyBearerResponseMessage::Cause_t cause)
{
  m_cause = cause;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcDeleteBearerCommandMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcDeleteBearerCommandMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcDeleteBearerCommandMessage> ();
  return tid;
}

GtpcDeleteBearerCommandMessage::GtpcDeleteBearerCommandMessage ()
{
  SetMessageType (GtpcHeader::DeleteBearerCommand);
  SetSequenceNumber (0);
}

GtpcDeleteBearerCommandMessage::~GtpcDeleteBearerCommandMessage ()
{
}

TypeId
GtpcDeleteBearerCommandMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcDeleteBearerCommandMessage::GetMessageSize (void) const
{
  uint32_t serializedSize = m_bearerContexts.size ()
                            * (serializedSizeBearerContextHeader + serializedSizeEbi);
  return serializedSize;
}

uint32_t
GtpcDeleteBearerCommandMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcDeleteBearerCommandMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  for (auto &bearerContext : m_bearerContexts)
    {
      SerializeBearerContextHeader (i, serializedSizeEbi);

      SerializeEbi (i, bearerContext.m_epsBearerId);
    }
}

uint32_t
GtpcDeleteBearerCommandMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  while (i.GetRemainingSize () > 0)
    {
      uint16_t length;
      DeserializeBearerContextHeader (i, length);

      BearerContext bearerContext;
      DeserializeEbi (i, bearerContext.m_epsBearerId);
      m_bearerContexts.push_back (bearerContext);
    }

  return GetSerializedSize ();
}

void
GtpcDeleteBearerCommandMessage::Print (std::ostream &os) const
{
  os << " bearerContexts [";
  for (auto &bearerContext : m_bearerContexts)
    {
      os << (uint16_t)bearerContext.m_epsBearerId << " ";
    }
  os << "]";
}

std::list<GtpcDeleteBearerCommandMessage::BearerContext>
GtpcDeleteBearerCommandMessage::GetBearerContexts () const
{
  return m_bearerContexts;
}

void
GtpcDeleteBearerCommandMessage::SetBearerContexts (std::list<GtpcDeleteBearerCommandMessage::BearerContext> bearerContexts)
{
  m_bearerContexts = bearerContexts;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcDeleteBearerRequestMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcDeleteBearerRequestMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcDeleteBearerRequestMessage> ();
  return tid;
}

GtpcDeleteBearerRequestMessage::GtpcDeleteBearerRequestMessage ()
{
  SetMessageType (GtpcHeader::DeleteBearerRequest);
  SetSequenceNumber (0);
}

GtpcDeleteBearerRequestMessage::~GtpcDeleteBearerRequestMessage ()
{
}

TypeId
GtpcDeleteBearerRequestMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcDeleteBearerRequestMessage::GetMessageSize (void) const
{
  uint32_t serializedSize = m_epsBearerIds.size () * serializedSizeEbi;
  return serializedSize;
}

uint32_t
GtpcDeleteBearerRequestMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcDeleteBearerRequestMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  for (auto &epsBearerId : m_epsBearerIds)
    {
      SerializeEbi (i, epsBearerId);
    }
}

uint32_t
GtpcDeleteBearerRequestMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  while (i.GetRemainingSize () > 0)
    {
      uint8_t epsBearerId;
      DeserializeEbi (i, epsBearerId);
      m_epsBearerIds.push_back (epsBearerId);
    }

  return GetSerializedSize ();
}

void
GtpcDeleteBearerRequestMessage::Print (std::ostream &os) const
{
  os << " epsBearerIds [";
  for (auto &epsBearerId : m_epsBearerIds)
    {
      os << (uint16_t)epsBearerId << " ";
    }
  os << "]";
}

std::list<uint8_t>
GtpcDeleteBearerRequestMessage::GetEpsBearerIds () const
{
  return m_epsBearerIds;
}

void
GtpcDeleteBearerRequestMessage::SetEpsBearerIds (std::list<uint8_t> epsBearerId)
{
  m_epsBearerIds = epsBearerId;
}

/////////////////////////////////////////////////////////////////////

TypeId
GtpcDeleteBearerResponseMessage::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GtpcDeleteBearerResponseMessage")
    .SetParent<Header> ()
    .SetGroupName ("Lte")
    .AddConstructor<GtpcDeleteBearerResponseMessage> ();
  return tid;
}

GtpcDeleteBearerResponseMessage::GtpcDeleteBearerResponseMessage ()
{
  SetMessageType (GtpcHeader::DeleteBearerResponse);
  SetSequenceNumber (0);
}

GtpcDeleteBearerResponseMessage::~GtpcDeleteBearerResponseMessage ()
{
}

TypeId
GtpcDeleteBearerResponseMessage::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
GtpcDeleteBearerResponseMessage::GetMessageSize (void) const
{
  uint32_t serializedSize = serializedSizeCause + m_epsBearerIds.size () * serializedSizeEbi;
  return serializedSize;
}

uint32_t
GtpcDeleteBearerResponseMessage::GetSerializedSize (void) const
{
  return GtpcHeader::GetSerializedSize () + GetMessageSize ();
}

void
GtpcDeleteBearerResponseMessage::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  GtpcHeader::PreSerialize (i);
  SerializeCause (i, m_cause);

  for (auto &epsBearerId : m_epsBearerIds)
    {
      SerializeEbi (i, epsBearerId);
    }
}

uint32_t
GtpcDeleteBearerResponseMessage::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  GtpcHeader::PreDeserialize (i);

  DeserializeCause (i, m_cause);

  while (i.GetRemainingSize () > 0)
    {
      uint8_t epsBearerId;
      DeserializeEbi (i, epsBearerId);
      m_epsBearerIds.push_back (epsBearerId);
    }

  return GetSerializedSize ();
}

void
GtpcDeleteBearerResponseMessage::Print (std::ostream &os) const
{
  os << " cause " << (uint16_t)m_cause << " epsBearerIds [";
  for (auto &epsBearerId : m_epsBearerIds)
    {
      os << (uint16_t)epsBearerId << " ";
    }
  os << "]";
}

GtpcDeleteBearerResponseMessage::Cause_t
GtpcDeleteBearerResponseMessage::GetCause () const
{
  return m_cause;
}

void
GtpcDeleteBearerResponseMessage::SetCause (GtpcDeleteBearerResponseMessage::Cause_t cause)
{
  m_cause = cause;
}

std::list<uint8_t>
GtpcDeleteBearerResponseMessage::GetEpsBearerIds () const
{
  return m_epsBearerIds;
}

void
GtpcDeleteBearerResponseMessage::SetEpsBearerIds (std::list<uint8_t> epsBearerId)
{
  m_epsBearerIds = epsBearerId;
}

} // namespace ns3
