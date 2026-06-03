/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2009 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *
 */

#ifndef WIMAX_TLV_H
#define WIMAX_TLV_H

#define WIMAX_TLV_EXTENDED_LENGTH_MASK 0x80

#include "ns3/ipv4-address.h"
#include <cstdlib>
#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/uinteger.h"
#include "ns3/header.h"
#include <vector>

namespace ns3 {

/**
 * \ingroup wimax
 * The value field of a tlv can take different values (uint8_t, uint16, 
 * vector, ...). This class is a virtual interface
 * from which all the types of tlv values should derive
 */
class TlvValue
{
public:
  virtual ~TlvValue ()
  {
  }
  /**
   * Get serialized size in bytes
   * \returns the serialized size
   */
  virtual uint32_t GetSerializedSize (void) const = 0;
  /**
   * Serialize to a buffer
   * \param start the iterator
   */
  virtual void Serialize (Buffer::Iterator start) const = 0;
  /**
   * Deserialize from a buffer
   * \param start the iterator
   * \param valueLen the maximum length of the value
   * \returns the  
   */
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLen ) = 0;
  /**
   * Copy function
   * \returns the TLV value
   */
  virtual TlvValue * Copy (void) const = 0;
private:
};


// =============================================================================
/**
 * \ingroup wimax
 * \brief This class implements the Type-Len-Value structure channel encodings as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 11. TLV encodings, page 645
 *
 */
class Tlv : public Header
{
public:
  /// CommonTypes enumeration
  enum CommonTypes
  {
    HMAC_TUPLE = 149,
    MAC_VERSION_ENCODING = 148,
    CURRENT_TRANSMIT_POWER = 147,
    DOWNLINK_SERVICE_FLOW = 146,
    UPLINK_SERVICE_FLOW = 145,
    VENDOR_ID_EMCODING = 144,
    VENDOR_SPECIFIC_INFORMATION = 143
  };
  /**
   * Constructor
   *
   * \param type type
   * \param length the length
   * \param value TLV value
   */
  Tlv (uint8_t type, uint64_t length, const TlvValue & value);
  Tlv (void);
  ~Tlv (void);
  /**
   * Register this type.
   * \return the TypeId.
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  /**
   * Get size of length field
   * \returns the size of length field
   */
  uint8_t GetSizeOfLen (void) const;
  /**
   * Get type value
   * \returns the type
   */
  uint8_t GetType (void) const;
  /**
   * Get length value
   * \returns the length
   */
  uint64_t GetLength (void) const;
  /**
   * Peek value 
   * \returns the TLV value
   */
  TlvValue* PeekValue (void);
  /**
   * Copy TLV
   * \returns a pointer to a TLV copy
   */
  Tlv * Copy (void) const;
  /**
   * Copy TlvValue
   * \returns the TLV value
   */
  TlvValue * CopyValue (void) const;
  /**
   * assignment operator
   * \param o the TLV to assign
   * \returns the TLV
   */
  Tlv &operator = (Tlv const& o);
  /**
   * type conversion operator
   * \param tlv the TLV
   */
  Tlv (const Tlv & tlv);

private:
  uint8_t m_type; ///< type
  uint64_t m_length; ///< length
  TlvValue * m_value; ///< value
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief U8TlvValue class 
 */
class U8TlvValue : public TlvValue
{
public:
  /**
   * Constructor
   *
   * \param value value to encode
   */
  U8TlvValue (uint8_t value);
  U8TlvValue ();
  ~U8TlvValue (void);
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start,uint64_t valueLen);
  /**
   * Deserialize from a buffer
   * \param start the iterator
   * \returns the size of the item
   */
  uint32_t Deserialize (Buffer::Iterator start);
  /**
   * Get value
   * \returns the value
   */
  uint8_t GetValue (void) const;
  /**
   * Copy 
   * \returns a U8 TLV value
   */
  U8TlvValue * Copy (void) const;
private:
  uint8_t  m_value; ///< value
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief U16TlvValue class 
 */
class U16TlvValue : public TlvValue
{
public:
  /**
   * Constructor
   *
   * \param value value to encode
   */
  U16TlvValue (uint16_t value);
  U16TlvValue ();
  ~U16TlvValue (void);
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start,uint64_t valueLen);
  /**
   * Deserialize from a buffer
   * \param start the iterator
   * \returns the size
   */
  uint32_t Deserialize (Buffer::Iterator start);
  /**
   * Get value 
   * \returns the value
   */
  uint16_t GetValue (void) const;
  /**
   * Copy 
   * \returns the U16 TLV value
   */
  virtual U16TlvValue * Copy (void) const;
private:
  uint16_t  m_value; ///< value
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief U32TlvValue class 
 */
class U32TlvValue : public TlvValue
{
public:
  /**
   * Constructor
   *
   * \param value to encode
   */
  U32TlvValue (uint32_t value);
  U32TlvValue ();
  ~U32TlvValue (void);

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLen);
  /**
   * Deserialize from a buffer
   * \param start the iterator
   * \returns the size
   */
  uint32_t Deserialize (Buffer::Iterator start);
  /**
   * Get value
   * \returns the value
   */
  uint32_t GetValue (void) const;
  /**
   * Copy
   * \returns the U32 TLV Value
   */
  virtual U32TlvValue * Copy (void) const;
private:
  uint32_t  m_value; ///< value
};

// ==============================================================================

/**
 * \ingroup wimax
 * \brief this class is used to implement a vector of values in one tlv value field
 */
class VectorTlvValue : public TlvValue
{
public:
  /// TLV vector iterator typedef
  typedef std::vector<Tlv*>::const_iterator Iterator;
  VectorTlvValue (void);
  ~VectorTlvValue (void);
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength) = 0;
  /**
   * Begin iterator
   * \returns the beginning element
   */
  Iterator Begin () const;
  /**
   * End iterator
   * \returns the ending element
   */
  Iterator End () const;
  /**
   * Add a TLV
   * \param val the TLV value
   */
  void Add (const Tlv & val);
  /**
   * Copy 
   * \returns the vector TLV value
   */
  virtual VectorTlvValue * Copy (void) const = 0;
private:
  std::vector<Tlv*>  * m_tlvList; ///< tlv list
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief SfVectorTlvValue class 
 */
class SfVectorTlvValue : public VectorTlvValue
{

public:
  /// Type enumeration
  enum Type
  {
    SFID = 1,
    CID = 2,
    Service_Class_Name = 3,
    reserved1 = 4,
    QoS_Parameter_Set_Type = 5,
    Traffic_Priority = 6,
    Maximum_Sustained_Traffic_Rate = 7,
    Maximum_Traffic_Burst = 8,
    Minimum_Reserved_Traffic_Rate = 9,
    Minimum_Tolerable_Traffic_Rate = 10,
    Service_Flow_Scheduling_Type = 11,
    Request_Transmission_Policy = 12,
    Tolerated_Jitter = 13,
    Maximum_Latency = 14,
    Fixed_length_versus_Variable_length_SDU_Indicator = 15,
    SDU_Size = 16,
    Target_SAID = 17,
    ARQ_Enable = 18,
    ARQ_WINDOW_SIZE = 19,
    ARQ_RETRY_TIMEOUT_Transmitter_Delay = 20,
    ARQ_RETRY_TIMEOUT_Receiver_Delay = 21,
    ARQ_BLOCK_LIFETIME = 22,
    ARQ_SYNC_LOSS = 23,
    ARQ_DELIVER_IN_ORDER = 24,
    ARQ_PURGE_TIMEOUT = 25,
    ARQ_BLOCK_SIZE = 26,
    reserved2 = 27,
    CS_Specification = 28,
    IPV4_CS_Parameters = 100
  };
  SfVectorTlvValue ();
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  virtual SfVectorTlvValue * Copy (void) const;

};
// ==============================================================================

/**
 * \ingroup wimax
 * \brief this class implements the convergence sub-layer descriptor as a tlv vector
 */
class CsParamVectorTlvValue : public VectorTlvValue
{
public:
  /// Type enumeration
  enum Type
  {
    Classifier_DSC_Action = 1,
    Packet_Classification_Rule = 3,
  };
  CsParamVectorTlvValue ();
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  virtual CsParamVectorTlvValue * Copy (void) const;
private:
};

// ==============================================================================

/**
 * \ingroup wimax
 * \brief this class implements the classifier descriptor as a tlv vector
 */
class ClassificationRuleVectorTlvValue : public VectorTlvValue
{
public:
  /// ClassificationRuleTlvType enumeration
  enum ClassificationRuleTlvType
  {
    Priority = 1,
    ToS = 2,
    Protocol = 3,
    IP_src = 4,
    IP_dst = 5,
    Port_src = 6,
    Port_dst = 7,
    Index = 14,
  };
  ClassificationRuleVectorTlvValue ();
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  virtual ClassificationRuleVectorTlvValue * Copy (void) const;
private:
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief TosTlvValue class
 */
class TosTlvValue : public TlvValue
{
public:
  TosTlvValue ();
  /**
   * Constructor
   *
   * \param low low value
   * \param high high value
   * \param mask the mask
   */
  TosTlvValue (uint8_t low, uint8_t high, uint8_t mask);
  ~TosTlvValue ();
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  /**
   * Get low part
   * \returns the low part
   */
  uint8_t GetLow (void) const;
  /**
   * Get high part
   * \returns the high part
   */
  uint8_t GetHigh (void) const;
  /**
   * Get the mask
   * \returns the mask
   */
  uint8_t GetMask (void) const;
  /**
   * Copy
   * \returns the TOS TLV value
   */
  virtual TosTlvValue * Copy () const;
private:
  uint8_t m_low; ///< low
  uint8_t m_high; ///< high
  uint8_t m_mask; ///< mask
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief PortRangeTlvValue class 
 */
class PortRangeTlvValue : public TlvValue
{
public:
  /// PortRange structure
  struct PortRange
  {
    uint16_t PortLow; ///< low
    uint16_t PortHigh; ///< high
  };
  /// PortRange vector iterator typedef
  typedef std::vector<struct PortRange>::const_iterator Iterator;
  PortRangeTlvValue ();
  ~PortRangeTlvValue ();
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  /**
   * Add a range
   * \param portLow the low port of the range
   * \param portHigh the high port of the range
   */
  void Add (uint16_t portLow, uint16_t portHigh);
  /**
   * Begin iterator
   * \returns the beginning element
   */
  Iterator Begin () const;
  /**
   * End iterator
   * \returns the ending element
   */
  Iterator End () const;
  /**
   * Copy
   * \returns the port range tlv value
   */
  virtual PortRangeTlvValue * Copy (void) const;
private:
  std::vector<struct PortRange> * m_portRange; ///< port range
};

// ==============================================================================
/**
 * \ingroup wimax
 * \brief ProtocolTlvValue class
 */
class ProtocolTlvValue : public TlvValue
{
public:
  ProtocolTlvValue ();
  ~ProtocolTlvValue ();
  /// Iterator typedef
  typedef std::vector<uint8_t>::const_iterator Iterator;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  /**
   * Add protocol number
   * \param protocol the protocol number
   */
  void Add (uint8_t protocol);
  /**
   * Begin iterator
   * \returns the beginning element
   */
  Iterator Begin () const;
  /**
   * End iterator
   * \return the ending element
   */
  Iterator End () const;
  /**
   * Copy
   * \returns the protocol tlv value
   */
  virtual ProtocolTlvValue * Copy (void) const;
private:
  std::vector<uint8_t> * m_protocol; ///< protocol
};

// ==============================================================================

/**
 * \ingroup wimax
 * \brief Ipv4AddressTlvValue class
 */
class Ipv4AddressTlvValue : public TlvValue
{
public:
  ///ipv4Addr structure
  struct ipv4Addr
  {
    Ipv4Address Address; ///< address
    Ipv4Mask Mask; ///< mask
  };
  /// IPv4 address vector iterator typedef
  typedef std::vector<struct ipv4Addr>::const_iterator Iterator;
  Ipv4AddressTlvValue ();
  ~Ipv4AddressTlvValue ();
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start, uint64_t valueLength);
  /**
   * Add IPv4 address and mask
   * \param address the IPv4 address
   * \param Mask the IPv4 mask
   */
  void Add (Ipv4Address address, Ipv4Mask Mask);
  /**
   * Begin iterator
   * \returns the beginning element
   */
  Iterator Begin () const;
  /**
   * End iterator
   * \returns the ending element
   */
  Iterator End () const;
  virtual Ipv4AddressTlvValue * Copy () const;
private:
  std::vector<struct ipv4Addr> * m_ipv4Addr; ///< ipv4 addr 
};

}

#endif /* WIMAX_TLV_H */
