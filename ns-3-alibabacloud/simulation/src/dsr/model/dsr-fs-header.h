/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Yufei Cheng
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
 * Author: Yufei Cheng   <yfcheng@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#ifndef DSR_FS_HEADER_H
#define DSR_FS_HEADER_H

#include <vector>
#include <list>
#include <ostream>

#include "ns3/header.h"
#include "ns3/ipv4-address.h"
#include "dsr-option-header.h"

namespace ns3 {
namespace dsr {
/**
 * \class DsrHeader
 * \brief Header for Dsr Routing.
 */

/**
* \ingroup dsr
* \brief Dsr fixed size header Format
  \verbatim
   |      0        |      1        |      2        |      3        |
   0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |  Next Header |F|     Reservd    |       Payload Length       |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                            Options                           |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  \endverbatim
*/

/**
* \ingroup dsr
* \brief The modified version of Dsr fixed size header Format
  \verbatim
   |      0        |      1        |      2        |      3        |
   0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |  Next Header |F|  Message Type  |       Payload Length       |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |             Source Id           |            Dest Id         |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                            Options                           |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  \endverbatim
*/
class DsrFsHeader : public Header
{
public:
  /**
   * \brief Get the type identificator.
   * \return type identificator
   */
  static TypeId GetTypeId ();
  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;
  /**
   * \brief Constructor.
   */
  DsrFsHeader ();
  /**
   * \brief Destructor.
   */
  virtual ~DsrFsHeader ();
  /**
   * \brief Set the "Next header" field.
   * \param protocol the next header number
   */
  void SetNextHeader (uint8_t protocol);
  /**
   * \brief Get the next header.
   * \return the next header number
   */
  uint8_t GetNextHeader () const;
  /**
   * brief Set the message type of the header.
   * \param messageType the message type of the header
   */
  void SetMessageType (uint8_t messageType);
  /**
   * brief Get the message type of the header.
   * \return message type the message type of the header
   */
  uint8_t GetMessageType () const;
  /**
   * brief Set the source ID of the header.
   * \param sourceId the source ID of the header
   */
  void SetSourceId (uint16_t sourceId);
  /**
   * brief Get the source ID of the header.
   * \return source ID the source ID of the header
   */
  uint16_t GetSourceId () const;
  /**
   * brief Set the dest ID of the header.
   * \param destId the destination ID of the header
   */
  void SetDestId (uint16_t destId);
  /**
   * brief Get the dest ID of the header.
   * \return dest ID the dest ID of the header
   */
  uint16_t GetDestId () const;
  /**
   * brief Set the payload length of the header.
   * \param length the payload length of the header in bytes
   */
  void SetPayloadLength (uint16_t length);
  /**
   * \brief Get the payload length of the header.
   * \return the payload length of the header
   */
  uint16_t GetPayloadLength () const;
  /**
   * \brief Print some information about the packet.
   * \param os output stream
   */
  virtual void Print (std::ostream &os) const;
  /**
   * \brief Get the serialized size of the packet.
   * \return size
   */
  virtual uint32_t GetSerializedSize () const;
  /**
   * \brief Serialize the packet.
   * \param start Buffer iterator
   */
  virtual void Serialize (Buffer::Iterator start) const;
  /**
   * \brief Deserialize the packet.
   * \param start Buffer iterator
   * \return size of the packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The "next header" field.
   */
  uint8_t m_nextHeader;
  /**
   * \brief The type of the message.
   */
  uint8_t m_messageType;
  /**
   * \brief The "payload length" field.
   */
  uint16_t m_payloadLen;
  /**
   * \brief The source node id
   */
  uint16_t m_sourceId;
  /**
   * \brief The destination node id
   */
  uint16_t m_destId;
  /**
   * \brief The data of the extension.
   */
  Buffer m_data;
};

/**
 * \class DsrOptionField
 * \brief Option field for an DsrFsHeader
 * Enables adding options to an DsrFsHeader
 *
 * Implementor's note: Make sure to add the result of
 * OptionField::GetSerializedSize () to your DsrFsHeader::GetSerializedSize ()
 * return value. Call OptionField::Serialize and OptionField::Deserialize at the
 * end of your corresponding DsrFsHeader methods.
 */
class DsrOptionField
{
public:
  /**
   * \brief Constructor.
   * \param optionsOffset option offset
   */
  DsrOptionField (uint32_t optionsOffset);
  /**
   * \brief Destructor.
   */
  ~DsrOptionField ();
  /**
   * \brief Get the serialized size of the packet.
   * \return size
   */
  uint32_t GetSerializedSize () const;
  /**
   * \brief Serialize all added options.
   * \param start Buffer iterator
   */
  void Serialize (Buffer::Iterator start) const;
  /**
   * \brief Deserialize the packet.
   * \param start Buffer iterator
   * \param length length
   * \return size of the packet
   */
  uint32_t Deserialize (Buffer::Iterator start, uint32_t length);
  /**
   * \brief Serialize the option, prepending pad1 or padn option as necessary
   * \param option the option header to serialize
   */
  void AddDsrOption (DsrOptionHeader const& option);
  /**
   * \brief Get the offset where the options begin, measured from the start of
   * the extension header.
   * \return the offset from the start of the extension header
   */
  uint32_t GetDsrOptionsOffset ();
  /**
   * \brief Get the buffer.
   * \return buffer
   */
  Buffer GetDsrOptionBuffer ();

private:
  /**
   * \brief Calculate padding.
   * \param alignment alignment
   * \return the number of bytes required to pad
   */
  uint32_t CalculatePad (DsrOptionHeader::Alignment alignment) const;
  /**
   * \brief Data payload.
   */
  Buffer m_optionData;
  /**
   * \brief Offset.
   */
  uint32_t m_optionsOffset;
};

/**
 * \class DsrRoutingHeader
 * \brief Header of Dsr Routing
 */
class DsrRoutingHeader : public DsrFsHeader,
                         public DsrOptionField
{
public:
  /**
   * \brief Get the type identificator.
   * \return type identificator
   */
  static TypeId GetTypeId ();
  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;
  /**
   * \brief Constructor.
   */
  DsrRoutingHeader ();
  /**
   * \brief Destructor.
   */
  virtual ~DsrRoutingHeader ();
  /**
   * \brief Print some information about the packet.
   * \param os output stream
   */
  virtual void Print (std::ostream &os) const;
  /**
   * \brief Get the serialized size of the packet.
   * \return size
   */
  virtual uint32_t GetSerializedSize () const;
  /**
   * \brief Serialize the packet.
   * \param start Buffer iterator
   */
  virtual void Serialize (Buffer::Iterator start) const;
  /**
   * \brief Deserialize the packet.
   * \param start Buffer iterator
   * \return size of the packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);
};

static inline std::ostream & operator<< (std::ostream& os, const DsrRoutingHeader & dsr)
{
  dsr.Print (os);
  return os;
}

}  // namespace dsr
}  // namespace ns3

#endif /* DSR_FS_HEADER_H */

