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
 */

#ifndef LTE_PDCP_HEADER_H
#define LTE_PDCP_HEADER_H

#include "ns3/header.h"

#include <list>

namespace ns3 {

/**
 * \ingroup lte
 * \brief The packet header for the Packet Data Convergence Protocol (PDCP) packets
 *
 * This class has fields corresponding to those in an PDCP header as well as
 * methods for serialization to and deserialization from a byte buffer.
 * It follows 3GPP TS 36.323 Packet Data Convergence Protocol (PDCP) specification.
 */
class LtePdcpHeader : public Header
{
public:

  /**
   * \brief Constructor
   *
   * Creates a null header
   */
  LtePdcpHeader ();
  ~LtePdcpHeader ();

  /**
   * \brief Set DC bit
   *
   * \param dcBit DC bit to set
   */
  void SetDcBit (uint8_t dcBit);
  /**
   * \brief Set sequence number
   *
   * \param sequenceNumber sequence number
   */
  void SetSequenceNumber (uint16_t sequenceNumber);

  /**
   * \brief Get DC bit
   *
   * \returns DC bit
   */
  uint8_t GetDcBit () const;
  /**
   * \brief Get sequence number
   *
   * \returns sequence number
   */
  uint16_t GetSequenceNumber () const;

  /// DcBit_t typedef
  enum {
    CONTROL_PDU   = 0,
    DATA_PDU      = 1
  } DcBit_t; ///< DcBit_t typedef 

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  uint8_t m_dcBit; ///< the DC bit
  uint16_t m_sequenceNumber; ///< the sequence number

};

} // namespace ns3

#endif // LTE_PDCP_HEADER_H
