/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009, 2010 CTTC
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


#ifndef ALOHA_NOACK_MAC_HEADER_H
#define ALOHA_NOACK_MAC_HEADER_H

#include <ns3/header.h>
#include <ns3/mac48-address.h>
#include <ns3/address-utils.h>

namespace ns3 {

/**
 * \ingroup spectrum
 *  Header for the AlohaNoack NetDevice
 *
 */
class AlohaNoackMacHeader : public Header
{
public:
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
   * Set the source address
   * \param source the source address
   */
  void SetSource (Mac48Address source);
  /**
   * Set the destination address
   * \param destination the destination address
   */
  void SetDestination (Mac48Address destination);

  /**
   * Get the source address
   * \returns the source address
   */
  Mac48Address GetSource () const;
  /**
   * Get the destination address
   * \returns the destination address
   */
  Mac48Address GetDestination () const;

private:
  Mac48Address m_source;      //!< source address
  Mac48Address m_destination; //!< destination address
};



} // namespace ns3

#endif /* ALOHA_NOACK_MAC_HEADER_H */
