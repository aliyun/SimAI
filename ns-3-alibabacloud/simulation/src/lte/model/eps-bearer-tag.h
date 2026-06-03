/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011,2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */
#ifndef EPS_BEARER_TAG_H
#define EPS_BEARER_TAG_H

#include "ns3/tag.h"

namespace ns3 {

class Tag;


/**
 * Tag used to define the RNTI and EPS bearer ID for packets
 * interchanged between the EpcEnbApplication and the LteEnbNetDevice 
 */

class EpsBearerTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * Create an empty EpsBearerTag
   */
  EpsBearerTag ();

  /**
   * Create a EpsBearerTag with the given RNTI and bearer id
   *
   * @param rnti the value of the RNTI to set
   * @param bid the value of the Bearer Id to set
   */
  EpsBearerTag (uint16_t  rnti, uint8_t bid);
  
  /**
   * Set the RNTI to the given value.
   *
   * @param rnti the value of the RNTI to set
   */
  void SetRnti (uint16_t rnti);

  /**
   * Set the bearer id to the given value.
   *
   * @param bid the value of the Bearer Id to set
   */
  void SetBid (uint8_t bid);

  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual uint32_t GetSerializedSize () const;
  virtual void Print (std::ostream &os) const;

  /**
   * Get RNTI function
   * \returns the RNTI
   */
  uint16_t GetRnti (void) const;
  /**
   * Get Bearer Id function
   * \returns the Bearer Id
   */
  uint8_t GetBid (void) const;

private:
  uint16_t m_rnti; ///< RNTI value
  uint8_t m_bid; ///< Bearer Id value

};



} // namespace ns3

#endif /* EPS_BEARER_TAG_H */
