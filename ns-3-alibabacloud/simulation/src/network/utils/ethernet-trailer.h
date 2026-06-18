/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Emmanuelle Laprise
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
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#ifndef ETHERNET_TRAILER_H
#define ETHERNET_TRAILER_H

#include "ns3/trailer.h"
#include "ns3/packet.h"
#include <string>

namespace ns3 {


/**
 * \ingroup network
 *
 * \brief Packet trailer for Ethernet
 *
 * This class can be used to add and verify the FCS at the end of an
 * Ethernet packet.
 */
class EthernetTrailer : public Trailer 
{
public:
  /**
   * \brief Construct a null ethernet trailer
   */
  EthernetTrailer ();

  /**
   * \brief Enable or disable FCS checking and calculations
   * \param enable If true, enables FCS calculations.
   */
  void EnableFcs (bool enable);

  /**
   * \brief Updates the Fcs Field to the correct FCS
   * \param p Reference to a packet on which the FCS should be
   * calculated. The packet should not currently contain an 
   * EthernetTrailer.
   */
  void CalcFcs (Ptr<const Packet> p);

  /**
   * \brief Sets the FCS to a new value
   * \param fcs New FCS value
   */
  void SetFcs (uint32_t fcs);

  /**
   * \return the FCS contained in this trailer
   */
  uint32_t GetFcs () const;

  /**
   * Calculate an FCS on the provided packet and check this value against
   * the FCS found when the trailer was deserialized (the one in the transmitted
   * packet).
   *
   * If FCS checking is disabled, this method will always
   * return true.
   *
   * \param p Reference to the packet on which the FCS should be
   * calculated. The packet should not contain an EthernetTrailer.
   *
   * \return Returns true if the Packet FCS matches the FCS in the trailer, 
   * false otherwise.
   */
  bool CheckFcs (Ptr<const Packet> p) const;

  /**
   *\return Returns the size of the trailer
   */ 
  uint32_t GetTrailerSize () const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator end) const;
  virtual uint32_t Deserialize (Buffer::Iterator end);
private:
  /**
   * Enabled FCS calculations. If false, m_fcs is set to 0 and CheckFcs
   * returns true.
   */
  bool m_calcFcs;
  uint32_t m_fcs; //!< Value of the fcs contained in the trailer

};

} // namespace ns3

#endif /* ETHERNET_TRAILER_H */
