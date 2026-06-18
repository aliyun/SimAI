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
 */


#ifndef COMPONENT_CARRIER_H
#define COMPONENT_CARRIER_H

#include <ns3/object.h>

namespace ns3 {

/**
 * \ingroup lte
 *
 * ComponentCarrier Object, it defines a single Carrier
 * This is the parent class for both ComponentCarrierBaseStation
 * and ComponentCarrierUe.
 * This class contains the main physical configuration
 * parameters for a carrier. Does not contain pointers to
 * the MAC/PHY objects of the carrier.
 */
class ComponentCarrier : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  ComponentCarrier ();

  virtual ~ComponentCarrier (void);
  virtual void DoDispose (void);

  /**
   * \return the uplink bandwidth in RBs
   */
  uint16_t GetUlBandwidth () const;

  /**
   * \param bw the uplink bandwidth in RBs
   */
  virtual void SetUlBandwidth (uint16_t bw);

  /**
   * \return the downlink bandwidth in RBs
   */
  uint16_t GetDlBandwidth () const;

  /**
   * \param bw the downlink bandwidth in RBs
   */
  virtual void SetDlBandwidth (uint16_t bw);

  /**
   * \return the downlink carrier frequency (EARFCN)
   */
  uint32_t GetDlEarfcn () const;

  /**
   * \param earfcn the downlink carrier frequency (EARFCN)
   */
  void SetDlEarfcn (uint32_t earfcn);

  /**
   * \return the uplink carrier frequency (EARFCN)
   */
  uint32_t GetUlEarfcn () const;

  /**
   * \param earfcn the uplink carrier frequency (EARFCN)
   */
  void SetUlEarfcn (uint32_t earfcn);

  /**
   * \brief Returns the CSG ID of the eNodeB.
   * \return the Closed Subscriber Group identity
   * \sa LteEnbNetDevice::SetCsgId
   */
  uint32_t GetCsgId () const;

  /**
   * \brief Associate the eNodeB device with a particular CSG.
   * \param csgId the intended Closed Subscriber Group identity
   *
   * CSG identity is a number identifying a Closed Subscriber Group which the
   * cell belongs to. eNodeB is associated with a single CSG identity.
   *
   * The same CSG identity can also be associated to several UEs, which is
   * equivalent as enlisting these UEs as the members of this particular CSG.
   *
   * \sa LteEnbNetDevice::SetCsgIndication
   */
  void SetCsgId (uint32_t csgId);

  /**
   * \brief Returns the CSG indication flag of the eNodeB.
   * \return the CSG indication flag
   * \sa LteEnbNetDevice::SetCsgIndication
   */
  bool GetCsgIndication () const;

  /**
   * \brief Enable or disable the CSG indication flag.
   * \param csgIndication if TRUE, only CSG members are allowed to access this
   *                      cell
   *
   * When the CSG indication field is set to TRUE, only UEs which are members of
   * the CSG (i.e. same CSG ID) can gain access to the eNodeB, therefore
   * enforcing closed access mode. Otherwise, the eNodeB operates as a non-CSG
   * cell and implements open access mode.
   *
   * \note This restriction only applies to initial cell selection and
   *       EPC-enabled simulation.
   *
   * \sa LteEnbNetDevice::SetCsgIndication
   */
  void SetCsgIndication (bool csgIndication);

  /**
   * \brief Set as primary carrier 
   * \param primaryCarrier true to set as primary carrier
   */
  void SetAsPrimary (bool primaryCarrier);

  /**
   * \brief Checks if the carrier is the primary carrier
   * \returns true if the carrier is primary
   */
  bool IsPrimary () const;

protected:
  uint32_t m_csgId {0}; ///< CSG ID
  bool m_csgIndication {false}; ///< CSG indication

  bool m_primaryCarrier {false}; ///< whether the carrier is primary

  uint16_t m_dlBandwidth {0};   ///< downlink bandwidth in RBs */
  uint16_t m_ulBandwidth {0};   ///< uplink bandwidth in RBs */

  uint32_t m_dlEarfcn {0};    ///< downlink carrier frequency */
  uint32_t m_ulEarfcn {0};    ///< uplink carrier frequency */
};

/**
 * \ingroup lte
 *
 * Defines a Base station, that is a ComponentCarrier but with a cell Id.
 *
 */
class ComponentCarrierBaseStation : public ComponentCarrier
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Constructor
   */
  ComponentCarrierBaseStation ();

  /**
   * \brief ~ComponentCarrierBaseStation
   */
  virtual ~ComponentCarrierBaseStation (void) override;

  /**
   * Get cell identifier
   * \return cell identifier
   */
  uint16_t GetCellId ();

  /**
   * Set physical cell identifier
   * \param cellId cell identifier
   */
  void SetCellId (uint16_t cellId);

protected:
  uint16_t m_cellId {0}; ///< Cell identifier
};

} // namespace ns3



#endif /* COMPONENT_CARRIER_H */
