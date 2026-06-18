/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Ritsumeikan University, Shiga, Japan.
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
 * Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 *
 * This file implements Information Fields present in IEEE 802.15.4-2011.
 * Information Fields are in practice similar to the Information Elements(IE)
 * introduced in later revisions of the standard, however, they lack
 * descriptors and common format unlike the IEs. To keep this implementation
 * consistent with the IEEE 802.15.4-2011 std. the present file implements
 * Information Fields not Information Elements.
 */
#ifndef LR_WPAN_FIELDS_H
#define LR_WPAN_FIELDS_H


#include <ns3/mac16-address.h>
#include <ns3/mac64-address.h>
#include "ns3/buffer.h"
#include <array>


namespace ns3 {

/**
 * \ingroup lr-wpan
 * The device Capabilities.
 */
enum DeviceType
{
  RFD = 0,                  //!< Reduced Functional Device (RFD)
  FFD = 1                   //!< Full Functional Device (FFD)
};

/**
 * \ingroup lr-wpan
 * Represent the Superframe Specification information field.
 * See IEEE 802.15.4-2011   Section 5.2.2.1.2 Figure 41
 */
class SuperframeField
{

public:
  SuperframeField ();
  /**
   * Set the whole Superframe Specification Information field.
   * \param superFrm The Superframe Specification information field.
   */
  void SetSuperframe (uint16_t superFrm);
  /**
   * Set the superframe specification Beacon Order field.
   * \param bcnOrder The beacon order value to set in the superframe.
   */
  void SetBeaconOrder (uint8_t bcnOrder);
  /**
   * Set the superframe specification Superframe Order field.
   * \param frmOrder The frame Order value to set on the superframe.
   */
  void SetSuperframeOrder (uint8_t frmOrder);
  /**
   * Set the superframe specification Final CAP slot field.
   * \param capSlot Set the final slot of the Contention Access Period (CAP).
   */
  void SetFinalCapSlot (uint8_t capSlot);
  /**
   * Set the Superframe Specification Battery Life Extension (BLE).
   * \param battLifeExt Sets true or false the value of the Battery Life Extension flag of the superframe field.
   */
  void SetBattLifeExt (bool battLifeExt);
  /**
   * Set the Superframe Specification PAN coordinator field.
   * \param panCoor set true or false the value for the PAN Coordinator flag of the superframe field.
   */
  void SetPanCoor (bool panCoor);
  /**
   * Set the Superframe Specification Association Permit field.
   * \param assocPermit set true or false the value of the Association Permit flag of the superframe field.
   */
  void SetAssocPermit (bool assocPermit);
  /**
   * Get the Superframe Specification Beacon Order field.
   * \return the Superframe Specification Beacon Order field.
   */
  uint8_t GetBeaconOrder (void) const;
  /**
   * Get the Superframe Specification Frame Order field.
   * \return The Superframe Specification Frame Order field.
   */
  uint8_t GetFrameOrder (void) const;
  /**
   * Get the the Final CAP Slot.
   * \returns The Final CAP Slot
   */
  uint8_t GetFinalCapSlot (void) const;
  /**
   * Check if the Battery Life Extension bit is enabled.
   * \returns true if the Battery Life Extension bit is enabled
   */
  bool IsBattLifeExt (void) const;
  /**
   * Check if the PAN Coordinator bit is enabled.
   * \returns true if the PAN Coordinator bit is enabled
   */
  bool IsPanCoor (void) const;
  /**
   * Check if the Association Permit bit is enabled.
   * \returns true if the Association Permit bit is enabled
   */
  bool IsAssocPermit (void) const;
  /**
   * Get the Superframe specification information field.
   * \return the Superframe Specification Information field bits.
   */
  uint16_t GetSuperframe (void) const;
  /**
   * Get the size of the serialized Superframe specification information field.
   * \return the size of the serialized field.
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * Serialize the entire superframe specification field.
   * \param i an iterator which points to where the superframe specification field should be written.
   * \return an iterator.
   */
  Buffer::Iterator Serialize (Buffer::Iterator i) const;
  /**
   * Deserialize the entire superframe specification field.
   * \param i an iterator which points to where the superframe specification field should be read.
   * \return an iterator.
   */
  Buffer::Iterator Deserialize (Buffer::Iterator i);


private:
  // Superframe Specification field
  // See IEEE 802.14.15-2011 5.2.2.1.2
  uint8_t m_sspecBcnOrder;    //!< Superframe Specification field Beacon Order (Bit 0-3)
  uint8_t m_sspecSprFrmOrder; //!< Superframe Specification field Superframe Order (Bit 4-7)
  uint8_t m_sspecFnlCapSlot;  //!< Superframe Specification field Final CAP slot (Bit 8-11)
  bool    m_sspecBatLifeExt;  //!< Superframe Specification field Battery Life Extension (Bit 12)
                              //!< Superframe Specification field Reserved (not necessary) (Bit 13)
  bool    m_sspecPanCoor;     //!< Superframe Specification field PAN Coordinator (Bit 14)
  bool    m_sspecAssocPermit; //!< Superframe Specification field Association Permit (Bit 15)

};

/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] superframeField The Superframe fields.
 * \returns The reference to the output stream.
 */
std::ostream &operator << (std::ostream &os, const SuperframeField &superframeField);

/**
 * \ingroup lr-wpan
 * Represent the GTS information fields.
 * See IEEE 802.15.4-2011   Section 5.2.2 Figure 39
 */
class GtsFields
{

public:
  GtsFields ();
  /**
   * Get the GTS Specification Field from the GTS Fields
   * \return The GTS Spcecification Field
   */
  uint8_t GetGtsSpecField (void) const;
  /**
   * Get the GTS Direction Field from the GTS Fields
   * \return The GTS Direction Field
   */
  uint8_t GetGtsDirectionField (void) const;
  /**
   * Set the GTS Specification Field to the GTS Fields
   * \param gtsSpec The GTS Specification Field to set.
   */
  void SetGtsSpecField (uint8_t gtsSpec);
  /**
   * Set the GTS direction field to the GTS Fields
   * \param gtsDir The GTS Direction Field to set
   */
  void SetGtsDirectionField (uint8_t gtsDir);
  /**
   * Get the size of the serialized GTS fields.
   * \return the size of the serialized fields.
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * Serialize the entire GTS fields.
   * \param i an iterator which points to where the superframe specification field should be written.
   * \return an iterator.
   */
  Buffer::Iterator Serialize (Buffer::Iterator i) const;
  /**
   * Deserialize the entire GTS fields.
   * \param i an iterator which points to where the superframe specification field should be read.
   * \return an iterator.
   */
  Buffer::Iterator Deserialize (Buffer::Iterator i);

private:
  /**
   * GTS Descriptor
   */
  struct gtsDescriptor
  {
    Mac16Address m_gtsDescDevShortAddr;      //!< GTS Descriptor Device Short Address (Bit 0-15)
    uint8_t m_gtsDescStartSlot;              //!< GTS Descriptor GTS Starting Slot(Bit 16-19)
    uint8_t m_gtsDescLength;                 //!< GTS Descriptor GTS Length (Bit 20-23)
  };

  //GTS specification field
  uint8_t m_gtsSpecDescCount;            //!< GTS specification field Descriptor Count (Bit 0-2)
                                         // GTS specification field Reserved (Not necessary) (Bit 3-6)
  uint8_t m_gtsSpecPermit;               //!< GTS specification field GTS Permit (Bit 7)
  //GTS Direction field
  uint8_t m_gtsDirMask;                  //!< GTS Direction field Directions Mask (Bit 0-6)
                                         // GTS Direction field Reserved (Not Necessary) (Bit 7)
  //GTS List
  gtsDescriptor m_gtsList[7];            //!< GTS List field (maximum descriptors stored == 7)
};


/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] gtsFields The GTS fields.
 * \returns The reference to the output stream.
 */
std::ostream &operator << (std::ostream &os, const GtsFields &gtsFields);



/**
 * \ingroup lr-wpan
 * Represent the Pending Address Specification field.
 * See IEEE 802.15.4-2011   Section 5.2.2.1.6. Figure 45
 */
class PendingAddrFields
{

public:
  PendingAddrFields ();
  /**
   * Add a short Pending Address to the Address List.
   * \param shortAddr The extended Pending Address List.
   */
  void AddAddress (Mac16Address shortAddr);
  /**
   * Add a extended Pending Address to the Address List.
   * \param extAddr The extended Pending Address List.
   */
  void AddAddress (Mac64Address extAddr);
  /**
   * Search for the short Pending Address in the Address List.
   * \param shortAddr The extended Address to look in the Address List.
   * \return True if the address exist in the extended Address List.
   */
  bool SearchAddress (Mac16Address shortAddr);
  /**
   * Search for the extended Pending Address in the Address List.
   * \param extAddr The extended Address to look in the Address List.
   * \return True if the address exist in the extended Address List.
   */
  bool SearchAddress (Mac64Address extAddr);
  /**
   * Get the whole Pending Address Specification Field from the Pending Address Fields.
   * \return The Pending Address Specification Field.
   */
  uint8_t GetPndAddrSpecField (void) const;
  /**
   * Get the number of Short Pending Address indicated in the Pending Address Specification Field.
   * \return The number Short Pending Address.
   */
  uint8_t GetNumShortAddr (void) const;
  /**
   * Get the number of Extended Pending Address indicated in the Pending Address Specification Field.
   * \return The number Short Pending Address.
   */
  uint8_t GetNumExtAddr (void) const;

  /**
    * Set the whole Pending Address Specification field. This field is part of the
    * Pending Address Fields header.
    * \param pndAddrSpecField The Pending Address Specification Field
    */
  void SetPndAddrSpecField (uint8_t pndAddrSpecField);
  /**
   * Get the size of the serialized Pending Address Fields.
   * \return the size of the serialized fields.
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * Serialize the entire Pending Address Fields.
   * \param i an iterator which points to where the Pending Address Fields should be written.
   * \return an iterator.
   */
  Buffer::Iterator Serialize (Buffer::Iterator i) const;
  /**
   * Deserialize the all the Pending Address Fields.
   * \param i an iterator which points to where the Pending Address Fields should be read.
   * \return an iterator.
   */
  Buffer::Iterator Deserialize (Buffer::Iterator i);

private:
  // Pending Address Specification Field
  uint8_t m_pndAddrSpecNumShortAddr;           //!< Pending Address Specification field Number of Short Address (Bits 0-2)
                                               //!< Pending Address Specification field Reserved (Not Necessary)(Bit  3)
  uint8_t m_pndAddrSpecNumExtAddr;             //!< Pending Address Specification field Number of Extended Address (Bits 4-6)
                                               //!< Pending Address Specification field Reserved (Not Necessary) (Bit  7)
  // Address List
  std::array <Mac16Address,7> m_shortAddrList;    //!< Pending Short Address List
  std::array<Mac64Address,7>  m_extAddrList;      //!< Pending Extended Address List

};

/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] pendingAddrFields The Pending Address fields.
 * \returns The reference to the output stream.
 */
std::ostream &operator << (std::ostream &os, const PendingAddrFields &pendingAddrFields);


}  //end namespace ns3

#endif /* LR_WPAN_FIELDS_H */
