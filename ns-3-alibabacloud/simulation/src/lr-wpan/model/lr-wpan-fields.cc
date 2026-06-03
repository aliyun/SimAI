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
 *  Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */



#include "lr-wpan-fields.h"
#include <ns3/log.h>
#include <ns3/address-utils.h>

namespace ns3 {

SuperframeField::SuperframeField ()
{
  SetBeaconOrder (15);
  SetSuperframeOrder (15);
  SetFinalCapSlot (0);
  SetBattLifeExt (false);
  SetPanCoor (false);
  SetAssocPermit (false);
}

void
SuperframeField::SetSuperframe (uint16_t superFrmSpec)
{
  m_sspecBcnOrder = (superFrmSpec) & (0x0F);          //Bits 0-3
  m_sspecSprFrmOrder = (superFrmSpec >> 4) & (0x0F);  //Bits 4-7
  m_sspecFnlCapSlot = (superFrmSpec >> 8) & (0x0F);   //Bits 8-11
  m_sspecBatLifeExt = (superFrmSpec >> 12) & (0x01);  //Bit 12
                                                      //Bit 13 (Reserved)
  m_sspecPanCoor = (superFrmSpec >> 14) & (0x01);     //Bit 14
  m_sspecAssocPermit = (superFrmSpec >> 15) & (0x01); //Bit 15
}

void
SuperframeField::SetBeaconOrder (uint8_t bcnOrder)
{
  if (bcnOrder > 15)
    {
      std::cout << "SuperframeField Beacon Order value must be 15 or less\n";
    }
  else
    {
      m_sspecBcnOrder = bcnOrder;
    }
}

void
SuperframeField::SetSuperframeOrder (uint8_t frmOrder)
{
  if (frmOrder > 15)
    {
      std::cout << "SuperframeField Frame Order value must be 15 or less\n";
    }
  else
    {
      m_sspecSprFrmOrder = frmOrder;
    }
}

void
SuperframeField::SetFinalCapSlot (uint8_t capSlot)
{
  if (capSlot > 15)
    {
      std::cout << "The final slot cannot greater than the slots in a CAP (15)\n";
    }
  else
    {
      m_sspecFnlCapSlot = capSlot;
    }
}

void
SuperframeField::SetBattLifeExt (bool battLifeExt)
{
  m_sspecBatLifeExt = battLifeExt;
}

void
SuperframeField::SetPanCoor (bool panCoor)
{
  m_sspecPanCoor = panCoor;
}

void
SuperframeField::SetAssocPermit (bool assocPermit)
{
  m_sspecAssocPermit = assocPermit;
}

uint8_t
SuperframeField::GetBeaconOrder (void) const
{
  return m_sspecBcnOrder;
}

uint8_t
SuperframeField::GetFrameOrder (void) const
{
  return m_sspecSprFrmOrder;
}

uint8_t
SuperframeField::GetFinalCapSlot (void) const
{
  return m_sspecFnlCapSlot;
}

bool
SuperframeField::IsBattLifeExt (void) const
{
  return m_sspecBatLifeExt;
}

bool
SuperframeField::IsPanCoor (void) const
{
  return m_sspecPanCoor;
}

bool
SuperframeField::IsAssocPermit (void) const
{
  return m_sspecAssocPermit;
}

uint16_t
SuperframeField::GetSuperframe (void) const
{
  uint16_t superframe;

  superframe = m_sspecBcnOrder & (0x0F);                       // Bits 0-3
  superframe     |= (m_sspecSprFrmOrder << 4) & (0x0F << 4);   // Bits 4-7
  superframe     |= (m_sspecFnlCapSlot << 8) & (0x0F << 8);    // Bits 8-11
  superframe     |= (m_sspecBatLifeExt << 12) & (0x01 << 12);  // Bit 12
                                                               // Bit 13 (Reserved)
  superframe     |= (m_sspecPanCoor << 14) & (0x01 << 14);     // Bit 14
  superframe     |= (m_sspecAssocPermit << 15) & (0x01 << 15); // Bit 15

  return superframe;
}

uint32_t
SuperframeField::GetSerializedSize (void) const
{
  return 2;  // 2 Octets (superframeSpec)
}

Buffer::Iterator
SuperframeField::Serialize (Buffer::Iterator i) const
{
  i.WriteHtolsbU16 (GetSuperframe ());
  return i;
}

Buffer::Iterator
SuperframeField::Deserialize (Buffer::Iterator i)
{
  uint16_t superframe = i.ReadLsbtohU16 ();
  SetSuperframe (superframe);

  return i;
}

std::ostream &
operator << (std::ostream &os, const SuperframeField &superframeField)
{
  os << " Beacon Order = "      << uint32_t  (superframeField.GetBeaconOrder ())
     << ", Frame Order = "       << uint32_t  (superframeField.GetFrameOrder ())
     << ", Final CAP slot = "    << uint32_t (superframeField.GetFinalCapSlot ())
     << ", Battery Life Ext = "  << bool (superframeField.IsBattLifeExt ())
     << ", PAN Coordinator = "   << bool (superframeField.IsPanCoor ())
     << ", Association Permit = " << bool (superframeField.IsAssocPermit ());
  return os;
}

/***********************************************************
 *         Guaranteed Time Slots (GTS) Fields
 ***********************************************************/

GtsFields::GtsFields ()
{
  // GTS Specification Field
  m_gtsSpecDescCount = 0;
  m_gtsSpecPermit = 0;
  // GTS Direction Field
  m_gtsDirMask = 0;
}

uint8_t
GtsFields::GetGtsSpecField (void) const
{
  uint8_t gtsSpecField;

  gtsSpecField  = m_gtsSpecDescCount & (0x07);            // Bits 0-2
                                                          // Bits 3-6 (Reserved)
  gtsSpecField |= (m_gtsSpecPermit << 7) & (0x01 << 7);   // Bit 7

  return gtsSpecField;
}

uint8_t
GtsFields::GetGtsDirectionField (void) const
{
  uint8_t gtsDirectionField;

  gtsDirectionField = m_gtsDirMask & (0x7F);              // Bit 0-6
                                                          // Bit 7 (Reserved)
  return gtsDirectionField;
}

void
GtsFields::SetGtsSpecField (uint8_t gtsSpec)
{
  m_gtsSpecDescCount = (gtsSpec) & (0x07);      // Bits 0-2
                                                // Bits 3-6 (Reserved)
  m_gtsSpecPermit = (gtsSpec >> 7) & (0x01);    // Bit 7
}

void
GtsFields::SetGtsDirectionField (uint8_t gtsDir)
{
  m_gtsDirMask = (gtsDir) & (0x7F);           // Bits 0-6
                                              // Bit 7 (Reserved)
}

uint32_t
GtsFields::GetSerializedSize (void) const
{
  uint32_t size;

  size = 1;                             // 1 octet  GTS Specification Field
  if (m_gtsSpecDescCount > 0)
    {
      size += 1;                            // 1 octet GTS Direction Field
      size += (m_gtsSpecDescCount * 3);     // 3 octets per GTS descriptor
    }

  return size;
}

Buffer::Iterator
GtsFields::Serialize (Buffer::Iterator i) const
{
  i.WriteU8 (GetGtsSpecField ());

  if (m_gtsSpecDescCount > 0)
    {
      uint8_t gtsDescStartAndLenght;
      i.WriteU8 (GetGtsDirectionField ());

      for (int j = 0; j < m_gtsSpecDescCount; j++)
        {
          WriteTo (i,m_gtsList[j].m_gtsDescDevShortAddr);

          gtsDescStartAndLenght = m_gtsList[j].m_gtsDescStartSlot & (0x0F);
          gtsDescStartAndLenght = (m_gtsList[j].m_gtsDescLength << 4) & (0x0F);

          i.WriteU8 (gtsDescStartAndLenght);
        }
    }
  return i;
}

Buffer::Iterator
GtsFields::Deserialize (Buffer::Iterator i)
{

  uint8_t gtsSpecField = i.ReadU8 ();
  SetGtsSpecField (gtsSpecField);

  if (m_gtsSpecDescCount > 0)
    {
      uint8_t gtsDirectionField = i.ReadU8 ();
      SetGtsDirectionField (gtsDirectionField);

      uint8_t gtsDescStartAndLenght;
      for (int j = 0; j < m_gtsSpecDescCount; j++)
        {
          ReadFrom (i, m_gtsList[j].m_gtsDescDevShortAddr);

          gtsDescStartAndLenght = i.ReadU8 ();
          m_gtsList[j].m_gtsDescStartSlot = (gtsDescStartAndLenght) & (0x0F);
          m_gtsList[j].m_gtsDescLength  = (gtsDescStartAndLenght >> 4) & (0x0F);
        }
    }
  return i;
}

std::ostream &
operator << (std::ostream &os, const GtsFields &gtsFields)
{
  os << " GTS specification = "  << uint32_t (gtsFields.GetGtsSpecField ())
     << ", GTS direction = "     << uint32_t (gtsFields.GetGtsDirectionField ());
  return os;
}

/***********************************************************
 *              Pending Address Fields
 ***********************************************************/

PendingAddrFields::PendingAddrFields ()
{
  m_pndAddrSpecNumShortAddr = 0;
  m_pndAddrSpecNumExtAddr = 0;
}


uint8_t
PendingAddrFields::GetNumShortAddr (void) const
{
  return m_pndAddrSpecNumShortAddr;
}


uint8_t
PendingAddrFields::GetNumExtAddr (void) const
{
  return m_pndAddrSpecNumExtAddr;
}

uint8_t
PendingAddrFields::GetPndAddrSpecField (void) const
{
  uint8_t pndAddrSpecField;

  pndAddrSpecField    = m_pndAddrSpecNumShortAddr & (0x07);            // Bits 0-2
                                                                       // Bit  3 (Reserved)
  pndAddrSpecField   |= (m_pndAddrSpecNumExtAddr << 4) & (0x07 << 4);  // Bits 4-6
                                                                       // Bit  7 (Reserved)

  return pndAddrSpecField;
}

void
PendingAddrFields::AddAddress (Mac16Address shortAddr)
{
  uint8_t totalPendAddr = m_pndAddrSpecNumShortAddr + m_pndAddrSpecNumExtAddr;

  if (totalPendAddr == 7)
    {
      return;
    }
  else
    {
      m_shortAddrList[m_pndAddrSpecNumShortAddr] = shortAddr;
      m_pndAddrSpecNumShortAddr++;
    }
}

void
PendingAddrFields::AddAddress (Mac64Address extAddr)
{
  uint8_t totalPendAddr = m_pndAddrSpecNumShortAddr + m_pndAddrSpecNumExtAddr;

  if (totalPendAddr == 7)
    {
      return;
    }
  else
    {
      m_extAddrList[m_pndAddrSpecNumExtAddr] = extAddr;
      m_pndAddrSpecNumExtAddr++;
    }
}

bool
PendingAddrFields::SearchAddress (Mac16Address shortAddr)
{
  for (int j = 0; j <= m_pndAddrSpecNumShortAddr; j++)
    {
      if (shortAddr == m_shortAddrList[j])
        {
          return true;
        }
    }

  return false;
}


bool
PendingAddrFields::SearchAddress (Mac64Address extAddr)
{
  for (int j = 0; j <= m_pndAddrSpecNumExtAddr; j++)
    {
      if (extAddr == m_extAddrList[j])
        {
          return true;
        }
    }

  return false;
}


void
PendingAddrFields::SetPndAddrSpecField (uint8_t pndAddrSpecField)
{
  m_pndAddrSpecNumShortAddr = (pndAddrSpecField) & (0x07);      // Bit 0-2
                                                                // Bit 3
  m_pndAddrSpecNumExtAddr   = (pndAddrSpecField >> 4) & (0x07); // Bit 4-6
                                                                // Bit 7
}


uint32_t
PendingAddrFields::GetSerializedSize (void) const
{
  uint32_t size;

  size = 1;                                      // 1 octet  (Pending Address Specification Field)
  size = size + (m_pndAddrSpecNumShortAddr * 2); // X octets (Short Pending Address List)
  size = size + (m_pndAddrSpecNumExtAddr * 8);   // X octets (Extended Pending Address List)

  return size;
}

Buffer::Iterator
PendingAddrFields::Serialize (Buffer::Iterator i) const
{
  i.WriteU8 (GetPndAddrSpecField ());

  for (int j = 0; j < m_pndAddrSpecNumShortAddr; j++)
    {
      WriteTo (i,m_shortAddrList[j]);
    }

  for (int k = 0; k < m_pndAddrSpecNumExtAddr; k++ )
    {
      WriteTo (i,m_extAddrList[k]);
    }

  return i;
}

Buffer::Iterator
PendingAddrFields::Deserialize (Buffer::Iterator i)
{
  uint8_t pndAddrSpecField = i.ReadU8 ();

  SetPndAddrSpecField (pndAddrSpecField);

  for (int j = 0; j < m_pndAddrSpecNumShortAddr; j++)
    {
      ReadFrom (i, m_shortAddrList[j]);
    }

  for (int k = 0; k < m_pndAddrSpecNumExtAddr; k++)
    {
      ReadFrom (i, m_extAddrList[k]);
    }

  return i;
}

std::ostream &
operator << (std::ostream &os, const PendingAddrFields &pendingAddrFields)
{
  os << " Num. Short Addr = "  << uint32_t  (pendingAddrFields.GetNumShortAddr ())
     << ", Num. Ext   Addr = "  << uint32_t  (pendingAddrFields.GetNumExtAddr ());
  return os;
}

} // ns-3 namespace
