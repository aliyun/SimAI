/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Dalian University of Technology
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#include <iomanip>
#include <iostream>
#include <cstring>
#include "ns3/log.h"
#include "ns3/assert.h"
#include "vendor-specific-action.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("VendorSpecificAction");

/*********** OrganizationIdentifier *******/

ATTRIBUTE_HELPER_CPP (OrganizationIdentifier);

OrganizationIdentifier::OrganizationIdentifier (void)
  : m_type (Unknown)
{
  NS_LOG_FUNCTION (this);
  m_type = Unknown;
  std::memset (m_oi, 0, 5);
}

OrganizationIdentifier::OrganizationIdentifier (const uint8_t *str, uint32_t length)
{
  NS_LOG_FUNCTION (this << str << length);
  if (length == 3)
    {
      m_type = OUI24;
      std::memcpy (m_oi, str, length);
    }
  else if (length == 5)
    {
      m_type = OUI36;
      std::memcpy (m_oi, str, length);
    }
  else
    {
      m_type = Unknown;
      NS_FATAL_ERROR ("cannot support organization identifier with length=" << length);
    }
}

OrganizationIdentifier&
OrganizationIdentifier::operator= (const OrganizationIdentifier& oi)
{
  this->m_type = oi.m_type;
  std::memcpy (this->m_oi, oi.m_oi, 5);
  return (*this);
}

OrganizationIdentifier::~OrganizationIdentifier (void)
{
  NS_LOG_FUNCTION (this);
}

uint8_t
OrganizationIdentifier::GetManagementId (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_type == OUI36);
  return (m_oi[4] & 0x0f);
}

bool
OrganizationIdentifier::IsNull (void) const
{
  NS_LOG_FUNCTION (this);
  return m_type == Unknown;
}

uint32_t
OrganizationIdentifier::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  switch (m_type)
    {
    case OUI24:
      return 3;
    case OUI36:
      return 5;
    case Unknown:
    default:
      NS_FATAL_ERROR_NO_MSG ();
      return 0;
    }
}

void
OrganizationIdentifier::SetType (enum OrganizationIdentifierType type)
{
  NS_LOG_FUNCTION (this);
  m_type = type;
}

enum OrganizationIdentifier::OrganizationIdentifierType
OrganizationIdentifier::GetType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_type;
}

void
OrganizationIdentifier::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.Write (m_oi, GetSerializedSize ());
}

/*  because OrganizationIdentifier field is not standard
 *  and the length of OrganizationIdentifier is variable
 *  so data parse here is troublesome
 */
uint32_t
OrganizationIdentifier::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  // first try to parse OUI24 with 3 bytes
  start.Read (m_oi,  3);
  for (std::vector<OrganizationIdentifier>::iterator  i = OrganizationIdentifiers.begin (); i != OrganizationIdentifiers.end (); ++i)
    {
      if ((i->m_type == OUI24)
          && (std::memcmp (i->m_oi, m_oi, 3) == 0 ))
        {
          m_type = OUI24;
          return 3;
        }
    }

  // then try to parse OUI36 with 5 bytes
  start.Read (m_oi + 3,  2);
  for (std::vector<OrganizationIdentifier>::iterator  i = OrganizationIdentifiers.begin (); i != OrganizationIdentifiers.end (); ++i)
    {
      if ((i->m_type == OUI36)
          && (std::memcmp (i->m_oi, m_oi, 4) == 0 ))
        {
          // OUI36 first check 4 bytes, then check half of the 5th byte
          if ((i->m_oi[4] & 0xf0) == (m_oi[4] & 0xf0))
            {
              m_type = OUI36;
              return 5;
            }
        }
    }

  // if we cannot deserialize the organization identifier field,
  // we will fail
  NS_FATAL_ERROR ("cannot deserialize the organization identifier field successfully");
  return 0;
}

/**
 * equality operator
 * \param a left side object
 * \param b right side object
 * \returns true if equal
 */
bool operator == (const OrganizationIdentifier& a, const OrganizationIdentifier& b)
{
  if (a.m_type != b.m_type)
    {
      return false;
    }

  if (a.m_type == OrganizationIdentifier::OUI24)
    {
      return memcmp (a.m_oi, b.m_oi, 3) == 0;
    }

  if (a.m_type == OrganizationIdentifier::OUI36)
    {
      return (memcmp (a.m_oi, b.m_oi, 4) == 0)
             && ((a.m_oi[4] & 0xf0) == (b.m_oi[4] & 0xf0));
    }

  return false;
}

/**
 * inequality operator
 * \param a left side object
 * \param b right side object
 * \returns true if not equal
 */
bool operator != (const OrganizationIdentifier& a, const OrganizationIdentifier& b)
{
  return !(a == b);
}

/**
 * less than operator
 * \param a left side object
 * \param b right side object
 * \returns true if a < b
 */
bool operator < (const OrganizationIdentifier& a, const OrganizationIdentifier& b)
{
  return memcmp (a.m_oi, b.m_oi, std::min (a.m_type, b.m_type)) < 0;
}

/**
 * output operator
 * \param os output stream
 * \param oi organization identifier
 * \returns output stream
 */
std::ostream& operator << (std::ostream& os, const OrganizationIdentifier& oi)
{
  for (int i = 0; i < oi.m_type; i++)
    {
      os << "0x" << std::hex << static_cast<int> (oi.m_oi[i]) << " ";
    }
  os << std::endl;
  return os;
}

/**
 * input operator
 * \param is input stream
 * \param oi organization identifier
 * \returns input stream
 */
std::istream& operator >> (std::istream& is, const OrganizationIdentifier& oi)
{
  return is;
}

/*********** VendorSpecificActionHeader *******/
NS_OBJECT_ENSURE_REGISTERED (VendorSpecificActionHeader);

VendorSpecificActionHeader::VendorSpecificActionHeader (void)
  : m_oi (),
    m_category (CATEGORY_OF_VSA)
{
  NS_LOG_FUNCTION (this);
}

VendorSpecificActionHeader::~VendorSpecificActionHeader (void)
{
  NS_LOG_FUNCTION (this);
}

void
VendorSpecificActionHeader::SetOrganizationIdentifier (OrganizationIdentifier oi)
{
  NS_LOG_FUNCTION (this << oi);
  m_oi = oi;
}

OrganizationIdentifier
VendorSpecificActionHeader::GetOrganizationIdentifier (void) const
{
  NS_LOG_FUNCTION (this);
  return m_oi;
}

TypeId
VendorSpecificActionHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::VendorSpecificActionHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wave")
    .AddConstructor<VendorSpecificActionHeader> ()
  ;

  return tid;
}

uint8_t
VendorSpecificActionHeader::GetCategory (void) const
{
  NS_LOG_FUNCTION (this);
  return m_category;
}

TypeId
VendorSpecificActionHeader::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}

void
VendorSpecificActionHeader::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "VendorSpecificActionHeader[ "
     << "category = 0x" << std::hex << (int)m_category
     << "organization identifier = " << m_oi
     << std::dec;
}

uint32_t
VendorSpecificActionHeader::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return sizeof(m_category) + m_oi.GetSerializedSize ();
}

void
VendorSpecificActionHeader::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.WriteU8 (m_category);
  m_oi.Serialize (start);
}

uint32_t
VendorSpecificActionHeader::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  m_category = start.ReadU8 ();
  if (m_category != CATEGORY_OF_VSA)
    {
      return 0;
    }
  m_oi.Deserialize (start);

  return GetSerializedSize ();
}

/********* VendorSpecificContentManager ***********/
VendorSpecificContentManager::VendorSpecificContentManager (void)
{
  NS_LOG_FUNCTION (this);
}

VendorSpecificContentManager::~VendorSpecificContentManager (void)
{
  NS_LOG_FUNCTION (this);
}

void
VendorSpecificContentManager::RegisterVscCallback (OrganizationIdentifier oi, VscCallback cb)
{
  NS_LOG_FUNCTION (this << oi << &cb);
  if (IsVscCallbackRegistered (oi))
    {
      NS_LOG_WARN ("there is already a VsaCallback registered for OrganizationIdentifier " << oi);
    }
  m_callbacks.insert (std::make_pair (oi, cb));
}

void
VendorSpecificContentManager::DeregisterVscCallback (OrganizationIdentifier &oi)
{
  NS_LOG_FUNCTION (this << oi);
  m_callbacks.erase (oi);
}

bool
VendorSpecificContentManager::IsVscCallbackRegistered (OrganizationIdentifier &oi)
{
  NS_LOG_FUNCTION (this << oi);
  if (m_callbacks.find (oi) == m_callbacks.end ())
    {
      OrganizationIdentifiers.push_back (oi);
      return false;
    }
  return true;
}

///VSC callback function
static VscCallback null_callback = MakeNullCallback<bool, Ptr<WifiMac>, const OrganizationIdentifier &,Ptr<const Packet>,const Address &> ();

VscCallback
VendorSpecificContentManager::FindVscCallback (OrganizationIdentifier &oi)
{
  NS_LOG_FUNCTION (this << oi);
  VscCallbacksI i;
  i = m_callbacks.find (oi);
  return (i == m_callbacks.end ()) ? null_callback : i->second;
}

} // namespace ns3
