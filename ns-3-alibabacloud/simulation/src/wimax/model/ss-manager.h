/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#ifndef SS_MANAGER_H
#define SS_MANAGER_H

#include <stdint.h>
#include "cid.h"
#include "ss-record.h"

namespace ns3 {

/**
 * \ingroup wimax
 * \brief this class manages a list of SSrecords
 * \see SSrecord
 */
class SSManager : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  SSManager (void);
  ~SSManager (void);
  /**
   * Create SS record
   * \param macAddress the MAC address
   * \returns pointer to the SS record
   */
  SSRecord* CreateSSRecord (const Mac48Address &macAddress);
  /**
   * Get SS record
   * \param macAddress the MAC address
   * \returns pointer to the SS record
   */
  SSRecord* GetSSRecord (const Mac48Address &macAddress) const;
  /**
   * \brief returns the ssrecord which has been assigned this cid. Since
   *        different types of cids (basic, primary, transport) are assigned
   *        different values, all cids (basic, primary and transport) of the
   *        ssrecord are matched.
   * \param cid the cid to be matched
   * \return pointer to the ss record matching the cid
   */
  SSRecord* GetSSRecord (Cid cid) const;
  /**
   * Get SS records
   * \returns a vector of pointers to the SS records
   */
  std::vector<SSRecord*>* GetSSRecords (void) const;
  /**
   * Check if address is in record
   * \param macAddress the MAC address
   * \returns whether the address is in the record
   */
  bool IsInRecord (const Mac48Address &macAddress) const;
  /**
   * Check if address is registered
   * \param macAddress the MAC address
   * \returns whether the address is registered
   */
  bool IsRegistered (const Mac48Address &macAddress) const;
  /**
   * Delete SS record
   * \param cid the CID
   */
  void DeleteSSRecord (Cid cid);
  /**
   * Get MAC address by CID
   * \param cid the CID
   * \returns the MAC address
   */
  Mac48Address GetMacAddress (Cid cid) const;
  /**
   * Get number of SSs
   * \returns the number of SSs
   */
  uint32_t GetNSSs (void) const;
  /**
   * Get number of registered SSs 
   * \returns the number of registered SSs
   */
  uint32_t GetNRegisteredSSs (void) const;
private:
  std::vector<SSRecord*> *m_ssRecords; ///< the SS records
};

} // namespace ns3

#endif /* SS_MANAGER_H */
