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


#ifndef CID_H
#define CID_H

#include <stdint.h>
#include <ostream>

namespace ns3 {

/**
 * \ingroup wimax
 * \class Cid
 * \brief Cid class
 */
class Cid
{
public:
  /// Type enumeration
  enum Type
  {
    BROADCAST = 1,
    INITIAL_RANGING,
    BASIC,
    PRIMARY,
    TRANSPORT,
    MULTICAST,
    PADDING
  };

  /// Create a cid of unknown type
  Cid (void);
  /**
   * Constructor
   *
   * \param cid
   */
  Cid (uint16_t cid);
  ~Cid (void);
  /**
   * \return the identifier of the cid
   */
  uint16_t GetIdentifier (void) const;
  /**
   * \return true if the cid is a multicast cid, false otherwise
   */
  bool IsMulticast (void) const;
  /**
   * \return true if the cid is a broadcast cid, false otherwise
   */
  bool IsBroadcast (void) const;
  /**
   * \return true if the cid is a padding cid, false otherwise
   */
  bool IsPadding (void) const;
  /**
   * \return true if the cid is an initial ranging cid, false otherwise
   */
  bool IsInitialRanging (void) const;
  /**
   * \return the broadcast cid
   */
  static Cid Broadcast (void);
  /**
   * \return the padding cid
   */
  static Cid Padding (void);
  /**
   * \return the initial ranging cid
   */
  static Cid InitialRanging (void);

private:
  /// allow CidFactory class friend access
  friend class CidFactory;
  /// equality operator
  friend bool operator == (const Cid &lhs, const Cid &rhs);
  uint16_t m_identifier; ///< identiifier
};

bool operator == (const Cid &lhs, const Cid &rhs);
bool operator != (const Cid &lhs, const Cid &rhs);

std::ostream & operator << (std::ostream &os, const Cid &cid);

} // namespace ns3

#endif /* CID_H */
