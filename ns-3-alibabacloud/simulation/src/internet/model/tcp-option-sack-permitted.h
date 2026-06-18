/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Adrian Sai-wah Tam
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
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
 * Original Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 * Documentation, test cases: Truc Anh N. Nguyen   <annguyen@ittc.ku.edu>
 *                            ResiliNets Research Group   http://wiki.ittc.ku.edu/resilinets
 *                            The University of Kansas
 *                            James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 */

#ifndef TCP_OPTION_SACK_PERMITTED_H
#define TCP_OPTION_SACK_PERMITTED_H

#include "ns3/tcp-option.h"

namespace ns3 {

/**
 * \brief Defines the TCP option of kind 4 (selective acknowledgment permitted
 * option) as in \RFC{2018}
 *
 * TCP Sack-Permitted Option is 2-byte in length and sent in a SYN segment by a
 * TCP host that can recognize and process SACK option during the lifetime of a
 * connection.
 */

class TcpOptionSackPermitted : public TcpOption
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  TcpOptionSackPermitted ();
  virtual ~TcpOptionSackPermitted ();

  virtual void Print (std::ostream &os) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  virtual uint8_t GetKind (void) const;
  virtual uint32_t GetSerializedSize (void) const;
};

} // namespace ns3

#endif /* TCP_OPTION_SACK_PERMITTED */
