/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 UPB
 * Copyright (c) 2017 NITK Surathkal
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
 * Author: Radu Lupu <rlupu@elcom.pub.ro>
 *         Ankit Deepak <adadeepak8@gmail.com>
 *         Deepti Rajagopal <deeptir96@gmail.com>
 *
 */

#ifndef DHCP_HEADER_H
#define DHCP_HEADER_H

#include "ns3/header.h"

namespace ns3 {

/**
 * \ingroup internet-apps
 * \defgroup dhcp DHCPv4 Client and Server
 */

/**
 * \ingroup dhcp
 *
 * \class DhcpHeader
 * \brief BOOTP header with DHCP messages supports the following options:
 *        Subnet Mask (1), Address Request (50), Refresh Lease Time (51),
 *        DHCP Message Type (53), DHCP Server ID (54), Renew Time (58),
 *        Rebind Time (59) and End (255) of BOOTP

  \verbatim
    0                   1                   2                   3
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |     op (1)    |   htype (1)   |   hlen (1)    |   hops (1)    |
   +---------------+---------------+---------------+---------------+
   |                            xid (4)                            |
   +-------------------------------+-------------------------------+
   |           secs (2)            |           flags (2)           |
   +-------------------------------+-------------------------------+
   |                          ciaddr  (4)                          |
   +---------------------------------------------------------------+
   |                          yiaddr  (4)                          |
   +---------------------------------------------------------------+
   |                          siaddr  (4)                          |
   +---------------------------------------------------------------+
   |                          giaddr  (4)                          |
   +---------------------------------------------------------------+
   |                                                               |
   |                          chaddr  (16)                         |
   |                                                               |
   |                                                               |
   +---------------------------------------------------------------+
   |                                                               |
   |                          sname   (64)                         |
   +---------------------------------------------------------------+
   |                                                               |
   |                          file    (128)                        |
   +---------------------------------------------------------------+
   |                                                               |
   |                          options (variable)                   |
   +---------------------------------------------------------------+
  \endverbatim

 */
class DhcpHeader : public Header
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
  DhcpHeader ();

  /**
   * \brief Destructor
   */
  ~DhcpHeader ();

  /// BOOTP options
  enum Options
  {
    OP_MASK = 1,        //!< BOOTP Option 1: Address Mask
    OP_ROUTE = 3,       //!< BOOTP Option 3: Router Option
    OP_ADDREQ = 50,     //!< BOOTP Option 50: Requested Address
    OP_LEASE = 51,      //!< BOOTP Option 51: Address Lease Time
    OP_MSGTYPE = 53,    //!< BOOTP Option 53: DHCP Message Type
    OP_SERVID = 54,     //!< BOOTP Option 54: Server Identifier
    OP_RENEW = 58,      //!< BOOTP Option 58: Address Renewal Time
    OP_REBIND = 59,     //!< BOOTP Option 59: Address Rebind Time
    OP_END = 255        //!< BOOTP Option 255: END
  };

  /// DHCP messages
  enum Messages
  {
    DHCPDISCOVER = 0,     //!< Code for DHCP Discover
    DHCPOFFER = 1,        //!< Code for DHCP Offer
    DHCPREQ = 2,          //!< Code for DHCP Request
    DHCPACK = 4,          //!< Code for DHCP ACK
    DHCPNACK = 5          //!< Code for DHCP NACK
  };

  /**
   * \brief Set the type of BOOTP and DHCP messages
   * \param type The type of message
   */
  void SetType (uint8_t type);

  /**
   * \brief Return the type of DHCP message
   * \return The type of message
   */
  uint8_t GetType (void) const;

  /**
   * \brief Set the hardware information
   * \param htype Hardware type
   * \param hlen Hardware length
   */
  void SetHWType (uint8_t htype, uint8_t hlen);

  /**
   * \brief Set the transaction ID
   * \param tran The transaction number
   */
  void SetTran (uint32_t tran);

  /**
   * \brief Get the transaction id
   * \return The transaction id
   */
  uint32_t GetTran (void) const;

  /**
   * \brief Set the time when message is sent
   */
  void SetTime ();

  /**
   * \brief Set the Address of the device.
   *
   * Only the relevant bits are considered (i.e., not the type and length)
   *
   * \param addr Address of the device
   */
  void SetChaddr (Address addr);

  /**
   * \brief Set the Address of the device
   * \param addr Address of the device
   * \param len Address length
   */
  void SetChaddr (uint8_t* addr, uint8_t len);

  /**
   * \brief Get the Address of the client.
   *
   * Note: the address is always 16-bytes long.
   *
   * \return Address of the client
   */
  Address GetChaddr (void);

  /**
   * \brief Set the IPv4Address of the client
   * \param addr The client Ipv4Address
   */
  void SetYiaddr (Ipv4Address addr);

  /**
   * \brief Get the IPv4Address of the client
   * \return IPv4Address of the client
   */
  Ipv4Address GetYiaddr (void) const;

  /**
   * \brief Set the DHCP server information
   * \param addr IPv4Address of the server
   */
  void SetDhcps (Ipv4Address addr);

  /**
   * \brief Get the information about the DHCP server
   * \return IPv4Address of DHCP server
   */
  Ipv4Address GetDhcps (void) const;

  /**
   * \brief Set the Ipv4Address requested by the client
   * \param addr Ipv4Address requested by the client
   */
  void SetReq (Ipv4Address addr);

  /**
   * \brief Get the IPv4Address requested by the client
   * \return IPv4Address requested by the client
   */
  Ipv4Address GetReq (void) const;

  /**
   * \brief Set the mask of the IPv4Address
   * \param addr 32 bit mask
   */
  void SetMask (uint32_t addr);

  /**
   * \brief Return the mask of the network
   * \return 32 bit mask
   */
  uint32_t GetMask (void) const;

  /**
   * \brief Set the Ipv4Address of gateway to be used
   * \param addr The Ipv4Address of the gateway
   */
  void SetRouter (Ipv4Address addr);

  /**
   * \brief Return the Ipv4Address of gateway to be used
   * \return The Ipv4Address of the gateway
   */
  Ipv4Address GetRouter (void) const;

  /**
   * \brief Set the lease time of the IPv4Address
   * \param time 32 bit time
   */
  void SetLease (uint32_t time);

  /**
   * \brief Return the lease time of the IPv4Address
   * \return 32 bit time
   */
  uint32_t GetLease (void) const;

  /**
   * \brief Set the Renewal time of the IPv4Address
   * \param time 32 bit time
   */
  void SetRenew (uint32_t time);

  /**
   * \brief Return the Renewal time of the address
   * \return 32 bit time
   */
  uint32_t GetRenew (void) const;

  /**
   * \brief Set the Rebind time of the IPv4Address
   * \param time 32 bit time
   */
  void SetRebind (uint32_t time);

  /**
   * \brief Return the Rebind time of the address
   * \return 32 bit time
   */
  uint32_t GetRebind (void) const;

  /**
   * \brief Reset the BOOTP options
   */
  void ResetOpt ();

private:
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  uint8_t m_op;                          //!< The DHCP Message type
  uint8_t m_bootp;                       //!< The BOOTP Message type
  uint8_t m_hType;                       //!< The hardware type
  uint8_t m_hLen;                        //!< The hardware length
  uint8_t m_hops;                        //!< The number of hops covered by the message
  uint32_t m_xid;                        //!< The transaction number
  uint32_t m_mask;                       //!< The mask of the network
  uint32_t m_len;                        //!< The length of the header
  uint16_t m_secs;                       //!< Seconds elapsed
  uint16_t m_flags;                      //!< BOOTP flags
  uint8_t m_chaddr[16];                  //!< The address identifier
  Ipv4Address m_yiAddr;                  //!< Your (client) IP address
  Ipv4Address m_ciAddr;                  //!< The IP address of the client
  Ipv4Address m_siAddr;                  //!< Next Server IP address
  Ipv4Address m_giAddr;                  //!< Relay Agent IP address
  Ipv4Address m_dhcps;                   //!< DHCP server IP address
  Ipv4Address m_req;                     //!< Requested Address
  Ipv4Address m_route;                   //!< Router Option Address
  uint8_t m_sname[64];                   //!< Server name (Padded for now)
  uint8_t m_file[128];                   //!< File name (Padded for now)
  uint8_t m_magic_cookie[4];             //!< DHCP Magic Cookie
  uint32_t m_lease;                      //!< The lease time of the address
  uint32_t m_renew;                      //!< The renewal time for the client
  uint32_t m_rebind;                     //!< The rebinding time for the client
  bool m_opt[255];                       //!< BOOTP option list
};

} // namespace ns3

#endif /* DHCP_HEADER_H */
