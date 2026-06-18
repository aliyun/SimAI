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

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/address-utils.h"
#include "dhcp-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DhcpHeader");
NS_OBJECT_ENSURE_REGISTERED (DhcpHeader);

DhcpHeader::DhcpHeader ()
{
  m_hType = 1;
  m_hLen = 6;
  m_xid = 0;
  m_secs = 0;
  m_hops = 0;
  m_flags = 0;
  Ipv4Address addr = Ipv4Address ("0.0.0.0");
  m_yiAddr = addr;
  m_ciAddr = addr;
  m_siAddr = addr;
  m_giAddr = addr;
  m_dhcps = addr;
  m_req = addr;
  m_route = addr;
  m_len = 240;

  uint32_t i;

  for (i = 0; i < 64; i++)
    {
      m_sname[i] = 0;
    }
  for (i = 0; i < 128; i++)
    {
      m_file[i] = 0;
    }
  m_magic_cookie[0] = 99;
  m_magic_cookie[1] = 130;
  m_magic_cookie[2] = 83;
  m_magic_cookie[3] = 99;
}

DhcpHeader::~DhcpHeader ()
{
}

void DhcpHeader::SetType (uint8_t type)
{
  if (m_opt[OP_MSGTYPE] == false)
    {
      m_len += 3;
      m_opt[OP_MSGTYPE] = true;
    }
  m_op = type;
  m_bootp = (m_op == 0||m_op == 2) ? 1 : 2;
}

uint8_t DhcpHeader::GetType (void) const
{
  return m_op;
}

void DhcpHeader::SetHWType (uint8_t htype, uint8_t hlen)
{
  m_hType = htype;
  m_hLen = hlen;
}

void DhcpHeader::SetTran (uint32_t tran)
{
  m_xid = tran;
}

uint32_t DhcpHeader::GetTran (void) const
{
  return m_xid;
}

void DhcpHeader::SetTime ()
{
  m_secs = (uint16_t) Simulator::Now ().GetSeconds ();
}

void DhcpHeader::SetChaddr (Address addr)
{
  std::memset (m_chaddr, 0, 16);
  NS_ASSERT_MSG (addr.GetLength () <= 16, "Address length too big");
  addr.CopyTo (m_chaddr);
}

void DhcpHeader::SetChaddr (uint8_t* addr, uint8_t len)
{
  std::memset (m_chaddr, 0, 16);
  NS_ASSERT_MSG (len <= 16, "Address length too big");
  std::memcpy (m_chaddr, addr, len);
}

Address DhcpHeader::GetChaddr ()
{
  Address addr;
  addr.CopyFrom (m_chaddr, 16);
  return addr;
}

void DhcpHeader::SetYiaddr (Ipv4Address addr)
{
  m_yiAddr = addr;
}

Ipv4Address DhcpHeader::GetYiaddr (void) const
{
  return m_yiAddr;
}

void DhcpHeader::SetDhcps (Ipv4Address addr)
{
  if (m_opt[OP_SERVID] == false)
    {
      m_len += 6;
      m_opt[OP_SERVID] = true;
    }
  m_dhcps = addr;
}

Ipv4Address DhcpHeader::GetDhcps (void) const
{
  return m_dhcps;
}

void DhcpHeader::SetReq (Ipv4Address addr)
{
  if (m_opt[OP_ADDREQ] == false)
    {
      m_len += 6;
      m_opt[OP_ADDREQ] = true;
    }
  m_req = addr;
}

Ipv4Address DhcpHeader::GetReq (void) const
{
  return m_req;
}

void DhcpHeader::SetMask (uint32_t addr)
{
  if (m_opt[OP_MASK] == false)
    {
      m_len += 6;
      m_opt[OP_MASK] = true;
    }
  m_mask = addr;
}

uint32_t DhcpHeader::GetMask (void) const
{
  return m_mask;
}

void DhcpHeader::SetRouter (Ipv4Address addr)
{
  if (m_opt[OP_ROUTE] == false)
    {
      m_len += 6;
      m_opt[OP_ROUTE] = true;
    }
  m_route = addr;
}

Ipv4Address DhcpHeader::GetRouter (void) const
{
  return m_route;
}

void DhcpHeader::SetLease (uint32_t time)
{
  if (m_opt[OP_LEASE] == false)
    {
      m_len += 6;
      m_opt[OP_LEASE] = true;
    }
  m_lease = time;
}

uint32_t DhcpHeader::GetLease (void) const
{
  return m_lease;
}

void DhcpHeader::SetRenew (uint32_t time)
{
  if (m_opt[OP_RENEW] == false)
    {
      m_len += 6;
      m_opt[OP_RENEW] = true;
    }
  m_renew = time;
}

uint32_t DhcpHeader::GetRenew (void) const
{
  return m_renew;
}

void DhcpHeader::SetRebind (uint32_t time)
{
  if (m_opt[OP_REBIND] == false)
    {
      m_len += 6;
      m_opt[OP_REBIND] = true;
    }
  m_rebind = time;
}

uint32_t DhcpHeader::GetRebind (void) const
{
  return m_rebind;
}

void DhcpHeader::ResetOpt ()
{
  m_len = 241;
  int i;
  for (i = 0; i < OP_END; i++)
    {
      m_opt[i] = false;
    }
}

uint32_t DhcpHeader::GetSerializedSize (void) const
{
  return m_len;
}

TypeId DhcpHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DhcpHeader")
    .SetParent<Header> ()
    .SetGroupName ("Internet-Apps")
    .AddConstructor<DhcpHeader> ()
  ;
  return tid;
}

TypeId DhcpHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void DhcpHeader::Print (std::ostream &os) const
{
  os << "(type=" << m_op << ")";
}

void
DhcpHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_bootp);
  i.WriteU8 (m_hType);
  i.WriteU8 (m_hLen);
  i.WriteU8 (m_hops);
  i.WriteU32 (m_xid);
  i.WriteHtonU16 (m_secs);
  i.WriteU16 ( m_flags);
  WriteTo (i, m_ciAddr);
  WriteTo (i, m_yiAddr);
  WriteTo (i, m_siAddr);
  WriteTo (i, m_giAddr);
  i.Write (m_chaddr, 16);
  i.Write (m_sname,64);
  i.Write (m_file,128);
  i.Write (m_magic_cookie,4);
  if (m_opt[OP_MASK])
    {
      i.WriteU8 (OP_MASK);
      i.WriteU8 (4);
      i.WriteHtonU32 (m_mask);
    }
  if (m_opt[OP_MSGTYPE])
    {
      i.WriteU8 (OP_MSGTYPE);
      i.WriteU8 (1);
      i.WriteU8 ((m_op + 1));
    }
  if (m_opt[OP_ADDREQ])
    {
      i.WriteU8 (OP_ADDREQ);
      i.WriteU8 (4);
      WriteTo (i, m_req);
    }
  if (m_opt[OP_SERVID])
    {
      i.WriteU8 (OP_SERVID);
      i.WriteU8 (4);
      WriteTo (i, m_dhcps);
    }
  if (m_opt[OP_ROUTE])
    {
      i.WriteU8 (OP_ROUTE);
      i.WriteU8 (4);
      WriteTo (i, m_route);
    }
  if (m_opt[OP_LEASE])
    {
      i.WriteU8 (OP_LEASE);
      i.WriteU8 (4);
      i.WriteHtonU32 (m_lease);
    }
  if (m_opt[OP_RENEW])
    {
      i.WriteU8 (OP_RENEW);
      i.WriteU8 (4);
      i.WriteHtonU32 (m_renew);
    }
  if (m_opt[OP_REBIND])
    {
      i.WriteU8 (OP_REBIND);
      i.WriteU8 (4);
      i.WriteHtonU32 (m_rebind);
    }
  i.WriteU8 (OP_END);
}

uint32_t DhcpHeader::Deserialize (Buffer::Iterator start)
{
  uint32_t len, clen = start.GetSize ();
  if (clen < 240)
    {
      NS_LOG_WARN ("Malformed Packet");
      return 0;
    }
  Buffer::Iterator i = start;
  m_bootp = i.ReadU8 ();
  m_hType = i.ReadU8 ();
  m_hLen = i.ReadU8 ();
  m_hops = i.ReadU8 ();
  m_xid = i.ReadU32 ();
  m_secs = i.ReadNtohU16 ();
  m_flags = i.ReadU16 ();
  ReadFrom (i, m_ciAddr);
  ReadFrom (i, m_yiAddr);
  ReadFrom (i, m_siAddr);
  ReadFrom (i, m_giAddr);
  i.Read (m_chaddr, 16);
  i.Read (m_sname, 64);
  i.Read (m_file, 128);
  i.Read (m_magic_cookie, 4);
  if ( m_magic_cookie[0] != 99 || m_magic_cookie[1] != 130 || m_magic_cookie[2] != 83 || m_magic_cookie[3] != 99)
    {
      NS_LOG_WARN ("Malformed Packet");
      return 0;
    }
  len = 240;
  uint8_t option;
  bool loop = true;
  do
    {
      if (len + 1 <= clen)
        {
          option = i.ReadU8 ();
          len += 1;
        }
      else
        {
          NS_LOG_WARN ("Malformed Packet");
          return 0;
        }
      switch (option)
        {
        case OP_MASK:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              m_mask = i.ReadNtohU32 ();
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_ROUTE:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              ReadFrom (i, m_route);
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_MSGTYPE:
          if (len + 2 < clen)
            {
              i.ReadU8 ();
              m_op = (i.ReadU8 () - 1);
              len += 2;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_SERVID:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              ReadFrom (i, m_dhcps);
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_ADDREQ:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              ReadFrom (i, m_req);
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_LEASE:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              m_lease = i.ReadNtohU32 ();
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_RENEW:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              m_renew = i.ReadNtohU32 ();
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_REBIND:
          if (len + 5 < clen)
            {
              i.ReadU8 ();
              m_rebind = i.ReadNtohU32 ();
              len += 5;
            }
          else
            {
              NS_LOG_WARN ("Malformed Packet");
              return 0;
            }
          break;
        case OP_END:
          loop = false;
          break;
        default:
          NS_LOG_WARN ("Malformed Packet");
          return 0;
        }
    }
  while (loop);

  m_len = len;
  return m_len;
}

} // namespace ns3
