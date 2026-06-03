/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
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
 */

#ifndef IPV6_QUEUE_DISC_ITEM_H
#define IPV6_QUEUE_DISC_ITEM_H

#include "ns3/packet.h"
#include "ns3/queue-item.h"
#include "ipv6-header.h"

namespace ns3 {

/**
 * \ingroup ipv6
 * \ingroup traffic-control
 *
 * Ipv6QueueDiscItem is a subclass of QueueDiscItem which stores IPv6 packets.
 * Header and payload are kept separate to allow the queue disc to manipulate
 * the header, which is added to the packet when the packet is dequeued.
 */
class Ipv6QueueDiscItem : public QueueDiscItem {
public:
  /**
   * \brief Create an IPv6 queue disc item containing an IPv6 packet.
   * \param p the packet included in the created item.
   * \param addr the destination MAC address
   * \param protocol the protocol number
   * \param header the IPv6 header
   */
  Ipv6QueueDiscItem (Ptr<Packet> p, const Address & addr, uint16_t protocol, const Ipv6Header & header);

  virtual ~Ipv6QueueDiscItem ();

  // Delete default constructor, copy constructor and assignment operator to avoid misuse
  Ipv6QueueDiscItem () = delete;
  Ipv6QueueDiscItem (const Ipv6QueueDiscItem &) = delete;
  Ipv6QueueDiscItem & operator = (const Ipv6QueueDiscItem &) = delete;

  /**
   * \return the correct packet size (header plus payload).
   */
  virtual uint32_t GetSize (void) const;

  /**
   * \return the header stored in this item..
   */
  const Ipv6Header & GetHeader (void) const;

  /**
   * \brief Add the header to the packet
   */
  virtual void AddHeader (void);

  /**
   * \brief Print the item contents.
   * \param os output stream in which the data should be printed.
   */
  virtual void Print (std::ostream &os) const;

  /*
   * The values for the fields of the Ipv6 header are taken from m_header and
   * thus might differ from those present in the packet in case the header is
   * modified after being added to the packet. However, this function is likely
   * to be called before the header is added to the packet (i.e., before the
   * packet is dequeued from the queue disc)
   */
  virtual bool GetUint8Value (Uint8Values field, uint8_t &value) const;

  /**
   * \brief Marks the packet by setting ECN_CE bits if the packet has 
   * ECN_ECT0 or ECN_ECT1 set.  If ECN_CE is already set, returns true.
   * \return true if the method results in a marked packet, false otherwise
   */
  virtual bool Mark (void);

  /**
   * \brief Computes the hash of the packet's 5-tuple
   *
   * Computes the hash of the source and destination IP addresses, protocol
   * number and, if the transport protocol is either UDP or TCP, the source
   * and destination port
   *
   * \param perturbation hash perturbation value
   * \return the hash of the packet's 5-tuple
   */
  virtual uint32_t Hash (uint32_t perturbation) const;

private:
  Ipv6Header m_header;  //!< The IPv6 header.
  bool m_headerAdded;   //!< True if the header has already been added to the packet.
};

} // namespace ns3

#endif /* IPV6_QUEUE_DISC_ITEM_H */
