/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Universita' degli Studi di Napoli Federico II
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

#ifndef ARP_QUEUE_DISC_ITEM_H
#define ARP_QUEUE_DISC_ITEM_H

#include "ns3/packet.h"
#include "ns3/queue-item.h"
#include "arp-header.h"

namespace ns3 {

/**
 * \ingroup arp
 * \ingroup traffic-control
 *
 * ArpQueueDiscItem is a subclass of QueueDiscItem which stores ARP packets.
 * Header and payload are kept separate to allow the queue disc to hash the
 * fields of the header, which is added to the packet when the packet is dequeued.
 */
class ArpQueueDiscItem : public QueueDiscItem {
public:
  /**
   * \brief Create an ARP queue disc item containing an ARP packet.
   * \param p the packet included in the created item.
   * \param addr the destination MAC address
   * \param protocol the protocol number
   * \param header the ARP header
   */
  ArpQueueDiscItem (Ptr<Packet> p, const Address & addr, uint16_t protocol, const ArpHeader & header);

  /** Destructor. */
  virtual ~ArpQueueDiscItem ();

  // Delete default constructor, copy constructor and assignment operator to avoid misuse
  ArpQueueDiscItem () = delete;
  ArpQueueDiscItem (const ArpQueueDiscItem &) = delete;
  ArpQueueDiscItem & operator = (const ArpQueueDiscItem &) = delete;

  /**
   * \return the correct packet size (header plus payload).
   */
  virtual uint32_t GetSize (void) const;

  /**
   * \return the header stored in this item..
   */
  const ArpHeader & GetHeader (void) const;

  /**
   * \brief Add the header to the packet
   */
  virtual void AddHeader (void);

  /**
   * \brief Print the item contents.
   * \param os output stream in which the data should be printed.
   */
  virtual void Print (std::ostream &os) const;

  /**
   * \brief Inherited from the base class, but we cannot mark ARP packets
   * \return false
   */
  virtual bool Mark (void);

  /**
   * \brief Computes the hash of the packet's 5-tuple
   *
   * \param perturbation hash perturbation value
   * \return the hash of the packet's 5-tuple
   */
  virtual uint32_t Hash (uint32_t perturbation) const;

private:
  ArpHeader m_header;  //!< The ARP header.
  bool m_headerAdded;  //!< True if the header has already been added to the packet.
};

} // namespace ns3

#endif /* ARP_QUEUE_DISC_ITEM_H */
