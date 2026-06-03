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
 *
 * Author: Stefano Avallone <stefano.avallone@unina.it>
 */
#ifndef QUEUE_ITEM_H
#define QUEUE_ITEM_H

#include "ns3/ptr.h"
#include "ns3/simple-ref-count.h"
#include <ns3/address.h>
#include "ns3/nstime.h"

namespace ns3 {

class Packet;

/**
 * \ingroup network
 * \defgroup netdevice Network Device
 */

/**
 * \ingroup netdevice
 * \brief Base class to represent items of packet Queues
 *
 * An item stored in an ns-3 packet Queue contains a packet and possibly other
 * information. An item of the base class only contains a packet. Subclasses
 * can be derived from this base class to allow items to contain additional
 * information.
 */
class QueueItem : public SimpleRefCount<QueueItem>
{
public:
  /**
   * \brief Create a queue item containing a packet.
   * \param p the packet included in the created item.
   */
  QueueItem (Ptr<Packet> p);

  virtual ~QueueItem ();

  // Delete default constructor, copy constructor and assignment operator to avoid misuse
  QueueItem () = delete;
  QueueItem (const QueueItem &) = delete;
  QueueItem & operator = (const QueueItem &) = delete;

  /**
   * \return the packet included in this item.
   */
  Ptr<Packet> GetPacket (void) const;

  /**
   * \brief Use this method (instead of GetPacket ()->GetSize ()) to get the packet size
   *
   * Subclasses may keep header and payload separate to allow manipulating the header,
   * so using this method ensures that the correct packet size is returned.
   *
   * \return the size of the packet included in this item.
   */
  virtual uint32_t GetSize (void) const;

  /**
   * \enum Uint8Values
   * \brief 1-byte fields of the packet whose value can be retrieved, if present
   */
  enum Uint8Values
    {
      IP_DSFIELD
    };

  /**
   * \brief Retrieve the value of a given field from the packet, if present
   * \param field the field whose value has to be retrieved
   * \param value the output parameter to store the retrieved value
   *
   * \return true if the requested field is present in the packet, false otherwise.
   */
  virtual bool GetUint8Value (Uint8Values field, uint8_t &value) const;

  /**
   * \brief Print the item contents.
   * \param os output stream in which the data should be printed.
   */
  virtual void Print (std::ostream &os) const;

  /**
   * TracedCallback signature for Ptr<QueueItem>
   *
   * \param [in] item The queue item.
   */
  typedef void (* TracedCallback) (Ptr<const QueueItem> item);

private:
  /**
   * The packet contained in the queue item.
   */
  Ptr<Packet> m_packet;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param item the item
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const QueueItem &item);


/**
 * \ingroup network
 *
 * QueueDiscItem is the abstract base class for items that are stored in a queue
 * disc. It is derived from QueueItem (which only consists of a Ptr<Packet>)
 * to additionally store the destination MAC address, the
 * L3 protocol number and the transmission queue index,
 */
class QueueDiscItem : public QueueItem {
public:
  /**
   * \brief Create a queue disc item.
   * \param p the packet included in the created item.
   * \param addr the destination MAC address
   * \param protocol the L3 protocol number
   */
  QueueDiscItem (Ptr<Packet> p, const Address & addr, uint16_t protocol);

  virtual ~QueueDiscItem ();

  // Delete default constructor, copy constructor and assignment operator to avoid misuse
  QueueDiscItem () = delete;
  QueueDiscItem (const QueueDiscItem &) = delete;
  QueueDiscItem & operator = (const QueueDiscItem &) = delete;

  /**
   * \brief Get the MAC address included in this item
   * \return the MAC address included in this item.
   */
  Address GetAddress (void) const;

  /**
   * \brief Get the L3 protocol included in this item
   * \return the L3 protocol included in this item.
   */
  uint16_t GetProtocol (void) const;

  /**
   * \brief Get the transmission queue index included in this item
   * \return the transmission queue index included in this item.
   */
  uint8_t GetTxQueueIndex (void) const;

  /**
   * \brief Set the transmission queue index to store in this item
   * \param txq the transmission queue index to store in this item.
   */
  void SetTxQueueIndex (uint8_t txq);

  /**
   * \brief Get the timestamp included in this item
   * \return the timestamp included in this item.
   */
  Time GetTimeStamp (void) const;

  /**
   * \brief Set the timestamp included in this item
   * \param t the timestamp to include in this item.
   */
  void SetTimeStamp (Time t);

  /**
   * \brief Add the header to the packet
   *
   * Subclasses may keep header and payload separate to allow manipulating the header,
   * so this method allows to add the header to the packet before sending the packet
   * to the device.
   */
  virtual void AddHeader (void) = 0;

  /**
   * \brief Print the item contents.
   * \param os output stream in which the data should be printed.
   */
  virtual void Print (std::ostream &os) const;

  /**
   * \brief Marks the packet as a substitute for dropping it, such as for Explicit Congestion Notification
   *
   * \return true if the packet is marked by this method or is already marked, false otherwise
   */
  virtual bool Mark (void) = 0;

  /**
   * \brief Computes the hash of various fields of the packet header
   *
   * This method just returns 0. Subclasses should implement a reasonable hash
   * for their protocol type, such as hashing on the TCP/IP 5-tuple.
   *
   * \param perturbation hash perturbation value
   * \return the hash of various fields of the packet header
   */
  virtual uint32_t Hash (uint32_t perturbation = 0) const;

private:
  Address m_address;      //!< MAC destination address
  uint16_t m_protocol;    //!< L3 Protocol number
  uint8_t m_txq;          //!< Transmission queue index
  Time m_tstamp;          //!< timestamp when the packet was enqueued
};

} // namespace ns3

#endif /* QUEUE_ITEM_H */
