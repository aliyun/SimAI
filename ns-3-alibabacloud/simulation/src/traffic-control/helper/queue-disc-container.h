/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef QUEUE_DISC_CONTAINER_H
#define QUEUE_DISC_CONTAINER_H

#include <stdint.h>
#include <vector>
#include "ns3/queue-disc.h"

namespace ns3 {

/**
 * \ingroup traffic-control
 *
 * \brief Holds a vector of ns3::QueueDisc pointers.
 *
 * Typically ns-3 QueueDiscs are installed on net devices using a traffic control
 * helper. The helper Install method takes a NetDeviceContainer which holds
 * some number of Ptr<NetDevice>. For each of the net devices in the
 * NetDeviceContainer the helper will instantiate a queue disc and install
 * it to the net device. For each of the queue discs, the helper also
 * adds the queue disc into a Container for later use by the caller.
 * This is that container used to hold the Ptr<QueueDisc> which are
 * instantiated by the traffic control helper.
 */
class QueueDiscContainer
{
public:
  /// QueueDisc container const iterator
  typedef std::vector<Ptr<QueueDisc> >::const_iterator ConstIterator;

  /**
   * Create an empty QueueDiscContainer.
   */
  QueueDiscContainer ();

  /**
   * \param qDisc a queue disc to add to the container
   *
   * Create a QueueDiscContainer with exactly one queue disc that has previously
   * been instantiated
   */
  QueueDiscContainer (Ptr<QueueDisc> qDisc);

  /**
   * \brief Get a const iterator which refers to the first QueueDisc in the
   * container.
   *
   * QueueDiscs can be retrieved from the container in two ways. First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a
   * for-loop to run through the QueueDiscs
   *
   * \code
   *   QueueDiscContainer::ConstIterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some QueueDisc method
   *     }
   * \endcode
   *
   * \returns a const iterator which refers to the first QueueDisc in the container.
   */
  ConstIterator Begin (void) const;

  /**
   * \brief Get a const iterator which indicates past-the-last QueueDisc in the
   * container.
   *
   * QueueDiscs can be retrieved from the container in two ways. First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a
   * for-loop to run through the QueueDiscs
   *
   * \code
   *   QueueDiscContainer::ConstIterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some QueueDisc method
   *     }
   * \endcode
   *
   * \returns a const iterator which indicates an ending condition for a loop.
   */
  ConstIterator End (void) const;

  /**
   * \brief Get the number of Ptr<QueueDisc> stored in this container.
   *
   * QueueDiscs can be retrieved from the container in two ways. First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is typically used to
   * define an ending condition in a for-loop that runs through the stored
   * QueueDiscs
   *
   * \code
   *   uint32_t nQueueDiscs = container.GetN ();
   *   for (uint32_t i = 0 i < nQueueDiscs; ++i)
   *     {
   *       Ptr<QueueDisc> p = container.Get (i)
   *       i->method ();  // some QueueDisc method
   *     }
   * \endcode
   *
   * \returns the number of Ptr<QueueDisc> stored in this container.
   */
  std::size_t GetN (void) const;

  /**
   * \brief Get the Ptr<QueueDisc> stored in this container at a given
   * index.
   *
   * QueueDiscs can be retrieved from the container in two ways. First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is used to retrieve the
   * indexed Ptr<QueueDisc>.
   *
   * \code
   *   uint32_t nQueueDiscs = container.GetN ();
   *   for (uint32_t i = 0 i < nQueueDiscs; ++i)
   *     {
   *       Ptr<QueueDisc> p = container.Get (i)
   *       i->method ();  // some QueueDisc method
   *     }
   * \endcode
   *
   * \param i the index of the requested queue disc pointer.
   * \returns the requested queue disc pointer.
   */
  Ptr<QueueDisc> Get (std::size_t i) const;

  /**
   * \brief Append the contents of another QueueDiscContainer to the end of
   * this container.
   *
   * \param other The QueueDiscContainer to append.
   */
  void Add (QueueDiscContainer other);

  /**
   * \brief Append a single Ptr<QueueDisc> to this container.
   *
   * \param qDisc The Ptr<QueueDisc> to append.
   */
  void Add (Ptr<QueueDisc> qDisc);

private:
  std::vector<Ptr<QueueDisc> > m_queueDiscs; //!< QueueDiscs smart pointers
};

} // namespace ns3

#endif /* QUEUE_DISC_CONTAINER_H */
