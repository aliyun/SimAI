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
 * Authors: Pasquale Imputato <p.imputato@gmail.com>
 *          Stefano Avallone <stefano.avallone@unina.it>
 */

#ifndef QUEUE_LIMITS_H
#define QUEUE_LIMITS_H

#include "ns3/object.h"

namespace ns3 {

/**
 * \ingroup network
 *
 * \brief Abstract base class for NetDevice queue length controller
 *
 * QueueLimits is an abstract base class providing the interface to
 * the NetDevice queue length controller.
 *
 * Child classes need to implement the methods used for a byte-based
 * measure of the queue length.
 *
 * The design and implementation of this class is inspired by Linux.
 * For more details, see the queue limits Sphinx documentation.
 */
class QueueLimits : public Object {
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  virtual ~QueueLimits ();

  /**
   * \brief Reset queue limits state
   */
  virtual void Reset () = 0;

  /**
   * \brief Record number of completed bytes and recalculate the limit
   * \param count the number of completed bytes
   */
  virtual void Completed (uint32_t count) = 0;

  /**
   * Available is called from NotifyTransmittedBytes to calculate the
   * number of bytes that can be passed again to the NetDevice.
   * A negative value means that no packets can be passed to the NetDevice.
   * In this case, NotifyTransmittedBytes stops the transmission queue.
   * \brief Returns how many bytes can be queued
   * \return the number of bytes that can be queued
   */
  virtual int32_t Available () const = 0;

  /**
   * \brief Record the number of bytes queued
   * \param count the number of bytes queued
   */
  virtual void Queued (uint32_t count) = 0;
};

} // namespace ns3

#endif /* QUEUE_LIMITS_H */
