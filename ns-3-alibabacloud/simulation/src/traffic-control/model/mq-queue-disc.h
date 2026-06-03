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
 *          Stefano Avallone <stavallo@unina.it>
 */

#ifndef MQ_QUEUE_DISC_H
#define MQ_QUEUE_DISC_H

#include "ns3/queue-disc.h"

namespace ns3 {

/**
 * \ingroup traffic-control
 *
 * mq is a classful multi-queue aware dummy scheduler. It has as many child
 * queue discs as the number of device transmission queues. Packets are
 * directly enqueued into and dequeued from child queue discs.
 */
class MqQueueDisc : public QueueDisc {
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief MqQueueDisc constructor
   */
  MqQueueDisc ();

  virtual ~MqQueueDisc();

 /**
   * \brief Return the wake mode adopted by this queue disc.
   * \return the wake mode adopted by this queue disc.
   */
  WakeMode GetWakeMode (void) const;

private:
  virtual bool DoEnqueue (Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue (void);
  virtual Ptr<const QueueDiscItem> DoPeek (void);
  virtual bool CheckConfig (void);
  virtual void InitializeParams (void);
};

} // namespace ns3

#endif /* MQ_QUEUE_DISC_H */
