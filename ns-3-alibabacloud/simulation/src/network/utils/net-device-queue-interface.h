/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stefano.avallone@.unina.it>
 */
#ifndef NET_DEVICE_QUEUE_INTERFACE_H
#define NET_DEVICE_QUEUE_INTERFACE_H

#include <vector>
#include <functional>
#include "ns3/callback.h"
#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/log.h"
#include "ns3/net-device.h"
#include "ns3/object-factory.h"

namespace ns3 {

class QueueLimits;
class NetDeviceQueueInterface;
class QueueItem;


/**
 * \ingroup network
 * \defgroup netdevice Network Device
 */

/**
 * \ingroup netdevice
 *
 * \brief Network device transmission queue
 *
 * This class stores information about a single transmission queue
 * of a network device that is exposed to queue discs. Such information
 * includes the state of the transmission queue (whether it has been
 * stopped or not) and data used by techniques such as Byte Queue Limits.
 *
 * This class roughly models the struct netdev_queue of Linux.
 */
class NetDeviceQueue : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  NetDeviceQueue ();
  virtual ~NetDeviceQueue();

  /**
   * Called by the device to start this device transmission queue.
   * This is the analogous to the netif_tx_start_queue function of the Linux kernel.
   */
  virtual void Start (void);

  /**
   * Called by the device to stop this device transmission queue.
   * This is the analogous to the netif_tx_stop_queue function of the Linux kernel.
   */
  virtual void Stop (void);

  /**
   * Called by the device to wake the queue disc associated with this
   * device transmission queue. This is done by invoking the wake callback.
   * This is the analogous to the netif_tx_wake_queue function of the Linux kernel.
   */
  virtual void Wake (void);

  /**
   * \brief Get the status of the device transmission queue.
   * \return true if the device transmission queue is stopped.
   *
   * Called by queue discs to enquire about the status of a given transmission queue.
   * This is the analogous to the netif_xmit_stopped function of the Linux kernel.
   */
  virtual bool IsStopped (void) const;

  /**
   * \brief Notify this NetDeviceQueue that the NetDeviceQueueInterface was
   *        aggregated to an object.
   *
   * \param ndqi the NetDeviceQueueInterface.
   *
   * This NetDeviceQueue stores a pointer to the NetDevice the NetDeviceQueueInterface
   * was aggregated to.
   */
  void NotifyAggregatedObject (Ptr<NetDeviceQueueInterface> ndqi);

  /// Callback invoked by netdevices to wake upper layers
  typedef Callback< void > WakeCallback;

  /**
   * \brief Set the wake callback
   * \param cb the callback to set
   *
   * Called by the traffic control layer to set the wake callback. The wake callback
   * is invoked by the device whenever it is needed to "wake" the upper layers (i.e.,
   * solicitate the queue disc associated with this transmission queue (in case of
   * multi-queue aware queue discs) or to the network device (otherwise) to send
   * packets down to the device).
   */
  virtual void SetWakeCallback (WakeCallback cb);

  /**
   * \brief Called by the netdevice to report the number of bytes queued to the device queue
   * \param bytes number of bytes queued to the device queue
   */
  virtual void NotifyQueuedBytes (uint32_t bytes);

  /**
   * \brief Called by the netdevice to report the number of bytes it is going to transmit
   * \param bytes number of bytes the device is going to transmit
   */
  virtual void NotifyTransmittedBytes (uint32_t bytes);

  /**
   * \brief Reset queue limits state
   */
  void ResetQueueLimits ();

  /**
   * \brief Set queue limits to this queue
   * \param ql the queue limits associated to this queue
   */
  void SetQueueLimits (Ptr<QueueLimits> ql);

  /**
   * \brief Get queue limits to this queue
   * \return the queue limits associated to this queue
   */
  Ptr<QueueLimits> GetQueueLimits ();

  /**
   * \brief Perform the actions required by flow control and dynamic queue
   *        limits when a packet is enqueued in the queue of a netdevice
   *
   * \param queue the device queue
   * \param item the enqueued packet
   *
   * This method must be connected to the "Enqueue" traced callback of a Queue
   * object (through a bound callback) in order for a netdevice to support
   * flow control and dynamic queue limits.
   */
  template <typename QueueType>
  void PacketEnqueued (QueueType* queue, Ptr<const typename QueueType::ItemType> item);

  /**
   * \brief Perform the actions required by flow control and dynamic queue
   *        limits when a packet is dequeued (or dropped after dequeue) from
   *        the queue of a netdevice
   *
   * \param queue the device queue
   * \param item the dequeued packet
   *
   * This method must be connected to the "Dequeue" traced callback of a Queue
   * object (through a bound callback) in order for
   * a netdevice to support flow control and dynamic queue limits.
   */
  template <typename QueueType>
  void PacketDequeued (QueueType* queue, Ptr<const typename QueueType::ItemType> item);

  /**
   * \brief Perform the actions required by flow control and dynamic queue
   *        limits when a packet is dropped before being enqueued in the queue
   *        of a netdevice (which likely indicates that the queue is full)
   *
   * \param queue the device queue
   * \param item the dropped packet
   *
   * This method must be connected to the "DropBeforeEnqueue" traced callback
   * of a Queue object (through a bound callback) in order for a netdevice to
   * support flow control and dynamic queue limits.
   */
  template <typename QueueType>
  void PacketDiscarded (QueueType* queue, Ptr<const typename QueueType::ItemType> item);

  /**
   * \brief Connect the traced callbacks of a queue to the methods providing support
   *        for flow control and dynamic queue limits. A queue can be any object providing:
   *        - "Enqueue", "Dequeue", "DropBeforeEnqueue" traces
   *        - an ItemType typedef for the type of stored items
   *        - GetCurrentSize and GetMaxSize methods
   * \param queue the queue
   */
  template <typename QueueType>
  void ConnectQueueTraces (Ptr<QueueType> queue);

private:
  bool m_stoppedByDevice;         //!< True if the queue has been stopped by the device
  bool m_stoppedByQueueLimits;    //!< True if the queue has been stopped by a queue limits object
  Ptr<QueueLimits> m_queueLimits; //!< Queue limits object
  WakeCallback m_wakeCallback;    //!< Wake callback
  Ptr<NetDevice> m_device;        //!< the netdevice aggregated to the NetDeviceQueueInterface

  NS_LOG_TEMPLATE_DECLARE;        //!< redefinition of the log component
};


/**
 * \ingroup netdevice
 *
 * \brief Network device transmission queue interface
 *
 * This interface is used by the traffic control layer and by the aggregated
 * device to access the transmission queues of the device. Additionally, through
 * this interface, traffic control aware netdevices can:
 * - set the number of transmission queues
 * - set the method used (by upper layers) to determine the transmission queue
 *   in which the netdevice would enqueue a given packet
 * NetDevice helpers create this interface and aggregate it to the device.
 */
class NetDeviceQueueInterface : public Object
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
  NetDeviceQueueInterface ();
  virtual ~NetDeviceQueueInterface ();

  /**
   * \brief Get the i-th transmission queue of the device.
   *
   * \param i the index of the requested queue.
   * \return the i-th transmission queue of the device.
   *
   * The index of the first transmission queue is zero.
   */
  Ptr<NetDeviceQueue> GetTxQueue (std::size_t i) const;

  /**
   * \brief Get the number of device transmission queues.
   * \return the number of device transmission queues.
   */
  std::size_t GetNTxQueues (void) const;

  /**
   * \brief Set the type of device transmission queues to create.
   * \param type type of device transmission queues to create.
   *
   * This method is called when the TxQueuesType attribute is set to create
   * the corresponding type of device transmission queues. It cannot be
   * called again afterwards.
   */
  void SetTxQueuesType (TypeId type);

  /**
   * \brief Set the number of device transmission queues to create.
   * \param numTxQueues number of device transmission queues to create.
   *
   * This method is called when the NTxQueues attribute is set to create
   * the corresponding number of device transmission queues. It cannot be
   * called again afterwards.
   */
  void SetNTxQueues (std::size_t numTxQueues);

  /// Callback invoked to determine the tx queue selected for a given packet
  typedef std::function<std::size_t (Ptr<QueueItem>)> SelectQueueCallback;

  /**
   * \brief Set the select queue callback.
   * \param cb the callback to set.
   *
   * This method is called to set the select queue callback, i.e., the
   * method used to select a device transmission queue for a given packet.
   */
  void SetSelectQueueCallback (SelectQueueCallback cb);

  /**
   * \brief Get the select queue callback.
   * \return the select queue callback.
   *
   * Called by the traffic control layer to get the select queue callback set
   * by a multi-queue device.
   */
  SelectQueueCallback GetSelectQueueCallback (void) const;

protected:
  /**
   * \brief Dispose of the object
   */
  virtual void DoDispose (void);
  /**
   * \brief Notify that an object was aggregated
   */
  virtual void NotifyNewAggregate (void);

private:
  ObjectFactory m_txQueues;   //!< Device transmission queues TypeId
  std::vector< Ptr<NetDeviceQueue> > m_txQueuesVector;   //!< Device transmission queues
  SelectQueueCallback m_selectQueueCallback;   //!< Select queue callback
};


/**
 * Implementation of the templates declared above.
 */

template <typename QueueType>
void
NetDeviceQueue::ConnectQueueTraces (Ptr<QueueType> queue)
{
  NS_ASSERT (queue != 0);

  queue->TraceConnectWithoutContext ("Enqueue",
                                     MakeCallback (&NetDeviceQueue::PacketEnqueued<QueueType>, this)
                                     .Bind (PeekPointer (queue)));
  queue->TraceConnectWithoutContext ("Dequeue",
                                     MakeCallback (&NetDeviceQueue::PacketDequeued<QueueType>, this)
                                     .Bind (PeekPointer (queue)));
  queue->TraceConnectWithoutContext ("DropBeforeEnqueue",
                                     MakeCallback (&NetDeviceQueue::PacketDiscarded<QueueType>, this)
                                     .Bind (PeekPointer (queue)));
}

template <typename QueueType>
void
NetDeviceQueue::PacketEnqueued (QueueType* queue, Ptr<const typename QueueType::ItemType> item)
{
  NS_LOG_FUNCTION (this << queue << item);

  // Inform BQL
  NotifyQueuedBytes (item->GetSize ());

  NS_ASSERT_MSG (m_device, "Aggregated NetDevice not set");
  // After enqueuing a packet, we need to check whether the queue is able to
  // store another packet. If not, we stop the queue

  if (queue->WouldOverflow (1, m_device->GetMtu ()))
    {
      NS_LOG_DEBUG ("The device queue is being stopped (" << queue->GetCurrentSize ()
                    << " inside)");
      Stop ();
    }
}

template <typename QueueType>
void
NetDeviceQueue::PacketDequeued (QueueType* queue, Ptr<const typename QueueType::ItemType> item)
{
  NS_LOG_FUNCTION (this << queue << item);

  // Inform BQL
  NotifyTransmittedBytes (item->GetSize ());

  NS_ASSERT_MSG (m_device, "Aggregated NetDevice not set");
  // After dequeuing a packet, if there is room for another packet we
  // call Wake () that ensures that the queue is not stopped and restarts
  // the queue disc if the queue was stopped

  if (!queue->WouldOverflow (1, m_device->GetMtu ()))
    {
      Wake ();
    }
}

template <typename QueueType>
void
NetDeviceQueue::PacketDiscarded (QueueType* queue, Ptr<const typename QueueType::ItemType> item)
{
  NS_LOG_FUNCTION (this << queue << item);

  // This method is called when a packet is discarded before being enqueued in the
  // device queue, likely because the queue is full. This should not happen if the
  // device correctly stops the queue. Anyway, stop the tx queue, so that the upper
  // layers do not send packets until there is room in the queue again.

  NS_LOG_ERROR ("BUG! No room in the device queue for the received packet! ("
                << queue->GetCurrentSize () << " inside)");

  Stop ();
}

} // namespace ns3

#endif /* NET_DEVICE_QUEUE_INTERFACE_H */
