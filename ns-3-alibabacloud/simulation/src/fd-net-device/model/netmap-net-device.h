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
 * Author: Pasquale Imputato <p.imputato@gmail.com>
 */

#ifndef NETMAP_NET_DEVICE_H
#define NETMAP_NET_DEVICE_H

#include "ns3/net-device-queue-interface.h"
#include <mutex>
#include <thread>
#include "fd-net-device.h"
#include <net/netmap_user.h>
#include <atomic>

namespace ns3 {

/**
 * \ingroup fd-net-device
 *
 * \brief Network device transmission queue with lock
 *
 * This class stores information about a single transmission queue
 * of a network device that is exposed to queue discs. This class extends
 * the NetDeviceQueue base class by introducing a lock for methods which
 * require mutual exclusion on data access in emulation.
 */
class NetDeviceQueueLock : public NetDeviceQueue
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  NetDeviceQueueLock ();
  virtual ~NetDeviceQueueLock ();

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
   * \brief Called by the netdevice to report the number of bytes queued to the device queue
   * \param bytes number of bytes queued to the device queue
   */
  virtual void NotifyQueuedBytes (uint32_t bytes);

  /**
   * \brief Called by the netdevice to report the number of bytes it is going to transmit
   * \param bytes number of bytes the device is going to transmit
   */
  virtual void NotifyTransmittedBytes (uint32_t bytes);

private:
  mutable std::mutex m_mutex; //!< Mutex to serialize the operations performed on the queue
};

/**
 * \ingroup fd-net-device
 *
 * \brief This class performs the actual data reading from the netmap ring.
 */
class NetmapNetDeviceFdReader : public FdReader
{
public:
  NetmapNetDeviceFdReader ();

  /**
   * \brief Set size of the read buffer.
   * \param bufferSize the size of the read buffer
   */
  void SetBufferSize (uint32_t bufferSize);

  /**
   * \brief Set netmap interface representation.
   * \param nifp the netmap interface representation
   */
  void SetNetmapIfp (struct netmap_if *nifp);

private:
  FdReader::Data DoRead (void);

  uint32_t m_bufferSize; //!< size of the read buffer
  struct netmap_if *m_nifp; //!< Netmap interface representation
};

/**
 * \ingroup fd-net-device
 *
 * \brief a NetDevice to read/write network traffic from/into a netmap file descriptor.
 *
 * A NetmapNetDevice object will read and write packets from/to a netmap file descriptor.
 *
 */
class NetmapNetDevice : public FdNetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  NetmapNetDevice ();
  virtual ~NetmapNetDevice ();

  /**
   * \brief Get the number of bytes currently in the netmap transmission ring.
   * \return the number of bytes in the netmap transmission ring.
   */
  uint32_t GetBytesInNetmapTxRing ();

  /**
   * \brief Get the number of slots currently available in the netmap transmission ring.
   * \return the number of slots currently available in the netmap transmission ring.
   */
  int GetSpaceInNetmapTxRing () const;

  /**
   * \brief Set the NetDeviceQueue
   * \param queue the NetDeviceQueue
   */
  void SetNetDeviceQueue (Ptr<NetDeviceQueue> queue);

  /**
   * \brief Set the netmap interface representation
   * \param nifp the pointer to netmap interface representation
   */
  void SetNetmapInterfaceRepresentation (struct netmap_if *nifp);

  /**
   * \brief Set the netmap transmission rings info
   * \param nTxRings the number of transmission rings
   * \param nTxRingsSlots the number of slots for each transmission ring
   */
  void SetTxRingsInfo (uint32_t nTxRings, uint32_t nTxRingsSlots);

  /**
   * \brief Set the netmap receiver rings info
   * \param nRxRings the number of receiver rings
   * \param nRxRingsSlots the number of slots for each receiver ring
   */
  void SetRxRingsInfo (uint32_t nRxRings, uint32_t nRxRingsSlots);

  /**
   * \brief The function Writes a packet into the netmap transmission ring.
   * \param buffer the pointer to packet
   * \param length the packet length
   * \return the number of written bytes
   */
  virtual ssize_t Write (uint8_t *buffer, size_t length);

private:
  Ptr<FdReader> DoCreateFdReader (void);
  void DoFinishStartingDevice (void);
  void DoFinishStoppingDevice (void);

  /**
   * \brief This function syncs netmap ring and notifies netdevice queue.
   * This function runs in a separate thread.
   */
  virtual void SyncAndNotifyQueue ();

  struct netmap_if *m_nifp; //!< Netmap interface representation
  uint32_t m_nTxRings; //!< Number of transmission rings
  uint32_t m_nTxRingsSlots; //!< Number of slots in the transmission rings
  uint32_t m_nRxRings; //!< Number of receiver rings
  uint32_t m_nRxRingsSlots; //!< Number of slots in the receiver rings
  Ptr<NetDeviceQueue> m_queue; //!< NetDevice queue
  uint32_t m_totalQueuedBytes; //!< Total queued bytes
  std::thread m_syncAndNotifyQueueThread; //!< Thread used to perform the flow control
  std::atomic<bool> m_syncAndNotifyQueueThreadRun; //!< Running flag of the flow control thread
  uint8_t m_syncAndNotifyQueuePeriod; //!< The period of time in us after which the device syncs the netmap ring and notifies queue status
};

} // namespace ns3

#endif /* NETMAP_NET_DEVICE_H */
