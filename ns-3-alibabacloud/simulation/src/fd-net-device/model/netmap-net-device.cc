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

#include "netmap-net-device.h"
#include "ns3/uinteger.h"
#include <sys/ioctl.h>
#include <unistd.h>
#include <thread>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NetmapNetDevice");

TypeId
NetDeviceQueueLock::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NetDeviceQueueLock")
    .SetParent<NetDeviceQueue> ()
    .SetGroupName ("Network")
    .AddConstructor<NetDeviceQueueLock> ();
  return tid;
}

NetDeviceQueueLock::NetDeviceQueueLock ()
{}

NetDeviceQueueLock::~NetDeviceQueueLock ()
{}

bool
NetDeviceQueueLock::IsStopped (void) const
{
  m_mutex.lock ();
  bool stopped = NetDeviceQueue::IsStopped ();
  m_mutex.unlock ();
  return stopped;
}

void
NetDeviceQueueLock::Start (void)
{
  m_mutex.lock ();
  NetDeviceQueue::Start ();
  m_mutex.unlock ();
}

void
NetDeviceQueueLock::Stop (void)
{
  m_mutex.lock ();
  NetDeviceQueue::Stop ();
  m_mutex.unlock ();
}

void
NetDeviceQueueLock::Wake (void)
{
  m_mutex.lock ();
  NetDeviceQueue::Wake ();
  m_mutex.unlock ();
}

void
NetDeviceQueueLock::NotifyQueuedBytes (uint32_t bytes)
{
  m_mutex.lock ();
  NetDeviceQueue::NotifyQueuedBytes (bytes);
  m_mutex.unlock ();
}

void
NetDeviceQueueLock::NotifyTransmittedBytes (uint32_t bytes)
{
  m_mutex.lock ();
  NetDeviceQueue::NotifyTransmittedBytes (bytes);
  m_mutex.unlock ();
}

NetmapNetDeviceFdReader::NetmapNetDeviceFdReader ()
  : m_bufferSize (65536),
    // Defaults to maximum TCP window size
    m_nifp (nullptr)
{}

void
NetmapNetDeviceFdReader::SetBufferSize (uint32_t bufferSize)
{
  NS_LOG_FUNCTION (this << bufferSize);
  m_bufferSize = bufferSize;
}

void
NetmapNetDeviceFdReader::SetNetmapIfp (struct netmap_if *nifp)
{
  NS_LOG_FUNCTION (this << nifp);
  m_nifp = nifp;
}

FdReader::Data
NetmapNetDeviceFdReader::DoRead (void)
{
  NS_LOG_FUNCTION (this);

  uint8_t *buf = (uint8_t *) malloc (m_bufferSize);
  NS_ABORT_MSG_IF (buf == 0, "malloc() failed");

  NS_LOG_LOGIC ("Calling read on fd " << m_fd);

  struct netmap_ring *rxring;
  uint16_t len = 0;
  uint32_t rxRingIndex = 0;

  // we have a packet in one of the receiver rings
  // we check for the first non empty receiver ring
  while (rxRingIndex < m_nifp->ni_rx_rings)
    {
      rxring = NETMAP_RXRING (m_nifp, rxRingIndex);

      if (!nm_ring_empty (rxring))
        {
          uint32_t i = rxring->cur;
          uint8_t *buffer = (uint8_t *) NETMAP_BUF (rxring, rxring->slot[i].buf_idx);
          len = rxring->slot[i].len;
          NS_LOG_DEBUG ("Received a packet of " << len << " bytes");

          // copy buffer in the destination memory area
          memcpy (buf, buffer, len);

          // advance the netmap pointers and sync the fd
          rxring->head = rxring->cur = nm_ring_next (rxring, i);

          ioctl (m_fd, NIOCRXSYNC, NULL);

          break;
        }

      rxRingIndex++;
    }

  if (len <= 0)
    {
      free (buf);
      buf = 0;
      len = 0;
    }
  NS_LOG_LOGIC ("Read " << len << " bytes on fd " << m_fd);
  return FdReader::Data (buf, len);
}

NS_OBJECT_ENSURE_REGISTERED (NetmapNetDevice);

TypeId
NetmapNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NetmapNetDevice")
    .SetParent<FdNetDevice> ()
    .SetGroupName ("FdNetDevice")
    .AddConstructor<NetmapNetDevice> ()
    .AddAttribute ("SyncAndNotifyQueuePeriod",
                   "The period of time (in number of us) after which the device syncs the netmap ring and notifies queue status.",
                   UintegerValue (50),
                   MakeUintegerAccessor (&NetmapNetDevice::m_syncAndNotifyQueuePeriod),
                   MakeUintegerChecker<uint8_t> ());
  return tid;
}

NetmapNetDevice::NetmapNetDevice ()
{
  NS_LOG_FUNCTION (this);
  m_nifp = nullptr;
  m_nTxRings = 0;
  m_nTxRingsSlots = 0;
  m_nRxRings = 0;
  m_nRxRingsSlots = 0;
  m_queue = nullptr;
  m_totalQueuedBytes = 0;
  m_syncAndNotifyQueueThreadRun = false;
}

NetmapNetDevice::~NetmapNetDevice ()
{
  NS_LOG_FUNCTION (this);
  m_nifp = nullptr;
  m_queue = nullptr;
}

Ptr<FdReader>
NetmapNetDevice::DoCreateFdReader (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<NetmapNetDeviceFdReader> fdReader = Create<NetmapNetDeviceFdReader> ();
  // 22 bytes covers 14 bytes Ethernet header with possible 8 bytes LLC/SNAP
  fdReader->SetBufferSize (GetMtu () + 22);
  fdReader->SetNetmapIfp (m_nifp);
  return fdReader;
}

void
NetmapNetDevice::DoFinishStartingDevice (void)
{
  NS_LOG_FUNCTION (this);

  m_syncAndNotifyQueueThreadRun = true;
  m_syncAndNotifyQueueThread = std::thread (&NetmapNetDevice::SyncAndNotifyQueue, this);
}


void
NetmapNetDevice::DoFinishStoppingDevice (void)
{
  NS_LOG_FUNCTION (this);

  m_queue->Stop ();

  m_syncAndNotifyQueueThreadRun = false;

  if (m_syncAndNotifyQueueThread.joinable ())
    {
      m_syncAndNotifyQueueThread.join ();
    }
}

uint32_t
NetmapNetDevice::GetBytesInNetmapTxRing ()
{
  NS_LOG_FUNCTION (this);

  struct netmap_ring *txring;
  txring = NETMAP_TXRING (m_nifp, 0);

  int tail = txring->tail;

  // the netmap ring has one slot reserved
  int inQueue = (m_nTxRingsSlots - 1) - nm_ring_space (txring);

  uint32_t bytesInQueue = 0;

  for (int i = 1; i < inQueue; i++)
    {
      bytesInQueue += txring->slot[tail].len;
      tail++;
      tail = tail % m_nTxRingsSlots;
    }

  return bytesInQueue;
}

void
NetmapNetDevice::SetNetDeviceQueue (Ptr<NetDeviceQueue> queue)
{
  NS_LOG_FUNCTION (this);

  m_queue = queue;
}

void
NetmapNetDevice::SetNetmapInterfaceRepresentation (struct netmap_if *nifp)
{
  NS_LOG_FUNCTION (this << nifp);

  m_nifp = nifp;
}

void
NetmapNetDevice::SetTxRingsInfo (uint32_t nTxRings, uint32_t nTxRingsSlots)
{
  NS_LOG_FUNCTION (this << nTxRings << nTxRingsSlots);

  m_nTxRings = nTxRings;
  m_nTxRingsSlots = nTxRingsSlots;
}

void
NetmapNetDevice::SetRxRingsInfo (uint32_t nRxRings, uint32_t nRxRingsSlots)
{
  NS_LOG_FUNCTION (this << nRxRings << nRxRingsSlots);

  m_nRxRings = nRxRings;
  m_nRxRingsSlots = nRxRingsSlots;
}

int
NetmapNetDevice::GetSpaceInNetmapTxRing () const
{
  NS_LOG_FUNCTION (this);

  struct netmap_ring *txring;
  txring = NETMAP_TXRING (m_nifp, 0);

  return nm_ring_space (txring);
}

// This function runs in a separate thread.
void
NetmapNetDevice::SyncAndNotifyQueue ()
{
  NS_LOG_FUNCTION (this);

  struct netmap_ring *txring = NETMAP_TXRING (m_nifp, 0);

  uint32_t prevTotalTransmittedBytes = 0;

  while (m_syncAndNotifyQueueThreadRun == true)
    {
      // we sync the netmap ring periodically.
      // the traffic control layer can write packets during the period between two syncs.
      ioctl (GetFileDescriptor (), NIOCTXSYNC, NULL);

      // we need of a nearly periodic notification to queue limits of the transmitted bytes.
      uint32_t totalTransmittedBytes = m_totalQueuedBytes - GetBytesInNetmapTxRing ();
      uint32_t deltaBytes = totalTransmittedBytes - prevTotalTransmittedBytes;
      NS_LOG_DEBUG (deltaBytes << " delta transmitted bytes");
      prevTotalTransmittedBytes = totalTransmittedBytes;
      if (m_queue)
        {
          m_queue->NotifyTransmittedBytes (deltaBytes);

          if (GetSpaceInNetmapTxRing () >= 32) // WAKE_THRESHOLD
            {
              if (m_queue->IsStopped ())
                {
                  m_queue->Wake ();
                }
            }
        }

      usleep (m_syncAndNotifyQueuePeriod);

      NS_LOG_DEBUG ("Space in the netmap ring of " << nm_ring_space (txring) << " packets");
    }

  ioctl (GetFileDescriptor (), NIOCTXSYNC, NULL);

}

ssize_t
NetmapNetDevice::Write (uint8_t *buffer, size_t length)
{
  NS_LOG_FUNCTION (this << buffer << length);

  struct netmap_ring *txring;

  // we use one ring also in case of multiqueue device to perform an accurate flow control on that ring
  txring = NETMAP_TXRING (m_nifp, 0);

  uint16_t ret = -1;

  if (m_queue->IsStopped ())
    {
      // the device queue is stopped and we cannot write other packets
      return ret;
    }

  if (!nm_ring_empty (txring))
    {

      uint32_t i = txring->cur;
      uint8_t *buf = (uint8_t *) NETMAP_BUF (txring, txring->slot[i].buf_idx);

      memcpy (buf, buffer, length);
      txring->slot[i].len = length;

      txring->head = txring->cur = nm_ring_next (txring, i);

      ret = length;

      // we update the total transmitted bytes counter and notify queue limits of the queued bytes
      m_totalQueuedBytes += length;
      m_queue->NotifyQueuedBytes (length);

      // if there is no room for other packets then stop the queue.
      if (nm_ring_space (txring) == 0)
        {
          m_queue->Stop ();
        }
    }

  return ret;
}

} // namespace ns3
