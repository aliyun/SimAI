/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 NITK Surathkal
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
 * Author: Harsh Patel <thadodaharsh10@gmail.com>
 *         Hrishikesh Hiraskar <hrishihiraskar@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

#ifndef DPDK_NET_DEVICE_H
#define DPDK_NET_DEVICE_H

#include "fd-net-device.h"

#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_cycles.h>

struct rte_eth_dev_tx_buffer;
struct rte_mbuf;

namespace ns3
{

/**
 * \ingroup fd-net-device
 *
 * \brief a NetDevice to read/write network traffic from/into a
 *        Dpdk enabled port.
 *
 * A DpdkNetDevice object will read and write frames/packets
 * from/to a Dpdk enabled port.
 *
 */
class DpdkNetDevice : public FdNetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Constructor for the DpdkNetDevice.
   */
  DpdkNetDevice ();

  /**
   * Destructor for the DpdkNetDevice.
   */
  ~DpdkNetDevice ();

  /**
   * Check the link status of all ports in up to 9s
   * and print them finally
   */
  void CheckAllPortsLinkStatus (void);

  /**
   * Initialize Dpdk.
   * Initializes EAL.
   *
   * \param argc Dpdk EAL args count.
   * \param argv Dpdk EAL args list.
   * \param dpdkDriver Dpdk Driver to bind NIC to.
   */
  void InitDpdk (int argc, char **argv, std::string dpdkDriver);

  /**
   * Set device name.
   *
   * \param deviceName The device name.
   */
  void SetDeviceName (std::string deviceName);

  /**
   * A signal handler for SIGINT and SIGTERM signals.
   *
   * \param signum The signal number.
   */
  static void SignalHandler (int signum);

  /**
   * A function to handle rx & tx operations.
   * \param arg a pointer to the DpdkNetDevice
   * \return zero on failure or exit.
   */
  static int LaunchCore (void *arg);

  /**
   * Transmit packets in burst from the tx_buffer to the nic.
   */
  void HandleTx ();

  /**
   * Receive packets in burst from the nic to the rx_buffer.
   */
  void HandleRx ();

  /**
   * Check the status of the link.
   * \return Status of the link - up/down as true/false.
   */
  bool IsLinkUp (void) const;

  /**
   * Free the given packet buffer.
   * \param buf the pointer to the buffer to be freed
   */
  virtual void FreeBuffer (uint8_t* buf);

  /**
   * Allocate packet buffer.
   * \param len the length of the buffer
   * \return A pointer to the newly created buffer.
   */
  virtual uint8_t* AllocateBuffer (size_t len);

protected:
  /**
   * Write packet data to device.
   * \param buffer The data.
   * \param length The data length.
   * \return The size of data written.
   */
  ssize_t Write (uint8_t *buffer, size_t length);

  /**
   * The port number of the device to be used.
   */
  uint16_t m_portId;

  /**
   * The device name;
   */
  std::string m_deviceName;

private:
  void DoFinishStoppingDevice (void);
  /**
   * Condition variable for Dpdk to stop
   */
  static volatile bool m_forceQuit;

  /**
   * Packet memory pool
   */
  struct rte_mempool *m_mempool;

  /**
   * Buffer to handle burst transmission
   */
  struct rte_eth_dev_tx_buffer *m_txBuffer;

  /**
   * Buffer to handle burst reception
   */
  struct rte_eth_dev_tx_buffer *m_rxBuffer;

  /**
   * Event for stale packet transmission
   */
  EventId m_txEvent;

  /**
   * The time to wait before transmitting burst from Tx buffer
   */
  Time m_txTimeout;

  /**
   * Size of Rx burst
   */
  uint32_t m_maxRxPktBurst;

  /**
   * Size of Tx burst
   */
  uint32_t m_maxTxPktBurst;

  /**
   * Mempool cache size
   */
  uint32_t m_mempoolCacheSize;

  /**
   * Number of Rx descriptors.
   */
  uint16_t m_nbRxDesc;

  /**
   * Number of Tx descriptors.
   */
  uint16_t m_nbTxDesc;
};

} //

#endif /* DPDK_NET_DEVICE_H */
