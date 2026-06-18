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

#include "dpdk-net-device.h"

#include "ns3/log.h"
#include "ns3/net-device-queue-interface.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <unistd.h>

#include <mutex>
#include <poll.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_common.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_cycles.h>
#include <rte_port.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DpdkNetDevice");

NS_OBJECT_ENSURE_REGISTERED (DpdkNetDevice);

volatile bool DpdkNetDevice::m_forceQuit = false;

TypeId
DpdkNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DpdkNetDevice")
    .SetParent<FdNetDevice> ()
    .SetGroupName ("FdNetDevice")
    .AddConstructor<DpdkNetDevice> ()
    .AddAttribute ("TxTimeout",
                   "The time to wait before transmitting burst from Tx buffer.",
                   TimeValue (MicroSeconds (2000)),
                   MakeTimeAccessor (&DpdkNetDevice::m_txTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("MaxRxBurst",
                   "Size of Rx Burst.",
                   UintegerValue (64),
                   MakeUintegerAccessor (&DpdkNetDevice::m_maxRxPktBurst),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MaxTxBurst",
                   "Size of Tx Burst.",
                   UintegerValue (64),
                   MakeUintegerAccessor (&DpdkNetDevice::m_maxTxPktBurst),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MempoolCacheSize",
                   "Size of mempool cache.",
                   UintegerValue (256),
                   MakeUintegerAccessor (&DpdkNetDevice::m_mempoolCacheSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("NbRxDesc",
                   "Number of Rx descriptors.",
                   UintegerValue (1024),
                   MakeUintegerAccessor (&DpdkNetDevice::m_nbRxDesc),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("NbTxDesc",
                   "Number of Tx descriptors.",
                   UintegerValue (1024),
                   MakeUintegerAccessor (&DpdkNetDevice::m_nbTxDesc),
                   MakeUintegerChecker<uint16_t> ())
  ;
  return tid;
}

DpdkNetDevice::DpdkNetDevice ()
  : m_mempool (NULL)
{
  NS_LOG_FUNCTION (this);
}

DpdkNetDevice::~DpdkNetDevice ()
{
  NS_LOG_FUNCTION (this);
  Simulator::Cancel (m_txEvent);
  m_forceQuit = true;

  rte_eal_wait_lcore (1);
  rte_eth_dev_stop (m_portId);
  rte_eth_dev_close (m_portId);
}

void
DpdkNetDevice::SetDeviceName (std::string deviceName)
{
  NS_LOG_FUNCTION (this);

  m_deviceName = deviceName;
}

void
DpdkNetDevice::CheckAllPortsLinkStatus (void)
{
  NS_LOG_FUNCTION (this);

  #define CHECK_INTERVAL 100 /* 100ms */
  #define MAX_CHECK_TIME 90 /* 9s (90 * 100ms) in total */
  uint8_t count, allPortsUp, printFlag = 0;
  struct rte_eth_link link;

  for (count = 0; count <= MAX_CHECK_TIME; count++)
    {

      allPortsUp = 1;

      if (m_forceQuit)
        {
          return;
        }
      if ((1 << m_portId) == 0)
        {
          continue;
        }
      memset (&link, 0, sizeof(link));
      rte_eth_link_get (m_portId, &link);
      /* print link status if flag set */
      if (printFlag == 1)
        {
          if (link.link_status)
            {
              continue;
            }
          else
            {
              printf ("Port %d Link Down\n", m_portId);
            }
          continue;
        }
      /* clear allPortsUp flag if any link down */
      if (link.link_status == ETH_LINK_DOWN)
        {
          allPortsUp = 0;
          break;
        }

      /* after finally printing all link status, get out */
      if (printFlag == 1)
        {
          break;
        }

      if (allPortsUp == 0)
        {
          fflush (stdout);
          rte_delay_ms (CHECK_INTERVAL);
        }

      /* set the printFlag if all ports up or timeout */
      if (allPortsUp == 1 || count == (MAX_CHECK_TIME - 1))
        {
          printFlag = 1;
        }
    }
}

void
DpdkNetDevice::SignalHandler (int signum)
{
  if (signum == SIGINT || signum == SIGTERM)
    {
      printf ("\n\nSignal %d received, preparing to exit...\n",
              signum);
      m_forceQuit = true;
    }
}

void
DpdkNetDevice::HandleTx ()
{
  int queueId = 0;
  rte_eth_tx_buffer_flush (m_portId, queueId, m_txBuffer);
}

void
DpdkNetDevice::HandleRx ()
{
  int queueId = 0;
  m_rxBuffer->length = rte_eth_rx_burst (m_portId,
                                         queueId,
                                         m_rxBuffer->pkts,
                                         m_maxRxPktBurst);

  for (uint16_t i = 0; i < m_rxBuffer->length; i++)
    {
      struct rte_mbuf *pkt = NULL;
      pkt = m_rxBuffer->pkts[i];

      if (!pkt)
        {
          continue;
        }

      uint8_t * buf = rte_pktmbuf_mtod (pkt, uint8_t *);
      size_t length = pkt->data_len;
      FdNetDevice::ReceiveCallback (buf,length);
    }

  m_rxBuffer->length = 0;
}

int
DpdkNetDevice::LaunchCore (void *arg)
{
  DpdkNetDevice *dpdkNetDevice = (DpdkNetDevice*) arg;
  unsigned lcoreId;
  lcoreId = rte_lcore_id ();
  if (lcoreId != 1)
    {
      return 0;
    }

  while (!m_forceQuit)
    {
      dpdkNetDevice->HandleRx ();
    }

  return 0;
}

bool
DpdkNetDevice::IsLinkUp (void) const
{
  // Refer https://mails.dpdk.org/archives/users/2018-December/003822.html
  return true;
}

void
DpdkNetDevice::InitDpdk (int argc, char** argv, std::string dpdkDriver)
{
  NS_LOG_FUNCTION (this << argc << argv);

  NS_LOG_INFO ("Binding device to DPDK");
  std::string command;
  command.append ("dpdk-devbind.py --force ");
  command.append ("--bind=");
  command.append (dpdkDriver.c_str ());
  command.append (" ");
  command.append (m_deviceName.c_str ());
  printf ("Executing: %s\n", command.c_str ());
  if (system (command.c_str ()))
    {
      rte_exit (EXIT_FAILURE, "Execution failed - bye\n");
    }

  // wait for the device to bind to Dpdk
  sleep (5);  /* 5 seconds */

  NS_LOG_INFO ("Initialize DPDK EAL");
  int ret = rte_eal_init (argc, argv);
  if (ret < 0)
    {
      rte_exit (EXIT_FAILURE, "Invalid EAL arguments\n");
    }

  m_forceQuit = false;
  signal (SIGINT, SignalHandler);
  signal (SIGTERM, SignalHandler);

  unsigned nbPorts = rte_eth_dev_count_avail ();
  if (nbPorts == 0)
    {
      rte_exit (EXIT_FAILURE, "No Ethernet ports - bye\n");
    }

  NS_LOG_INFO ("Get port id of the device");
  if (rte_eth_dev_get_port_by_name (m_deviceName.c_str (), &m_portId) != 0)
    {
      rte_exit (EXIT_FAILURE, "Cannot get port id - bye\n");
    }

  // Set number of logical cores to 2
  unsigned int nbLcores = 2;

  unsigned int nbMbufs = RTE_MAX (nbPorts * (m_nbRxDesc + m_nbTxDesc + m_maxRxPktBurst +
                                             m_maxTxPktBurst +
                                             nbLcores * m_mempoolCacheSize),
                                  8192U);

  NS_LOG_INFO ("Create the mbuf pool");
  m_mempool = rte_pktmbuf_pool_create ("mbuf_pool", nbMbufs,
                                       m_mempoolCacheSize, 0,
                                       RTE_MBUF_DEFAULT_BUF_SIZE,
                                       rte_socket_id ());

  if (m_mempool == NULL)
    {
      rte_exit (EXIT_FAILURE, "Cannot init mbuf pool\n");
    }

  NS_LOG_INFO ("Initialize port");
  static struct rte_eth_conf portConf = {};
  portConf.rxmode = {};
  portConf.rxmode.split_hdr_size = 0;
  portConf.txmode = {};
  portConf.txmode.mq_mode = ETH_MQ_TX_NONE;

  struct rte_eth_rxconf reqConf;
  struct rte_eth_txconf txqConf;
  struct rte_eth_conf localPortConf = portConf;
  struct rte_eth_dev_info devInfo;

  fflush (stdout);
  rte_eth_dev_info_get (m_portId, &devInfo);
  if (devInfo.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
    {
      localPortConf.txmode.offloads |=
        DEV_TX_OFFLOAD_MBUF_FAST_FREE;
    }
  ret = rte_eth_dev_configure (m_portId, 1, 1, &localPortConf);
  if (ret < 0)
    {
      rte_exit (EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n",
                ret, m_portId);
    }

  ret = rte_eth_dev_adjust_nb_rx_tx_desc (m_portId, &m_nbRxDesc, &m_nbTxDesc);
  if (ret < 0)
    {
      rte_exit (EXIT_FAILURE,
                "Cannot adjust number of descriptors: err=%d, port=%u\n",
                ret, m_portId);
    }

  NS_LOG_INFO ("Initialize one Rx queue");
  fflush (stdout);
  reqConf = devInfo.default_rxconf;
  reqConf.offloads = localPortConf.rxmode.offloads;
  ret = rte_eth_rx_queue_setup (m_portId, 0, m_nbRxDesc,
                                rte_eth_dev_socket_id (m_portId),
                                &reqConf,
                                m_mempool);
  if (ret < 0)
    {
      rte_exit (EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n",
                ret, m_portId);
    }

  NS_LOG_INFO ("Initialize one Tx queue per port");
  fflush (stdout);
  txqConf = devInfo.default_txconf;
  txqConf.offloads = localPortConf.txmode.offloads;
  ret = rte_eth_tx_queue_setup (m_portId, 0, m_nbTxDesc,
                                rte_eth_dev_socket_id (m_portId),
                                &txqConf);
  if (ret < 0)
    {
      rte_exit (EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n",
                ret, m_portId);
    }

  NS_LOG_INFO ("Initialize Tx buffers");
  m_txBuffer = (rte_eth_dev_tx_buffer*)
    rte_zmalloc_socket ("tx_buffer",
                        RTE_ETH_TX_BUFFER_SIZE (m_maxTxPktBurst), 0,
                        rte_eth_dev_socket_id (m_portId));
  NS_LOG_INFO ("Initialize Rx buffers");
  m_rxBuffer = (rte_eth_dev_tx_buffer*)
    rte_zmalloc_socket ("rx_buffer",
                        RTE_ETH_TX_BUFFER_SIZE (m_maxRxPktBurst), 0,
                        rte_eth_dev_socket_id (m_portId));
  if (m_txBuffer == NULL || m_rxBuffer == NULL)
    {
      rte_exit (EXIT_FAILURE, "Cannot allocate buffer for rx/tx on port %u\n",
                m_portId);
    }

  rte_eth_tx_buffer_init (m_txBuffer, m_maxTxPktBurst);
  rte_eth_tx_buffer_init (m_rxBuffer, m_maxRxPktBurst);

  NS_LOG_INFO ("Start the device");
  ret = rte_eth_dev_start (m_portId);
  if (ret < 0)
    {
      rte_exit (EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n",
                ret, m_portId);
    }

  rte_eth_promiscuous_enable (m_portId);

  CheckAllPortsLinkStatus ();

  NS_LOG_INFO ("Launching core threads");
  rte_eal_mp_remote_launch (LaunchCore, this, CALL_MASTER);
}

uint8_t*
DpdkNetDevice::AllocateBuffer (size_t len)
{
  struct rte_mbuf *pkt = rte_pktmbuf_alloc (m_mempool);
  if (!pkt)
    {
      return NULL;
    }
  uint8_t *buf = rte_pktmbuf_mtod (pkt, uint8_t *);
  return buf;
}

void
DpdkNetDevice::FreeBuffer (uint8_t* buf)
{
  struct rte_mbuf *pkt;

  if (!buf)
    {
      return;
    }
  pkt = (struct rte_mbuf *)
    RTE_PTR_SUB ( buf,
                  sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM);

  rte_pktmbuf_free (pkt);
}

ssize_t
DpdkNetDevice::Write (uint8_t *buffer, size_t length)
{
  struct rte_mbuf ** pkt = new struct rte_mbuf*[1];
  int queueId = 0;

  if (buffer == NULL || m_txBuffer->length == m_maxTxPktBurst)
    {
      NS_LOG_ERROR ("Error allocating mbuf" << buffer);
      return -1;
    }

  pkt[0] = (struct rte_mbuf *)
    RTE_PTR_SUB ( buffer,
                  sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM);

  pkt[0]->pkt_len = length;
  pkt[0]->data_len = length;
  rte_eth_tx_buffer (m_portId, queueId, m_txBuffer, pkt[0]);

  if (m_txBuffer->length == 1)
    {
      // If this is a first packet in buffer, schedule a tx.
      Simulator::Cancel (m_txEvent);
      m_txEvent = Simulator::Schedule (m_txTimeout, &DpdkNetDevice::HandleTx, this);
    }

  return length;
}

void
DpdkNetDevice::DoFinishStoppingDevice (void)
{
  std::unique_lock lock {m_pendingReadMutex};

  while (!m_pendingQueue.empty ())
    {
      std::pair<uint8_t *, ssize_t> next = m_pendingQueue.front ();
      m_pendingQueue.pop ();

      FreeBuffer (next.first);
    }
}

} // namespace ns3
