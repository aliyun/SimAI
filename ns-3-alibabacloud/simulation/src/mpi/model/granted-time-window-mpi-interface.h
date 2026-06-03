/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: George Riley <riley@ece.gatech.edu>
 */

/**
 * \file
 * \ingroup mpi
 * Declaration of classes ns3::SentBuffer and ns3::GrantedTimeWindowMpiInterface.
 */

// This object contains static methods that provide an easy interface
// to the necessary MPI information.

#ifndef NS3_GRANTED_TIME_WINDOW_MPI_INTERFACE_H
#define NS3_GRANTED_TIME_WINDOW_MPI_INTERFACE_H

#include <stdint.h>
#include <atomic>
#include <list>

#include "ns3/nstime.h"
#include "ns3/buffer.h"

#include "parallel-communication-interface.h"

#include "mpi.h"

namespace ns3 {

/**
 * maximum MPI message size for easy
 * buffer creation
 */
const uint32_t MAX_MPI_MSG_SIZE = 2000;

/**
 * \ingroup mpi
 *
 * \brief Tracks non-blocking sends
 *
 * This class is used to keep track of the asynchronous non-blocking
 * sends that have been posted.
 */
class SentBuffer
{
public:
  SentBuffer ();
  ~SentBuffer ();

  /**
   * \return pointer to sent buffer
   */
  uint8_t* GetBuffer ();
  /**
   * \param buffer pointer to sent buffer
   */
  void SetBuffer (uint8_t* buffer);
  /**
   * \return MPI request
   */
  MPI_Request* GetRequest ();

private:
  uint8_t* m_buffer;      /**< The buffer. */
  MPI_Request m_request;  /**< The MPI request handle. */
};

class Packet;
class DistributedSimulatorImpl;
class HybridSimulatorImpl;

/**
 * \ingroup mpi
 *
 * \brief Interface between ns-3 and MPI
 *
 * Implements the interface used by the singleton parallel controller
 * to interface between NS3 and the communications layer being
 * used for inter-task packet transfers.
 */
class GrantedTimeWindowMpiInterface : public ParallelCommunicationInterface, Object
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // Inherited
  virtual void Destroy ();
  virtual uint32_t GetSystemId ();
  virtual uint32_t GetSize ();
  virtual bool IsEnabled ();
  virtual void Enable (int* pargc, char*** pargv);
  virtual void Enable (MPI_Comm communicator);
  virtual void Disable();
  virtual void SendPacket (Ptr<Packet> p, const Time &rxTime, uint32_t node, uint32_t dev);
  virtual MPI_Comm GetCommunicator();

  /*
   * The granted time window implementation is a collaboration of several
   * classes.  Methods that should be invoked only by the
   * collaborators are private to restrict use.
   * It is not intended for state to be shared.
   */
  friend ns3::DistributedSimulatorImpl;
  friend ns3::HybridSimulatorImpl;

  /**
   * Check for received messages complete
   */
  static void ReceiveMessages ();
  /**
   * Check for completed sends
   */
  static void TestSendComplete ();
  /**
   * \return received count in packets
   */
  static uint32_t GetRxCount ();
  /**
   * \return transmitted count in packets
   */
  static uint32_t GetTxCount ();

  /** System ID (rank) for this task. */
  static uint32_t g_sid;
  /** Size of the MPI COM_WORLD group. */
  static uint32_t g_size;

  /** Total packets received. */
  static uint32_t g_rxCount;

  /** Total packets sent. */
  static uint32_t g_txCount;

  /** Has this interface been enabled. */
  static bool     g_enabled;

  /**
   * Has MPI Init been called by this interface.
   * Alternatively user supplies a communicator.
   */
  static bool     g_mpiInitCalled;

  /** Pending non-blocking receives. */
  static MPI_Request* g_requests;

  /** Data buffers for non-blocking reads. */
  static char**   g_pRxBuffers;

  /** List of pending non-blocking sends. */
  static std::list<SentBuffer> g_pendingTx;

  /** MPI communicator being used for ns-3 tasks. */
  static MPI_Comm g_communicator;

  /** Did ns-3 create the communicator?  Have to free it. */
  static bool g_freeCommunicator;

#ifdef NS3_MTP
  static std::atomic<bool> g_sending;
#endif
};

} // namespace ns3

#endif /* NS3_GRANTED_TIME_WINDOW_MPI_INTERFACE_H */
