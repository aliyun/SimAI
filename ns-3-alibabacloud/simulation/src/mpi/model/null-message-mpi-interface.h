/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright 2013. Lawrence Livermore National Security, LLC.
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
 * Author: Steven Smith <smith84@llnl.gov>
 */

/**
 * \file
 * \ingroup mpi
 * Declaration of classes ns3::NullMessageSentBuffer and ns3::NullMessageMpiInterface.
 */

#ifndef NS3_NULLMESSAGE_MPI_INTERFACE_H
#define NS3_NULLMESSAGE_MPI_INTERFACE_H

#include "parallel-communication-interface.h"

#include <ns3/nstime.h>
#include <ns3/buffer.h>

#include "mpi.h"
#include <list>

namespace ns3 {

class NullMessageSimulatorImpl;
class NullMessageSentBuffer;
class RemoteChannelBundle;
class Packet;

/**
 * \ingroup mpi
 *
 * \brief Interface between ns-3 and MPI for the Null Message
 * distributed simulation implementation.
 */
class NullMessageMpiInterface : public ParallelCommunicationInterface, Object
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  NullMessageMpiInterface ();
  ~NullMessageMpiInterface ();

  // Inherited
  virtual void Destroy ();
  virtual uint32_t GetSystemId ();
  virtual uint32_t GetSize ();
  virtual bool IsEnabled ();
  virtual void Enable (int* pargc, char*** pargv);
  virtual void Enable (MPI_Comm communicator);
  virtual void Disable ();
  virtual void SendPacket (Ptr<Packet> p, const Time &rxTime, uint32_t node, uint32_t dev);
  virtual MPI_Comm GetCommunicator();

private:

  /**
   * The null message implementation is a collaboration of several
   * classes.  Methods that should be invoked only by the
   * collaborators are private to restrict use.
   * It is not intended for state to be shared.
   */
  friend ns3::RemoteChannelBundle;
  friend ns3::NullMessageSimulatorImpl;
  
  /**
   * \brief Send a Null Message to across the specified bundle.  
   *
   * Null Messages are sent when a packet has not been sent across
   * this bundle in order to allow time advancement on the remote
   * MPI task.
   *
   * \param [in] guaranteeUpdate Lower bound time on the next
   * possible event from this MPI task to the remote MPI task across
   * the bundle.  Remote task may execute events up to this time.
   *
   * \param [in] bundle The bundle of links between two ranks.
   *
   * \internal The Null Message MPI buffer format uses the same packet
   * metadata format as sending a normal packet with the time,
   * destination node, and destination device set to zero.  Using the
   * same packet metadata simplifies receive logic.
   */
  static void SendNullMessage (const Time& guaranteeUpdate, Ptr<RemoteChannelBundle> bundle);
  /**
   * Non-blocking check for received messages complete.  Will
   * receive all messages that are queued up locally.
   */
  static void ReceiveMessagesNonBlocking ();
  /**
   * Blocking message receive.  Will block until at least one message
   * has been received.
   */
  static void ReceiveMessagesBlocking ();
  /**
   * Check for completed sends
   */
  static void TestSendComplete ();

  /**
   * \brief Initialize send and receive buffers.
   *
   * This method should be called after all links have been added to the RemoteChannelBundle
   * manager to setup any required send and receive buffers.
   */
  static void InitializeSendReceiveBuffers (void);

  /**
   * Check for received messages complete.  Will block until message
   * has been received if blocking flag is true.  When blocking will
   * return after the first message is received.   Non-blocking mode will
   * only check for received messages complete, and return 
   * all messages that are queued up locally.
   *
   * \param [in] blocking Whether this call should block.
   */
  static void ReceiveMessages (bool blocking = false);

  /** System ID (rank) for this task. */
  static uint32_t g_sid;

  /** Size of the MPI COM_WORLD group. */
  static uint32_t g_size;

  /** Number of neighbor tasks, tasks that this task shares a link with. */
  static uint32_t g_numNeighbors;

  /** Has this interface been enabled. */
  static bool     g_enabled;

  /**
   * Has MPI Init been called by this interface.
   * Alternatively user supplies a communicator.
   */
  static bool     g_mpiInitCalled;

  /** Pending non-blocking receives. */
  static MPI_Request* g_requests;

  /** Data buffers for non-blocking receives. */
  static char**   g_pRxBuffers;

  /** List of pending non-blocking sends. */
  static std::list<NullMessageSentBuffer> g_pendingTx;

  /** MPI communicator being used for ns-3 tasks. */
  static MPI_Comm g_communicator;

  /** Did we create the communicator?  Have to free it. */
  static bool g_freeCommunicator;
};

} // namespace ns3

#endif /* NS3_NULL_MESSAGE_MPI_INTERFACE_H */
