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
 * Implementation of classes ns3::SentBuffer and ns3::GrantedTimeWindowMpiInterface.
 */

// This object contains static methods that provide an easy interface
// to the necessary MPI information.

#include <iostream>
#include <iomanip>
#include <list>

#include "granted-time-window-mpi-interface.h"
#include "mpi-receiver.h"
#include "mpi-interface.h"

#include "ns3/node.h"
#include "ns3/node-list.h"
#include "ns3/net-device.h"
#include "ns3/simulator.h"
#include "ns3/simulator-impl.h"
#include "ns3/nstime.h"
#include "ns3/log.h"
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif

#include <mpi.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GrantedTimeWindowMpiInterface");

NS_OBJECT_ENSURE_REGISTERED (GrantedTimeWindowMpiInterface);

SentBuffer::SentBuffer ()
{
  m_buffer = 0;
  m_request = 0;
}

SentBuffer::~SentBuffer ()
{
  delete [] m_buffer;
}

uint8_t*
SentBuffer::GetBuffer ()
{
  return m_buffer;
}

void
SentBuffer::SetBuffer (uint8_t* buffer)
{
  m_buffer = buffer;
}

MPI_Request*
SentBuffer::GetRequest ()
{
  return &m_request;
}

uint32_t              GrantedTimeWindowMpiInterface::g_sid = 0;
uint32_t              GrantedTimeWindowMpiInterface::g_size = 1;
bool                  GrantedTimeWindowMpiInterface::g_enabled = false;
bool                  GrantedTimeWindowMpiInterface::g_mpiInitCalled = false;
uint32_t              GrantedTimeWindowMpiInterface::g_rxCount = 0;
uint32_t              GrantedTimeWindowMpiInterface::g_txCount = 0;
std::list<SentBuffer> GrantedTimeWindowMpiInterface::g_pendingTx;

MPI_Request* GrantedTimeWindowMpiInterface::g_requests;
char**       GrantedTimeWindowMpiInterface::g_pRxBuffers;
MPI_Comm     GrantedTimeWindowMpiInterface::g_communicator = MPI_COMM_WORLD;
bool         GrantedTimeWindowMpiInterface::g_freeCommunicator = false;;

#ifdef NS3_MTP
std::atomic<bool>     GrantedTimeWindowMpiInterface::g_sending (false);
#endif

TypeId 
GrantedTimeWindowMpiInterface::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GrantedTimeWindowMpiInterface")
    .SetParent<Object> ()
    .SetGroupName ("Mpi")
  ;
  return tid;
}

void
GrantedTimeWindowMpiInterface::Destroy ()
{
  NS_LOG_FUNCTION (this);

  for (uint32_t i = 0; i < GetSize (); ++i)
    {
      delete [] g_pRxBuffers[i];
    }
  delete [] g_pRxBuffers;
  delete [] g_requests;

  g_pendingTx.clear ();
}

uint32_t
GrantedTimeWindowMpiInterface::GetRxCount ()
{
  NS_ASSERT (g_enabled);
  return g_rxCount;
}

uint32_t
GrantedTimeWindowMpiInterface::GetTxCount ()
{
  NS_ASSERT (g_enabled);
  return g_txCount;
}

uint32_t
GrantedTimeWindowMpiInterface::GetSystemId ()
{
  NS_ASSERT (g_enabled);
  return g_sid;
}

uint32_t
GrantedTimeWindowMpiInterface::GetSize ()
{
  NS_ASSERT (g_enabled);
  return g_size;
}

bool
GrantedTimeWindowMpiInterface::IsEnabled ()
{
  return g_enabled;
}

MPI_Comm 
GrantedTimeWindowMpiInterface::GetCommunicator() 
{
  NS_ASSERT (g_enabled);
  return g_communicator;
}

void
GrantedTimeWindowMpiInterface::Enable (int* pargc, char*** pargv)
{
  NS_LOG_FUNCTION (this << pargc << pargv);

  NS_ASSERT (g_enabled == false);

  // Initialize the MPI interface
  MPI_Init (pargc, pargv);
  Enable (MPI_COMM_WORLD);
  g_mpiInitCalled = true;
  g_enabled = true;
}

void
GrantedTimeWindowMpiInterface::Enable (MPI_Comm communicator)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT (g_enabled == false);

  // Standard MPI practice is to duplicate the communicator for
  // library to use.  Library communicates in isolated communication
  // context.
  MPI_Comm_dup (communicator, &g_communicator);
  g_freeCommunicator = true;
  
  MPI_Barrier (g_communicator);
  
  int mpiSystemId;
  int mpiSize;
  MPI_Comm_rank (g_communicator, &mpiSystemId);
  MPI_Comm_size (g_communicator, &mpiSize);
  g_sid = mpiSystemId;
  g_size = mpiSize;
  
  g_enabled = true;
  // Post a non-blocking receive for all peers
  g_pRxBuffers = new char*[g_size];
  g_requests = new MPI_Request[g_size];
  for (uint32_t i = 0; i < GetSize (); ++i)
    {
      g_pRxBuffers[i] = new char[MAX_MPI_MSG_SIZE];
      MPI_Irecv (g_pRxBuffers[i], MAX_MPI_MSG_SIZE, MPI_CHAR, MPI_ANY_SOURCE, 0,
                 g_communicator, &g_requests[i]);
    }
}

void
GrantedTimeWindowMpiInterface::SendPacket (Ptr<Packet> p, const Time& rxTime, uint32_t node, uint32_t dev)
{
  NS_LOG_FUNCTION (this << p << rxTime.GetTimeStep () << node << dev);

#ifdef NS3_MTP
  while (g_sending.exchange (true, std::memory_order_acquire))
    ;
#endif

  SentBuffer sendBuf;
  g_pendingTx.push_back (sendBuf);
  std::list<SentBuffer>::reverse_iterator i = g_pendingTx.rbegin (); // Points to the last element

  uint32_t serializedSize = p->GetSerializedSize ();
  uint8_t* buffer =  new uint8_t[serializedSize + 16];
  i->SetBuffer (buffer);
  // Add the time, dest node and dest device
  uint64_t t = rxTime.GetInteger ();
  uint64_t* pTime = reinterpret_cast <uint64_t *> (buffer);
  *pTime++ = t;
  uint32_t* pData = reinterpret_cast<uint32_t *> (pTime);
  *pData++ = node;
  *pData++ = dev;
  // Serialize the packet
  p->Serialize (reinterpret_cast<uint8_t *> (pData), serializedSize);

  // Find the system id for the destination node
  Ptr<Node> destNode = NodeList::GetNode (node);
#ifdef NS3_MTP
  uint32_t nodeSysId = destNode->GetSystemId () & 0xFFFF;
#else
  uint32_t nodeSysId = destNode->GetSystemId ();
#endif

  MPI_Isend (reinterpret_cast<void *> (i->GetBuffer ()), serializedSize + 16, MPI_CHAR, nodeSysId,
             0, g_communicator, (i->GetRequest ()));
  g_txCount++;

#ifdef NS3_MTP
  g_sending.store(false, std::memory_order_release);
#endif
}

void
GrantedTimeWindowMpiInterface::ReceiveMessages ()
{ 
  NS_LOG_FUNCTION_NOARGS ();

  // Poll the non-block reads to see if data arrived
  while (true)
    {
      int flag = 0;
      int index = 0;
      MPI_Status status;

      MPI_Testany (MpiInterface::GetSize (), g_requests, &index, &flag, &status);
      if (!flag)
        {
          break;        // No more messages
        }
      int count;
      MPI_Get_count (&status, MPI_CHAR, &count);
      g_rxCount++; // Count this receive

      // Get the meta data first
      uint64_t* pTime = reinterpret_cast<uint64_t *> (g_pRxBuffers[index]);
      uint64_t time = *pTime++;
      uint32_t* pData = reinterpret_cast<uint32_t *> (pTime);
      uint32_t node = *pData++;
      uint32_t dev  = *pData++;

      Time rxTime (time);

      count -= sizeof (time) + sizeof (node) + sizeof (dev);

      Ptr<Packet> p = Create<Packet> (reinterpret_cast<uint8_t *> (pData), count, true);

      // Find the correct node/device to schedule receive event
      Ptr<Node> pNode = NodeList::GetNode (node);
      Ptr<MpiReceiver> pMpiRec = 0;
      uint32_t nDevices = pNode->GetNDevices ();
      for (uint32_t i = 0; i < nDevices; ++i)
        {
          Ptr<NetDevice> pThisDev = pNode->GetDevice (i);
          if (pThisDev->GetIfIndex () == dev)
            {
              pMpiRec = pThisDev->GetObject<MpiReceiver> ();
              break;
            }
        }

      NS_ASSERT (pNode && pMpiRec);

      // Schedule the rx event
#ifdef NS3_MTP
      MtpInterface::GetSystem (pNode->GetSystemId () >> 16)
          ->ScheduleAt (pNode->GetId (), rxTime, MakeEvent (&MpiReceiver::Receive, pMpiRec, p));
#else
      Simulator::ScheduleWithContext (pNode->GetId (), rxTime - Simulator::Now (),
                                      &MpiReceiver::Receive, pMpiRec, p);
#endif

      // Re-queue the next read
      MPI_Irecv (g_pRxBuffers[index], MAX_MPI_MSG_SIZE, MPI_CHAR, MPI_ANY_SOURCE, 0,
                 g_communicator, &g_requests[index]);
    }
}

void
GrantedTimeWindowMpiInterface::TestSendComplete ()
{
  NS_LOG_FUNCTION_NOARGS ();

  std::list<SentBuffer>::iterator i = g_pendingTx.begin ();
  while (i != g_pendingTx.end ())
    {
      MPI_Status status;
      int flag = 0;
      MPI_Test (i->GetRequest (), &flag, &status);
      std::list<SentBuffer>::iterator current = i; // Save current for erasing
      i++;                                    // Advance to next
      if (flag)
        { // This message is complete
          g_pendingTx.erase (current);
        }
    }
}

void
GrantedTimeWindowMpiInterface::Disable ()
{
  NS_LOG_FUNCTION_NOARGS ();

  if (g_freeCommunicator)
    {
      MPI_Comm_free (&g_communicator);
      g_freeCommunicator = false;
    }

  // ns-3 should MPI finalize only if ns-3 was used to initialize
  if (g_mpiInitCalled)
    {
      int flag = 0;
      MPI_Initialized (&flag);
      if (flag)
        {
          MPI_Finalize ();
        }
      else
        {
          NS_FATAL_ERROR ("Cannot disable MPI environment without Initializing it first");
        }
      g_mpiInitCalled = false;
    }

  g_enabled = false;
}


} // namespace ns3
