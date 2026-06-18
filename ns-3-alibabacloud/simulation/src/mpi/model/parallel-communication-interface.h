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
 *
 */

/**
 * \file
 * \ingroup mpi
 * Declaration of class ns3::ParallelCommunicationInterface.
 */

#ifndef NS3_PARALLEL_COMMUNICATION_INTERFACE_H
#define NS3_PARALLEL_COMMUNICATION_INTERFACE_H

#include <stdint.h>
#include <list>

#include <ns3/object.h>
#include <ns3/nstime.h>
#include <ns3/buffer.h>
#include <ns3/packet.h>

#include "mpi.h"

namespace ns3 {

/**
 * \ingroup mpi
 *
 * \brief Pure virtual base class for the interface between ns-3 and
 * the parallel communication layer being used.
 *
 * This class is implemented for each of the parallel versions of
 * SimulatorImpl to manage communication between process (ranks).
 *
 * This interface is called through the singleton MpiInterface class.
 * MpiInterface has the same API as ParallelCommunicationInterface but
 * being a singleton uses static methods to delegate to methods
 * defined in classes that implement the
 * ParallelCommunicationInterface.  For example, SendPacket is likely
 * to be specialized for a specific parallel SimulatorImpl.
 */
class ParallelCommunicationInterface
{
public:
  /**
    * Destructor
    */
  virtual ~ParallelCommunicationInterface() {}
  /**
   * \copydoc MpiInterface::Destroy
   */
  virtual void Destroy () = 0;
  /**
   * \copydoc MpiInterface::GetSystemId
   */
  virtual uint32_t GetSystemId () = 0;
  /**
   * \copydoc MpiInterface::GetSize
   */
  virtual uint32_t GetSize () = 0;
  /**
   * \copydoc MpiInterface::IsEnabled
   */
  virtual bool IsEnabled () = 0;
  /**
   * \copydoc MpiInterface::Enable(int* pargc,char*** pargv)
   */
  virtual void Enable (int* pargc, char*** pargv) = 0;
  /**
   * \copydoc MpiInterface::Enable(MPI_Comm communicator)
   */
  virtual void Enable (MPI_Comm communicator) = 0;
  /**
   * \copydoc MpiInterface::Disable
   */
  virtual void Disable () = 0;
  /**
   * \copydoc MpiInterface::SendPacket
   */
  virtual void SendPacket (Ptr<Packet> p, const Time &rxTime, uint32_t node, uint32_t dev) = 0;
  /**
   * \copydoc MpiInterface::GetCommunicator
   */
  virtual MPI_Comm GetCommunicator () = 0;
private:
};

} // namespace ns3

#endif /* NS3_PARALLEL_COMMUNICATION_INTERFACE_H */
