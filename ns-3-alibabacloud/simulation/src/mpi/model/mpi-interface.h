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
 * Declaration of class ns3::MpiInterface.
 */

#ifndef NS3_MPI_INTERFACE_H
#define NS3_MPI_INTERFACE_H

#include <ns3/nstime.h>
#include <ns3/packet.h>

#include "mpi.h"

namespace ns3 {
/**
 * \defgroup mpi MPI Distributed Simulation
 */

/**
 * \ingroup mpi
 * \ingroup tests
 * \defgroup mpi-tests MPI Distributed Simulation tests
 */

class ParallelCommunicationInterface;

/**
 * \ingroup mpi
 *
 * \brief Singleton used to interface to the communications infrastructure
 * when running NS3 in parallel.  
 *
 * Delegates the implementation to the specific parallel
 * infrastructure being used.  Implementation is defined in the
 * ParallelCommunicationInterface virtual base class; this API mirrors
 * that interface.  This singleton is responsible for instantiating an
 * instance of the communication interface based on
 * SimulatorImplementationType attribute in ns3::GlobalValues.  The
 * attribute must be set before Enable is invoked.
 */
class MpiInterface
{
public:
  /**
   * \brief Deletes storage used by the parallel environment.
   */
  static void Destroy ();
  /**
   * \brief Get the id number of this rank.
   *
   * When running a sequential simulation this will return a systemID of 0.
   *
   * \return system identification
   */
  static uint32_t GetSystemId ();
  /**
   * \brief Get the number of ranks used by ns-3.
   *
   * Returns the size (number of MPI ranks) of the communicator used by
   * ns-3.  When running a sequential simulation this will return a
   * size of 1.
   *
   * \return number of parallel tasks
   */
  static uint32_t GetSize ();
  /**
   * \brief Returns enabled state of parallel environment.
   *
   * \return true if parallel communication is enabled
   */
  static bool IsEnabled ();
  /**
   * \brief Setup the parallel communication interface.
   *
   * There are two ways to setup the communications interface.  This
   * Enable method is the easiest method and should be used in most
   * situations.
   *
   * Disable() must be invoked at end of an ns-3 application to 
   * properly cleanup the parallel communication interface.
   *
   * This method will call MPI_Init and configure ns-3 to use the
   * MPI_COMM_WORLD communicator.
   *
   * For more complex situations, such as embedding ns-3 with other
   * MPI simulators or libraries, the Enable(MPI_Comm communcicator)
   * may be used if MPI is initialized externally or if ns-3 needs to
   * be run unique communicator.  For example if there are two
   * parallel simulators and the goal is to run each simulator on a
   * different set of ranks.
   *
   * \note The `SimulatorImplementationType attribute in
   * ns3::GlobalValues must be set before calling Enable()
   *
   * \param pargc number of command line arguments
   * \param pargv command line arguments
   */
  static void Enable (int* pargc, char*** pargv);
  /**
   * \brief Setup the parallel communication interface using the specified communicator.
   *
   * See @ref Enable (int* pargc, char*** pargv) for additional information.
   *
   * \param communicator MPI Communicator that should be used by ns-3
   */
  static void Enable (MPI_Comm communicator);
  /**
   * \brief Clean up the ns-3 parallel communications interface.
   *
   * MPI_Finalize will be called only if Enable (int* pargc, char***
   * pargv) was called.
   */
  static void Disable ();
  /**
   * \brief Send a packet to a remote node.
   *
   * \param p packet to send
   * \param rxTime received time at destination node
   * \param node destination node
   * \param dev destination device
   *
   * Serialize and send a packet to the specified node and net device
   */
  static void SendPacket (Ptr<Packet> p, const Time &rxTime, uint32_t node, uint32_t dev);

  /**
   * \brief Return the communicator used to run ns-3.
   *
   * The communicator returned will be MPI_COMM_WORLD if Enable (int*
   * pargc, char*** pargv) is used to enable or the user specified
   * communicator if Enable (MPI_Comm communicator) is used.
   *
   * \return The MPI Communicator.
   */
  static MPI_Comm GetCommunicator();

private:

  /**
   * Common enable logic.
   */
  static void SetParallelSimulatorImpl (void);

  /**
   * Static instance of the instantiated parallel controller.
   */
  static ParallelCommunicationInterface* g_parallelCommunicationInterface;
};

} // namespace ns3

#endif /* NS3_MPI_INTERFACE_H */
