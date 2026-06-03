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
 * Implementation of class ns3::MpiInterface.
 */

#include "mpi-interface.h"

#include <ns3/global-value.h>
#include <ns3/string.h>
#include <ns3/log.h>

#include "null-message-mpi-interface.h"
#include "granted-time-window-mpi-interface.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MpiInterface");

ParallelCommunicationInterface* MpiInterface::g_parallelCommunicationInterface = 0;

void
MpiInterface::Destroy ()
{
  NS_ASSERT (g_parallelCommunicationInterface);
  g_parallelCommunicationInterface->Destroy ();
}

uint32_t
MpiInterface::GetSystemId ()
{
  if ( g_parallelCommunicationInterface )
    return g_parallelCommunicationInterface->GetSystemId ();
  else
    return 0;
}

uint32_t
MpiInterface::GetSize ()
{
  if ( g_parallelCommunicationInterface )
    return g_parallelCommunicationInterface->GetSize ();
  else
    return 1;
}

bool
MpiInterface::IsEnabled ()
{
  if (g_parallelCommunicationInterface)
    {
      return g_parallelCommunicationInterface->IsEnabled ();
    }
  else
    {
      return false;
    }
}

void
MpiInterface::SetParallelSimulatorImpl (void)
{
  StringValue simulationTypeValue;
  bool useDefault = true;

  if (GlobalValue::GetValueByNameFailSafe ("SimulatorImplementationType", simulationTypeValue))
    {
      std::string simulationType = simulationTypeValue.Get ();

      // Set communication interface based on the simulation type being used.
      // Defaults to synchronous.
      if (simulationType.compare ("ns3::NullMessageSimulatorImpl") == 0)
        {
          g_parallelCommunicationInterface = new NullMessageMpiInterface ();
          useDefault = false;
        }
      else if (simulationType.compare ("ns3::DistributedSimulatorImpl") == 0 || simulationType.compare("ns3::HybridSimulatorImpl") == 0)
        {
          g_parallelCommunicationInterface = new GrantedTimeWindowMpiInterface ();
          useDefault = false;
        }
    }

  // User did not specify a valid parallel simulator; use the default.
  if (useDefault)
    {
      g_parallelCommunicationInterface = new GrantedTimeWindowMpiInterface ();
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::DistributedSimulatorImpl"));
      NS_LOG_WARN ("SimulatorImplementationType was set to non-parallel simulator; setting type to ns3::DistributedSimulatorImp");
    }
}

void
MpiInterface::Enable (int* pargc, char*** pargv)
{

  SetParallelSimulatorImpl ();

  g_parallelCommunicationInterface->Enable (pargc, pargv);
}

void
MpiInterface::Enable (MPI_Comm communicator)
{
  SetParallelSimulatorImpl ();
  g_parallelCommunicationInterface->Enable (communicator);
}

void
MpiInterface::SendPacket (Ptr<Packet> p, const Time& rxTime, uint32_t node, uint32_t dev)
{
  NS_ASSERT (g_parallelCommunicationInterface);
  g_parallelCommunicationInterface->SendPacket (p, rxTime, node, dev);
}

MPI_Comm 
MpiInterface::GetCommunicator()
{
  NS_ASSERT (g_parallelCommunicationInterface);
  return g_parallelCommunicationInterface->GetCommunicator ();
}


void
MpiInterface::Disable ()
{
  NS_ASSERT (g_parallelCommunicationInterface);
  g_parallelCommunicationInterface->Disable ();
  delete g_parallelCommunicationInterface;
  g_parallelCommunicationInterface = 0;
}


} // namespace ns3
