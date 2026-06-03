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
 * Implementation of class ns3::RemoteChannelBundleManager.
 */

#include "remote-channel-bundle-manager.h"

#include "remote-channel-bundle.h"
#include "null-message-simulator-impl.h"

#include "ns3/simulator.h"

namespace ns3 {

bool ns3::RemoteChannelBundleManager::g_initialized = false;
ns3::RemoteChannelBundleManager::RemoteChannelMap ns3::RemoteChannelBundleManager::g_remoteChannelBundles;

Ptr<RemoteChannelBundle>
RemoteChannelBundleManager::Find (uint32_t systemId)
{
  ns3::RemoteChannelBundleManager::RemoteChannelMap::iterator kv = g_remoteChannelBundles.find (systemId);

  if ( kv == g_remoteChannelBundles.end ())
    {
      return 0;
    }
  else
    {
      return kv->second;
    }
}

Ptr<RemoteChannelBundle>
RemoteChannelBundleManager::Add (uint32_t systemId)
{
  NS_ASSERT (!g_initialized);
  NS_ASSERT (g_remoteChannelBundles.find (systemId) == g_remoteChannelBundles.end ());

  Ptr<RemoteChannelBundle> remoteChannelBundle = Create<RemoteChannelBundle> (systemId);

  g_remoteChannelBundles[systemId] = remoteChannelBundle;

  return remoteChannelBundle;
}

std::size_t
RemoteChannelBundleManager::Size (void) 
{
  return g_remoteChannelBundles.size();
}

void
RemoteChannelBundleManager::InitializeNullMessageEvents (void)
{
  NS_ASSERT (!g_initialized);

  for ( RemoteChannelMap::const_iterator iter = g_remoteChannelBundles.begin ();
        iter != g_remoteChannelBundles.end ();
        ++iter )
    {
      Ptr<RemoteChannelBundle> bundle = iter->second;
      bundle->Send (bundle->GetDelay ());

      NullMessageSimulatorImpl::GetInstance ()->ScheduleNullMessageEvent (bundle);
    }

  g_initialized = true;
}

Time
RemoteChannelBundleManager::GetSafeTime (void)
{
  NS_ASSERT (g_initialized);

  Time safeTime = Simulator::GetMaximumSimulationTime ();

  for (RemoteChannelMap::const_iterator kv = g_remoteChannelBundles.begin ();
       kv != g_remoteChannelBundles.end ();
       ++kv)
    {
      safeTime = Min (safeTime, kv->second->GetGuaranteeTime ());
    }

  return safeTime;
}

void
RemoteChannelBundleManager::Destroy (void)
{
  NS_ASSERT (g_initialized);

  g_remoteChannelBundles.clear();
  g_initialized = false;
}

} // namespace ns3
