/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#include "bs-scheduler.h"
#include "ns3/simulator.h"
#include "bs-net-device.h"
#include "ns3/packet-burst.h"
#include "cid.h"
#include "wimax-mac-header.h"
#include "ss-record.h"
#include "wimax-mac-queue.h"
#include "ns3/log.h"
#include "burst-profile-manager.h"
#include "wimax-connection.h"
#include "connection-manager.h"
#include "ss-manager.h"
#include "service-flow.h"
#include "service-flow-record.h"
#include "service-flow-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BSScheduler");
  
NS_OBJECT_ENSURE_REGISTERED (BSScheduler);

TypeId
BSScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BSScheduler")
    .SetParent<Object> ()
    .SetGroupName("Wimax")
    // No AddConstructor because this is an abstract class.
    ;
  return tid;
}

BSScheduler::BSScheduler ()
  : m_downlinkBursts (new std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > > ())
{
  // m_downlinkBursts is filled by AddDownlinkBurst and emptied by
  // wimax-bs-net-device::sendBurst and wimax-ss-net-device::sendBurst
}

BSScheduler::BSScheduler (Ptr<BaseStationNetDevice> bs)
  : m_downlinkBursts (new std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > > ())
{

}

BSScheduler::~BSScheduler (void)
{
  std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > > *downlinkBursts = m_downlinkBursts;
  std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > pair;
  while (downlinkBursts->size ())
    {
      pair = downlinkBursts->front ();
      pair.second = 0;
      delete pair.first;
    }
  SetBs (0);
  delete m_downlinkBursts;
  m_downlinkBursts = 0;
}

void
BSScheduler::SetBs (Ptr<BaseStationNetDevice> bs)
{
  m_bs = bs;
}

Ptr<BaseStationNetDevice> BSScheduler::GetBs (void)
{
  return m_bs;
}

bool
BSScheduler::CheckForFragmentation (Ptr<WimaxConnection> connection,
                                    int availableSymbols,
                                    WimaxPhy::ModulationType modulationType)
{
  NS_LOG_INFO ("BS Scheduler, CheckForFragmentation");
  if (connection->GetType () != Cid::TRANSPORT)
    {
      NS_LOG_INFO ("\t No Transport connction, Fragmentation IS NOT possible");
      return false;
    }
  uint32_t availableByte = GetBs ()->GetPhy ()->
    GetNrBytes (availableSymbols, modulationType);

  uint32_t headerSize = connection->GetQueue ()->GetFirstPacketHdrSize (
      MacHeaderType::HEADER_TYPE_GENERIC);
  NS_LOG_INFO ("\t availableByte = " << availableByte <<
               " headerSize = " << headerSize);

  if (availableByte > headerSize)
    {
      NS_LOG_INFO ("\t Fragmentation IS possible");
      return true;
    }
  else
    {
      NS_LOG_INFO ("\t Fragmentation IS NOT possible");
      return false;
    }
}
} // namespace ns3
