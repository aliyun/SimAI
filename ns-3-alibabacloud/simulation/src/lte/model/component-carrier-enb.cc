/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 */

#include "component-carrier-enb.h"
#include <ns3/uinteger.h>
#include <ns3/boolean.h>
#include <ns3/simulator.h>
#include <ns3/log.h>
#include <ns3/abort.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/pointer.h>
#include <ns3/lte-enb-mac.h>
#include <ns3/lte-ffr-algorithm.h>
#include <ns3/ff-mac-scheduler.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ComponentCarrierEnb");
NS_OBJECT_ENSURE_REGISTERED (ComponentCarrierEnb);

TypeId ComponentCarrierEnb::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::ComponentCarrierEnb")
    .SetParent<ComponentCarrier> ()
    .AddConstructor<ComponentCarrierEnb> ()
    .AddAttribute ("LteEnbPhy",
                   "The PHY associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&ComponentCarrierEnb::m_phy),
                   MakePointerChecker <LteEnbPhy> ())
    .AddAttribute ("LteEnbMac",
                   "The MAC associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&ComponentCarrierEnb::m_mac),
                   MakePointerChecker <LteEnbMac> ())
    .AddAttribute ("FfMacScheduler",
                   "The scheduler associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&ComponentCarrierEnb::m_scheduler),
                   MakePointerChecker <FfMacScheduler> ())
    .AddAttribute ("LteFfrAlgorithm",
                   "The FFR algorithm associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&ComponentCarrierEnb::m_ffrAlgorithm),
                   MakePointerChecker <LteFfrAlgorithm> ())
  ;
  return tid;
}
ComponentCarrierEnb::ComponentCarrierEnb ()
{
  NS_LOG_FUNCTION (this);
}

ComponentCarrierEnb::~ComponentCarrierEnb (void)
{
  NS_LOG_FUNCTION (this);
}

void
ComponentCarrierEnb::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  if (m_phy)
    {
      m_phy->Dispose ();
      m_phy = 0;
    }
  if (m_mac)
    {
      m_mac->Dispose ();
      m_mac = 0;
    }
  if (m_scheduler)
    {
      m_scheduler->Dispose ();
      m_scheduler = 0;
    }
  if (m_ffrAlgorithm)
    {
      m_ffrAlgorithm->Dispose ();
      m_ffrAlgorithm = 0;
    }

  Object::DoDispose ();
}


void
ComponentCarrierEnb::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  m_phy->Initialize ();
  m_mac->Initialize ();
  m_ffrAlgorithm->Initialize ();
  m_scheduler->Initialize();

}

Ptr<LteEnbPhy>
ComponentCarrierEnb::GetPhy ()
{
  NS_LOG_FUNCTION (this);
  return m_phy;
}


void 
ComponentCarrierEnb::SetPhy (Ptr<LteEnbPhy> s)
{
  NS_LOG_FUNCTION (this);
  m_phy = s;
}

Ptr<LteEnbMac>
ComponentCarrierEnb::GetMac ()
{
  NS_LOG_FUNCTION (this);
  return m_mac;
}
void 
ComponentCarrierEnb::SetMac (Ptr<LteEnbMac> s)
{
  NS_LOG_FUNCTION (this);
  m_mac = s;
}

Ptr<LteFfrAlgorithm>
ComponentCarrierEnb::GetFfrAlgorithm ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrAlgorithm;
}

void 
ComponentCarrierEnb::SetFfrAlgorithm (Ptr<LteFfrAlgorithm> s)
{
  NS_LOG_FUNCTION (this);
  m_ffrAlgorithm = s;
}

Ptr<FfMacScheduler>
ComponentCarrierEnb::GetFfMacScheduler ()
{
  NS_LOG_FUNCTION (this);
  return m_scheduler;
}

void
ComponentCarrierEnb::SetFfMacScheduler (Ptr<FfMacScheduler> s)
{
  NS_LOG_FUNCTION (this);
  m_scheduler = s;
}

} // namespace ns3


