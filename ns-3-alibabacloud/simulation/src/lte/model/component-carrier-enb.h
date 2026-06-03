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


#ifndef COMPONENT_CARRIER_ENB_H
#define COMPONENT_CARRIER_ENB_H

#include "component-carrier.h"
#include <ns3/object.h>
#include <ns3/packet.h>
#include <ns3/nstime.h>
#include "ns3/lte-phy.h"
#include <ns3/lte-enb-phy.h>
#include <ns3/pointer.h>
//#include <ns3/lte-enb-mac.h>


namespace ns3 {

class LteEnbMac;
class FfMacScheduler;
class LteFfrAlgorithm;

/**
 * \ingroup lte
 *
 * Defines a single carrier for enb, and contains pointers to LteEnbPhy,
 * LteEnbMac, LteFfrAlgorithm, and FfMacScheduler objects.
 *
 */
class ComponentCarrierEnb : public ComponentCarrierBaseStation
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  ComponentCarrierEnb ();

  virtual ~ComponentCarrierEnb (void);
  virtual void DoDispose (void);

  /**
   * \return a pointer to the physical layer.
   */
  Ptr<LteEnbPhy> GetPhy (void);

  /**
   * \return a pointer to the MAC layer.
   */
  Ptr<LteEnbMac> GetMac (void);

  /**
   * \return a pointer to the Ffr Algorithm.
   */
  Ptr<LteFfrAlgorithm> GetFfrAlgorithm ();

  /**
   * \return a pointer to the Mac Scheduler.
   */
  Ptr<FfMacScheduler> GetFfMacScheduler ();

  /**
   * Set the LteEnbPhy
   * \param s a pointer to the LteEnbPhy
   */
  void SetPhy (Ptr<LteEnbPhy> s);
  /**
   * Set the LteEnbMac
   * \param s a pointer to the LteEnbMac
   */
  void SetMac (Ptr<LteEnbMac> s);

  /**
   * Set the FfMacScheduler Algorithm
   * \param s a pointer to the FfMacScheduler
   */
  void SetFfMacScheduler (Ptr<FfMacScheduler> s);

  /**
   * Set the LteFfrAlgorithm
   * \param s a pointer to the LteFfrAlgorithm
   */
  void SetFfrAlgorithm (Ptr<LteFfrAlgorithm> s);
  
protected:

  virtual void DoInitialize (void);

private:
  Ptr<LteEnbPhy> m_phy; ///< the Phy instance of this eNodeB component carrier
  Ptr<LteEnbMac> m_mac; ///< the MAC instance of this eNodeB component carrier
  Ptr<FfMacScheduler> m_scheduler; ///< the scheduler instance of this eNodeB component carrier
  Ptr<LteFfrAlgorithm> m_ffrAlgorithm; ///< the FFR algorithm instance of this eNodeB component carrier
};

} // namespace ns3

#endif /* COMPONENT_CARRIER_H */
