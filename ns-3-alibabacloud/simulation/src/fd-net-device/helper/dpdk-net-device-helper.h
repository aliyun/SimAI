/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 NITK Surathkal
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
 * Authors: Harsh Patel <thadodaharsh10@gmail.com>
 *          Hrishikesh Hiraskar <hrishihiraskar@gmail.com>
 */

#ifndef DPDK_NET_DEVICE_HELPER_H
#define DPDK_NET_DEVICE_HELPER_H

#include "ns3/emu-fd-net-device-helper.h"

namespace ns3 {

/**
 * \ingroup dpdk-net-device
 * \brief build a DpdkNetDevice object attached to a physical network
 * interface
 *
 */
class DpdkNetDeviceHelper : public EmuFdNetDeviceHelper
{
public:
  /**
   * Construct a DpdkNetDeviceHelper and initialize DPDK EAL
   *
   * \param argc Dpdk EAL args count.
   * \param argv Dpdk EAL args list.
   */
  DpdkNetDeviceHelper ();

  virtual ~DpdkNetDeviceHelper ()
  {}

  /**
   * \brief Sets list of logical cores to use
   *
   * \param lCoreList Comma separated logical core list (e.g., "0,1")
   */
  void SetLCoreList (std::string lCoreList);

  /**
   * \brief Sets PMD Library to be used for the NIC
   *
   * \param pmdLibrary The PMD Library
   */
  void SetPmdLibrary (std::string pmdLibrary);

  /**
   * \brief Sets DPDK Driver to bind NIC to
   *
   * \param dpdkDriver The DPDK Driver
   */
  void SetDpdkDriver (std::string dpdkDriver);

protected:
  /**
   * \brief This method creates an ns3::FdNetDevice attached to a physical network
   * interface
   *
   * \param node The node to install the device in
   * \returns A container holding the added net device.
   */
  Ptr<NetDevice> InstallPriv (Ptr<Node> node) const;

  /**
   * Logical cores to use
   */
  std::string m_lCoreList;

  /**
   * PMD Library to be used for NIC
   */
  std::string m_pmdLibrary;

  /**
   * DPDK Driver to bind NIC to
   */
  std::string m_dpdkDriver;

};

} // namespace ns3

#endif /* DPDK_NET_DEVICE_HELPER_H */
