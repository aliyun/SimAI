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

#include "dpdk-net-device-helper.h"

#include "ns3/dpdk-net-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DpdkNetDeviceHelper");

DpdkNetDeviceHelper::DpdkNetDeviceHelper ()
  : m_lCoreList ("0,1"),
    m_pmdLibrary ("librte_pmd_e1000.so"),
    m_dpdkDriver ("uio_pci_generic")
{
  NS_LOG_FUNCTION (this);
  SetTypeId ("ns3::DpdkNetDevice");
}

void
DpdkNetDeviceHelper::SetLCoreList (std::string lCoreList)
{
  m_lCoreList = lCoreList;
}

void
DpdkNetDeviceHelper::SetPmdLibrary (std::string pmdLibrary)
{
  m_pmdLibrary = pmdLibrary;
}

void
DpdkNetDeviceHelper::SetDpdkDriver (std::string dpdkDriver)
{
  m_dpdkDriver = dpdkDriver;
}

Ptr<NetDevice>
DpdkNetDeviceHelper::InstallPriv (Ptr<Node> node) const
{
  NS_LOG_FUNCTION (this);
  Ptr<NetDevice> device = FdNetDeviceHelper::InstallPriv (node);
  Ptr<DpdkNetDevice> dpdkDevice = DynamicCast<DpdkNetDevice> (device);
  dpdkDevice->SetDeviceName (m_deviceName);

  // Set EAL arguments
  char **ealArgv = new char*[20];

  // arg[0] is program name (optional)
  ealArgv[0] = new char[20];
  strcpy (ealArgv[0], "");

  // Logical core usage
  ealArgv[1] = new char[20];
  strcpy (ealArgv[1], "-l");
  ealArgv[2] = new char[20];
  strcpy (ealArgv[2], m_lCoreList.c_str ());

  // Load library
  ealArgv[3] = new char[20];
  strcpy (ealArgv[3], "-d");
  ealArgv[4] = new char[50];
  strcpy (ealArgv[4], m_pmdLibrary.c_str ());

  // Use mempool ring library
  ealArgv[5] = new char[20];
  strcpy (ealArgv[5], "-d");
  ealArgv[6] = new char[50];
  strcpy (ealArgv[6], "librte_mempool_ring.so");

  dpdkDevice->InitDpdk (7, ealArgv, m_dpdkDriver);
  return dpdkDevice;
}

} // namespace ns3
