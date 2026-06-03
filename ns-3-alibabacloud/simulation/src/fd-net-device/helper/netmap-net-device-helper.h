/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
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
 * Author: Pasquale Imputato <p.imputato@gmail.com>
 */

#ifndef NETMAP_NET_DEVICE_HELPER_H
#define NETMAP_NET_DEVICE_HELPER_H

#include <string>

#include "ns3/attribute.h"
#include "ns3/fd-net-device.h"
#include "ns3/fd-net-device-helper.h"
#include "ns3/object-factory.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/netmap-net-device.h"

namespace ns3 {

class NetmapNetDevice;

/**
 * \ingroup fd-net-device
 * \brief build a set of FdNetDevice objects attached to a physical network
 * interface
 *
 */
class NetmapNetDeviceHelper : public FdNetDeviceHelper
{
public:

  NetmapNetDeviceHelper ();
  virtual ~NetmapNetDeviceHelper ()
  {}

  /**
   * \brief Get the device name of this device.
   * \returns The device name of this device.
   */
  std::string GetDeviceName (void);

  /**
   * \brief Set the device name of this device.
   * \param deviceName The device name of this device.
   */
  void SetDeviceName (std::string deviceName);

protected:

  /**
   * \brief This method creates an ns3::FdNetDevice attached to a physical network
   * interface
   * \param node The node to install the device in
   * \returns A container holding the added net device.
   */
  Ptr<NetDevice> InstallPriv (Ptr<Node> node) const;

  /**
   * \brief Sets device flags and MTU.
   * \param device the FdNetDevice
   */
  virtual void SetDeviceAttributes (Ptr<FdNetDevice> device) const;

  /**
   * \brief Call out to a separate process running as suid root in order to get a raw
   * socket.  We do this to avoid having the entire simulation running as root.
   * \return the rawSocket number
   */
  virtual int CreateFileDescriptor (void) const;

  /**
   * \brief Switch the fd in netmap mode.
   * \param fd the file descriptor
   * \param device the NetmapNetDevice
   */
  void SwitchInNetmapMode (int fd, Ptr<NetmapNetDevice> device) const;

  std::string m_deviceName; //!< The unix/linux name of the underlying device (e.g., eth0)
};

} // namespace ns3

#endif /* NETMAP_NET_DEVICE_HELPER_H */
