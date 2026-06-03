/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 INRIA, 2012 University of Washington
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
 */

#ifndef EMU_FD_NET_DEVICE_HELPER_H
#define EMU_FD_NET_DEVICE_HELPER_H

#include <string>

#include "ns3/attribute.h"
#include "ns3/fd-net-device.h"
#include "ns3/fd-net-device-helper.h"
#include "ns3/object-factory.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"

namespace ns3 {

/**
 * \ingroup fd-net-device
 * \brief build a set of FdNetDevice objects attached to a physical network
 * interface
 *
 */
class EmuFdNetDeviceHelper : public FdNetDeviceHelper
{
public:
  /**
   * Construct a EmuFdNetDeviceHelper.
   */
  EmuFdNetDeviceHelper ();
  virtual ~EmuFdNetDeviceHelper ()
  {}

  /**
   * Get the device name of this device.
   *
   * \returns The device name of this device.
   */
  std::string GetDeviceName (void);

  /**
   * Set the device name of this device.
   *
   * \param deviceName The device name of this device.
   */
  void SetDeviceName (std::string deviceName);

  /**
   * \brief Request host qdisc bypass
   * \param hostQdiscBypass to enable host qdisc bypass
   */
  void HostQdiscBypass (bool hostQdiscBypass);

protected:

  /**
   * This method creates an ns3::FdNetDevice attached to a physical network
   * interface
   *
   * \param node The node to install the device in
   * \returns A container holding the added net device.
   */
  Ptr<NetDevice> InstallPriv (Ptr<Node> node) const;

  /**
   * Sets a file descriptor on the FileDescriptorNetDevice.
   * \param device the device to install the file descriptor in
   */
  virtual void SetFileDescriptor (Ptr<FdNetDevice> device) const;

  /**
   * Call out to a separate process running as suid root in order to get a raw
   * socket.  We do this to avoid having the entire simulation running as root.
   * \return the rawSocket number
   */
  virtual int CreateFileDescriptor (void) const;

  /**
   * The Unix/Linux name of the underlying device (e.g., eth0)
   */
  std::string m_deviceName;
  bool m_hostQdiscBypass; //!< True if request host qdisc bypass
};

} // namespace ns3

#endif /* EMU_FD_NET_DEVICE_HELPER_H */
