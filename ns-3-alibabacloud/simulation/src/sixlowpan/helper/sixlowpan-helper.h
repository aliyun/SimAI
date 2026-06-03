/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#ifndef SIXLOWPAN_HELPER_H
#define SIXLOWPAN_HELPER_H

#include "ns3/net-device-container.h"
#include "ns3/object-factory.h"
#include <string>

namespace ns3 {

class Node;
class AttributeValue;
class Time;

/**
 * \ingroup sixlowpan
 *
 * \brief Setup a sixlowpan stack to be used as a shim between IPv6 and a generic NetDevice.
 */
class SixLowPanHelper
{
public:
  /*
   * Construct a SixlowpanHelper
   */
  SixLowPanHelper ();
  /**
   * Set an attribute on each ns3::SixlowpanNetDevice created by
   * SixlowpanHelper::Install.
   *
   * \param n1 [in] The name of the attribute to set.
   * \param v1 [in] The value of the attribute to set.
   */
  void SetDeviceAttribute (std::string n1,
                           const AttributeValue &v1);

  /**
   * \brief Install the SixLoWPAN stack on top of an existing NetDevice.
   *
   * This function requires a set of properly configured NetDevices
   * passed in as the parameter "c". The new NetDevices will have to
   * be used instead of the original ones. In this way these
   * SixLoWPAN devices will behave as shims between the NetDevices
   * passed in and IPv6.
   *
   * Note that only IPv6 (and related protocols, such as ICMPv6) can
   * be transmitted over a 6LoWPAN interface.
   * Any other protocol (e.g., IPv4) will be discarded by 6LoWPAN.
   *
   * Other protocols (e.g., IPv4) could be used on the original NetDevices
   * with some limitations.
   * See the manual for a complete discussion.
   *
   * \note IPv6 stack must be installed \a after SixLoWPAN,
   * using the SixLoWPAN NetDevices. See the example in the
   * examples directory.
   *
   *
   * \param [in] c The NetDevice container.
   * \return A container with the newly created SixLowPanNetDevices.
   */
  NetDeviceContainer Install (NetDeviceContainer c);

  /**
   * \brief Adds a compression Context to a set of NetDevices.
   *
   * This function installs one Compression Context on a set of NetDevices.
   * The context is used only in IPHC compression and decompression.
   *
   * \param [in] c The NetDevice container.
   * \param [in] contextId The context id (must be less than 16).
   * \param [in] context The context prefix.
   * \param [in] validity the context validity time (relative to the actual time).
   */
  void AddContext (NetDeviceContainer c, uint8_t contextId, Ipv6Prefix context, Time validity);

  /**
   * \brief Renew a compression Context in a set of NetDevices.
   *
   * The context will have its lifetime extended and its validity for compression re-enabled.
   *
   * \param [in] c The NetDevice container.
   * \param [in] contextId The context id (must be less than 16).
   * \param [in] validity the context validity time (relative to the actual time).
   */
  void RenewContext (NetDeviceContainer c, uint8_t contextId, Time validity);

  /**
   * \brief Invalidates a compression Context in a set of NetDevices.
   *
   * An invalid context is used only in IPHC decompression and not
   * in IPHC compression.
   *
   * This is necessary to avoid that a context reaching its validity lifetime
   * can not be used for decompression whie packets are traveling the network.
   *
   * \param [in] c The NetDevice container.
   * \param [in] contextId The context id (must be less than 16).
   */
  void InvalidateContext (NetDeviceContainer c, uint8_t contextId);

  /**
   * \brief Remove a compression Context in a set of NetDevices.
   *
   * The context is removed immediately from the contexts in the devices.
   *
   * \param [in] c The NetDevice container.
   * \param [in] contextId The context id (must be less than 16).
   */
  void RemoveContext (NetDeviceContainer c, uint8_t contextId);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model. Return the number of streams (possibly zero) that
   * have been assigned. The Install() method should have previously been
   * called by the user.
   *
   * \param [in] c NetDeviceContainer of the set of net devices for which the
   *          SixLowPanNetDevice should be modified to use a fixed stream.
   * \param [in] stream First stream index to use.
   * \return The number of stream indices assigned by this helper.
   */
  int64_t AssignStreams (NetDeviceContainer c, int64_t stream);

private:
  ObjectFactory m_deviceFactory; //!< Object factory.
};

} // namespace ns3


#endif /* SIXLOWPAN_HELPER_H */
