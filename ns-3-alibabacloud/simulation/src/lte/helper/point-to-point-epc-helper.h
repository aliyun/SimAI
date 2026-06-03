/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011-2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Jaume Nin <jnin@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 *         Manuel Requena <manuel.requena@cttc.es>
 *         (most of the code refactored to no-backhaul-epc-helper.h)
 */

#ifndef POINT_TO_POINT_EPC_HELPER_H
#define POINT_TO_POINT_EPC_HELPER_H

#include "ns3/no-backhaul-epc-helper.h"

namespace ns3 {

/**
 * \ingroup lte
 * \brief Create an EPC network with PointToPoint links in the backhaul network.
 *
 * This Helper extends NoBackhaulEpcHelper creating PointToPoint links in the
 * backhaul network (i.e. in the S1-U and S1-MME interfaces)
 */
class PointToPointEpcHelper : public NoBackhaulEpcHelper
{
public:
  /**
   * Constructor
   */
  PointToPointEpcHelper ();

  /**
   * Destructor
   */
  virtual ~PointToPointEpcHelper ();

  // inherited from Object
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  TypeId GetInstanceTypeId () const;
  virtual void DoDispose ();

  // inherited from EpcHelper
  virtual void AddEnb (Ptr<Node> enbNode, Ptr<NetDevice> lteEnbNetDevice, std::vector<uint16_t> cellIds);

private:

  /**
   * S1-U interfaces
   */

  /**
   * Helper to assign addresses to S1-U NetDevices 
   */
  Ipv4AddressHelper m_s1uIpv4AddressHelper; 

  /**
   * The data rate to be used for the next S1-U link to be created
   */
  DataRate m_s1uLinkDataRate;

  /**
   * The delay to be used for the next S1-U link to be created
   */
  Time     m_s1uLinkDelay;

  /**
   * The MTU of the next S1-U link to be created. Note that,
   * because of the additional GTP/UDP/IP tunneling overhead,
   * you need a MTU larger than the end-to-end MTU that you
   * want to support.
   */
  uint16_t m_s1uLinkMtu;

  /**
   * Helper to assign addresses to S1-MME NetDevices
   */
  Ipv4AddressHelper m_s1apIpv4AddressHelper;

  /**
   * Enable PCAP generation for S1 link
   */
  bool        m_s1uLinkEnablePcap;

  /**
   * Prefix for the PCAP file for the S1 link
   */
  std::string m_s1uLinkPcapPrefix;
};

} // namespace ns3

#endif // POINT_TO_POINT_EPC_HELPER_H
