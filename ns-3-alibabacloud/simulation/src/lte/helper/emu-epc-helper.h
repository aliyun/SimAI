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
 */

#ifndef EMU_EPC_HELPER_H
#define EMU_EPC_HELPER_H

#include "ns3/no-backhaul-epc-helper.h"

namespace ns3 {

/**
 * \ingroup lte
 *
 * \brief Create an EPC network using EmuFdNetDevice 
 *
 * This Helper will create an EPC network topology comprising of a
 * single node that implements both the SGW and PGW functionality, and
 * an MME node. The S1-U, X2-U and X2-C interfaces are realized using
 * EmuFdNetDevice; in particular, one device is used to send all the
 * traffic related to these interfaces. 
 */
class EmuEpcHelper : public NoBackhaulEpcHelper
{
public:
  /**
   * Constructor
   */
  EmuEpcHelper ();

  /**
   * Destructor
   */
  virtual ~EmuEpcHelper ();

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
  virtual void AddX2Interface (Ptr<Node> enbNode1, Ptr<Node> enbNode2);

private:

  /**
   * helper to assign addresses to S1-U NetDevices
   */
  Ipv4AddressHelper m_epcIpv4AddressHelper;

  /**
   * Container for Ipv4Interfaces of the SGW
   */
  Ipv4InterfaceContainer m_sgwIpIfaces; 

  /**
   * The name of the device used for the S1-U interface of the SGW
   */
  std::string m_sgwDeviceName;

  /**
   * The name of the device used for the S1-U interface of the eNB
   */
  std::string m_enbDeviceName;

  /**
   * MAC address used for the SGW
   */
  std::string m_sgwMacAddress;

  /**
   * First 5 bytes of the Enb MAC address base
   */
  std::string m_enbMacAddressBase;
};

} // namespace ns3

#endif // EMU_EPC_HELPER_H
