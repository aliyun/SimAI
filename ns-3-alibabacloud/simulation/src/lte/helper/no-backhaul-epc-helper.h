/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 *         (based on the original point-to-point-epc-helper.h)
 */

#ifndef NO_BACKHAUL_EPC_HELPER_H
#define NO_BACKHAUL_EPC_HELPER_H

#include "ns3/epc-helper.h"

namespace ns3 {

class EpcSgwApplication;
class EpcPgwApplication;
class EpcMmeApplication;

/**
 * \ingroup lte
 * \brief Create an EPC network with PointToPoint links between the core network nodes.
 *
 * This Helper will create an EPC network topology comprising of
 * three nodes: SGW, PGW and MME.
 * The X2-U, X2-C, S5 and S11 interfaces are realized over PointToPoint links.
 *
 * The S1 interface is not created. So, no backhaul network is built.
 * You have to build your own backhaul network in the simulation program.
 * Or you can use PointToPointEpcHelper or CsmaEpcHelper
 * (instead of this NoBackhaulEpcHelper) to use reference backhaul networks.
 */
class NoBackhaulEpcHelper : public EpcHelper
{
public:
  /**
   * Constructor
   */
  NoBackhaulEpcHelper ();

  /**
   * Destructor
   */
  virtual ~NoBackhaulEpcHelper ();

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
  virtual void AddUe (Ptr<NetDevice> ueLteDevice, uint64_t imsi);
  virtual void AddX2Interface (Ptr<Node> enbNode1, Ptr<Node> enbNode2);
  virtual void AddS1Interface (Ptr<Node> enb, Ipv4Address enbAddress, Ipv4Address sgwAddress, std::vector<uint16_t> cellIds);
  virtual uint8_t ActivateEpsBearer (Ptr<NetDevice> ueLteDevice, uint64_t imsi, Ptr<EpcTft> tft, EpsBearer bearer);
  virtual Ptr<Node> GetSgwNode () const;
  virtual Ptr<Node> GetPgwNode () const;
  virtual Ipv4InterfaceContainer AssignUeIpv4Address (NetDeviceContainer ueDevices);
  virtual Ipv6InterfaceContainer AssignUeIpv6Address (NetDeviceContainer ueDevices);
  virtual Ipv4Address GetUeDefaultGatewayAddress ();
  virtual Ipv6Address GetUeDefaultGatewayAddress6 ();

protected:
  /**
   * \brief DoAddX2Interface: Call AddX2Interface on top of the Enb device pointers
   *
   * \param enb1X2 EPCX2 of ENB1
   * \param enb1LteDev LTE device of ENB1
   * \param enb1X2Address Address for ENB1
   * \param enb2X2 EPCX2 of ENB2
   * \param enb2LteDev LTE device of ENB2
   * \param enb2X2Address Address for ENB2
   */
  virtual void DoAddX2Interface(const Ptr<EpcX2> &enb1X2, const Ptr<NetDevice> &enb1LteDev,
                                const Ipv4Address &enb1X2Address,
                                const Ptr<EpcX2> &enb2X2, const Ptr<NetDevice> &enb2LteDev,
                                const Ipv4Address &enb2X2Address) const;

  /**
   * \brief DoActivateEpsBearerForUe: Schedule ActivateEpsBearer on the UE
   * \param ueDevice LTE device for the UE
   * \param tft TFT
   * \param bearer Bearer
   */
  virtual void DoActivateEpsBearerForUe (const Ptr<NetDevice> &ueDevice,
                                         const Ptr<EpcTft> &tft,
                                         const EpsBearer &bearer) const;

private:

  /**
   * helper to assign IPv4 addresses to UE devices as well as to the TUN device of the SGW/PGW
   */
  Ipv4AddressHelper m_uePgwAddressHelper;
  /**
   * helper to assign IPv6 addresses to UE devices as well as to the TUN device of the SGW/PGW
   */
  Ipv6AddressHelper m_uePgwAddressHelper6;

  /**
   * PGW network element
   */
  Ptr<Node> m_pgw;

  /**
   * SGW network element
   */
  Ptr<Node> m_sgw;

  /**
   * MME network element
   */
  Ptr<Node> m_mme;

  /**
   * SGW application
   */
  Ptr<EpcSgwApplication> m_sgwApp;

  /**
   * PGW application
   */
  Ptr<EpcPgwApplication> m_pgwApp;

  /**
   * MME application
   */
  Ptr<EpcMmeApplication> m_mmeApp;

  /**
   * TUN device implementing tunneling of user data over GTP-U/UDP/IP
   */
  Ptr<VirtualNetDevice> m_tunDevice;

  /**
   * UDP port where the GTP-U Socket is bound, fixed by the standard as 2152
   */
  uint16_t m_gtpuUdpPort;

  /**
   * Helper to assign addresses to S11 NetDevices
   */
  Ipv4AddressHelper m_s11Ipv4AddressHelper;

  /**
   * The data rate to be used for the next S11 link to be created
   */
  DataRate m_s11LinkDataRate;

  /**
   * The delay to be used for the next S11 link to be created
   */
  Time     m_s11LinkDelay;

  /**
   * The MTU of the next S11 link to be created
   */
  uint16_t m_s11LinkMtu;

  /**
   * UDP port where the GTPv2-C Socket is bound, fixed by the standard as 2123
   */
  uint16_t m_gtpcUdpPort;

  /**
   * S5 interfaces
   */

  /**
   * Helper to assign addresses to S5 NetDevices
   */
  Ipv4AddressHelper m_s5Ipv4AddressHelper; 

  /**
   * The data rate to be used for the next S5 link to be created
   */
  DataRate m_s5LinkDataRate;

  /**
   * The delay to be used for the next S5 link to be created
   */
  Time     m_s5LinkDelay;

  /**
   * The MTU of the next S5 link to be created
   */
  uint16_t m_s5LinkMtu;

  /**
   * Map storing for each IMSI the corresponding eNB NetDevice
   */
  std::map<uint64_t, Ptr<NetDevice> > m_imsiEnbDeviceMap;

  /**
   * helper to assign addresses to X2 NetDevices
   */
  Ipv4AddressHelper m_x2Ipv4AddressHelper;

  /**
   * The data rate to be used for the next X2 link to be created
   */
  DataRate m_x2LinkDataRate;

  /**
   * The delay to be used for the next X2 link to be created
   */
  Time     m_x2LinkDelay;

  /**
   * The MTU of the next X2 link to be created. Note that,
   * because of some big X2 messages, you need a big MTU.
   */
  uint16_t m_x2LinkMtu;

  /**
   * Enable PCAP generation for X2 link
   */
  bool        m_x2LinkEnablePcap;

  /**
   * Prefix for the PCAP file for the X2 link
   */
  std::string m_x2LinkPcapPrefix;

};

} // namespace ns3

#endif // NO_BACKHAUL_EPC_HELPER_H
