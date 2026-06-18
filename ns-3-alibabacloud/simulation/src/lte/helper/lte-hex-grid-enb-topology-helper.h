/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef LTE_HEX_GRID_ENB_TOPOLOGY_HELPER_H
#define LTE_HEX_GRID_ENB_TOPOLOGY_HELPER_H

#include <ns3/lte-helper.h>

namespace ns3 {

/**
 * \ingroup lte
 *
 * This helper class allows to easily create a topology with eNBs
 * grouped in three-sector sites laid out on an hexagonal grid. The
 * layout is done row-wise. 
 *
 */
class LteHexGridEnbTopologyHelper : public Object
{
public:
  LteHexGridEnbTopologyHelper (void);
  virtual ~LteHexGridEnbTopologyHelper (void);

  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  virtual void DoDispose (void);


  /** 
   * Set the LteHelper to be used to actually create the EnbNetDevices
   *
   * \note if no EpcHelper is ever set, then LteHexGridEnbTopologyHelper will default
   * to creating an LTE-only simulation with no EPC, using LteRlcSm as
   * the RLC model, and without supporting any IP networking. In other
   * words, it will be a radio-level simulation involving only LTE PHY
   * and MAC and the FF Scheduler, with a saturation traffic model for
   * the RLC.
   * 
   * \param h a pointer to the EpcHelper to be used
   */
  void SetLteHelper (Ptr<LteHelper> h);

  /**
   * Position the nodes on a hex grid and install the corresponding
   * EnbNetDevices with antenna boresight configured properly
   *
   * \param c the node container where the devices are to be installed
   *
   * \return the NetDeviceContainer with the newly created devices
   */
  NetDeviceContainer SetPositionAndInstallEnbDevice (NodeContainer c);

private:
  /**
   * Pointer to LteHelper object
   */
  Ptr<LteHelper> m_lteHelper;

  /**
   * The offset [m] in the position for the node of each sector with
   * respect to the center of the three-sector site
   */
  double m_offset;

  /**
   * The distance [m] between nearby sites
   */
  double m_d;

  /**
   * The x coordinate where the hex grid starts
   */
  double m_xMin;

  /**
   * The y coordinate where the hex grid starts
   */
  double m_yMin;

  /**
   * The number of sites in even rows (odd rows will have
   * one additional site)
   */
  uint32_t m_gridWidth;

  /**
   * The height [m] of each site
   */
  uint32_t m_siteHeight;

};


} // namespace ns3



#endif // LTE_HEX_GRID_ENB_TOPOLOGY_HELPER_H
