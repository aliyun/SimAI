/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#ifndef UAN_RC_EXAMPLE_H
#define UAN_RC_EXAMPLE_H

#include "ns3/network-module.h"
#include "ns3/stats-module.h"
#include "ns3/uan-module.h"

using namespace ns3;

/**
 * \ingroup uan
 *
 * Container for the parameters describing a single experiment.
 *
 * An experiment samples a range of parameter values.  The parameter
 * controls the number of nodes (if m_doNode is true), or the
 * UanMacRcGw MaxReservations attribute.
 *
 * \see uan-rc-example.cc
 *
 */
class Experiment
{
public:
  uint32_t m_simMin;     //!< Minimum parameter to test.
  uint32_t m_simMax;     //!< Maximum parameter to test.
  uint32_t m_simStep;    //!< Amount to increment param per trial
  uint32_t m_numRates;   //!< Number of divided rates ( (NumberRates+1)%TotalRate should be 0).
  uint32_t m_totalRate;  //!< Total channel capacity.
  uint32_t m_maxRange;   //!< Maximum range between gateway and acoustic node.
  uint32_t m_numNodes;   //!< Number of nodes (invalid for m_doNode true).
  uint32_t m_pktSize;    //!< Packet size in bytes.
  bool m_doNode;         //!< 1 for do max nodes simulation (invalidates AMin and AMax values).
  Time m_sifs;           //!< SIFS time duration.
  Time m_simTime;        //!< Simulation time per trial

  std::string m_gnuplotfile;    //!< Filename for GnuPlot.

  uint32_t m_bytesTotal; //!< Total number of bytes received in a simulation run.

  UanModesList m_dataModes;     //!< List of UanTxModes used for data channels.
  UanModesList m_controlModes;  //!< List of UanTxModes used for control channels.

  /**
   * Callback to receive a packet.
   *
   * \param socket The socket receiving packets.
   */
  void ReceivePacket (Ptr<Socket> socket);
  /**
   * Create a UanTxMode.
   *
   * The mode physical rate is set equal to m_totalRate.
   * The data rate and bandwidth are set to
   *
   * \f[{\rm{rate, bandwidth}} = {\rm{kass}}\frac{{{\rm{m\_totalRate}}}}{{{\rm{m\_numRates}} + 1}}\f]
   *
   * The center frequency is set to
   *
   * \f[{f_{center}} = {\rm{fc}} + \frac{{ \pm {\rm{m\_totalRate}} \mp {\rm{rate}}}}{2}\f]
   *
   * where the upper sign is taken if upperblock is true.
   *
   * \param kass Fraction of total bandwidth assigned to mode.
   * \param fc Mode center frequency offset.
   * \param upperblock Sign choice in setting the center frequency.
   * \param name Mode name.
   * \return The new mode.
   */
  UanTxMode CreateMode (uint32_t kass,
                        uint32_t fc,
                        bool upperblock,
                        std::string name);
  /**
   * Create m_numRates matching control and data modes.
   *
   * \param fc Mode center frequency offset.
   */
  void CreateDualModes (uint32_t fc);
  /**
   * Run a parametrized experiment.
   *
   * The parameter sets either the number of nodes (if m_doNode is true)
   * or the \"a\" parameter, which controls the UanMacRcGw MaxReservations
   * attribute.
   *
   * \param param The parameter value.
   * \return The total number of bytes delivered.
   */
  uint32_t Run (uint32_t param);

  /** Default constructor. */
  Experiment();

};

#endif /* UAN_RC_EXAMPLE_H */
