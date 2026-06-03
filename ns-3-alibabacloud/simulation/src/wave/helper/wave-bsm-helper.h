/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 North Carolina State University
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
 * Author: Scott E. Carpenter <scarpen@ncsu.edu>
 *
 */

#ifndef WAVE_BSM_HELPER_H
#define WAVE_BSM_HELPER_H

#include <vector>
#include "ns3/wave-bsm-stats.h"
#include "ns3/bsm-application.h"
#include "ns3/object-factory.h"
#include "ns3/application-container.h"
#include "ns3/nstime.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/mobility-model.h"

namespace ns3 {
/**
 * \ingroup wave
 * \brief The WaveBsmHelper class manages
 * IEEE 1609 WAVE (Wireless Access in Vehicular Environments)
 * Basic Safety Messages (BSMs) and uses the WaveBsmStats class
 * to manage statistics about BSMs transmitted and received
 * The BSM is a ~200-byte packet that is
 * generally broadcast from every vehicle at a nominal rate of 10 Hz.
 */
class WaveBsmHelper
{
public:
  /**
   * \brief Constructor
   */
  WaveBsmHelper ();

  /**
   * Helper function used to set the underlying application attributes.
   *
   * \param name the name of the application attribute to set
   * \param value the value of the application attribute to set
   */
  void SetAttribute (std::string name, const AttributeValue &value);

  /**
   * Install an ns3::BsmApplication on each node of the input container
   * configured with all the attributes set with SetAttribute.
   *
   * \param i Ipv4InterfaceContainer of the set of interfaces on which an BsmApplication
   * will be installed on the nodes.
   * \returns Container of Ptr to the applications installed.
   */
  ApplicationContainer Install (Ipv4InterfaceContainer i) const;

  /**
   * Install an ns3::BsmApplication on the node configured with all the
   * attributes set with SetAttribute.
   *
   * \param node The node on which an BsmApplication will be installed.
   * \returns Container of Ptr to the applications installed.
   */
  ApplicationContainer Install (Ptr<Node> node) const;

  /**
   * \brief Installs BSM generation on devices for nodes
   * and their interfaces
   * \param i IPv4 interface container
   * \param totalTime total amount of time that BSM packets should be transmitted
   * \param wavePacketSize the size, in bytes, of a WAVE BSM
   * \param waveInterval the time, in seconds, between each WAVE BSM transmission,
   * typically 10 Hz (0.1 second)
   * \param gpsAccuracyNs the timing synchronization accuracy of GPS time, in nanoseconds.
   * GPS time-sync is ~40-100 ns.  Universally synchronized time among all vehicles
   * will result in all vehicles transmitting safety messages simultaneously, leading
   * to excessive wireless collisions.
   * \param ranges the expected transmission range, in m.
   * \param chAccessMode channel access mode (0=continuous; 1=switching)
   * \param txMaxDelay max delay prior to transmit
   */
  void Install (Ipv4InterfaceContainer & i,
                Time totalTime,          // seconds
                uint32_t wavePacketSize, // bytes
                Time waveInterval,       // seconds
                double gpsAccuracyNs,    // clock drift range in number of ns
                std::vector <double> ranges,          // m
                int chAccessMode,        // channel access mode (0=continuous; 1=switching)
                Time txMaxDelay);        // max delay prior to transmit

  /**
   * \brief Returns the WaveBsmStats instance
   * \return the WaveBsmStats instance
   */
  Ptr<WaveBsmStats> GetWaveBsmStats ();

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.  The Install() method should have previously been
   * called by the user.
   *
   * \param stream first stream index to use
   * \param c NodeContainer of the set of nodes for which the BsmApplication
   *          should be modified to use a fixed stream
   * \return the number of stream indices assigned by this helper
   */
  int64_t AssignStreams (NodeContainer c, int64_t stream);

  /**
   * \brief Returns the list of moving nove indicators
   * \return the list of moving node indicators
   */
  static std::vector<int>& GetNodesMoving ();

private:
  /**
   * Install an ns3::BsmApplication on the node
   *
   * \param node The node on which an BsmApplication will be installed.
   * \returns Ptr to the application installed.
   */
  Ptr<Application> InstallPriv (Ptr<Node> node) const;

  ObjectFactory m_factory; //!< Object factory.
  WaveBsmStats m_waveBsmStats; ///< wave BSM stats
  /// tx safety range squared, for optimization
  std::vector <double> m_txSafetyRangesSq;
  static std::vector<int> nodesMoving; ///< nodes moving
};

} // namespace ns3

#endif /* WAVE_BSM_HELPER_H*/
