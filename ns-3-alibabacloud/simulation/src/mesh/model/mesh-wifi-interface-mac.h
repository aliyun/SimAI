/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */

#ifndef MESH_WIFI_INTERFACE_MAC_H
#define MESH_WIFI_INTERFACE_MAC_H

#include <stdint.h>
#include <map>
#include "ns3/mac48-address.h"
#include "ns3/mgt-headers.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/nstime.h"
#include "ns3/wifi-mac.h"
#include "ns3/mesh-wifi-interface-mac-plugin.h"
#include "ns3/event-id.h"

namespace ns3 {

class UniformRandomVariable;
/**
 * \ingroup mesh
 *
 * \brief Basic MAC of mesh point Wi-Fi interface. Its function is extendable through plugins mechanism.
 *
 * Now only three output queues are used:
 *  - beacons (PIFS and no backoff),
 *  - background traffic,
 *  - management and priority traffic.
 *
 */
class MeshWifiInterfaceMac : public WifiMac
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  /// C-tor
  MeshWifiInterfaceMac ();
  /// D-tor
  virtual ~MeshWifiInterfaceMac ();

  // Inherited from WifiMac
  virtual void  Enqueue (Ptr<Packet> packet, Mac48Address to, Mac48Address from);
  virtual void  Enqueue (Ptr<Packet> packet, Mac48Address to);
  virtual bool  SupportsSendFrom () const;
  virtual void  SetLinkUpCallback (Callback<void> linkUp);
  virtual bool CanForwardPacketsTo (Mac48Address to) const;

  /// \name Each mesh point interface must know the mesh point address
  ///@{
  /**
   * Set the mesh point address
   * \param addr the mesh point address
   */
  void SetMeshPointAddress (Mac48Address addr);
  /**
   * Get the mesh point address
   * \return The mesh point address
   */
  Mac48Address GetMeshPointAddress () const;
  ///@}

  /// \name Beacons
  ///@{
  /**
   *  Set maximum initial random delay before first beacon
   *  \param interval maximum random interval
   */
  void SetRandomStartDelay (Time interval);
  /**
   *  Set interval between two successive beacons
   *  \param interval beacon interval
   */
  void SetBeaconInterval (Time interval);
  /// \return interval between two beacons
  Time GetBeaconInterval () const;
  /**
   * \brief Next beacon frame time
   * \return TBTT time
   *
   * This is supposed to be used by any entity managing beacon collision avoidance (e.g. Peer management protocol in 802.11s)
   */
  Time GetTbtt () const;
  /**
   * \brief Shift TBTT.
   * \param shift
   *
   * This is supposed to be used by any entity managing beacon collision avoidance (e.g. Peer management protocol in 802.11s)
   *
   * \attention User of ShiftTbtt () must take care to not shift it to the past.
   */
  void ShiftTbtt (Time shift);
  ///@}

  /**
   * Install plugin.
   *
   * \param plugin 
   *
   * \todo return unique ID to allow user to unregister plugins
   */
  void InstallPlugin (Ptr<MeshWifiInterfaceMacPlugin> plugin);

  /*
   * Channel center frequency = Channel starting frequency + 5 * channel_id (MHz),
   * where Starting channel frequency is standard-dependent as defined in IEEE 802.11-2007 17.3.8.3.2.
   *
   * Number of channels to use must be limited elsewhere.
   */

  /**
   * Current channel Id
   * \returns the frequency channel
   */
  uint16_t GetFrequencyChannel () const;
  /**
   * Switch frequency channel.
   *
   * \param new_id 
   */
  void SwitchFrequencyChannel (uint16_t new_id);

  /**
   * To be used by plugins sending management frames.
   *
   * \param frame the management frame
   * \param hdr the wifi MAC header
   */
  void SendManagementFrame (Ptr<Packet> frame, const WifiMacHeader& hdr);
  /**
   * Check supported rates.
   *
   * \param rates 
   * \return true if rates are supported
   */
  bool CheckSupportedRates (SupportedRates rates) const;
  /// \return list of supported bitrates
  SupportedRates GetSupportedRates () const;

  /// \name Metric Calculation routines:
  ///@{
  /**
   * Set the link metric callback
   * \param cb the callback
   */
  void SetLinkMetricCallback (Callback<uint32_t, Mac48Address, Ptr<MeshWifiInterfaceMac> > cb);
  /**
   * Get the link metric
   * \param peerAddress the peer address
   * \return The metric
   */
  uint32_t GetLinkMetric (Mac48Address peerAddress);
  ///@}

  /**
   * \brief Report statistics
   * \param os the output stream
   */
  void Report (std::ostream & os) const;
  /// Reset statistics function
  void ResetStats ();

  /**
   * Enable/disable beacons
   *
   * \param enable enable / disable flag
   */
  void SetBeaconGeneration (bool enable);
  /**
   * Finish configuration based on the WifiStandard being provided
   *
   * \param standard the WifiStandard being configured
   */
  virtual void ConfigureStandard (enum WifiStandard standard);
  /**
   * \param cwMin the minimum contention window size
   * \param cwMax the maximum contention window size
   *
   * This method is called to set the minimum and the maximum
   * contention window size.
   */
  virtual void ConfigureContentionWindow (uint32_t cwMin, uint32_t cwMax);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);
private:
  /**
   * Frame receive handler
   *
   * \param mpdu the received MPDU
   */
  void Receive (Ptr<WifiMacQueueItem> mpdu);
  /**
   * Send frame. Frame is supposed to be tagged by routing information.
   *
   * \param packet the packet to forward
   * \param from the from address
   * \param to the to address
   */
  void ForwardDown (Ptr<Packet> packet, Mac48Address from, Mac48Address to);
  /// Send beacon
  void SendBeacon ();
  /// Schedule next beacon
  void ScheduleNextBeacon ();
  /**
   * Get current beaconing status
   *
   * \returns true if beacon active
   */
  bool GetBeaconGeneration () const;
  /// Real d-tor
  virtual void DoDispose ();

private:
  typedef std::vector<Ptr<MeshWifiInterfaceMacPlugin> > PluginList; ///< PluginList typedef

  virtual void DoInitialize ();

  /// \name Mesh timing intervals
  ///@{
  /// whether beaconing is enabled
  bool m_beaconEnable;
  /// Beaconing interval.
  Time m_beaconInterval;
  /// Maximum delay before first beacon
  Time m_randomStart;
  /// Time for the next frame
  Time m_tbtt;
  ///@}

  /// Mesh point address
  Mac48Address m_mpAddress;

  /// "Timer" for the next beacon
  EventId m_beaconSendEvent;
  /// List of all installed plugins
  PluginList m_plugins;
  Callback<uint32_t, Mac48Address, Ptr<MeshWifiInterfaceMac> > m_linkMetricCallback; ///< linkMetricCallback
  /// Statistics:
  struct Statistics
  {
    uint16_t recvBeacons; ///< receive beacons
    uint32_t sentFrames; ///< sent frames
    uint32_t sentBytes; ///< sent bytes
    uint32_t recvFrames; ///< receive frames
    uint32_t recvBytes; ///< receive bytes
    /**
     * Print statistics.
     *
     * \param os 
     */
    void Print (std::ostream & os) const;
    /// constructor
    Statistics ();
  };
  Statistics m_stats; ///< statistics

  /// Current standard: needed to configure metric
  WifiStandard m_standard;

  /// Add randomness to beacon generation
  Ptr<UniformRandomVariable> m_coefficient;
};

} // namespace ns3

#endif /* MESH_WIFI_INTERFACE_MAC_H */
