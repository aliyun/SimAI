/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef WIFI_MAC_H
#define WIFI_MAC_H

#include <set>
#include <unordered_map>
#include "wifi-standards.h"
#include "wifi-remote-station-manager.h"
#include "qos-utils.h"
#include "ssid.h"

namespace ns3 {

class Txop;
class WifiNetDevice;
class QosTxop;
class WifiPsdu;
class MacRxMiddle;
class MacTxMiddle;
class WifiMacQueue;
class WifiMacQueueItem;
class HtConfiguration;
class VhtConfiguration;
class HeConfiguration;
class FrameExchangeManager;
class ChannelAccessManager;
class ExtendedCapabilities;

/**
 * Enumeration for type of station
 */
enum TypeOfStation
{
  STA,
  AP,
  ADHOC_STA,
  MESH,
  OCB
};

/**
 * \ingroup wifi
  * \enum WifiMacDropReason
  * \brief The reason why an MPDU was dropped
  */
enum WifiMacDropReason : uint8_t
{
  WIFI_MAC_DROP_FAILED_ENQUEUE = 0,
  WIFI_MAC_DROP_EXPIRED_LIFETIME,
  WIFI_MAC_DROP_REACHED_RETRY_LIMIT,
  WIFI_MAC_DROP_QOS_OLD_PACKET
};

typedef std::unordered_map <uint16_t /* staId */, Ptr<WifiPsdu> /* PSDU */> WifiPsduMap;

/**
 * \brief base class for all MAC-level wifi objects.
 * \ingroup wifi
 *
 * This class encapsulates all the low-level MAC functionality
 * DCA, EDCA, etc) and all the high-level MAC functionality
 * (association/disassociation state machines).
 *
 */
class WifiMac : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  WifiMac ();
  virtual ~WifiMac ();

  /**
   * Sets the device this PHY is associated with.
   *
   * \param device the device this PHY is associated with
   */
  void SetDevice (const Ptr<WifiNetDevice> device);
  /**
   * Return the device this PHY is associated with
   *
   * \return the device this PHY is associated with
   */
  Ptr<WifiNetDevice> GetDevice (void) const;

  /**
   * Get the Frame Exchange Manager
   *
   * \return the Frame Exchange Manager
   */
  Ptr<FrameExchangeManager> GetFrameExchangeManager (void) const;

  /**
   * Accessor for the Txop object
   *
   * \return a smart pointer to Txop
   */
  Ptr<Txop> GetTxop (void) const;
  /**
   * Accessor for a specified EDCA object
   *
   * \param ac the Access Category
   * \return a smart pointer to a QosTxop
   */
  Ptr<QosTxop> GetQosTxop (AcIndex ac) const;
  /**
   * Accessor for a specified EDCA object
   *
   * \param tid the Traffic ID
   * \return a smart pointer to a QosTxop
   */
  Ptr<QosTxop> GetQosTxop (uint8_t tid) const;
  /**
   * Get the wifi MAC queue of the (Qos)Txop associated with the given AC,
   * if such (Qos)Txop is installed, or a null pointer, otherwise.
   *
   * \param ac the given Access Category
   * \return the wifi MAC queue of the (Qos)Txop associated with the given AC,
   *         if such (Qos)Txop is installed, or a null pointer, otherwise
   */
  virtual Ptr<WifiMacQueue> GetTxopQueue (AcIndex ac) const;

   /**
   * This method is invoked by a subclass to specify what type of
   * station it is implementing. This is something that the channel
   * access functions need to know.
   *
   * \param type the type of station.
   */
  void SetTypeOfStation (TypeOfStation type);
  /**
   * Return the type of station.
   *
   * \return the type of station.
   */
  TypeOfStation GetTypeOfStation (void) const;

 /**
   * \param ssid the current SSID of this MAC layer.
   */
  void SetSsid (Ssid ssid);
  /**
   * \brief Sets the interface in promiscuous mode.
   *
   * Enables promiscuous mode on the interface. Note that any further
   * filtering on the incoming frame path may affect the overall
   * behavior.
   */
  void SetPromisc (void);
  /**
   * Enable or disable CTS-to-self feature.
   *
   * \param enable true if CTS-to-self is to be supported,
   *               false otherwise
   */
  void SetCtsToSelfSupported (bool enable);

  /**
   * \return the MAC address associated to this MAC layer.
   */
  Mac48Address GetAddress (void) const;
  /**
   * \return the SSID which this MAC layer is going to try to stay in.
   */
  Ssid GetSsid (void) const;
  /**
   * \param address the current address of this MAC layer.
   */
  virtual void SetAddress (Mac48Address address);
  /**
   * \return the BSSID of the network this device belongs to.
   */
  Mac48Address GetBssid (void) const;
  /**
   * \param bssid the BSSID of the network that this device belongs to.
   */
  void SetBssid (Mac48Address bssid);

  /**
   * Return true if packets can be forwarded to the given destination,
   * false otherwise.
   *
   * \param to the address to which the packet should be sent
   * \return whether packets can be forwarded to the given destination
   */
  virtual bool CanForwardPacketsTo (Mac48Address to) const = 0;
  /**
   * \param packet the packet to send.
   * \param to the address to which the packet should be sent.
   * \param from the address from which the packet should be sent.
   *
   * The packet should be enqueued in a TX queue, and should be
   * dequeued as soon as the DCF function determines that
   * access it granted to this MAC. The extra parameter "from" allows
   * this device to operate in a bridged mode, forwarding received
   * frames without altering the source address.
   */
  virtual void Enqueue (Ptr<Packet> packet, Mac48Address to, Mac48Address from);
  /**
   * \param packet the packet to send.
   * \param to the address to which the packet should be sent.
   *
   * The packet should be enqueued in a TX queue, and should be
   * dequeued as soon as the DCF function determines that
   * access it granted to this MAC.
   */
  virtual void Enqueue (Ptr<Packet> packet, Mac48Address to) = 0;
  /**
   * \return if this MAC supports sending from arbitrary address.
   *
   * The interface may or may not support sending from arbitrary address.
   * This function returns true if sending from arbitrary address is supported,
   * false otherwise.
   */
  virtual bool SupportsSendFrom (void) const;

  /**
   * \param phy the physical layer attached to this MAC.
   */
  virtual void SetWifiPhy (Ptr<WifiPhy> phy);
  /**
   * \return the physical layer attached to this MAC
   */
  Ptr<WifiPhy> GetWifiPhy (void) const;
  /**
   * Remove currently attached WifiPhy device from this MAC.
   */
  void ResetWifiPhy (void);

  /**
   * \param stationManager the station manager attached to this MAC.
   */
  void SetWifiRemoteStationManager (Ptr<WifiRemoteStationManager> stationManager);
  /**
   * \return the station manager attached to this MAC.
   */
  Ptr<WifiRemoteStationManager> GetWifiRemoteStationManager (void) const;

  /**
   * This type defines the callback of a higher layer that a
   * WifiMac(-derived) object invokes to pass a packet up the stack.
   *
   * \param packet the packet that has been received.
   * \param from the MAC address of the device that sent the packet.
   * \param to the MAC address of the device that the packet is destined for.
   */
  typedef Callback<void, Ptr<const Packet>, Mac48Address, Mac48Address> ForwardUpCallback;

  /**
   * \param upCallback the callback to invoke when a packet must be
   *        forwarded up the stack.
   */
  void SetForwardUpCallback (ForwardUpCallback upCallback);
  /**
   * \param linkUp the callback to invoke when the link becomes up.
   */
  virtual void SetLinkUpCallback (Callback<void> linkUp);
  /**
   * \param linkDown the callback to invoke when the link becomes down.
   */
  void SetLinkDownCallback (Callback<void> linkDown);
  /* Next functions are not pure virtual so non QoS WifiMacs are not
   * forced to implement them.
   */

  /**
   * Notify that channel has been switched.
   */
  virtual void NotifyChannelSwitching (void);

  /**
   * \param packet the packet being enqueued
   *
   * Public method used to fire a MacTx trace. Implemented for encapsulation purposes.
   * Note this trace indicates that the packet was accepted by the device only.
   * The packet may be dropped later (e.g. if the queue is full).
   */
  void NotifyTx (Ptr<const Packet> packet);
  /**
   * \param packet the packet being dropped
   *
   * Public method used to fire a MacTxDrop trace.
   * This trace indicates that the packet was dropped before it was queued for
   * transmission (e.g. when a STA is not associated with an AP).
   */
  void NotifyTxDrop (Ptr<const Packet> packet);
  /**
   * \param packet the packet we received
   *
   * Public method used to fire a MacRx trace. Implemented for encapsulation purposes.
   */
  void NotifyRx (Ptr<const Packet> packet);
  /**
   * \param packet the packet we received promiscuously
   *
   * Public method used to fire a MacPromiscRx trace. Implemented for encapsulation purposes.
   */
  void NotifyPromiscRx (Ptr<const Packet> packet);
  /**
   * \param packet the packet we received but is not destined for us
   *
   * Public method used to fire a MacRxDrop trace. Implemented for encapsulation purposes.
   */
  void NotifyRxDrop (Ptr<const Packet> packet);

  /**
   * \param standard the wifi standard to be configured
   *
   * This method completes the configuration process for a requested PHY standard.
   * Subclasses should implement this method to configure their DCF queues
   * according to the requested standard.
   */
  virtual void ConfigureStandard (WifiStandard standard);

  /**
   * \return pointer to HtConfiguration if it exists
   */
  Ptr<HtConfiguration> GetHtConfiguration (void) const;
  /**
   * \return pointer to VhtConfiguration if it exists
   */
  Ptr<VhtConfiguration> GetVhtConfiguration (void) const;
  /**
   * \return pointer to HeConfiguration if it exists
   */
  Ptr<HeConfiguration> GetHeConfiguration (void) const;

  /**
   * Return the extended capabilities of the device.
   *
   * \return the extended capabilities that we support
   */
  ExtendedCapabilities GetExtendedCapabilities (void) const;
  /**
   * Return the HT capabilities of the device.
   *
   * \return the HT capabilities that we support
   */
  HtCapabilities GetHtCapabilities (void) const;
  /**
   * Return the VHT capabilities of the device.
   *
   * \return the VHT capabilities that we support
   */
  VhtCapabilities GetVhtCapabilities (void) const;
  /**
   * Return the HE capabilities of the device.
   *
   * \return the HE capabilities that we support
   */
  HeCapabilities GetHeCapabilities (void) const;

  /**
   * Return whether the device supports QoS.
   *
   * \return true if QoS is supported, false otherwise
   */
  bool GetQosSupported () const;
  /**
   * Return whether the device supports ERP.
   *
   * \return true if ERP is supported, false otherwise
   */
  bool GetErpSupported () const;
  /**
   * Return whether the device supports DSSS.
   *
   * \return true if DSSS is supported, false otherwise
   */
  bool GetDsssSupported () const;
  /**
   * Return whether the device supports HT.
   *
   * \return true if HT is supported, false otherwise
   */
  bool GetHtSupported () const;
  /**
   * Return whether the device supports VHT.
   *
   * \return true if VHT is supported, false otherwise
   */
  bool GetVhtSupported () const;
  /**
   * Return whether the device supports HE.
   *
   * \return true if HE is supported, false otherwise
   */
  bool GetHeSupported () const;

  /**
   * Return the maximum A-MPDU size of the given Access Category.
   *
   * \param ac Access Category index
   * \return the maximum A-MPDU size
   */
  uint32_t GetMaxAmpduSize (AcIndex ac) const;
  /**
   * Return the maximum A-MSDU size of the given Access Category.
   *
   * \param ac Access Category index
   * \return the maximum A-MSDU size
   */
  uint16_t GetMaxAmsduSize (AcIndex ac) const;


protected:
  void DoInitialize () override;
  void DoDispose () override;

  /**
   * \param cwMin the minimum contention window size
   * \param cwMax the maximum contention window size
   *
   * This method is called to set the minimum and the maximum
   * contention window size.
   */
  virtual void ConfigureContentionWindow (uint32_t cwMin, uint32_t cwMax);

  /**
   * Enable or disable QoS support for the device. Construct a Txop object
   * or QosTxop objects accordingly. This method can only be called before
   * initialization.
   *
   * \param enable whether QoS is supported
   */
  void SetQosSupported (bool enable);

  /**
   * Enable or disable short slot time feature.
   *
   * \param enable true if short slot time is to be supported,
   *               false otherwise
   */
  void SetShortSlotTimeSupported (bool enable);
  /**
   * \return whether the device supports short slot time capability.
   */
  bool GetShortSlotTimeSupported (void) const;

  /**
   * Accessor for the AC_VO channel access function
   *
   * \return a smart pointer to QosTxop
   */
  Ptr<QosTxop> GetVOQueue (void) const;
  /**
   * Accessor for the AC_VI channel access function
   *
   * \return a smart pointer to QosTxop
   */
  Ptr<QosTxop> GetVIQueue (void) const;
  /**
   * Accessor for the AC_BE channel access function
   *
   * \return a smart pointer to QosTxop
   */
  Ptr<QosTxop> GetBEQueue (void) const;
  /**
   * Accessor for the AC_BK channel access function
   *
   * \return a smart pointer to QosTxop
   */
  Ptr<QosTxop> GetBKQueue (void) const;

  /**
   * This method acts as the MacRxMiddle receive callback and is
   * invoked to notify us that a frame has been received. The
   * implementation is intended to capture logic that is going to be
   * common to all (or most) derived classes. Specifically, handling
   * of Block Ack management frames is dealt with here.
   *
   * This method will need, however, to be overridden by derived
   * classes so that they can perform their data handling before
   * invoking the base version.
   *
   * \param mpdu the MPDU that has been received.
   */
  virtual void Receive (Ptr<WifiMacQueueItem> mpdu);
  /**
   * Forward the packet up to the device.
   *
   * \param packet the packet that we are forwarding up to the device
   * \param from the address of the source
   * \param to the address of the destination
   */
  void ForwardUp (Ptr<const Packet> packet, Mac48Address from, Mac48Address to);

  /**
   * This method can be called to de-aggregate an A-MSDU and forward
   * the constituent packets up the stack.
   *
   * \param mpdu the MPDU containing the A-MSDU.
   */
  virtual void DeaggregateAmsduAndForward (Ptr<WifiMacQueueItem> mpdu);

  Ptr<MacRxMiddle> m_rxMiddle;                      //!< RX middle (defragmentation etc.)
  Ptr<MacTxMiddle> m_txMiddle;                      //!< TX middle (aggregation etc.)
  Ptr<ChannelAccessManager> m_channelAccessManager; //!< channel access manager
  Ptr<FrameExchangeManager> m_feManager;            //!< Frame Exchange Manager
  Ptr<Txop> m_txop;                                 //!<  TXOP used for transmission of frames to non-QoS peers.

  Callback<void> m_linkUp;       //!< Callback when a link is up
  Callback<void> m_linkDown;     //!< Callback when a link is down


private:
  /// type conversion operator
  WifiMac (const WifiMac&);
  /**
   * assignment operator
   *
   * \param mac the WifiMac to assign
   * \returns the assigned value
   */
  WifiMac & operator= (const WifiMac& mac);

  /**
   * \param dcf the DCF to be configured
   * \param cwmin the minimum contention window for the DCF
   * \param cwmax the maximum contention window for the DCF
   * \param isDsss flag to indicate whether PHY is DSSS or HR/DSSS
   * \param ac the access category for the DCF
   *
   * Configure the DCF with appropriate values depending on the given access category.
   */
  void ConfigureDcf (Ptr<Txop> dcf, uint32_t cwmin, uint32_t cwmax, bool isDsss, AcIndex ac);

  /**
   * Configure PHY dependent parameters such as CWmin and CWmax.
   */
  void ConfigurePhyDependentParameters (void);

  /**
   * This method is a private utility invoked to configure the channel
   * access function for the specified Access Category.
   *
   * \param ac the Access Category of the queue to initialise.
   */
  void SetupEdcaQueue (AcIndex ac);

  /**
   * Create a Frame Exchange Manager depending on the supported version
   * of the standard.
   *
   * \param standard the supported version of the standard
   */
  void SetupFrameExchangeManager (WifiStandard standard);

  /**
   * Enable or disable ERP support for the device.
   *
   * \param enable whether ERP is supported
   */
  void SetErpSupported (bool enable);
  /**
   * Enable or disable DSSS support for the device.
   *
   * \param enable whether DSSS is supported
   */
  void SetDsssSupported (bool enable);

  /**
   * Set the block ack threshold for AC_VO.
   *
   * \param threshold the block ack threshold for AC_VO.
   */
  void SetVoBlockAckThreshold (uint8_t threshold);
  /**
   * Set the block ack threshold for AC_VI.
   *
   * \param threshold the block ack threshold for AC_VI.
   */
  void SetViBlockAckThreshold (uint8_t threshold);
  /**
   * Set the block ack threshold for AC_BE.
   *
   * \param threshold the block ack threshold for AC_BE.
   */
  void SetBeBlockAckThreshold (uint8_t threshold);
  /**
   * Set the block ack threshold for AC_BK.
   *
   * \param threshold the block ack threshold for AC_BK.
   */
  void SetBkBlockAckThreshold (uint8_t threshold);

  /**
   * Set VO block ack inactivity timeout.
   *
   * \param timeout the VO block ack inactivity timeout.
   */
  void SetVoBlockAckInactivityTimeout (uint16_t timeout);
  /**
   * Set VI block ack inactivity timeout.
   *
   * \param timeout the VI block ack inactivity timeout.
   */
  void SetViBlockAckInactivityTimeout (uint16_t timeout);
  /**
   * Set BE block ack inactivity timeout.
   *
   * \param timeout the BE block ack inactivity timeout.
   */
  void SetBeBlockAckInactivityTimeout (uint16_t timeout);
  /**
   * Set BK block ack inactivity timeout.
   *
   * \param timeout the BK block ack inactivity timeout.
   */
  void SetBkBlockAckInactivityTimeout (uint16_t timeout);

  /**
   * This Boolean is set \c true iff this WifiMac is to model
   * 802.11e/WMM style Quality of Service. It is exposed through the
   * attribute system.
   *
   * At the moment, this flag is the sole selection between QoS and
   * non-QoS operation for the STA (whether IBSS, AP, or
   * non-AP). Ultimately, we will want a QoS-enabled STA to be able to
   * fall back to non-QoS operation with a non-QoS peer. This'll
   * require further intelligence - i.e., per-association QoS
   * state. Having a big switch seems like a good intermediate stage,
   * however.
   */
  bool m_qosSupported;
  /**
   * This Boolean is set \c true iff this WifiMac is to model
   * 802.11g. It is exposed through the attribute system.
   */
  bool m_erpSupported;
  /**
   * This Boolean is set \c true iff this WifiMac is to model
   * 802.11b. It is exposed through the attribute system.
   */
  bool m_dsssSupported;

  bool m_shortSlotTimeSupported; ///< flag whether short slot time is supported
  bool m_ctsToSelfSupported;     ///< flag indicating whether CTS-To-Self is supported

  TypeOfStation m_typeOfStation; //!< the type of station

  Ptr<WifiNetDevice> m_device;                    //!< Pointer to the device
  Ptr<WifiPhy> m_phy;                             //!< Wifi PHY
  Ptr<WifiRemoteStationManager> m_stationManager; //!< Remote station manager (rate control, RTS/CTS/fragmentation thresholds etc.)

  Mac48Address m_address; //!< MAC address of this station
  Ssid m_ssid;            //!< Service Set ID (SSID)
  Mac48Address m_bssid;   //!< the BSSID

  /** This type defines a mapping between an Access Category index,
  and a pointer to the corresponding channel access function */
  typedef std::map<AcIndex, Ptr<QosTxop> > EdcaQueues;

  /** This is a map from Access Category index to the corresponding
  channel access function */
  EdcaQueues m_edca;

  uint16_t m_voMaxAmsduSize; ///< maximum A-MSDU size for AC_VO (in bytes)
  uint16_t m_viMaxAmsduSize; ///< maximum A-MSDU size for AC_VI (in bytes)
  uint16_t m_beMaxAmsduSize; ///< maximum A-MSDU size for AC_BE (in bytes)
  uint16_t m_bkMaxAmsduSize; ///< maximum A-MSDU size for AC_BK (in bytes)

  uint32_t m_voMaxAmpduSize; ///< maximum A-MPDU size for AC_VO (in bytes)
  uint32_t m_viMaxAmpduSize; ///< maximum A-MPDU size for AC_VI (in bytes)
  uint32_t m_beMaxAmpduSize; ///< maximum A-MPDU size for AC_BE (in bytes)
  uint32_t m_bkMaxAmpduSize; ///< maximum A-MPDU size for AC_BK (in bytes)

  ForwardUpCallback m_forwardUp; //!< Callback to forward packet up the stack

  /**
   * The trace source fired when packets come into the "top" of the device
   * at the L3/L2 transition, before being queued for transmission.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_macTxTrace;
  /**
   * The trace source fired when packets coming into the "top" of the device
   * are dropped at the MAC layer before being queued for transmission.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_macTxDropTrace;
  /**
   * The trace source fired for packets successfully received by the device
   * immediately before being forwarded up to higher layers (at the L2/L3
   * transition).  This is a promiscuous trace.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_macPromiscRxTrace;
  /**
   * The trace source fired for packets successfully received by the device
   * immediately before being forwarded up to higher layers (at the L2/L3
   * transition).  This is a non- promiscuous trace.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_macRxTrace;
  /**
   * The trace source fired when packets coming into the "top" of the device
   * are dropped at the MAC layer during reception.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_macRxDropTrace;

  TracedCallback<const WifiMacHeader &> m_txOkCallback;  ///< transmit OK callback
  TracedCallback<const WifiMacHeader &> m_txErrCallback; ///< transmit error callback

  /**
   * TracedCallback signature for MPDU drop events.
   *
   * \param reason the reason why the MPDU was dropped (\see WifiMacDropReason)
   * \param mpdu the dropped MPDU
   */
  typedef void (* DroppedMpduCallback)(WifiMacDropReason reason, Ptr<const WifiMacQueueItem> mpdu);

  /// TracedCallback for MPDU drop events typedef
  typedef TracedCallback<WifiMacDropReason, Ptr<const WifiMacQueueItem>> DroppedMpduTracedCallback;

  /**
   * This trace indicates that an MPDU was dropped for the given reason.
   */
  DroppedMpduTracedCallback m_droppedMpduCallback;

  /// TracedCallback for acked/nacked MPDUs typedef
  typedef TracedCallback<Ptr<const WifiMacQueueItem>> MpduTracedCallback;

  MpduTracedCallback m_ackedMpduCallback;  ///< ack'ed MPDU callback
  MpduTracedCallback m_nackedMpduCallback; ///< nack'ed MPDU callback

  /**
   * TracedCallback signature for MPDU response timeout events.
   *
   * \param reason the reason why the timer was started
   * \param mpdu the MPDU whose response was not received before the timeout
   * \param txVector the TXVECTOR used to transmit the MPDU
   */
  typedef void (* MpduResponseTimeoutCallback)(uint8_t reason, Ptr<const WifiMacQueueItem> mpdu,
                                               const WifiTxVector& txVector);

  /// TracedCallback for MPDU response timeout events typedef
  typedef TracedCallback<uint8_t, Ptr<const WifiMacQueueItem>, const WifiTxVector&> MpduResponseTimeoutTracedCallback;

  /**
   * MPDU response timeout traced callback.
   * This trace source is fed by a WifiTxTimer object.
   */
  MpduResponseTimeoutTracedCallback m_mpduResponseTimeoutCallback;

  /**
   * TracedCallback signature for PSDU response timeout events.
   *
   * \param reason the reason why the timer was started
   * \param psdu the PSDU whose response was not received before the timeout
   * \param txVector the TXVECTOR used to transmit the PSDU
   */
  typedef void (* PsduResponseTimeoutCallback)(uint8_t reason, Ptr<const WifiPsdu> psdu,
                                               const WifiTxVector& txVector);

  /// TracedCallback for PSDU response timeout events typedef
  typedef TracedCallback<uint8_t, Ptr<const WifiPsdu>, const WifiTxVector&> PsduResponseTimeoutTracedCallback;

  /**
   * PSDU response timeout traced callback.
   * This trace source is fed by a WifiTxTimer object.
   */
  PsduResponseTimeoutTracedCallback m_psduResponseTimeoutCallback;

  /**
   * TracedCallback signature for PSDU map response timeout events.
   *
   * \param reason the reason why the timer was started
   * \param psduMap the PSDU map for which not all responses were received before the timeout
   * \param missingStations the MAC addresses of the stations that did not respond
   * \param nTotalStations the total number of stations that had to respond
   */
  typedef void (* PsduMapResponseTimeoutCallback)(uint8_t reason, WifiPsduMap* psduMap,
                                                  const std::set<Mac48Address>* missingStations,
                                                  std::size_t nTotalStations);

  /// TracedCallback for PSDU map response timeout events typedef
  typedef TracedCallback<uint8_t, WifiPsduMap*, const std::set<Mac48Address>*, std::size_t> PsduMapResponseTimeoutTracedCallback;

  /**
   * PSDU map response timeout traced callback.
   * This trace source is fed by a WifiTxTimer object.
   */
  PsduMapResponseTimeoutTracedCallback m_psduMapResponseTimeoutCallback;
};

} //namespace ns3

#endif /* WIFI_MAC_H */

