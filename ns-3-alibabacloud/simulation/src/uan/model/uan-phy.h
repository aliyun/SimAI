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
 *         Andrea Sacco <andrea.sacco85@gmail.com>
 */


#ifndef UAN_PHY_H
#define UAN_PHY_H

#include "ns3/object.h"
#include "ns3/uan-mac.h"
#include "ns3/uan-tx-mode.h"
#include "ns3/uan-prop-model.h"
#include "ns3/uan-transducer.h"
#include "ns3/device-energy-model.h"

namespace ns3 {


/**
 * \ingroup uan
 *
 * Class used for calculating SINR of packet in UanPhy.
 *
 * Can be set to any derived class using attributes of UanPhy
 * to implement different models.
 */
class UanPhyCalcSinr : public Object
{
public:
  /**
   * Calculate the SINR value for a packet.
   *
   * \param pkt Packet to calculate SINR for.
   * \param arrTime Arrival time of pkt.
   * \param rxPowerDb The received signal strength of the packet in dB re 1 uPa.
   * \param ambNoiseDb Ambient channel noise in dB re 1 uPa.
   * \param mode TX Mode of pkt.
   * \param pdp  Power delay profile of pkt.
   * \param arrivalList  List of interfering arrivals given from Transducer.
   * \return The SINR in dB re 1 uPa.
   */
  virtual double CalcSinrDb (Ptr<Packet> pkt,
                             Time arrTime,
                             double rxPowerDb,
                             double ambNoiseDb,
                             UanTxMode mode,
                             UanPdp pdp,
                             const UanTransducer::ArrivalList &arrivalList
                             ) const = 0;
  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /** Clear all pointer references. */
  virtual void Clear (void);

  /**
   * Convert dB re 1 uPa to kilopascals.
   *
   * \param db dB value.
   * \return Value in kilopascals.
   */
  inline double DbToKp (double db) const
  {
    return std::pow (10, db / 10.0);
  }
  /**
   * Convert kilopascals to dB re 1 uPa.
   *
   * \param kp Value in kilopascals.
   * \return Valeu in dB re 1 uPa
   */
  inline double KpToDb (double kp) const
  {
    return 10 * std::log10 (kp);
  }

protected:
  virtual void DoDispose (void);

};  // class UanPhyCalcSinr

/**
 * \ingroup uan
 *
 * Calculate packet error probability, based on received SINR
 * and modulation (mode).
 *
 * Can be set in UanPhy via attributes.
 */
class UanPhyPer : public Object
{
public:
  /**
   * Calculate the packet error probability based on
   * SINR at the receiver and a tx mode.
   *
   * \param pkt Packet which is under consideration.
   * \param sinrDb SINR at receiver.
   * \param mode TX mode used to transmit packet.
   * \return Probability of packet error.
   */
  virtual double CalcPer (Ptr<Packet> pkt, double sinrDb, UanTxMode mode) = 0;

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  /** Clear all pointer references. */
  virtual void Clear (void);

protected:
  virtual void DoDispose (void);

};  // class UanPhyPer


/**
 * \ingroup uan
 *
 * Interface for PHY event listener.
 *
 * A class which implements this interface may register with Phy object
 * to receive notification of TX/RX/CCA events
 */
class UanPhyListener
{
public:
  /** Default destructor */
  virtual ~UanPhyListener ()
  {
  }
  /** Called when UanPhy begins receiving packet. */
  virtual void NotifyRxStart (void) = 0;
  /** Called when UanPhy finishes receiving packet without error. */
  virtual void NotifyRxEndOk (void) = 0;
  /** Called when UanPhy finishes receiving packet in error. */
  virtual void NotifyRxEndError (void) = 0;
  /** Called when UanPhy begins sensing channel is busy. */
  virtual void NotifyCcaStart (void) = 0;
  /** Called when UanPhy stops sensing channel is busy. */
  virtual void NotifyCcaEnd (void) = 0;
  /**
   * Called when transmission starts from Phy object.
   *
   * \param duration Duration of transmission.
   */
  virtual void NotifyTxStart (Time duration) = 0;
  /** Function called when Phy object finishes transmitting packet */
  virtual void NotifyTxEnd (void) = 0;
};  // class UanPhyListener

/**
 * \ingroup uan
 *
 * Base class for UAN Phy models.
 */
class UanPhy : public Object
{
public:
  /// Enum defining possible Phy states.
  enum State
  {
    IDLE,     //!< Idle state.
    CCABUSY,  //!< Channel busy.
    RX,       //!< Receiving.
    TX,       //!< Transmitting.
    SLEEP,    //!< Sleeping.
    DISABLED  //!< Disabled.
  };

  /**
   * Packet received successfully callback function type.
   *
   * \pname{arg1} Packet received successfully.
   * \pname{arg2} SNIR of packet.
   * \pname{arg3} Mode of packet.
   */
  typedef Callback<void, Ptr<Packet>, double, UanTxMode> RxOkCallback;

  /**
   * Packet receive error callback function type.
   *
   * \pname{arg1} Packet received successfully.
   * \pname{arg2} SNIR of packet.
   */
  typedef Callback<void, Ptr<Packet>, double > RxErrCallback;

  /**
   * TracedCallback signature for UanPhy packet send/receive events.
   *
   * \param [in] pkt The packet.
   * \param [in] sinr The SINR.
   * \param [in] mode The channel mode.
   */
  typedef void (* TracedCallback)(Ptr<const Packet> pkt, double sinr, UanTxMode mode);


  /**
   * Set the DeviceEnergyModel callback for UanPhy device.
   *
   * \param callback The DeviceEnergyModel change state callback.
   */
  virtual void SetEnergyModelCallback (DeviceEnergyModel::ChangeStateCallback callback) = 0;
  /**
   * Handle the energy depletion event.
   */
  virtual void EnergyDepletionHandler (void) = 0;
  /**
   * Handle the energy recharge event.
   */
  virtual void EnergyRechargeHandler (void) = 0;
  /**
   * Send a packet using a specific transmission mode.
   *
   * \param pkt  Packet to transmit.
   * \param modeNum  Index of mode in SupportedModes list to use for transmission.
   */
  virtual void SendPacket (Ptr<Packet> pkt, uint32_t modeNum) = 0;

  /**
   * Register a UanPhyListener to be notified of common UanPhy events.
   *
   * \param listener New listener to register.
   */
  virtual void RegisterListener (UanPhyListener *listener) = 0;

  /**
   * Packet arriving from channel:  i.e.  leading bit of packet has arrived.
   *
   * \param pkt Packet which is arriving.
   * \param rxPowerDb Signal power of incoming packet in dB re 1 uPa.
   * \param txMode Transmission mode defining modulation of incoming packet.
   * \param pdp Power delay profile of incoming packet.
   */
  virtual void StartRxPacket (Ptr<Packet> pkt, double rxPowerDb, UanTxMode txMode, UanPdp pdp) = 0;

  /**
   * Set the callback to be used when a packet is received without error.
   *
   * \param cb The callback.
   */
  virtual void SetReceiveOkCallback (RxOkCallback cb) = 0;

  /**
   * Set the callback to be used when a packet is received with errors.
   *
   * \param cb The callback.
   */
  virtual void SetReceiveErrorCallback (RxErrCallback cb) = 0;


  /**
   * Set the transmit power.
   *
   * \param txpwr Final output transmission power, in dB.
   */
  virtual void SetTxPowerDb (double txpwr) = 0;

  /**
   * Set the minimum SINR threshold to receive a packet without errors.
   *
   * \deprecated See UanPhyPer.
   *
   * \param thresh Threshold SINR for proper reception in dB re 1 uPa.
   */
  virtual void SetRxThresholdDb (double thresh) = 0;

  /**
   * Set the threshold for detecting channel busy.
   *
   * \param thresh Signal power threshold at receiver.
   */
  virtual void SetCcaThresholdDb (double thresh) = 0;


  /**
   * Get the current transmit power, in dB.
   *
   * \return The transmit power.
   */
  virtual double GetTxPowerDb (void) = 0;

  /**
   * Get the minimum received signal strength required
   * to receive a packet without errors.
   *
   * \return The minimum required signal strength, in dB.
   */
  virtual double GetRxThresholdDb (void) = 0;

  /**
   * Get the CCA threshold signal strength required to detect channel busy.
   *
   * \return The CCA threshold signal strength in dB.
   */
  virtual double GetCcaThresholdDb (void) = 0;
  /** \return True if Phy is in SLEEP state. */
  virtual bool IsStateSleep (void) = 0;
  /** \return True if Phy is in IDLE state. */
  virtual bool IsStateIdle (void) = 0;
  /** \return True if Phy is neither IDLE nor SLEEP. */
  virtual bool IsStateBusy (void) = 0;
  /** \return True if Phy is currently in receive mode. */
  virtual bool IsStateRx (void) = 0;
  /** \return True if Phy is busy transmitting. */
  virtual bool IsStateTx (void) = 0;
  /** \return True if Phy is in CCABUSY state. */
  virtual bool IsStateCcaBusy (void) = 0;

  /**
   * Get the attached channel.
   *
   * \return The channel.
   */
  virtual Ptr<UanChannel> GetChannel (void) const = 0;

  /**
   * Get the device hosting this Phy.
   *
   * \return The net device.
   */
  virtual Ptr<UanNetDevice> GetDevice (void) const = 0;

  /**
   * Attach to a channel.
   *
   * \param channel The channel to attach to.
   */
  virtual void SetChannel (Ptr<UanChannel> channel) = 0;

  /**
   * Set the device hosting this Phy.
   *
   * \param device The device.
   */
  virtual void SetDevice (Ptr<UanNetDevice> device) = 0;

  /**
   * Set the MAC forwarding messages to this Phy.
   *
   * \param mac The MAC.
   */
  virtual void SetMac (Ptr<UanMac> mac) = 0;

  /**
   * Called when a transmission is beginning
   * on the attached transducer.
   *
   * \param packet Packet that is beginning transmission.
   * \param txPowerDb Transmit power of packet.
   * \param txMode  Transmission mode of packet.
   */
  virtual void NotifyTransStartTx (Ptr<Packet> packet, double txPowerDb, UanTxMode txMode) = 0;

  /**
   * Called when there has been a change in the
   * amount of interference this node is experiencing
   * from other transmissions.
   */
  virtual void NotifyIntChange (void) = 0;

  /**
   * Attach a transducer to this Phy.
   *
   * \param trans The transducer.
   */
  virtual void SetTransducer (Ptr<UanTransducer> trans) = 0;

  /**
   * Get the attached transducer.
   *
   * \return The transducer.
   */
  virtual Ptr<UanTransducer> GetTransducer (void) = 0;

  /**
   * Get the number of transmission modes supported by this Phy.
   *
   * \return The number modes.
   */
  virtual uint32_t GetNModes (void) = 0;

  /**
   * Get a specific transmission mode.
   *
   * \param n The mode number.
   * \return The mode.
   */
  virtual UanTxMode GetMode (uint32_t n) = 0;

  /**
   * Get the packet currently being received.
   *
   * \warning Returns non-valid pointer if IsStateRx == false.
   * \return The packet.
   */
  virtual Ptr<Packet> GetPacketRx (void) const = 0;

  /** Clear all pointer references. */
  virtual void Clear (void) = 0;

  /**
   * Set the Phy SLEEP mode.
   *
   * \param sleep SLEEP on or off.
   */
  virtual void SetSleepMode (bool sleep) = 0;


  /**
   * Called when the transducer begins transmitting a packet.
   *
   * This fires a PhyTxBegin trace.  Implemented for encapsulation
   * purposes.
   *
   * \param packet The packet.
   */
  void NotifyTxBegin (Ptr<const Packet> packet);

  /**
   * Called when the transducer finishes transmitting a packet.
   *
   * This fires a PhyTxEnd trace.  Implemented for encapsulation
   * purposes.
   *
   * \param packet The packet.
   */
  void NotifyTxEnd (Ptr<const Packet> packet);

  /**
   * Called when the transducer attempts to transmit a new packet while
   * already transmitting a prior packet.
   *
   * This fires a PhyTxDrop trace.  Implemented for encapsulation
   * purposes.
   *
   * \param packet The packet.
   */
  void NotifyTxDrop (Ptr<const Packet> packet);

  /**
   * Called when the Phy begins to receive a packet.
   *
   * This fires a PhyRxBegin trace.  Implemented for encapsulation
   * purposes.
   *
   * \param packet The packet.
   */
  void NotifyRxBegin (Ptr<const Packet> packet);

  /**
   * Called when a packet is received without error.
   *
   * This fires a PhyRxEnd trace.  Implemented for encapsulation
   * purposes.
   *
   * \param packet The packet.
   */
  void NotifyRxEnd (Ptr<const Packet> packet);

  /**
   * Called when the Phy drops a packet.
   *
   * This fires a PhyRxDrop trace.  Implemented for encapsulation
   * purposes.
   *
   * \param packet The packet.
   */
  void NotifyRxDrop (Ptr<const Packet> packet);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream First stream index to use.
   * \return The number of stream indices assigned by this model.
   */
  virtual int64_t AssignStreams (int64_t stream) = 0;

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

private:
  /**
   * Trace source indicating a packet has begun transmitting
   * over the channel medium.
   *
   * \see class CallBackTraceSource
   */
  ns3::TracedCallback<Ptr<const Packet> > m_phyTxBeginTrace;

  /**
   * Trace source indicating a packet has been completely transmitted
   * over the channel.
   *
   * \see class CallBackTraceSource
   */
  ns3::TracedCallback<Ptr<const Packet> > m_phyTxEndTrace;

  /**
   * Trace source indicating a packet has been dropped by the device
   * during transmission.
   *
   * \see class CallBackTraceSource
   */
  ns3::TracedCallback<Ptr<const Packet> > m_phyTxDropTrace;

  /**
   * Trace source indicating a packet has begun being received
   * from the channel medium by the device.
   *
   * \see class CallBackTraceSource
   */
  ns3::TracedCallback<Ptr<const Packet> > m_phyRxBeginTrace;

  /**
   * Trace source indicating a packet has been completely received
   * from the channel medium by the device.
   *
   * \see class CallBackTraceSource
   */
  ns3::TracedCallback<Ptr<const Packet> > m_phyRxEndTrace;

  /**
   * Trace source indicating a packet has been dropped by the device
   * during reception.
   *
   * \see class CallBackTraceSource
   */
  ns3::TracedCallback<Ptr<const Packet> > m_phyRxDropTrace;

};  // class UanPhy

} // namespace ns3

#endif /* UAN_PHY_H */
