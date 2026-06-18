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

#ifndef UAN_PHY_GEN_H
#define UAN_PHY_GEN_H


#include "uan-phy.h"
#include "ns3/traced-callback.h"
#include "ns3/nstime.h"
#include "ns3/device-energy-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/event-id.h"
#include <list>

namespace ns3 {

/**
 * \ingroup uan
 *
 * Default Packet Error Rate calculator for UanPhyGen
 *
 * Considers no error if SINR is > user defined threshold
 * (configured by an attribute).
 */
class UanPhyPerGenDefault : public UanPhyPer
{
public:
  /** Constructor */
  UanPhyPerGenDefault ();
  /** Destructor */
  virtual ~UanPhyPerGenDefault ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  virtual double CalcPer (Ptr<Packet> pkt, double sinrDb, UanTxMode mode);
private:
  double m_thresh;  //!< SINR threshold.

};  // class UanPhyPerGenDefault


/**
 * \ingroup uan
 *
 * Packet error rate calculation assuming WHOI Micromodem-like PHY (FH-FSK)
 *
 * Calculates PER assuming rate 1/2 convolutional code with
 * constraint length 9 with soft decision viterbi decoding and
 * a CRC capable of correcting 1 bit error.
 */
class UanPhyPerUmodem : public UanPhyPer
{
public:
  /** Constructor */
  UanPhyPerUmodem ();
  /** Destructor */
  virtual ~UanPhyPerUmodem ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Calculate the packet error probability based on
   * SINR at the receiver and a tx mode.
   *
   * This implementation uses calculations
   * for binary FSK modulation coded by a rate 1/2 convolutional code
   * with constraint length = 9 and a viterbi decoder and finally a CRC capable
   * of correcting one bit error.  These equations can be found in
   * the book, Digital Communications, by Proakis (any version I think).
   *
   * \param pkt Packet which is under consideration.
   * \param sinrDb SINR at receiver.
   * \param mode TX mode used to transmit packet.
   * \return Probability of packet error.
   */
  virtual double CalcPer (Ptr<Packet> pkt, double sinrDb, UanTxMode mode);

private:
  /**
   * Binomial coefficient
   *
   * \param n Pool size.
   * \param k Number of draws.
   * \return Binomial coefficient n choose k.
   */
  double NChooseK (uint32_t n, uint32_t k);

};  // class UanPhyPerUmodem


/**
 * \ingroup uan
 *
 * Packet error rate calculation for common tx modes based on UanPhyPerUmodem
 *
 * Calculates PER for common UanTxMode modulations, by deriving
 * PER from the BER taken from well known literature's formulas.
 */
class UanPhyPerCommonModes : public UanPhyPer
{
public:
  /** Constructor */
  UanPhyPerCommonModes ();
  /** Destructor */
  virtual ~UanPhyPerCommonModes ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Calculate the Packet ERror probability based on
   * SINR at the receiver and a tx mode.
   *
   * This implementation calculates PER for common UanTxMode modulations,
   * by deriving PER from the BER taken from literature's formulas.
   *
   * \param pkt Packet which is under consideration.
   * \param sinrDb SINR at receiver.
   * \param mode TX mode used to transmit packet.
   * \return Probability of packet error.
   */
  virtual double CalcPer (Ptr<Packet> pkt, double sinrDb, UanTxMode mode);

};  // class UanPhyPerCommonModes


/**
 * \ingroup uan
 *
 * Default SINR calculator for UanPhyGen.
 *
 * The default ignores mode data and assumes that all rxpower transmitted is
 * captured by the receiver, and that all signal power associated with
 * interfering packets affects SINR identically to additional ambient noise.
 */
class UanPhyCalcSinrDefault : public UanPhyCalcSinr
{

public:
  /** Constructor */
  UanPhyCalcSinrDefault ();
  /** Destructor */
  virtual ~UanPhyCalcSinrDefault ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Calculate the SINR value for a packet.
   *
   * This implementation simply adds all arriving signal power
   * and assumes it acts identically to additional noise.
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
                             ) const;

};  // class UanPhyCalcSinrDefault


/**
 * \ingroup uan
 *
 * WHOI Micromodem like FH-FSK model.
 *
 * Model of interference calculation for FH-FSK wherein all nodes
 * use an identical hopping pattern.  In this case, there is an (M-1)*SymbolTime
 * clearing time between symbols transmitted on the same frequency.
 * This clearing time combats ISI from channel delay spread and also has
 * a byproduct of possibly reducing interference from other transmitted packets.
 *
 * Thanks to Randall Plate for the latest model revision based on the following
 * papers:
 * <ul>
 * <li>Parrish, "System Design Considerations for Undersea Networks: Link and Multiple Access Protocols"
 * <li>Siderius, "Effects of Ocean Thermocline Variability on Noncoherent Underwater Acoustic Communications"
 * <li>Rao, "Channel Coding Techniques for Wireless Communications", ch 2
 * </ul>
 */
class UanPhyCalcSinrFhFsk : public UanPhyCalcSinr
{

public:
  /** Constructor */
  UanPhyCalcSinrFhFsk ();
  /** Destructor */
  virtual ~UanPhyCalcSinrFhFsk ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  virtual double CalcSinrDb (Ptr<Packet> pkt,
                             Time arrTime,
                             double rxPowerDb,
                             double ambNoiseDb,
                             UanTxMode mode,
                             UanPdp pdp,
                             const UanTransducer::ArrivalList &arrivalList
                             ) const;
private:
  uint32_t m_hops;  //!< Number of hops.

};  // class UanPhyCalcSinrFhFsk


/**
 * \ingroup uan
 *
 * Generic PHY model.
 *
 * This is a generic PHY class.  SINR and PER information
 * are controlled via attributes.  By adapting the SINR
 * and PER models to a specific situation, this PHY should
 * be able to model a wide variety of networks.
 */
class UanPhyGen : public UanPhy
{
public:
  /** Constructor */
  UanPhyGen ();
  /** Dummy destructor, see DoDispose */
  virtual ~UanPhyGen ();
  /**
   * Get the default transmission modes.
   *
   * \return The default mode list.
   */
  static UanModesList GetDefaultModes (void);

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  // Inherited methods
  virtual void SetEnergyModelCallback (DeviceEnergyModel::ChangeStateCallback cb);
  virtual void EnergyDepletionHandler (void);
  virtual void EnergyRechargeHandler (void);
  virtual void SendPacket (Ptr<Packet> pkt, uint32_t modeNum);
  virtual void RegisterListener (UanPhyListener *listener);
  virtual void StartRxPacket (Ptr<Packet> pkt, double rxPowerDb, UanTxMode txMode, UanPdp pdp);
  virtual void SetReceiveOkCallback (RxOkCallback cb);
  virtual void SetReceiveErrorCallback (RxErrCallback cb);
  virtual bool IsStateSleep (void);
  virtual bool IsStateIdle (void);
  virtual bool IsStateBusy (void);
  virtual bool IsStateRx (void);
  virtual bool IsStateTx (void);
  virtual bool IsStateCcaBusy (void);
  virtual void SetTxPowerDb (double txpwr);
  virtual void SetRxThresholdDb (double thresh);
  virtual void SetCcaThresholdDb (double thresh);
  virtual double GetTxPowerDb (void);
  virtual double GetRxThresholdDb (void);
  virtual double GetCcaThresholdDb (void);
  virtual Ptr<UanChannel> GetChannel (void) const;
  virtual Ptr<UanNetDevice> GetDevice (void) const;
  virtual Ptr<UanTransducer> GetTransducer (void);
  virtual void SetChannel (Ptr<UanChannel> channel);
  virtual void SetDevice (Ptr<UanNetDevice> device);
  virtual void SetMac (Ptr<UanMac> mac);
  virtual void SetTransducer (Ptr<UanTransducer> trans);
  virtual void NotifyTransStartTx (Ptr<Packet> packet, double txPowerDb, UanTxMode txMode);
  virtual void NotifyIntChange (void);
  virtual uint32_t GetNModes (void);
  virtual UanTxMode GetMode (uint32_t n);
  virtual Ptr<Packet> GetPacketRx (void) const;
  virtual void Clear (void);
  virtual void SetSleepMode (bool sleep);
  int64_t AssignStreams (int64_t stream);

private:
  /** List of Phy Listeners. */
  typedef std::list<UanPhyListener *> ListenerList;

  UanModesList m_modes;             //!< List of modes supported by this PHY.

  State m_state;                    //!< Phy state.
  ListenerList m_listeners;         //!< List of listeners.
  RxOkCallback m_recOkCb;           //!< Callback for packets received without error.
  RxErrCallback m_recErrCb;         //!< Callback for packets received with errors.
  Ptr<UanChannel> m_channel;        //!< Attached channel.
  Ptr<UanTransducer> m_transducer;  //!< Associated transducer.
  Ptr<UanNetDevice> m_device;       //!< Device hosting this Phy.
  Ptr<UanMac> m_mac;                //!< MAC layer.
  Ptr<UanPhyPer> m_per;             //!< Error model.
  Ptr<UanPhyCalcSinr> m_sinr;       //!< SINR calculator.

  double m_txPwrDb;                 //!< Transmit power.
  double m_rxThreshDb;              //!< Receive SINR threshold.
  double m_ccaThreshDb;             //!< CCA busy threshold.

  Ptr<Packet> m_pktRx;              //!< Received packet.
  Ptr<Packet> m_pktTx;              //!< Sent packet.
  double m_minRxSinrDb;             //!< Minimum receive SINR during packet reception.
  double m_rxRecvPwrDb;             //!< Receiver power.
  Time m_pktRxArrTime;              //!< Packet arrival time.
  UanPdp m_pktRxPdp;                //!< Power delay profile of packet.
  UanTxMode m_pktRxMode;            //!< Packet transmission mode at receiver.

  bool m_cleared;                   //!< Flag when we've been cleared.

  EventId m_txEndEvent;             //!< Tx event
  EventId m_rxEndEvent;             //!< Rx event

  /** Provides uniform random variables. */
  Ptr<UniformRandomVariable> m_pg;

  /** Energy model callback. */
  DeviceEnergyModel::ChangeStateCallback m_energyCallback;
  /** A packet destined for this Phy was received without error. */
  ns3::TracedCallback<Ptr<const Packet>, double, UanTxMode > m_rxOkLogger;
  /** A packet destined for this Phy was received with error. */
  ns3::TracedCallback<Ptr<const Packet>, double, UanTxMode > m_rxErrLogger;
  /** A packet was sent from this Phy. */
  ns3::TracedCallback<Ptr<const Packet>, double, UanTxMode > m_txLogger;

  /**
   * Calculate the SINR value for a packet.
   *
   * \param pkt Packet to calculate SINR for.
   * \param arrTime Arrival time of pkt.
   * \param rxPowerDb The received signal strength of the packet in dB re 1 uPa.
   * \param mode TX Mode of pkt.
   * \param pdp  Power delay profile of pkt.
   * \return The SINR in dB re 1 uPa.
   */
  double CalculateSinrDb (Ptr<Packet> pkt, Time arrTime, double rxPowerDb,
                          UanTxMode mode, UanPdp pdp);

  /**
   * Calculate interference power from overlapping packet arrivals, in dB.
   *
   * The "signal" packet power is excluded.  Use
   * GetInterferenceDb ( (Ptr<Packet>) 0) to treat all signals as
   * interference, for instance in calculating the CCA busy.
   *
   * \param pkt The arriving (signal) packet.
   * \return The total interference power, in dB.
   */
  double GetInterferenceDb (Ptr<Packet> pkt);
  /**
   * Convert dB to kilopascals.
   *
   *   \f[{\rm{kPa}} = {10^{\frac{{{\rm{dB}}}}{{10}}}}\f]
   *
   * \param db Signal level in dB.
   * \return Sound pressure in kPa.
   */
  double DbToKp (double db);
  /**
   * Convert kilopascals to dB.
   *
   *   \f[{\rm{dB}} = 10{\log _{10}}{\rm{kPa}}\f]
   *
   * \param kp Sound pressure in kPa.
   * \return Signal level in dB.
   */
  double KpToDb (double kp);
  /**
   * Event to process end of packet reception.
   *
   * \param pkt The packet.
   * \param rxPowerDb Received signal power.
   * \param txMode Transmission mode.
   */
  void RxEndEvent (Ptr<Packet> pkt, double rxPowerDb, UanTxMode txMode);
  /** Event to process end of packet transmission. */
  void TxEndEvent ();
  /**
   * Update energy source with new state.
   *
   * \param state The new Phy state.
   */
  void UpdatePowerConsumption (const State state);


  /** Call UanListener::NotifyRxStart on all listeners. */
  void NotifyListenersRxStart (void);
  /** Call UanListener::NotifyRxEndOk on all listeners. */
  void NotifyListenersRxGood (void);
  /** Call UanListener::NotifyRxEndError on all listeners. */
  void NotifyListenersRxBad (void);
  /** Call UanListener::NotifyCcaStart on all listeners. */
  void NotifyListenersCcaStart (void);
  /** Call UanListener::NotifyCcaEnd on all listeners. */
  void NotifyListenersCcaEnd (void);
  /**
   * Call UanListener::NotifyTxStart on all listeners.
   *
   * \param duration Duration of transmission.
   */
  void NotifyListenersTxStart (Time duration);
  /**
   * Call UanListener::NotifyTxEnd on all listeners.
   */
  void NotifyListenersTxEnd (void);

protected:
  virtual void DoDispose ();

};  // class UanPhyGen

} // namespace ns3

#endif /* UAN_PHY_GEN_H */
