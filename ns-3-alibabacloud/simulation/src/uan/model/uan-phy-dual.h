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

#ifndef UAN_PHY_DUAL_H
#define UAN_PHY_DUAL_H

#include "ns3/uan-phy.h"



namespace ns3 {

class UanTxMode;
class UanModesList;

/**
 * Default SINR model for UanPhyDual
 *
 * Considers interfering packet power as additional ambient noise only
 * if there is overlap in frequency band as found from supplied UanTxMode.
 * If there is no overlap, then the packets are considered not to interfere.
 */
class UanPhyCalcSinrDual : public UanPhyCalcSinr
{
public:
  /** Constructor */
  UanPhyCalcSinrDual ();
  /** Destructor */
  virtual ~UanPhyCalcSinrDual ();
  
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

};  // class UanPhyCalcSinrDual

/**
 * \ingroup uan
 *
 * Two channel Phy.
 *
 * A class that wraps two generic UAN Phy layers (UanPhyGen) into a single PHY.
 * This is used to simulate two receivers (and transmitters) that use
 * the same front end hardware.  When attached to a UanTransducerHd,
 * this results in a net device able to transmit on one or two channels
 * simultaneously or receive on one or two channels simultaneously but
 * that cannot transmit and receive simultaneously.
 *
 * Many of the standard PHY functions here become ambiguous.  In most cases
 * information for "Phy1" are returned.
 */
class UanPhyDual : public UanPhy
{
public:
  /** Constructor */
  UanPhyDual ();
  /** Dummy destructor \see DoDispose */
  virtual ~UanPhyDual ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId ();

  // Inherited methods:
  virtual void SetEnergyModelCallback (DeviceEnergyModel::ChangeStateCallback callback);
  virtual void EnergyDepletionHandler (void);
  virtual void EnergyRechargeHandler (void);
  virtual void SendPacket (Ptr<Packet> pkt, uint32_t modeNum);

  /**
   * Register a UanPhyListener to be notified of common UanPhy events.
   *
   * \param listener New listener to register.
   *
   * \note You may receive duplicate
   * messages as underneath there are two generic phys here.
   * Each will notify of state changes independently.
   */
  virtual void RegisterListener (UanPhyListener *listener);
  virtual void StartRxPacket (Ptr<Packet> pkt, double rxPowerDb, UanTxMode txMode, UanPdp pdp);
  virtual void SetReceiveOkCallback (RxOkCallback cb);
  virtual void SetReceiveErrorCallback (RxErrCallback cb);
  virtual void SetTxPowerDb (double txpwr);
  virtual void SetRxThresholdDb (double thresh);
  virtual void SetCcaThresholdDb (double thresh);
  virtual double GetTxPowerDb (void);
  virtual double GetRxThresholdDb (void);
  virtual double GetCcaThresholdDb (void);
  virtual bool IsStateSleep (void);
  virtual bool IsStateIdle (void);
  virtual bool IsStateBusy (void);
  virtual bool IsStateRx (void);
  virtual bool IsStateTx (void);
  virtual bool IsStateCcaBusy (void);
  virtual Ptr<UanChannel> GetChannel (void) const;
  virtual Ptr<UanNetDevice> GetDevice (void) const;
  virtual void SetChannel (Ptr<UanChannel> channel);
  virtual void SetDevice (Ptr<UanNetDevice> device);
  virtual void SetMac (Ptr<UanMac> mac);
  virtual void NotifyTransStartTx (Ptr<Packet> packet, double txPowerDb, UanTxMode txMode);
  virtual void NotifyIntChange (void);
  virtual void SetTransducer (Ptr<UanTransducer> trans);
  virtual Ptr<UanTransducer> GetTransducer (void);
  virtual uint32_t GetNModes (void);
  virtual UanTxMode GetMode (uint32_t n);
  virtual void Clear (void);
  virtual void SetSleepMode ([[maybe_unused]] bool sleep)
  {
    /// \todo This method has to be implemented
  }
  int64_t AssignStreams (int64_t stream);
  Ptr<Packet> GetPacketRx (void) const;
  

  /** \copydoc UanPhy::IsStateIdle */
  bool IsPhy1Idle (void);
  /** \copydoc UanPhy::IsStateIdle */
  bool IsPhy2Idle (void);
  /** \copydoc UanPhy::IsStateRx */
  bool IsPhy1Rx (void);
  /** \copydoc UanPhy::IsStateRx */
  bool IsPhy2Rx (void);
  /** \copydoc UanPhy::IsStateTx */
  bool IsPhy1Tx (void);
  /** \copydoc UanPhy::IsStateTx */
  bool IsPhy2Tx (void);

  // Attribute getters and setters
  /** \copydoc ns3::UanPhy::GetCcaThresholdDb() */
  double GetCcaThresholdPhy1 (void) const;
  /** \copydoc UanPhy::GetCcaThresholdDb() */
  double GetCcaThresholdPhy2 (void) const;
  /** \copydoc UanPhy::SetCcaThresholdDb */
  void SetCcaThresholdPhy1 (double thresh);
  /** \copydoc UanPhy::SetCcaThresholdDb */
  void SetCcaThresholdPhy2 (double thresh);

  /** \copydoc UanPhy::GetTxPowerDb */
  double GetTxPowerDbPhy1 (void) const;
  /** \copydoc UanPhy::GetTxPowerDb */
  double GetTxPowerDbPhy2 (void) const;
  /** \copydoc UanPhy::SetTxPowerDb */
  void SetTxPowerDbPhy1 (double txpwr);
  /** \copydoc UanPhy::SetTxPowerDb */
  void SetTxPowerDbPhy2 (double txpwr);


  /**
   * Get the list of available modes.
   *
   * \return The mode list.
   */
  UanModesList GetModesPhy1 (void) const;
  /** \copydoc GetModesPhy1 */
  UanModesList GetModesPhy2 (void) const;

  /**
   * Set the available modes.
   * 
   * \param modes List of modes.
   */
  void SetModesPhy1 (UanModesList modes);
  /** \copydoc SetModesPhy1 */
  void SetModesPhy2 (UanModesList modes);

  /**
   * Get the error probability model.
   *
   * \return The error model.
   */
  Ptr<UanPhyPer> GetPerModelPhy1 (void) const;
  /** \copydoc GetPerModelPhy1() */
  Ptr<UanPhyPer> GetPerModelPhy2 (void) const;

  /**
   * Set the error probability model.
   *
   * \param per The error model.
   */
  void SetPerModelPhy1 (Ptr<UanPhyPer> per);
  /** \copydoc SetPerModelPhy1 */
  void SetPerModelPhy2 (Ptr<UanPhyPer> per);

  /**
   * Get the SINR calculator.
   *
   * \return The SINR calculator.
   */
  Ptr<UanPhyCalcSinr> GetSinrModelPhy1 (void) const;
  /** \copydoc GetSinrModelPhy1 */
  Ptr<UanPhyCalcSinr> GetSinrModelPhy2 (void) const;
  
  /**
   * Set the SINR calculator.
   *
   * \param calcSinr The SINR calculator.
   */
  void SetSinrModelPhy1 (Ptr<UanPhyCalcSinr> calcSinr);
  /** \copydoc SetSinrModelPhy1 */
  void SetSinrModelPhy2 (Ptr<UanPhyCalcSinr> calcSinr);

  
  /** \copydoc UanPhy::GetPacketRx */
  Ptr<Packet> GetPhy1PacketRx (void) const;
  /** \copydoc UanPhy::GetPacketRx */
  Ptr<Packet> GetPhy2PacketRx (void) const;


private:
  /** First Phy layer. */
  Ptr<UanPhy> m_phy1;
  /** Second Phy layer. */
  Ptr<UanPhy> m_phy2;

  /** A packet was received successfully. */
  ns3::TracedCallback<Ptr<const Packet>, double, UanTxMode > m_rxOkLogger;
  /** A packet was received unsuccessfully. */
  ns3::TracedCallback<Ptr<const Packet>, double, UanTxMode > m_rxErrLogger;
  /** A packet was sent from this Phy. */
  ns3::TracedCallback<Ptr<const Packet>, double, UanTxMode > m_txLogger;
  /** Callback when packet received without errors. */
  RxOkCallback m_recOkCb;
  /** Callback when packet received with errors. */
  RxErrCallback m_recErrCb;


  /**
   * Handle callback and logger for packets received without error.
   *
   * \param pkt The packet.
   * \param sinr The SINR.
   * \param mode The channel mode.
   */
  void RxOkFromSubPhy (Ptr<Packet> pkt, double sinr, UanTxMode mode);
  /**
   * Handle callback and logger for packets received with error.
   *
   * \param pkt The packet.
   * \param sinr The SINR.
   */
  void RxErrFromSubPhy (Ptr<Packet> pkt, double sinr);
  
protected:
  virtual void DoDispose ();

};  // class UanPhyDual

} // namespace ns3

#endif /* UAN_PHY_DUAL_H */
