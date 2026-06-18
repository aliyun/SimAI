/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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

#ifndef HALF_DUPLEX_IDEAL_PHY_H
#define HALF_DUPLEX_IDEAL_PHY_H


#include <ns3/spectrum-value.h>
#include <ns3/mobility-model.h>
#include <ns3/packet.h>
#include <ns3/nstime.h>
#include <ns3/net-device.h>
#include <ns3/spectrum-phy.h>
#include <ns3/spectrum-channel.h>
#include <ns3/spectrum-interference.h>
#include <ns3/data-rate.h>
#include <ns3/generic-phy.h>
#include <ns3/event-id.h>
#include <ns3/spectrum-signal-parameters.h>

namespace ns3 {

/**
 * \ingroup spectrum
 *
 * This PHY layer implementation realizes an ideal OFDM PHY which
 * transmits half-duplex (i.e., it can either receive or transmit at a
 * given time). The device is ideal in the sense that:
 * 1) it uses an error model based on the Shannon capacity, which
 * assumes ideal channel coding;
 * 2) it uses ideal signal acquisition, i.e., preamble detection and
 * synchronization are always successful
 * 3) it has no PHY layer overhead
 *
 * Being half duplex, if a RX is ongoing but a TX is requested, the RX
 * is aborted and the TX is started. Of course, no RX can be performed
 * while there is an ongoing TX.
 *
 * The use of OFDM is modeled by means of the Spectrum framework. By
 * calling the method SetTxPowerSpectralDensity(), the
 * user can specify how much of the spectrum is used, how many
 * subcarriers are used, and what power is allocated to each
 * subcarrier.
 *
 * The user can also specify the PHY rate
 * at which communications take place by using SetRate(). This is
 * equivalent to choosing a particular modulation and coding scheme.
 *
 * The use of the ShannonSpectrumErrorModel allows us to account for
 * the following aspects in determining whether a
 * transmission is successful or not:
 * - the PHY rate (trades off communication speed with reliability)
 * - the power spectral density (trade-off among total power consumed,
 * total bandwidth used (i.e., how much of the spectrum is occupied),
 * and communication reliability)
 * - the signal propagation
 *
 * This PHY model supports a single antenna model instance which is
 * used for both transmission and reception.  
 */
class HalfDuplexIdealPhy : public SpectrumPhy
{

public:
  HalfDuplexIdealPhy ();
  virtual ~HalfDuplexIdealPhy ();

  /**
   *  PHY states
   */
  enum State
  {
    IDLE, //!< Idle state
    TX,   //!< Transmitting state
    RX    //!< Receiving state
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited from SpectrumPhy
  void SetChannel (Ptr<SpectrumChannel> c);
  void SetMobility (Ptr<MobilityModel> m);
  void SetDevice (Ptr<NetDevice> d);
  Ptr<MobilityModel> GetMobility () const;
  Ptr<NetDevice> GetDevice () const;
  Ptr<const SpectrumModel> GetRxSpectrumModel () const;
  Ptr<Object> GetAntenna () const;
  void StartRx (Ptr<SpectrumSignalParameters> params);


  /**
   * \brief Set the Power Spectral Density of outgoing signals in power units
   * (Watt, Pascal...) per Hz.
   *
   * @param txPsd Tx Power Spectral Density
   */
  void SetTxPowerSpectralDensity (Ptr<SpectrumValue> txPsd);

  /**
   * \brief Set the Noise Power Spectral Density in power units
   * (Watt, Pascal...) per Hz.
   * @param noisePsd the Noise Power Spectral Density
   */
  void SetNoisePowerSpectralDensity (Ptr<const SpectrumValue> noisePsd);


  /**
   * Start a transmission
   *
   *
   * @param p the packet to be transmitted
   *
   * @return true if an error occurred and the transmission was not
   * started, false otherwise.
   */
  bool StartTx (Ptr<Packet> p);

  /**
   * Set the PHY rate to be used by this PHY.
   *
   * @param rate DataRate
   */
  void SetRate (DataRate rate);

  /**
   * Get the PHY rate to be used by this PHY.
   *
   * @return the PHY rate used by this PHY.
   */
  DataRate GetRate () const;

  /**
   * Set the callback for the end of a TX, as part of the
   * interconnections between the PHY and the MAC
   *
   * @param c the callback
   */
  void SetGenericPhyTxEndCallback (GenericPhyTxEndCallback c);

  /**
   * Set the callback for the start of RX, as part of the
   * interconnections between the PHY and the MAC
   *
   * @param c the callback
   */
  void SetGenericPhyRxStartCallback (GenericPhyRxStartCallback c);

  /**
   * set the callback for the end of a RX in error, as part of the
   * interconnections between the PHY and the MAC
   *
   * @param c the callback
   */
  void SetGenericPhyRxEndErrorCallback (GenericPhyRxEndErrorCallback c);

  /**
   * set the callback for the successful end of a RX, as part of the
   * interconnections between the PHY and the MAC
   *
   * @param c the callback
   */
  void SetGenericPhyRxEndOkCallback (GenericPhyRxEndOkCallback c);

  /** 
   * set the AntennaModel to be used
   * 
   * \param a the Antenna Model
   */
  void SetAntenna (Ptr<AntennaModel> a);

private:
  virtual void DoDispose (void);

  /**
   * Change the PHY state
   * \param newState new state
   */
  void ChangeState (State newState);
  /**
   * End the current Tx
   */
  void EndTx ();
  /**
   * About current Rx
   */
  void AbortRx ();
  /**
   * End current Rx
   */
  void EndRx ();

  EventId m_endRxEventId; //!< End Rx event

  Ptr<MobilityModel> m_mobility;  //!< Mobility model
  Ptr<AntennaModel> m_antenna;    //!< Antenna model
  Ptr<NetDevice> m_netDevice;     //!< NetDevice connected to theis phy
  Ptr<SpectrumChannel> m_channel; //!< Channel

  Ptr<SpectrumValue> m_txPsd;       //!< Tx power spectral density
  Ptr<const SpectrumValue> m_rxPsd; //!< Rx power spectral density
  Ptr<Packet> m_txPacket; //!< Tx packet
  Ptr<Packet> m_rxPacket; //!< Rx packet

  DataRate m_rate;  //!< Datarate
  State m_state;    //!< PHY state

  TracedCallback<Ptr<const Packet> > m_phyTxStartTrace; //!< Trace - Tx start
  TracedCallback<Ptr<const Packet> > m_phyTxEndTrace;   //!< Trace - Tx end
  TracedCallback<Ptr<const Packet> > m_phyRxStartTrace; //!< Trace - Rx start
  TracedCallback<Ptr<const Packet> > m_phyRxAbortTrace; //!< Trace - Rx abort
  TracedCallback<Ptr<const Packet> > m_phyRxEndOkTrace; //!< Trace - Tx end (ok)
  TracedCallback<Ptr<const Packet> > m_phyRxEndErrorTrace;  //!< Trace - Rx end (error)

  GenericPhyTxEndCallback        m_phyMacTxEndCallback;       //!< Callback - Tx end
  GenericPhyRxStartCallback      m_phyMacRxStartCallback;     //!< Callback - Rx start
  GenericPhyRxEndErrorCallback   m_phyMacRxEndErrorCallback;  //!< Callback - Rx error
  GenericPhyRxEndOkCallback      m_phyMacRxEndOkCallback;     //!< Callback - Rx end

  SpectrumInterference m_interference; //!< Received interference

};

}


#endif /* HALF_DUPLEX_IDEAL_PHY_H */
