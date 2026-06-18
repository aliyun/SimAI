/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author:
 *  Gary Pei <guangyu.pei@boeing.com>
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 */
#ifndef LR_WPAN_PHY_H
#define LR_WPAN_PHY_H

#include "lr-wpan-interference-helper.h"

#include <ns3/spectrum-phy.h>
#include <ns3/traced-callback.h>
#include <ns3/traced-value.h>
#include <ns3/event-id.h>

namespace ns3 {

class Packet;
class SpectrumValue;
class LrWpanErrorModel;
struct LrWpanSpectrumSignalParameters;
class MobilityModel;
class SpectrumChannel;
class SpectrumModel;
class AntennaModel;
class NetDevice;
class UniformRandomVariable;

/**
 * \ingroup lr-wpan
 *
 * Helper structure to manage the power measurement during ED.
 */
typedef struct
{
  double averagePower;    //!< Average measured power
  Time lastUpdate;        //!< Last update time
  Time measurementLength; //!< Total measurement period
} LrWpanEdPower;

/**
 * \ingroup lr-wpan
 *
 * This data structure provides the Bit rate and Symbol rate for a given channel
 * See IEEE802.15.4-2006 Table 1 and 2 in section 6.1.1 and 6.1.2
 */
typedef  struct
{
  double bitRate;    //!< bit rate
  double symbolRate; //!< symbol rate
} LrWpanPhyDataAndSymbolRates;

/**
 * \ingroup lr-wpan
 *
 * This data structure provides number of symbols for the PPDU headers: SHR and PHR
 * See IEEE802.15.4-2006 Figure 16, Table 19 and 20 in section 6.3
 */
typedef  struct
{
  double shrPreamble; //!< Number of symbols for the SHR preamble
  double shrSfd;      //!< Number of symbols for the SHR SFD
  double phr;         //!< Number of symbols for the PHR
} LrWpanPhyPpduHeaderSymbolNumber;

/**
 * \ingroup lr-wpan
 *
 * This Phy option will be used to index various Tables in IEEE802.15.4-2006
 */
typedef enum
{
  IEEE_802_15_4_868MHZ_BPSK         = 0,
  IEEE_802_15_4_915MHZ_BPSK         = 1,
  IEEE_802_15_4_868MHZ_ASK          = 2,
  IEEE_802_15_4_915MHZ_ASK          = 3,
  IEEE_802_15_4_868MHZ_OQPSK        = 4,
  IEEE_802_15_4_915MHZ_OQPSK        = 5,
  IEEE_802_15_4_2_4GHZ_OQPSK        = 6,
  IEEE_802_15_4_INVALID_PHY_OPTION  = 7
} LrWpanPhyOption;

/**
 * \ingroup lr-wpan
 *
 * IEEE802.15.4-2006 PHY Emumerations Table 18
 * in section 6.2.3
 */
typedef enum
{
  IEEE_802_15_4_PHY_BUSY  = 0x00,
  IEEE_802_15_4_PHY_BUSY_RX = 0x01,
  IEEE_802_15_4_PHY_BUSY_TX = 0x02,
  IEEE_802_15_4_PHY_FORCE_TRX_OFF = 0x03,
  IEEE_802_15_4_PHY_IDLE = 0x04,
  IEEE_802_15_4_PHY_INVALID_PARAMETER = 0x05,
  IEEE_802_15_4_PHY_RX_ON = 0x06,
  IEEE_802_15_4_PHY_SUCCESS = 0x07,
  IEEE_802_15_4_PHY_TRX_OFF = 0x08,
  IEEE_802_15_4_PHY_TX_ON = 0x09,
  IEEE_802_15_4_PHY_UNSUPPORTED_ATTRIBUTE = 0xa,
  IEEE_802_15_4_PHY_READ_ONLY = 0xb,
  IEEE_802_15_4_PHY_UNSPECIFIED = 0xc // all cases not covered by ieee802.15.4
} LrWpanPhyEnumeration;

namespace TracedValueCallback
{
/**
 * \ingroup lr-wpan
 * TracedValue callback signature for LrWpanPhyEnumeration.
 *
 * \param [in] oldValue original value of the traced variable
 * \param [in] newValue new value of the traced variable
 */
  typedef void (* LrWpanPhyEnumeration)(LrWpanPhyEnumeration oldValue,
                                        LrWpanPhyEnumeration newValue);
}  // namespace TracedValueCallback

/**
 * \ingroup lr-wpan
 *
 * IEEE802.15.4-2006 PHY PIB Attribute Identifiers Table 23 in section 6.4.2
 */
typedef enum
{
  phyCurrentChannel = 0x00,
  phyChannelsSupported = 0x01,
  phyTransmitPower = 0x02,
  phyCCAMode = 0x03,
  phyCurrentPage = 0x04,
  phyMaxFrameDuration = 0x05,
  phySHRDuration = 0x06,
  phySymbolsPerOctet = 0x07
} LrWpanPibAttributeIdentifier;

/**
 * \ingroup lr-wpan
 *
 * IEEE802.15.4-2006 PHY PIB Attributes Table 23 in section 6.4.2
 */
typedef struct
{
  uint8_t phyCurrentChannel;         //!< The RF channel to use
  uint32_t phyChannelsSupported[32]; //!< BitField representing the available channels supported by a channel page.
  uint8_t phyTransmitPower;          //!< 2 MSB: tolerance on the transmit power, 6 LSB: Tx power in dBm relative to 1mW (signed int in 2-complement format)
  uint8_t phyCCAMode;                //!< CCA mode
  uint32_t phyCurrentPage;           //!< Current channel page
  uint32_t phyMaxFrameDuration;      //!< The maximum number of symbols in a frame
  uint32_t phySHRDuration;           //!< The duration of the synchronization header (SHR) in symbols
  double phySymbolsPerOctet;         //!< The number of symbols per octet
} LrWpanPhyPibAttributes;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PdDataIndication
 *
 *  @param psduLength number of bytes in the PSDU
 *  @param p the packet to be transmitted
 *  @param lqi Link quality (LQI) value measured during reception of the PPDU
 */
typedef Callback< void, uint32_t, Ptr<Packet>, uint8_t > PdDataIndicationCallback;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PdDataConfirm
 *
 * @param status the status to be transmitted
 */
typedef Callback< void, LrWpanPhyEnumeration > PdDataConfirmCallback;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PlmeCcaConfirm
 *
 * @param status the status of CCA
 */
typedef Callback< void, LrWpanPhyEnumeration > PlmeCcaConfirmCallback;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PlmeEdConfirm
 *
 * @param status the status of ED
 * @param energyLevel the energy level of ED
 */
typedef Callback< void, LrWpanPhyEnumeration,uint8_t > PlmeEdConfirmCallback;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PlmeGetAttributeConfirm
 *
 * @param status the status of PlmeGetAttributeRequest
 * @param id the identifier of attribute
 * @param attribute the pointer to attribute struct
 */
typedef Callback< void, LrWpanPhyEnumeration,
                  LrWpanPibAttributeIdentifier,
                  LrWpanPhyPibAttributes* > PlmeGetAttributeConfirmCallback;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PlmeSetTRXStateConfirm
 *
 * @param status the status of PlmeSetTRXStateRequest
 */
typedef Callback< void, LrWpanPhyEnumeration > PlmeSetTRXStateConfirmCallback;

/**
 * \ingroup lr-wpan
 *
 * This method implements the PD SAP: PlmeSetAttributeConfirm
 *
 * @param status the status of PlmeSetAttributeRequest
 * @param id the identifier of attribute
 */
typedef Callback< void, LrWpanPhyEnumeration,
                  LrWpanPibAttributeIdentifier > PlmeSetAttributeConfirmCallback;

/**
 * \ingroup lr-wpan
 *
 * Make LrWpanPhy a SpectrumPhy so we can enable the eventual modeling of
 * device interference
 */
class LrWpanPhy : public SpectrumPhy
{

public:
  /**
   * Get the type ID.
   *
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * The maximum packet size accepted by the PHY.
   * See Table 22 in section 6.4.1 of IEEE 802.15.4-2006
   */
  static const uint32_t aMaxPhyPacketSize;

  /**
   * The turnaround time for switching the transceiver from RX to TX or vice
   * versa.
   * See Table 22 in section 6.4.1 of IEEE 802.15.4-2006
   */
  static const uint32_t aTurnaroundTime;

  /**
   * Default constructor.
   */
  LrWpanPhy (void);
  virtual ~LrWpanPhy (void);

  // inherited from SpectrumPhy
  void SetMobility (Ptr<MobilityModel> m);
  Ptr<MobilityModel> GetMobility (void) const;
  void SetChannel (Ptr<SpectrumChannel> c);

  /**
   * Get the currently attached channel.
   *
   * \return the channel
   */
  Ptr<SpectrumChannel> GetChannel (void);
  void SetDevice (Ptr<NetDevice> d);
  Ptr<NetDevice> GetDevice (void) const;

  /**
   * Set the attached antenna.
   *
   * \param a the antenna
   */
  void SetAntenna (Ptr<AntennaModel> a);
  Ptr<Object> GetAntenna (void) const;
  virtual Ptr<const SpectrumModel> GetRxSpectrumModel (void) const;

  /**
   * Set the Power Spectral Density of outgoing signals in W/Hz.
   *
   * @param txPsd the Power Spectral Density value
   */
  void SetTxPowerSpectralDensity (Ptr<SpectrumValue> txPsd);

  /**
   * Set the noise power spectral density.
   *
   * @param noisePsd the Noise Power Spectral Density in power units
   * (Watt, Pascal...) per Hz.
   */
  void SetNoisePowerSpectralDensity (Ptr<const SpectrumValue> noisePsd);

  /**
   * Get the noise power spectral density.
   *
   * @return the Noise Power Spectral Density
   */
  Ptr<const SpectrumValue> GetNoisePowerSpectralDensity (void);

  /**
    * Notify the SpectrumPhy instance of an incoming waveform.
    *
    * @param params the SpectrumSignalParameters associated with the incoming waveform
    */
  virtual void StartRx (Ptr<SpectrumSignalParameters> params);

  /**
   *  IEEE 802.15.4-2006 section 6.2.1.1
   *  PD-DATA.request
   *  Request to transfer MPDU from MAC (transmitting)
   *  @param psduLength number of bytes in the PSDU
   *  @param p the packet to be transmitted
   */
  void PdDataRequest (const uint32_t psduLength, Ptr<Packet> p);

  /**
   *  IEEE 802.15.4-2006 section 6.2.2.1
   *  PLME-CCA.request
   *  Perform a CCA per section 6.9.9
   */
  void PlmeCcaRequest (void);

  /**
   *  IEEE 802.15.4-2006 section 6.2.2.3
   *  PLME-ED.request
   *  Perform an ED per section 6.9.7
   */
  void PlmeEdRequest (void);

  /**
   *  IEEE 802.15.4-2006 section 6.2.2.5
   *  PLME-GET.request
   *  Get attributes per definition from Table 23 in section 6.4.2
   *  @param id the attributed identifier
   */
  void PlmeGetAttributeRequest (LrWpanPibAttributeIdentifier id);

  /**
   *  IEEE 802.15.4-2006 section 6.2.2.7
   *  PLME-SET-TRX-STATE.request
   *  Set PHY state
   *  @param state in RX_ON,TRX_OFF,FORCE_TRX_OFF,TX_ON
   */
  void PlmeSetTRXStateRequest (LrWpanPhyEnumeration state);

  /**
   *  IEEE 802.15.4-2006 section 6.2.2.9
   *  PLME-SET.request
   *  Set attributes per definition from Table 23 in section 6.4.2
   *  @param id the attributed identifier
   *  @param attribute the attribute value
   */
  void PlmeSetAttributeRequest (LrWpanPibAttributeIdentifier id, LrWpanPhyPibAttributes* attribute);

  /**
   * set the callback for the end of a RX, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implements PD Indication SAP.
   * @param c the callback
   */
  void SetPdDataIndicationCallback (PdDataIndicationCallback c);

  /**
   * set the callback for the end of a TX, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implements PD SAP.
   * @param c the callback
   */
  void SetPdDataConfirmCallback (PdDataConfirmCallback c);

  /**
   * set the callback for the end of a CCA, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implement PLME CCA confirm SAP
   * @param c the callback
   */
  void SetPlmeCcaConfirmCallback (PlmeCcaConfirmCallback c);

  /**
   * set the callback for the end of an ED, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implement PLME ED confirm SAP
   * @param c the callback
   */
  void SetPlmeEdConfirmCallback (PlmeEdConfirmCallback c);

  /**
   * set the callback for the end of an GetAttribute, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implement PLME GetAttribute confirm SAP
   * @param c the callback
   */
  void SetPlmeGetAttributeConfirmCallback (PlmeGetAttributeConfirmCallback c);

  /**
   * set the callback for the end of an SetTRXState, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implement PLME SetTRXState confirm SAP
   * @param c the callback
   */
  void SetPlmeSetTRXStateConfirmCallback (PlmeSetTRXStateConfirmCallback c);

  /**
   * set the callback for the end of an SetAttribute, as part of the
   * interconnections betweenthe PHY and the MAC. The callback
   * implement PLME SetAttribute confirm SAP
   * @param c the callback
   */
  void SetPlmeSetAttributeConfirmCallback (PlmeSetAttributeConfirmCallback c);

  /**
   * implement PLME SetAttribute confirm SAP
   * bit rate is in kbit/s.  Symbol rate is in ksymbol/s.
   * @param isData is true for data rate or false for symbol rate
   * @return the rate value of this PHY
   */
  double GetDataOrSymbolRate (bool isData);

  /**
   * set the error model to use
   *
   * @param e pointer to LrWpanErrorModel to use
   */
  void SetErrorModel (Ptr<LrWpanErrorModel> e);

  /**
   * get the error model in use
   *
   * @return pointer to LrWpanErrorModel in use
   */
  Ptr<LrWpanErrorModel> GetErrorModel (void) const;

  /**
   * Get the duration of the SHR (preamble and SFD) in symbols, depending on
   * the currently selected channel.
   *
   * \return the SHR duration in symbols
   */
  uint64_t GetPhySHRDuration (void) const;

  /**
   * Get the number of symbols per octet, depending on the currently selected
   * channel.
   *
   * \return the number of symbols per octet
   */
  double GetPhySymbolsPerOctet (void) const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams that have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

  /**
   * TracedCallback signature for Trx state change events.
   *
   * \param [in] time The time of the state change.
   * \param [in] oldState The old state.
   * \param [in] newState The new state.
   * \deprecated The LrWpanPhyEnumeration state is now accessible as the
   * TracedValue \c TrxStateValue.  The \c TrxState TracedCallback will
   * be removed in a future release.
   */
  typedef void (* StateTracedCallback)
    (Time time, LrWpanPhyEnumeration oldState, LrWpanPhyEnumeration newState);

protected:
  /**
   * The data and symbol rates for the different PHY options.
   * See Table 2 in section 6.1.2 IEEE 802.15.4-2006.
   * Bit rate is in kbit/s.  Symbol rate is in ksymbol/s.
   */
  static const LrWpanPhyDataAndSymbolRates dataSymbolRates[7];
  /**
   * The preamble, SFD, and PHR lengths in symbols for the different PHY options.
   * See Table 19 and Table 20 in section 6.3 IEEE 802.15.4-2006
   */
  static const LrWpanPhyPpduHeaderSymbolNumber ppduHeaderSymbolNumbers[7];

private:
  /**
   * The second is true if the first is flagged as error/invalid.
   */
  typedef std::pair<Ptr<Packet>, bool>  PacketAndStatus;

  // Inherited from Object.
  virtual void DoDispose (void);

  /**
   * Change the PHY state to the given new state, firing the state change trace.
   *
   * \param newState the new state
   */
  void ChangeTrxState (LrWpanPhyEnumeration newState);

  /**
   * Configure the PHY option according to the current channel and channel page.
   * See IEEE 802.15.4-2006, section 6.1.2, Table 2.
   */
  void SetMyPhyOption (void);

  /**
   * Get the currently configured PHY option.
   * See IEEE 802.15.4-2006, section 6.1.2, Table 2.
   *
   * \return the PHY option
   */
  LrWpanPhyOption GetMyPhyOption (void);

  /**
   * Finish the transmission of a frame. This is called at the end of a frame
   * transmission, applying possibly pending PHY state changes and fireing the
   * appropriate trace sources and confirm callbacks to the MAC.
   */
  void EndTx (void);

  /**
   * Check if the interference destroys a frame currently received. Called
   * whenever a change in interference is detected.
   */
  void CheckInterference (void);

  /**
   * Finish the reception of a frame. This is called at the end of a frame
   * reception, applying possibly pending PHY state changes and fireing the
   * appropriate trace sources and indication callbacks to the MAC. A frame
   * destroyed by noise/interference is dropped here, but not during reception.
   * This method is also called for every packet which only contributes to
   * interference.
   *
   * \param params signal parameters of the packet
   */
  void EndRx (Ptr<SpectrumSignalParameters> params);

  /**
   * Cancel an ongoing ED procedure. This is called when the transceiver is
   * switched off or set to TX mode. This calls the appropriate confirm callback
   * of the MAC.
   *
   * \param state the new state which is the cause for canceling ED
   */
  void CancelEd (LrWpanPhyEnumeration state);

  /**
   * Called at the end of the ED procedure. The average energy detected is
   * reported to the MAC.
   */
  void EndEd (void);

  /**
   * Called at the end of the CCA. The channel condition (busy or idle) is
   * reported to the MAC or CSMA/CA.
   */
  void EndCca (void);

  /**
   * Called after applying a deferred transceiver state switch. The result of
   * the state switch is reported to the MAC.
   */
  void EndSetTRXState (void);

  /**
   * Calculate the time required for sending the given packet, including
   * preamble, SFD and PHR.
   *
   * \param packet the packet for which the transmission time should be calculated
   * \return the time required for transmitting the packet
   */
  Time CalculateTxTime (Ptr<const Packet> packet);

  /**
   * Calculate the time required for sending the PPDU header, that is the
   * preamble, SFD and PHR.
   * \returns The time required for sending the PPDU header.
   */
  Time GetPpduHeaderTxTime (void);

  /**
   * Check if the given channel is supported by the PHY.
   *
   * \param channel the channel to check
   * \return true, if the channel is supported, false otherwise
   */
  bool ChannelSupported (uint8_t channel);

  /**
   * Check if the PHY is busy, which is the case if the PHY is currently sending
   * or receiving a frame.
   *
   * \return true, if the PHY is busy, false otherwise
   */
  bool PhyIsBusy (void) const;

  // Trace sources
  /**
   * The trace source fired when a packet begins the transmission process on
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyTxBeginTrace;

  /**
   * The trace source fired when a packet ends the transmission process on
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyTxEndTrace;

  /**
   * The trace source fired when the phy layer drops a packet as it tries
   * to transmit it.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyTxDropTrace;

  /**
   * The trace source fired when a packet begins the reception process from
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyRxBeginTrace;

  /**
   * The trace source fired when a packet ends the reception process from
   * the medium.  Second quantity is received SINR.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet>, double > m_phyRxEndTrace;

  /**
   * The trace source fired when the phy layer drops a packet it has received.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyRxDropTrace;

  /**
   * The trace source fired when the phy layer changes the transceiver state.
   *
   * \see class CallBackTraceSource
   * \deprecated The LrWpanPhyEnumeration state is now accessible as the
   * TracedValue \c TrxStateValue.  This TracedCallback will
   * be removed in a future release.
   */
  TracedCallback<Time, LrWpanPhyEnumeration, LrWpanPhyEnumeration> m_trxStateLogger;

  /**
   * Calculates the nominal transmit power of the device in decibels relative to 1 mW
   * according to the representation of the PIB attribute phyTransmitPower.
   *
   * The phyTransmitPower (table 23 of IEEE 802.15.4-2006) is a 8-bit bitmap, stored
   * as a `uint8_t`. The bitmap has the following meaning:
   * The 2 MSBs represent the tolerance on the transmit power.
   * The 6 LSBs represent a signed integer in twos-complement format, corresponding
   * to the nominal transmit power of the device in decibels relative to 1 mW.
   *
   * This function extracts the 6 LSBs corresponding to the nominal transmit power and
   * converts it to a `int8_t`.
   *
   * \param phyTransmitPower the PIB attribute
   * \return The nominal transmit power in dBm.
   */
  int8_t GetNominalTxPowerFromPib (uint8_t phyTransmitPower);

  /**
   * The mobility model used by the PHY.
   */
  Ptr<MobilityModel> m_mobility;

  /**
   * The configured net device.
   */
  Ptr<NetDevice> m_device;

  /**
   * The channel attached to this transceiver.
   */
  Ptr<SpectrumChannel> m_channel;

  /**
   * The antenna used by the transceiver.
   */
  Ptr<AntennaModel> m_antenna;

  /**
   * The transmit power spectral density.
   */
  Ptr<SpectrumValue> m_txPsd;

  /**
   * The spectral density for for the noise.
   */
  Ptr<const SpectrumValue> m_noise;

  /**
   * The error model describing the bit and packet error rates.
   */
  Ptr<LrWpanErrorModel> m_errorModel;

  /**
   * The current PHY PIB attributes.
   */
  LrWpanPhyPibAttributes m_phyPIBAttributes;

  // State variables
  /**
   * The current transceiver state.
   */
  TracedValue<LrWpanPhyEnumeration> m_trxState;

  /**
   * The next pending state to applied after the current action of the PHY is
   * completed.
   */
  LrWpanPhyEnumeration m_trxStatePending;

  // Callbacks
  /**
   * This callback is used to notify incoming packets to the MAC layer.
   * See IEEE 802.15.4-2006, section 6.2.1.3.
   */
  PdDataIndicationCallback m_pdDataIndicationCallback;

  /**
   * This callback is used to report packet transmission status to the MAC layer.
   * See IEEE 802.15.4-2006, section 6.2.1.2.
   */
  PdDataConfirmCallback m_pdDataConfirmCallback;

  /**
   * This callback is used to report CCA status to the MAC or CSMA/CA.
   * See IEEE 802.15.4-2006, section 6.2.2.2.
   */
  PlmeCcaConfirmCallback m_plmeCcaConfirmCallback;

  /**
   * This callback is used to report ED status to the MAC.
   * See IEEE 802.15.4-2006, section 6.2.2.4.
   */
  PlmeEdConfirmCallback m_plmeEdConfirmCallback;

  /**
   * This callback is used to report requested attribute values back to the MAC.
   * See IEEE 802.15.4-2006, section 6.2.2.6.
   */
  PlmeGetAttributeConfirmCallback m_plmeGetAttributeConfirmCallback;

  /**
   * This callback is used to report transceiver state change status to the MAC.
   * See IEEE 802.15.4-2006, section 6.2.2.8.
   */
  PlmeSetTRXStateConfirmCallback m_plmeSetTRXStateConfirmCallback;

  /**
   * This callback is used to report attribute set results back to the MAC.
   * See IEEE 802.15.4-2006, section 6.2.2.10.
   */
  PlmeSetAttributeConfirmCallback m_plmeSetAttributeConfirmCallback;

  /**
   * The currently configured PHY type.
   */
  LrWpanPhyOption m_phyOption;

  /**
   * Helper value for tracking the average power during ED.
   */
  LrWpanEdPower m_edPower;

  /**
   * Helper value for the peak power value during CCA.
   */
  double m_ccaPeakPower;

  /**
   * The receiver sensitivity.
   */
  double m_rxSensitivity;

  /**
   * The accumulated signals currently received by the transceiver, including
   * the signal of a possibly received packet, as well as all signals
   * considered noise.
   */
  Ptr<LrWpanInterferenceHelper> m_signal;

  /**
   * Timestamp of the last calculation of the PER of a packet currently received.
   */
  Time m_rxLastUpdate;

  /**
   * Statusinformation of the currently received packet. The first parameter
   * contains the frame, as well the signal power of the frame. The second
   * parameter is set to false, if the frame is either invalid or destroyed
   * due to interference.
   */
  std::pair<Ptr<LrWpanSpectrumSignalParameters>, bool>  m_currentRxPacket;

  /**
   * Statusinformation of the currently transmitted packet. The first parameter
   * contains the frame. The second parameter is set to false, if the frame not
   * completely transmitted, in the event of a force transceiver switch, for
   * example.
   */
  PacketAndStatus m_currentTxPacket;

  /**
   * Scheduler event of a currently running CCA request.
   */
  EventId m_ccaRequest;

  /**
   * Scheduler event of a currently running ED request.
   */
  EventId m_edRequest;

  /**
   * Scheduler event of a currently running deferred transceiver state switch.
   */
  EventId m_setTRXState;

  /**
   * Scheduler event of a currently running data transmission request.
   */
  EventId m_pdDataRequest;

  /**
   * Uniform random variable stream.
   */
  Ptr<UniformRandomVariable> m_random;
};


} // namespace ns3

#endif /* LR_WPAN_PHY_H */
