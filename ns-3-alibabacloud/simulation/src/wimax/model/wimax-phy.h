/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#ifndef WIMAX_PHY_H
#define WIMAX_PHY_H

#include <stdint.h>
#include <list>
#include "ns3/callback.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "bvec.h"
#include "send-params.h"
#include "ns3/log.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"

namespace ns3 {

class WimaxChannel;
class WimaxNetDevice;
class NetDevice;
class Packet;

/**
 * \ingroup wimax
 * 
 * WiMAX PHY entity
 */
class WimaxPhy : public Object
{
public:
  /// ModulationType enumeration
  enum ModulationType // Table 356 and 362
  {
    MODULATION_TYPE_BPSK_12,
    MODULATION_TYPE_QPSK_12,
    MODULATION_TYPE_QPSK_34,
    MODULATION_TYPE_QAM16_12,
    MODULATION_TYPE_QAM16_34,
    MODULATION_TYPE_QAM64_23,
    MODULATION_TYPE_QAM64_34
  };

  /// PhyState enumeration
  enum PhyState
  {
    PHY_STATE_IDLE,
    PHY_STATE_SCANNING,
    PHY_STATE_TX,
    PHY_STATE_RX
  };

  /// PhyType enumeration
  enum PhyType
  {
    SimpleWimaxPhy,
    simpleOfdmWimaxPhy
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  WimaxPhy (void);
  virtual ~WimaxPhy (void);
  /**
   * Attach the physical layer to a channel.
   * \param channel the channel to which the physical layer will be attached
   */
  void Attach (Ptr<WimaxChannel> channel);
  /**
   * \return the channel to which this physical layer is attached
   */
  Ptr<WimaxChannel> GetChannel (void) const;
  /**
   * \brief Set the device in which this physical layer is installed
   * \param device the device in which this physical layer is installed
   */
  void SetDevice (Ptr<WimaxNetDevice> device);
  /**
   * \return the the device in which this physical layer is installed
   */
  Ptr<NetDevice> GetDevice (void) const;
  /**
   * \brief set the callback function to call when a burst is received
   * \param callback the callback function to call when a burst is received
   */
  void SetReceiveCallback (Callback<void, Ptr<const PacketBurst> > callback);
  /**
   * \return the receive callback
   */
  Callback<void, Ptr<const PacketBurst> > GetReceiveCallback (void) const;
  /**
   * \brief send a packet on the channel
   * \param params the parameters used to send the packet
   */
  virtual void Send (SendParams *params) = 0;
  /**
   * \brief Get the type of the physical layer
   * \returns the phy type
   */
  virtual PhyType GetPhyType (void) const = 0;
  /**
   * \brief configure the physical layer in duplex mode
   * \param rxFrequency the reception frequency
   * \param txFrequency the transmission frequency
   */
  void SetDuplex (uint64_t rxFrequency, uint64_t txFrequency);
  /**
   * \brief configure the physical layer in simplex mode
   * \param frequency  the frequency to be used for reception and transmission process
   */
  void SetSimplex (uint64_t frequency);
  /**
   * Get the reception frequency
   * \return the reception frequency
   */
  uint64_t GetRxFrequency (void) const;
  /**
   * Get the transmission frequency
   * \return the transmission frequency
   */
  uint64_t GetTxFrequency (void) const;
  /**
   * Get the scanning frequency
   * \return the scanning frequency
   */
  uint64_t GetScanningFrequency (void) const;
  /**
   * Set the number of carriers in the physical frame
   * \brief Set the number of carriers in the physical frame
   * \param  nrCarriers the number of carriers in the frame
   */
  void SetNrCarriers (uint8_t nrCarriers);
  /**
   * Get the number of carriers in the physical frame
   * \return the number of carriers in the physical frame
   */
  uint8_t GetNrCarriers (void) const;
  /**
   * \brief Set the frame duration
   * \param frameDuration the frame duration
   */
  void SetFrameDuration (Time frameDuration);
  /**
   * \brief Get the frame duration
   * This method is redundant with GetFrameDuration ()
   * \return the frame duration
   */
  Time GetFrameDurationSec (void) const;
  /**
   * \brief Get the frame duration
   * \return the frame duration
   */
  Time GetFrameDuration (void) const;
  /**
   * \brief set the frequency on which the device should lock
   * \param frequency the frequency to configure
   */
  void SetFrequency (uint32_t frequency);
  /**
   * Get the frequency on which the device is locked
   * \return the frequency on which the device is locked
   */
  uint32_t GetFrequency (void) const;
  /**
   * \brief Set the channel bandwidth
   * \param channelBandwidth The channel bandwidth
   */
  void SetChannelBandwidth (uint32_t channelBandwidth);
  /**
   * Get the channel bandwidth
   * \return the channel bandwidth
   */
  uint32_t GetChannelBandwidth (void) const;
  /**
   * Get the size of the FFT
   * \return the size of the FFT
   */
  uint16_t GetNfft (void) const;
  /**
   * Get the sampling factor
   * \return the sampling factor
   */
  double GetSamplingFactor (void) const;
  /**
   * Get the sampling frequency
   * \return the sampling frequency
   */
  double GetSamplingFrequency (void) const;
  /**
   * \brief set the physical slot duration
   * \param psDuration the physical slot duration
   */
  void SetPsDuration (Time psDuration);
  /**
   * Get the physical slot duration
   * \return the physical slot duration
   */
  Time GetPsDuration (void) const;
  /**
   * \brief set the OFDM symbol duration
   * \param symbolDuration the symbol duration
   */
  void SetSymbolDuration (Time symbolDuration);
  /**
   * Get the OFDM symbol duration
   * \return the symbol duration in second
   */
  Time GetSymbolDuration (void) const;
  /**
   * Get the guard interval factor (the ratio TG/Td)
   * \return the guard interval factor
   */
  double GetGValue (void) const;
  /**
   * \brief set the number of physical slots per symbol
   * \param psPerSymbol the number of physical slots per symbol
   */
  void SetPsPerSymbol (uint16_t psPerSymbol);
  /**
   * Get the number of physical slots per symbol
   * \return the number of physical slots per symbol
   */
  uint16_t GetPsPerSymbol (void) const;

  /**
   * \brief set the number of physical slots per frame
   * \param psPerFrame the number of physical slots per frame
   */
  void SetPsPerFrame (uint16_t psPerFrame);
  /**
   * Get the number of physical slots per frame
   * \return the number of physical slot per frame
   */
  uint16_t GetPsPerFrame (void) const;
  /**
   * \brief set the number of symbols per frame
   * \param symbolsPerFrame the number of symbols per frame
   */
  void SetSymbolsPerFrame (uint32_t symbolsPerFrame);
  /**
   * Get the number of symbols per frame
   * \return the number of symbols per frame
   */
  uint32_t GetSymbolsPerFrame (void) const;
  /**
   * Check if configured in duplex mode
   * \return true if the device is configured in duplex mode
   */
  bool IsDuplex (void) const;
  /**
   * \brief set the state of the device
   * \param state the state to be set (PHY_STATE_IDLE, PHY_STATE_SCANNING, PHY_STATE_TX, PHY_STATE_RX)
   */
  void SetState (PhyState state);
  /**
   * Get the state of the device
   * \return the state of the device (PHY_STATE_IDLE, PHY_STATE_SCANNING, PHY_STATE_TX, PHY_STATE_RX)
   */
  PhyState GetState (void) const;
  /**
   * \brief scan a frequency for maximum timeout seconds and call the callback if the frequency can be used
   * \param frequency the frequency to scan
   * \param timeout the timeout before considering the channel as unusable
   * \param callback the function to call if the channel could be used
   */
  void StartScanning (uint64_t frequency, Time timeout, Callback<void, bool, uint64_t> callback);

  /**
   * \brief calls the scanning call back function
   */
  void SetScanningCallback (void) const;
  /**
   * \brief Get channel search timeout event
   * \return event ID
   */
  EventId GetChnlSrchTimeoutEvent (void) const;
  /**
   * \brief calculates the data rate of each modulation and save them for future use
   */
  void SetDataRates (void);
  /**
   * Get the data rate corresponding to a modulation type
   * \return the data rate
   * \param modulationType the modulation that you want to get its data rate
   */
  uint32_t GetDataRate (ModulationType modulationType) const;
  /**
   * Get transmission time needed to send bytes at a given modulation
   * \return the time needed
   * \param size the number of byte to transmit
   * \param modulationType the modulation that will be used to transmit the bytes
   */
  Time GetTransmissionTime (uint32_t size, ModulationType modulationType) const;
  /**
   * Get the number of symbols needed to transmit size bytes using the modulation modulationType
   * \return the number of symbols needed
   * \param size the number of byte to transmit
   * \param modulationType the modulation that will be used to transmit the bytes
   */
  uint64_t GetNrSymbols (uint32_t size, ModulationType modulationType) const;
  /**
   * Get the maximum number of bytes that could be carried by symbols symbols using the modulation modulationType
   * \return the maximum number of bytes
   * \param symbols the number of symbols to use
   * \param modulationType the modulation that will be used
   */
  uint64_t GetNrBytes (uint32_t symbols, ModulationType modulationType) const;
  /**
   * Get the transmit/receive transition gap
   * \return the transmit/receive transition gap
   */
  uint16_t GetTtg (void) const;
  /**
   * Get the receive/transmit transition gap
   * \return the receive/transmit transition gap
   */
  uint16_t GetRtg (void) const;
  /**
   * Get the frame duration code
   * \return the frame duration code
   */
  uint8_t GetFrameDurationCode (void) const;
  /**
   * Get the frame duration corresponding to a given code
   * \param frameDurationCode the frame duration code to use
   * \return the frame duration
   */
  Time GetFrameDuration (uint8_t frameDurationCode) const;
  /**
   * \brief computes the Physical parameters and store them
   */
  void SetPhyParameters (void);
  virtual void DoDispose (void);
  /**
   * Get the mobility model of the device
   * \return the mobility model of the device
   */
  virtual Ptr<Object> GetMobility (void);
  /**
   * \brief set the mobility model of the device
   * \param mobility the mobility model to set
   */
  virtual void SetMobility (Ptr<Object> mobility);

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  virtual int64_t AssignStreams (int64_t stream) = 0;

private:
  /**
   * Get modulation FEC parameters 
   * \param modulationType the modultion type
   * \param bitsPerSymbol the number of bits per symbol page
   * \param fecCode the FEC code
   */
  void GetModulationFecParams (ModulationType modulationType, uint8_t &bitsPerSymbol, double &fecCode) const;
  /// End scanning
  void EndScanning (void);
  /**
   * Get transmission time
   * \param size the transmission size
   * \param modulationType the modulation type
   * \returns the transmission time
   */
  virtual Time DoGetTransmissionTime (uint32_t size, ModulationType modulationType) const = 0;
  /**
   * Attach channel
   * \param channel the wimax channel
   */
  virtual void DoAttach (Ptr<WimaxChannel> channel) = 0;
  /// Set data rates
  virtual void DoSetDataRates (void) = 0;
  /**
   * Get data rate
   * \param modulationType the modulation type
   * \returns the data rate
   */
  virtual uint32_t DoGetDataRate (ModulationType modulationType) const = 0;
  /**
   * Get number of symbols
   * \param size the transmission size
   * \param modulationType the modulation type
   * \returns the number of symbols
   */
  virtual uint64_t DoGetNrSymbols (uint32_t size, ModulationType modulationType) const = 0;
  /**
   * Get number of bytes
   * \param symbols the number of symbols
   * \param modulationType the modulation type
   * \returns the number of bytes
   */
  virtual uint64_t DoGetNrBytes (uint32_t symbols, ModulationType modulationType) const = 0;
  /**
   * Get TTG
   * \returns the TTG
   */
  virtual uint16_t DoGetTtg (void) const = 0;
  /**
   * Get RTG
   * \returns the RTG
   */
  virtual uint16_t DoGetRtg (void) const = 0;

  /**
   * Get frame duration code
   * \returns the frame duration code
   */
  virtual uint8_t DoGetFrameDurationCode (void) const = 0;
  /**
   * Get frame duration
   * \param frameDurationCode the frame duration code
   * \returns the frame duration time
   */
  virtual Time DoGetFrameDuration (uint8_t frameDurationCode) const = 0;
  /**
   * Set phy parameters
   */
  virtual void DoSetPhyParameters (void) = 0;
  /**
   * Get sampling factor
   * \return the sampling factor
   */
  virtual double DoGetSamplingFactor (void) const = 0;
  /**
   * Get NFFT
   * \returns the NFFT
   */
  virtual uint16_t DoGetNfft (void) const = 0;
  /**
   * Get sampling frequency
   * \returns the sampling frequency
   */
  virtual double DoGetSamplingFrequency (void) const = 0;
  /**
   * Get G value
   * \returns he G value
   */
  virtual double DoGetGValue (void) const = 0;

  Ptr<WimaxNetDevice> m_device; ///< the device
  Ptr<WimaxChannel> m_channel; ///< channel

  uint64_t m_txFrequency; ///< transmit frequency
  uint64_t m_rxFrequency; ///< receive frequency
  uint64_t m_scanningFrequency; ///< scanning frequency
  EventId m_dlChnlSrchTimeoutEvent; ///< DL channel search timeout event
  bool m_duplex; ///< duplex
  PhyState m_state; ///< state

  Callback<void, Ptr<const PacketBurst> > m_rxCallback; ///< receive callback function
  Callback<void, bool, uint64_t> m_scanningCallback; ///< scanning callback function

  uint8_t m_nrCarriers; ///< number of carriers
  Time m_frameDuration; ///< in seconds
  uint32_t m_frequency; ///< in KHz
  uint32_t m_channelBandwidth; ///< in Hz
  Time m_psDuration; ///< in seconds
  Time m_symbolDuration; ///< in seconds
  uint16_t m_psPerSymbol; ///< ps per sumbol
  uint16_t m_psPerFrame; ///< ps per framce
  uint32_t m_symbolsPerFrame; ///< symbols per frame
  Ptr<Object> m_mobility; ///< modility model
};

} // namespace ns3

#endif /* WIMAX_PHY_H */
