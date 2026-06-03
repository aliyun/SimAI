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
 *         Giuseppe Piro  <g.piro@poliba.it>
 * Modified by: Marco Miozzo <mmiozzo@cttc.es> (introduce physical error model)
 */

#ifndef LTE_SPECTRUM_PHY_H
#define LTE_SPECTRUM_PHY_H

#include <ns3/event-id.h>
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
#include <ns3/packet-burst.h>
#include <ns3/lte-interference.h>
#include "ns3/random-variable-stream.h"
#include <map>
#include <ns3/ff-mac-common.h>
#include <ns3/lte-harq-phy.h>
#include <ns3/lte-common.h>

namespace ns3 {

/// TbId_t structure
struct TbId_t
{
  uint16_t m_rnti; ///< RNTI
  uint8_t m_layer; ///< layer
  
  public:
  TbId_t ();
  /**
   * Constructor
   *
   * \param a RNTI
   * \param b Layer
   */
  TbId_t (const uint16_t a, const uint8_t b);
  
  friend bool operator == (const TbId_t &a, const TbId_t &b);
  friend bool operator < (const TbId_t &a, const TbId_t &b);
};

  
/// tbInfo_t structure
struct tbInfo_t
{
  uint8_t ndi; ///< New data indicator
  uint16_t size; ///< Transport block size
  uint8_t mcs; ///< MCS
  std::vector<int> rbBitmap; ///< Resource block bitmap
  uint8_t harqProcessId; ///< HARQ process id
  uint8_t rv; ///< Redundancy version
  double mi; ///< Mutual information
  bool downlink; ///< whether is downlink
  bool corrupt; ///< whether is corrupt
  bool harqFeedbackSent; ///< is HARQ feedback sent
};

typedef std::map<TbId_t, tbInfo_t> expectedTbs_t; ///< expectedTbs_t typedef


class LteNetDevice;
class AntennaModel;
class LteControlMessage;
struct LteSpectrumSignalParametersDataFrame;
struct LteSpectrumSignalParametersDlCtrlFrame;
struct LteSpectrumSignalParametersUlSrsFrame;

/**
* This method is used by the LteSpectrumPhy to notify the PHY that a
* previously started RX attempt has terminated without success
*/
typedef Callback< void > LtePhyRxDataEndErrorCallback;
/**
* This method is used by the LteSpectrumPhy to notify the PHY that a
* previously started RX attempt has been successfully completed.
*
* @param packet the received Packet
*/
typedef Callback< void, Ptr<Packet> > LtePhyRxDataEndOkCallback;


/**
* This method is used by the LteSpectrumPhy to notify the PHY that a
* previously started RX of a control frame attempt has been 
* successfully completed.
*
* @param packet the received Packet
*/
typedef Callback< void, std::list<Ptr<LteControlMessage> > > LtePhyRxCtrlEndOkCallback;

/**
* This method is used by the LteSpectrumPhy to notify the PHY that a
* previously started RX of a control frame attempt has terminated 
* without success.
*/
typedef Callback< void > LtePhyRxCtrlEndErrorCallback;

/**
* This method is used by the LteSpectrumPhy to notify the UE PHY that a
* PSS has been received
*/
typedef Callback< void, uint16_t, Ptr<SpectrumValue> > LtePhyRxPssCallback;


/**
* This method is used by the LteSpectrumPhy to notify the PHY about
* the status of a certain DL HARQ process
*/
typedef Callback< void, DlInfoListElement_s > LtePhyDlHarqFeedbackCallback;

/**
* This method is used by the LteSpectrumPhy to notify the PHY about
* the status of a certain UL HARQ process
*/
typedef Callback< void, UlInfoListElement_s > LtePhyUlHarqFeedbackCallback;



/**
 * \ingroup lte
 * \class LteSpectrumPhy
 *
 * The LteSpectrumPhy models the physical layer of LTE
 *
 * It supports a single antenna model instance which is
 * used for both transmission and reception.  
 */
class LteSpectrumPhy : public SpectrumPhy
{

public:
  LteSpectrumPhy ();
  virtual ~LteSpectrumPhy ();

  /**
   *  PHY states
   */
  enum State
  {
    IDLE, TX_DL_CTRL, TX_DATA, TX_UL_SRS, RX_DL_CTRL, RX_DATA, RX_UL_SRS
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  // inherited from Object
  virtual void DoDispose ();

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
   * \brief Start receive data function
   * \param params Ptr<LteSpectrumSignalParametersDataFrame>
   */
  void StartRxData (Ptr<LteSpectrumSignalParametersDataFrame> params);
  /**
   * \brief Start receive DL control function
   * \param lteDlCtrlRxParams Ptr<LteSpectrumSignalParametersDlCtrlFrame>
   */
  void StartRxDlCtrl (Ptr<LteSpectrumSignalParametersDlCtrlFrame> lteDlCtrlRxParams);
  /**
   * \brief Start receive UL SRS function
   * \param lteUlSrsRxParams Ptr<LteSpectrumSignalParametersUlSrsFrame>
   */
  void StartRxUlSrs (Ptr<LteSpectrumSignalParametersUlSrsFrame> lteUlSrsRxParams);
  /**
   * \brief Set HARQ phy function
   * \param harq the HARQ phy module
   */
  void SetHarqPhyModule (Ptr<LteHarqPhy> harq);

  /**
   * set the Power Spectral Density of outgoing signals in W/Hz.
   *
   * @param txPsd
   */
  void SetTxPowerSpectralDensity (Ptr<SpectrumValue> txPsd);

  /**
   * \brief set the noise power spectral density
   * @param noisePsd the Noise Power Spectral Density in power units
   * (Watt, Pascal...) per Hz.
   */
  void SetNoisePowerSpectralDensity (Ptr<const SpectrumValue> noisePsd);

  /** 
   * reset the internal state
   * 
   */
  void Reset ();
 
  /**
   * set the AntennaModel to be used
   * 
   * \param a the Antenna Model
   */
  void SetAntenna (Ptr<AntennaModel> a);
  
  /**
  * Start a transmission of data frame in DL and UL
  *
  *
  * @param pb the burst of packets to be transmitted in PDSCH/PUSCH
  * @param ctrlMsgList the list of LteControlMessage to send
  * @param duration the duration of the data frame 
  *
  * @return true if an error occurred and the transmission was not
  * started, false otherwise.
  */
  bool StartTxDataFrame (Ptr<PacketBurst> pb, std::list<Ptr<LteControlMessage> > ctrlMsgList, Time duration);
  
  /**
  * Start a transmission of control frame in DL
  *
  *
  * @param ctrlMsgList the burst of control messages to be transmitted
  * @param pss the flag for transmitting the primary synchronization signal
  *
  * @return true if an error occurred and the transmission was not
  * started, false otherwise.
  */
  bool StartTxDlCtrlFrame (std::list<Ptr<LteControlMessage> > ctrlMsgList, bool pss);
  
  
  /**
  * Start a transmission of control frame in UL
  *
  * @return true if an error occurred and the transmission was not
  * started, false otherwise.
  */
  bool StartTxUlSrsFrame ();

  /**
   * set the callback for the end of a RX in error, as part of the
   * interconnections between the PHY and the MAC
   *
   * @param c the callback
   */
  void SetLtePhyRxDataEndErrorCallback (LtePhyRxDataEndErrorCallback c);

  /**
   * set the callback for the successful end of a RX, as part of the
   * interconnections between the PHY and the MAC
   *
   * @param c the callback
   */
  void SetLtePhyRxDataEndOkCallback (LtePhyRxDataEndOkCallback c);
  
  /**
  * set the callback for the successful end of a RX ctrl frame, as part 
  * of the interconnections between the LteSpectrumPhy and the PHY
  *
  * @param c the callback
  */
  void SetLtePhyRxCtrlEndOkCallback (LtePhyRxCtrlEndOkCallback c);
  
  /**
  * set the callback for the erroneous end of a RX ctrl frame, as part 
  * of the interconnections between the LteSpectrumPhy and the PHY
  *
  * @param c the callback
  */
  void SetLtePhyRxCtrlEndErrorCallback (LtePhyRxCtrlEndErrorCallback c);

  /**
  * set the callback for the reception of the PSS as part
  * of the interconnections between the LteSpectrumPhy and the UE PHY
  *
  * @param c the callback
  */
  void SetLtePhyRxPssCallback (LtePhyRxPssCallback c);

  /**
  * set the callback for the DL HARQ feedback as part of the 
  * interconnections between the LteSpectrumPhy and the PHY
  *
  * @param c the callback
  */
  void SetLtePhyDlHarqFeedbackCallback (LtePhyDlHarqFeedbackCallback c);

  /**
  * set the callback for the UL HARQ feedback as part of the
  * interconnections between the LteSpectrumPhy and the PHY
  *
  * @param c the callback
  */
  void SetLtePhyUlHarqFeedbackCallback (LtePhyUlHarqFeedbackCallback c);

  /**
   * \brief Set the state of the phy layer
   * \param newState the state
   */
  void SetState (State newState);

  /** 
   * 
   * 
   * \param cellId the Cell Identifier
   */
  void SetCellId (uint16_t cellId);

  /**
   *
   * \param componentCarrierId the component carrier id
   */
  void SetComponentCarrierId (uint8_t componentCarrierId);

  /**
  *
  *
  * \param p the new LteChunkProcessor to be added to the RS power
  *          processing chain
  */
  void AddRsPowerChunkProcessor (Ptr<LteChunkProcessor> p);
  
  /**
  *
  *
  * \param p the new LteChunkProcessor to be added to the Data Channel power
  *          processing chain
  */
  void AddDataPowerChunkProcessor (Ptr<LteChunkProcessor> p);

  /** 
  * 
  * 
  * \param p the new LteChunkProcessor to be added to the data processing chain
  */
  void AddDataSinrChunkProcessor (Ptr<LteChunkProcessor> p);

  /**
  *  LteChunkProcessor devoted to evaluate interference + noise power
  *  in control symbols of the subframe
  *
  * \param p the new LteChunkProcessor to be added to the data processing chain
  */
  void AddInterferenceCtrlChunkProcessor (Ptr<LteChunkProcessor> p);

  /**
  *  LteChunkProcessor devoted to evaluate interference + noise power
  *  in data symbols of the subframe
  *
  * \param p the new LteChunkProcessor to be added to the data processing chain
  */
  void AddInterferenceDataChunkProcessor (Ptr<LteChunkProcessor> p);
  
  
  /** 
  * 
  * 
  * \param p the new LteChunkProcessor to be added to the ctrl processing chain
  */
  void AddCtrlSinrChunkProcessor (Ptr<LteChunkProcessor> p);
  
  /** 
  * 
  * 
  * \param rnti the RNTI of the source of the TB
  * \param ndi new data indicator flag
  * \param size the size of the TB
  * \param mcs the MCS of the TB
  * \param map the map of RB(s) used
  * \param layer the layer (in case of MIMO tx)
  * \param harqId the id of the HARQ process (valid only for DL)
  * \param rv the redundancy version
  * \param downlink true when the TB is for DL
  */
  void AddExpectedTb (uint16_t  rnti, uint8_t ndi, uint16_t size, uint8_t mcs, std::vector<int> map, uint8_t layer, uint8_t harqId, uint8_t rv, bool downlink);
  /**
   * \brief Remove expected transport block.
   *
   * When UE context at eNodeB is removed and if UL TB is expected to be received
   * but not transmitted due to break in radio link. The TB with different rnti or lcid
   * remains during the transmission of a new TB and causes problems with
   * m_ulPhyReception trace, since the UE context was already removed. TB has to be
   * removed when ue context at eNodeB is removed
   *
   * \param rnti The RNTI of the UE
   */
  void RemoveExpectedTb (uint16_t  rnti);

  /** 
  * 
  * 
  * \param sinr vector of sinr perceived per each RB
  */
  void UpdateSinrPerceived (const SpectrumValue& sinr);
  
  /** 
  * 
  * 
  * \param txMode UE transmission mode (SISO, MIMO tx diversity, ...)
  */
  void SetTransmissionMode (uint8_t txMode);
  

  /** 
   * 
   * \return the previously set channel
   */
  Ptr<SpectrumChannel> GetChannel ();

  /// allow LteUePhy class friend access
  friend class LteUePhy;
  
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
  * \brief Change state function
  * 
  * \param newState the new state to set
  */
  void ChangeState (State newState);
  /// End transmit data function
  void EndTxData ();
  /// End transmit DL control function
  void EndTxDlCtrl ();
  /// End transmit UL SRS function
  void EndTxUlSrs ();
  /// End receive data function
  void EndRxData ();
  /// End receive DL control function
  void EndRxDlCtrl ();
  /// End receive UL SRS function
  void EndRxUlSrs ();
  
  /** 
  * \brief Set transmit mode gain function
  * 
  * \param txMode the transmit mode
  * \param gain the gain to set
  */
  void SetTxModeGain (uint8_t txMode, double gain);
  

  Ptr<MobilityModel> m_mobility; ///< the modility model
  Ptr<AntennaModel> m_antenna; ///< the antenna model
  Ptr<NetDevice> m_device; ///< the device

  Ptr<SpectrumChannel> m_channel; ///< the channel

  Ptr<const SpectrumModel> m_rxSpectrumModel; ///< the spectrum model
  Ptr<SpectrumValue> m_txPsd; ///< the transmit PSD
  Ptr<PacketBurst> m_txPacketBurst; ///< the transmit packet burst
  std::list<Ptr<PacketBurst> > m_rxPacketBurstList; ///< the receive burst list
  
  std::list<Ptr<LteControlMessage> > m_txControlMessageList; ///< the transmit control message list
  std::list<Ptr<LteControlMessage> > m_rxControlMessageList; ///< the receive control message list
  
  
  State m_state; ///< the state
  Time m_firstRxStart; ///< the first receive start
  Time m_firstRxDuration; ///< the first receive duration

  TracedCallback<Ptr<const PacketBurst> > m_phyTxStartTrace; ///< the phy transmit start trace callback
  TracedCallback<Ptr<const PacketBurst> > m_phyTxEndTrace; ///< the phy transmit end trace callback
  TracedCallback<Ptr<const PacketBurst> > m_phyRxStartTrace; ///< the phy receive start trace callback
  TracedCallback<Ptr<const Packet> >      m_phyRxEndOkTrace; ///< the phy receive end ok trace callback
  TracedCallback<Ptr<const Packet> >      m_phyRxEndErrorTrace; ///< the phy receive end error trace callback

  LtePhyRxDataEndErrorCallback   m_ltePhyRxDataEndErrorCallback; ///< the LTE phy receive data end error callback 
  LtePhyRxDataEndOkCallback      m_ltePhyRxDataEndOkCallback; ///< the LTE phy receive data end ok callback
  
  LtePhyRxCtrlEndOkCallback     m_ltePhyRxCtrlEndOkCallback; ///< the LTE phy receive control end ok callback
  LtePhyRxCtrlEndErrorCallback  m_ltePhyRxCtrlEndErrorCallback; ///< the LTE phy receive control end error callback
  LtePhyRxPssCallback  m_ltePhyRxPssCallback; ///< the LTE phy receive PSS callback

  Ptr<LteInterference> m_interferenceData; ///< the data interference
  Ptr<LteInterference> m_interferenceCtrl; ///< the control interference

  uint16_t m_cellId; ///< the cell ID
  
  uint8_t m_componentCarrierId; ///< the component carrier ID
  expectedTbs_t m_expectedTbs; ///< the expected TBS
  SpectrumValue m_sinrPerceived; ///< the preceived SINR 

  /// Provides uniform random variables.
  Ptr<UniformRandomVariable> m_random;
  bool m_dataErrorModelEnabled; ///< when true (default) the phy error model is enabled
  bool m_ctrlErrorModelEnabled; ///< when true (default) the phy error model is enabled for DL ctrl frame
  
  uint8_t m_transmissionMode; ///< for UEs: store the transmission mode
  uint8_t m_layersNum; ///< layers num
  std::vector <double> m_txModeGain; ///< duplicate value of LteUePhy

  Ptr<LteHarqPhy> m_harqPhyModule; ///< the HARQ phy module
  LtePhyDlHarqFeedbackCallback m_ltePhyDlHarqFeedbackCallback; ///< the LTE phy DL HARQ feedback callback
  LtePhyUlHarqFeedbackCallback m_ltePhyUlHarqFeedbackCallback; ///< the LTE phy UL HARQ feedback callback


  /**
   * Trace information regarding PHY stats from DL Rx perspective
   * PhyReceptionStatParameters (see lte-common.h)
   */
  TracedCallback<PhyReceptionStatParameters> m_dlPhyReception;

  
  /**
   * Trace information regarding PHY stats from UL Rx perspective
   * PhyReceptionStatParameters (see lte-common.h)
   */
  TracedCallback<PhyReceptionStatParameters> m_ulPhyReception;

  EventId m_endTxEvent; ///< end transmit event
  EventId m_endRxDataEvent; ///< end receive data event
  EventId m_endRxDlCtrlEvent; ///< end receive DL control event
  EventId m_endRxUlSrsEvent; ///< end receive UL SRS event
  

};






}

#endif /* LTE_SPECTRUM_PHY_H */
