/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
 * Copyright (c) 2018 Fraunhofer ESK : RLF extensions
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
 * Author: Giuseppe Piro  <g.piro@poliba.it>
 * Author: Marco Miozzo <mmiozzo@cttc.es>
 * Modified by:
 *          Vignesh Babu <ns3-dev@esk.fraunhofer.de> (RLF extensions)
 */

#ifndef LTE_UE_PHY_H
#define LTE_UE_PHY_H


#include <ns3/lte-phy.h>
#include <ns3/ff-mac-common.h>

#include <ns3/lte-control-messages.h>
#include <ns3/lte-amc.h>
#include <ns3/lte-ue-phy-sap.h>
#include <ns3/lte-ue-cphy-sap.h>
#include <ns3/ptr.h>
#include <ns3/lte-amc.h>
#include <set>
#include <ns3/lte-ue-power-control.h>


namespace ns3 {

class PacketBurst;
class LteEnbPhy;
class LteHarqPhy;


/**
 * \ingroup lte
 *
 * The LteSpectrumPhy models the physical layer of LTE
 */
class LteUePhy : public LtePhy
{

  /// allow UeMemberLteUePhySapProvider class friend access
  friend class UeMemberLteUePhySapProvider;
  /// allow MemberLteUeCphySapProvider<LteUePhy> class friend access
  friend class MemberLteUeCphySapProvider<LteUePhy>;

public:
  /**
   * \brief The states of the UE PHY entity
   */
  enum State
  {
    CELL_SEARCH = 0,
    SYNCHRONIZED,
    NUM_STATES
  };

  /**
   * @warning the default constructor should not be used
   */
  LteUePhy ();

  /**
   *
   * \param dlPhy the downlink LteSpectrumPhy instance
   * \param ulPhy the uplink LteSpectrumPhy instance
   */
  LteUePhy (Ptr<LteSpectrumPhy> dlPhy, Ptr<LteSpectrumPhy> ulPhy);

  virtual ~LteUePhy ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  // inherited from Object
  virtual void DoInitialize (void);
  virtual void DoDispose (void);

  /**
   * \brief Get the PHY SAP provider
   *
   * \return a pointer to the SAP Provider 
   */
  LteUePhySapProvider* GetLteUePhySapProvider ();

  /**
   * \brief Set the PHY SAP User
   *
   * \param s a pointer to the SAP user
   */
  void SetLteUePhySapUser (LteUePhySapUser* s);

  /**
   * \brief Get the CPHY SAP provider
   *
   * \return a pointer to the SAP Provider
   */
  LteUeCphySapProvider* GetLteUeCphySapProvider ();

  /**
   * \brief Set the CPHY SAP User
   *
   * \param s a pointer to the SAP user
   */
  void SetLteUeCphySapUser (LteUeCphySapUser* s);


  /**
   * \brief Set transmit power
   *
   * \param pow the transmission power in dBm
   */
  void SetTxPower (double pow);

  /**
   * \brief Get transmit power
   *
   * \return the transmission power in dBm
   */
  double GetTxPower () const;

  /**
   * \brief Get Uplink power control
   *
   * \return ptr to UE Uplink Power Control entity
   */
  Ptr<LteUePowerControl> GetUplinkPowerControl () const;

  /**
   * \brief Set noise figure
   *
   * \param nf the noise figure in dB
   */
  void SetNoiseFigure (double nf);

  /**
   * \brief Get noise figure
   *
   * \return the noise figure in dB
   */
  double GetNoiseFigure () const;

  /**
   * \brief Get MAC to Channel delay
   *
   * \returns the TTI delay between MAC and channel
   */
  uint8_t GetMacChDelay (void) const;

  /**
   * \brief Get Downlink spectrum phy
   *
   * \return a pointer to the LteSpectrumPhy instance relative to the downlink
   */
  Ptr<LteSpectrumPhy> GetDlSpectrumPhy () const;

  /**
   * \brief Get Uplink spectrum phy
   *
   * \return a pointer to the LteSpectrumPhy instance relative to the uplink
   */
  Ptr<LteSpectrumPhy> GetUlSpectrumPhy () const;

  /**
   * \brief Create the PSD for the TX
   *
   * \return the pointer to the PSD
   */
  virtual Ptr<SpectrumValue> CreateTxPowerSpectralDensity ();

  /**
   * \brief Set a list of sub channels to use in TX
   *
   * \param mask a list of sub channels
   */
  void SetSubChannelsForTransmission (std::vector <int> mask);
  /**
   * \brief Get a list of sub channels to use in RX
   *
   * \return a list of sub channels
   */
  std::vector <int> GetSubChannelsForTransmission (void);

  /**
   * \brief Get a list of sub channels to use in RX
   *
   * \param mask list of sub channels
   */
  void SetSubChannelsForReception (std::vector <int> mask);
  /**
   * \brief Get a list of sub channels to use in RX
   *
   * \return a list of sub channels
   */
  std::vector <int> GetSubChannelsForReception (void);

  /**
   * \brief Create the DL CQI feedback from SINR values perceived at
   * the physical layer with the signal received from eNB
   *
   * \param sinr SINR values vector
   * \return a DL CQI control message containing the CQI feedback
   */
  Ptr<DlCqiLteControlMessage> CreateDlCqiFeedbackMessage (const SpectrumValue& sinr);



  // inherited from LtePhy
  virtual void GenerateCtrlCqiReport (const SpectrumValue& sinr);
  virtual void GenerateDataCqiReport (const SpectrumValue& sinr);
  /**
   * \brief Create the mixed CQI report
   *
   * \param sinr SINR values vector
   */
  virtual void GenerateMixedCqiReport (const SpectrumValue& sinr);
  virtual void ReportInterference (const SpectrumValue& interf);
  /**
   * \brief Create the mixed CQI report
   *
   * \param interf interference values vector
   */
  virtual void ReportDataInterference (const SpectrumValue& interf);
  virtual void ReportRsReceivedPower (const SpectrumValue& power);

  // callbacks for LteSpectrumPhy
  /**
   * \brief Receive LTE control message list function
   *
   * \param msgList LTE control message list
   */
  virtual void ReceiveLteControlMessageList (std::list<Ptr<LteControlMessage> > msgList);
  /**
   * \brief Receive PSS function
   *
   * \param cellId the cell ID
   * \param p PSS list
   */
  virtual void ReceivePss (uint16_t cellId, Ptr<SpectrumValue> p);


  /**
   * \brief PhySpectrum received a new PHY-PDU
   *
   * \param p the packet received
   */
  void PhyPduReceived (Ptr<Packet> p);


  /**
   * \brief trigger from eNB the start from a new frame
   *
   * \param frameNo frame number
   * \param subframeNo subframe number
   */
  void SubframeIndication (uint32_t frameNo, uint32_t subframeNo);


  /**
   * \brief Send the SRS signal in the last symbols of the frame
   */
  void SendSrs ();

  /**
   * \brief Enqueue the downlink HARQ feedback generated by LteSpectrumPhy
   *
   * \param mes the DlInfoListElement_s
   */
  virtual void EnqueueDlHarqFeedback (DlInfoListElement_s mes);

  /**
   * \brief Set the HARQ PHY module
   *
   * \param harq the HARQ PHY module
   */
  void SetHarqPhyModule (Ptr<LteHarqPhy> harq);

  /**
   * \brief Get state of the UE physical layer
   *
   * \return The current state
   */
  State GetState () const;

  /**
   * TracedCallback signature for state transition events.
   *
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] oldState
   * \param [in] newState
   */
  typedef void (* StateTracedCallback)
    (uint16_t cellId, uint16_t rnti, State oldState, State newState);

  /**
   * TracedCallback signature for cell RSRP and SINR report.
   *
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] rsrp
   * \param [in] sinr
   * \param [in] componentCarrierId
   */
  typedef void (* RsrpSinrTracedCallback)
    (uint16_t cellId, uint16_t rnti,
     double rsrp, double sinr, uint8_t componentCarrierId);

  /**
   * TracedCallback signature for cell RSRP and RSRQ.
   *
   * \param [in] rnti
   * \param [in] cellId
   * \param [in] rsrp
   * \param [in] rsrq
   * \param [in] isServingCell
   * \param [in] componentCarrierId
   */
  typedef void (* RsrpRsrqTracedCallback)
    (uint16_t rnti, uint16_t cellId, double rsrp, double rsrq,
     bool isServingCell, uint8_t componentCarrierId);

  /**
   * TracedCallback signature for UL Phy resource blocks.
   *
   * \param [in] rnti
   * \param [in] rbs Vector of resource blocks allocated for UL.
   */
  typedef void (* UlPhyResourceBlocksTracedCallback)
    (uint16_t rnti, const std::vector<int>& rbs);

  /**
   * TracedCallback signature for spectral value.
   *
   * \param [in] rnti
   * \param [in] psd The spectral power density.
   */
  typedef void (* PowerSpectralDensityTracedCallback)
      (uint16_t rnti, Ptr<SpectrumValue> psd);

private:

  /**
   * \brief Set transmit mode 1 gain function
   *
   * \param [in] gain
   */
  void SetTxMode1Gain (double gain);
  /**
   * Set transmit mode 2 gain function
   *
   * \param [in] gain
   */
  void SetTxMode2Gain (double gain);
  /**
   * \brief Set transmit mode 3 gain function
   *
   * \param [in] gain
   */
  void SetTxMode3Gain (double gain);
  /**
   * \brief Set transmit mode 4 gain function
   *
   * \param [in] gain
   */
  void SetTxMode4Gain (double gain);
  /**
   * \brief Set transmit mode 5 gain function
   *
   * \param [in] gain
   */
  void SetTxMode5Gain (double gain);
  /**
   * \brief Set transmit mode 6 gain function
   *
   * \param [in] gain
   */
  void SetTxMode6Gain (double gain);
  /**
   * \brief Set transmit mode 7 gain function
   *
   * \param [in] gain
   */
  void SetTxMode7Gain (double gain);
  /**
   * \brief Set transmit mode gain function
   *
   * \param [in] txMode
   * \param [in] gain
   */
  void SetTxModeGain (uint8_t txMode, double gain);
  /**
   * \brief Queue subchannels for transmission function
   *
   * \param [in] rbMap
   */
  void QueueSubChannelsForTransmission (std::vector <int> rbMap);
  /** 
   * \brief Get CQI, RSRP, and RSRQ
   *
   * internal method that takes care of generating CQI reports,
   * calculating the RSRP and RSRQ metrics, and generating RSRP+SINR traces
   * 
   * \param sinr 
   */
  void GenerateCqiRsrpRsrq (const SpectrumValue& sinr);
  /**
   * \brief Layer-1 filtering of RSRP and RSRQ measurements and reporting to
   *        the RRC entity.
   *
   * Initially executed at +0.200s, and then repeatedly executed with
   * periodicity as indicated by the *UeMeasurementsFilterPeriod* attribute.
   */
  void ReportUeMeasurements ();
  /**
   * \brief Set the periodicty for the downlink periodic
   * wideband and aperiodic subband CQI reporting.
   *
   * \param cqiPeriodicity The downlink CQI reporting periodicity in milliseconds
   */
  void SetDownlinkCqiPeriodicity (Time cqiPeriodicity);
  /**
   * \brief Switch the UE PHY to the given state.
   * \param s the destination state
   */
  void SwitchToState (State s);
  /**
   * \brief Set number of Qout evaluation subframes
   *
   * The number passed to this method should be multiple
   * of 10. This number specifies the total number of consecutive
   * subframes, which corresponds to the Qout evaluation period.
   *
   * \param numSubframes the number of subframes
   */
  void SetNumQoutEvalSf (uint16_t numSubframes);
  /**
   * \brief Set number of Qin evaluation subframes
   *
   * The number passed to this method should be multiple
   * of 10. This number specifies the total number of consecutive
   * subframes, which corresponds to the Qin evaluation period.
   *
   * \param numSubframes the number of subframes
   */
  void SetNumQinEvalSf (uint16_t numSubframes);
  /**
   * \brief Get number of Qout evaluation subframes
   *
   * The number returned by this method specifies the
   * total number of consecutive subframes, which corresponds
   * to the Qout evaluation period.
   *
   * \return the number of consecutive subframes used for Qout evaluation
   */
  uint16_t GetNumQoutEvalSf () const;
  /**
   * \brief Get number of Qin evaluation subframes
   *
   * The number returned by this method specifies the
   * total number of consecutive subframes, which corresponds
   * to the Qin evaluation period.
   *
   * \return the number of consecutive subframes used for Qin evaluation
   */
  uint16_t GetNumQinEvalSf () const;

  // UE CPHY SAP methods
  /**
   * \brief Do Reset function
   */
  void DoReset ();
  /**
   * \brief Start the cell search function
   *
   * \param dlEarfcn the DL EARFCN
   */
  void DoStartCellSearch (uint32_t dlEarfcn);
  /**
   * \brief Synchronize with ENB function
   *
   * \param cellId the cell ID
   */
  void DoSynchronizeWithEnb (uint16_t cellId);
  /**
   * \brief Synchronize with ENB function
   *
   * \param cellId the cell ID
   * \param dlEarfcn the DL EARFCN
   */
  void DoSynchronizeWithEnb (uint16_t cellId, uint32_t dlEarfcn);
  /**
   *
   * Get cell ID
   * \returns cell ID
   */
  uint16_t DoGetCellId ();
  /**
   * Get DL EARFCN
   * \returns DL EARFCN
   */
  uint32_t DoGetDlEarfcn ();
  /**
   * Set DL bandwidth function
   *
   * \param dlBandwidth the DL bandwidth
   */
  void DoSetDlBandwidth (uint16_t dlBandwidth);
  /**
   * \brief Configure UL uplink function
   *
   * \param ulEarfcn UL EARFCN
   * \param ulBandwidth the UL bandwidth
   */
  void DoConfigureUplink (uint32_t ulEarfcn, uint16_t ulBandwidth);
  /**
   * \brief Configure reference signal power function
   *
   * \param referenceSignalPower reference signal power in dBm
   */
  void DoConfigureReferenceSignalPower (int8_t referenceSignalPower);
  /**
   * \brief Set RNTI function
   *
   * \param rnti the RNTI
   */
  void DoSetRnti (uint16_t rnti);
  /**
   * \brief Set transmission mode function
   *
   * \param txMode the transmission mode
   */
  void DoSetTransmissionMode (uint8_t txMode);
  /**
   * \brief Set SRS configuration index function
   *
   * \param srcCi the SRS configuration index
   */
  void DoSetSrsConfigurationIndex (uint16_t srcCi);
  /**
   * \brief Set PA function
   *
   * \param pa the PA value
   */
  void DoSetPa (double pa);
  /**
   * \brief Reset Phy after radio link failure function
   *
   * It resets the physical layer parameters of the
   * UE after RLF.
   *
   */
  void DoResetPhyAfterRlf ();
  /**
   * \brief Reset radio link failure parameters
   *
   * Upon receiving N311 in Sync indications from the UE
   * PHY, the UE RRC instructs the UE PHY to reset the
   * RLF parameters so, it can start RLF detection again.
   *
   */
  void DoResetRlfParams ();

  /**
   * \brief Start in Snyc detection function
   *
   * When T310 timer is started, it indicates that physical layer
   * problems are detected at the UE and the recovery process is
   * started by checking if the radio frames are in-sync for N311
   * consecutive times.
   *
   */
  void DoStartInSnycDetection ();

  /**
   * \brief Radio link failure detection function
   *
   * Radio link monitoring is started to detect downlink radio link
   * quality when the UE is both uplink and downlink synchronized
   * (UE in CONNECTED_NORMALLY state).
   * Upon detection of radio link failure, RRC connection is released
   * and the UE starts the cell selection again. The procedure is implemented
   * as per 3GPP TS 36.213 4.2.1 and TS 36.133 7.6. When the downlink
   * radio link quality estimated over the last 200 ms period becomes worse than
   * the threshold Qout, an out-of-sync indication is sent to RRC. When the
   * downlink radio link quality estimated over the last 100 ms period becomes
   * better than the threshold Qin, an in-sync indication is sent to RRC.
   *
   * \param sinrdB the average SINR value in dB measured across
   * all resource blocks
   *
   */
  void RlfDetection (double sinrdB);
  /**
   * \brief Initialize radio link failure parameters
   *
   * Upon receiving the notification about the successful RRC connection
   * establishment, the UE phy initialize the RLF parameters to be
   * ready for RLF detection.
   *
   */
  void InitializeRlfParams ();
  /**
   * Set IMSI
   *
   * \param imsi the IMSI of the UE
   */
  void DoSetImsi (uint64_t imsi);
  /**
   * \brief Do set RSRP filter coefficient
   *
   * \param rsrpFilterCoefficient value. Determines the strength of
   * smoothing effect induced by layer 3 filtering of RSRP
   * used for uplink power control in all attached UE.
   * If equals to 0, no layer 3 filtering is applicable.
   */
  void DoSetRsrpFilterCoefficient (uint8_t rsrpFilterCoefficient);
  /**
   * \brief Compute average SINR among the RBs
   *
   * \param sinr
   * \return the average SINR value
   */
  double ComputeAvgSinr (const SpectrumValue& sinr);

  // UE PHY SAP methods 
  virtual void DoSendMacPdu (Ptr<Packet> p);
  /**
   * \brief Send LTE control message function
   *
   * \param msg the LTE control message
   */
  virtual void DoSendLteControlMessage (Ptr<LteControlMessage> msg);
  /**
   * \brief Send RACH preamble function
   *
   * \param prachId the RACH preamble ID
   * \param raRnti the rnti
   */
  virtual void DoSendRachPreamble (uint32_t prachId, uint32_t raRnti);
  /**
   * \brief Notify PHY about the successful RRC connection
   * establishment.
   */
  virtual void DoNotifyConnectionSuccessful ();

  /// A list of sub channels to use in TX.
  std::vector <int> m_subChannelsForTransmission;
  /// A list of sub channels to use in RX.
  std::vector <int> m_subChannelsForReception;

  std::vector< std::vector <int> > m_subChannelsForTransmissionQueue; ///< subchannels for transmission queue


  Ptr<LteAmc> m_amc; ///< AMC

  /**
   * The `EnableUplinkPowerControl` attribute. If true, Uplink Power Control
   * will be enabled.
   */
  bool m_enableUplinkPowerControl;
  /// Pointer to UE Uplink Power Control entity.
  Ptr<LteUePowerControl> m_powerControl;

  /// Wideband Periodic CQI. 2, 5, 10, 16, 20, 32, 40, 64, 80 or 160 ms.
  Time m_p10CqiPeriodicity;
  Time m_p10CqiLast; ///< last periodic CQI

  /**
   * SubBand Aperiodic CQI. Activated by DCI format 0 or Random Access Response
   * Grant.
   * \note Defines a periodicity for academic studies.
   */
  Time m_a30CqiPeriodicity;
  Time m_a30CqiLast; ///< last aperiodic CQI

  LteUePhySapProvider* m_uePhySapProvider; ///< UE Phy SAP provider
  LteUePhySapUser* m_uePhySapUser; ///< UE Phy SAP user

  LteUeCphySapProvider* m_ueCphySapProvider; ///< UE CPhy SAP provider
  LteUeCphySapUser* m_ueCphySapUser; ///< UE CPhy SAP user

  uint16_t  m_rnti; ///< the RNTI
 
  uint8_t m_transmissionMode; ///< the transmission mode
  std::vector <double> m_txModeGain; ///< the transmit mode gain

  uint16_t m_srsPeriodicity; ///< SRS periodicity
  uint16_t m_srsSubframeOffset; ///< SRS subframe offset
  uint16_t m_srsConfigured; ///< SRS configured
  Time     m_srsStartTime; ///< SRS start time

  double m_paLinear; ///< PA linear

  bool m_dlConfigured; ///< DL configured?
  bool m_ulConfigured; ///< UL configured?

  /// The current UE PHY state.
  State m_state;
  /**
   * The `StateTransition` trace source. Fired upon every UE PHY state
   * transition. Exporting the serving cell ID, RNTI, old state, and new state.
   */
  TracedCallback<uint16_t, uint16_t, State, State> m_stateTransitionTrace;

  /// \todo Can be removed.
  uint8_t m_subframeNo;

  bool m_rsReceivedPowerUpdated; ///< RS receive power updated?
  SpectrumValue m_rsReceivedPower; ///< RS receive power

  bool m_rsInterferencePowerUpdated; ///< RS interference power updated?
  SpectrumValue m_rsInterferencePower; ///< RS interference power

  bool m_dataInterferencePowerUpdated; ///< data interference power updated?
  SpectrumValue m_dataInterferencePower; ///< data interference power

  bool m_pssReceived; ///< PSS received?
  /// PssElement structure
  struct PssElement 
  {
    uint16_t cellId; ///< cell ID
    double pssPsdSum; ///< PSS PSD sum
    uint16_t nRB; ///< number of RB
  };
  std::list <PssElement> m_pssList; ///< PSS list

  /**
   * The `RsrqUeMeasThreshold` attribute. Receive threshold for PSS on RSRQ
   * in dB.
   */
  double m_pssReceptionThreshold;

  /// Summary results of measuring a specific cell. Used for layer-1 filtering.
  struct UeMeasurementsElement
  {
    double rsrpSum;   ///< Sum of RSRP sample values in linear unit.
    uint8_t rsrpNum;  ///< Number of RSRP samples.
    double rsrqSum;   ///< Sum of RSRQ sample values in linear unit.
    uint8_t rsrqNum;  ///< Number of RSRQ samples.
  };

  /**
   * Store measurement results during the last layer-1 filtering period.
   * Indexed by the cell ID where the measurements come from.
   */
  std::map <uint16_t, UeMeasurementsElement> m_ueMeasurementsMap;
  /**
   * The `UeMeasurementsFilterPeriod` attribute. Time period for reporting UE
   * measurements, i.e., the length of layer-1 filtering (default 200 ms).
   */
  Time m_ueMeasurementsFilterPeriod;
  /// \todo Can be removed.
  Time m_ueMeasurementsFilterLast;

  Ptr<LteHarqPhy> m_harqPhyModule; ///< HARQ phy module

  uint32_t m_raPreambleId; ///< RA preamble ID
  uint32_t m_raRnti; ///< RA RNTI

  /**
   * The `ReportCurrentCellRsrpSinr` trace source. Trace information regarding
   * RSRP and average SINR (see TS 36.214). Exporting cell ID, RNTI, RSRP, and
   * SINR. Moreover it reports the m_componentCarrierId.
   */
  TracedCallback<uint16_t, uint16_t, double, double, uint8_t> m_reportCurrentCellRsrpSinrTrace;
  /**
   * The `RsrpSinrSamplePeriod` attribute. The sampling period for reporting
   * RSRP-SINR stats.
   */
  uint16_t m_rsrpSinrSamplePeriod;
  /**
   * The `RsrpSinrSampleCounter` attribute. The sampling counter for reporting
   * RSRP-SINR stats.
   */
  uint16_t m_rsrpSinrSampleCounter;

  /**
   * The `ReportUeMeasurements` trace source. Contains trace information
   * regarding RSRP and RSRQ measured from a specific cell (see TS 36.214).
   * Exporting RNTI, the ID of the measured cell, RSRP (in dBm), RSRQ (in dB),
   * and whether the cell is the serving cell. Moreover it report the m_componentCarrierId.
   */
  TracedCallback<uint16_t, uint16_t, double, double, bool, uint8_t> m_reportUeMeasurements;

  EventId m_sendSrsEvent; ///< send SRS event

  /**
   * The `UlPhyTransmission` trace source. Contains trace information regarding
   * PHY stats from UL Tx perspective. Exporting a structure with type
   * PhyTransmissionStatParameters.
   */
  TracedCallback<PhyTransmissionStatParameters> m_ulPhyTransmission;

  /**
   * The `ReportUlPhyResourceBlocks` trace source. Contains trace information
   * regarding PHY stats from UL Resource Blocks (RBs). Exporting an RNTI of a
   * UE and a vector containing the indices of the RBs used for UL.
   */
  TracedCallback< uint16_t, const std::vector<int>& > m_reportUlPhyResourceBlocks;

  /**
   * The `ReportsPowerSpectralDensity` trace source. Contains trace information
   * regarding Power Spectral Density. Exporting an RNTI of a UE and a pointer
   * to Spectrum Values.
   */
  TracedCallback< uint16_t, Ptr<SpectrumValue> > m_reportPowerSpectralDensity;

  
  Ptr<SpectrumValue> m_noisePsd; ///< Noise power spectral density for
                                 ///the configured bandwidth

  bool m_isConnected; ///< set when UE RRC is in CONNECTED_NORMALLY state
  /**
   * The 'Qin' attribute.
   * corresponds to 2% block error rate of a hypothetical PDCCH transmission
   * taking into account the PCFICH errors.
   */
  double m_qIn;

  /**
   * The 'Qout' attribute.
   * corresponds to 2% block error rate of a hypothetical PDCCH transmission
   * taking into account the PCFICH errors.
   */
  double m_qOut;

  uint16_t m_numOfQoutEvalSf; ///< the downlink radio link quality is estimated over this period for detecting out-of-syncs
  uint16_t m_numOfQinEvalSf; ///< the downlink radio link quality is estimated over this period for detecting in-syncs

  bool m_downlinkInSync; ///< when set, DL SINR evaluation for out-of-sync indications is conducted.
  uint16_t m_numOfSubframes; ///< count the number of subframes for which the downlink radio link quality is estimated
  uint16_t m_numOfFrames; ///< count the number of frames for which the downlink radio link quality is estimated
  double m_sinrDbFrame; ///< the average SINR per radio frame
  SpectrumValue m_ctrlSinrForRlf; ///< the CTRL SINR used for RLF detection
  uint64_t m_imsi; ///< the IMSI of the UE
  bool m_enableRlfDetection; ///< Flag to enable/disable RLF detection

}; // end of `class LteUePhy`


}

#endif /* LTE_UE_PHY_H */
