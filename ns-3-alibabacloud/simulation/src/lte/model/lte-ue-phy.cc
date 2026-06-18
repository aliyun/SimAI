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
 *         Marco Miozzo <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 * Modified by:
 *          Vignesh Babu <ns3-dev@esk.fraunhofer.de> (RLF extensions)
 */

#include <ns3/object-factory.h>
#include <ns3/log.h>
#include <ns3/node.h>
#include <cfloat>
#include <cmath>
#include <ns3/simulator.h>
#include <ns3/double.h>
#include "lte-ue-phy.h"
#include "lte-enb-phy.h"
#include "lte-net-device.h"
#include "lte-ue-net-device.h"
#include "lte-enb-net-device.h"
#include "lte-spectrum-value-helper.h"
#include "lte-amc.h"
#include "lte-ue-mac.h"
#include "ff-mac-common.h"
#include "lte-chunk-processor.h"
#include <ns3/lte-common.h>
#include <ns3/pointer.h>
#include <ns3/boolean.h>
#include <ns3/lte-ue-power-control.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteUePhy");



/**
 * Duration of the data portion of a UL subframe.
 * Equals to "TTI length - 1 symbol length for SRS - margin".
 * The margin is 1 nanosecond and is intended to avoid overlapping simulator
 * events. The duration of one symbol is TTI/14 (rounded). In other words,
 * duration of data portion of UL subframe = 1 ms * (13/14) - 1 ns.
 */
static const Time UL_DATA_DURATION = NanoSeconds (1e6 - 71429 - 1);

/**
 * Delay from subframe start to transmission of SRS.
 * Equals to "TTI length - 1 symbol for SRS".
 */
static const Time UL_SRS_DELAY_FROM_SUBFRAME_START = NanoSeconds (1e6 - 71429);




////////////////////////////////////////
// member SAP forwarders
////////////////////////////////////////

/// UeMemberLteUePhySapProvider class
class UeMemberLteUePhySapProvider : public LteUePhySapProvider
{
public:
  /**
   * Constructor
   *
   * \param phy the LTE UE Phy
   */
  UeMemberLteUePhySapProvider (LteUePhy* phy);

  // inherited from LtePhySapProvider
  virtual void SendMacPdu (Ptr<Packet> p);
  virtual void SendLteControlMessage (Ptr<LteControlMessage> msg);
  virtual void SendRachPreamble (uint32_t prachId, uint32_t raRnti);
  virtual void NotifyConnectionSuccessful ();

private:
  LteUePhy* m_phy; ///< the Phy
};

UeMemberLteUePhySapProvider::UeMemberLteUePhySapProvider (LteUePhy* phy) : m_phy (phy)
{}

void
UeMemberLteUePhySapProvider::SendMacPdu (Ptr<Packet> p)
{
  m_phy->DoSendMacPdu (p);
}

void
UeMemberLteUePhySapProvider::SendLteControlMessage (Ptr<LteControlMessage> msg)
{
  m_phy->DoSendLteControlMessage (msg);
}

void
UeMemberLteUePhySapProvider::SendRachPreamble (uint32_t prachId, uint32_t raRnti)
{
  m_phy->DoSendRachPreamble (prachId, raRnti);
}

void
UeMemberLteUePhySapProvider::NotifyConnectionSuccessful ()
{
  m_phy->DoNotifyConnectionSuccessful ();
}


////////////////////////////////////////
// LteUePhy methods
////////////////////////////////////////

/// Map each of UE PHY states to its string representation.
static const std::string g_uePhyStateName[LteUePhy::NUM_STATES] =
{
  "CELL_SEARCH",
  "SYNCHRONIZED"
};

/**
 * \param s The UE PHY state.
 * \return The string representation of the given state.
 */
static inline const std::string & ToString (LteUePhy::State s)
{
  return g_uePhyStateName[s];
}


NS_OBJECT_ENSURE_REGISTERED (LteUePhy);


LteUePhy::LteUePhy ()
{
  NS_LOG_FUNCTION (this);
  NS_FATAL_ERROR ("This constructor should not be called");
}

LteUePhy::LteUePhy (Ptr<LteSpectrumPhy> dlPhy, Ptr<LteSpectrumPhy> ulPhy)
  : LtePhy (dlPhy, ulPhy),
    m_uePhySapUser (0),
    m_ueCphySapUser (0),
    m_state (CELL_SEARCH),
    m_subframeNo (0),
    m_rsReceivedPowerUpdated (false),
    m_rsInterferencePowerUpdated (false),
    m_dataInterferencePowerUpdated (false),
    m_pssReceived (false),
    m_ueMeasurementsFilterPeriod (MilliSeconds (200)),
    m_ueMeasurementsFilterLast (MilliSeconds (0)),
    m_rsrpSinrSampleCounter (0),
    m_imsi (0)
{
  m_amc = CreateObject <LteAmc> ();
  m_powerControl = CreateObject <LteUePowerControl> ();
  m_uePhySapProvider = new UeMemberLteUePhySapProvider (this);
  m_ueCphySapProvider = new MemberLteUeCphySapProvider<LteUePhy> (this);
  m_macChTtiDelay = UL_PUSCH_TTIS_DELAY;

  NS_ASSERT_MSG (Simulator::Now ().GetNanoSeconds () == 0,
                 "Cannot create UE devices after simulation started");
  Simulator::Schedule (m_ueMeasurementsFilterPeriod, &LteUePhy::ReportUeMeasurements, this);

  DoReset ();
}


LteUePhy::~LteUePhy ()
{
  m_txModeGain.clear ();
}

void
LteUePhy::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_uePhySapProvider;
  delete m_ueCphySapProvider;
  LtePhy::DoDispose ();
}



TypeId
LteUePhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteUePhy")
    .SetParent<LtePhy> ()
    .SetGroupName ("Lte")
    .AddConstructor<LteUePhy> ()
    .AddAttribute ("TxPower",
                   "Transmission power in dBm",
                   DoubleValue (10.0),
                   MakeDoubleAccessor (&LteUePhy::SetTxPower,
                                       &LteUePhy::GetTxPower),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("NoiseFigure",
                   "Loss (dB) in the Signal-to-Noise-Ratio due to non-idealities in the receiver."
                   " According to Wikipedia (http://en.wikipedia.org/wiki/Noise_figure), this is "
                   "\"the difference in decibels (dB) between"
                   " the noise output of the actual receiver to the noise output of an "
                   " ideal receiver with the same overall gain and bandwidth when the receivers "
                   " are connected to sources at the standard noise temperature T0.\" "
                   "In this model, we consider T0 = 290K.",
                   DoubleValue (9.0),
                   MakeDoubleAccessor (&LteUePhy::SetNoiseFigure,
                                       &LteUePhy::GetNoiseFigure),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode1Gain",
                   "Transmission mode 1 gain in dB",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode1Gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode2Gain",
                   "Transmission mode 2 gain in dB",
                   DoubleValue (4.2),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode2Gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode3Gain",
                   "Transmission mode 3 gain in dB",
                   DoubleValue (-2.8),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode3Gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode4Gain",
                   "Transmission mode 4 gain in dB",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode4Gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode5Gain",
                   "Transmission mode 5 gain in dB",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode5Gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode6Gain",
                   "Transmission mode 6 gain in dB",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode6Gain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxMode7Gain",
                   "Transmission mode 7 gain in dB",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&LteUePhy::SetTxMode7Gain),
                   MakeDoubleChecker<double> ())
    .AddTraceSource ("ReportCurrentCellRsrpSinr",
                     "RSRP and SINR statistics.",
                     MakeTraceSourceAccessor (&LteUePhy::m_reportCurrentCellRsrpSinrTrace),
                     "ns3::LteUePhy::RsrpSinrTracedCallback")
    .AddAttribute ("RsrpSinrSamplePeriod",
                   "The sampling period for reporting RSRP-SINR stats (default value 1)",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteUePhy::m_rsrpSinrSamplePeriod),
                   MakeUintegerChecker<uint16_t> ())
    .AddTraceSource ("ReportUlPhyResourceBlocks",
                     "UL transmission PHY layer resource blocks.",
                     MakeTraceSourceAccessor (&LteUePhy::m_reportUlPhyResourceBlocks),
                     "ns3::LteUePhy::UlPhyResourceBlocksTracedCallback")
    .AddTraceSource ("ReportPowerSpectralDensity",
                     "Power Spectral Density data.",
                     MakeTraceSourceAccessor (&LteUePhy::m_reportPowerSpectralDensity),
                     "ns3::LteUePhy::PowerSpectralDensityTracedCallback")
    .AddTraceSource ("UlPhyTransmission",
                     "DL transmission PHY layer statistics.",
                     MakeTraceSourceAccessor (&LteUePhy::m_ulPhyTransmission),
                     "ns3::PhyTransmissionStatParameters::TracedCallback")
    .AddAttribute ("DlSpectrumPhy",
                   "The downlink LteSpectrumPhy associated to this LtePhy",
                   TypeId::ATTR_GET,
                   PointerValue (),
                   MakePointerAccessor (&LteUePhy::GetDlSpectrumPhy),
                   MakePointerChecker <LteSpectrumPhy> ())
    .AddAttribute ("UlSpectrumPhy",
                   "The uplink LteSpectrumPhy associated to this LtePhy",
                   TypeId::ATTR_GET,
                   PointerValue (),
                   MakePointerAccessor (&LteUePhy::GetUlSpectrumPhy),
                   MakePointerChecker <LteSpectrumPhy> ())
    .AddAttribute ("RsrqUeMeasThreshold",
                   "Receive threshold for PSS on RSRQ [dB]",
                   DoubleValue (-1000.0),
                   MakeDoubleAccessor (&LteUePhy::m_pssReceptionThreshold),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("UeMeasurementsFilterPeriod",
                   "Time period for reporting UE measurements, i.e., the"
                   "length of layer-1 filtering.",
                   TimeValue (MilliSeconds (200)),
                   MakeTimeAccessor (&LteUePhy::m_ueMeasurementsFilterPeriod),
                   MakeTimeChecker ())
    .AddAttribute ("DownlinkCqiPeriodicity",
                   "Periodicity in milliseconds for reporting the"
                   "wideband and subband downlink CQIs to the eNB",
                   TimeValue (MilliSeconds (1)),
                   MakeTimeAccessor (&LteUePhy::SetDownlinkCqiPeriodicity),
                   MakeTimeChecker ())
    .AddTraceSource ("ReportUeMeasurements",
                     "Report UE measurements RSRP (dBm) and RSRQ (dB).",
                     MakeTraceSourceAccessor (&LteUePhy::m_reportUeMeasurements),
                     "ns3::LteUePhy::RsrpRsrqTracedCallback")
    .AddTraceSource ("StateTransition",
                     "Trace fired upon every UE PHY state transition",
                     MakeTraceSourceAccessor (&LteUePhy::m_stateTransitionTrace),
                     "ns3::LteUePhy::StateTracedCallback")
    .AddAttribute ("EnableUplinkPowerControl",
                   "If true, Uplink Power Control will be enabled.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteUePhy::m_enableUplinkPowerControl),
                   MakeBooleanChecker ())
    .AddAttribute ("Qout",
                   "corresponds to 10% block error rate of a hypothetical PDCCH transmission"
                   "taking into account the PCFICH errors with transmission parameters."
                   "see 3GPP TS 36.213 4.2.1 and TS 36.133 7.6",
                   DoubleValue (-5),
                   MakeDoubleAccessor (&LteUePhy::m_qOut),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Qin",
                   "corresponds to 2% block error rate of a hypothetical PDCCH transmission"
                   "taking into account the PCFICH errors with transmission parameters."
                   "see 3GPP TS 36.213 4.2.1 and TS 36.133 7.6",
                   DoubleValue (-3.9),
                   MakeDoubleAccessor (&LteUePhy::m_qIn),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("NumQoutEvalSf",
                   "This specifies the total number of consecutive subframes"
                   "which corresponds to the Qout evaluation period",
                   UintegerValue (200), //see 3GPP 3GPP TS 36.133 7.6.2.1
                   MakeUintegerAccessor (&LteUePhy::SetNumQoutEvalSf,
                                         &LteUePhy::GetNumQoutEvalSf),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("NumQinEvalSf",
                   "This specifies the total number of consecutive subframes"
                   "which corresponds to the Qin evaluation period",
                   UintegerValue (100), //see 3GPP 3GPP TS 36.133 7.6.2.1
                   MakeUintegerAccessor (&LteUePhy::SetNumQinEvalSf,
                                         &LteUePhy::GetNumQinEvalSf),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("EnableRlfDetection",
                   "If true, RLF detection will be enabled.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteUePhy::m_enableRlfDetection),
                   MakeBooleanChecker ())
  ;
  return tid;
}

void
LteUePhy::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  NS_ABORT_MSG_IF (m_netDevice == nullptr, "LteNetDevice is not available in LteUePhy");
  Ptr<Node> node = m_netDevice->GetNode ();
  NS_ABORT_MSG_IF (node == nullptr, "Node is not available in the LteNetDevice of LteUePhy");
  uint32_t nodeId = node->GetId ();

  //ScheduleWithContext() is needed here to set context for logs,
  //because Initialize() is called outside of Node::AddDevice().

  Simulator::ScheduleWithContext (nodeId, Seconds (0), &LteUePhy::SubframeIndication, this, 1, 1);

  LtePhy::DoInitialize ();
}

void
LteUePhy::SetLteUePhySapUser (LteUePhySapUser* s)
{
  NS_LOG_FUNCTION (this);
  m_uePhySapUser = s;
}

LteUePhySapProvider*
LteUePhy::GetLteUePhySapProvider ()
{
  NS_LOG_FUNCTION (this);
  return (m_uePhySapProvider);
}


void
LteUePhy::SetLteUeCphySapUser (LteUeCphySapUser* s)
{
  NS_LOG_FUNCTION (this);
  m_ueCphySapUser = s;
}

LteUeCphySapProvider*
LteUePhy::GetLteUeCphySapProvider ()
{
  NS_LOG_FUNCTION (this);
  return (m_ueCphySapProvider);
}

void
LteUePhy::SetNoiseFigure (double nf)
{
  NS_LOG_FUNCTION (this << nf);
  m_noiseFigure = nf;
}

double
LteUePhy::GetNoiseFigure () const
{
  NS_LOG_FUNCTION (this);
  return m_noiseFigure;
}

void
LteUePhy::SetTxPower (double pow)
{
  NS_LOG_FUNCTION (this << pow);
  m_txPower = pow;
  m_powerControl->SetTxPower (pow);
}

double
LteUePhy::GetTxPower () const
{
  NS_LOG_FUNCTION (this);
  return m_txPower;
}

Ptr<LteUePowerControl>
LteUePhy::GetUplinkPowerControl () const
{
  NS_LOG_FUNCTION (this);
  return m_powerControl;
}

uint8_t
LteUePhy::GetMacChDelay (void) const
{
  return (m_macChTtiDelay);
}

Ptr<LteSpectrumPhy>
LteUePhy::GetDlSpectrumPhy () const
{
  return m_downlinkSpectrumPhy;
}

Ptr<LteSpectrumPhy>
LteUePhy::GetUlSpectrumPhy () const
{
  return m_uplinkSpectrumPhy;
}

void
LteUePhy::SetNumQoutEvalSf (uint16_t numSubframes)
{
  NS_LOG_FUNCTION (this << numSubframes);
  NS_ABORT_MSG_IF (numSubframes % 10 != 0, "Number of subframes used for Qout "
                   "evaluation must be multiple of 10");
  m_numOfQoutEvalSf = numSubframes;
}

void
LteUePhy::SetNumQinEvalSf (uint16_t numSubframes)
{
  NS_LOG_FUNCTION (this << numSubframes);
  NS_ABORT_MSG_IF (numSubframes % 10 != 0, "Number of subframes used for Qin "
                   "evaluation must be multiple of 10");
  m_numOfQinEvalSf = numSubframes;
}

uint16_t
LteUePhy::GetNumQoutEvalSf (void) const
{
  NS_LOG_FUNCTION (this);
  return m_numOfQoutEvalSf;
}

uint16_t
LteUePhy::GetNumQinEvalSf (void) const
{
  NS_LOG_FUNCTION (this);
  return m_numOfQinEvalSf;
}

void
LteUePhy::DoSendMacPdu (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this);

  SetMacPdu (p);
}


void
LteUePhy::PhyPduReceived (Ptr<Packet> p)
{
  m_uePhySapUser->ReceivePhyPdu (p);
}

void
LteUePhy::SetSubChannelsForTransmission (std::vector <int> mask)
{
  NS_LOG_FUNCTION (this);

  m_subChannelsForTransmission = mask;

  Ptr<SpectrumValue> txPsd = CreateTxPowerSpectralDensity ();
  m_uplinkSpectrumPhy->SetTxPowerSpectralDensity (txPsd);
}


void
LteUePhy::SetSubChannelsForReception (std::vector <int> mask)
{
  NS_LOG_FUNCTION (this);
  m_subChannelsForReception = mask;
}


std::vector <int>
LteUePhy::GetSubChannelsForTransmission ()
{
  NS_LOG_FUNCTION (this);
  return m_subChannelsForTransmission;
}


std::vector <int>
LteUePhy::GetSubChannelsForReception ()
{
  NS_LOG_FUNCTION (this);
  return m_subChannelsForReception;
}


Ptr<SpectrumValue>
LteUePhy::CreateTxPowerSpectralDensity ()
{
  NS_LOG_FUNCTION (this);
  LteSpectrumValueHelper psdHelper;
  Ptr<SpectrumValue> psd = psdHelper.CreateUlTxPowerSpectralDensity (m_ulEarfcn, m_ulBandwidth, m_txPower, GetSubChannelsForTransmission ());
  m_reportPowerSpectralDensity (m_rnti, psd);

  return psd;
}

void
LteUePhy::GenerateCtrlCqiReport (const SpectrumValue& sinr)
{
  NS_LOG_FUNCTION (this);
  /**
   * We do not generate the CQI report
   * when the UE is not synchronized to any cell.
   *
   * Also, the RLF is detected after the DL CTRL
   * is received by the UE,therefore, we do not need
   * to generate the CQI reports and the UE measurements
   * for a CTRL for which the RLF has been detected.
   */
  if (m_cellId == 0)
    {
      return;
    }
  m_ctrlSinrForRlf = sinr;
  GenerateCqiRsrpRsrq (sinr);
}

void
LteUePhy::GenerateCqiRsrpRsrq (const SpectrumValue& sinr)
{
  NS_LOG_FUNCTION (this << sinr);

  NS_ASSERT (m_state != CELL_SEARCH);
  NS_ASSERT (m_cellId > 0);

  if (m_dlConfigured && m_ulConfigured && (m_rnti > 0))
    {
      // check periodic wideband CQI
      if (Simulator::Now () > m_p10CqiLast + m_p10CqiPeriodicity)
        {
          NS_LOG_DEBUG ("Reporting P10 CQI at : " << Simulator::Now ().As (Time::MS)
                                                  << ". Last reported at : " << m_p10CqiLast.As (Time::MS));
          Ptr<LteUeNetDevice> thisDevice = GetDevice ()->GetObject<LteUeNetDevice> ();
          Ptr<DlCqiLteControlMessage> msg = CreateDlCqiFeedbackMessage (sinr);
          if (msg)
            {
              DoSendLteControlMessage (msg);
            }
          m_p10CqiLast = Simulator::Now ();
        }
      // check aperiodic high-layer configured subband CQI
      if  (Simulator::Now () > m_a30CqiLast + m_a30CqiPeriodicity)
        {
          NS_LOG_DEBUG ("Reporting A30 CQI at : " << Simulator::Now ().As (Time::MS)
                                                  << ". Last reported at : " << m_a30CqiLast.As (Time::MS));
          Ptr<LteUeNetDevice> thisDevice = GetDevice ()->GetObject<LteUeNetDevice> ();
          Ptr<DlCqiLteControlMessage> msg = CreateDlCqiFeedbackMessage (sinr);
          if (msg)
            {
              DoSendLteControlMessage (msg);
            }
          m_a30CqiLast = Simulator::Now ();
        }
    }

  // Generate PHY trace
  m_rsrpSinrSampleCounter++;
  if (m_rsrpSinrSampleCounter == m_rsrpSinrSamplePeriod)
    {
      NS_ASSERT_MSG (m_rsReceivedPowerUpdated, " RS received power info obsolete");
      // RSRP evaluated as averaged received power among RBs
      double sum = 0.0;
      uint8_t rbNum = 0;
      Values::const_iterator it;
      for (it = m_rsReceivedPower.ConstValuesBegin (); it != m_rsReceivedPower.ConstValuesEnd (); it++)
        {
          // convert PSD [W/Hz] to linear power [W] for the single RE
          // we consider only one RE for the RS since the channel is
          // flat within the same RB
          double powerTxW = ((*it) * 180000.0) / 12.0;
          sum += powerTxW;
          rbNum++;
        }
      double rsrp = (rbNum > 0) ? (sum / rbNum) : DBL_MAX;
      // averaged SINR among RBs
      double avSinr = ComputeAvgSinr (sinr);

      NS_LOG_INFO (this << " cellId " << m_cellId << " rnti " << m_rnti << " RSRP " << rsrp << " SINR " << avSinr << " ComponentCarrierId " << (uint16_t) m_componentCarrierId);
      //trigger RLF detection only when UE has an active RRC connection
      //and RLF detection attribute is set to true
      if (m_isConnected && m_enableRlfDetection)
        {
          double avrgSinrForRlf = ComputeAvgSinr (m_ctrlSinrForRlf);
          RlfDetection (10 * log10 (avrgSinrForRlf));
        }

      m_reportCurrentCellRsrpSinrTrace (m_cellId, m_rnti, rsrp, avSinr, (uint16_t) m_componentCarrierId);
      m_rsrpSinrSampleCounter = 0;
    }

  if (m_pssReceived)
    {
      // measure instantaneous RSRQ now
      NS_ASSERT_MSG (m_rsInterferencePowerUpdated, " RS interference power info obsolete");

      std::list <PssElement>::iterator itPss = m_pssList.begin ();
      while (itPss != m_pssList.end ())
        {
          uint16_t rbNum = 0;
          double rssiSum = 0.0;

          Values::const_iterator itIntN = m_rsInterferencePower.ConstValuesBegin ();
          Values::const_iterator itPj = m_rsReceivedPower.ConstValuesBegin ();
          for (itPj = m_rsReceivedPower.ConstValuesBegin ();
               itPj != m_rsReceivedPower.ConstValuesEnd ();
               itIntN++, itPj++)
            {
              rbNum++;
              // convert PSD [W/Hz] to linear power [W] for the single RE
              double interfPlusNoisePowerTxW = ((*itIntN) * 180000.0) / 12.0;
              double signalPowerTxW = ((*itPj) * 180000.0) / 12.0;
              rssiSum += (2 * (interfPlusNoisePowerTxW + signalPowerTxW));
            }

          NS_ASSERT (rbNum == (*itPss).nRB);
          double rsrq_dB = 10 * log10 ((*itPss).pssPsdSum / rssiSum);

          if (rsrq_dB > m_pssReceptionThreshold)
            {
              NS_LOG_INFO (this << " PSS RNTI " << m_rnti << " cellId " << m_cellId
                                << " has RSRQ " << rsrq_dB << " and RBnum " << rbNum);
              // store measurements
              std::map <uint16_t, UeMeasurementsElement>::iterator itMeasMap;
              itMeasMap = m_ueMeasurementsMap.find ((*itPss).cellId);
              if (itMeasMap != m_ueMeasurementsMap.end ())
                {
                  (*itMeasMap).second.rsrqSum += rsrq_dB;
                  (*itMeasMap).second.rsrqNum++;
                }
              else
                {
                  NS_LOG_WARN ("race condition of bug 2091 occurred");
                }
            }

          itPss++;

        } // end of while (itPss != m_pssList.end ())

      m_pssList.clear ();

    } // end of if (m_pssReceived)

} // end of void LteUePhy::GenerateCtrlCqiReport (const SpectrumValue& sinr)

double
LteUePhy::ComputeAvgSinr (const SpectrumValue& sinr)
{
  NS_LOG_FUNCTION (this);

  // averaged SINR among RBs
  double sum = 0.0;
  uint8_t rbNum = 0;
  Values::const_iterator it;

  for (it = sinr.ConstValuesBegin (); it != sinr.ConstValuesEnd (); it++)
    {
      sum += (*it);
      rbNum++;
    }

  double avrgSinr = (rbNum > 0) ? (sum / rbNum) : DBL_MAX;

  return avrgSinr;
}

void
LteUePhy::GenerateDataCqiReport (const SpectrumValue& sinr)
{
  // Not used by UE, CQI are based only on RS
}

void
LteUePhy::GenerateMixedCqiReport (const SpectrumValue& sinr)
{
  NS_LOG_FUNCTION (this);

  /**
   * We do not generate the CQI report
   * when the UE is not synchronized to any cell.
   *
   * Also, the RLF is detected after the DL CTRL
   * is received by the UE,therefore, we do not need
   * to generate the CQI reports and the UE measurements
   * for a CTRL for which the RLF has been detected.
   */
  if (m_cellId == 0)
    {
      return;
    }

  NS_ASSERT (m_state != CELL_SEARCH);
  //NOTE: The SINR received by this method is
  //based on CTRL, which is not used to compute
  //PDSCH (i.e., data) based SINR. It is used
  //for RLF detection.
  m_ctrlSinrForRlf = sinr;

  SpectrumValue mixedSinr = (m_rsReceivedPower * m_paLinear);
  if (m_dataInterferencePowerUpdated)
    {
      // we have a measurement of interf + noise for the denominator
      // of SINR = S/(I+N)
      mixedSinr /= m_dataInterferencePower;
      m_dataInterferencePowerUpdated = false;
      NS_LOG_LOGIC ("data interf measurement available, SINR = " << mixedSinr);
    }
  else
    {
      // we did not see any interference on data, so interference is
      // there and we have only noise at the denominator of SINR
      mixedSinr /= (*m_noisePsd);
      NS_LOG_LOGIC ("no data interf measurement available, SINR = " << mixedSinr);
    }

  /*
   * some RBs are not used in PDSCH and their SINR is very high
   * for example with bandwidth 25, last RB is not used
   * it can make avgSinr value very high, what is incorrect
   */
  uint32_t rbgSize = GetRbgSize ();
  uint32_t modulo = m_dlBandwidth % rbgSize;
  double avgMixedSinr = 0;
  uint32_t usedRbgNum = 0;
  for (uint32_t i = 0; i < (m_dlBandwidth - 1 - modulo); i++)
    {
      usedRbgNum++;
      avgMixedSinr += mixedSinr[i];
    }
  avgMixedSinr = avgMixedSinr / usedRbgNum;
  for (uint32_t i = 0; i < modulo; i++)
    {
      mixedSinr[m_dlBandwidth - 1 - i] = avgMixedSinr;
    }

  GenerateCqiRsrpRsrq (mixedSinr);
}

void
LteUePhy::ReportInterference (const SpectrumValue& interf)
{
  NS_LOG_FUNCTION (this << interf);
  m_rsInterferencePowerUpdated = true;
  m_rsInterferencePower = interf;
}

void
LteUePhy::ReportDataInterference (const SpectrumValue& interf)
{
  NS_LOG_FUNCTION (this << interf);

  m_dataInterferencePowerUpdated = true;
  m_dataInterferencePower = interf;
}

void
LteUePhy::ReportRsReceivedPower (const SpectrumValue& power)
{
  NS_LOG_FUNCTION (this << power);
  m_rsReceivedPowerUpdated = true;
  m_rsReceivedPower = power;

  if (m_enableUplinkPowerControl)
    {
      double sum = 0;
      uint32_t rbNum = 0;
      Values::const_iterator it;
      for (it = m_rsReceivedPower.ConstValuesBegin (); it != m_rsReceivedPower.ConstValuesEnd (); it++)
        {
          double powerTxW = ((*it) * 180000);
          sum += powerTxW;
          rbNum++;
        }
      double rsrp = 10 * log10 (sum) + 30;

      NS_LOG_INFO ("RSRP: " << rsrp);
      m_powerControl->SetRsrp (rsrp);
    }
}

Ptr<DlCqiLteControlMessage>
LteUePhy::CreateDlCqiFeedbackMessage (const SpectrumValue& sinr)
{
  NS_LOG_FUNCTION (this);


  // apply transmission mode gain
  NS_ASSERT (m_transmissionMode < m_txModeGain.size ());
  SpectrumValue newSinr = sinr;
  newSinr *= m_txModeGain.at (m_transmissionMode);

  // CREATE DlCqiLteControlMessage
  Ptr<DlCqiLteControlMessage> msg = Create<DlCqiLteControlMessage> ();
  CqiListElement_s dlcqi;
  std::vector<int> cqi;
  if (Simulator::Now () > m_p10CqiLast + m_p10CqiPeriodicity)
    {
      cqi = m_amc->CreateCqiFeedbacks (newSinr, m_dlBandwidth);

      int nLayer = TransmissionModesLayers::TxMode2LayerNum (m_transmissionMode);
      int nbSubChannels = cqi.size ();
      double cqiSum = 0.0;
      int activeSubChannels = 0;
      // average the CQIs of the different RBs
      for (int i = 0; i < nbSubChannels; i++)
        {
          if (cqi.at (i) != -1)
            {
              cqiSum += cqi.at (i);
              activeSubChannels++;
            }
          NS_LOG_DEBUG (this << " subch " << i << " cqi " <<  cqi.at (i));
        }
      dlcqi.m_rnti = m_rnti;
      dlcqi.m_ri = 1; // not yet used
      dlcqi.m_cqiType = CqiListElement_s::P10; // Peridic CQI using PUCCH wideband
      NS_ASSERT_MSG (nLayer > 0, " nLayer negative");
      NS_ASSERT_MSG (nLayer < 3, " nLayer limit is 2s");
      for (int i = 0; i < nLayer; i++)
        {
          if (activeSubChannels > 0)
            {
              dlcqi.m_wbCqi.push_back ((uint16_t) cqiSum / activeSubChannels);
            }
          else
            {
              // approximate with the worst case -> CQI = 1
              dlcqi.m_wbCqi.push_back (1);
            }
        }
      //NS_LOG_DEBUG (this << " Generate P10 CQI feedback " << (uint16_t) cqiSum / activeSubChannels);
      dlcqi.m_wbPmi = 0; // not yet used
      // dl.cqi.m_sbMeasResult others CQI report modes: not yet implemented
    }
  else if (Simulator::Now () > m_a30CqiLast + m_a30CqiPeriodicity)
    {
      cqi = m_amc->CreateCqiFeedbacks (newSinr, GetRbgSize ());
      int nLayer = TransmissionModesLayers::TxMode2LayerNum (m_transmissionMode);
      int nbSubChannels = cqi.size ();
      int rbgSize = GetRbgSize ();
      double cqiSum = 0.0;
      int cqiNum = 0;
      SbMeasResult_s rbgMeas;
      //NS_LOG_DEBUG (this << " Create A30 CQI feedback, RBG " << rbgSize << " cqiNum " << nbSubChannels << " band "  << (uint16_t)m_dlBandwidth);
      for (int i = 0; i < nbSubChannels; i++)
        {
          if (cqi.at (i) != -1)
            {
              cqiSum += cqi.at (i);
            }
          // else "nothing" no CQI is treated as CQI = 0 (worst case scenario)
          cqiNum++;
          if (cqiNum == rbgSize)
            {
              // average the CQIs of the different RBGs
              //NS_LOG_DEBUG (this << " RBG CQI "  << (uint16_t) cqiSum / rbgSize);
              HigherLayerSelected_s hlCqi;
              hlCqi.m_sbPmi = 0; // not yet used
              for (int i = 0; i < nLayer; i++)
                {
                  hlCqi.m_sbCqi.push_back ((uint16_t) cqiSum / rbgSize);
                }
              rbgMeas.m_higherLayerSelected.push_back (hlCqi);
              cqiSum = 0.0;
              cqiNum = 0;
            }
        }
      dlcqi.m_rnti = m_rnti;
      dlcqi.m_ri = 1; // not yet used
      dlcqi.m_cqiType = CqiListElement_s::A30; // Aperidic CQI using PUSCH
      //dlcqi.m_wbCqi.push_back ((uint16_t) cqiSum / nbSubChannels);
      dlcqi.m_wbPmi = 0; // not yet used
      dlcqi.m_sbMeasResult = rbgMeas;
    }

  msg->SetDlCqi (dlcqi);
  return msg;
}


void
LteUePhy::ReportUeMeasurements ()
{
  NS_LOG_FUNCTION (this << Simulator::Now ());
  NS_LOG_DEBUG (this << " Report UE Measurements ");

  LteUeCphySapUser::UeMeasurementsParameters ret;

  std::map <uint16_t, UeMeasurementsElement>::iterator it;
  for (it = m_ueMeasurementsMap.begin (); it != m_ueMeasurementsMap.end (); it++)
    {
      double avg_rsrp = (*it).second.rsrpSum / (double)(*it).second.rsrpNum;
      double avg_rsrq = (*it).second.rsrqSum / (double)(*it).second.rsrqNum;
      /*
       * In CELL_SEARCH state, this may result in avg_rsrq = 0/0 = -nan.
       * UE RRC must take this into account when receiving measurement reports.
       * TODO remove this shortcoming by calculating RSRQ during CELL_SEARCH
       */
      NS_LOG_DEBUG (this << " CellId " << (*it).first
                         << " RSRP " << avg_rsrp
                         << " (nSamples " << (uint16_t)(*it).second.rsrpNum << ")"
                         << " RSRQ " << avg_rsrq
                         << " (nSamples " << (uint16_t)(*it).second.rsrqNum << ")"
                         << " ComponentCarrierID " << (uint16_t)m_componentCarrierId);

      LteUeCphySapUser::UeMeasurementsElement newEl;
      newEl.m_cellId = (*it).first;
      newEl.m_rsrp = avg_rsrp;
      newEl.m_rsrq = avg_rsrq;
      ret.m_ueMeasurementsList.push_back (newEl);
      ret.m_componentCarrierId = m_componentCarrierId;

      // report to UE measurements trace
      m_reportUeMeasurements (m_rnti, (*it).first, avg_rsrp, avg_rsrq, ((*it).first == m_cellId ? 1 : 0), m_componentCarrierId);
    }

  // report to RRC
  m_ueCphySapUser->ReportUeMeasurements (ret);

  m_ueMeasurementsMap.clear ();
  Simulator::Schedule (m_ueMeasurementsFilterPeriod, &LteUePhy::ReportUeMeasurements, this);
}

void
LteUePhy::SetDownlinkCqiPeriodicity (Time cqiPeriodicity)
{
  NS_LOG_FUNCTION (this << cqiPeriodicity);
  m_a30CqiPeriodicity = cqiPeriodicity;
  m_p10CqiPeriodicity = cqiPeriodicity;
}

void
LteUePhy::DoSendLteControlMessage (Ptr<LteControlMessage> msg)
{
  NS_LOG_FUNCTION (this << msg);

  SetControlMessages (msg);
}

void
LteUePhy::DoSendRachPreamble (uint32_t raPreambleId, uint32_t raRnti)
{
  NS_LOG_FUNCTION (this << raPreambleId);

  // unlike other control messages, RACH preamble is sent ASAP
  Ptr<RachPreambleLteControlMessage> msg = Create<RachPreambleLteControlMessage> ();
  msg->SetRapId (raPreambleId);
  m_raPreambleId = raPreambleId;
  m_raRnti = raRnti;
  m_controlMessagesQueue.at (0).push_back (msg);
}

void
LteUePhy::DoNotifyConnectionSuccessful ()
{
  /**
   * Radio link failure detection should take place only on the
   * primary carrier to avoid errors due to multiple calls to the
   * same methods at the RRC layer
   */
  if (m_componentCarrierId == 0)
    {
      m_isConnected = true;
      // Initialize the parameters for radio link failure detection
      InitializeRlfParams ();
    }
}



void
LteUePhy::ReceiveLteControlMessageList (std::list<Ptr<LteControlMessage> > msgList)
{
  NS_LOG_FUNCTION (this);

  std::list<Ptr<LteControlMessage> >::iterator it;
  NS_LOG_DEBUG (this << " I am rnti = " << m_rnti << " and I received msgs " << (uint16_t) msgList.size ());
  for (it = msgList.begin (); it != msgList.end (); it++)
    {
      Ptr<LteControlMessage> msg = (*it);

      if (msg->GetMessageType () == LteControlMessage::DL_DCI)
        {
          Ptr<DlDciLteControlMessage> msg2 = DynamicCast<DlDciLteControlMessage> (msg);

          DlDciListElement_s dci = msg2->GetDci ();
          if (dci.m_rnti != m_rnti)
            {
              // DCI not for me
              continue;
            }

          if (dci.m_resAlloc != 0)
            {
              NS_FATAL_ERROR ("Resource Allocation type not implemented");
            }

          std::vector <int> dlRb;

          // translate the DCI to Spectrum framework
          uint32_t mask = 0x1;
          for (int i = 0; i < 32; i++)
            {
              if (((dci.m_rbBitmap & mask) >> i) == 1)
                {
                  for (int k = 0; k < GetRbgSize (); k++)
                    {
                      dlRb.push_back ((i * GetRbgSize ()) + k);
//             NS_LOG_DEBUG(this << " RNTI " << m_rnti << " RBG " << i << " DL-DCI allocated PRB " << (i*GetRbgSize()) + k);
                    }
                }
              mask = (mask << 1);
            }
          if (m_enableUplinkPowerControl)
            {
              m_powerControl->ReportTpc (dci.m_tpc);
            }


          // send TB info to LteSpectrumPhy
          NS_LOG_DEBUG (this << " UE " << m_rnti << " DL-DCI " << dci.m_rnti << " bitmap "  << dci.m_rbBitmap);
          for (uint8_t i = 0; i < dci.m_tbsSize.size (); i++)
            {
              m_downlinkSpectrumPhy->AddExpectedTb (dci.m_rnti, dci.m_ndi.at (i), dci.m_tbsSize.at (i), dci.m_mcs.at (i), dlRb, i, dci.m_harqProcess, dci.m_rv.at (i), true /* DL */);
            }

          SetSubChannelsForReception (dlRb);


        }
      else if (msg->GetMessageType () == LteControlMessage::UL_DCI)
        {
          // set the uplink bandwidth according to the UL-CQI
          Ptr<UlDciLteControlMessage> msg2 = DynamicCast<UlDciLteControlMessage> (msg);
          UlDciListElement_s dci = msg2->GetDci ();
          if (dci.m_rnti != m_rnti)
            {
              // DCI not for me
              continue;
            }
          NS_LOG_INFO (this << " UL DCI");
          std::vector <int> ulRb;
          for (int i = 0; i < dci.m_rbLen; i++)
            {
              ulRb.push_back (i + dci.m_rbStart);
              //NS_LOG_DEBUG (this << " UE RB " << i + dci.m_rbStart);
            }
          m_reportUlPhyResourceBlocks (m_rnti, ulRb);
          QueueSubChannelsForTransmission (ulRb);
          // fire trace of UL Tx PHY stats
          HarqProcessInfoList_t harqInfoList = m_harqPhyModule->GetHarqProcessInfoUl (m_rnti, 0);
          PhyTransmissionStatParameters params;
          params.m_cellId = m_cellId;
          params.m_imsi = 0; // it will be set by DlPhyTransmissionCallback in LteHelper
          params.m_timestamp = Simulator::Now ().GetMilliSeconds () + UL_PUSCH_TTIS_DELAY;
          params.m_rnti = m_rnti;
          params.m_txMode = 0; // always SISO for UE
          params.m_layer = 0;
          params.m_mcs = dci.m_mcs;
          params.m_size = dci.m_tbSize;
          params.m_rv = harqInfoList.size ();
          params.m_ndi = dci.m_ndi;
          params.m_ccId = m_componentCarrierId;
          m_ulPhyTransmission (params);
          // pass the info to the MAC
          m_uePhySapUser->ReceiveLteControlMessage (msg);
        }
      else if (msg->GetMessageType () == LteControlMessage::RAR)
        {
          Ptr<RarLteControlMessage> rarMsg = DynamicCast<RarLteControlMessage> (msg);
          if (rarMsg->GetRaRnti () == m_raRnti)
            {
              for (std::list<RarLteControlMessage::Rar>::const_iterator it = rarMsg->RarListBegin (); it != rarMsg->RarListEnd (); ++it)
                {
                  if (it->rapId != m_raPreambleId)
                    {
                      // UL grant not for me
                      continue;
                    }
                  else
                    {
                      NS_LOG_INFO ("received RAR RNTI " << m_raRnti);
                      // set the uplink bandwidth according to the UL grant
                      std::vector <int> ulRb;
                      for (int i = 0; i < it->rarPayload.m_grant.m_rbLen; i++)
                        {
                          ulRb.push_back (i + it->rarPayload.m_grant.m_rbStart);
                        }

                      QueueSubChannelsForTransmission (ulRb);
                      // pass the info to the MAC
                      m_uePhySapUser->ReceiveLteControlMessage (msg);
                      // reset RACH variables with out of range values
                      m_raPreambleId = 255;
                      m_raRnti = 11;
                    }
                }
            }
        }
      else if (msg->GetMessageType () == LteControlMessage::MIB)
        {
          NS_LOG_INFO ("received MIB");
          NS_ASSERT (m_cellId > 0);
          Ptr<MibLteControlMessage> msg2 = DynamicCast<MibLteControlMessage> (msg);
          m_ueCphySapUser->RecvMasterInformationBlock (m_cellId, msg2->GetMib ());
        }
      else if (msg->GetMessageType () == LteControlMessage::SIB1)
        {
          NS_LOG_INFO ("received SIB1");
          NS_ASSERT (m_cellId > 0);
          Ptr<Sib1LteControlMessage> msg2 = DynamicCast<Sib1LteControlMessage> (msg);
          m_ueCphySapUser->RecvSystemInformationBlockType1 (m_cellId, msg2->GetSib1 ());
        }
      else
        {
          // pass the message to UE-MAC
          m_uePhySapUser->ReceiveLteControlMessage (msg);
        }

    }


}


void
LteUePhy::ReceivePss (uint16_t cellId, Ptr<SpectrumValue> p)
{
  NS_LOG_FUNCTION (this << cellId << (*p));

  double sum = 0.0;
  uint16_t nRB = 0;
  Values::const_iterator itPi;
  for (itPi = p->ConstValuesBegin (); itPi != p->ConstValuesEnd (); itPi++)
    {
      // convert PSD [W/Hz] to linear power [W] for the single RE
      double powerTxW = ((*itPi) * 180000.0) / 12.0;
      sum += powerTxW;
      nRB++;
    }

  // measure instantaneous RSRP now
  double rsrp_dBm = 10 * log10 (1000 * (sum / (double)nRB));
  NS_LOG_INFO (this << " PSS RNTI " << m_rnti << " cellId " << m_cellId
                    << " has RSRP " << rsrp_dBm << " and RBnum " << nRB);
  // note that m_pssReceptionThreshold does not apply here

  // store measurements
  std::map <uint16_t, UeMeasurementsElement>::iterator itMeasMap = m_ueMeasurementsMap.find (cellId);
  if (itMeasMap == m_ueMeasurementsMap.end ())
    {
      // insert new entry
      UeMeasurementsElement newEl;
      newEl.rsrpSum = rsrp_dBm;
      newEl.rsrpNum = 1;
      newEl.rsrqSum = 0;
      newEl.rsrqNum = 0;
      m_ueMeasurementsMap.insert (std::pair <uint16_t, UeMeasurementsElement> (cellId, newEl));
    }
  else
    {
      (*itMeasMap).second.rsrpSum += rsrp_dBm;
      (*itMeasMap).second.rsrpNum++;
    }

  /*
   * Collect the PSS for later processing in GenerateCtrlCqiReport()
   * (to be called from ChunkProcessor after RX is finished).
   */
  m_pssReceived = true;
  PssElement el;
  el.cellId = cellId;
  el.pssPsdSum = sum;
  el.nRB = nRB;
  m_pssList.push_back (el);

} // end of void LteUePhy::ReceivePss (uint16_t cellId, Ptr<SpectrumValue> p)


void
LteUePhy::QueueSubChannelsForTransmission (std::vector <int> rbMap)
{
  m_subChannelsForTransmissionQueue.at (m_macChTtiDelay - 1) = rbMap;
}


void
LteUePhy::SubframeIndication (uint32_t frameNo, uint32_t subframeNo)
{
  NS_LOG_FUNCTION (this << frameNo << subframeNo);

  NS_ASSERT_MSG (frameNo > 0, "the SRS index check code assumes that frameNo starts at 1");

  // refresh internal variables
  m_rsReceivedPowerUpdated = false;
  m_rsInterferencePowerUpdated = false;
  m_pssReceived = false;

  if (m_ulConfigured)
    {
      // update uplink transmission mask according to previous UL-CQIs
      std::vector <int> rbMask = m_subChannelsForTransmissionQueue.at (0);
      SetSubChannelsForTransmission (m_subChannelsForTransmissionQueue.at (0));

      // shift the queue
      for (uint8_t i = 1; i < m_macChTtiDelay; i++)
        {
          m_subChannelsForTransmissionQueue.at (i - 1) = m_subChannelsForTransmissionQueue.at (i);
        }
      m_subChannelsForTransmissionQueue.at (m_macChTtiDelay - 1).clear ();

      if (m_srsConfigured && (m_srsStartTime <= Simulator::Now ()))
        {

          NS_ASSERT_MSG (subframeNo > 0 && subframeNo <= 10, "the SRS index check code assumes that subframeNo starts at 1");
          if ((((frameNo - 1) * 10 + (subframeNo - 1)) % m_srsPeriodicity) == m_srsSubframeOffset)
            {
              NS_LOG_INFO ("frame " << frameNo << " subframe " << subframeNo << " sending SRS (offset=" << m_srsSubframeOffset << ", period=" << m_srsPeriodicity << ")");
              m_sendSrsEvent = Simulator::Schedule (UL_SRS_DELAY_FROM_SUBFRAME_START,
                                                    &LteUePhy::SendSrs,
                                                    this);
            }
        }

      std::list<Ptr<LteControlMessage> > ctrlMsg = GetControlMessages ();
      // send packets in queue
      NS_LOG_LOGIC (this << " UE - start slot for PUSCH + PUCCH - RNTI " << m_rnti << " CELLID " << m_cellId);
      // send the current burts of packets
      Ptr<PacketBurst> pb = GetPacketBurst ();
      if (pb)
        {
          if (m_enableUplinkPowerControl)
            {
              m_txPower = m_powerControl->GetPuschTxPower (rbMask);
              SetSubChannelsForTransmission (rbMask);
            }
          m_uplinkSpectrumPhy->StartTxDataFrame (pb, ctrlMsg, UL_DATA_DURATION);
        }
      else
        {
          // send only PUCCH (ideal: fake null bandwidth signal)
          if (ctrlMsg.size () > 0)
            {
              NS_LOG_LOGIC (this << " UE - start TX PUCCH (NO PUSCH)");
              std::vector <int> dlRb;

              if (m_enableUplinkPowerControl)
                {
                  m_txPower = m_powerControl->GetPucchTxPower (dlRb);
                }

              SetSubChannelsForTransmission (dlRb);
              m_uplinkSpectrumPhy->StartTxDataFrame (pb, ctrlMsg, UL_DATA_DURATION);
            }
          else
            {
              NS_LOG_LOGIC (this << " UE - UL NOTHING TO SEND");
            }
        }
    }  // m_configured

  // trigger the MAC
  m_uePhySapUser->SubframeIndication (frameNo, subframeNo);

  m_subframeNo = subframeNo;
  ++subframeNo;
  if (subframeNo > 10)
    {
      ++frameNo;
      subframeNo = 1;
    }

  // schedule next subframe indication
  Simulator::Schedule (Seconds (GetTti ()), &LteUePhy::SubframeIndication, this, frameNo, subframeNo);
}


void
LteUePhy::SendSrs ()
{
  NS_LOG_FUNCTION (this << " UE " << m_rnti << " start tx SRS, cell Id " << (uint32_t) m_cellId);
  NS_ASSERT (m_cellId > 0);
  // set the current tx power spectral density (full bandwidth)
  std::vector <int> dlRb;
  for (uint8_t i = 0; i < m_ulBandwidth; i++)
    {
      dlRb.push_back (i);
    }

  if (m_enableUplinkPowerControl)
    {
      m_txPower = m_powerControl->GetSrsTxPower (dlRb);
    }

  SetSubChannelsForTransmission (dlRb);
  m_uplinkSpectrumPhy->StartTxUlSrsFrame ();
}


void
LteUePhy::DoReset ()
{
  NS_LOG_FUNCTION (this);

  m_rnti = 0;
  m_cellId = 0;
  m_isConnected = false;
  m_transmissionMode = 0;
  m_srsPeriodicity = 0;
  m_srsConfigured = false;
  m_dlConfigured = false;
  m_ulConfigured = false;
  m_raPreambleId = 255; // value out of range
  m_raRnti = 11; // value out of range
  m_rsrpSinrSampleCounter = 0;
  m_p10CqiLast = Simulator::Now ();
  m_a30CqiLast = Simulator::Now ();
  m_paLinear = 1;

  m_rsReceivedPowerUpdated = false;
  m_rsInterferencePowerUpdated = false;
  m_dataInterferencePowerUpdated = false;

  m_packetBurstQueue.clear ();
  m_controlMessagesQueue.clear ();
  m_subChannelsForTransmissionQueue.clear ();
  for (int i = 0; i < m_macChTtiDelay; i++)
    {
      Ptr<PacketBurst> pb = CreateObject <PacketBurst> ();
      m_packetBurstQueue.push_back (pb);
      std::list<Ptr<LteControlMessage> > l;
      m_controlMessagesQueue.push_back (l);
    }
  std::vector <int> ulRb;
  m_subChannelsForTransmissionQueue.resize (m_macChTtiDelay, ulRb);

  m_sendSrsEvent.Cancel ();
  m_downlinkSpectrumPhy->Reset ();
  m_uplinkSpectrumPhy->Reset ();
  m_pssList.clear ();
  /**
   * Call the EndRx() method of the interference model for DL control and data
   * to cancel any ongoing downlink reception of control and data info.
   */
  m_downlinkSpectrumPhy->m_interferenceCtrl->EndRx ();
  m_downlinkSpectrumPhy->m_interferenceData->EndRx ();

} // end of void LteUePhy::DoReset ()

void
LteUePhy::DoStartCellSearch (uint32_t dlEarfcn)
{
  NS_LOG_FUNCTION (this << dlEarfcn);
  m_dlEarfcn = dlEarfcn;
  DoSetDlBandwidth (6); // configure DL for receiving PSS
  SwitchToState (CELL_SEARCH);
}

void
LteUePhy::DoSynchronizeWithEnb (uint16_t cellId, uint32_t dlEarfcn)
{
  NS_LOG_FUNCTION (this << cellId << dlEarfcn);
  m_dlEarfcn = dlEarfcn;
  DoSynchronizeWithEnb (cellId);
}

void
LteUePhy::DoSynchronizeWithEnb (uint16_t cellId)
{
  NS_LOG_FUNCTION (this << cellId);

  if (cellId == 0)
    {
      NS_FATAL_ERROR ("Cell ID shall not be zero");
    }

  m_cellId = cellId;
  m_downlinkSpectrumPhy->SetCellId (cellId);
  m_uplinkSpectrumPhy->SetCellId (cellId);

  // configure DL for receiving the BCH with the minimum bandwidth
  DoSetDlBandwidth (6);

  m_dlConfigured = false;
  m_ulConfigured = false;

  SwitchToState (SYNCHRONIZED);
}

uint16_t
LteUePhy::DoGetCellId ()
{
  return m_cellId;
}

uint32_t
LteUePhy::DoGetDlEarfcn ()
{
  return m_dlEarfcn;
}

void
LteUePhy::DoSetDlBandwidth (uint16_t dlBandwidth)
{
  NS_LOG_FUNCTION (this << (uint32_t) dlBandwidth);
  if (m_dlBandwidth != dlBandwidth or !m_dlConfigured)
    {
      m_dlBandwidth = dlBandwidth;

      static const int Type0AllocationRbg[4] = {
        10,     // RGB size 1
        26,     // RGB size 2
        63,     // RGB size 3
        110     // RGB size 4
      };  // see table 7.1.6.1-1 of 36.213
      for (int i = 0; i < 4; i++)
        {
          if (dlBandwidth < Type0AllocationRbg[i])
            {
              m_rbgSize = i + 1;
              break;
            }
        }

      m_noisePsd = LteSpectrumValueHelper::CreateNoisePowerSpectralDensity (m_dlEarfcn, m_dlBandwidth, m_noiseFigure);
      m_downlinkSpectrumPhy->SetNoisePowerSpectralDensity (m_noisePsd);
      m_downlinkSpectrumPhy->GetChannel ()->AddRx (m_downlinkSpectrumPhy);
    }
  m_dlConfigured = true;
}


void
LteUePhy::DoConfigureUplink (uint32_t ulEarfcn, uint16_t ulBandwidth)
{
  m_ulEarfcn = ulEarfcn;
  m_ulBandwidth = ulBandwidth;
  m_ulConfigured = true;
}

void
LteUePhy::DoConfigureReferenceSignalPower (int8_t referenceSignalPower)
{
  NS_LOG_FUNCTION (this);
  m_powerControl->ConfigureReferenceSignalPower (referenceSignalPower);
}

void
LteUePhy::DoSetRnti (uint16_t rnti)
{
  NS_LOG_FUNCTION (this << rnti);
  m_rnti = rnti;

  m_powerControl->SetCellId (m_cellId);
  m_powerControl->SetRnti (m_rnti);
}

void
LteUePhy::DoSetTransmissionMode (uint8_t txMode)
{
  NS_LOG_FUNCTION (this << (uint16_t)txMode);
  m_transmissionMode = txMode;
  m_downlinkSpectrumPhy->SetTransmissionMode (txMode);
}

void
LteUePhy::DoSetSrsConfigurationIndex (uint16_t srcCi)
{
  NS_LOG_FUNCTION (this << srcCi);
  m_srsPeriodicity = GetSrsPeriodicity (srcCi);
  m_srsSubframeOffset = GetSrsSubframeOffset (srcCi);
  m_srsConfigured = true;

  // a guard time is needed for the case where the SRS periodicity is changed dynamically at run time
  // if we use a static one, we can have a 0ms guard time
  m_srsStartTime = Simulator::Now () + MilliSeconds (0);
  NS_LOG_DEBUG (this << " UE SRS P " << m_srsPeriodicity << " RNTI " << m_rnti << " offset " << m_srsSubframeOffset << " cellId " << m_cellId << " CI " << srcCi);
}

void
LteUePhy::DoSetPa (double pa)
{
  NS_LOG_FUNCTION (this << pa);
  m_paLinear = pow (10,(pa / 10));
}

void
LteUePhy::DoSetRsrpFilterCoefficient (uint8_t rsrpFilterCoefficient)
{
  NS_LOG_FUNCTION (this << (uint16_t) (rsrpFilterCoefficient));
  m_powerControl->SetRsrpFilterCoefficient (rsrpFilterCoefficient);
}

void
LteUePhy::DoResetPhyAfterRlf ()
{
  NS_LOG_FUNCTION (this);
  m_downlinkSpectrumPhy->m_harqPhyModule->ClearDlHarqBuffer (m_rnti); //flush HARQ buffers
  m_dataInterferencePowerUpdated = false;
  m_rsInterferencePowerUpdated = false;
  m_pssReceived = false;
  DoReset ();
}

void
LteUePhy::DoResetRlfParams ()
{
  NS_LOG_FUNCTION (this);

  InitializeRlfParams ();
}

void
LteUePhy::DoStartInSnycDetection ()
{
  NS_LOG_FUNCTION (this);
  // indicates that the downlink radio link quality has to be monitored for in-sync indications
  m_downlinkInSync = false;
}

void
LteUePhy::DoSetImsi (uint64_t imsi)
{
  NS_LOG_FUNCTION (this);
  m_imsi = imsi;
}

void
LteUePhy::InitializeRlfParams ()
{
  NS_LOG_FUNCTION (this);
  m_numOfSubframes = 0;
  m_sinrDbFrame = 0;
  m_numOfFrames = 0;
  m_downlinkInSync = true;
}

void
LteUePhy::RlfDetection (double sinrDb)
{
  NS_LOG_FUNCTION (this << sinrDb);
  m_sinrDbFrame += sinrDb;
  m_numOfSubframes++;
  NS_LOG_LOGIC ("No of Subframes: " << m_numOfSubframes << " UE synchronized: " << m_downlinkInSync);
  //check for out_of_snyc indications first when UE is both DL and UL synchronized
  //m_downlinkInSync=true indicates that the evaluation is for out-of-sync indications
  if (m_downlinkInSync && m_numOfSubframes == 10)
    {
      /**
       * For every frame, if the downlink radio link quality(avg SINR)
       * is less than the threshold Qout, then the frame cannot be decoded
       */
      if ((m_sinrDbFrame / m_numOfSubframes) < m_qOut)
        {
          m_numOfFrames++; //increment the counter if a frame cannot be decoded
          NS_LOG_LOGIC ("No of Frames which cannot be decoded: " << m_numOfFrames);
        }
      else
        {
          /**
           * If the downlink radio link quality(avg SINR) is greater
           * than the threshold Qout, then the frame counter is reset
           * since only consecutive frames should be considered.
           */
          NS_LOG_INFO ("Resetting frame counter at phy. Current value = " << m_numOfFrames);
          m_numOfFrames = 0;
          // Also reset the sync indicator counter at RRC
          m_ueCphySapUser->ResetSyncIndicationCounter ();
        }
      m_numOfSubframes = 0;
      m_sinrDbFrame = 0;
    }
  /**
   * Once the number of consecutive frames which cannot be decoded equals
   * the Qout evaluation period (i.e 200ms), then an out-of-sync indication
   * is sent to the RRC layer
   */
  if (m_downlinkInSync && (m_numOfFrames * 10) == m_numOfQoutEvalSf)
    {
      NS_LOG_LOGIC ("At " << Simulator::Now ().As (Time::MS)
                          << " ms UE PHY sending out of snyc indication to UE RRC layer");
      m_ueCphySapUser->NotifyOutOfSync ();
      m_numOfFrames = 0;
    }
  //check for in_snyc indications when T310 timer is started
  //m_downlinkInSync=false indicates that the evaluation is for in-sync indications
  if (!m_downlinkInSync && m_numOfSubframes == 10)
    {
      /**
       * For every frame, if the downlink radio link quality(avg SINR)
       * is greater than the threshold Qin, then the frame can be
       * successfully decoded.
       */
      if ((m_sinrDbFrame / m_numOfSubframes) > m_qIn)
        {
          m_numOfFrames++; //increment the counter if a frame can be decoded
          NS_LOG_LOGIC ("No of Frames successfully decoded: " << m_numOfFrames);
        }
      else
        {
          /**
           * If the downlink radio link quality(avg SINR) is less
           * than the threshold Qin, then the frame counter is reset
           * since only consecutive frames should be considered
           */
          m_numOfFrames = 0;
          // Also reset the sync indicator counter at RRC
          m_ueCphySapUser->ResetSyncIndicationCounter ();
        }
      m_numOfSubframes = 0;
      m_sinrDbFrame = 0;
    }
  /**
   * Once the number of consecutive frames which can be decoded equals the Qin evaluation period (i.e 100ms),
   * then an in-sync indication is sent to the RRC layer
   */
  if (!m_downlinkInSync && (m_numOfFrames * 10) == m_numOfQinEvalSf)
    {
      NS_LOG_LOGIC ("At " << Simulator::Now ().As (Time::MS)
                          << " ms UE PHY sending in snyc indication to UE RRC layer");
      m_ueCphySapUser->NotifyInSync ();
      m_numOfFrames = 0;
    }
}


void
LteUePhy::SetTxMode1Gain (double gain)
{
  SetTxModeGain (1, gain);
}

void
LteUePhy::SetTxMode2Gain (double gain)
{
  SetTxModeGain (2, gain);
}

void
LteUePhy::SetTxMode3Gain (double gain)
{
  SetTxModeGain (3, gain);
}

void
LteUePhy::SetTxMode4Gain (double gain)
{
  SetTxModeGain (4, gain);
}

void
LteUePhy::SetTxMode5Gain (double gain)
{
  SetTxModeGain (5, gain);
}

void
LteUePhy::SetTxMode6Gain (double gain)
{
  SetTxModeGain (6, gain);
}

void
LteUePhy::SetTxMode7Gain (double gain)
{
  SetTxModeGain (7, gain);
}


void
LteUePhy::SetTxModeGain (uint8_t txMode, double gain)
{
  NS_LOG_FUNCTION (this << gain);
  // convert to linear
  double gainLin = std::pow (10.0, (gain / 10.0));
  if (m_txModeGain.size () < txMode)
    {
      m_txModeGain.resize (txMode);
    }
  std::vector <double> temp;
  temp = m_txModeGain;
  m_txModeGain.clear ();
  for (uint8_t i = 0; i < temp.size (); i++)
    {
      if (i == txMode - 1)
        {
          m_txModeGain.push_back (gainLin);
        }
      else
        {
          m_txModeGain.push_back (temp.at (i));
        }
    }
  // forward the info to DL LteSpectrumPhy
  m_downlinkSpectrumPhy->SetTxModeGain (txMode, gain);
}



void
LteUePhy::EnqueueDlHarqFeedback (DlInfoListElement_s m)
{
  NS_LOG_FUNCTION (this);
  // get the feedback from LteSpectrumPhy and send it through ideal PUCCH to eNB
  Ptr<DlHarqFeedbackLteControlMessage> msg = Create<DlHarqFeedbackLteControlMessage> ();
  msg->SetDlHarqFeedback (m);
  SetControlMessages (msg);
}

void
LteUePhy::SetHarqPhyModule (Ptr<LteHarqPhy> harq)
{
  m_harqPhyModule = harq;
}


LteUePhy::State
LteUePhy::GetState () const
{
  NS_LOG_FUNCTION (this);
  return m_state;
}


void
LteUePhy::SwitchToState (State newState)
{
  NS_LOG_FUNCTION (this << newState);
  State oldState = m_state;
  m_state = newState;
  NS_LOG_INFO (this << " cellId=" << m_cellId << " rnti=" << m_rnti
                    << " UePhy " << ToString (oldState)
                    << " --> " << ToString (newState));
  m_stateTransitionTrace (m_cellId, m_rnti, oldState, newState);
}


} // namespace ns3
