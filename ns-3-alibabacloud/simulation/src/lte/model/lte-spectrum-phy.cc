/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009, 2011 CTTC
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
 *         Marco Miozzo <marco.miozzo@cttc.es> (add physical error model)
 */


#include <ns3/object-factory.h>
#include <ns3/log.h>
#include <cmath>
#include <ns3/simulator.h>
#include <ns3/trace-source-accessor.h>
#include <ns3/antenna-model.h>
#include "lte-spectrum-phy.h"
#include "lte-spectrum-signal-parameters.h"
#include "lte-net-device.h"
#include "lte-radio-bearer-tag.h"
#include "lte-chunk-processor.h"
#include "lte-phy-tag.h"
#include <ns3/lte-mi-error-model.h>
#include <ns3/lte-radio-bearer-tag.h>
#include <ns3/boolean.h>
#include <ns3/double.h>
#include <ns3/config.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteSpectrumPhy");


/// duration of SRS portion of UL subframe
/// = 1 symbol for SRS -1ns as margin to avoid overlapping simulator events
static const Time UL_SRS_DURATION = NanoSeconds (71429 - 1);

/// duration of the control portion of a subframe
/// = 0.001 / 14 * 3 (ctrl fixed to 3 symbols) -1ns as margin to avoid overlapping simulator events
static const Time DL_CTRL_DURATION = NanoSeconds (214286 - 1);

/// Effective coding rate
static const double EffectiveCodingRate[29] = {
  0.08,
  0.1,
  0.11,
  0.15,
  0.19,
  0.24,
  0.3,
  0.37,
  0.44,
  0.51,
  0.3,
  0.33,
  0.37,
  0.42,
  0.48,
  0.54,
  0.6,
  0.43,
  0.45,
  0.5,
  0.55,
  0.6,
  0.65,
  0.7,
  0.75,
  0.8,
  0.85,
  0.89,
  0.92
};




TbId_t::TbId_t ()
{}

TbId_t::TbId_t (const uint16_t a, const uint8_t b)
  : m_rnti (a),
    m_layer (b)
{}

/**
 * Equality operator
 *
 * \param a lhs
 * \param b rhs
 * \returns true if rnti and layer are equal
 */
bool
operator == (const TbId_t &a, const TbId_t &b)
{
  return ( (a.m_rnti == b.m_rnti) && (a.m_layer == b.m_layer) );
}

/**
 * Less than operator
 *
 * \param a lhs
 * \param b rhs
 * \returns true if rnti less than ro rnti equal and layer less than
 */
bool
operator < (const TbId_t& a, const TbId_t& b)
{
  return ( (a.m_rnti < b.m_rnti) || ( (a.m_rnti == b.m_rnti) && (a.m_layer < b.m_layer) ) );
}

NS_OBJECT_ENSURE_REGISTERED (LteSpectrumPhy);

LteSpectrumPhy::LteSpectrumPhy ()
  : m_state (IDLE),
    m_cellId (0),
    m_componentCarrierId (0),
    m_transmissionMode (0),
    m_layersNum (1)
{
  NS_LOG_FUNCTION (this);
  m_random = CreateObject<UniformRandomVariable> ();
  m_random->SetAttribute ("Min", DoubleValue (0.0));
  m_random->SetAttribute ("Max", DoubleValue (1.0));
  m_interferenceData = CreateObject<LteInterference> ();
  m_interferenceCtrl = CreateObject<LteInterference> ();

  for (uint8_t i = 0; i < 7; i++)
    {
      m_txModeGain.push_back (1.0);
    }
}


LteSpectrumPhy::~LteSpectrumPhy ()
{
  NS_LOG_FUNCTION (this);
  m_expectedTbs.clear ();
  m_txModeGain.clear ();
}

void LteSpectrumPhy::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_channel = 0;
  m_mobility = 0;
  m_device = 0;
  m_interferenceData->Dispose ();
  m_interferenceData = 0;
  m_interferenceCtrl->Dispose ();
  m_interferenceCtrl = 0;
  m_ltePhyRxDataEndErrorCallback = MakeNullCallback< void > ();
  m_ltePhyRxDataEndOkCallback    = MakeNullCallback< void, Ptr<Packet> >  ();
  m_ltePhyRxCtrlEndOkCallback = MakeNullCallback< void, std::list<Ptr<LteControlMessage> > > ();
  m_ltePhyRxCtrlEndErrorCallback = MakeNullCallback< void > ();
  m_ltePhyDlHarqFeedbackCallback = MakeNullCallback< void, DlInfoListElement_s > ();
  m_ltePhyUlHarqFeedbackCallback = MakeNullCallback< void, UlInfoListElement_s > ();
  m_ltePhyRxPssCallback = MakeNullCallback< void, uint16_t, Ptr<SpectrumValue> > ();
  SpectrumPhy::DoDispose ();
}

/**
 * Output stream output operator
 *
 * \param os output stream
 * \param s state
 * \returns output stream
 */
std::ostream& operator<< (std::ostream& os, LteSpectrumPhy::State s)
{
  switch (s)
    {
      case LteSpectrumPhy::IDLE:
        os << "IDLE";
        break;
      case LteSpectrumPhy::RX_DATA:
        os << "RX_DATA";
        break;
      case LteSpectrumPhy::RX_DL_CTRL:
        os << "RX_DL_CTRL";
        break;
      case LteSpectrumPhy::TX_DATA:
        os << "TX_DATA";
        break;
      case LteSpectrumPhy::TX_DL_CTRL:
        os << "TX_DL_CTRL";
        break;
      case LteSpectrumPhy::TX_UL_SRS:
        os << "TX_UL_SRS";
        break;
      default:
        os << "UNKNOWN";
        break;
    }
  return os;
}

TypeId
LteSpectrumPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteSpectrumPhy")
    .SetParent<SpectrumPhy> ()
    .SetGroupName ("Lte")
    .AddTraceSource ("TxStart",
                     "Trace fired when a new transmission is started",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_phyTxStartTrace),
                     "ns3::PacketBurst::TracedCallback")
    .AddTraceSource ("TxEnd",
                     "Trace fired when a previously started transmission is finished",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_phyTxEndTrace),
                     "ns3::PacketBurst::TracedCallback")
    .AddTraceSource ("RxStart",
                     "Trace fired when the start of a signal is detected",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_phyRxStartTrace),
                     "ns3::PacketBurst::TracedCallback")
    .AddTraceSource ("RxEndOk",
                     "Trace fired when a previously started RX terminates successfully",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_phyRxEndOkTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("RxEndError",
                     "Trace fired when a previously started RX terminates with an error",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_phyRxEndErrorTrace),
                     "ns3::Packet::TracedCallback")
    .AddAttribute ("DataErrorModelEnabled",
                   "Activate/Deactivate the error model of data (TBs of PDSCH and PUSCH) [by default is active].",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteSpectrumPhy::m_dataErrorModelEnabled),
                   MakeBooleanChecker ())
    .AddAttribute ("CtrlErrorModelEnabled",
                   "Activate/Deactivate the error model of control (PCFICH-PDCCH decodification) [by default is active].",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteSpectrumPhy::m_ctrlErrorModelEnabled),
                   MakeBooleanChecker ())
    .AddTraceSource ("DlPhyReception",
                     "DL reception PHY layer statistics.",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_dlPhyReception),
                     "ns3::PhyReceptionStatParameters::TracedCallback")
    .AddTraceSource ("UlPhyReception",
                     "DL reception PHY layer statistics.",
                     MakeTraceSourceAccessor (&LteSpectrumPhy::m_ulPhyReception),
                     "ns3::PhyReceptionStatParameters::TracedCallback")
  ;
  return tid;
}



Ptr<NetDevice>
LteSpectrumPhy::GetDevice () const
{
  NS_LOG_FUNCTION (this);
  return m_device;
}


Ptr<MobilityModel>
LteSpectrumPhy::GetMobility () const
{
  NS_LOG_FUNCTION (this);
  return m_mobility;
}


void
LteSpectrumPhy::SetDevice (Ptr<NetDevice> d)
{
  NS_LOG_FUNCTION (this << d);
  m_device = d;
}


void
LteSpectrumPhy::SetMobility (Ptr<MobilityModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_mobility = m;
}


void
LteSpectrumPhy::SetChannel (Ptr<SpectrumChannel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}

Ptr<SpectrumChannel>
LteSpectrumPhy::GetChannel ()
{
  return m_channel;
}

Ptr<const SpectrumModel>
LteSpectrumPhy::GetRxSpectrumModel () const
{
  return m_rxSpectrumModel;
}


void
LteSpectrumPhy::SetTxPowerSpectralDensity (Ptr<SpectrumValue> txPsd)
{
  NS_LOG_FUNCTION (this << txPsd);
  NS_ASSERT (txPsd);
  m_txPsd = txPsd;
}


void
LteSpectrumPhy::SetNoisePowerSpectralDensity (Ptr<const SpectrumValue> noisePsd)
{
  NS_LOG_FUNCTION (this << noisePsd);
  NS_ASSERT (noisePsd);
  m_rxSpectrumModel = noisePsd->GetSpectrumModel ();
  m_interferenceData->SetNoisePowerSpectralDensity (noisePsd);
  m_interferenceCtrl->SetNoisePowerSpectralDensity (noisePsd);
}


void
LteSpectrumPhy::Reset ()
{
  NS_LOG_FUNCTION (this);
  m_cellId = 0;
  m_state = IDLE;
  m_transmissionMode = 0;
  m_layersNum = 1;
  m_endTxEvent.Cancel ();
  m_endRxDataEvent.Cancel ();
  m_endRxDlCtrlEvent.Cancel ();
  m_endRxUlSrsEvent.Cancel ();
  m_rxControlMessageList.clear ();
  m_expectedTbs.clear ();
  m_txControlMessageList.clear ();
  m_rxPacketBurstList.clear ();
  m_txPacketBurst = 0;
  m_rxSpectrumModel = 0;

  // Detach from the channel, because receiving any signal without
  // spectrum model is an error.
  if (m_channel)
    {
      m_channel->RemoveRx (this);
    }
}


void
LteSpectrumPhy::SetLtePhyRxDataEndErrorCallback (LtePhyRxDataEndErrorCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyRxDataEndErrorCallback = c;
}


void
LteSpectrumPhy::SetLtePhyRxDataEndOkCallback (LtePhyRxDataEndOkCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyRxDataEndOkCallback = c;
}

void
LteSpectrumPhy::SetLtePhyRxCtrlEndOkCallback (LtePhyRxCtrlEndOkCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyRxCtrlEndOkCallback = c;
}

void
LteSpectrumPhy::SetLtePhyRxCtrlEndErrorCallback (LtePhyRxCtrlEndErrorCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyRxCtrlEndErrorCallback = c;
}


void
LteSpectrumPhy::SetLtePhyRxPssCallback (LtePhyRxPssCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyRxPssCallback = c;
}

void
LteSpectrumPhy::SetLtePhyDlHarqFeedbackCallback (LtePhyDlHarqFeedbackCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyDlHarqFeedbackCallback = c;
}

void
LteSpectrumPhy::SetLtePhyUlHarqFeedbackCallback (LtePhyUlHarqFeedbackCallback c)
{
  NS_LOG_FUNCTION (this);
  m_ltePhyUlHarqFeedbackCallback = c;
}


Ptr<Object>
LteSpectrumPhy::GetAntenna () const
{
  return m_antenna;
}

void
LteSpectrumPhy::SetAntenna (Ptr<AntennaModel> a)
{
  NS_LOG_FUNCTION (this << a);
  m_antenna = a;
}

void
LteSpectrumPhy::SetState (State newState)
{
  ChangeState (newState);
}


void
LteSpectrumPhy::ChangeState (State newState)
{
  NS_LOG_LOGIC (this << " state: " << m_state << " -> " << newState);
  m_state = newState;
}


void
LteSpectrumPhy::SetHarqPhyModule (Ptr<LteHarqPhy> harq)
{
  m_harqPhyModule = harq;
}




bool
LteSpectrumPhy::StartTxDataFrame (Ptr<PacketBurst> pb, std::list<Ptr<LteControlMessage> > ctrlMsgList, Time duration)
{
  NS_LOG_FUNCTION (this << pb);
  NS_LOG_LOGIC (this << " state: " << m_state);

  m_phyTxStartTrace (pb);

  switch (m_state)
    {
      case RX_DATA:
      case RX_DL_CTRL:
      case RX_UL_SRS:
        NS_FATAL_ERROR ("cannot TX while RX: according to FDD channel access, the physical layer for transmission cannot be used for reception");
        break;

      case TX_DATA:
      case TX_DL_CTRL:
      case TX_UL_SRS:
        NS_FATAL_ERROR ("cannot TX while already TX: the MAC should avoid this");
        break;

      case IDLE:
        {
          /*
          m_txPsd must be set by the device, according to
          (i) the available subchannel for transmission
          (ii) the power transmission
          */
          NS_ASSERT (m_txPsd);
          m_txPacketBurst = pb;

          // we need to convey some PHY meta information to the receiver
          // to be used for simulation purposes (e.g., the CellId). This
          // is done by setting the ctrlMsgList parameter of
          // LteSpectrumSignalParametersDataFrame
          ChangeState (TX_DATA);
          NS_ASSERT (m_channel);
          Ptr<LteSpectrumSignalParametersDataFrame> txParams = Create<LteSpectrumSignalParametersDataFrame> ();
          txParams->duration = duration;
          txParams->txPhy = GetObject<SpectrumPhy> ();
          txParams->txAntenna = m_antenna;
          txParams->psd = m_txPsd;
          txParams->packetBurst = pb;
          txParams->ctrlMsgList = ctrlMsgList;
          txParams->cellId = m_cellId;
          m_channel->StartTx (txParams);
          m_endTxEvent = Simulator::Schedule (duration, &LteSpectrumPhy::EndTxData, this);
        }
        return false;
        break;

      default:
        NS_FATAL_ERROR ("unknown state");
        return true;
        break;
    }
}

bool
LteSpectrumPhy::StartTxDlCtrlFrame (std::list<Ptr<LteControlMessage> > ctrlMsgList, bool pss)
{
  NS_LOG_FUNCTION (this << " PSS " << (uint16_t)pss);
  NS_LOG_LOGIC (this << " state: " << m_state);

  switch (m_state)
    {
      case RX_DATA:
      case RX_DL_CTRL:
      case RX_UL_SRS:
        NS_FATAL_ERROR ("cannot TX while RX: according to FDD channel access, the physical layer for transmission cannot be used for reception");
        break;

      case TX_DATA:
      case TX_DL_CTRL:
      case TX_UL_SRS:
        NS_FATAL_ERROR ("cannot TX while already TX: the MAC should avoid this");
        break;

      case IDLE:
        {
          /*
          m_txPsd must be set by the device, according to
          (i) the available subchannel for transmission
          (ii) the power transmission
          */
          NS_ASSERT (m_txPsd);

          // we need to convey some PHY meta information to the receiver
          // to be used for simulation purposes (e.g., the CellId). This
          // is done by setting the cellId parameter of
          // LteSpectrumSignalParametersDlCtrlFrame
          ChangeState (TX_DL_CTRL);
          NS_ASSERT (m_channel);

          Ptr<LteSpectrumSignalParametersDlCtrlFrame> txParams = Create<LteSpectrumSignalParametersDlCtrlFrame> ();
          txParams->duration = DL_CTRL_DURATION;
          txParams->txPhy = GetObject<SpectrumPhy> ();
          txParams->txAntenna = m_antenna;
          txParams->psd = m_txPsd;
          txParams->cellId = m_cellId;
          txParams->pss = pss;
          txParams->ctrlMsgList = ctrlMsgList;
          m_channel->StartTx (txParams);
          m_endTxEvent = Simulator::Schedule (DL_CTRL_DURATION, &LteSpectrumPhy::EndTxDlCtrl, this);
        }
        return false;
        break;

      default:
        NS_FATAL_ERROR ("unknown state");
        return true;
        break;
    }
}


bool
LteSpectrumPhy::StartTxUlSrsFrame ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " state: " << m_state);

  switch (m_state)
    {
      case RX_DATA:
      case RX_DL_CTRL:
      case RX_UL_SRS:
        NS_FATAL_ERROR ("cannot TX while RX: according to FDD channel access, the physical layer for transmission cannot be used for reception");
        break;

      case TX_DL_CTRL:
      case TX_DATA:
      case TX_UL_SRS:
        NS_FATAL_ERROR ("cannot TX while already TX: the MAC should avoid this");
        break;

      case IDLE:
        {
          /*
          m_txPsd must be set by the device, according to
          (i) the available subchannel for transmission
          (ii) the power transmission
          */
          NS_ASSERT (m_txPsd);
          NS_LOG_LOGIC (this << " m_txPsd: " << *m_txPsd);

          // we need to convey some PHY meta information to the receiver
          // to be used for simulation purposes (e.g., the CellId). This
          // is done by setting the cellId parameter of
          // LteSpectrumSignalParametersDlCtrlFrame
          ChangeState (TX_UL_SRS);
          NS_ASSERT (m_channel);
          Ptr<LteSpectrumSignalParametersUlSrsFrame> txParams = Create<LteSpectrumSignalParametersUlSrsFrame> ();
          txParams->duration = UL_SRS_DURATION;
          txParams->txPhy = GetObject<SpectrumPhy> ();
          txParams->txAntenna = m_antenna;
          txParams->psd = m_txPsd;
          txParams->cellId = m_cellId;
          m_channel->StartTx (txParams);
          m_endTxEvent = Simulator::Schedule (UL_SRS_DURATION, &LteSpectrumPhy::EndTxUlSrs, this);
        }
        return false;
        break;

      default:
        NS_FATAL_ERROR ("unknown state");
        return true;
        break;
    }
}



void
LteSpectrumPhy::EndTxData ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " state: " << m_state);

  NS_ASSERT (m_state == TX_DATA);
  m_phyTxEndTrace (m_txPacketBurst);
  m_txPacketBurst = 0;
  ChangeState (IDLE);
}

void
LteSpectrumPhy::EndTxDlCtrl ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " state: " << m_state);

  NS_ASSERT (m_state == TX_DL_CTRL);
  NS_ASSERT (m_txPacketBurst == 0);
  ChangeState (IDLE);
}

void
LteSpectrumPhy::EndTxUlSrs ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " state: " << m_state);

  NS_ASSERT (m_state == TX_UL_SRS);
  NS_ASSERT (m_txPacketBurst == 0);
  ChangeState (IDLE);
}




void
LteSpectrumPhy::StartRx (Ptr<SpectrumSignalParameters> spectrumRxParams)
{
  NS_LOG_FUNCTION (this << spectrumRxParams);
  NS_LOG_LOGIC (this << " state: " << m_state);

  Ptr <const SpectrumValue> rxPsd = spectrumRxParams->psd;
  Time duration = spectrumRxParams->duration;

  // the device might start RX only if the signal is of a type
  // understood by this device - in this case, an LTE signal.
  Ptr<LteSpectrumSignalParametersDataFrame> lteDataRxParams = DynamicCast<LteSpectrumSignalParametersDataFrame> (spectrumRxParams);
  Ptr<LteSpectrumSignalParametersDlCtrlFrame> lteDlCtrlRxParams = DynamicCast<LteSpectrumSignalParametersDlCtrlFrame> (spectrumRxParams);
  Ptr<LteSpectrumSignalParametersUlSrsFrame> lteUlSrsRxParams = DynamicCast<LteSpectrumSignalParametersUlSrsFrame> (spectrumRxParams);
  if (lteDataRxParams != 0)
    {
      m_interferenceData->AddSignal (rxPsd, duration);
      StartRxData (lteDataRxParams);
    }
  else if (lteDlCtrlRxParams != 0)
    {
      m_interferenceCtrl->AddSignal (rxPsd, duration);
      StartRxDlCtrl (lteDlCtrlRxParams);
    }
  else if (lteUlSrsRxParams != 0)
    {
      m_interferenceCtrl->AddSignal (rxPsd, duration);
      StartRxUlSrs (lteUlSrsRxParams);
    }
  else
    {
      // other type of signal (could be 3G, GSM, whatever) -> interference
      m_interferenceData->AddSignal (rxPsd, duration);
      m_interferenceCtrl->AddSignal (rxPsd, duration);
    }
}

void
LteSpectrumPhy::StartRxData (Ptr<LteSpectrumSignalParametersDataFrame> params)
{
  NS_LOG_FUNCTION (this);
  switch (m_state)
    {
      case TX_DATA:
      case TX_DL_CTRL:
      case TX_UL_SRS:
        NS_FATAL_ERROR ("cannot RX while TX: according to FDD channel access, the physical layer for transmission cannot be used for reception");
        break;
      case RX_DL_CTRL:
        NS_FATAL_ERROR ("cannot RX Data while receiving control");
        break;
      case IDLE:
      case RX_DATA:
        // the behavior is similar when
        // we're IDLE or RX because we can receive more signals
        // simultaneously (e.g., at the eNB).
        {
          // To check if we're synchronized to this signal, we check
          // for the CellId which is reported in the
          //  LteSpectrumSignalParametersDataFrame
          if (params->cellId  == m_cellId)
            {
              NS_LOG_LOGIC (this << " synchronized with this signal (cellId=" << params->cellId << ")");
              if ((m_rxPacketBurstList.empty ())&&(m_rxControlMessageList.empty ()))
                {
                  NS_ASSERT (m_state == IDLE);
                  // first transmission, i.e., we're IDLE and we
                  // start RX
                  m_firstRxStart = Simulator::Now ();
                  m_firstRxDuration = params->duration;
                  NS_LOG_LOGIC (this << " scheduling EndRx with delay " << params->duration.As (Time::S));
                  m_endRxDataEvent = Simulator::Schedule (params->duration, &LteSpectrumPhy::EndRxData, this);
                }
              else
                {
                  NS_ASSERT (m_state == RX_DATA);
                  // sanity check: if there are multiple RX events, they
                  // should occur at the same time and have the same
                  // duration, otherwise the interference calculation
                  // won't be correct
                  NS_ASSERT ((m_firstRxStart == Simulator::Now ())
                             && (m_firstRxDuration == params->duration));
                }

              ChangeState (RX_DATA);
              if (params->packetBurst)
                {
                  m_rxPacketBurstList.push_back (params->packetBurst);
                  m_interferenceData->StartRx (params->psd);

                  m_phyRxStartTrace (params->packetBurst);
                }
              NS_LOG_DEBUG (this << " insert msgs " << params->ctrlMsgList.size ());
              m_rxControlMessageList.insert (m_rxControlMessageList.end (), params->ctrlMsgList.begin (), params->ctrlMsgList.end ());

              NS_LOG_LOGIC (this << " numSimultaneousRxEvents = " << m_rxPacketBurstList.size ());
            }
          else
            {
              NS_LOG_LOGIC (this << " not in sync with this signal (cellId="
                                 << params->cellId  << ", m_cellId=" << m_cellId << ")");
            }
        }
        break;

      default:
        NS_FATAL_ERROR ("unknown state");
        break;
    }

  NS_LOG_LOGIC (this << " state: " << m_state);
}



void
LteSpectrumPhy::StartRxDlCtrl (Ptr<LteSpectrumSignalParametersDlCtrlFrame> lteDlCtrlRxParams)
{
  NS_LOG_FUNCTION (this);

  // To check if we're synchronized to this signal, we check
  // for the CellId which is reported in the
  // LteSpectrumSignalParametersDlCtrlFrame
  uint16_t cellId;
  NS_ASSERT (lteDlCtrlRxParams != 0);
  cellId = lteDlCtrlRxParams->cellId;

  switch (m_state)
    {
      case TX_DATA:
      case TX_DL_CTRL:
      case TX_UL_SRS:
      case RX_DATA:
      case RX_UL_SRS:
        NS_FATAL_ERROR ("unexpected event in state " << m_state);
        break;

      case RX_DL_CTRL:
      case IDLE:

        // common code for the two states
        // check presence of PSS for UE measuerements
        if (lteDlCtrlRxParams->pss == true)
          {
            if (!m_ltePhyRxPssCallback.IsNull ())
              {
                m_ltePhyRxPssCallback (cellId, lteDlCtrlRxParams->psd);
              }
          }

        // differentiated code for the two states
        switch (m_state)
          {
            case RX_DL_CTRL:
              NS_ASSERT_MSG (m_cellId != cellId, "any other DlCtrl should be from a different cell");
              NS_LOG_LOGIC (this << " ignoring other DlCtrl (cellId="
                                 << cellId  << ", m_cellId=" << m_cellId << ")");
              break;

            case IDLE:
              if (cellId  == m_cellId)
                {
                  NS_LOG_LOGIC (this << " synchronized with this signal (cellId=" << cellId << ")");

                  NS_ASSERT (m_rxControlMessageList.empty ());
                  m_firstRxStart = Simulator::Now ();
                  m_firstRxDuration = lteDlCtrlRxParams->duration;
                  NS_LOG_LOGIC (this << " scheduling EndRx with delay " << lteDlCtrlRxParams->duration);

                  // store the DCIs
                  m_rxControlMessageList = lteDlCtrlRxParams->ctrlMsgList;
                  m_endRxDlCtrlEvent = Simulator::Schedule (lteDlCtrlRxParams->duration, &LteSpectrumPhy::EndRxDlCtrl, this);
                  ChangeState (RX_DL_CTRL);
                  m_interferenceCtrl->StartRx (lteDlCtrlRxParams->psd);
                }
              else
                {
                  NS_LOG_LOGIC (this << " not synchronizing with this signal (cellId="
                                     << cellId  << ", m_cellId=" << m_cellId << ")");
                }
              break;

            default:
              NS_FATAL_ERROR ("unexpected event in state " << m_state);
              break;
          }
        break; // case RX_DL_CTRL or IDLE

      default:
        NS_FATAL_ERROR ("unknown state");
        break;
    }

  NS_LOG_LOGIC (this << " state: " << m_state);
}




void
LteSpectrumPhy::StartRxUlSrs (Ptr<LteSpectrumSignalParametersUlSrsFrame> lteUlSrsRxParams)
{
  NS_LOG_FUNCTION (this);
  switch (m_state)
    {
      case TX_DATA:
      case TX_DL_CTRL:
      case TX_UL_SRS:
        NS_FATAL_ERROR ("cannot RX while TX: according to FDD channel access, the physical layer for transmission cannot be used for reception");
        break;

      case RX_DATA:
      case RX_DL_CTRL:
        NS_FATAL_ERROR ("cannot RX SRS while receiving something else");
        break;

      case IDLE:
      case RX_UL_SRS:
        // the behavior is similar when
        // we're IDLE or RX_UL_SRS because we can receive more signals
        // simultaneously at the eNB
        {
          // To check if we're synchronized to this signal, we check
          // for the CellId which is reported in the
          // LteSpectrumSignalParametersDlCtrlFrame
          uint16_t cellId;
          cellId = lteUlSrsRxParams->cellId;
          if (cellId  == m_cellId)
            {
              NS_LOG_LOGIC (this << " synchronized with this signal (cellId=" << cellId << ")");
              if (m_state == IDLE)
                {
                  // first transmission, i.e., we're IDLE and we
                  // start RX
                  NS_ASSERT (m_rxControlMessageList.empty ());
                  m_firstRxStart = Simulator::Now ();
                  m_firstRxDuration = lteUlSrsRxParams->duration;
                  NS_LOG_LOGIC (this << " scheduling EndRx with delay " << lteUlSrsRxParams->duration);

                  m_endRxUlSrsEvent = Simulator::Schedule (lteUlSrsRxParams->duration, &LteSpectrumPhy::EndRxUlSrs, this);
                }
              else if (m_state == RX_UL_SRS)
                {
                  // sanity check: if there are multiple RX events, they
                  // should occur at the same time and have the same
                  // duration, otherwise the interference calculation
                  // won't be correct
                  NS_ASSERT ((m_firstRxStart == Simulator::Now ())
                             && (m_firstRxDuration == lteUlSrsRxParams->duration));
                }
              ChangeState (RX_UL_SRS);
              m_interferenceCtrl->StartRx (lteUlSrsRxParams->psd);
            }
          else
            {
              NS_LOG_LOGIC (this << " not in sync with this signal (cellId="
                                 << cellId  << ", m_cellId=" << m_cellId << ")");
            }
        }
        break;

      default:
        NS_FATAL_ERROR ("unknown state");
        break;
    }

  NS_LOG_LOGIC (this << " state: " << m_state);
}


void
LteSpectrumPhy::UpdateSinrPerceived (const SpectrumValue& sinr)
{
  NS_LOG_FUNCTION (this << sinr);
  m_sinrPerceived = sinr;
}


void
LteSpectrumPhy::AddExpectedTb (uint16_t  rnti, uint8_t ndi, uint16_t size, uint8_t mcs, std::vector<int> map, uint8_t layer, uint8_t harqId,uint8_t rv,  bool downlink)
{
  NS_LOG_FUNCTION (this << " rnti: " << rnti << " NDI " << (uint16_t)ndi << " size " << size << " mcs " << (uint16_t)mcs << " layer " << (uint16_t)layer << " rv " << (uint16_t)rv);
  TbId_t tbId;
  tbId.m_rnti = rnti;
  tbId.m_layer = layer;
  expectedTbs_t::iterator it;
  it = m_expectedTbs.find (tbId);
  if (it != m_expectedTbs.end ())
    {
      // migth be a TB of an unreceived packet (due to high progpalosses)
      m_expectedTbs.erase (it);
    }
  // insert new entry
  tbInfo_t tbInfo = {ndi, size, mcs, map, harqId, rv, 0.0, downlink, false, false};
  m_expectedTbs.insert (std::pair<TbId_t, tbInfo_t> (tbId,tbInfo));
}

void
LteSpectrumPhy::RemoveExpectedTb (uint16_t  rnti)
{
  NS_LOG_FUNCTION (this << rnti);
  TbId_t tbId;
  tbId.m_rnti = rnti;
  //Remove TB of both the layers
  for (uint8_t i = 0; i < 2; i++)
    {
      tbId.m_layer = i;
      expectedTbs_t::iterator it;
      it = m_expectedTbs.find (tbId);
      if (it != m_expectedTbs.end ())
        {
          m_expectedTbs.erase (it);
        }
    }
}




void
LteSpectrumPhy::EndRxData ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " state: " << m_state);

  NS_ASSERT (m_state == RX_DATA);

  // this will trigger CQI calculation and Error Model evaluation
  // as a side effect, the error model should update the error status of all TBs
  m_interferenceData->EndRx ();
  NS_LOG_DEBUG (this << " No. of burts " << m_rxPacketBurstList.size ());
  NS_LOG_DEBUG (this << " Expected TBs " << m_expectedTbs.size ());
  expectedTbs_t::iterator itTb = m_expectedTbs.begin ();

  // apply transmission mode gain
  NS_LOG_DEBUG (this << " txMode " << (uint16_t)m_transmissionMode << " gain " << m_txModeGain.at (m_transmissionMode));
  NS_ASSERT (m_transmissionMode < m_txModeGain.size ());
  m_sinrPerceived *= m_txModeGain.at (m_transmissionMode);

  while (itTb != m_expectedTbs.end ())
    {
      if ((m_dataErrorModelEnabled)&&(m_rxPacketBurstList.size () > 0)) // avoid to check for errors when there is no actual data transmitted
        {
          // retrieve HARQ info
          HarqProcessInfoList_t harqInfoList;
          if ((*itTb).second.ndi == 0)
            {
              // TB retxed: retrieve HARQ history
              uint16_t ulHarqId = 0;
              if ((*itTb).second.downlink)
                {
                  harqInfoList = m_harqPhyModule->GetHarqProcessInfoDl ((*itTb).second.harqProcessId, (*itTb).first.m_layer);
                }
              else
                {
                  harqInfoList = m_harqPhyModule->GetHarqProcessInfoUl ((*itTb).first.m_rnti, ulHarqId);
                }
            }
          TbStats_t tbStats = LteMiErrorModel::GetTbDecodificationStats (m_sinrPerceived, (*itTb).second.rbBitmap, (*itTb).second.size, (*itTb).second.mcs, harqInfoList);
          (*itTb).second.mi = tbStats.mi;
          (*itTb).second.corrupt = !(m_random->GetValue () > tbStats.tbler);
          NS_LOG_DEBUG (this << "RNTI " << (*itTb).first.m_rnti << " size " << (*itTb).second.size << " mcs " << (uint32_t)(*itTb).second.mcs << " bitmap " << (*itTb).second.rbBitmap.size () << " layer " << (uint16_t)(*itTb).first.m_layer << " TBLER " << tbStats.tbler << " corrupted " << (*itTb).second.corrupt);
          // fire traces on DL/UL reception PHY stats
          PhyReceptionStatParameters params;
          params.m_timestamp = Simulator::Now ().GetMilliSeconds ();
          params.m_cellId = m_cellId;
          params.m_imsi = 0; // it will be set by DlPhyTransmissionCallback in LteHelper
          params.m_rnti = (*itTb).first.m_rnti;
          params.m_txMode = m_transmissionMode;
          params.m_layer =  (*itTb).first.m_layer;
          params.m_mcs = (*itTb).second.mcs;
          params.m_size = (*itTb).second.size;
          params.m_rv = (*itTb).second.rv;
          params.m_ndi = (*itTb).second.ndi;
          params.m_correctness = (uint8_t) !(*itTb).second.corrupt;
          params.m_ccId = m_componentCarrierId;
          if ((*itTb).second.downlink)
            {
              // DL
              m_dlPhyReception (params);
            }
          else
            {
              // UL
              params.m_rv = harqInfoList.size ();
              m_ulPhyReception (params);
            }
        }

      itTb++;
    }
  std::map <uint16_t, DlInfoListElement_s> harqDlInfoMap;
  for (std::list<Ptr<PacketBurst> >::const_iterator i = m_rxPacketBurstList.begin ();
       i != m_rxPacketBurstList.end (); ++i)
    {
      for (std::list<Ptr<Packet> >::const_iterator j = (*i)->Begin (); j != (*i)->End (); ++j)
        {
          // retrieve TB info of this packet
          LteRadioBearerTag tag;
          (*j)->PeekPacketTag (tag);
          TbId_t tbId;
          tbId.m_rnti = tag.GetRnti ();
          tbId.m_layer = tag.GetLayer ();
          itTb = m_expectedTbs.find (tbId);
          NS_LOG_INFO (this << " Packet of " << tbId.m_rnti << " layer " <<  (uint16_t) tag.GetLayer ());
          if (itTb != m_expectedTbs.end ())
            {
              if (!(*itTb).second.corrupt)
                {
                  m_phyRxEndOkTrace (*j);

                  if (!m_ltePhyRxDataEndOkCallback.IsNull ())
                    {
                      m_ltePhyRxDataEndOkCallback (*j);
                    }
                }
              else
                {
                  // TB received with errors
                  m_phyRxEndErrorTrace (*j);
                }

              // send HARQ feedback (if not already done for this TB)
              if (!(*itTb).second.harqFeedbackSent)
                {
                  (*itTb).second.harqFeedbackSent = true;
                  if (!(*itTb).second.downlink)
                    {
                      UlInfoListElement_s harqUlInfo;
                      harqUlInfo.m_rnti = tbId.m_rnti;
                      harqUlInfo.m_tpc = 0;
                      if ((*itTb).second.corrupt)
                        {
                          harqUlInfo.m_receptionStatus = UlInfoListElement_s::NotOk;
                          NS_LOG_DEBUG (this << " RNTI " << tbId.m_rnti << " send UL-HARQ-NACK");
                          m_harqPhyModule->UpdateUlHarqProcessStatus (tbId.m_rnti, (*itTb).second.mi, (*itTb).second.size, (*itTb).second.size / EffectiveCodingRate [(*itTb).second.mcs]);
                        }
                      else
                        {
                          harqUlInfo.m_receptionStatus = UlInfoListElement_s::Ok;
                          NS_LOG_DEBUG (this << " RNTI " << tbId.m_rnti << " send UL-HARQ-ACK");
                          m_harqPhyModule->ResetUlHarqProcessStatus (tbId.m_rnti, (*itTb).second.harqProcessId);
                        }
                      if (!m_ltePhyUlHarqFeedbackCallback.IsNull ())
                        {
                          m_ltePhyUlHarqFeedbackCallback (harqUlInfo);
                        }
                    }
                  else
                    {
                      std::map <uint16_t, DlInfoListElement_s>::iterator itHarq = harqDlInfoMap.find (tbId.m_rnti);
                      if (itHarq == harqDlInfoMap.end ())
                        {
                          DlInfoListElement_s harqDlInfo;
                          harqDlInfo.m_harqStatus.resize (m_layersNum, DlInfoListElement_s::ACK);
                          harqDlInfo.m_rnti = tbId.m_rnti;
                          harqDlInfo.m_harqProcessId = (*itTb).second.harqProcessId;
                          if ((*itTb).second.corrupt)
                            {
                              harqDlInfo.m_harqStatus.at (tbId.m_layer) = DlInfoListElement_s::NACK;
                              NS_LOG_DEBUG (this << " RNTI " << tbId.m_rnti << " harqId " << (uint16_t)(*itTb).second.harqProcessId << " layer " << (uint16_t)tbId.m_layer << " send DL-HARQ-NACK");
                              m_harqPhyModule->UpdateDlHarqProcessStatus ((*itTb).second.harqProcessId, tbId.m_layer, (*itTb).second.mi, (*itTb).second.size, (*itTb).second.size / EffectiveCodingRate [(*itTb).second.mcs]);
                            }
                          else
                            {

                              harqDlInfo.m_harqStatus.at (tbId.m_layer) = DlInfoListElement_s::ACK;
                              NS_LOG_DEBUG (this << " RNTI " << tbId.m_rnti << " harqId " << (uint16_t)(*itTb).second.harqProcessId << " layer " << (uint16_t)tbId.m_layer << " size " << (*itTb).second.size << " send DL-HARQ-ACK");
                              m_harqPhyModule->ResetDlHarqProcessStatus ((*itTb).second.harqProcessId);
                            }
                          harqDlInfoMap.insert (std::pair <uint16_t, DlInfoListElement_s> (tbId.m_rnti, harqDlInfo));
                        }
                      else
                        {
                          if ((*itTb).second.corrupt)
                            {
                              (*itHarq).second.m_harqStatus.at (tbId.m_layer) = DlInfoListElement_s::NACK;
                              NS_LOG_DEBUG (this << " RNTI " << tbId.m_rnti << " harqId " << (uint16_t)(*itTb).second.harqProcessId << " layer " << (uint16_t)tbId.m_layer << " size " << (*itHarq).second.m_harqStatus.size () << " send DL-HARQ-NACK");
                              m_harqPhyModule->UpdateDlHarqProcessStatus ((*itTb).second.harqProcessId, tbId.m_layer, (*itTb).second.mi, (*itTb).second.size, (*itTb).second.size / EffectiveCodingRate [(*itTb).second.mcs]);
                            }
                          else
                            {
                              NS_ASSERT_MSG (tbId.m_layer < (*itHarq).second.m_harqStatus.size (), " layer " << (uint16_t)tbId.m_layer);
                              (*itHarq).second.m_harqStatus.at (tbId.m_layer) = DlInfoListElement_s::ACK;
                              NS_LOG_DEBUG (this << " RNTI " << tbId.m_rnti << " harqId " << (uint16_t)(*itTb).second.harqProcessId << " layer " << (uint16_t)tbId.m_layer << " size " << (*itHarq).second.m_harqStatus.size () << " send DL-HARQ-ACK");
                              m_harqPhyModule->ResetDlHarqProcessStatus ((*itTb).second.harqProcessId);
                            }
                        }
                    }   // end if ((*itTb).second.downlink) HARQ
                }   // end if (!(*itTb).second.harqFeedbackSent)
            }
        }
    }

  // send DL HARQ feedback to LtePhy
  std::map <uint16_t, DlInfoListElement_s>::iterator itHarq;
  for (itHarq = harqDlInfoMap.begin (); itHarq != harqDlInfoMap.end (); itHarq++)
    {
      if (!m_ltePhyDlHarqFeedbackCallback.IsNull ())
        {
          m_ltePhyDlHarqFeedbackCallback ((*itHarq).second);
        }
    }
  // forward control messages of this frame to LtePhy
  if (!m_rxControlMessageList.empty ())
    {
      if (!m_ltePhyRxCtrlEndOkCallback.IsNull ())
        {
          m_ltePhyRxCtrlEndOkCallback (m_rxControlMessageList);
        }
    }
  ChangeState (IDLE);
  m_rxPacketBurstList.clear ();
  m_rxControlMessageList.clear ();
  m_expectedTbs.clear ();
}


void
LteSpectrumPhy::EndRxDlCtrl ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " state: " << m_state);

  NS_ASSERT (m_state == RX_DL_CTRL);

  // this will trigger CQI calculation and Error Model evaluation
  // as a side effect, the error model should update the error status of all TBs
  m_interferenceCtrl->EndRx ();
  // apply transmission mode gain
  NS_LOG_DEBUG (this << " txMode " << (uint16_t)m_transmissionMode << " gain " << m_txModeGain.at (m_transmissionMode));
  NS_ASSERT (m_transmissionMode < m_txModeGain.size ());
  if (m_transmissionMode > 0)
    {
      // in case of MIMO, ctrl is always txed as TX diversity
      m_sinrPerceived *= m_txModeGain.at (1);
    }
//   m_sinrPerceived *= m_txModeGain.at (m_transmissionMode);
  bool error = false;
  if (m_ctrlErrorModelEnabled)
    {
      double  errorRate = LteMiErrorModel::GetPcfichPdcchError (m_sinrPerceived);
      error = !(m_random->GetValue () > errorRate);
      NS_LOG_DEBUG (this << " PCFICH-PDCCH Decodification, errorRate " << errorRate << " error " << error);
    }

  if (!error)
    {
      if (!m_ltePhyRxCtrlEndOkCallback.IsNull ())
        {
          NS_LOG_DEBUG (this << " PCFICH-PDCCH Rxed OK");
          m_ltePhyRxCtrlEndOkCallback (m_rxControlMessageList);
        }
    }
  else
    {
      if (!m_ltePhyRxCtrlEndErrorCallback.IsNull ())
        {
          NS_LOG_DEBUG (this << " PCFICH-PDCCH Error");
          m_ltePhyRxCtrlEndErrorCallback ();
        }
    }
  ChangeState (IDLE);
  m_rxControlMessageList.clear ();
}

void
LteSpectrumPhy::EndRxUlSrs ()
{
  NS_ASSERT (m_state == RX_UL_SRS);
  ChangeState (IDLE);
  m_interferenceCtrl->EndRx ();
  // nothing to do (used only for SRS at this stage)
}

void
LteSpectrumPhy::SetCellId (uint16_t cellId)
{
  m_cellId = cellId;
}

void
LteSpectrumPhy::SetComponentCarrierId (uint8_t componentCarrierId)
{
  m_componentCarrierId = componentCarrierId;
}

void
LteSpectrumPhy::AddRsPowerChunkProcessor (Ptr<LteChunkProcessor> p)
{
  m_interferenceCtrl->AddRsPowerChunkProcessor (p);
}

void
LteSpectrumPhy::AddDataPowerChunkProcessor (Ptr<LteChunkProcessor> p)
{
  m_interferenceData->AddRsPowerChunkProcessor (p);
}

void
LteSpectrumPhy::AddDataSinrChunkProcessor (Ptr<LteChunkProcessor> p)
{
  m_interferenceData->AddSinrChunkProcessor (p);
}

void
LteSpectrumPhy::AddInterferenceCtrlChunkProcessor (Ptr<LteChunkProcessor> p)
{
  m_interferenceCtrl->AddInterferenceChunkProcessor (p);
}

void
LteSpectrumPhy::AddInterferenceDataChunkProcessor (Ptr<LteChunkProcessor> p)
{
  m_interferenceData->AddInterferenceChunkProcessor (p);
}

void
LteSpectrumPhy::AddCtrlSinrChunkProcessor (Ptr<LteChunkProcessor> p)
{
  m_interferenceCtrl->AddSinrChunkProcessor (p);
}

void
LteSpectrumPhy::SetTransmissionMode (uint8_t txMode)
{
  NS_LOG_FUNCTION (this << (uint16_t) txMode);
  NS_ASSERT_MSG (txMode < m_txModeGain.size (), "TransmissionMode not available: 1.." << m_txModeGain.size ());
  m_transmissionMode = txMode;
  m_layersNum = TransmissionModesLayers::TxMode2LayerNum (txMode);
}


void
LteSpectrumPhy::SetTxModeGain (uint8_t txMode, double gain)
{
  NS_LOG_FUNCTION (this << " txmode " << (uint16_t)txMode << " gain " << gain);
  // convert to linear
  gain = std::pow (10.0, (gain / 10.0));
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
          m_txModeGain.push_back (gain);
        }
      else
        {
          m_txModeGain.push_back (temp.at (i));
        }
    }
}

int64_t
LteSpectrumPhy::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_random->SetStream (stream);
  return 1;
}



} // namespace ns3
