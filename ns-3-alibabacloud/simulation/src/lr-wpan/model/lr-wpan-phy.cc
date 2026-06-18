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
#include "lr-wpan-phy.h"
#include "lr-wpan-lqi-tag.h"
#include "lr-wpan-spectrum-signal-parameters.h"
#include "lr-wpan-spectrum-value-helper.h"
#include "lr-wpan-error-model.h"
#include "lr-wpan-net-device.h"
#include <ns3/log.h>
#include <ns3/abort.h>
#include <ns3/simulator.h>
#include <ns3/spectrum-value.h>
#include <ns3/antenna-model.h>
#include <ns3/mobility-model.h>
#include <ns3/spectrum-channel.h>
#include <ns3/packet.h>
#include <ns3/packet-burst.h>
#include <ns3/net-device.h>
#include <ns3/random-variable-stream.h>
#include <ns3/double.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LrWpanPhy");

NS_OBJECT_ENSURE_REGISTERED (LrWpanPhy);

// Table 22 in section 6.4.1 of ieee802.15.4
const uint32_t LrWpanPhy::aMaxPhyPacketSize = 127; // max PSDU in octets
const uint32_t LrWpanPhy::aTurnaroundTime = 12;  // RX-to-TX or TX-to-RX in symbol periods

// IEEE802.15.4-2006 Table 2 in section 6.1.2 (kb/s and ksymbol/s)
// The index follows LrWpanPhyOption
const LrWpanPhyDataAndSymbolRates
LrWpanPhy::dataSymbolRates[7] = { { 20.0, 20.0},
                                  { 40.0, 40.0},
                                  { 250.0, 12.5},
                                  { 250.0, 50.0},
                                  { 100.0, 25.0},
                                  { 250.0, 62.5},
                                  { 250.0, 62.5}};
// IEEE802.15.4-2006 Table 19 and Table 20 in section 6.3.
// The PHR is 1 octet and it follows phySymbolsPerOctet in Table 23
// The index follows LrWpanPhyOption
const LrWpanPhyPpduHeaderSymbolNumber
LrWpanPhy::ppduHeaderSymbolNumbers[7] = { { 32.0, 8.0, 8.0},
                                          { 32.0, 8.0, 8.0},
                                          { 2.0, 1.0, 0.4},
                                          { 6.0, 1.0, 1.6},
                                          { 8.0, 2.0, 2.0},
                                          { 8.0, 2.0, 2.0},
                                          { 8.0, 2.0, 2.0}};

TypeId
LrWpanPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LrWpanPhy")
    .SetParent<SpectrumPhy> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<LrWpanPhy> ()
    .AddTraceSource ("TrxStateValue",
                     "The state of the transceiver",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_trxState),
                     "ns3::TracedValueCallback::LrWpanPhyEnumeration")
    .AddTraceSource ("TrxState",
                     "The state of the transceiver",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_trxStateLogger),
                     "ns3::LrWpanPhy::StateTracedCallback")
    .AddTraceSource ("PhyTxBegin",
                     "Trace source indicating a packet has "
                     "begun transmitting over the channel medium",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_phyTxBeginTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxEnd",
                     "Trace source indicating a packet has been "
                     "completely transmitted over the channel.",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_phyTxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxDrop",
                     "Trace source indicating a packet has been "
                     "dropped by the device during transmission",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_phyTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxBegin",
                     "Trace source indicating a packet has begun "
                     "being received from the channel medium by the device",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_phyRxBeginTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxEnd",
                     "Trace source indicating a packet has been "
                     "completely received from the channel medium "
                     "by the device",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_phyRxEndTrace),
                     "ns3::Packet::SinrTracedCallback")
    .AddTraceSource ("PhyRxDrop",
                     "Trace source indicating a packet has been "
                     "dropped by the device during reception",
                     MakeTraceSourceAccessor (&LrWpanPhy::m_phyRxDropTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

LrWpanPhy::LrWpanPhy (void)
  : m_edRequest (),
  m_setTRXState ()
{
  m_trxState = IEEE_802_15_4_PHY_TRX_OFF;
  m_trxStatePending = IEEE_802_15_4_PHY_IDLE;

  // default PHY PIB attributes
  m_phyPIBAttributes.phyCurrentChannel = 11;
  m_phyPIBAttributes.phyTransmitPower = 0;
  m_phyPIBAttributes.phyCurrentPage = 0;
  for (uint32_t i = 0; i < 32; i++)
    {
      m_phyPIBAttributes.phyChannelsSupported[i] = 0x07ffffff;
    }
  m_phyPIBAttributes.phyCCAMode = 1;

  SetMyPhyOption ();

  m_edPower.averagePower = 0.0;
  m_edPower.lastUpdate = Seconds (0.0);
  m_edPower.measurementLength = Seconds (0.0);

  // default -110 dBm in W for 2.4 GHz
  m_rxSensitivity = pow (10.0, -106.58 / 10.0) / 1000.0;
  LrWpanSpectrumValueHelper psdHelper;
  m_txPsd = psdHelper.CreateTxPowerSpectralDensity (GetNominalTxPowerFromPib (m_phyPIBAttributes.phyTransmitPower),
                                                    m_phyPIBAttributes.phyCurrentChannel);
  m_noise = psdHelper.CreateNoisePowerSpectralDensity (m_phyPIBAttributes.phyCurrentChannel);
  m_signal = Create<LrWpanInterferenceHelper> (m_noise->GetSpectrumModel ());
  m_rxLastUpdate = Seconds (0);
  Ptr<Packet> none_packet = 0;
  Ptr<LrWpanSpectrumSignalParameters> none_params = 0;
  m_currentRxPacket = std::make_pair (none_params, true);
  m_currentTxPacket = std::make_pair (none_packet, true);
  m_errorModel = 0;

  m_random = CreateObject<UniformRandomVariable> ();
  m_random->SetAttribute ("Min", DoubleValue (0.0));
  m_random->SetAttribute ("Max", DoubleValue (1.0));


  ChangeTrxState (IEEE_802_15_4_PHY_TRX_OFF);
}

LrWpanPhy::~LrWpanPhy (void)
{
}

void
LrWpanPhy::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  // Cancel pending transceiver state change, if one is in progress.
  m_setTRXState.Cancel ();
  m_trxState = IEEE_802_15_4_PHY_TRX_OFF;
  m_trxStatePending = IEEE_802_15_4_PHY_IDLE;

  m_mobility = 0;
  m_device = 0;
  m_channel = 0;
  m_txPsd = 0;
  m_noise = 0;
  m_signal = 0;
  m_errorModel = 0;
  m_pdDataIndicationCallback = MakeNullCallback< void, uint32_t, Ptr<Packet>, uint8_t > ();
  m_pdDataConfirmCallback = MakeNullCallback< void, LrWpanPhyEnumeration > ();
  m_plmeCcaConfirmCallback = MakeNullCallback< void, LrWpanPhyEnumeration > ();
  m_plmeEdConfirmCallback = MakeNullCallback< void, LrWpanPhyEnumeration,uint8_t > ();
  m_plmeGetAttributeConfirmCallback = MakeNullCallback< void, LrWpanPhyEnumeration, LrWpanPibAttributeIdentifier, LrWpanPhyPibAttributes* > ();
  m_plmeSetTRXStateConfirmCallback = MakeNullCallback< void, LrWpanPhyEnumeration > ();
  m_plmeSetAttributeConfirmCallback = MakeNullCallback< void, LrWpanPhyEnumeration, LrWpanPibAttributeIdentifier > ();

  SpectrumPhy::DoDispose ();
}

Ptr<NetDevice>
LrWpanPhy::GetDevice (void) const
{
  NS_LOG_FUNCTION (this);
  return m_device;
}


Ptr<MobilityModel>
LrWpanPhy::GetMobility (void) const
{
  NS_LOG_FUNCTION (this);
  return m_mobility;
}


void
LrWpanPhy::SetDevice (Ptr<NetDevice> d)
{
  NS_LOG_FUNCTION (this << d);
  m_device = d;
}


void
LrWpanPhy::SetMobility (Ptr<MobilityModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_mobility = m;
}


void
LrWpanPhy::SetChannel (Ptr<SpectrumChannel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}


Ptr<SpectrumChannel>
LrWpanPhy::GetChannel (void)
{
  NS_LOG_FUNCTION (this);
  return m_channel;
}


Ptr<const SpectrumModel>
LrWpanPhy::GetRxSpectrumModel (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_txPsd)
    {
      return m_txPsd->GetSpectrumModel ();
    }
  else
    {
      return 0;
    }
}

Ptr<Object>
LrWpanPhy::GetAntenna (void) const
{
  NS_LOG_FUNCTION (this);
  return m_antenna;
}

void
LrWpanPhy::SetAntenna (Ptr<AntennaModel> a)
{
  NS_LOG_FUNCTION (this << a);
  m_antenna = a;
}

void
LrWpanPhy::StartRx (Ptr<SpectrumSignalParameters> spectrumRxParams)
{
  NS_LOG_FUNCTION (this << spectrumRxParams);
  LrWpanSpectrumValueHelper psdHelper;

  if (!m_edRequest.IsExpired ())
    {
      // Update the average receive power during ED.
      Time now = Simulator::Now ();
      m_edPower.averagePower += LrWpanSpectrumValueHelper::TotalAvgPower (m_signal->GetSignalPsd (), m_phyPIBAttributes.phyCurrentChannel) * (now - m_edPower.lastUpdate).GetTimeStep () / m_edPower.measurementLength.GetTimeStep ();
      m_edPower.lastUpdate = now;
    }

  Ptr<LrWpanSpectrumSignalParameters> lrWpanRxParams = DynamicCast<LrWpanSpectrumSignalParameters> (spectrumRxParams);

  if (lrWpanRxParams == 0)
    {
      CheckInterference ();
      m_signal->AddSignal (spectrumRxParams->psd);

      // Update peak power if CCA is in progress.
      if (!m_ccaRequest.IsExpired ())
        {
          double power = LrWpanSpectrumValueHelper::TotalAvgPower (m_signal->GetSignalPsd (), m_phyPIBAttributes.phyCurrentChannel);
          if (m_ccaPeakPower < power)
            {
              m_ccaPeakPower = power;
            }
        }

      Simulator::Schedule (spectrumRxParams->duration, &LrWpanPhy::EndRx, this, spectrumRxParams);
      return;
    }

  Ptr<Packet> p = (lrWpanRxParams->packetBurst->GetPackets ()).front ();
  NS_ASSERT (p != 0);

  // Prevent PHY from receiving another packet while switching the transceiver state.
  if (m_trxState == IEEE_802_15_4_PHY_RX_ON && !m_setTRXState.IsRunning ())
    {
      // The specification doesn't seem to refer to BUSY_RX, but vendor
      // data sheets suggest that this is a substate of the RX_ON state
      // that is entered after preamble detection when the digital receiver
      // is enabled.  Here, for now, we use BUSY_RX to mark the period between
      // StartRx() and EndRx() states.

      // We are going to BUSY_RX state when receiving the first bit of an SHR,
      // as opposed to real receivers, which should go to this state only after
      // successfully receiving the SHR.

      // If synchronizing to the packet is possible, change to BUSY_RX state,
      // otherwise drop the packet and stay in RX state. The actual synchronization
      // is not modeled.

      // Add any incoming packet to the current interference before checking the
      // SINR.
      NS_LOG_DEBUG (this << " receiving packet with power: " << 10 * log10 (LrWpanSpectrumValueHelper::TotalAvgPower (lrWpanRxParams->psd, m_phyPIBAttributes.phyCurrentChannel)) + 30 << "dBm");
      m_signal->AddSignal (lrWpanRxParams->psd);
      Ptr<SpectrumValue> interferenceAndNoise = m_signal->GetSignalPsd ();
      *interferenceAndNoise -= *lrWpanRxParams->psd;
      *interferenceAndNoise += *m_noise;
      double sinr = LrWpanSpectrumValueHelper::TotalAvgPower (lrWpanRxParams->psd, m_phyPIBAttributes.phyCurrentChannel) / LrWpanSpectrumValueHelper::TotalAvgPower (interferenceAndNoise, m_phyPIBAttributes.phyCurrentChannel);

      // Std. 802.15.4-2006, appendix E, Figure E.2
      // At SNR < -5 the BER is less than 10e-1.
      // It's useless to even *try* to decode the packet.
      if (10 * log10 (sinr) > -5)
        {
          ChangeTrxState (IEEE_802_15_4_PHY_BUSY_RX);
          m_currentRxPacket = std::make_pair (lrWpanRxParams, false);
          m_phyRxBeginTrace (p);

          m_rxLastUpdate = Simulator::Now ();
        }
      else
        {
          m_phyRxDropTrace (p);
        }
    }
  else if (m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
    {
      // Drop the new packet.
      NS_LOG_DEBUG (this << " packet collision");
      m_phyRxDropTrace (p);

      // Check if we correctly received the old packet up to now.
      CheckInterference ();

      // Add the incoming packet to the current interference after we have
      // checked for successful reception of the current packet for the time
      // before the additional interference.
      m_signal->AddSignal (lrWpanRxParams->psd);
    }
  else
    {
      // Simply drop the packet.
      NS_LOG_DEBUG (this << " transceiver not in RX state");
      m_phyRxDropTrace (p);

      // Add the signal power to the interference, anyway.
      m_signal->AddSignal (lrWpanRxParams->psd);
    }

  // Update peak power if CCA is in progress.
  if (!m_ccaRequest.IsExpired ())
    {
      double power = LrWpanSpectrumValueHelper::TotalAvgPower (m_signal->GetSignalPsd (), m_phyPIBAttributes.phyCurrentChannel);
      if (m_ccaPeakPower < power)
        {
          m_ccaPeakPower = power;
        }
    }

  // Always call EndRx to update the interference.
  // \todo: Do we need to keep track of these events to unschedule them when disposing off the PHY?

  Simulator::Schedule (spectrumRxParams->duration, &LrWpanPhy::EndRx, this, spectrumRxParams);
}

void
LrWpanPhy::CheckInterference (void)
{
  // Calculate whether packet was lost.
  LrWpanSpectrumValueHelper psdHelper;
  Ptr<LrWpanSpectrumSignalParameters> currentRxParams = m_currentRxPacket.first;

  // We are currently receiving a packet.
  if (m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
    {
      // NS_ASSERT (currentRxParams && !m_currentRxPacket.second);

      Ptr<Packet> currentPacket = currentRxParams->packetBurst->GetPackets ().front ();
      if (m_errorModel != 0)
        {
          // How many bits did we receive since the last calculation?
          double t = (Simulator::Now () - m_rxLastUpdate).ToDouble (Time::MS);
          uint32_t chunkSize = ceil (t * (GetDataOrSymbolRate (true) / 1000));
          Ptr<SpectrumValue> interferenceAndNoise = m_signal->GetSignalPsd ();
          *interferenceAndNoise -= *currentRxParams->psd;
          *interferenceAndNoise += *m_noise;
          double sinr = LrWpanSpectrumValueHelper::TotalAvgPower (currentRxParams->psd, m_phyPIBAttributes.phyCurrentChannel) / LrWpanSpectrumValueHelper::TotalAvgPower (interferenceAndNoise, m_phyPIBAttributes.phyCurrentChannel);
          double per = 1.0 - m_errorModel->GetChunkSuccessRate (sinr, chunkSize);

          // The LQI is the total packet success rate scaled to 0-255.
          // If not already set, initialize to 255.
          LrWpanLqiTag tag (std::numeric_limits<uint8_t>::max ());
          currentPacket->PeekPacketTag (tag);
          uint8_t lqi = tag.Get ();
          tag.Set (lqi - (per * lqi));
          currentPacket->ReplacePacketTag (tag);

          if (m_random->GetValue () < per)
            {
              // The packet was destroyed, drop the packet after reception.
              m_currentRxPacket.second = true;
            }
        }
      else
        {
          NS_LOG_WARN ("Missing ErrorModel");
        }
    }
  m_rxLastUpdate = Simulator::Now ();
}

void
LrWpanPhy::EndRx (Ptr<SpectrumSignalParameters> par)
{
  NS_LOG_FUNCTION (this);

  Ptr<LrWpanSpectrumSignalParameters> params = DynamicCast<LrWpanSpectrumSignalParameters> (par);

  if (!m_edRequest.IsExpired ())
    {
      // Update the average receive power during ED.
      Time now = Simulator::Now ();
      m_edPower.averagePower += LrWpanSpectrumValueHelper::TotalAvgPower (m_signal->GetSignalPsd (), m_phyPIBAttributes.phyCurrentChannel) * (now - m_edPower.lastUpdate).GetTimeStep () / m_edPower.measurementLength.GetTimeStep ();
      m_edPower.lastUpdate = now;
    }

  Ptr<LrWpanSpectrumSignalParameters> currentRxParams = m_currentRxPacket.first;
  if (currentRxParams == params)
    {
      CheckInterference ();
    }

  // Update the interference.
  m_signal->RemoveSignal (par->psd);

  if (params == 0)
    {
      NS_LOG_LOGIC ("Node: " << m_device->GetAddress () << " Removing interferent: " << *(par->psd));
      return;
    }

  // If this is the end of the currently received packet, check if reception was successful.
  if (currentRxParams == params)
    {
      Ptr<Packet> currentPacket = currentRxParams->packetBurst->GetPackets ().front ();
      NS_ASSERT (currentPacket != 0);

      // If there is no error model attached to the PHY, we always report the maximum LQI value.
      LrWpanLqiTag tag (std::numeric_limits<uint8_t>::max ());
      currentPacket->PeekPacketTag (tag);
      m_phyRxEndTrace (currentPacket, tag.Get ());

      if (!m_currentRxPacket.second)
        {
          // The packet was successfully received, push it up the stack.
          if (!m_pdDataIndicationCallback.IsNull ())
            {
              m_pdDataIndicationCallback (currentPacket->GetSize (), currentPacket, tag.Get ());
            }
        }
      else
        {
          // The packet was destroyed, drop it.
          m_phyRxDropTrace (currentPacket);
        }
      Ptr<LrWpanSpectrumSignalParameters> none = 0;
      m_currentRxPacket = std::make_pair (none, true);

      // We may be waiting to apply a pending state change.
      if (m_trxStatePending != IEEE_802_15_4_PHY_IDLE)
        {
          // Only change the state immediately, if the transceiver is not already
          // switching the state.
          if (!m_setTRXState.IsRunning ())
            {
              NS_LOG_LOGIC ("Apply pending state change to " << m_trxStatePending);
              ChangeTrxState (m_trxStatePending);
              m_trxStatePending = IEEE_802_15_4_PHY_IDLE;
              if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
                {
                  m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_SUCCESS);
                }
            }
        }
      else
        {
          ChangeTrxState (IEEE_802_15_4_PHY_RX_ON);
        }
    }
}

void
LrWpanPhy::PdDataRequest (const uint32_t psduLength, Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << psduLength << p);

  if (psduLength > aMaxPhyPacketSize)
    {
      if (!m_pdDataConfirmCallback.IsNull ())
        {
          m_pdDataConfirmCallback (IEEE_802_15_4_PHY_UNSPECIFIED);
        }
      NS_LOG_DEBUG ("Drop packet because psduLength too long: " << psduLength);
      return;
    }

  // Prevent PHY from sending a packet while switching the transceiver state.
  if (!m_setTRXState.IsRunning ())
    {
      if (m_trxState == IEEE_802_15_4_PHY_TX_ON)
        {
          //send down
          NS_ASSERT (m_channel);

          // Remove a possible LQI tag from a previous transmission of the packet.
          LrWpanLqiTag lqiTag;
          p->RemovePacketTag (lqiTag);

          m_phyTxBeginTrace (p);
          m_currentTxPacket.first = p;
          m_currentTxPacket.second = false;

          Ptr<LrWpanSpectrumSignalParameters> txParams = Create<LrWpanSpectrumSignalParameters> ();
          txParams->duration = CalculateTxTime (p);
          txParams->txPhy = GetObject<SpectrumPhy> ();
          txParams->psd = m_txPsd;
          txParams->txAntenna = m_antenna;
          Ptr<PacketBurst> pb = CreateObject<PacketBurst> ();
          pb->AddPacket (p);
          txParams->packetBurst = pb;
          m_channel->StartTx (txParams);
          m_pdDataRequest = Simulator::Schedule (txParams->duration, &LrWpanPhy::EndTx, this);
          ChangeTrxState (IEEE_802_15_4_PHY_BUSY_TX);
          return;
        }
      else if ((m_trxState == IEEE_802_15_4_PHY_RX_ON)
               || (m_trxState == IEEE_802_15_4_PHY_TRX_OFF)
               || (m_trxState == IEEE_802_15_4_PHY_BUSY_TX) )
        {
          if (!m_pdDataConfirmCallback.IsNull ())
            {
              m_pdDataConfirmCallback (m_trxState);
            }
          // Drop packet, hit PhyTxDrop trace
          m_phyTxDropTrace (p);
          return;
        }
      else
        {
          NS_FATAL_ERROR ("This should be unreachable, or else state " << m_trxState << " should be added as a case");
        }
    }
  else
    {
      // TODO: This error code is not covered by the standard.
      // What is the correct behavior in this case?
      if (!m_pdDataConfirmCallback.IsNull ())
        {
          m_pdDataConfirmCallback (IEEE_802_15_4_PHY_UNSPECIFIED);
        }
      // Drop packet, hit PhyTxDrop trace
      m_phyTxDropTrace (p);
      return;
    }
}

void
LrWpanPhy::PlmeCcaRequest (void)
{
  NS_LOG_FUNCTION (this);

  if (m_trxState == IEEE_802_15_4_PHY_RX_ON || m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
    {
      m_ccaPeakPower = 0.0;
      Time ccaTime = Seconds (8.0 / GetDataOrSymbolRate (false));
      m_ccaRequest = Simulator::Schedule (ccaTime, &LrWpanPhy::EndCca, this);
    }
  else
    {
      if (!m_plmeCcaConfirmCallback.IsNull ())
        {
          if (m_trxState == IEEE_802_15_4_PHY_TRX_OFF)
            {
              m_plmeCcaConfirmCallback (IEEE_802_15_4_PHY_TRX_OFF);
            }
          else
            {
              m_plmeCcaConfirmCallback (IEEE_802_15_4_PHY_BUSY);
            }
        }
    }
}

void
LrWpanPhy::PlmeEdRequest (void)
{
  NS_LOG_FUNCTION (this);
  if (m_trxState == IEEE_802_15_4_PHY_RX_ON || m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
    {
      // Average over the powers of all signals received until EndEd()
      m_edPower.averagePower = 0;
      m_edPower.lastUpdate = Simulator::Now ();
      m_edPower.measurementLength = Seconds (8.0 / GetDataOrSymbolRate (false));
      m_edRequest = Simulator::Schedule (m_edPower.measurementLength, &LrWpanPhy::EndEd, this);
    }
  else
    {
      LrWpanPhyEnumeration result = m_trxState;
      if (m_trxState == IEEE_802_15_4_PHY_BUSY_TX)
        {
          result = IEEE_802_15_4_PHY_TX_ON;
        }

      if (!m_plmeEdConfirmCallback.IsNull ())
        {
          m_plmeEdConfirmCallback (result, 0);
        }
    }
}

void
LrWpanPhy::PlmeGetAttributeRequest (LrWpanPibAttributeIdentifier id)
{
  NS_LOG_FUNCTION (this << id);
  LrWpanPhyEnumeration status;

  switch (id)
    {
    case phyCurrentChannel:
    case phyChannelsSupported:
    case phyTransmitPower:
    case phyCCAMode:
    case phyCurrentPage:
    case phyMaxFrameDuration:
    case phySHRDuration:
    case phySymbolsPerOctet:
      {
        status = IEEE_802_15_4_PHY_SUCCESS;
        break;
      }
    default:
      {
        status = IEEE_802_15_4_PHY_UNSUPPORTED_ATTRIBUTE;
        break;
      }
    }
  if (!m_plmeGetAttributeConfirmCallback.IsNull ())
    {
      LrWpanPhyPibAttributes retValue;
      memcpy (&retValue, &m_phyPIBAttributes, sizeof(LrWpanPhyPibAttributes));
      m_plmeGetAttributeConfirmCallback (status,id,&retValue);
    }
}

// Section 6.2.2.7.3
void
LrWpanPhy::PlmeSetTRXStateRequest (LrWpanPhyEnumeration state)
{
  NS_LOG_FUNCTION (this << state);

  // Check valid states (Table 14)
  NS_ABORT_IF ( (state != IEEE_802_15_4_PHY_RX_ON)
                && (state != IEEE_802_15_4_PHY_TRX_OFF)
                && (state != IEEE_802_15_4_PHY_FORCE_TRX_OFF)
                && (state != IEEE_802_15_4_PHY_TX_ON) );

  NS_LOG_LOGIC ("Trying to set m_trxState from " << m_trxState << " to " << state);
  // this method always overrides previous state setting attempts
  if (!m_setTRXState.IsExpired ())
    {
      if (m_trxStatePending == state)
        {
          // Simply wait for the ongoing state switch.
          return;
        }
      else
        {
          NS_LOG_DEBUG ("Cancel m_setTRXState");
          // Keep the transceiver state as the old state before the switching attempt.
          m_setTRXState.Cancel ();
        }
    }
  if (m_trxStatePending != IEEE_802_15_4_PHY_IDLE)
    {
      m_trxStatePending = IEEE_802_15_4_PHY_IDLE;
    }

  if (state == m_trxState)
    {
      if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
        {
          m_plmeSetTRXStateConfirmCallback (state);
        }
      return;
    }

  if ( ((state == IEEE_802_15_4_PHY_RX_ON)
        || (state == IEEE_802_15_4_PHY_TRX_OFF))
       && (m_trxState == IEEE_802_15_4_PHY_BUSY_TX) )
    {
      NS_LOG_DEBUG ("Phy is busy; setting state pending to " << state);
      m_trxStatePending = state;
      return;  // Send PlmeSetTRXStateConfirm later
    }

  // specification talks about being in RX_ON and having received
  // a valid SFD.  Here, we are not modelling at that level of
  // granularity, so we just test for BUSY_RX state (any part of
  // a packet being actively received)
  if (state == IEEE_802_15_4_PHY_TRX_OFF)
    {
      CancelEd (state);

      if ((m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
          && (m_currentRxPacket.first) && (!m_currentRxPacket.second))
        {
          NS_LOG_DEBUG ("Receiver has valid SFD; defer state change");
          m_trxStatePending = state;
          return;  // Send PlmeSetTRXStateConfirm later
        }
      else if (m_trxState == IEEE_802_15_4_PHY_RX_ON || m_trxState == IEEE_802_15_4_PHY_TX_ON)
        {
          ChangeTrxState (IEEE_802_15_4_PHY_TRX_OFF);
          if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
            {
              m_plmeSetTRXStateConfirmCallback (state);
            }
          return;
        }
    }

  if (state == IEEE_802_15_4_PHY_TX_ON)
    {
      CancelEd (state);

      NS_LOG_DEBUG ("turn on PHY_TX_ON");
      if ((m_trxState == IEEE_802_15_4_PHY_BUSY_RX) || (m_trxState == IEEE_802_15_4_PHY_RX_ON))
        {
          if (m_currentRxPacket.first)
            {
              //terminate reception if needed
              //incomplete reception -- force packet discard
              NS_LOG_DEBUG ("force TX_ON, terminate reception");
              m_currentRxPacket.second = true;
            }

          // If CCA is in progress, cancel CCA and return BUSY.
          if (!m_ccaRequest.IsExpired ())
            {
              m_ccaRequest.Cancel ();
              if (!m_plmeCcaConfirmCallback.IsNull ())
                {
                  m_plmeCcaConfirmCallback (IEEE_802_15_4_PHY_BUSY);
                }
            }

          m_trxStatePending = IEEE_802_15_4_PHY_TX_ON;

          // Delay for turnaround time
          // TODO: Does it also take aTurnaroundTime to switch the transceiver state,
          //       even when the receiver is not busy? (6.9.2)
          Time setTime = Seconds ( (double) aTurnaroundTime / GetDataOrSymbolRate (false));
          m_setTRXState = Simulator::Schedule (setTime, &LrWpanPhy::EndSetTRXState, this);
          return;
        }
      else if (m_trxState == IEEE_802_15_4_PHY_BUSY_TX || m_trxState == IEEE_802_15_4_PHY_TX_ON)
        {
          // We do NOT change the transceiver state here. We only report that
          // the transceiver is already in TX_ON state.
          if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
            {
              m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_TX_ON);
            }
          return;
        }
      else if (m_trxState == IEEE_802_15_4_PHY_TRX_OFF)
        {
          ChangeTrxState (IEEE_802_15_4_PHY_TX_ON);
          if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
            {
              m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_TX_ON);
            }
          return;
        }
    }

  if (state == IEEE_802_15_4_PHY_FORCE_TRX_OFF)
    {
      if (m_trxState == IEEE_802_15_4_PHY_TRX_OFF)
        {
          NS_LOG_DEBUG ("force TRX_OFF, was already off");
        }
      else
        {
          NS_LOG_DEBUG ("force TRX_OFF, SUCCESS");
          if (m_currentRxPacket.first)
            {   //terminate reception if needed
                //incomplete reception -- force packet discard
              NS_LOG_DEBUG ("force TRX_OFF, terminate reception");
              m_currentRxPacket.second = true;
            }
          if (m_trxState == IEEE_802_15_4_PHY_BUSY_TX)
            {
              NS_LOG_DEBUG ("force TRX_OFF, terminate transmission");
              m_currentTxPacket.second = true;
            }
          ChangeTrxState (IEEE_802_15_4_PHY_TRX_OFF);
          // Clear any other state
          m_trxStatePending = IEEE_802_15_4_PHY_IDLE;
        }
      if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
        {
          m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_SUCCESS);
        }
      return;
    }

  if (state == IEEE_802_15_4_PHY_RX_ON)
    {
      if (m_trxState == IEEE_802_15_4_PHY_TX_ON || m_trxState == IEEE_802_15_4_PHY_TRX_OFF)
        {
          // Turnaround delay
          // TODO: Does it really take aTurnaroundTime to switch the transceiver state,
          //       even when the transmitter is not busy? (6.9.1)
          m_trxStatePending = IEEE_802_15_4_PHY_RX_ON;

          Time setTime = Seconds ( (double) aTurnaroundTime / GetDataOrSymbolRate (false));
          m_setTRXState = Simulator::Schedule (setTime, &LrWpanPhy::EndSetTRXState, this);
          return;
        }
      else if (m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
        {
          if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
            {
              m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_RX_ON);
            }
          return;
        }
    }

  NS_FATAL_ERROR ("Unexpected transition from state " << m_trxState << " to state " << state);
}

bool
LrWpanPhy::ChannelSupported (uint8_t channel)
{
  NS_LOG_FUNCTION (this << channel);
  bool retValue = false;

  for (uint32_t i = 0; i < 32; i++)
    {
      if ((m_phyPIBAttributes.phyChannelsSupported[i] & (1 << channel)) != 0)
        {
          retValue = true;
          break;
        }
    }

  return retValue;
}

void
LrWpanPhy::PlmeSetAttributeRequest (LrWpanPibAttributeIdentifier id,
                                    LrWpanPhyPibAttributes* attribute)
{
  NS_LOG_FUNCTION (this << id << attribute);
  NS_ASSERT (attribute);
  LrWpanPhyEnumeration status = IEEE_802_15_4_PHY_SUCCESS;

  switch (id)
    {
    case phyCurrentChannel:
      {
        if (!ChannelSupported (attribute->phyCurrentChannel))
          {
            status = IEEE_802_15_4_PHY_INVALID_PARAMETER;
          }
        if (m_phyPIBAttributes.phyCurrentChannel != attribute->phyCurrentChannel)
          {
            // Cancel a pending transceiver state change.
            // Switch off the transceiver.
            // TODO: Is switching off the transceiver the right choice?
            m_trxState = IEEE_802_15_4_PHY_TRX_OFF;
            if (m_trxStatePending != IEEE_802_15_4_PHY_IDLE)
              {
                m_trxStatePending = IEEE_802_15_4_PHY_IDLE;
                m_setTRXState.Cancel ();
                if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
                  {
                    m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_TRX_OFF);
                  }
              }

            // Any packet in transmission or reception will be corrupted.
            if (m_currentRxPacket.first)
              {
                m_currentRxPacket.second = true;
              }
            if (PhyIsBusy ())
              {
                m_currentTxPacket.second = true;
                m_pdDataRequest.Cancel ();
                m_currentTxPacket.first = 0;
                if (!m_pdDataConfirmCallback.IsNull ())
                  {
                    m_pdDataConfirmCallback (IEEE_802_15_4_PHY_TRX_OFF);
                  }
              }
            m_phyPIBAttributes.phyCurrentChannel = attribute->phyCurrentChannel;
            LrWpanSpectrumValueHelper psdHelper;
            m_txPsd = psdHelper.CreateTxPowerSpectralDensity (GetNominalTxPowerFromPib (m_phyPIBAttributes.phyTransmitPower),
                                                                                        m_phyPIBAttributes.phyCurrentChannel);
          }
        break;
      }
    case phyChannelsSupported:
      {   // only the first element is considered in the array
        if ((attribute->phyChannelsSupported[0] & 0xf8000000) != 0)
          {    //5 MSBs reserved
            status = IEEE_802_15_4_PHY_INVALID_PARAMETER;
          }
        else
          {
            m_phyPIBAttributes.phyChannelsSupported[0] = attribute->phyChannelsSupported[0];
          }
        break;
      }
    case phyTransmitPower:
      {
        if (attribute->phyTransmitPower & 0xC0)
          {
            NS_LOG_LOGIC ("LrWpanPhy::PlmeSetAttributeRequest error - can not change read-only attribute bits.");
            status = IEEE_802_15_4_PHY_INVALID_PARAMETER;
          }
        else
          {
            m_phyPIBAttributes.phyTransmitPower = attribute->phyTransmitPower;
            LrWpanSpectrumValueHelper psdHelper;
            m_txPsd = psdHelper.CreateTxPowerSpectralDensity (GetNominalTxPowerFromPib (m_phyPIBAttributes.phyTransmitPower),
                                                              m_phyPIBAttributes.phyCurrentChannel);
          }
        break;
      }
    case phyCCAMode:
      {
        if ((attribute->phyCCAMode < 1) || (attribute->phyCCAMode > 3))
          {
            status = IEEE_802_15_4_PHY_INVALID_PARAMETER;
          }
        else
          {
            m_phyPIBAttributes.phyCCAMode = attribute->phyCCAMode;
          }
        break;
      }
    default:
      {
        status = IEEE_802_15_4_PHY_UNSUPPORTED_ATTRIBUTE;
        break;
      }
    }

  if (!m_plmeSetAttributeConfirmCallback.IsNull ())
    {
      m_plmeSetAttributeConfirmCallback (status, id);
    }
}

void
LrWpanPhy::SetPdDataIndicationCallback (PdDataIndicationCallback c)
{
  NS_LOG_FUNCTION (this);
  m_pdDataIndicationCallback = c;
}

void
LrWpanPhy::SetPdDataConfirmCallback (PdDataConfirmCallback c)
{
  NS_LOG_FUNCTION (this);
  m_pdDataConfirmCallback = c;
}

void
LrWpanPhy::SetPlmeCcaConfirmCallback (PlmeCcaConfirmCallback c)
{
  NS_LOG_FUNCTION (this);
  m_plmeCcaConfirmCallback = c;
}

void
LrWpanPhy::SetPlmeEdConfirmCallback (PlmeEdConfirmCallback c)
{
  NS_LOG_FUNCTION (this);
  m_plmeEdConfirmCallback = c;
}

void
LrWpanPhy::SetPlmeGetAttributeConfirmCallback (PlmeGetAttributeConfirmCallback c)
{
  NS_LOG_FUNCTION (this);
  m_plmeGetAttributeConfirmCallback = c;
}

void
LrWpanPhy::SetPlmeSetTRXStateConfirmCallback (PlmeSetTRXStateConfirmCallback c)
{
  NS_LOG_FUNCTION (this);
  m_plmeSetTRXStateConfirmCallback = c;
}

void
LrWpanPhy::SetPlmeSetAttributeConfirmCallback (PlmeSetAttributeConfirmCallback c)
{
  NS_LOG_FUNCTION (this);
  m_plmeSetAttributeConfirmCallback = c;
}

void
LrWpanPhy::ChangeTrxState (LrWpanPhyEnumeration newState)
{
  NS_LOG_LOGIC (this << " state: " << m_trxState << " -> " << newState);
  m_trxStateLogger (Simulator::Now (), m_trxState, newState);
  m_trxState = newState;
}

bool
LrWpanPhy::PhyIsBusy (void) const
{
  NS_LOG_FUNCTION (this << m_trxState);
  return ((m_trxState == IEEE_802_15_4_PHY_BUSY_TX)
          || (m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
          || (m_trxState == IEEE_802_15_4_PHY_BUSY));
}

void
LrWpanPhy::CancelEd (LrWpanPhyEnumeration state)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (state == IEEE_802_15_4_PHY_TRX_OFF || state == IEEE_802_15_4_PHY_TX_ON);

  if (!m_edRequest.IsExpired ())
    {
      m_edRequest.Cancel ();
      if (!m_plmeEdConfirmCallback.IsNull ())
        {
          m_plmeEdConfirmCallback (state, 0);
        }
    }
}

void
LrWpanPhy::EndEd (void)
{
  NS_LOG_FUNCTION (this);

  m_edPower.averagePower += LrWpanSpectrumValueHelper::TotalAvgPower (m_signal->GetSignalPsd (), m_phyPIBAttributes.phyCurrentChannel) * (Simulator::Now () - m_edPower.lastUpdate).GetTimeStep () / m_edPower.measurementLength.GetTimeStep ();

  uint8_t energyLevel;

  // Per IEEE802.15.4-2006 sec 6.9.7
  double ratio = m_edPower.averagePower / m_rxSensitivity;
  ratio = 10.0 * log10 (ratio);
  if (ratio <= 10.0)
    {  // less than 10 dB
      energyLevel = 0;
    }
  else if (ratio >= 40.0)
    { // less than 40 dB
      energyLevel = 255;
    }
  else
    {
      // in-between with linear increase per sec 6.9.7
      energyLevel = static_cast<uint8_t> (((ratio - 10.0) / 30.0) * 255.0);
    }

  if (!m_plmeEdConfirmCallback.IsNull ())
    {
      m_plmeEdConfirmCallback (IEEE_802_15_4_PHY_SUCCESS, energyLevel);
    }
}

void
LrWpanPhy::EndCca (void)
{
  NS_LOG_FUNCTION (this);
  LrWpanPhyEnumeration sensedChannelState = IEEE_802_15_4_PHY_UNSPECIFIED;

  // Update peak power.
  double power = LrWpanSpectrumValueHelper::TotalAvgPower (m_signal->GetSignalPsd (), m_phyPIBAttributes.phyCurrentChannel);
  if (m_ccaPeakPower < power)
    {
      m_ccaPeakPower = power;
    }

  if (PhyIsBusy ())
    {
      sensedChannelState = IEEE_802_15_4_PHY_BUSY;
    }
  else if (m_phyPIBAttributes.phyCCAMode == 1)
    { //sec 6.9.9 ED detection
      // -- ED threshold at most 10 dB above receiver sensitivity.
      if (10 * log10 (m_ccaPeakPower / m_rxSensitivity) >= 10.0)
        {
          sensedChannelState = IEEE_802_15_4_PHY_BUSY;
        }
      else
        {
          sensedChannelState = IEEE_802_15_4_PHY_IDLE;
        }
    }
  else if (m_phyPIBAttributes.phyCCAMode == 2)
    {
      //sec 6.9.9 carrier sense only
      if (m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
        {
          // We currently do not model PPDU reception in detail. Instead we model
          // packet reception starting with the first bit of the preamble.
          // Therefore, this code will never be reached, as PhyIsBusy() would
          // already lead to a channel busy condition.
          // TODO: Change this, if we also model preamble and SFD detection.
          sensedChannelState = IEEE_802_15_4_PHY_BUSY;
        }
      else
        {
          sensedChannelState = IEEE_802_15_4_PHY_IDLE;
        }
    }
  else if (m_phyPIBAttributes.phyCCAMode == 3)
    { //sect 6.9.9 both
      if ((10 * log10 (m_ccaPeakPower / m_rxSensitivity) >= 10.0)
          && m_trxState == IEEE_802_15_4_PHY_BUSY_RX)
        {
          // Again, this code will never be reached, if we are already receiving
          // a packet, as PhyIsBusy() would already lead to a channel busy condition.
          // TODO: Change this, if we also model preamble and SFD detection.
          sensedChannelState = IEEE_802_15_4_PHY_BUSY;
        }
      else
        {
          sensedChannelState = IEEE_802_15_4_PHY_IDLE;
        }
    }
  else
    {
      NS_ASSERT_MSG (false, "Invalid CCA mode");
    }

  NS_LOG_LOGIC (this << "channel sensed state: " << sensedChannelState);

  if (!m_plmeCcaConfirmCallback.IsNull ())
    {
      m_plmeCcaConfirmCallback (sensedChannelState);
    }
}

void
LrWpanPhy::EndSetTRXState (void)
{
  NS_LOG_FUNCTION (this);

  NS_ABORT_IF ( (m_trxStatePending != IEEE_802_15_4_PHY_RX_ON) && (m_trxStatePending != IEEE_802_15_4_PHY_TX_ON) );
  ChangeTrxState (m_trxStatePending);
  m_trxStatePending = IEEE_802_15_4_PHY_IDLE;

  if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
    {
      m_plmeSetTRXStateConfirmCallback (m_trxState);
    }
}

void
LrWpanPhy::EndTx (void)
{
  NS_LOG_FUNCTION (this);

  NS_ABORT_IF ( (m_trxState != IEEE_802_15_4_PHY_BUSY_TX) && (m_trxState != IEEE_802_15_4_PHY_TRX_OFF));

  if (m_currentTxPacket.second == false)
    {
      NS_LOG_DEBUG ("Packet successfully transmitted");
      m_phyTxEndTrace (m_currentTxPacket.first);
      if (!m_pdDataConfirmCallback.IsNull ())
        {
          m_pdDataConfirmCallback (IEEE_802_15_4_PHY_SUCCESS);
        }
    }
  else
    {
      NS_LOG_DEBUG ("Packet transmission aborted");
      m_phyTxDropTrace (m_currentTxPacket.first);
      if (!m_pdDataConfirmCallback.IsNull ())
        {
          // See if this is ever entered in another state
          NS_ASSERT (m_trxState ==  IEEE_802_15_4_PHY_TRX_OFF);
          m_pdDataConfirmCallback (m_trxState);
        }
    }
  m_currentTxPacket.first = 0;
  m_currentTxPacket.second = false;


  // We may be waiting to apply a pending state change.
  if (m_trxStatePending != IEEE_802_15_4_PHY_IDLE)
    {
      // Only change the state immediately, if the transceiver is not already
      // switching the state.
      if (!m_setTRXState.IsRunning ())
        {
          NS_LOG_LOGIC ("Apply pending state change to " << m_trxStatePending);
          ChangeTrxState (m_trxStatePending);
          m_trxStatePending = IEEE_802_15_4_PHY_IDLE;
          if (!m_plmeSetTRXStateConfirmCallback.IsNull ())
            {
              m_plmeSetTRXStateConfirmCallback (IEEE_802_15_4_PHY_SUCCESS);
            }
        }
    }
  else
    {
      if (m_trxState != IEEE_802_15_4_PHY_TRX_OFF)
        {
          ChangeTrxState (IEEE_802_15_4_PHY_TX_ON);
        }
    }
}

Time
LrWpanPhy::CalculateTxTime (Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (this << packet);

  bool isData = true;
  Time txTime = GetPpduHeaderTxTime ();

  txTime += Seconds (packet->GetSize () * 8.0 / GetDataOrSymbolRate (isData));

  return txTime;
}

double
LrWpanPhy::GetDataOrSymbolRate (bool isData)
{
  NS_LOG_FUNCTION (this << isData);

  double rate = 0.0;

  NS_ASSERT (m_phyOption < IEEE_802_15_4_INVALID_PHY_OPTION);

  if (isData)
    {
      rate = dataSymbolRates [m_phyOption].bitRate;
    }
  else
    {
      rate = dataSymbolRates [m_phyOption].symbolRate;
    }

  return (rate * 1000.0);
}

Time
LrWpanPhy::GetPpduHeaderTxTime (void)
{
  NS_LOG_FUNCTION (this);

  bool isData = false;
  double totalPpduHdrSymbols;

  NS_ASSERT (m_phyOption < IEEE_802_15_4_INVALID_PHY_OPTION);

  totalPpduHdrSymbols = ppduHeaderSymbolNumbers[m_phyOption].shrPreamble
    + ppduHeaderSymbolNumbers[m_phyOption].shrSfd
    + ppduHeaderSymbolNumbers[m_phyOption].phr;

  return Seconds (totalPpduHdrSymbols / GetDataOrSymbolRate (isData));
}

// IEEE802.15.4-2006 Table 2 in section 6.1.2
void
LrWpanPhy::SetMyPhyOption (void)
{
  NS_LOG_FUNCTION (this);

  m_phyOption = IEEE_802_15_4_INVALID_PHY_OPTION;

  if (m_phyPIBAttributes.phyCurrentPage == 0)
    {
      if (m_phyPIBAttributes.phyCurrentChannel == 0)
        {  // 868 MHz BPSK
          m_phyOption = IEEE_802_15_4_868MHZ_BPSK;
        }
      else if (m_phyPIBAttributes.phyCurrentChannel <= 10)
        {  // 915 MHz BPSK
          m_phyOption = IEEE_802_15_4_915MHZ_BPSK;
        }
      else if (m_phyPIBAttributes.phyCurrentChannel <= 26)
        {  // 2.4 GHz MHz O-QPSK
          m_phyOption = IEEE_802_15_4_2_4GHZ_OQPSK;
        }
    }
  else if (m_phyPIBAttributes.phyCurrentPage == 1)
    {
      if (m_phyPIBAttributes.phyCurrentChannel == 0)
        {  // 868 MHz ASK
          m_phyOption = IEEE_802_15_4_868MHZ_ASK;
        }
      else if (m_phyPIBAttributes.phyCurrentChannel <= 10)
        {  // 915 MHz ASK
          m_phyOption = IEEE_802_15_4_915MHZ_ASK;
        }
    }
  else if (m_phyPIBAttributes.phyCurrentPage == 2)
    {
      if (m_phyPIBAttributes.phyCurrentChannel == 0)
        {  // 868 MHz O-QPSK
          m_phyOption = IEEE_802_15_4_868MHZ_OQPSK;
        }
      else if (m_phyPIBAttributes.phyCurrentChannel <= 10)
        {  // 915 MHz O-QPSK
          m_phyOption = IEEE_802_15_4_915MHZ_OQPSK;
        }
    }
}

LrWpanPhyOption
LrWpanPhy::GetMyPhyOption (void)
{
  NS_LOG_FUNCTION (this);
  return m_phyOption;
}

void
LrWpanPhy::SetTxPowerSpectralDensity (Ptr<SpectrumValue> txPsd)
{
  NS_LOG_FUNCTION (this << txPsd);
  NS_ASSERT (txPsd);
  m_txPsd = txPsd;
  NS_LOG_INFO ("\t computed tx_psd: " << *txPsd << "\t stored tx_psd: " << *m_txPsd);
}

void
LrWpanPhy::SetNoisePowerSpectralDensity (Ptr<const SpectrumValue> noisePsd)
{
  NS_LOG_FUNCTION (this << noisePsd);
  NS_LOG_INFO ("\t computed noise_psd: " << *noisePsd );
  NS_ASSERT (noisePsd);
  m_noise = noisePsd;
}

Ptr<const SpectrumValue>
LrWpanPhy::GetNoisePowerSpectralDensity (void)
{
  NS_LOG_FUNCTION (this);
  return m_noise;
}

void
LrWpanPhy::SetErrorModel (Ptr<LrWpanErrorModel> e)
{
  NS_LOG_FUNCTION (this << e);
  NS_ASSERT (e);
  m_errorModel = e;
}

Ptr<LrWpanErrorModel>
LrWpanPhy::GetErrorModel (void) const
{
  NS_LOG_FUNCTION (this);
  return m_errorModel;
}

uint64_t
LrWpanPhy::GetPhySHRDuration (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_phyOption < IEEE_802_15_4_INVALID_PHY_OPTION);

  return ppduHeaderSymbolNumbers[m_phyOption].shrPreamble
         + ppduHeaderSymbolNumbers[m_phyOption].shrSfd;
}

double
LrWpanPhy::GetPhySymbolsPerOctet (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_phyOption < IEEE_802_15_4_INVALID_PHY_OPTION);

  return dataSymbolRates [m_phyOption].symbolRate / (dataSymbolRates [m_phyOption].bitRate / 8);
}

int8_t
LrWpanPhy::GetNominalTxPowerFromPib (uint8_t phyTransmitPower)
{
  NS_LOG_FUNCTION (this << +phyTransmitPower);

  // The nominal Tx power is stored in the PIB as a 6-bit
  // twos-complement, signed number.

  // The 5 LSBs can be copied - as their representation
  // is the same for unsigned and signed integers.
  int8_t nominalTxPower = phyTransmitPower & 0x1F;

  // Now check the 6th LSB (the "sign" bit).
  // It's a twos-complement format, so the "sign"
  // bit represents -2^5 = -32.
  if (phyTransmitPower & 0x20)
    {
      nominalTxPower -= 32;
    }
  return nominalTxPower;
}


int64_t
LrWpanPhy::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this);
  m_random->SetStream (stream);
  return 1;
}

} // namespace ns3
