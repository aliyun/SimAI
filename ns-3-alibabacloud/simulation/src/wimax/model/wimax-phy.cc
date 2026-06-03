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

#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/node.h"
#include "wimax-net-device.h"
#include "wimax-phy.h"
#include "wimax-channel.h"
#include "ns3/packet-burst.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/pointer.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WimaxPhy");

NS_OBJECT_ENSURE_REGISTERED (WimaxPhy);

TypeId WimaxPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WimaxPhy")
    .SetParent<Object> ()
    .SetGroupName("Wimax")

    // No AddConstructor because this is an abstract class.

    .AddAttribute ("Channel",
                   "Wimax channel",
                   PointerValue (),
                   MakePointerAccessor (&WimaxPhy::GetChannel, &WimaxPhy::Attach),
                   MakePointerChecker<WimaxChannel> ())

    .AddAttribute ("FrameDuration",
                   "The frame duration in seconds.",
                   TimeValue (Seconds (0.01)),
                   MakeTimeAccessor (&WimaxPhy::SetFrameDuration, &WimaxPhy::GetFrameDurationSec),
                   MakeTimeChecker ())

    .AddAttribute ("Frequency",
                   "The central frequency in KHz.",
                   UintegerValue (5000000),
                   MakeUintegerAccessor (&WimaxPhy::SetFrequency, &WimaxPhy::GetFrequency),
                   MakeUintegerChecker<uint32_t> (1000000, 11000000))

    .AddAttribute ("Bandwidth",
                   "The channel bandwidth in Hz.",
                   UintegerValue (10000000),
                   MakeUintegerAccessor (&WimaxPhy::SetChannelBandwidth, &WimaxPhy::GetChannelBandwidth),
                   MakeUintegerChecker<uint32_t> (5000000, 30000000))

  ;
  return tid;
}

WimaxPhy::WimaxPhy (void)
  : m_state (PHY_STATE_IDLE),
    m_nrCarriers (0),
    m_frameDuration (Seconds (0.01)),
    m_frequency (5000000),
    m_channelBandwidth (10000000),
    m_psDuration (Seconds (0)),
    m_symbolDuration (Seconds (0)),
    m_psPerSymbol (0),
    m_psPerFrame (0),
    m_symbolsPerFrame (0)
{
  m_mobility = 0;
  m_duplex = 0;
  m_txFrequency = 0;
  m_rxFrequency = 0;

}

WimaxPhy::~WimaxPhy (void)
{
}

void
WimaxPhy::DoDispose (void)
{
  m_device = 0;
  m_channel = 0;
}

void
WimaxPhy::Attach (Ptr<WimaxChannel> channel)
{
  m_channel = channel;
  DoAttach (channel);
}

Ptr<WimaxChannel>
WimaxPhy::GetChannel (void) const
{
  return m_channel;
}

void
WimaxPhy::SetDevice (Ptr<WimaxNetDevice> device)
{
  m_device = device;
}

Ptr<NetDevice>
WimaxPhy::GetDevice (void) const
{
  return m_device;
}

void
WimaxPhy::StartScanning (uint64_t frequency, Time timeout, Callback<void, bool, uint64_t> callback)
{
  NS_ASSERT_MSG (m_state == PHY_STATE_IDLE || m_state == PHY_STATE_SCANNING,
                 "Error while scanning: The PHY state should be PHY_STATE_SCANNING or PHY_STATE_IDLE");

  m_state = PHY_STATE_SCANNING;
  m_scanningFrequency = frequency;
  m_dlChnlSrchTimeoutEvent = Simulator::Schedule (timeout, &WimaxPhy::EndScanning, this);
  m_scanningCallback = callback;
}

void
WimaxPhy::EndScanning (void)
{
  m_scanningCallback (false, m_scanningFrequency);
}

void
WimaxPhy::SetReceiveCallback (Callback<void, Ptr<const PacketBurst> > callback)
{
  m_rxCallback = callback;
}

Callback<void, Ptr<const PacketBurst> >
WimaxPhy::GetReceiveCallback (void) const
{
  return m_rxCallback;
}

void
WimaxPhy::SetDuplex (uint64_t rxFrequency, uint64_t txFrequency)
{
  m_txFrequency = txFrequency;
  m_rxFrequency = rxFrequency;
}

void
WimaxPhy::SetSimplex (uint64_t frequency)
{
  m_txFrequency = frequency;
  m_rxFrequency = frequency;
}

uint64_t
WimaxPhy::GetRxFrequency (void) const
{
  return m_rxFrequency;
}

uint64_t
WimaxPhy::GetTxFrequency (void) const
{
  return m_txFrequency;
}

uint64_t
WimaxPhy::GetScanningFrequency (void) const
{
  return m_scanningFrequency;
}

void
WimaxPhy::SetState (PhyState state)
{
  m_state = state;
}

WimaxPhy::PhyState WimaxPhy::GetState (void) const
{
  return m_state;
}

bool
WimaxPhy::IsDuplex (void) const
{
  return m_duplex;
}

EventId
WimaxPhy::GetChnlSrchTimeoutEvent (void) const
{
  return m_dlChnlSrchTimeoutEvent;
}

void
WimaxPhy::SetScanningCallback (void) const
{
  m_scanningCallback (true, GetScanningFrequency ());
}

void
WimaxPhy::SetDataRates (void)
{
  DoSetDataRates ();
}

uint32_t
WimaxPhy::GetDataRate (WimaxPhy::ModulationType modulationType) const
{
  return DoGetDataRate (modulationType);
}

Time
WimaxPhy::GetTransmissionTime (uint32_t size, WimaxPhy::ModulationType modulationType) const
{
  return DoGetTransmissionTime (size, modulationType);
}

uint64_t
WimaxPhy::GetNrSymbols (uint32_t size, WimaxPhy::ModulationType modulationType) const
{
  return DoGetNrSymbols (size, modulationType);
}

uint64_t
WimaxPhy::GetNrBytes (uint32_t symbols, WimaxPhy::ModulationType modulationType) const
{
  return DoGetNrBytes (symbols, modulationType);
}

uint16_t
WimaxPhy::GetTtg (void) const
{
  return DoGetTtg ();
}

uint16_t
WimaxPhy::GetRtg (void) const
{
  return DoGetRtg ();
}

uint8_t
WimaxPhy::GetFrameDurationCode (void) const
{
  return DoGetFrameDurationCode ();
}

Time
WimaxPhy::GetFrameDuration (uint8_t frameDurationCode) const
{
  return DoGetFrameDuration (frameDurationCode);
}

/*---------------------PHY parameters functions-----------------------*/

void
WimaxPhy::SetPhyParameters (void)
{
  DoSetPhyParameters ();
}

void
WimaxPhy::SetNrCarriers (uint8_t nrCarriers)
{
  m_nrCarriers = nrCarriers;
}

uint8_t
WimaxPhy::GetNrCarriers (void) const
{
  return m_nrCarriers;
}

void
WimaxPhy::SetFrameDuration (Time frameDuration)
{
  m_frameDuration = frameDuration;
}

Time
WimaxPhy::GetFrameDuration (void) const
{
  return GetFrameDurationSec ();
}

Time
WimaxPhy::GetFrameDurationSec (void) const
{
  return m_frameDuration;
}

void
WimaxPhy::SetFrequency (uint32_t frequency)
{
  m_frequency = frequency;
}

uint32_t
WimaxPhy::GetFrequency (void) const
{
  return m_frequency;
}

void
WimaxPhy::SetChannelBandwidth (uint32_t channelBandwidth)
{
  m_channelBandwidth = channelBandwidth;
}

uint32_t
WimaxPhy::GetChannelBandwidth (void) const
{
  return m_channelBandwidth;
}

uint16_t
WimaxPhy::GetNfft (void) const
{
  return DoGetNfft ();
}

double
WimaxPhy::GetSamplingFactor (void) const
{
  return DoGetSamplingFactor ();
}

double
WimaxPhy::GetSamplingFrequency (void) const
{
  return DoGetSamplingFrequency ();
}

void
WimaxPhy::SetPsDuration (Time psDuration)
{
  m_psDuration = psDuration;
}

Time
WimaxPhy::GetPsDuration (void) const
{
  return m_psDuration;
}

void
WimaxPhy::SetSymbolDuration (Time symbolDuration)
{
  m_symbolDuration = symbolDuration;
}

Time
WimaxPhy::GetSymbolDuration (void) const
{
  return m_symbolDuration;
}

double
WimaxPhy::GetGValue (void) const
{
  return DoGetGValue ();
}

void
WimaxPhy::SetPsPerSymbol (uint16_t psPerSymbol)
{
  m_psPerSymbol = psPerSymbol;
}

uint16_t
WimaxPhy::GetPsPerSymbol (void) const
{
  return m_psPerSymbol;
}

void
WimaxPhy::SetPsPerFrame (uint16_t psPerFrame)
{
  m_psPerFrame = psPerFrame;
}

uint16_t
WimaxPhy::GetPsPerFrame (void) const
{
  return m_psPerFrame;
}

void
WimaxPhy::SetSymbolsPerFrame (uint32_t symbolsPerFrame)
{
  m_symbolsPerFrame = symbolsPerFrame;
}

uint32_t
WimaxPhy::GetSymbolsPerFrame (void) const
{
  return m_symbolsPerFrame;
}
Ptr<Object>
WimaxPhy::GetMobility (void)
{
  return m_mobility;
}

void
WimaxPhy::SetMobility (Ptr<Object> mobility)
{
  m_mobility = mobility;

}

} // namespace ns3
