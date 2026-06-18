/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include <algorithm>
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/string.h"
#include "ns3/tuple.h"
#include "ns3/mobility-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/error-model.h"
#include "wifi-net-device.h"
#include "wifi-phy.h"
#include "wifi-utils.h"
#include "interference-helper.h"
#include "frame-capture-model.h"
#include "preamble-detection-model.h"
#include "wifi-radio-energy-model.h"
#include "error-rate-model.h"
#include "wifi-net-device.h"
#include "wifi-psdu.h"
#include "wifi-ppdu.h"
#include "ns3/dsss-phy.h"
#include "ns3/erp-ofdm-phy.h"
#include "ns3/he-phy.h" //includes OFDM, HT, and VHT

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiPhy");

/****************************************************************
 *       The actual WifiPhy class
 ****************************************************************/

NS_OBJECT_ENSURE_REGISTERED (WifiPhy);

TypeId
WifiPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiPhy")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddAttribute ("ChannelSettings",
                   "Tuple {channel number, channel width (MHz), PHY band, primary20 index} "
                   "describing the settings of the operating channel. The primary20 index is "
                   "the index of the primary 20 MHz channel within the operating channel "
                   "(0 indicates the 20 MHz subchannel with the lowest center frequency) and "
                   "is only valid if the width of the operating channel is a multiple of 20 MHz. "
                   "If the standard for this object has not been set yet, the value of this "
                   "attribute is saved and will be used to set the operating channel when the "
                   "standard is configured. If the PHY band is left unspecified, the default "
                   "band for the configured standard is used. If the channel width and the "
                   "channel number are both 0, the default channel width for the configured "
                   "standard and band are used. If the channel number is 0, the default "
                   "channel number for the configured standard, band and channel width is used."
                   "Note that the channel width can be left unspecified (0) if the channel "
                   "number uniquely identify a frequency channel for the given standard and band. ",
                   StringValue ("{0, 0, BAND_UNSPECIFIED, 0}"),
                   MakeTupleAccessor <UintegerValue, UintegerValue, EnumValue, UintegerValue> ((void (WifiPhy::*) (const ChannelTuple&))(&WifiPhy::SetOperatingChannel)),
                   MakeTupleChecker<UintegerValue, UintegerValue, EnumValue, UintegerValue>
                     (MakeUintegerChecker<uint8_t> (0, 233),
                      MakeUintegerChecker<uint16_t> (0, 160),
                      MakeEnumChecker (WifiPhyBand::WIFI_PHY_BAND_2_4GHZ, "BAND_2_4GHZ",
                                       WifiPhyBand::WIFI_PHY_BAND_5GHZ, "BAND_5GHZ",
                                       WifiPhyBand::WIFI_PHY_BAND_6GHZ, "BAND_6GHZ",
                                       WifiPhyBand::WIFI_PHY_BAND_UNSPECIFIED, "BAND_UNSPECIFIED"),
                      MakeUintegerChecker<uint8_t> (0, 7)))
    .AddAttribute ("Frequency",
                   "The center frequency (MHz) of the current operating channel.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiPhy::GetFrequency),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("ChannelNumber",
                   "The channel number of the current operating channel.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiPhy::GetChannelNumber),
                   MakeUintegerChecker<uint8_t> (0, 233))
    .AddAttribute ("ChannelWidth",
                   "The width in MHz of the current operating channel (5, 10, 20, 22, 40, 80 or 160).",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiPhy::GetChannelWidth),
                   MakeUintegerChecker<uint16_t> (5, 160))
    .AddAttribute ("Primary20MHzIndex",
                   "The index of the primary 20 MHz channel within the current operating channel "
                   "(0 indicates the 20 MHz subchannel with the lowest center frequency).",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiPhy::GetPrimary20Index),
                   MakeUintegerChecker<uint8_t> (0, 7))
    .AddAttribute ("RxSensitivity",
                   "The energy of a received signal should be higher than "
                   "this threshold (dBm) for the PHY to detect the signal. "
                   "This threshold refers to a width of 20 MHz and will be "
                   "scaled to match the width of the received signal.",
                   DoubleValue (-101.0),
                   MakeDoubleAccessor (&WifiPhy::SetRxSensitivity,
                                       &WifiPhy::GetRxSensitivity),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("CcaEdThreshold",
                   "The energy of a non Wi-Fi received signal should be higher than "
                   "this threshold (dBm) to allow the PHY layer to declare CCA BUSY state. "
                   "This check is performed on the 20 MHz primary channel only.",
                   DoubleValue (-62.0),
                   MakeDoubleAccessor (&WifiPhy::SetCcaEdThreshold,
                                       &WifiPhy::GetCcaEdThreshold),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxGain",
                   "Transmission gain (dB).",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&WifiPhy::SetTxGain,
                                       &WifiPhy::GetTxGain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("RxGain",
                   "Reception gain (dB).",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&WifiPhy::SetRxGain,
                                       &WifiPhy::GetRxGain),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxPowerLevels",
                   "Number of transmission power levels available between "
                   "TxPowerStart and TxPowerEnd included.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&WifiPhy::m_nTxPower),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("TxPowerEnd",
                   "Maximum available transmission level (dBm).",
                   DoubleValue (16.0206),
                   MakeDoubleAccessor (&WifiPhy::SetTxPowerEnd,
                                       &WifiPhy::GetTxPowerEnd),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxPowerStart",
                   "Minimum available transmission level (dBm).",
                   DoubleValue (16.0206),
                   MakeDoubleAccessor (&WifiPhy::SetTxPowerStart,
                                       &WifiPhy::GetTxPowerStart),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("RxNoiseFigure",
                   "Loss (dB) in the Signal-to-Noise-Ratio due to non-idealities in the receiver."
                   " According to Wikipedia (http://en.wikipedia.org/wiki/Noise_figure), this is "
                   "\"the difference in decibels (dB) between"
                   " the noise output of the actual receiver to the noise output of an "
                   " ideal receiver with the same overall gain and bandwidth when the receivers "
                   " are connected to sources at the standard noise temperature T0 (usually 290 K)\".",
                   DoubleValue (7),
                   MakeDoubleAccessor (&WifiPhy::SetRxNoiseFigure),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("State",
                   "The state of the PHY layer.",
                   PointerValue (),
                   MakePointerAccessor (&WifiPhy::m_state),
                   MakePointerChecker<WifiPhyStateHelper> ())
    .AddAttribute ("ChannelSwitchDelay",
                   "Delay between two short frames transmitted on different frequencies.",
                   TimeValue (MicroSeconds (250)),
                   MakeTimeAccessor (&WifiPhy::m_channelSwitchDelay),
                   MakeTimeChecker ())
    .AddAttribute ("Antennas",
                   "The number of antennas on the device.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&WifiPhy::GetNumberOfAntennas,
                                         &WifiPhy::SetNumberOfAntennas),
                   MakeUintegerChecker<uint8_t> (1, 8))
    .AddAttribute ("MaxSupportedTxSpatialStreams",
                   "The maximum number of supported TX spatial streams."
                   "This parameter is only valuable for 802.11n/ac/ax STAs and APs.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&WifiPhy::GetMaxSupportedTxSpatialStreams,
                                         &WifiPhy::SetMaxSupportedTxSpatialStreams),
                   MakeUintegerChecker<uint8_t> (1, 8))
    .AddAttribute ("MaxSupportedRxSpatialStreams",
                   "The maximum number of supported RX spatial streams."
                   "This parameter is only valuable for 802.11n/ac/ax STAs and APs.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&WifiPhy::GetMaxSupportedRxSpatialStreams,
                                         &WifiPhy::SetMaxSupportedRxSpatialStreams),
                   MakeUintegerChecker<uint8_t> (1, 8))
    .AddAttribute ("ShortPlcpPreambleSupported",
                   "Whether or not short PHY preamble is supported."
                   "This parameter is only valuable for 802.11b STAs and APs."
                   "Note: 802.11g APs and STAs always support short PHY preamble.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&WifiPhy::GetShortPhyPreambleSupported,
                                        &WifiPhy::SetShortPhyPreambleSupported),
                   MakeBooleanChecker ())
    .AddAttribute ("FrameCaptureModel",
                   "Ptr to an object that implements the frame capture model",
                   PointerValue (),
                   MakePointerAccessor (&WifiPhy::m_frameCaptureModel),
                   MakePointerChecker <FrameCaptureModel> ())
    .AddAttribute ("PreambleDetectionModel",
                   "Ptr to an object that implements the preamble detection model",
                   PointerValue (),
                   MakePointerAccessor (&WifiPhy::m_preambleDetectionModel),
                   MakePointerChecker <PreambleDetectionModel> ())
    .AddAttribute ("PostReceptionErrorModel",
                   "An optional packet error model can be added to the receive "
                   "packet process after any propagation-based (SNR-based) error "
                   "models have been applied. Typically this is used to force "
                   "specific packet drops, for testing purposes.",
                   PointerValue (),
                   MakePointerAccessor (&WifiPhy::m_postReceptionErrorModel),
                   MakePointerChecker<ErrorModel> ())
    .AddAttribute ("Sifs",
                   "The duration of the Short Interframe Space. "
                   "NOTE that the default value is overwritten by the value defined "
                   "by the standard; if you want to set this attribute, you have to "
                   "do it after that the PHY object is initialized.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&WifiPhy::m_sifs),
                   MakeTimeChecker ())
    .AddAttribute ("Slot",
                   "The duration of a slot. "
                   "NOTE that the default value is overwritten by the value defined "
                   "by the standard; if you want to set this attribute, you have to "
                   "do it after that the PHY object is initialized.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&WifiPhy::m_slot),
                   MakeTimeChecker ())
    .AddAttribute ("Pifs",
                   "The duration of the PCF Interframe Space. "
                   "NOTE that the default value is overwritten by the value defined "
                   "by the standard; if you want to set this attribute, you have to "
                   "do it after that the PHY object is initialized.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&WifiPhy::m_pifs),
                   MakeTimeChecker ())
    .AddAttribute ("PowerDensityLimit",
                   "The mean equivalent isotropically radiated power density"
                   "limit (in dBm/MHz) set by regulators.",
                   DoubleValue (100.0), //set to a high value so as to have no effect
                   MakeDoubleAccessor (&WifiPhy::m_powerDensityLimit),
                   MakeDoubleChecker<double> ())
    .AddTraceSource ("PhyTxBegin",
                     "Trace source indicating a packet "
                     "has begun transmitting over the channel medium",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyTxBeginTrace),
                     "ns3::WifiPhy::PhyTxBeginTracedCallback")
    .AddTraceSource ("PhyTxPsduBegin",
                     "Trace source indicating a PSDU "
                     "has begun transmitting over the channel medium",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyTxPsduBeginTrace),
                     "ns3::WifiPhy::PsduTxBeginCallback")
    .AddTraceSource ("PhyTxEnd",
                     "Trace source indicating a packet "
                     "has been completely transmitted over the channel.",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyTxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxDrop",
                     "Trace source indicating a packet "
                     "has been dropped by the device during transmission",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxBegin",
                     "Trace source indicating a packet "
                     "has begun being received from the channel medium "
                     "by the device",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyRxBeginTrace),
                     "ns3::WifiPhy::PhyRxBeginTracedCallback")
    .AddTraceSource ("PhyRxPayloadBegin",
                     "Trace source indicating the reception of the "
                     "payload of a PPDU has begun",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyRxPayloadBeginTrace),
                     "ns3::WifiPhy::PhyRxPayloadBeginTracedCallback")
    .AddTraceSource ("PhyRxEnd",
                     "Trace source indicating a packet "
                     "has been completely received from the channel medium "
                     "by the device",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyRxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxDrop",
                     "Trace source indicating a packet "
                     "has been dropped by the device during reception",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyRxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MonitorSnifferRx",
                     "Trace source simulating a wifi device in monitor mode "
                     "sniffing all received frames",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyMonitorSniffRxTrace),
                     "ns3::WifiPhy::MonitorSnifferRxTracedCallback")
    .AddTraceSource ("MonitorSnifferTx",
                     "Trace source simulating the capability of a wifi device "
                     "in monitor mode to sniff all frames being transmitted",
                     MakeTraceSourceAccessor (&WifiPhy::m_phyMonitorSniffTxTrace),
                     "ns3::WifiPhy::MonitorSnifferTxTracedCallback")
  ;
  return tid;
}

WifiPhy::WifiPhy ()
  : m_txMpduReferenceNumber (0xffffffff),
    m_rxMpduReferenceNumber (0xffffffff),
    m_endPhyRxEvent (),
    m_endTxEvent (),
    m_currentEvent (0),
    m_previouslyRxPpduUid (UINT64_MAX),
    m_standard (WIFI_STANDARD_UNSPECIFIED),
    m_band (WIFI_PHY_BAND_UNSPECIFIED),
    m_sifs (Seconds (0)),
    m_slot (Seconds (0)),
    m_pifs (Seconds (0)),
    m_ackTxTime (Seconds (0)),
    m_blockAckTxTime (Seconds (0)),
    m_powerRestricted (false),
    m_channelAccessRequested (false),
    m_txSpatialStreams (0),
    m_rxSpatialStreams (0),
    m_wifiRadioEnergyModel (0),
    m_timeLastPreambleDetected (Seconds (0))
{
  NS_LOG_FUNCTION (this);
  m_random = CreateObject<UniformRandomVariable> ();
  m_state = CreateObject<WifiPhyStateHelper> ();
}

WifiPhy::~WifiPhy ()
{
  NS_LOG_FUNCTION (this);
}

void
WifiPhy::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_endTxEvent.Cancel ();
  m_endPhyRxEvent.Cancel ();
  for (auto & phyEntity : m_phyEntities)
    {
      phyEntity.second->CancelAllEvents ();
    }
  m_device = 0;
  m_mobility = 0;
  m_frameCaptureModel = 0;
  m_preambleDetectionModel = 0;
  m_wifiRadioEnergyModel = 0;
  m_postReceptionErrorModel = 0;
  m_supportedChannelWidthSet.clear ();
  if (m_interference != nullptr)
    {
      m_interference->Dispose ();
    }
  m_interference = 0;
  m_random = 0;
  m_state = 0;
  m_currentEvent = 0;
  for (auto & preambleEvent : m_currentPreambleEvents)
    {
      preambleEvent.second = 0;
    }
  m_currentPreambleEvents.clear ();

  for (auto & phyEntity : m_phyEntities)
    {
      phyEntity.second = 0;
    }
  m_phyEntities.clear ();
}

std::map<WifiModulationClass, Ptr<PhyEntity> > &
WifiPhy::GetStaticPhyEntities (void)
{
    static std::map<WifiModulationClass, Ptr<PhyEntity> > g_staticPhyEntities;
    return g_staticPhyEntities;
}

Ptr<WifiPhyStateHelper>
WifiPhy::GetState (void) const
{
  return m_state;
}

void
WifiPhy::SetReceiveOkCallback (RxOkCallback callback)
{
  m_state->SetReceiveOkCallback (callback);
}

void
WifiPhy::SetReceiveErrorCallback (RxErrorCallback callback)
{
  m_state->SetReceiveErrorCallback (callback);
}

void
WifiPhy::RegisterListener (WifiPhyListener *listener)
{
  m_state->RegisterListener (listener);
}

void
WifiPhy::UnregisterListener (WifiPhyListener *listener)
{
  m_state->UnregisterListener (listener);
}

void
WifiPhy::SetCapabilitiesChangedCallback (Callback<void> callback)
{
  m_capabilitiesChangedCallback = callback;
}

void
WifiPhy::SetRxSensitivity (double threshold)
{
  NS_LOG_FUNCTION (this << threshold);
  m_rxSensitivityW = DbmToW (threshold);
}

double
WifiPhy::GetRxSensitivity (void) const
{
  return WToDbm (m_rxSensitivityW);
}

void
WifiPhy::SetCcaEdThreshold (double threshold)
{
  NS_LOG_FUNCTION (this << threshold);
  m_ccaEdThresholdW = DbmToW (threshold);
}

double
WifiPhy::GetCcaEdThreshold (void) const
{
  return WToDbm (m_ccaEdThresholdW);
}

void
WifiPhy::SetRxNoiseFigure (double noiseFigureDb)
{
  NS_LOG_FUNCTION (this << noiseFigureDb);
  if (m_interference)
    {
      m_interference->SetNoiseFigure (DbToRatio (noiseFigureDb));
    }
  m_noiseFigureDb = noiseFigureDb;
}

void
WifiPhy::SetTxPowerStart (double start)
{
  NS_LOG_FUNCTION (this << start);
  m_txPowerBaseDbm = start;
}

double
WifiPhy::GetTxPowerStart (void) const
{
  return m_txPowerBaseDbm;
}

void
WifiPhy::SetTxPowerEnd (double end)
{
  NS_LOG_FUNCTION (this << end);
  m_txPowerEndDbm = end;
}

double
WifiPhy::GetTxPowerEnd (void) const
{
  return m_txPowerEndDbm;
}

void
WifiPhy::SetNTxPower (uint8_t n)
{
  NS_LOG_FUNCTION (this << +n);
  m_nTxPower = n;
}

uint8_t
WifiPhy::GetNTxPower (void) const
{
  return m_nTxPower;
}

void
WifiPhy::SetTxGain (double gain)
{
  NS_LOG_FUNCTION (this << gain);
  m_txGainDb = gain;
}

double
WifiPhy::GetTxGain (void) const
{
  return m_txGainDb;
}

void
WifiPhy::SetRxGain (double gain)
{
  NS_LOG_FUNCTION (this << gain);
  m_rxGainDb = gain;
}

double
WifiPhy::GetRxGain (void) const
{
  return m_rxGainDb;
}

void
WifiPhy::SetShortPhyPreambleSupported (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  m_shortPreamble = enable;
}

bool
WifiPhy::GetShortPhyPreambleSupported (void) const
{
  return m_shortPreamble;
}

void
WifiPhy::SetDevice (const Ptr<WifiNetDevice> device)
{
  m_device = device;
}

Ptr<WifiNetDevice>
WifiPhy::GetDevice (void) const
{
  return m_device;
}

void
WifiPhy::SetMobility (const Ptr<MobilityModel> mobility)
{
  m_mobility = mobility;
}

Ptr<MobilityModel>
WifiPhy::GetMobility (void) const
{
  if (m_mobility != 0)
    {
      return m_mobility;
    }
  else
    {
      return m_device->GetNode ()->GetObject<MobilityModel> ();
    }
}

void
WifiPhy::SetInterferenceHelper (const Ptr<InterferenceHelper> helper)
{
  m_interference = helper;
  m_interference->SetNoiseFigure (DbToRatio (m_noiseFigureDb));
  m_interference->SetNumberOfReceiveAntennas (m_numberOfAntennas);
}

void
WifiPhy::SetErrorRateModel (const Ptr<ErrorRateModel> model)
{
  NS_ASSERT (m_interference != nullptr);
  m_interference->SetErrorRateModel (model);
}

void
WifiPhy::SetPostReceptionErrorModel (const Ptr<ErrorModel> em)
{
  NS_LOG_FUNCTION (this << em);
  m_postReceptionErrorModel = em;
}

void
WifiPhy::SetFrameCaptureModel (const Ptr<FrameCaptureModel> model)
{
  m_frameCaptureModel = model;
}

void
WifiPhy::SetPreambleDetectionModel (const Ptr<PreambleDetectionModel> model)
{
  m_preambleDetectionModel = model;
}

void
WifiPhy::SetWifiRadioEnergyModel (const Ptr<WifiRadioEnergyModel> wifiRadioEnergyModel)
{
  m_wifiRadioEnergyModel = wifiRadioEnergyModel;
}

double
WifiPhy::GetPowerDbm (uint8_t power) const
{
  NS_ASSERT (m_txPowerBaseDbm <= m_txPowerEndDbm);
  NS_ASSERT (m_nTxPower > 0);
  double dbm;
  if (m_nTxPower > 1)
    {
      dbm = m_txPowerBaseDbm + power * (m_txPowerEndDbm - m_txPowerBaseDbm) / (m_nTxPower - 1);
    }
  else
    {
      NS_ASSERT_MSG (m_txPowerBaseDbm == m_txPowerEndDbm, "cannot have TxPowerEnd != TxPowerStart with TxPowerLevels == 1");
      dbm = m_txPowerBaseDbm;
    }
  return dbm;
}

Time
WifiPhy::GetChannelSwitchDelay (void) const
{
  return m_channelSwitchDelay;
}

double
WifiPhy::CalculateSnr (const WifiTxVector& txVector, double ber) const
{
  return m_interference->GetErrorRateModel ()->CalculateSnr (txVector, ber);
}

const Ptr<const PhyEntity>
WifiPhy::GetStaticPhyEntity (WifiModulationClass modulation)
{
  const auto it = GetStaticPhyEntities ().find (modulation);
  NS_ABORT_MSG_IF (it == GetStaticPhyEntities ().end (), "Unimplemented Wi-Fi modulation class");
  return it->second;
}

Ptr<PhyEntity>
WifiPhy::GetPhyEntity (WifiModulationClass modulation) const
{
  const auto it = m_phyEntities.find (modulation);
  NS_ABORT_MSG_IF (it == m_phyEntities.end (), "Unsupported Wi-Fi modulation class " << modulation);
  return it->second;
}

void
WifiPhy::AddStaticPhyEntity (WifiModulationClass modulation, Ptr<PhyEntity> phyEntity)
{
  NS_LOG_FUNCTION (modulation);
  NS_ASSERT_MSG (GetStaticPhyEntities ().find (modulation) == GetStaticPhyEntities ().end (), "The PHY entity has already been added. The setting should only be done once per modulation class");
    GetStaticPhyEntities ()[modulation] = phyEntity;
}

void
WifiPhy::AddPhyEntity (WifiModulationClass modulation, Ptr<PhyEntity> phyEntity)
{
  NS_LOG_FUNCTION (this << modulation);
  NS_ABORT_MSG_IF (GetStaticPhyEntities ().find (modulation) == GetStaticPhyEntities ().end (), "Cannot add an unimplemented PHY to supported list. Update the former first.");
  NS_ASSERT_MSG (m_phyEntities.find (modulation) == m_phyEntities.end (), "The PHY entity has already been added. The setting should only be done once per modulation class");
  phyEntity->SetOwner (this);
  m_phyEntities[modulation] = phyEntity;
}

void
WifiPhy::SetSifs (Time sifs)
{
  m_sifs = sifs;
}

Time
WifiPhy::GetSifs (void) const
{
  return m_sifs;
}

void
WifiPhy::SetSlot (Time slot)
{
  m_slot = slot;
}

Time
WifiPhy::GetSlot (void) const
{
  return m_slot;
}

void
WifiPhy::SetPifs (Time pifs)
{
  m_pifs = pifs;
}

Time
WifiPhy::GetPifs (void) const
{
  return m_pifs;
}

Time
WifiPhy::GetAckTxTime (void) const
{
  return m_ackTxTime;
}

Time
WifiPhy::GetBlockAckTxTime (void) const
{
  return m_blockAckTxTime;
}

void
WifiPhy::Configure80211a (void)
{
  NS_LOG_FUNCTION (this);
  AddPhyEntity (WIFI_MOD_CLASS_OFDM, Create<OfdmPhy> ());

  // See Table 17-21 "OFDM PHY characteristics" of 802.11-2016
  SetSifs (MicroSeconds (16));
  SetSlot (MicroSeconds (9));
  SetPifs (GetSifs () + GetSlot ());
  // See Table 10-5 "Determination of the EstimatedAckTxTime based on properties
  // of the PPDU causing the EIFS" of 802.11-2016
  m_ackTxTime = MicroSeconds (44);
}

void
WifiPhy::Configure80211b (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<DsssPhy> phyEntity = Create<DsssPhy> ();
  AddPhyEntity (WIFI_MOD_CLASS_HR_DSSS, phyEntity);
  AddPhyEntity (WIFI_MOD_CLASS_DSSS, phyEntity); //when plain DSSS modes are used

  // See Table 16-4 "HR/DSSS PHY characteristics" of 802.11-2016
  SetSifs (MicroSeconds (10));
  SetSlot (MicroSeconds (20));
  SetPifs (GetSifs () + GetSlot ());
  // See Table 10-5 "Determination of the EstimatedAckTxTime based on properties
  // of the PPDU causing the EIFS" of 802.11-2016
  m_ackTxTime = MicroSeconds (304);
}

void
WifiPhy::Configure80211g (void)
{
  NS_LOG_FUNCTION (this);
  // See Table 18-5 "ERP characteristics" of 802.11-2016
  // Slot time defaults to the "long slot time" of 20 us in the standard
  // according to mixed 802.11b/g deployments.  Short slot time is enabled
  // if the user sets the ShortSlotTimeSupported flag to true and when the BSS
  // consists of only ERP STAs capable of supporting this option.
  Configure80211b ();
  AddPhyEntity (WIFI_MOD_CLASS_ERP_OFDM, Create<ErpOfdmPhy> ());
}

void
WifiPhy::Configure80211p (void)
{
  NS_LOG_FUNCTION (this);
  if (GetChannelWidth () == 10)
    {
      AddPhyEntity (WIFI_MOD_CLASS_OFDM, Create<OfdmPhy> (OFDM_PHY_10_MHZ));

      // See Table 17-21 "OFDM PHY characteristics" of 802.11-2016
      SetSifs (MicroSeconds (32));
      SetSlot (MicroSeconds (13));
      SetPifs (GetSifs () + GetSlot ());
      m_ackTxTime = MicroSeconds (88);
    }
  else if (GetChannelWidth () == 5)
    {
      AddPhyEntity (WIFI_MOD_CLASS_OFDM, Create<OfdmPhy> (OFDM_PHY_5_MHZ));

      // See Table 17-21 "OFDM PHY characteristics" of 802.11-2016
      SetSifs (MicroSeconds (64));
      SetSlot (MicroSeconds (21));
      SetPifs (GetSifs () + GetSlot ());
      m_ackTxTime = MicroSeconds (176);
    }
  else
    {
      NS_FATAL_ERROR ("802.11p configured with a wrong channel width!");
    }
}

void
WifiPhy::Configure80211n (void)
{
  NS_LOG_FUNCTION (this);
  if (m_band == WIFI_PHY_BAND_2_4GHZ)
    {
      Configure80211g ();
    }
  else
    {
      Configure80211a ();
    }
  AddPhyEntity (WIFI_MOD_CLASS_HT, Create<HtPhy> (m_txSpatialStreams));

  // See Table 10-5 "Determination of the EstimatedAckTxTime based on properties
  // of the PPDU causing the EIFS" of 802.11-2016
  m_blockAckTxTime = MicroSeconds (68);
}

void
WifiPhy::Configure80211ac (void)
{
  NS_LOG_FUNCTION (this);
  Configure80211n ();
  AddPhyEntity (WIFI_MOD_CLASS_VHT, Create<VhtPhy> ());
}

void
WifiPhy::Configure80211ax (void)
{
  NS_LOG_FUNCTION (this);
  if (m_band == WIFI_PHY_BAND_2_4GHZ)
    {
      Configure80211n ();
    }
  else
    {
      Configure80211ac ();
    }
  AddPhyEntity (WIFI_MOD_CLASS_HE, Create<HePhy> ());
}

void
WifiPhy::ConfigureStandard (WifiStandard standard)
{
  NS_LOG_FUNCTION (this << standard);

  NS_ABORT_MSG_IF (m_standard != WIFI_STANDARD_UNSPECIFIED && standard != m_standard,
                   "Cannot change standard");

  m_standard = standard;

  if (!m_operatingChannel.IsSet ())
    {
      NS_LOG_DEBUG ("Setting the operating channel first");
      SetOperatingChannel (m_channelSettings);
      // return because we are called back by SetOperatingChannel
      return;
    }

  // this function is called when changing PHY band, hence we have to delete
  // the previous PHY entities
  m_phyEntities.clear ();

  switch (standard)
    {
    case WIFI_STANDARD_80211a:
      Configure80211a ();
      break;
    case WIFI_STANDARD_80211b:
      Configure80211b ();
      break;
    case WIFI_STANDARD_80211g:
      Configure80211g ();
      break;
    case WIFI_STANDARD_80211p:
      Configure80211p ();
      break;
    case WIFI_STANDARD_80211n:
      Configure80211n ();
      break;
    case WIFI_STANDARD_80211ac:
      Configure80211ac ();
      break;
    case WIFI_STANDARD_80211ax:
      Configure80211ax ();
      break;
    case WIFI_STANDARD_UNSPECIFIED:
    default:
      NS_ASSERT_MSG (false, "Unsupported standard");
      break;
    }
}

WifiPhyBand
WifiPhy::GetPhyBand (void) const
{
  return m_band;
}


WifiStandard
WifiPhy::GetStandard (void) const
{
  return m_standard;
}

const WifiPhyOperatingChannel&
WifiPhy::GetOperatingChannel (void) const
{
  return m_operatingChannel;
}

uint16_t
WifiPhy::GetFrequency (void) const
{
  return m_operatingChannel.GetFrequency ();
}

uint8_t
WifiPhy::GetChannelNumber (void) const
{
  return m_operatingChannel.GetNumber ();
}

uint16_t
WifiPhy::GetChannelWidth (void) const
{
  return m_operatingChannel.GetWidth ();
}

uint8_t
WifiPhy::GetPrimary20Index (void) const
{
  return m_operatingChannel.GetPrimaryChannelIndex (20);
}

void
WifiPhy::SetOperatingChannel (const ChannelTuple& channelTuple)
{
  // the generic operator<< for tuples does not give a pretty result
  NS_LOG_FUNCTION (this << +std::get<0> (channelTuple) << std::get<1> (channelTuple)
                   << static_cast<WifiPhyBand> (std::get<2> (channelTuple))
                   << +std::get<3> (channelTuple));

  m_channelSettings = channelTuple;

  if (m_standard == WIFI_STANDARD_UNSPECIFIED)
    {
      NS_LOG_DEBUG ("Channel information will be applied when a standard is configured");
      return;
    }

  Time delay = Seconds (0);

  if (IsInitialized ())
    {
      delay = GetDelayUntilChannelSwitch ();
    }

  if (delay.IsStrictlyNegative ())
    {
      // switching channel is not possible now
      return;
    }
  if (delay.IsStrictlyPositive ())
    {
      // switching channel has been postponed
      void (WifiPhy::*fp) (const ChannelTuple&) = &WifiPhy::SetOperatingChannel;
      Simulator::Schedule (delay, fp, this, channelTuple);
      return;
    }

  // channel can be switched now.
  DoChannelSwitch ();
}

Time
WifiPhy::GetDelayUntilChannelSwitch (void)
{
  m_powerRestricted = false;
  m_channelAccessRequested = false;
  m_currentEvent = 0;
  m_currentPreambleEvents.clear ();
  if (!IsInitialized ())
    {
      //this is not channel switch, this is initialization
      NS_LOG_DEBUG ("Before initialization, nothing to do");
      return Seconds (0);
    }

  Time delay = Seconds (0);

  NS_ASSERT (!IsStateSwitching ());
  switch (m_state->GetState ())
    {
    case WifiPhyState::RX:
      NS_LOG_DEBUG ("drop packet because of channel switching while reception");
      m_endPhyRxEvent.Cancel ();
      for (auto & phyEntity : m_phyEntities)
        {
          phyEntity.second->CancelAllEvents ();
        }
      break;
    case WifiPhyState::TX:
      NS_LOG_DEBUG ("channel switching postponed until end of current transmission");
      delay = GetDelayUntilIdle ();
      break;
    case WifiPhyState::CCA_BUSY:
    case WifiPhyState::IDLE:
      m_endPhyRxEvent.Cancel ();
      for (auto & phyEntity : m_phyEntities)
        {
          phyEntity.second->CancelAllEvents ();
        }
      break;
    case WifiPhyState::SLEEP:
      NS_LOG_DEBUG ("channel switching ignored in sleep mode");
      delay = Seconds (-1);  // negative value to indicate switching not possible
      break;
    default:
      NS_ASSERT (false);
      break;
    }

  return delay;
}

void
WifiPhy::DoChannelSwitch (void)
{
  NS_LOG_FUNCTION (this);

  // Update unspecified parameters with default values
  if (auto& [number, width, band, primary20] = m_channelSettings; true)
    {
      if (band == static_cast<int> (WIFI_PHY_BAND_UNSPECIFIED))
        {
          band = static_cast<int> (GetDefaultPhyBand (m_standard));
        }
      if (width == 0 && number == 0)
        {
          width = GetDefaultChannelWidth (m_standard, static_cast<WifiPhyBand> (band));
        }
      if (number == 0)
        {
          number = WifiPhyOperatingChannel::GetDefaultChannelNumber (width, m_standard, static_cast<WifiPhyBand> (band));
        }
    }

  // We need to call SetStandard if this is the first time we set a channel or we
  // are changing PHY band. Checking if the new PHY band is different than the
  // previous one covers both cases because initially the PHY band is unspecified
  bool changingPhyBand = (static_cast<WifiPhyBand> (std::get<2> (m_channelSettings)) != m_band);

  m_band = static_cast<WifiPhyBand> (std::get<2> (m_channelSettings));

  NS_LOG_DEBUG ("switching channel");
  m_operatingChannel.Set (std::get<0> (m_channelSettings), 0, std::get<1> (m_channelSettings),
                          m_standard, m_band);
  m_operatingChannel.SetPrimary20Index (std::get<3> (m_channelSettings));

  if (changingPhyBand)
    {
      ConfigureStandard (m_standard);
    }

  AddSupportedChannelWidth (GetChannelWidth ());

  if (IsInitialized ())
    {
      // notify channel switching
      m_state->SwitchToChannelSwitching (GetChannelSwitchDelay ());
      m_interference->EraseEvents ();
      /*
      * Needed here to be able to correctly sensed the medium for the first
      * time after the switching. The actual switching is not performed until
      * after m_channelSwitchDelay. Packets received during the switching
      * state are added to the event list and are employed later to figure
      * out the state of the medium after the switching.
      */
    }
}

void
WifiPhy::SetNumberOfAntennas (uint8_t antennas)
{
  NS_LOG_FUNCTION (this << +antennas);
  NS_ASSERT_MSG (antennas > 0 && antennas <= 4, "unsupported number of antennas");
  m_numberOfAntennas = antennas;
  if (m_interference)
    {
      m_interference->SetNumberOfReceiveAntennas (antennas);
    }
}

uint8_t
WifiPhy::GetNumberOfAntennas (void) const
{
  return m_numberOfAntennas;
}

void
WifiPhy::SetMaxSupportedTxSpatialStreams (uint8_t streams)
{
  NS_ASSERT (streams <= GetNumberOfAntennas ());
  bool changed = (m_txSpatialStreams != streams);
  m_txSpatialStreams = streams;
  if (changed)
    {
      auto phyEntity = m_phyEntities.find (WIFI_MOD_CLASS_HT);
      if (phyEntity != m_phyEntities.end ())
        {
          Ptr<HtPhy> htPhy = DynamicCast<HtPhy> (phyEntity->second);
          if (htPhy)
            {
              htPhy->SetMaxSupportedNss (m_txSpatialStreams); //this is essential to have the right MCSs configured
            }

          if (!m_capabilitiesChangedCallback.IsNull ())
            {
              m_capabilitiesChangedCallback ();
            }
        }
    }
}

uint8_t
WifiPhy::GetMaxSupportedTxSpatialStreams (void) const
{
  return m_txSpatialStreams;
}

void
WifiPhy::SetMaxSupportedRxSpatialStreams (uint8_t streams)
{
  NS_ASSERT (streams <= GetNumberOfAntennas ());
  bool changed = (m_rxSpatialStreams != streams);
  m_rxSpatialStreams = streams;
  if (changed && !m_capabilitiesChangedCallback.IsNull ())
    {
      m_capabilitiesChangedCallback ();
    }
}

uint8_t
WifiPhy::GetMaxSupportedRxSpatialStreams (void) const
{
  return m_rxSpatialStreams;
}

std::list<uint8_t>
WifiPhy::GetBssMembershipSelectorList (void) const
{
  std::list<uint8_t> list;
  for (const auto & phyEntity : m_phyEntities)
    {
      Ptr<HtPhy> htPhy = DynamicCast<HtPhy> (phyEntity.second);
      if (htPhy)
        {
          list.emplace_back (htPhy->GetBssMembershipSelector ());
        }
    }
  return list;
}

void
WifiPhy::AddSupportedChannelWidth (uint16_t width)
{
  NS_LOG_FUNCTION (this << width);
  for (std::vector<uint32_t>::size_type i = 0; i != m_supportedChannelWidthSet.size (); i++)
    {
      if (m_supportedChannelWidthSet[i] == width)
        {
          return;
        }
    }
  NS_LOG_FUNCTION ("Adding " << width << " to supported channel width set");
  m_supportedChannelWidthSet.push_back (width);
}

std::vector<uint16_t>
WifiPhy::GetSupportedChannelWidthSet (void) const
{
  return m_supportedChannelWidthSet;
}

void
WifiPhy::SetSleepMode (void)
{
  NS_LOG_FUNCTION (this);
  m_powerRestricted = false;
  m_channelAccessRequested = false;
  switch (m_state->GetState ())
    {
    case WifiPhyState::TX:
      NS_LOG_DEBUG ("setting sleep mode postponed until end of current transmission");
      Simulator::Schedule (GetDelayUntilIdle (), &WifiPhy::SetSleepMode, this);
      break;
    case WifiPhyState::RX:
      NS_LOG_DEBUG ("setting sleep mode postponed until end of current reception");
      Simulator::Schedule (GetDelayUntilIdle (), &WifiPhy::SetSleepMode, this);
      break;
    case WifiPhyState::SWITCHING:
      NS_LOG_DEBUG ("setting sleep mode postponed until end of channel switching");
      Simulator::Schedule (GetDelayUntilIdle (), &WifiPhy::SetSleepMode, this);
      break;
    case WifiPhyState::CCA_BUSY:
    case WifiPhyState::IDLE:
      NS_LOG_DEBUG ("setting sleep mode");
      m_state->SwitchToSleep ();
      break;
    case WifiPhyState::SLEEP:
      NS_LOG_DEBUG ("already in sleep mode");
      break;
    default:
      NS_ASSERT (false);
      break;
    }
}

void
WifiPhy::SetOffMode (void)
{
  NS_LOG_FUNCTION (this);
  m_powerRestricted = false;
  m_channelAccessRequested = false;
  m_endPhyRxEvent.Cancel ();
  m_endTxEvent.Cancel ();
  for (auto & phyEntity : m_phyEntities)
    {
      phyEntity.second->CancelAllEvents ();
    }
  m_state->SwitchToOff ();
}

void
WifiPhy::ResumeFromSleep (void)
{
  NS_LOG_FUNCTION (this);
  m_currentPreambleEvents.clear ();
  switch (m_state->GetState ())
    {
    case WifiPhyState::TX:
    case WifiPhyState::RX:
    case WifiPhyState::IDLE:
    case WifiPhyState::CCA_BUSY:
    case WifiPhyState::SWITCHING:
      {
        NS_LOG_DEBUG ("not in sleep mode, there is nothing to resume");
        break;
      }
    case WifiPhyState::SLEEP:
      {
        NS_LOG_DEBUG ("resuming from sleep mode");
        m_state->SwitchFromSleep ();
        SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (nullptr));
        break;
      }
    default:
      {
        NS_ASSERT (false);
        break;
      }
    }
}

void
WifiPhy::ResumeFromOff (void)
{
  NS_LOG_FUNCTION (this);
  switch (m_state->GetState ())
    {
    case WifiPhyState::TX:
    case WifiPhyState::RX:
    case WifiPhyState::IDLE:
    case WifiPhyState::CCA_BUSY:
    case WifiPhyState::SWITCHING:
    case WifiPhyState::SLEEP:
      {
        NS_LOG_DEBUG ("not in off mode, there is nothing to resume");
        break;
      }
    case WifiPhyState::OFF:
      {
        NS_LOG_DEBUG ("resuming from off mode");
        m_state->SwitchFromOff ();
        SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (nullptr));
        break;
      }
    default:
      {
        NS_ASSERT (false);
        break;
      }
    }
}

Time
WifiPhy::GetPreambleDetectionDuration (void)
{
  return MicroSeconds (4);
}

Time
WifiPhy::GetStartOfPacketDuration (const WifiTxVector& txVector)
{
  return MicroSeconds (4);
}

Time
WifiPhy::GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype, uint16_t staId)
{
  uint32_t totalAmpduSize;
  double totalAmpduNumSymbols;
  return GetPayloadDuration (size, txVector, band, mpdutype, false, totalAmpduSize, totalAmpduNumSymbols, staId);
}

Time
WifiPhy::GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype,
                             bool incFlag, uint32_t &totalAmpduSize, double &totalAmpduNumSymbols,
                             uint16_t staId)
{
  return GetStaticPhyEntity (txVector.GetModulationClass ())->GetPayloadDuration (size, txVector, band, mpdutype,
                                                                                  incFlag, totalAmpduSize, totalAmpduNumSymbols,
                                                                                  staId);
}

Time
WifiPhy::CalculatePhyPreambleAndHeaderDuration (const WifiTxVector& txVector)
{
  return GetStaticPhyEntity (txVector.GetModulationClass ())->CalculatePhyPreambleAndHeaderDuration (txVector);
}

Time
WifiPhy::CalculateTxDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, uint16_t staId)
{
  Time duration = CalculatePhyPreambleAndHeaderDuration (txVector)
    + GetPayloadDuration (size, txVector, band, NORMAL_MPDU, staId);
  NS_ASSERT (duration.IsStrictlyPositive ());
  return duration;
}

Time
WifiPhy::CalculateTxDuration (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, WifiPhyBand band)
{
  return CalculateTxDuration (GetWifiConstPsduMap (psdu, txVector), txVector, band);
}

Time
WifiPhy::CalculateTxDuration (WifiConstPsduMap psduMap, const WifiTxVector& txVector, WifiPhyBand band)
{
  return GetStaticPhyEntity (txVector.GetModulationClass ())->CalculateTxDuration (psduMap, txVector, band);
}

uint32_t
WifiPhy::GetMaxPsduSize (WifiModulationClass modulation)
{
  return GetStaticPhyEntity (modulation)->GetMaxPsduSize ();
}

void
WifiPhy::NotifyTxBegin (WifiConstPsduMap psdus, double txPowerW)
{
  if (!m_phyTxBeginTrace.IsEmpty ())
    {
      for (auto const& psdu : psdus)
        {
          for (auto& mpdu : *PeekPointer (psdu.second))
            {
              m_phyTxBeginTrace (mpdu->GetProtocolDataUnit (), txPowerW);
            }
        }
    }
}

void
WifiPhy::NotifyTxEnd (WifiConstPsduMap psdus)
{
  if (!m_phyTxEndTrace.IsEmpty ())
    {
      for (auto const& psdu : psdus)
        {
          for (auto& mpdu : *PeekPointer (psdu.second))
            {
              m_phyTxEndTrace (mpdu->GetProtocolDataUnit ());
            }
        }
    }
}

void
WifiPhy::NotifyTxDrop (Ptr<const WifiPsdu> psdu)
{
  if (!m_phyTxDropTrace.IsEmpty ())
    {
      for (auto& mpdu : *PeekPointer (psdu))
        {
          m_phyTxDropTrace (mpdu->GetProtocolDataUnit ());
        }
    }
}

void
WifiPhy::NotifyRxBegin (Ptr<const WifiPsdu> psdu, const RxPowerWattPerChannelBand& rxPowersW)
{
  if (psdu && !m_phyRxBeginTrace.IsEmpty ())
    {
      for (auto& mpdu : *PeekPointer (psdu))
        {
          m_phyRxBeginTrace (mpdu->GetProtocolDataUnit (), rxPowersW);
        }
    }
}

void
WifiPhy::NotifyRxEnd (Ptr<const WifiPsdu> psdu)
{
  if (psdu && !m_phyRxEndTrace.IsEmpty ())
    {
      for (auto& mpdu : *PeekPointer (psdu))
        {
          m_phyRxEndTrace (mpdu->GetProtocolDataUnit ());
        }
    }
}

void
WifiPhy::NotifyRxDrop (Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason)
{
  if (psdu && !m_phyRxDropTrace.IsEmpty ())
    {
      for (auto& mpdu : *PeekPointer (psdu))
        {
          m_phyRxDropTrace (mpdu->GetProtocolDataUnit (), reason);
        }
    }
}

void
WifiPhy::NotifyMonitorSniffRx (Ptr<const WifiPsdu> psdu, uint16_t channelFreqMhz, WifiTxVector txVector,
                               SignalNoiseDbm signalNoise, std::vector<bool> statusPerMpdu, uint16_t staId)
{
  MpduInfo aMpdu;
  if (psdu->IsAggregate ())
    {
      //Expand A-MPDU
      NS_ASSERT_MSG (txVector.IsAggregation (), "TxVector with aggregate flag expected here according to PSDU");
      aMpdu.mpduRefNumber = ++m_rxMpduReferenceNumber;
      size_t nMpdus = psdu->GetNMpdus ();
      NS_ASSERT_MSG (statusPerMpdu.size () == nMpdus, "Should have one reception status per MPDU");
      if (!m_phyMonitorSniffRxTrace.IsEmpty ())
        {
          aMpdu.type = (psdu->IsSingle ()) ? SINGLE_MPDU : FIRST_MPDU_IN_AGGREGATE;
          for (size_t i = 0; i < nMpdus;)
            {
              if (statusPerMpdu.at (i)) //packet received without error, hand over to sniffer
                {
                  m_phyMonitorSniffRxTrace (psdu->GetAmpduSubframe (i), channelFreqMhz, txVector, aMpdu, signalNoise, staId);
                }
              ++i;
              aMpdu.type = (i == (nMpdus - 1)) ? LAST_MPDU_IN_AGGREGATE : MIDDLE_MPDU_IN_AGGREGATE;
            }
        }
    }
  else
    {
      NS_ASSERT_MSG (statusPerMpdu.size () == 1, "Should have one reception status for normal MPDU");
      if (!m_phyMonitorSniffRxTrace.IsEmpty ())
        {
          aMpdu.type = NORMAL_MPDU;
          m_phyMonitorSniffRxTrace (psdu->GetPacket (), channelFreqMhz, txVector, aMpdu, signalNoise, staId);
        }
    }
}

void
WifiPhy::NotifyMonitorSniffTx (Ptr<const WifiPsdu> psdu, uint16_t channelFreqMhz, WifiTxVector txVector, uint16_t staId)
{
  MpduInfo aMpdu;
  if (psdu->IsAggregate ())
    {
      //Expand A-MPDU
      NS_ASSERT_MSG (txVector.IsAggregation (), "TxVector with aggregate flag expected here according to PSDU");
      aMpdu.mpduRefNumber = ++m_rxMpduReferenceNumber;
      if (!m_phyMonitorSniffTxTrace.IsEmpty ())
        {
          size_t nMpdus = psdu->GetNMpdus ();
          aMpdu.type = (psdu->IsSingle ()) ? SINGLE_MPDU: FIRST_MPDU_IN_AGGREGATE;
          for (size_t i = 0; i < nMpdus;)
            {
              m_phyMonitorSniffTxTrace (psdu->GetAmpduSubframe (i), channelFreqMhz, txVector, aMpdu, staId);
              ++i;
              aMpdu.type = (i == (nMpdus - 1)) ? LAST_MPDU_IN_AGGREGATE : MIDDLE_MPDU_IN_AGGREGATE;
            }
        }
    }
  else
    {
      if (!m_phyMonitorSniffTxTrace.IsEmpty ())
        {
          aMpdu.type = NORMAL_MPDU;
          m_phyMonitorSniffTxTrace (psdu->GetPacket (), channelFreqMhz, txVector, aMpdu, staId);
        }
    }
}

WifiConstPsduMap
WifiPhy::GetWifiConstPsduMap (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector)
{
  return GetStaticPhyEntity (txVector.GetModulationClass ())->GetWifiConstPsduMap (psdu, txVector);
}

void
WifiPhy::Send (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << *psdu << txVector);
  Send (GetWifiConstPsduMap (psdu, txVector), txVector);
}

void
WifiPhy::Send (WifiConstPsduMap psdus, const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << psdus << txVector);
  /* Transmission can happen if:
   *  - we are syncing on a packet. It is the responsibility of the
   *    MAC layer to avoid doing this but the PHY does nothing to
   *    prevent it.
   *  - we are idle
   */
  NS_ASSERT (!m_state->IsStateTx () && !m_state->IsStateSwitching ());
  NS_ASSERT (m_endTxEvent.IsExpired ());

  if (txVector.GetNssMax () > GetMaxSupportedTxSpatialStreams ())
    {
      NS_FATAL_ERROR ("Unsupported number of spatial streams!");
    }

  if (m_state->IsStateSleep ())
    {
      NS_LOG_DEBUG ("Dropping packet because in sleep mode");
      for (auto const& psdu : psdus)
        {
          NotifyTxDrop (psdu.second);
        }
      return;
    }

  Time txDuration = CalculateTxDuration (psdus, txVector, GetPhyBand ());

  bool noEndPreambleDetectionEvent = true;
  for (const auto & it : m_phyEntities)
    {
      noEndPreambleDetectionEvent &= it.second->NoEndPreambleDetectionEvents ();
    }
  if (!noEndPreambleDetectionEvent || ((m_currentEvent != 0) && (m_currentEvent->GetEndTime () > (Simulator::Now () + m_state->GetDelayUntilIdle ()))))
    {
      AbortCurrentReception (RECEPTION_ABORTED_BY_TX);
    }

  for (auto & it : m_phyEntities)
    {
      it.second->CancelRunningEndPreambleDetectionEvents ();
    }
  m_currentPreambleEvents.clear ();
  m_endPhyRxEvent.Cancel ();

  if (m_powerRestricted)
    {
      NS_LOG_DEBUG ("Transmitting with power restriction for " << txDuration.As (Time::NS));
    }
  else
    {
      NS_LOG_DEBUG ("Transmitting without power restriction for " << txDuration.As (Time::NS));
    }

  if (m_state->GetState () == WifiPhyState::OFF)
    {
      NS_LOG_DEBUG ("Transmission canceled because device is OFF");
      return;
    }

  Ptr<WifiPpdu> ppdu = GetPhyEntity (txVector.GetModulationClass ())->BuildPpdu (psdus, txVector, txDuration);
  m_previouslyRxPpduUid = UINT64_MAX; //reset (after creation of PPDU) to use it only once

  double txPowerW = DbmToW (GetTxPowerForTransmission (ppdu) + GetTxGain ());
  NotifyTxBegin (psdus, txPowerW);
  if (!m_phyTxPsduBeginTrace.IsEmpty ())
    {
      m_phyTxPsduBeginTrace (psdus, txVector, txPowerW);
    }
  for (auto const& psdu : psdus)
    {
      NotifyMonitorSniffTx (psdu.second, GetFrequency (), txVector, psdu.first);
    }
  m_state->SwitchToTx (txDuration, psdus, GetPowerDbm (txVector.GetTxPowerLevel ()), txVector);

  if (m_wifiRadioEnergyModel != 0 && m_wifiRadioEnergyModel->GetMaximumTimeInState (WifiPhyState::TX) < txDuration)
    {
      ppdu->SetTruncatedTx ();
    }

  m_endTxEvent = Simulator::Schedule (txDuration, &WifiPhy::NotifyTxEnd, this, psdus); //TODO: fix for MU

  StartTx (ppdu);

  m_channelAccessRequested = false;
  m_powerRestricted = false;

  Simulator::Schedule (txDuration, &WifiPhy::Reset, this);
}

uint64_t
WifiPhy::GetPreviouslyRxPpduUid (void) const
{
  return m_previouslyRxPpduUid;
}

void
WifiPhy::Reset (void)
{
  NS_LOG_FUNCTION (this);
  m_currentPreambleEvents.clear ();
  m_currentEvent = 0;
  for (auto & phyEntity : m_phyEntities)
    {
      phyEntity.second->CancelAllEvents ();
    }
  SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (nullptr));
}

void
WifiPhy::StartReceivePreamble (Ptr<WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW, Time rxDuration)
{
  WifiModulationClass modulation = ppdu->GetTxVector ().GetModulationClass ();
  auto it = m_phyEntities.find (modulation);
  if (it != m_phyEntities.end ())
    {
      it->second->StartReceivePreamble (ppdu, rxPowersW, rxDuration);
    }
  else
    {
      //TODO find a fallback PHY for receiving the PPDU (e.g. 11a for 11ax due to preamble structure)
      NS_LOG_DEBUG ("Unsupported modulation received (" << modulation << "), consider as noise");
      m_interference->Add (ppdu, ppdu->GetTxVector (), rxDuration, rxPowersW);
      SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (nullptr));
    }
}

WifiSpectrumBand
WifiPhy::ConvertHeRuSubcarriers (uint16_t bandWidth, uint16_t guardBandwidth,
                                 HeRu::SubcarrierRange range, uint8_t bandIndex) const
{
  NS_ASSERT_MSG (false, "802.11ax can only be used with SpectrumWifiPhy");
  WifiSpectrumBand convertedSubcarriers;
  return convertedSubcarriers;
}

void
WifiPhy::EndReceiveInterBss (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_channelAccessRequested)
    {
      m_powerRestricted = false;
    }
}

void
WifiPhy::ResetReceive (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_ASSERT (!IsStateRx ());
  m_interference->NotifyRxEnd (Simulator::Now ());
  m_currentEvent = 0;
  m_currentPreambleEvents.clear ();
  SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (event->GetPpdu ()));
}

void
WifiPhy::NotifyChannelAccessRequested (void)
{
  NS_LOG_FUNCTION (this);
  m_channelAccessRequested = true;
}

bool
WifiPhy::IsModeSupported (WifiMode mode) const
{
  for (const auto & phyEntity : m_phyEntities)
    {
      if (phyEntity.second->IsModeSupported (mode))
        {
          return true;
        }
    }
  return false;
}

WifiMode
WifiPhy::GetDefaultMode (void) const
{
  //Start from oldest standards and move up (guaranteed by fact that WifModulationClass is ordered)
  for (const auto & phyEntity : m_phyEntities)
    {
      for (const auto & mode : *(phyEntity.second))
        {
          return mode;
        }
    }
  NS_ASSERT_MSG (false, "Should have found at least one default mode");
  return WifiMode ();
}

bool
WifiPhy::IsMcsSupported (WifiModulationClass modulation, uint8_t mcs) const
{
  const auto phyEntity = m_phyEntities.find (modulation);
  if (phyEntity == m_phyEntities.end ())
    {
      return false;
    }
  return phyEntity->second->IsMcsSupported (mcs);
}

std::list<WifiMode>
WifiPhy::GetModeList (void) const
{
  std::list<WifiMode> list;
  for (const auto & phyEntity : m_phyEntities)
    {
      if (!phyEntity.second->HandlesMcsModes ()) //to exclude MCSs from search
        {
          for (const auto & mode : *(phyEntity.second))
            {
              list.emplace_back (mode);
            }
        }
    }
  return list;
}

std::list<WifiMode>
WifiPhy::GetModeList (WifiModulationClass modulation) const
{
  std::list<WifiMode> list;
  const auto phyEntity = m_phyEntities.find (modulation);
  if (phyEntity != m_phyEntities.end ())
    {
      if (!phyEntity->second->HandlesMcsModes ()) //to exclude MCSs from search
        {
          for (const auto & mode : *(phyEntity->second))
            {
              list.emplace_back (mode);
            }
        }
    }
  return list;
}

uint16_t
WifiPhy::GetNMcs (void) const
{
  uint16_t numMcs = 0;
  for (const auto & phyEntity : m_phyEntities)
    {
      if (phyEntity.second->HandlesMcsModes ()) //to exclude non-MCS modes from search
        {
          numMcs += phyEntity.second->GetNumModes ();
        }
    }
  return numMcs;
}

std::list<WifiMode>
WifiPhy::GetMcsList (void) const
{
  std::list<WifiMode> list;
  for (const auto & phyEntity : m_phyEntities)
    {
      if (phyEntity.second->HandlesMcsModes ()) //to exclude non-MCS modes from search
        {
          for (const auto & mode : *(phyEntity.second))
            {
              list.emplace_back (mode);
            }
        }
    }
  return list;
}

std::list<WifiMode>
WifiPhy::GetMcsList (WifiModulationClass modulation) const
{
  std::list<WifiMode> list;
  auto phyEntity = m_phyEntities.find (modulation);
  if (phyEntity != m_phyEntities.end ())
    {
      if (phyEntity->second->HandlesMcsModes ()) //to exclude non-MCS modes from search
        {
          for (const auto & mode : *(phyEntity->second))
            {
              list.emplace_back (mode);
            }
        }
    }
  return list;
}

WifiMode
WifiPhy::GetMcs (WifiModulationClass modulation, uint8_t mcs) const
{
  NS_ASSERT_MSG (IsMcsSupported (modulation, mcs), "Unsupported MCS");
  return m_phyEntities.at (modulation)->GetMcs (mcs);
}

bool
WifiPhy::IsStateCcaBusy (void) const
{
  return m_state->IsStateCcaBusy ();
}

bool
WifiPhy::IsStateIdle (void) const
{
  return m_state->IsStateIdle ();
}

bool
WifiPhy::IsStateRx (void) const
{
  return m_state->IsStateRx ();
}

bool
WifiPhy::IsStateTx (void) const
{
  return m_state->IsStateTx ();
}

bool
WifiPhy::IsStateSwitching (void) const
{
  return m_state->IsStateSwitching ();
}

bool
WifiPhy::IsStateSleep (void) const
{
  return m_state->IsStateSleep ();
}

bool
WifiPhy::IsStateOff (void) const
{
  return m_state->IsStateOff ();
}

Time
WifiPhy::GetDelayUntilIdle (void)
{
  return m_state->GetDelayUntilIdle ();
}

Time
WifiPhy::GetLastRxStartTime (void) const
{
  return m_state->GetLastRxStartTime ();
}

Time
WifiPhy::GetLastRxEndTime (void) const
{
  return m_state->GetLastRxEndTime ();
}

void
WifiPhy::SwitchMaybeToCcaBusy (uint16_t channelWidth)
{
  NS_LOG_FUNCTION (this << channelWidth);
  //We are here because we have received the first bit of a packet and we are
  //not going to be able to synchronize on it
  //In this model, CCA becomes busy when the aggregation of all signals as
  //tracked by the InterferenceHelper class is higher than the CcaBusyThreshold
  Time delayUntilCcaEnd = m_interference->GetEnergyDuration (m_ccaEdThresholdW, GetPrimaryBand (channelWidth));
  if (!delayUntilCcaEnd.IsZero ())
    {
      NS_LOG_DEBUG ("Calling SwitchMaybeToCcaBusy for " << delayUntilCcaEnd.As (Time::S));
      m_state->SwitchMaybeToCcaBusy (delayUntilCcaEnd);
    }
}

void
WifiPhy::AbortCurrentReception (WifiPhyRxfailureReason reason)
{
  NS_LOG_FUNCTION (this << reason);
  if (reason != OBSS_PD_CCA_RESET || m_currentEvent) //Otherwise abort has already been called previously
    {
      for (auto & phyEntity : m_phyEntities)
        {
          phyEntity.second->CancelAllEvents ();
        }
      if (m_endPhyRxEvent.IsRunning ())
        {
          m_endPhyRxEvent.Cancel ();
        }
      m_interference->NotifyRxEnd (Simulator::Now ());
      if (!m_currentEvent)
        {
          return;
        }
      NotifyRxDrop (GetAddressedPsduInPpdu (m_currentEvent->GetPpdu ()), reason);
      if (reason == OBSS_PD_CCA_RESET)
        {
          m_state->SwitchFromRxAbort ();
        }
      for (auto it = m_currentPreambleEvents.begin (); it != m_currentPreambleEvents.end (); ++it)
        {
          if (it->second == m_currentEvent)
            {
              it = m_currentPreambleEvents.erase (it);
              break;
            }
        }
      m_currentEvent = 0;
    }
}

void
WifiPhy::ResetCca (bool powerRestricted, double txPowerMaxSiso, double txPowerMaxMimo)
{
  NS_LOG_FUNCTION (this << powerRestricted << txPowerMaxSiso << txPowerMaxMimo);
  // This method might be called multiple times when receiving TB PPDUs with a BSS color
  // different than the one of the receiver. The first time this method is called, the call
  // to AbortCurrentReception sets m_currentEvent to 0. Therefore, we need to check whether
  // m_currentEvent is not 0 before executing the instructions below.
  if (m_currentEvent != 0)
    {
      m_powerRestricted = powerRestricted;
      m_txPowerMaxSiso = txPowerMaxSiso;
      m_txPowerMaxMimo = txPowerMaxMimo;
      NS_ASSERT ((m_currentEvent->GetEndTime () - Simulator::Now ()).IsPositive ());
      Simulator::Schedule (m_currentEvent->GetEndTime () - Simulator::Now (), &WifiPhy::EndReceiveInterBss, this);
      Simulator::ScheduleNow (&WifiPhy::AbortCurrentReception, this, OBSS_PD_CCA_RESET); //finish processing field first
    }
}

double
WifiPhy::GetTxPowerForTransmission (Ptr<const WifiPpdu> ppdu) const
{
  NS_LOG_FUNCTION (this << m_powerRestricted << ppdu);
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  // Get transmit power before antenna gain
  double txPowerDbm;
  if (!m_powerRestricted)
    {
      txPowerDbm = GetPowerDbm (txVector.GetTxPowerLevel ());
    }
  else
    {
      if (txVector.GetNssMax () > 1)
        {
          txPowerDbm = std::min (m_txPowerMaxMimo, GetPowerDbm (txVector.GetTxPowerLevel ()));
        }
      else
        {
          txPowerDbm = std::min (m_txPowerMaxSiso, GetPowerDbm (txVector.GetTxPowerLevel ()));
        }
    }

  //Apply power density constraint on EIRP
  uint16_t channelWidth = ppdu->GetTransmissionChannelWidth ();
  double txPowerDbmPerMhz = (txPowerDbm + GetTxGain ()) - RatioToDb (channelWidth); //account for antenna gain since EIRP
  NS_LOG_INFO ("txPowerDbm=" << txPowerDbm << " with txPowerDbmPerMhz=" << txPowerDbmPerMhz << " over " << channelWidth << " MHz");
  txPowerDbm = std::min (txPowerDbmPerMhz, m_powerDensityLimit) + RatioToDb (channelWidth);
  txPowerDbm -= GetTxGain (); //remove antenna gain since will be added right afterwards
  NS_LOG_INFO ("txPowerDbm=" << txPowerDbm << " after applying m_powerDensityLimit=" << m_powerDensityLimit);
  return txPowerDbm;
}

Ptr<const WifiPsdu>
WifiPhy::GetAddressedPsduInPpdu (Ptr<const WifiPpdu> ppdu) const
{
  //TODO: wrapper. See if still needed
  return GetPhyEntity (ppdu->GetModulation ())->GetAddressedPsduInPpdu (ppdu);
}

uint16_t
WifiPhy::GetMeasurementChannelWidth (const Ptr<const WifiPpdu> ppdu) const
{
  if (ppdu == nullptr)
    {
      // Here because PHY was not receiving anything (e.g. resuming from OFF) nor expecting anything (e.g. sleep)
      // nor processing a Wi-Fi signal.
      return GetChannelWidth () >= 40 ? 20 : GetChannelWidth ();
    }
  return GetPhyEntity (ppdu->GetModulation ())->GetMeasurementChannelWidth (ppdu);
}

WifiSpectrumBand
WifiPhy::GetBand (uint16_t /*bandWidth*/, uint8_t /*bandIndex*/)
{
  WifiSpectrumBand band;
  band.first = 0;
  band.second = 0;
  return band;
}

WifiSpectrumBand
WifiPhy::GetPrimaryBand (uint16_t bandWidth)
{
  if (GetChannelWidth () % 20 != 0)
    {
      return GetBand (bandWidth);
    }

  return GetBand (bandWidth, m_operatingChannel.GetPrimaryChannelIndex (bandWidth));
}

int64_t
WifiPhy::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  int64_t currentStream = stream;
  m_random->SetStream (currentStream++);
  currentStream += m_interference->GetErrorRateModel ()->AssignStreams (currentStream);
  return (currentStream - stream);
}

std::ostream& operator<< (std::ostream& os, RxSignalInfo rxSignalInfo)
{
  os << "SNR:" << RatioToDb (rxSignalInfo.snr) << " dB"
     << ", RSSI:" << rxSignalInfo.rssi << " dBm";
  return os;
}

uint8_t
WifiPhy::GetPrimaryChannelNumber (uint16_t primaryChannelWidth) const
{
  return m_operatingChannel.GetPrimaryChannelNumber (primaryChannelWidth, m_standard);
}

} //namespace ns3
