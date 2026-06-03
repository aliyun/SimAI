/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

//
// This script is used to verify the behavior of InterferenceHelper.
//
// The scenario consists of two IEEE 802.11 hidden stations and an access point.
// The two stations have both a packet to transmit to the access point.
//
//
// (xA,0,0)     (0,0,0)      (xB,0,0)
//
//    *   ----->   *   <-----   *
//    |            |            |
//   STA A         AP          STA B
//
//
// The program can be configured at run-time by passing command-line arguments.
// It enables to configure the delay between the transmission from station A
// and the transmission from station B (--delay option). It is also possible to
// select the tx power level (--txPowerA and --txPowerB options), the packet size
// (--packetSizeA and --packetSizeB options) and the modulation (--txModeA and
// --txModeB options) used for the respective transmissions.
//
// By default, IEEE 802.11a with long preamble type is considered, but those
// parameters can be also picked among other IEEE 802.11 flavors and preamble
// types available in the simulator (--standard and --preamble options).
// Note that the program checks the consistency between the selected standard
// the selected preamble type.
//
// The output of the program displays InterfenceHelper and SpectrumWifiPhy trace
// logs associated to the chosen scenario.
//

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/simulator.h"
#include "ns3/command-line.h"
#include "ns3/single-model-spectrum-channel.h"
#include "ns3/spectrum-wifi-phy.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/interference-helper.h"
#include "ns3/nist-error-rate-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/simple-frame-capture-model.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-mac-trailer.h"
#include "ns3/wifi-net-device.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("test-interference-helper");

bool checkResults = false;
bool expectRxASuccessfull = false;
bool expectRxBSuccessfull = false;

/// InterferenceExperiment
class InterferenceExperiment
{
public:
  /// Input structure
  struct Input
  {
    Input ();
    Time interval; ///< interval
    double xA; ///< x A
    double xB; ///< x B
    std::string txModeA; ///< transmit mode A
    std::string txModeB; ///< transmit mode B
    double txPowerLevelA; ///< transmit power level A
    double txPowerLevelB; ///< transmit power level B
    uint32_t packetSizeA; ///< packet size A
    uint32_t packetSizeB; ///< packet size B
    uint16_t channelA; ///< channel number A
    uint16_t channelB; ///< channel number B
    uint16_t widthA; ///< channel width A
    uint16_t widthB; ///< channel width B
    WifiStandard standard; ///< standard
    WifiPhyBand band; ///< band
    WifiPreamble preamble; ///< preamble
    bool captureEnabled; ///< whether physical layer capture is enabled
    double captureMargin; ///< margin used for physical layer capture
  };

  InterferenceExperiment ();
  /**
   * Run function
   * \param input the interference experiment data
   */
  void Run (struct InterferenceExperiment::Input input);

private:
  /**
   * Function triggered when a packet is dropped
   * \param packet the packet that was dropped
   * \param reason the reason why it was dropped
   */
  void PacketDropped (Ptr<const Packet> packet, WifiPhyRxfailureReason reason);
  /// Send A function
  void SendA (void) const;
  /// Send B function
  void SendB (void) const;
  Ptr<SpectrumWifiPhy> m_txA; ///< transmit A function
  Ptr<SpectrumWifiPhy> m_txB; ///< transmit B function
  struct Input m_input; ///< input
  bool m_droppedA; ///< flag to indicate whether packet A has been dropped
  bool m_droppedB; ///< flag to indicate whether packet B has been dropped
  mutable uint64_t m_uidA; ///< UID to use for packet A
  mutable uint64_t m_uidB; ///< UID to use for packet B
};

void
InterferenceExperiment::SendA (void) const
{
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_ACK); //so that size may not be empty while being as short as possible
  Ptr<Packet> p = Create<Packet> (m_input.packetSizeA - hdr.GetSerializedSize () - WIFI_MAC_FCS_LENGTH);
  m_uidA = p->GetUid ();
  Ptr<WifiPsdu> psdu = Create<WifiPsdu> (p, hdr);
  WifiTxVector txVector;
  txVector.SetTxPowerLevel (0); //only one TX power level
  txVector.SetMode (WifiMode (m_input.txModeA));
  txVector.SetChannelWidth (m_input.widthA);
  txVector.SetPreambleType (m_input.preamble);
  m_txA->Send (psdu, txVector);
}

void
InterferenceExperiment::SendB (void) const
{
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_ACK); //so that size may not be empty while being as short as possible
  Ptr<Packet> p = Create<Packet> (m_input.packetSizeB - hdr.GetSerializedSize () - WIFI_MAC_FCS_LENGTH);
  m_uidB = p->GetUid ();
  Ptr<WifiPsdu> psdu = Create<WifiPsdu> (p, hdr);
  WifiTxVector txVector;
  txVector.SetTxPowerLevel (0); //only one TX power level
  txVector.SetMode (WifiMode (m_input.txModeB));
  txVector.SetChannelWidth (m_input.widthB);
  txVector.SetPreambleType (m_input.preamble);
  m_txB->Send (psdu, txVector);
}

void
InterferenceExperiment::PacketDropped (Ptr<const Packet> packet, WifiPhyRxfailureReason reason)
{
  if (packet->GetUid () == m_uidA)
    {
      m_droppedA = true;
    }
  else if (packet->GetUid () == m_uidB)
    {
      m_droppedB = true;
    }
  else
    {
      NS_LOG_ERROR ("Unknown packet!");
      exit (1);
    }
}

InterferenceExperiment::InterferenceExperiment ()
  : m_droppedA (false),
    m_droppedB (false),
    m_uidA (0),
    m_uidB (0)
{
}

InterferenceExperiment::Input::Input ()
  : interval (MicroSeconds (0)),
    xA (-5),
    xB (5),
    txModeA ("OfdmRate54Mbps"),
    txModeB ("OfdmRate54Mbps"),
    txPowerLevelA (16.0206),
    txPowerLevelB (16.0206),
    packetSizeA (1500),
    packetSizeB (1500),
    channelA (36),
    channelB (36),
    widthA (20),
    widthB (20),
    standard (WIFI_STANDARD_80211a),
    band (WIFI_PHY_BAND_5GHZ),
    preamble (WIFI_PREAMBLE_LONG),
    captureEnabled (false),
    captureMargin (0)
{
}

void
InterferenceExperiment::Run (struct InterferenceExperiment::Input input)
{
  m_input = input;

  double range = std::max (std::abs (input.xA), input.xB);
  Config::SetDefault ("ns3::RangePropagationLossModel::MaxRange", DoubleValue (range));

  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  channel->SetPropagationDelayModel (CreateObject<ConstantSpeedPropagationDelayModel> ());
  Ptr<RangePropagationLossModel> loss = CreateObject<RangePropagationLossModel> ();
  channel->AddPropagationLossModel (loss);

  Ptr<MobilityModel> posTxA = CreateObject<ConstantPositionMobilityModel> ();
  posTxA->SetPosition (Vector (input.xA, 0.0, 0.0));
  Ptr<MobilityModel> posTxB = CreateObject<ConstantPositionMobilityModel> ();
  posTxB->SetPosition (Vector (input.xB, 0.0, 0.0));
  Ptr<MobilityModel> posRx = CreateObject<ConstantPositionMobilityModel> ();
  posRx->SetPosition (Vector (0.0, 0.0, 0.0));

  Ptr<Node> nodeA = CreateObject<Node> ();
  Ptr<WifiNetDevice> devA = CreateObject<WifiNetDevice> ();
  m_txA = CreateObject<SpectrumWifiPhy> ();
  m_txA->CreateWifiSpectrumPhyInterface (devA);
  m_txA->SetDevice (devA);
  m_txA->SetTxPowerStart (input.txPowerLevelA);
  m_txA->SetTxPowerEnd (input.txPowerLevelA);

  Ptr<Node> nodeB = CreateObject<Node> ();
  Ptr<WifiNetDevice> devB = CreateObject<WifiNetDevice> ();
  m_txB = CreateObject<SpectrumWifiPhy> ();
  m_txB->CreateWifiSpectrumPhyInterface (devB);
  m_txB->SetDevice (devB);
  m_txB->SetTxPowerStart (input.txPowerLevelB);
  m_txB->SetTxPowerEnd (input.txPowerLevelB);

  Ptr<Node> nodeRx = CreateObject<Node> ();
  Ptr<WifiNetDevice> devRx = CreateObject<WifiNetDevice> ();
  Ptr<SpectrumWifiPhy> rx = CreateObject<SpectrumWifiPhy> ();
  rx->CreateWifiSpectrumPhyInterface (devRx);
  rx->SetDevice (devRx);

  Ptr<InterferenceHelper> interferenceTxA = CreateObject<InterferenceHelper> ();
  m_txA->SetInterferenceHelper (interferenceTxA);
  Ptr<ErrorRateModel> errorTxA = CreateObject<NistErrorRateModel> ();
  m_txA->SetErrorRateModel (errorTxA);
  Ptr<InterferenceHelper> interferenceTxB = CreateObject<InterferenceHelper> ();
  m_txB->SetInterferenceHelper (interferenceTxB);
  Ptr<ErrorRateModel> errorTxB = CreateObject<NistErrorRateModel> ();
  m_txB->SetErrorRateModel (errorTxB);
  Ptr<InterferenceHelper> interferenceRx = CreateObject<InterferenceHelper> ();
  rx->SetInterferenceHelper (interferenceRx);
  Ptr<ErrorRateModel> errorRx = CreateObject<NistErrorRateModel> ();
  rx->SetErrorRateModel (errorRx);
  m_txA->SetChannel (channel);
  m_txB->SetChannel (channel);
  rx->SetChannel (channel);
  m_txA->SetMobility (posTxA);
  m_txB->SetMobility (posTxB);
  rx->SetMobility (posRx);
  if (input.captureEnabled)
    {
      Ptr<SimpleFrameCaptureModel> frameCaptureModel = CreateObject<SimpleFrameCaptureModel> ();
      frameCaptureModel->SetMargin (input.captureMargin);
      rx->SetFrameCaptureModel (frameCaptureModel);
    }

  m_txA->ConfigureStandard (input.standard);
  m_txB->ConfigureStandard (input.standard);
  rx->ConfigureStandard (input.standard);

  devA->SetPhy (m_txA);
  nodeA->AddDevice (devA);
  devB->SetPhy (m_txB);
  nodeB->AddDevice (devB);
  devRx->SetPhy (rx);
  nodeRx->AddDevice (devRx);

  m_txA->SetOperatingChannel (WifiPhy::ChannelTuple {input.channelA, 0, (int)(input.band), 0});
  m_txB->SetOperatingChannel (WifiPhy::ChannelTuple {input.channelB, 0, (int)(input.band), 0});
  rx->SetOperatingChannel (WifiPhy::ChannelTuple {std::max (input.channelA, input.channelB), 0,
                                                  (int)(input.band), 0});

  rx->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&InterferenceExperiment::PacketDropped, this));

  Simulator::Schedule (Seconds (0), &InterferenceExperiment::SendA, this);
  Simulator::Schedule (Seconds (0) + input.interval, &InterferenceExperiment::SendB, this);

  Simulator::Run ();
  Simulator::Destroy ();
  m_txB->Dispose ();
  m_txA->Dispose ();
  rx->Dispose ();

  if (checkResults && (m_droppedA == expectRxASuccessfull || m_droppedB == expectRxBSuccessfull))
    {
      NS_LOG_ERROR ("Results are not expected!");
      exit (1);
    }
}

int main (int argc, char *argv[])
{
  InterferenceExperiment::Input input;
  std::string str_standard = "WIFI_PHY_STANDARD_80211a";
  std::string str_preamble = "WIFI_PREAMBLE_LONG";
  uint64_t delay = 0; //microseconds

  CommandLine cmd (__FILE__);
  cmd.AddValue ("delay", "Delay in microseconds between frame transmission from sender A and frame transmission from sender B", delay);
  cmd.AddValue ("xA", "The position of transmitter A (< 0)", input.xA);
  cmd.AddValue ("xB", "The position of transmitter B (> 0)", input.xB);
  cmd.AddValue ("packetSizeA", "Packet size in bytes of transmitter A", input.packetSizeA);
  cmd.AddValue ("packetSizeB", "Packet size in bytes of transmitter B", input.packetSizeB);
  cmd.AddValue ("txPowerA", "TX power level of transmitter A", input.txPowerLevelA);
  cmd.AddValue ("txPowerB", "TX power level of transmitter B", input.txPowerLevelB);
  cmd.AddValue ("txModeA", "Wifi mode used for payload transmission of sender A", input.txModeA);
  cmd.AddValue ("txModeB", "Wifi mode used for payload transmission of sender B", input.txModeB);
  cmd.AddValue ("channelA", "The selected channel number of sender A", input.channelA);
  cmd.AddValue ("channelB", "The selected channel number of sender B", input.channelB);
  cmd.AddValue ("widthA", "The selected channel width (MHz) of sender A", input.widthA);
  cmd.AddValue ("widthB", "The selected channel width (MHz) of sender B", input.widthB);
  cmd.AddValue ("standard", "IEEE 802.11 flavor", str_standard);
  cmd.AddValue ("preamble", "Type of preamble", str_preamble);
  cmd.AddValue ("enableCapture", "Enable/disable physical layer capture", input.captureEnabled);
  cmd.AddValue ("captureMargin", "Margin used for physical layer capture", input.captureMargin);
  cmd.AddValue ("checkResults", "Used to check results at the end of the test", checkResults);
  cmd.AddValue ("expectRxASuccessfull", "Indicate whether packet A is expected to be successfully received", expectRxASuccessfull);
  cmd.AddValue ("expectRxBSuccessfull", "Indicate whether packet B is expected to be successfully received", expectRxBSuccessfull);
  cmd.Parse (argc, argv);

  input.interval = MicroSeconds (delay);

  if (input.xA >= 0 || input.xB <= 0)
    {
      std::cout << "Value of xA must be smaller than 0 and value of xB must be bigger than 0!" << std::endl;
      return 0;
    }

  if (str_standard == "WIFI_PHY_STANDARD_80211a")
    {
      input.standard = WIFI_STANDARD_80211a;
      input.band = WIFI_PHY_BAND_5GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211b")
    {
      input.standard = WIFI_STANDARD_80211b;
      input.band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211g")
    {
      input.standard = WIFI_STANDARD_80211g;
      input.band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211n_2_4GHZ")
    {
      input.standard = WIFI_STANDARD_80211n;
      input.band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211n_5GHZ")
    {
      input.standard = WIFI_STANDARD_80211n;
      input.band = WIFI_PHY_BAND_5GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211ac")
    {
      input.standard = WIFI_STANDARD_80211ac;
      input.band = WIFI_PHY_BAND_5GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211ax_2_4GHZ")
    {
      input.standard = WIFI_STANDARD_80211ax;
      input.band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (str_standard == "WIFI_PHY_STANDARD_80211ax_5GHZ")
    {
      input.standard = WIFI_STANDARD_80211ax;
      input.band = WIFI_PHY_BAND_5GHZ;
    }

  if (str_preamble == "WIFI_PREAMBLE_LONG" && (input.standard == WIFI_STANDARD_80211a || input.standard == WIFI_STANDARD_80211b || input.standard == WIFI_STANDARD_80211g))
    {
      input.preamble = WIFI_PREAMBLE_LONG;
    }
  else if (str_preamble == "WIFI_PREAMBLE_SHORT" && (input.standard == WIFI_STANDARD_80211b || input.standard == WIFI_STANDARD_80211g))
    {
      input.preamble = WIFI_PREAMBLE_SHORT;
    }
  else if (str_preamble == "WIFI_PREAMBLE_HT_MF" && input.standard == WIFI_STANDARD_80211n)
    {
      input.preamble = WIFI_PREAMBLE_HT_MF;
    }
  else if (str_preamble == "WIFI_PREAMBLE_VHT_SU" && input.standard == WIFI_STANDARD_80211ac)
    {
      input.preamble = WIFI_PREAMBLE_VHT_SU;
    }
  else if (str_preamble == "WIFI_PREAMBLE_HE_SU" && input.standard == WIFI_STANDARD_80211ax)
    {
      input.preamble = WIFI_PREAMBLE_HE_SU;
    }
  else
    {
      std::cout << "Preamble does not exist or is not compatible with the selected standard!" << std::endl;
      return 0;
    }

  InterferenceExperiment experiment;
  experiment.Run (input);

  return 0;
}
