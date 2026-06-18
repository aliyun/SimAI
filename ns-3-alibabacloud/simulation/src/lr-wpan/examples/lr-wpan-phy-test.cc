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
 * Author: Gary Pei <guangyu.pei@boeing.com>
 */
#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/command-line.h>
#include <ns3/packet.h>
#include <ns3/lr-wpan-phy.h>
#include <ns3/lr-wpan-mac.h>
#include <ns3/simulator.h>
#include <ns3/single-model-spectrum-channel.h>
#include <ns3/constant-position-mobility-model.h>

using namespace ns3;

/**
 * Function called when a the PHY state change is confirmed
 * \param status PHY state
 */
void
GetSetTRXStateConfirm (LrWpanPhyEnumeration status)
{
  NS_LOG_UNCOND ("At: " << Simulator::Now ()
                        << " Received Set TRX Confirm: " << status);
}

/**
 * Function called when a the PHY receives a packet
 * \param psduLength PSDU length
 * \param p packet
 * \param lqi link quality indication
 */
void
ReceivePdDataIndication (uint32_t psduLength,
                         Ptr<Packet> p,
                         uint8_t lqi)
{
  NS_LOG_UNCOND ("At: " << Simulator::Now ()
                        << " Received frame size: " << psduLength << " LQI: " <<
                 (uint16_t) lqi);
}

/**
 * Send one packet
 * \param sender sender PHY
 * \param receiver receiver PHY
 */
void
SendOnePacket (Ptr<LrWpanPhy> sender, Ptr<LrWpanPhy> receiver)
{
  uint32_t n = 10;
  Ptr<Packet> p = Create<Packet> (n);
  sender->PdDataRequest (p->GetSize (), p);
}


int main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  LogComponentEnableAll (LOG_PREFIX_FUNC);
  LogComponentEnable ("LrWpanPhy", LOG_LEVEL_ALL);
  LogComponentEnable ("SingleModelSpectrumChannel", LOG_LEVEL_ALL);

  Ptr<LrWpanPhy> sender = CreateObject<LrWpanPhy> ();
  Ptr<LrWpanPhy> receiver = CreateObject<LrWpanPhy> ();

  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  sender->SetChannel (channel);
  receiver->SetChannel (channel);
  channel->AddRx (sender);
  channel->AddRx (receiver);

  // CONFIGURE MOBILITY
  Ptr<ConstantPositionMobilityModel> senderMobility = CreateObject<ConstantPositionMobilityModel> ();
  sender->SetMobility (senderMobility);
  Ptr<ConstantPositionMobilityModel> receiverMobility = CreateObject<ConstantPositionMobilityModel> ();
  receiver->SetMobility (receiverMobility);


  sender->SetPlmeSetTRXStateConfirmCallback (MakeCallback (&GetSetTRXStateConfirm));
  receiver->SetPlmeSetTRXStateConfirmCallback (MakeCallback (&GetSetTRXStateConfirm));

  sender->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TX_ON);
  receiver->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);

  receiver->SetPdDataIndicationCallback (MakeCallback (&ReceivePdDataIndication));

  Simulator::Schedule (Seconds (1.0), &SendOnePacket, sender, receiver);

  Simulator::Stop (Seconds (10.0));

  Simulator::Run ();

  Simulator::Destroy ();

  return 0;
}
