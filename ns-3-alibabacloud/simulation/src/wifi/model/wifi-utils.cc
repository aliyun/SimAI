/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016
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

#include <cmath>
#include "ns3/packet.h"
#include "wifi-utils.h"
#include "ctrl-headers.h"
#include "wifi-mac-header.h"
#include "wifi-mac-trailer.h"

namespace ns3 {

double
DbToRatio (double dB)
{
  return std::pow (10.0, 0.1 * dB);
}

double
DbmToW (double dBm)
{
  return std::pow (10.0, 0.1 * (dBm - 30.0));
}

double
WToDbm (double w)
{
  return 10.0 * std::log10 (w) + 30.0;
}

double
RatioToDb (double ratio)
{
  return 10.0 * std::log10 (ratio);
}

uint32_t
GetAckSize (void)
{
  WifiMacHeader ack;
  ack.SetType (WIFI_MAC_CTL_ACK);
  return ack.GetSize () + 4;
}

uint32_t
GetBlockAckSize (BlockAckType type)
{
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_BACKRESP);
  CtrlBAckResponseHeader blockAck;
  blockAck.SetType (type);
  return hdr.GetSize () + blockAck.GetSerializedSize () + 4;
}

uint32_t
GetBlockAckRequestSize (BlockAckReqType type)
{
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_BACKREQ);
  CtrlBAckRequestHeader bar;
  bar.SetType (type);
  return hdr.GetSize () + bar.GetSerializedSize () + 4;
}

uint32_t
GetMuBarSize (std::list<BlockAckReqType> types)
{
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_TRIGGER);
  CtrlTriggerHeader trigger;
  trigger.SetType (MU_BAR_TRIGGER);
  for (auto& t : types)
    {
      auto userInfo = trigger.AddUserInfoField ();
      CtrlBAckRequestHeader bar;
      bar.SetType (t);
      userInfo.SetMuBarTriggerDepUserInfo (bar);
    }
  return hdr.GetSize () + trigger.GetSerializedSize () + 4;
}

uint32_t
GetRtsSize (void)
{
  WifiMacHeader rts;
  rts.SetType (WIFI_MAC_CTL_RTS);
  return rts.GetSize () + 4;
}

uint32_t
GetCtsSize (void)
{
  WifiMacHeader cts;
  cts.SetType (WIFI_MAC_CTL_CTS);
  return cts.GetSize () + 4;
}

bool
IsInWindow (uint16_t seq, uint16_t winstart, uint16_t winsize)
{
  return ((seq - winstart + 4096) % 4096) < winsize;
}

void
AddWifiMacTrailer (Ptr<Packet> packet)
{
  WifiMacTrailer fcs;
  packet->AddTrailer (fcs);
}

uint32_t
GetSize (Ptr<const Packet> packet, const WifiMacHeader *hdr, bool isAmpdu)
{
  uint32_t size;
  WifiMacTrailer fcs;
  if (isAmpdu)
    {
      size = packet->GetSize ();
    }
  else
    {
      size = packet->GetSize () + hdr->GetSize () + fcs.GetSerializedSize ();
    }
  return size;
}

} //namespace ns3
