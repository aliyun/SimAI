/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/packet.h"
#include "adhoc-wifi-mac.h"
#include "qos-txop.h"
#include "ns3/ht-capabilities.h"
#include "ns3/vht-capabilities.h"
#include "ns3/he-capabilities.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AdhocWifiMac");

NS_OBJECT_ENSURE_REGISTERED (AdhocWifiMac);

TypeId
AdhocWifiMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AdhocWifiMac")
    .SetParent<WifiMac> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AdhocWifiMac> ()
  ;
  return tid;
}

AdhocWifiMac::AdhocWifiMac ()
{
  NS_LOG_FUNCTION (this);
  //Let the lower layers know that we are acting in an IBSS
  SetTypeOfStation (ADHOC_STA);
}

AdhocWifiMac::~AdhocWifiMac ()
{
  NS_LOG_FUNCTION (this);
}

void
AdhocWifiMac::SetAddress (Mac48Address address)
{
  NS_LOG_FUNCTION (this << address);
  //In an IBSS, the BSSID is supposed to be generated per Section
  //11.1.3 of IEEE 802.11. We don't currently do this - instead we
  //make an IBSS STA a bit like an AP, with the BSSID for frames
  //transmitted by each STA set to that STA's address.
  //
  //This is why we're overriding this method.
  WifiMac::SetAddress (address);
  WifiMac::SetBssid (address);
}

bool
AdhocWifiMac::CanForwardPacketsTo (Mac48Address to) const
{
  return true;
}

void
AdhocWifiMac::Enqueue (Ptr<Packet> packet, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << to);
  if (GetWifiRemoteStationManager ()->IsBrandNew (to))
    {
      //In ad hoc mode, we assume that every destination supports all the rates we support.
      if (GetHtSupported ())
        {
          GetWifiRemoteStationManager ()->AddAllSupportedMcs (to);
          GetWifiRemoteStationManager ()->AddStationHtCapabilities (to, GetHtCapabilities ());
        }
      if (GetVhtSupported ())
        {
          GetWifiRemoteStationManager ()->AddStationVhtCapabilities (to, GetVhtCapabilities ());
        }
      if (GetHeSupported ())
        {
          GetWifiRemoteStationManager ()->AddStationHeCapabilities (to, GetHeCapabilities ());
        }
      GetWifiRemoteStationManager ()->AddAllSupportedModes (to);
      GetWifiRemoteStationManager ()->RecordDisassociated (to);
    }

  WifiMacHeader hdr;

  //If we are not a QoS STA then we definitely want to use AC_BE to
  //transmit the packet. A TID of zero will map to AC_BE (through \c
  //QosUtilsMapTidToAc()), so we use that as our default here.
  uint8_t tid = 0;

  //For now, a STA that supports QoS does not support non-QoS
  //associations, and vice versa. In future the STA model should fall
  //back to non-QoS if talking to a peer that is also non-QoS. At
  //that point there will need to be per-station QoS state maintained
  //by the association state machine, and consulted here.
  if (GetQosSupported ())
    {
      hdr.SetType (WIFI_MAC_QOSDATA);
      hdr.SetQosAckPolicy (WifiMacHeader::NORMAL_ACK);
      hdr.SetQosNoEosp ();
      hdr.SetQosNoAmsdu ();
      //Transmission of multiple frames in the same TXOP is not
      //supported for now
      hdr.SetQosTxopLimit (0);

      //Fill in the QoS control field in the MAC header
      tid = QosUtilsGetTidForPacket (packet);
      //Any value greater than 7 is invalid and likely indicates that
      //the packet had no QoS tag, so we revert to zero, which will
      //mean that AC_BE is used.
      if (tid > 7)
        {
          tid = 0;
        }
      hdr.SetQosTid (tid);
    }
  else
    {
      hdr.SetType (WIFI_MAC_DATA);
    }

  if (GetHtSupported ())
    {
      hdr.SetNoOrder (); // explicitly set to 0 for the time being since HT control field is not yet implemented (set it to 1 when implemented)
    }
  hdr.SetAddr1 (to);
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (GetBssid ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();

  if (GetQosSupported ())
    {
      //Sanity check that the TID is valid
      NS_ASSERT (tid < 8);
      GetQosTxop (tid)->Queue (packet, hdr);
    }
  else
    {
      GetTxop ()->Queue (packet, hdr);
    }
}

void
AdhocWifiMac::SetLinkUpCallback (Callback<void> linkUp)
{
  NS_LOG_FUNCTION (this << &linkUp);
  WifiMac::SetLinkUpCallback (linkUp);

  //The approach taken here is that, from the point of view of a STA
  //in IBSS mode, the link is always up, so we immediately invoke the
  //callback if one is set
  linkUp ();
}

void
AdhocWifiMac::Receive (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  const WifiMacHeader* hdr = &mpdu->GetHeader ();
  NS_ASSERT (!hdr->IsCtl ());
  Mac48Address from = hdr->GetAddr2 ();
  Mac48Address to = hdr->GetAddr1 ();
  if (GetWifiRemoteStationManager ()->IsBrandNew (from))
    {
      //In ad hoc mode, we assume that every destination supports all the rates we support.
      if (GetHtSupported ())
        {
          GetWifiRemoteStationManager ()->AddAllSupportedMcs (from);
          GetWifiRemoteStationManager ()->AddStationHtCapabilities (from, GetHtCapabilities ());
        }
      if (GetVhtSupported ())
        {
          GetWifiRemoteStationManager ()->AddStationVhtCapabilities (from, GetVhtCapabilities ());
        }
      if (GetHeSupported ())
        {
          GetWifiRemoteStationManager ()->AddStationHeCapabilities (from, GetHeCapabilities ());
        }
      GetWifiRemoteStationManager ()->AddAllSupportedModes (from);
      GetWifiRemoteStationManager ()->RecordDisassociated (from);
    }
  if (hdr->IsData ())
    {
      if (hdr->IsQosData () && hdr->IsQosAmsdu ())
        {
          NS_LOG_DEBUG ("Received A-MSDU from" << from);
          DeaggregateAmsduAndForward (mpdu);
        }
      else
        {
          ForwardUp (mpdu->GetPacket ()->Copy (), from, to);
        }
      return;
    }

  //Invoke the receive handler of our parent class to deal with any
  //other frames. Specifically, this will handle Block Ack-related
  //Management Action frames.
  WifiMac::Receive (mpdu);
}

} //namespace ns3
