/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "uan-transducer-hd.h"
#include "ns3/simulator.h"
#include "ns3/uan-prop-model.h"
#include "uan-phy.h"
#include "uan-channel.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/double.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("UanTransducerHd");

NS_OBJECT_ENSURE_REGISTERED (UanTransducerHd);
  
UanTransducerHd::UanTransducerHd ()
  : UanTransducer (),
    m_state (RX),
    m_endTxTime (Seconds (0)),
    m_cleared (false),
    m_rxGainDb (0)
{
}

UanTransducerHd::~UanTransducerHd ()
{
}

void
UanTransducerHd::Clear ()
{
  if (m_cleared)
    {
      return;
    }
  m_cleared = true;
  if (m_channel)
    {
      m_channel->Clear ();
      m_channel = 0;
    }

  UanPhyList::iterator it = m_phyList.begin ();
  for (; it != m_phyList.end (); it++)
    {
      if (*it)
        {
          (*it)->Clear ();
          *it = 0;
        }
    }
  ArrivalList::iterator ait = m_arrivalList.begin ();
  for (; ait != m_arrivalList.end (); ait++)
    {
      ait->GetPacket () = 0;
    }
  m_phyList.clear ();
  m_arrivalList.clear ();
  m_endTxEvent.Cancel ();
}

void
UanTransducerHd::DoDispose ()
{
  Clear ();
  UanTransducer::DoDispose ();
}
TypeId
UanTransducerHd::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::UanTransducerHd")
    .SetParent<UanTransducer> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanTransducerHd> ()
    .AddAttribute ("RxGainDb",
                   "Gain in Db added to incoming signal at receiver.",
                   DoubleValue (0),
                   MakeDoubleAccessor (&UanTransducerHd::m_rxGainDb),
                   MakeDoubleChecker<double> ())
  ;
  return tid;
}

UanTransducer::State
UanTransducerHd::GetState () const
{
  return m_state;
}


bool
UanTransducerHd::IsRx (void) const
{
  return m_state == RX;
}

bool
UanTransducerHd::IsTx (void) const
{
  return m_state == TX;

}

const UanTransducer::ArrivalList &
UanTransducerHd::GetArrivalList (void) const
{
  return m_arrivalList;
}

void
UanTransducerHd::SetRxGainDb (double gainDb)
{
  m_rxGainDb = gainDb;
}

double
UanTransducerHd::GetRxGainDb (void)
{
  return m_rxGainDb;
}

double
UanTransducerHd::ApplyRxGainDb (double rxPowerDb, UanTxMode mode)
{
  NS_LOG_FUNCTION (this << rxPowerDb << mode);
  rxPowerDb += GetRxGainDb ();
  NS_LOG_DEBUG ("Rx power after RX gain = " << rxPowerDb << " db re uPa");
  return rxPowerDb;
}

void
UanTransducerHd::Receive (Ptr<Packet> packet,
                          double rxPowerDb,
                          UanTxMode txMode,
                          UanPdp pdp)
{
  NS_LOG_FUNCTION (this << packet << rxPowerDb << txMode << pdp);
  //Apply receiver gain in dB
  rxPowerDb = ApplyRxGainDb (rxPowerDb, txMode);

  UanPacketArrival arrival (packet,
                            rxPowerDb,
                            txMode,
                            pdp,
                            Simulator::Now ());

  m_arrivalList.push_back (arrival);
  Time txDelay = Seconds (packet->GetSize () * 8.0 / txMode.GetDataRateBps ());
  Simulator::Schedule (txDelay, &UanTransducerHd::RemoveArrival, this, arrival);
  NS_LOG_DEBUG (Now ().As (Time::S) << " Transducer in receive");
  if (m_state == RX)
    {
      NS_LOG_DEBUG ("Transducer state = RX");
      UanPhyList::const_iterator it = m_phyList.begin ();
      for (; it != m_phyList.end (); it++)
        {
          NS_LOG_DEBUG ("Calling StartRx");
          (*it)->StartRxPacket (packet, rxPowerDb, txMode, pdp);
        }
    }
}

void
UanTransducerHd::Transmit (Ptr<UanPhy> src,
                           Ptr<Packet> packet,
                           double txPowerDb,
                           UanTxMode txMode)
{
  if (m_state == TX)
    {
      m_endTxEvent.Cancel ();
      src->NotifyTxDrop(packet);           // traced source netanim
    }
  else
    {
      m_state = TX;
      src->NotifyTxBegin(packet);             // traced source netanim
    }


  Time delay = Seconds (packet->GetSize () * 8.0 / txMode.GetDataRateBps ());
  NS_LOG_DEBUG ("Transducer transmitting:  TX delay = "
                << delay << " seconds for packet size "
                << packet->GetSize () << " bytes and rate = "
                << txMode.GetDataRateBps () << " bps");
  UanPhyList::const_iterator it = m_phyList.begin ();
  for (; it != m_phyList.end (); it++)
    {
      if (src != (*it))
        {
          (*it)->NotifyTransStartTx (packet, txPowerDb, txMode);
        }
    }
  m_channel->TxPacket (Ptr<UanTransducer> (this), packet, txPowerDb, txMode);


  delay = std::max (delay, m_endTxTime - Simulator::Now ());

  m_endTxEvent = Simulator::Schedule (delay, &UanTransducerHd::EndTx, this);
  m_endTxTime = Simulator::Now () + delay;
  Simulator::Schedule(delay, &UanPhy::NotifyTxEnd, src, packet);    // traced source netanim
}

void
UanTransducerHd::EndTx (void)
{
  NS_ASSERT (m_state == TX);
  m_state = RX;
  m_endTxTime = Seconds (0);
}
void
UanTransducerHd::SetChannel (Ptr<UanChannel> chan)
{
  NS_LOG_DEBUG ("Transducer setting channel");
  m_channel = chan;

}
Ptr<UanChannel>
UanTransducerHd::GetChannel (void) const
{
  return m_channel;
}
void
UanTransducerHd::AddPhy (Ptr<UanPhy> phy)
{
  m_phyList.push_back (phy);
}

const UanTransducer::UanPhyList &
UanTransducerHd::GetPhyList (void) const
{
  return m_phyList;
}

void
UanTransducerHd::RemoveArrival (UanPacketArrival arrival)
{

  // Remove entry from arrival list
  ArrivalList::iterator it = m_arrivalList.begin ();
  for (; it != m_arrivalList.end (); it++)
    {
      if (it->GetPacket () == arrival.GetPacket ())
        {
          m_arrivalList.erase (it);
          break;
        }
    }
  UanPhyList::const_iterator ait = m_phyList.begin ();
  for (; ait != m_phyList.end (); ait++)
    {
      (*ait)->NotifyIntChange ();
    }

}

} // namespace ns3
