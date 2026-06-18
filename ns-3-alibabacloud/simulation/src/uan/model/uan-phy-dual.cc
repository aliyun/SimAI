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
 *         Andrea Sacco <andrea.sacco85@gmail.com>
 */

#include "uan-phy.h"
#include "uan-phy-dual.h"
#include "uan-phy-gen.h"
#include "uan-tx-mode.h"
#include "uan-net-device.h"
#include "uan-channel.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/ptr.h"
#include "ns3/traced-callback.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/simulator.h"
#include "uan-header-common.h"
#include "uan-mac-rc.h"

#include <cmath>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("UanPhyDual");

NS_OBJECT_ENSURE_REGISTERED (UanPhyDual);
NS_OBJECT_ENSURE_REGISTERED (UanPhyCalcSinrDual);

UanPhyCalcSinrDual::UanPhyCalcSinrDual ()
{

}
UanPhyCalcSinrDual::~UanPhyCalcSinrDual ()
{

}

TypeId
UanPhyCalcSinrDual::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanPhyCalcSinrDual")
    .SetParent<UanPhyCalcSinr> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanPhyCalcSinrDual> ()
  ;
  return tid;
}

double
UanPhyCalcSinrDual::CalcSinrDb (Ptr<Packet> pkt,
                                Time arrTime,
                                double rxPowerDb,
                                double ambNoiseDb,
                                UanTxMode mode,
                                UanPdp pdp,
                                const UanTransducer::ArrivalList &arrivalList) const
{

  if (mode.GetModType () != UanTxMode::OTHER)
    {
      NS_LOG_WARN ("Calculating SINR for unsupported modulation type");
    }

  double intKp = -DbToKp (rxPowerDb); // This packet is in the arrivalList
  UanTransducer::ArrivalList::const_iterator it = arrivalList.begin ();
  for (; it != arrivalList.end (); it++)
    {
      // Only count interference if there is overlap in incoming frequency
      if (std::abs ( (double) it->GetTxMode ().GetCenterFreqHz () - (double) mode.GetCenterFreqHz ())
          < (double)(it->GetTxMode ().GetBandwidthHz () / 2 + mode.GetBandwidthHz () / 2) - 0.5)
        {
          UanHeaderCommon ch, ch2;
          if (pkt)
            {
              pkt->PeekHeader (ch);
            }
          it->GetPacket ()->PeekHeader (ch2);

          if (pkt)
            {
              if (ch.GetType () == UanMacRc::TYPE_DATA)
                {
                  NS_LOG_DEBUG ("Adding interferer from " << ch2.GetSrc () << " against " << ch.GetSrc () << ": PktRxMode: "
                                                          << mode.GetName () << " Int mode: " << it->GetTxMode ().GetName () << " Separation: "
                                                          << std::abs ( (double) it->GetTxMode ().GetCenterFreqHz () - (double) mode.GetCenterFreqHz ())
                                                          << " Combined bandwidths: " << (double)(it->GetTxMode ().GetBandwidthHz () / 2 + mode.GetBandwidthHz () / 2) - 0.5);
                }
            }
          intKp += DbToKp (it->GetRxPowerDb ());
        }
    }

  double totalIntDb = KpToDb (intKp + DbToKp (ambNoiseDb));

  NS_LOG_DEBUG (Now ().As (Time::S) << " Calculating SINR:  RxPower = " << rxPowerDb << " dB.  Number of interferers = " << arrivalList.size () << "  Interference + noise power = " << totalIntDb << " dB.  SINR = " << rxPowerDb - totalIntDb << " dB.");
  return rxPowerDb - totalIntDb;
}

UanPhyDual::UanPhyDual ()
  :  UanPhy ()
{

  m_phy1 = CreateObject<UanPhyGen> ();
  m_phy2 = CreateObject<UanPhyGen> ();

  m_phy1->SetReceiveOkCallback (m_recOkCb);
  m_phy2->SetReceiveOkCallback (m_recOkCb);

  m_phy1->SetReceiveErrorCallback (m_recErrCb);
  m_phy2->SetReceiveErrorCallback (m_recErrCb);

}

UanPhyDual::~UanPhyDual ()
{
}

void
UanPhyDual::Clear ()
{
  if (m_phy1)
    {
      m_phy1->Clear ();
      m_phy1 = 0;
    }
  if (m_phy2)
    {
      m_phy2->Clear ();
      m_phy2 = 0;
    }
}
void
UanPhyDual::DoDispose ()
{
  Clear ();
  UanPhy::DoDispose ();
}

TypeId
UanPhyDual::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanPhyDual")
    .SetParent<UanPhy> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanPhyDual> ()
    .AddAttribute  ("CcaThresholdPhy1",
                    "Aggregate energy of incoming signals to move to CCA Busy state dB of Phy1.",
                    DoubleValue (10),
                    MakeDoubleAccessor (&UanPhyDual::GetCcaThresholdPhy1, &UanPhyDual::SetCcaThresholdPhy1),
                    MakeDoubleChecker<double> ())
    .AddAttribute ("CcaThresholdPhy2",
                   "Aggregate energy of incoming signals to move to CCA Busy state dB of Phy2.",
                   DoubleValue (10),
                   MakeDoubleAccessor (&UanPhyDual::GetCcaThresholdPhy2, &UanPhyDual::SetCcaThresholdPhy2),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxPowerPhy1",
                   "Transmission output power in dB of Phy1.",
                   DoubleValue (190),
                   MakeDoubleAccessor (&UanPhyDual::GetTxPowerDbPhy1, &UanPhyDual::SetTxPowerDbPhy1),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxPowerPhy2",
                   "Transmission output power in dB of Phy2.",
                   DoubleValue (190),
                   MakeDoubleAccessor (&UanPhyDual::GetTxPowerDbPhy2, &UanPhyDual::SetTxPowerDbPhy2),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("SupportedModesPhy1",
                   "List of modes supported by Phy1.",
                   UanModesListValue (UanPhyGen::GetDefaultModes ()),
                   MakeUanModesListAccessor (&UanPhyDual::GetModesPhy1, &UanPhyDual::SetModesPhy1),
                   MakeUanModesListChecker () )
    .AddAttribute ("SupportedModesPhy2",
                   "List of modes supported by Phy2.",
                   UanModesListValue (UanPhyGen::GetDefaultModes ()),
                   MakeUanModesListAccessor (&UanPhyDual::GetModesPhy2, &UanPhyDual::SetModesPhy2),
                   MakeUanModesListChecker () )
    .AddAttribute ("PerModelPhy1",
                   "Functor to calculate PER based on SINR and TxMode for Phy1.",
                   StringValue ("ns3::UanPhyPerGenDefault"),
                   MakePointerAccessor (&UanPhyDual::GetPerModelPhy1, &UanPhyDual::SetPerModelPhy1),
                   MakePointerChecker<UanPhyPer> ())
    .AddAttribute ("PerModelPhy2",
                   "Functor to calculate PER based on SINR and TxMode for Phy2.",
                   StringValue ("ns3::UanPhyPerGenDefault"),
                   MakePointerAccessor (&UanPhyDual::GetPerModelPhy2, &UanPhyDual::SetPerModelPhy2),
                   MakePointerChecker<UanPhyPer> ())
    .AddAttribute ("SinrModelPhy1",
                   "Functor to calculate SINR based on pkt arrivals and modes for Phy1.",
                   StringValue ("ns3::UanPhyCalcSinrDual"),
                   MakePointerAccessor (&UanPhyDual::GetSinrModelPhy1, &UanPhyDual::SetSinrModelPhy1),
                   MakePointerChecker<UanPhyCalcSinr> ())
    .AddAttribute ("SinrModelPhy2",
                   "Functor to calculate SINR based on pkt arrivals and modes for Phy2.",
                   StringValue ("ns3::UanPhyCalcSinrDual"),
                   MakePointerAccessor (&UanPhyDual::GetSinrModelPhy2, &UanPhyDual::SetSinrModelPhy2),
                   MakePointerChecker<UanPhyCalcSinr> ())
    .AddTraceSource ("RxOk",
                     "A packet was received successfully.",
                     MakeTraceSourceAccessor (&UanPhyDual::m_rxOkLogger),
                     "ns3::UanPhy::TracedCallback")
    .AddTraceSource ("RxError",
                     "A packet was received unsuccessfully.",
                     MakeTraceSourceAccessor (&UanPhyDual::m_rxErrLogger),
                     "ns3::UanPhy::TracedCallback")
    .AddTraceSource ("Tx",
                     "Packet transmission beginning.",
                     MakeTraceSourceAccessor (&UanPhyDual::m_txLogger),
                     "ns3::UanPhy::TracedCallback")

  ;

  return tid;
}

void
UanPhyDual::SetEnergyModelCallback (DeviceEnergyModel::ChangeStateCallback callback)
{
  NS_LOG_DEBUG ("Not Implemented");
}

void
UanPhyDual::EnergyDepletionHandler ()
{
  NS_LOG_DEBUG ("Not Implemented");
}

void
UanPhyDual::EnergyRechargeHandler ()
{
  NS_LOG_DEBUG ("Not Implemented");
}

void
UanPhyDual::SendPacket (Ptr<Packet> pkt, uint32_t modeNum)
{
  if (modeNum <= m_phy1->GetNModes () - 1)
    {
      NS_LOG_DEBUG (Now ().As (Time::S) << " Sending packet on Phy1 with mode number " << modeNum);
      m_txLogger (pkt, m_phy1->GetTxPowerDb (), m_phy1->GetMode (modeNum));
      m_phy1->SendPacket (pkt, modeNum);

    }
  else
    {
      NS_LOG_DEBUG (Now ().As (Time::S) << " Sending packet on Phy2 with mode number " << modeNum - m_phy1->GetNModes ());
      m_txLogger (pkt, m_phy2->GetTxPowerDb (), m_phy2->GetMode (modeNum - m_phy1->GetNModes ()));
      m_phy2->SendPacket (pkt, modeNum - m_phy1->GetNModes ());
    }
}
void
UanPhyDual::RegisterListener (UanPhyListener *listener)
{
  m_phy1->RegisterListener (listener);
  m_phy2->RegisterListener (listener);
}

void
UanPhyDual::StartRxPacket (Ptr<Packet> pkt, [[maybe_unused]] double rxPowerDb, UanTxMode txMode, UanPdp pdp)
{
  // Not called.  StartRxPacket in m_phy1 and m_phy2 are called directly from Transducer.
}

void
UanPhyDual::SetReceiveOkCallback (RxOkCallback cb)
{
  m_phy1->SetReceiveOkCallback (cb);
  m_phy2->SetReceiveOkCallback (cb);
}

void
UanPhyDual::SetReceiveErrorCallback (RxErrCallback cb)
{
  m_phy1->SetReceiveErrorCallback (cb);
  m_phy2->SetReceiveErrorCallback (cb);
}


void
UanPhyDual::SetTxPowerDb (double txpwr)
{
  m_phy1->SetTxPowerDb (txpwr);
  m_phy2->SetTxPowerDb (txpwr);
}

void
UanPhyDual::SetTxPowerDbPhy1 (double txpwr)
{
  m_phy1->SetTxPowerDb (txpwr);
}
void
UanPhyDual::SetTxPowerDbPhy2 (double txpwr)
{
  m_phy2->SetTxPowerDb (txpwr);
}

void
UanPhyDual::SetRxThresholdDb (double thresh)
{
  m_phy1->SetRxThresholdDb (thresh);
  m_phy2->SetRxThresholdDb (thresh);
}
void
UanPhyDual::SetCcaThresholdDb (double thresh)
{
  m_phy1->SetCcaThresholdDb (thresh);
  m_phy2->SetCcaThresholdDb (thresh);
}

void
UanPhyDual::SetCcaThresholdPhy1 (double thresh)
{
  m_phy1->SetCcaThresholdDb (thresh);
}
void
UanPhyDual::SetCcaThresholdPhy2 (double thresh)
{
  m_phy2->SetCcaThresholdDb (thresh);
}


double
UanPhyDual::GetTxPowerDb (void)
{
  NS_LOG_WARN ("Warning: Dual Phy only returns TxPowerDb of Phy 1");
  return m_phy1->GetTxPowerDb ();
}

double
UanPhyDual::GetTxPowerDbPhy1 (void) const
{
  return m_phy1->GetTxPowerDb ();
}

double
UanPhyDual::GetTxPowerDbPhy2 (void) const
{
  return m_phy2->GetTxPowerDb ();
}

double
UanPhyDual::GetRxThresholdDb (void)
{
  return m_phy1->GetRxThresholdDb ();
}

double
UanPhyDual::GetCcaThresholdDb (void)
{
  NS_LOG_WARN ("Dual Phy only returns CCAThreshold of Phy 1");
  return m_phy1->GetCcaThresholdDb ();
}
double
UanPhyDual::GetCcaThresholdPhy1 (void) const
{
  return m_phy1->GetCcaThresholdDb ();
}
double
UanPhyDual::GetCcaThresholdPhy2 (void) const
{
  return m_phy2->GetCcaThresholdDb ();
}

bool
UanPhyDual::IsPhy1Idle (void)
{
  return m_phy1->IsStateIdle ();
}
bool
UanPhyDual::IsPhy2Idle (void)
{
  return m_phy2->IsStateIdle ();
}

bool
UanPhyDual::IsPhy1Rx (void)
{
  return m_phy1->IsStateRx ();
}

bool
UanPhyDual::IsPhy2Rx (void)
{
  return m_phy2->IsStateRx ();
}

bool
UanPhyDual::IsPhy1Tx (void)
{
  return m_phy1->IsStateTx ();
}

Ptr<Packet>
UanPhyDual::GetPhy1PacketRx (void) const
{
  return m_phy1->GetPacketRx ();
}

Ptr<Packet>
UanPhyDual::GetPhy2PacketRx (void) const
{
  return m_phy2->GetPacketRx ();
}

bool
UanPhyDual::IsPhy2Tx (void)
{
  return m_phy2->IsStateTx ();
}
bool
UanPhyDual::IsStateSleep (void)
{
  return m_phy1->IsStateSleep () && m_phy2->IsStateSleep ();
}
bool
UanPhyDual::IsStateIdle (void)
{
  return m_phy1->IsStateIdle () && m_phy2->IsStateIdle ();
}
bool
UanPhyDual::IsStateBusy (void)
{
  return !IsStateIdle () || !IsStateSleep ();
}
bool
UanPhyDual::IsStateRx (void)
{
  return m_phy1->IsStateRx () || m_phy2->IsStateRx ();
}
bool
UanPhyDual::IsStateTx (void)
{
  return m_phy1->IsStateTx () || m_phy2->IsStateTx ();
}
bool
UanPhyDual::IsStateCcaBusy (void)
{
  return m_phy1->IsStateCcaBusy () || m_phy2->IsStateCcaBusy ();
}
Ptr<UanChannel>
UanPhyDual::GetChannel (void) const
{
  return m_phy1->GetChannel ();
}
Ptr<UanNetDevice>
UanPhyDual::GetDevice (void) const
{
  return m_phy1->GetDevice ();
}
void
UanPhyDual::SetChannel (Ptr<UanChannel> channel)
{
  m_phy1->SetChannel (channel);
  m_phy2->SetChannel (channel);
}
void
UanPhyDual::SetDevice (Ptr<UanNetDevice> device)
{
  m_phy1->SetDevice (device);
  m_phy2->SetDevice (device);
}
void
UanPhyDual::SetMac (Ptr<UanMac> mac)
{
  m_phy1->SetMac (mac);
  m_phy2->SetMac (mac);
}
void
UanPhyDual::NotifyTransStartTx (Ptr<Packet> packet, [[maybe_unused]] double txPowerDb, UanTxMode txMode)
{
}
void
UanPhyDual::NotifyIntChange (void)
{
  m_phy1->NotifyIntChange ();
  m_phy2->NotifyIntChange ();

}
void
UanPhyDual::SetTransducer (Ptr<UanTransducer> trans)
{
  m_phy1->SetTransducer (trans);
  m_phy2->SetTransducer (trans);
}
Ptr<UanTransducer>
UanPhyDual::GetTransducer (void)
{
  NS_LOG_WARN ("DualPhy Returning transducer of Phy1");
  return m_phy1->GetTransducer ();
}
uint32_t
UanPhyDual::GetNModes (void)
{
  return m_phy1->GetNModes () + m_phy2->GetNModes ();
}
UanTxMode
UanPhyDual::GetMode (uint32_t n)
{
  if (n < m_phy1->GetNModes ())
    {
      return m_phy1->GetMode (n);
    }
  else
    {
      return m_phy2->GetMode (n - m_phy1->GetNModes ());
    }
}

UanModesList
UanPhyDual::GetModesPhy1 (void) const
{

  UanModesListValue modeValue;
  m_phy1->GetAttribute ("SupportedModes", modeValue);
  return modeValue.Get ();
}

UanModesList
UanPhyDual::GetModesPhy2 (void) const
{
  UanModesListValue modeValue;
  m_phy2->GetAttribute ("SupportedModes", modeValue);
  return modeValue.Get ();
}

void
UanPhyDual::SetModesPhy1 (UanModesList modes)
{
  m_phy1->SetAttribute ("SupportedModes", UanModesListValue (modes));
}

void
UanPhyDual::SetModesPhy2 (UanModesList modes)
{
  m_phy2->SetAttribute ("SupportedModes", UanModesListValue (modes));
}

Ptr<UanPhyPer>
UanPhyDual::GetPerModelPhy1 (void) const
{
  PointerValue perValue;
  m_phy1->GetAttribute ("PerModel", perValue);
  return perValue;
}

Ptr<UanPhyPer>
UanPhyDual::GetPerModelPhy2 (void) const
{
  PointerValue perValue;
  m_phy2->GetAttribute ("PerModel", perValue);
  return perValue;
}

void
UanPhyDual::SetPerModelPhy1 (Ptr<UanPhyPer> per)
{
  m_phy1->SetAttribute ("PerModel", PointerValue (per));
}

void
UanPhyDual::SetPerModelPhy2 (Ptr<UanPhyPer> per)
{
  m_phy2->SetAttribute ("PerModel", PointerValue (per));
}

Ptr<UanPhyCalcSinr>
UanPhyDual::GetSinrModelPhy1 (void) const
{
  PointerValue sinrValue;
  m_phy1->GetAttribute ("SinrModel", sinrValue);
  return sinrValue;
}

Ptr<UanPhyCalcSinr>
UanPhyDual::GetSinrModelPhy2 (void) const
{
  PointerValue sinrValue;
  m_phy2->GetAttribute ("SinrModel", sinrValue);
  return sinrValue;
}

void
UanPhyDual::SetSinrModelPhy1 (Ptr<UanPhyCalcSinr> sinr)
{
  m_phy1->SetAttribute ("SinrModel", PointerValue (sinr));
}

void
UanPhyDual::SetSinrModelPhy2 (Ptr<UanPhyCalcSinr> sinr)
{
  m_phy2->SetAttribute ("SinrModel", PointerValue (sinr));
}

void
UanPhyDual::RxOkFromSubPhy (Ptr<Packet> pkt, double sinr, UanTxMode mode)
{
  NS_LOG_DEBUG (Now ().As (Time::S) << " Received packet");
  m_recOkCb (pkt, sinr, mode);
  m_rxOkLogger (pkt, sinr, mode);
}

void
UanPhyDual::RxErrFromSubPhy (Ptr<Packet> pkt, double sinr)
{
  m_recErrCb (pkt, sinr);
  m_rxErrLogger (pkt, sinr, m_phy1->GetMode (0));
}

Ptr<Packet>
UanPhyDual::GetPacketRx (void) const
{
  NS_FATAL_ERROR ("GetPacketRx not valid for UanPhyDual.  Must specify GetPhy1PacketRx or GetPhy2PacketRx");
  return Create<Packet> ();
}

int64_t
UanPhyDual::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  return 0;
}

}
