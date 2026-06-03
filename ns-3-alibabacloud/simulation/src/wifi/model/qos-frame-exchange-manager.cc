/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "qos-frame-exchange-manager.h"
#include "wifi-mac-queue.h"
#include "wifi-mac-trailer.h"
#include "ap-wifi-mac.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT std::clog << "[mac=" << m_self << "] "

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("QosFrameExchangeManager");

NS_OBJECT_ENSURE_REGISTERED (QosFrameExchangeManager);

TypeId
QosFrameExchangeManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::QosFrameExchangeManager")
    .SetParent<FrameExchangeManager> ()
    .AddConstructor<QosFrameExchangeManager> ()
    .SetGroupName ("Wifi")
    .AddAttribute ("PifsRecovery",
                   "Perform a PIFS recovery as a response to transmission failure "
                   "within a TXOP",
                   BooleanValue (true),
                   MakeBooleanAccessor (&QosFrameExchangeManager::m_pifsRecovery),
                   MakeBooleanChecker ())
    .AddAttribute ("SetQueueSize",
                   "Whether to set the Queue Size subfield of the QoS Control field "
                   "of QoS data frames sent by non-AP stations",
                   BooleanValue (false),
                   MakeBooleanAccessor (&QosFrameExchangeManager::m_setQosQueueSize),
                   MakeBooleanChecker ())
  ;
  return tid;
}

QosFrameExchangeManager::QosFrameExchangeManager ()
  : m_initialFrame (false)
{
  NS_LOG_FUNCTION (this);
}

QosFrameExchangeManager::~QosFrameExchangeManager ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
QosFrameExchangeManager::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_edca = 0;
  m_edcaBackingOff = 0;
  m_pifsRecoveryEvent.Cancel ();
  FrameExchangeManager::DoDispose ();
}

bool
QosFrameExchangeManager::SendCfEndIfNeeded (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_edca != 0);
  NS_ASSERT (m_edca->GetTxopLimit ().IsStrictlyPositive ());

  WifiMacHeader cfEnd;
  cfEnd.SetType (WIFI_MAC_CTL_END);
  cfEnd.SetDsNotFrom ();
  cfEnd.SetDsNotTo ();
  cfEnd.SetNoRetry ();
  cfEnd.SetNoMoreFragments ();
  cfEnd.SetDuration (Seconds (0));
  cfEnd.SetAddr1 (Mac48Address::GetBroadcast ());
  cfEnd.SetAddr2 (m_self);

  WifiTxVector cfEndTxVector = m_mac->GetWifiRemoteStationManager ()->GetRtsTxVector (cfEnd.GetAddr1 ());

  Time txDuration = m_phy->CalculateTxDuration (cfEnd.GetSize () + WIFI_MAC_FCS_LENGTH,
                                                cfEndTxVector, m_phy->GetPhyBand ());

  // Send the CF-End frame if the remaining duration is long enough to transmit this frame
  if (m_edca->GetRemainingTxop () > txDuration)
    {
      NS_LOG_DEBUG ("Send CF-End frame");
      m_phy->Send (Create<WifiPsdu> (Create<Packet> (), cfEnd), cfEndTxVector);
      Simulator::Schedule (txDuration, &Txop::NotifyChannelReleased, m_edca);
      return true;
    }

  m_edca->NotifyChannelReleased ();
  m_edca = 0;
  return false;
}

void
QosFrameExchangeManager::PifsRecovery (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_edca != 0);
  NS_ASSERT (m_edca->IsTxopStarted ());

  // Release the channel if it has not been idle for the last PIFS interval
  if (m_channelAccessManager->GetAccessGrantStart () - m_phy->GetSifs ()
      > Simulator::Now () - m_phy->GetPifs ())
    {
      m_edca->NotifyChannelReleased ();
      m_edca = 0;
    }
  else
    {
      // the txopDuration parameter is unused because we are not starting a new TXOP
      StartTransmission (m_edca, Seconds (0));
    }
}

void
QosFrameExchangeManager::CancelPifsRecovery (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_pifsRecoveryEvent.IsRunning ());
  NS_ASSERT (m_edca != 0);

  NS_LOG_DEBUG ("Cancel PIFS recovery being attempted by EDCAF " << m_edca);
  m_pifsRecoveryEvent.Cancel ();
  m_edca->NotifyChannelReleased ();
}

bool
QosFrameExchangeManager::StartTransmission (Ptr<Txop> edca)
{
  NS_LOG_FUNCTION (this << edca);

  if (m_pifsRecoveryEvent.IsRunning ())
    {
      // Another AC (having AIFS=1 or lower, if the user changed the default settings)
      // gained channel access while performing PIFS recovery. Abort PIFS recovery
      CancelPifsRecovery ();
    }

  // TODO This will become an assert once no Txop is installed on a QoS station
  if (!edca->IsQosTxop ())
    {
      m_edca = 0;
      return FrameExchangeManager::StartTransmission (edca);
    }

  Ptr<QosTxop> qosTxop = StaticCast<QosTxop> (edca);
  return StartTransmission (qosTxop, qosTxop->GetTxopLimit ());
}

bool
QosFrameExchangeManager::StartTransmission (Ptr<QosTxop> edca, Time txopDuration)
{
  NS_LOG_FUNCTION (this << edca << txopDuration);

  if (m_pifsRecoveryEvent.IsRunning ())
    {
      // Another AC (having AIFS=1 or lower, if the user changed the default settings)
      // gained channel access while performing PIFS recovery. Abort PIFS recovery
      CancelPifsRecovery ();
    }

  if (m_txTimer.IsRunning ())
    {
      m_txTimer.Cancel ();
    }
  m_dcf = edca;
  m_edca = edca;

  // We check if this EDCAF invoked the backoff procedure (without terminating
  // the TXOP) because the transmission of a non-initial frame of a TXOP failed
  bool backingOff = (m_edcaBackingOff == m_edca);

  if (backingOff)
    {
      NS_ASSERT (m_edca->GetTxopLimit ().IsStrictlyPositive ());
      NS_ASSERT (m_edca->IsTxopStarted ());
      NS_ASSERT (!m_pifsRecovery);
      NS_ASSERT (!m_initialFrame);

      // clear the member variable
      m_edcaBackingOff = 0;
    }

  if (m_edca->GetTxopLimit ().IsStrictlyPositive ())
    {
      // TXOP limit is not null. We have to check if this EDCAF is starting a
      // new TXOP. This includes the case when the transmission of a non-initial
      // frame of a TXOP failed and backoff was invoked without terminating the
      // TXOP. In such a case, we assume that a new TXOP is being started if it
      // elapsed more than TXOPlimit since the start of the paused TXOP. Note
      // that GetRemainingTxop returns 0 iff Now - TXOPstart >= TXOPlimit
      if (!m_edca->IsTxopStarted ()
          || (backingOff && m_edca->GetRemainingTxop ().IsZero ()))
        {
          // starting a new TXOP
          m_edca->NotifyChannelAccessed (txopDuration);

          if (StartFrameExchange (m_edca, txopDuration, true))
            {
              m_initialFrame = true;
              return true;
            }

          // TXOP not even started, return false
          NS_LOG_DEBUG ("No frame transmitted");
          m_edca->NotifyChannelReleased ();
          m_edca = 0;
          return false;
        }

      // We are continuing a TXOP, check if we can transmit another frame
      NS_ASSERT (!m_initialFrame);

      if (!StartFrameExchange (m_edca, m_edca->GetRemainingTxop (), false))
        {
          NS_LOG_DEBUG ("Not enough remaining TXOP time");
          return SendCfEndIfNeeded ();
        }

      return true;
    }

  // we get here if TXOP limit is null
  m_initialFrame = true;

  if (StartFrameExchange (m_edca, Time::Min (), true))
    {
      m_edca->NotifyChannelAccessed (Seconds (0));
      return true;
    }

  NS_LOG_DEBUG ("No frame transmitted");
  m_edca->NotifyChannelReleased ();
  m_edca = 0;
  return false;
}

bool
QosFrameExchangeManager::StartFrameExchange (Ptr<QosTxop> edca, Time availableTime, bool initialFrame)
{
  NS_LOG_FUNCTION (this << edca << availableTime << initialFrame);

  Ptr<const WifiMacQueueItem> mpdu = edca->PeekNextMpdu ();

  // Even though channel access is requested when the queue is not empty, at
  // the time channel access is granted the lifetime of the packet might be
  // expired and the queue might be empty.
  if (mpdu == 0)
    {
      NS_LOG_DEBUG ("Queue empty");
      return false;
    }

  WifiTxParameters txParams;
  txParams.m_txVector = m_mac->GetWifiRemoteStationManager ()->GetDataTxVector (mpdu->GetHeader ());

  Ptr<WifiMacQueueItem> item = edca->GetNextMpdu (mpdu, txParams, availableTime, initialFrame);

  if (item == nullptr)
    {
      NS_LOG_DEBUG ("Not enough time to transmit a frame");
      return false;
    }

  NS_ASSERT_MSG (!item->GetHeader ().IsQosData () || !item->GetHeader ().IsQosAmsdu (),
                 "We should not get an A-MSDU here");

  // check if the MSDU needs to be fragmented
  item = GetFirstFragmentIfNeeded (item);

  // update the protection method if the frame was fragmented
  if (item->IsFragment () && item->GetSize () != mpdu->GetSize ())
    {
      WifiTxParameters fragmentTxParams;
      fragmentTxParams.m_txVector = txParams.m_txVector;
      txParams.m_protection = GetProtectionManager ()->TryAddMpdu (item, fragmentTxParams);
      NS_ASSERT (txParams.m_protection != nullptr);
    }

  SendMpduWithProtection (item, txParams);

  return true;
}

bool
QosFrameExchangeManager::TryAddMpdu (Ptr<const WifiMacQueueItem> mpdu, WifiTxParameters& txParams,
                                     Time availableTime) const
{
  NS_ASSERT (mpdu != 0);
  NS_LOG_FUNCTION (this << *mpdu << &txParams << availableTime);

  // check if adding the given MPDU requires a different protection method
  Time protectionTime = Time::Min ();  // uninitialized
  if (txParams.m_protection)
    {
      protectionTime = txParams.m_protection->protectionTime;
    }

  std::unique_ptr<WifiProtection> protection;
  protection = GetProtectionManager ()->TryAddMpdu (mpdu, txParams);
  bool protectionSwapped = false;

  if (protection)
    {
      // the protection method has changed, calculate the new protection time
      CalculateProtectionTime (protection.get ());
      protectionTime = protection->protectionTime;
      // swap unique pointers, so that the txParams that is passed to the next
      // call to IsWithinLimitsIfAddMpdu is the most updated one
      txParams.m_protection.swap (protection);
      protectionSwapped = true;
    }
  NS_ASSERT (protectionTime != Time::Min ());
  NS_LOG_DEBUG ("protection time=" << protectionTime);

  // check if adding the given MPDU requires a different acknowledgment method
  Time acknowledgmentTime = Time::Min ();  // uninitialized
  if (txParams.m_acknowledgment)
    {
      acknowledgmentTime = txParams.m_acknowledgment->acknowledgmentTime;
    }

  std::unique_ptr<WifiAcknowledgment> acknowledgment;
  acknowledgment = GetAckManager ()->TryAddMpdu (mpdu, txParams);
  bool acknowledgmentSwapped = false;

  if (acknowledgment)
    {
      // the acknowledgment method has changed, calculate the new acknowledgment time
      CalculateAcknowledgmentTime (acknowledgment.get ());
      acknowledgmentTime = acknowledgment->acknowledgmentTime;
      // swap unique pointers, so that the txParams that is passed to the next
      // call to IsWithinLimitsIfAddMpdu is the most updated one
      txParams.m_acknowledgment.swap (acknowledgment);
      acknowledgmentSwapped = true;
    }
  NS_ASSERT (acknowledgmentTime != Time::Min ());
  NS_LOG_DEBUG ("acknowledgment time=" << acknowledgmentTime);

  Time ppduDurationLimit = Time::Min ();
  if (availableTime != Time::Min ())
    {
      ppduDurationLimit = availableTime - protectionTime - acknowledgmentTime;
    }

  if (!IsWithinLimitsIfAddMpdu (mpdu, txParams, ppduDurationLimit))
    {
      // adding MPDU failed, restore protection and acknowledgment methods
      // if they were swapped
      if (protectionSwapped)
        {
          txParams.m_protection.swap (protection);
        }
      if (acknowledgmentSwapped)
        {
          txParams.m_acknowledgment.swap (acknowledgment);
        }
      return false;
    }

  // the given MPDU can be added, hence update the txParams
  txParams.AddMpdu (mpdu);
  UpdateTxDuration (mpdu->GetHeader ().GetAddr1 (), txParams);

  return true;
}

bool
QosFrameExchangeManager::IsWithinLimitsIfAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                                  const WifiTxParameters& txParams,
                                                  Time ppduDurationLimit) const
{
  NS_ASSERT (mpdu != 0);
  NS_LOG_FUNCTION (this << *mpdu << &txParams << ppduDurationLimit);

  // A QoS station only has to check that the MPDU transmission time does not
  // exceed the given limit
  return IsWithinSizeAndTimeLimits (mpdu->GetSize (), mpdu->GetHeader ().GetAddr1 (),
                                    txParams, ppduDurationLimit);
}

bool
QosFrameExchangeManager::IsWithinSizeAndTimeLimits (uint32_t ppduPayloadSize, Mac48Address receiver,
                                                    const WifiTxParameters& txParams,
                                                    Time ppduDurationLimit) const
{
  NS_LOG_FUNCTION (this << ppduPayloadSize << receiver << &txParams << ppduDurationLimit);

  if (ppduDurationLimit != Time::Min () && ppduDurationLimit.IsNegative ())
    {
      NS_LOG_DEBUG ("ppduDurationLimit is null or negative, time limit is trivially exceeded");
      return false;
    }

  if (ppduPayloadSize > WifiPhy::GetMaxPsduSize (txParams.m_txVector.GetModulationClass ()))
    {
      NS_LOG_DEBUG ("the frame exceeds the max PSDU size");
      return false;
    }

  // Get the maximum PPDU Duration based on the preamble type
  Time maxPpduDuration = GetPpduMaxTime (txParams.m_txVector.GetPreambleType ());

  Time txTime = GetTxDuration (ppduPayloadSize, receiver, txParams);
  NS_LOG_DEBUG ("PPDU duration: " << txTime.As (Time::MS));

  if ((ppduDurationLimit.IsStrictlyPositive () && txTime > ppduDurationLimit)
      || (maxPpduDuration.IsStrictlyPositive () && txTime > maxPpduDuration))
    {
      NS_LOG_DEBUG ("the frame does not meet the constraint on max PPDU duration or PPDU duration limit");
      return false;
    }

  return true;
}

Time
QosFrameExchangeManager::GetFrameDurationId (const WifiMacHeader& header, uint32_t size,
                                             const WifiTxParameters& txParams,
                                             Ptr<Packet> fragmentedPacket) const
{
  NS_LOG_FUNCTION (this << header << size << &txParams << fragmentedPacket);

  // TODO This will be removed once no Txop is installed on a QoS station
  if (m_edca == 0)
    {
      return FrameExchangeManager::GetFrameDurationId (header, size, txParams, fragmentedPacket);
    }

  if (m_edca->GetTxopLimit ().IsZero ())
    {
      return FrameExchangeManager::GetFrameDurationId (header, size, txParams, fragmentedPacket);
    }

  NS_ASSERT (txParams.m_acknowledgment && txParams.m_acknowledgment->acknowledgmentTime != Time::Min ());

  // under multiple protection settings, if the TXOP limit is not null, Duration/ID
  // is set to cover the remaining TXOP time (Sec. 9.2.5.2 of 802.11-2016).
  // The TXOP holder may exceed the TXOP limit in some situations (Sec. 10.22.2.8
  // of 802.11-2016)
  return std::max (m_edca->GetRemainingTxop ()
                   - m_phy->CalculateTxDuration (size, txParams.m_txVector, m_phy->GetPhyBand ()),
                   txParams.m_acknowledgment->acknowledgmentTime);
}

Time
QosFrameExchangeManager::GetRtsDurationId (const WifiTxVector& rtsTxVector, Time txDuration, Time response) const
{
  NS_LOG_FUNCTION (this << rtsTxVector << txDuration << response);

  // TODO This will be removed once no Txop is installed on a QoS station
  if (m_edca == 0)
    {
      return FrameExchangeManager::GetRtsDurationId (rtsTxVector, txDuration, response);
    }

  if (m_edca->GetTxopLimit ().IsZero ())
    {
      return FrameExchangeManager::GetRtsDurationId (rtsTxVector, txDuration, response);
    }

  // under multiple protection settings, if the TXOP limit is not null, Duration/ID
  // is set to cover the remaining TXOP time (Sec. 9.2.5.2 of 802.11-2016).
  // The TXOP holder may exceed the TXOP limit in some situations (Sec. 10.22.2.8
  // of 802.11-2016)
  return std::max (m_edca->GetRemainingTxop ()
                   - m_phy->CalculateTxDuration (GetRtsSize (), rtsTxVector, m_phy->GetPhyBand ()),
                   Seconds (0));
}

Time
QosFrameExchangeManager::GetCtsToSelfDurationId (const WifiTxVector& ctsTxVector,
                                                  Time txDuration, Time response) const
{
  NS_LOG_FUNCTION (this << ctsTxVector << txDuration << response);

  // TODO This will be removed once no Txop is installed on a QoS station
  if (m_edca == 0)
    {
      return FrameExchangeManager::GetCtsToSelfDurationId (ctsTxVector, txDuration, response);
    }

  if (m_edca->GetTxopLimit ().IsZero ())
    {
      return FrameExchangeManager::GetCtsToSelfDurationId (ctsTxVector, txDuration, response);
    }

  // under multiple protection settings, if the TXOP limit is not null, Duration/ID
  // is set to cover the remaining TXOP time (Sec. 9.2.5.2 of 802.11-2016).
  // The TXOP holder may exceed the TXOP limit in some situations (Sec. 10.22.2.8
  // of 802.11-2016)
  return std::max (m_edca->GetRemainingTxop ()
                   - m_phy->CalculateTxDuration (GetCtsSize (), ctsTxVector, m_phy->GetPhyBand ()),
                   Seconds (0));
}

void
QosFrameExchangeManager::ForwardMpduDown (Ptr<WifiMacQueueItem> mpdu, WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << *mpdu << txVector);

  WifiMacHeader& hdr = mpdu->GetHeader ();

  if (hdr.IsQosData () && m_mac->GetTypeOfStation () == STA
      && (m_setQosQueueSize || hdr.IsQosEosp ()))
    {
      uint8_t tid = hdr.GetQosTid ();
      hdr.SetQosEosp ();
      hdr.SetQosQueueSize (m_mac->GetQosTxop (tid)->GetQosQueueSize (tid, hdr.GetAddr1 ()));
    }
  FrameExchangeManager::ForwardMpduDown (mpdu, txVector);
}

void
QosFrameExchangeManager::TransmissionSucceeded (void)
{
  NS_LOG_DEBUG (this);

  // TODO This will be removed once no Txop is installed on a QoS station
  if (m_edca == 0)
    {
      FrameExchangeManager::TransmissionSucceeded ();
      return;
    }

  if (m_edca->GetTxopLimit ().IsStrictlyPositive ()
      && m_edca->GetRemainingTxop () > m_phy->GetSifs ())
    {
      NS_LOG_DEBUG ("Schedule another transmission in a SIFS");
      bool (QosFrameExchangeManager::*fp) (Ptr<QosTxop>, Time) = &QosFrameExchangeManager::StartTransmission;

      // we are continuing a TXOP, hence the txopDuration parameter is unused
      Simulator::Schedule (m_phy->GetSifs (), fp, this, m_edca, Seconds (0));
    }
  else
    {
      m_edca->NotifyChannelReleased ();
      m_edca = 0;
    }
  m_initialFrame = false;
}

void
QosFrameExchangeManager::TransmissionFailed (void)
{
  NS_LOG_FUNCTION (this);

  // TODO This will be removed once no Txop is installed on a QoS station
  if (m_edca == 0)
    {
      FrameExchangeManager::TransmissionFailed ();
      return;
    }

  if (m_initialFrame)
    {
      // The backoff procedure shall be invoked by an EDCAF when the transmission
      // of an MPDU in the initial PPDU of a TXOP fails (Sec. 10.22.2.2 of 802.11-2016)
      NS_LOG_DEBUG ("TX of the initial frame of a TXOP failed: terminate TXOP");
      m_edca->NotifyChannelReleased ();
      m_edca = 0;
    }
  else
    {
      NS_ASSERT_MSG (m_edca->GetTxopLimit ().IsStrictlyPositive (),
                     "Cannot transmit more than one frame if TXOP Limit is zero");

      // A STA can perform a PIFS recovery or perform a backoff as a response to
      // transmission failure within a TXOP. How it chooses between these two is
      // implementation dependent. (Sec. 10.22.2.2 of 802.11-2016)
      if (m_pifsRecovery)
        {
          // we can continue the TXOP if the carrier sense mechanism indicates that
          // the medium is idle in a PIFS
          NS_LOG_DEBUG ("TX of a non-initial frame of a TXOP failed: perform PIFS recovery");
          NS_ASSERT (!m_pifsRecoveryEvent.IsRunning ());
          m_pifsRecoveryEvent = Simulator::Schedule (m_phy->GetPifs (), &QosFrameExchangeManager::PifsRecovery, this);
        }
      else
        {
          // In order not to terminate (yet) the TXOP, we call the NotifyChannelReleased
          // method of the Txop class, which only generates a new backoff value and
          // requests channel access if needed,
          NS_LOG_DEBUG ("TX of a non-initial frame of a TXOP failed: invoke backoff");
          m_edca->Txop::NotifyChannelReleased ();
          m_edcaBackingOff = m_edca;
          m_edca = 0;
        }
    }
  m_initialFrame = false;
}

void
QosFrameExchangeManager::PreProcessFrame (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << psdu << txVector);

  SetTxopHolder (psdu, txVector);

  // APs store buffer size report of associated stations
  if (m_mac->GetTypeOfStation () == AP && psdu->GetAddr1 () == m_self)
    {
      for (const auto& mpdu : *PeekPointer (psdu))
        {
          const WifiMacHeader& hdr = mpdu->GetHeader ();

          if (hdr.IsQosData () && hdr.IsQosEosp ())
            {
              NS_LOG_DEBUG ("Station " << hdr.GetAddr2 () << " reported a buffer status of "
                                      << +hdr.GetQosQueueSize () << " for tid=" << +hdr.GetQosTid ());
              StaticCast<ApWifiMac> (m_mac)->SetBufferStatus (hdr.GetQosTid (), hdr.GetAddr2 (), hdr.GetQosQueueSize ());
            }
        }
    }

  FrameExchangeManager::PreProcessFrame (psdu, txVector);
}

void
QosFrameExchangeManager::SetTxopHolder (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << psdu << txVector);

  const WifiMacHeader& hdr = psdu->GetHeader (0);

  if (hdr.IsQosData () || hdr.IsMgt () || hdr.IsRts ())
    {
      m_txopHolder = psdu->GetAddr2 ();
    }
  else if (hdr.IsCts () || hdr.IsAck ())
    {
      m_txopHolder = psdu->GetAddr1 ();
    }
}

void
QosFrameExchangeManager::ReceiveMpdu (Ptr<WifiMacQueueItem> mpdu, RxSignalInfo rxSignalInfo,
                                      const WifiTxVector& txVector, bool inAmpdu)
{
  // The received MPDU is either broadcast or addressed to this station
  NS_ASSERT (mpdu->GetHeader ().GetAddr1 ().IsGroup ()
             || mpdu->GetHeader ().GetAddr1 () == m_self);

  double rxSnr = rxSignalInfo.snr;
  const WifiMacHeader& hdr = mpdu->GetHeader ();

  if (hdr.IsCfEnd ())
    {
      // reset NAV
      NavResetTimeout ();
      return;
    }

  if (hdr.IsRts ())
    {
      NS_ABORT_MSG_IF (inAmpdu, "Received RTS as part of an A-MPDU");

      // If a non-VHT STA receives an RTS frame with the RA address matching the
      // MAC address of the STA and the MAC address in the TA field in the RTS
      // frame matches the saved TXOP holder address, then the STA shall send the
      // CTS frame after SIFS, without regard for, and without resetting, its NAV.
      // (sec. 10.22.2.4 of 802.11-2016)
      if (hdr.GetAddr2 () == m_txopHolder || m_navEnd <= Simulator::Now ())
        {
          NS_LOG_DEBUG ("Received RTS from=" << hdr.GetAddr2 () << ", schedule CTS");
          Simulator::Schedule (m_phy->GetSifs (), &QosFrameExchangeManager::SendCtsAfterRts,
                                this, hdr, txVector.GetMode (), rxSnr);
        }
      else
        {
          NS_LOG_DEBUG ("Received RTS from=" << hdr.GetAddr2 () << ", cannot schedule CTS");
        }
      return;
    }

  if (hdr.IsQosData ())
    {
      if (hdr.GetAddr1 () == m_self && hdr.GetQosAckPolicy () == WifiMacHeader::NORMAL_ACK)
        {
          NS_LOG_DEBUG ("Received " << hdr.GetTypeString () << " from=" << hdr.GetAddr2 () << ", schedule ACK");
          Simulator::Schedule (m_phy->GetSifs (), &QosFrameExchangeManager::SendNormalAck,
                               this, hdr, txVector, rxSnr);
        }

      // Forward up the frame if it is not a QoS Null frame
      if (hdr.HasData ())
        {
          m_rxMiddle->Receive (mpdu);
        }

      // the received data frame has been processed
      return;
    }

  return FrameExchangeManager::ReceiveMpdu (mpdu, rxSignalInfo, txVector, inAmpdu);
}

} //namespace ns3
