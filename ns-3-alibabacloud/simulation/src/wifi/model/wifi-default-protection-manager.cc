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
#include "wifi-default-protection-manager.h"
#include "wifi-tx-parameters.h"
#include "wifi-mac-queue-item.h"
#include "wifi-mac.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiDefaultProtectionManager");

NS_OBJECT_ENSURE_REGISTERED (WifiDefaultProtectionManager);

TypeId
WifiDefaultProtectionManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiDefaultProtectionManager")
    .SetParent<WifiProtectionManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiDefaultProtectionManager> ()
  ;
  return tid;
}

WifiDefaultProtectionManager::WifiDefaultProtectionManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiDefaultProtectionManager::~WifiDefaultProtectionManager ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::TryAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                          const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);

  // TB PPDUs need no protection (the soliciting Trigger Frame can be protected
  // by an MU-RTS). Until MU-RTS is implemented, we disable protection also for:
  // - Trigger Frames
  // - DL MU PPDUs containing more than one PSDU
  if (txParams.m_txVector.IsUlMu ()
      || mpdu->GetHeader ().IsTrigger ()
      || (txParams.m_txVector.IsDlMu () && txParams.GetPsduInfoMap ().size () > 1))
    {
      if (txParams.m_protection)
        {
          NS_ASSERT (txParams.m_protection->method == WifiProtection::NONE);
          return nullptr;
        }
      return std::unique_ptr<WifiProtection> (new WifiNoProtection);
    }

  // If we are adding a second PSDU to a DL MU PPDU, switch to no protection
  // (until MU-RTS is implemented)
  if (txParams.m_txVector.IsDlMu () && txParams.GetPsduInfoMap ().size () == 1
      && txParams.GetPsduInfo (mpdu->GetHeader ().GetAddr1 ()) == nullptr)
    {
      return std::unique_ptr<WifiProtection> (new WifiNoProtection);
    }

  // if the current protection method (if any) is already RTS/CTS or CTS-to-Self,
  // it will not change by adding an MPDU
  if (txParams.m_protection
      && (txParams.m_protection->method == WifiProtection::RTS_CTS
          || txParams.m_protection->method == WifiProtection::CTS_TO_SELF))
    {
      return nullptr;
    }

  // if a protection method is set, it must be NONE
  NS_ASSERT (!txParams.m_protection || txParams.m_protection->method == WifiProtection::NONE);

  std::unique_ptr<WifiProtection> protection;
  protection = GetPsduProtection (mpdu->GetHeader (), txParams.GetSizeIfAddMpdu (mpdu),
                                  txParams.m_txVector);

  // return the newly computed method if none was set or it is not NONE
  if (!txParams.m_protection || protection->method != WifiProtection::NONE)
    {
      return protection;
    }
  // the protection method has not changed
  return nullptr;
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                                const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *msdu << &txParams);

  // if the current protection method is already RTS/CTS or CTS-to-Self,
  // it will not change by aggregating an MSDU
  NS_ASSERT (txParams.m_protection);
  if (txParams.m_protection->method == WifiProtection::RTS_CTS
      || txParams.m_protection->method == WifiProtection::CTS_TO_SELF)
    {
      return nullptr;
    }

  NS_ASSERT (txParams.m_protection->method == WifiProtection::NONE);

  // No protection for TB PPDUs and DL MU PPDUs containing more than one PSDU
  if (txParams.m_txVector.IsUlMu ()
      || (txParams.m_txVector.IsDlMu () && txParams.GetPsduInfoMap ().size () > 1))
    {
      return nullptr;
    }

  std::unique_ptr<WifiProtection> protection;
  protection = GetPsduProtection (msdu->GetHeader (), txParams.GetSizeIfAggregateMsdu (msdu).second,
                                  txParams.m_txVector);

  // the protection method may still be none
  if (protection->method == WifiProtection::NONE)
    {
      return nullptr;
    }

  // the protection method has changed
  return protection;
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::GetPsduProtection (const WifiMacHeader& hdr, uint32_t size,
                                                 const WifiTxVector& txVector) const
{
  NS_LOG_FUNCTION (this << hdr << size << txVector);

  // a non-initial fragment does not need to be protected, unless it is being retransmitted
  if (hdr.GetFragmentNumber () > 0 && !hdr.IsRetry ())
    {
      return std::unique_ptr<WifiProtection> (new WifiNoProtection ());
    }

  // check if RTS/CTS is needed
  if (m_mac->GetWifiRemoteStationManager ()->NeedRts (hdr, size))
    {
      WifiRtsCtsProtection* protection = new WifiRtsCtsProtection;
      protection->rtsTxVector = m_mac->GetWifiRemoteStationManager ()->GetRtsTxVector (hdr.GetAddr1 ());
      protection->ctsTxVector = m_mac->GetWifiRemoteStationManager ()->GetCtsTxVector (hdr.GetAddr1 (),
                                                                                       protection->rtsTxVector.GetMode ());
      return std::unique_ptr<WifiProtection> (protection);
    }

  // check if CTS-to-Self is needed
  if (m_mac->GetWifiRemoteStationManager ()->GetUseNonErpProtection ()
      && m_mac->GetWifiRemoteStationManager ()->NeedCtsToSelf (txVector))
    {
      WifiCtsToSelfProtection* protection = new WifiCtsToSelfProtection;
      protection->ctsTxVector = m_mac->GetWifiRemoteStationManager ()->GetCtsToSelfTxVector ();
      return std::unique_ptr<WifiProtection> (protection);
    }

  return std::unique_ptr<WifiProtection> (new WifiNoProtection ());
}

} //namespace ns3
