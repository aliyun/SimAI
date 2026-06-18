/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 *
 */

#include "lte-ue-component-carrier-manager.h"
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteUeComponentCarrierManager");
NS_OBJECT_ENSURE_REGISTERED (LteUeComponentCarrierManager);

LteUeComponentCarrierManager::LteUeComponentCarrierManager ()
: m_ccmRrcSapUser (0), m_ccmRrcSapProvider (0), m_noOfComponentCarriers(0)
{
  NS_LOG_FUNCTION (this);
}

LteUeComponentCarrierManager::~LteUeComponentCarrierManager ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
LteUeComponentCarrierManager::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteUeComponentCarrierManager")
                      .SetParent<Object> ()
                      .SetGroupName("Lte");
  return tid;
}

void
LteUeComponentCarrierManager::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

void
LteUeComponentCarrierManager::SetLteCcmRrcSapUser (LteUeCcmRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ccmRrcSapUser = s;
}

LteUeCcmRrcSapProvider*
LteUeComponentCarrierManager::GetLteCcmRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ccmRrcSapProvider;
}


bool 
LteUeComponentCarrierManager::SetComponentCarrierMacSapProviders (uint8_t componentCarrierId, LteMacSapProvider* sap)
{
  NS_LOG_FUNCTION (this);
  bool result = false;
  std::map <uint8_t, LteMacSapProvider*>::iterator it;
  it=m_macSapProvidersMap.find (componentCarrierId);
  if (componentCarrierId > m_noOfComponentCarriers)
    {
      NS_FATAL_ERROR ("Inconsistent componentCarrierId or you didn't call SetNumberOfComponentCarriers before calling this method");
    }
  if (it != m_macSapProvidersMap.end ())
    {
      NS_FATAL_ERROR ("Tried to allocated an existing componentCarrierId");
    }
  else
    {
      m_macSapProvidersMap.insert (std::pair<uint8_t, LteMacSapProvider*>(componentCarrierId, sap));
      result = true;
    }
  return result;
  
}

void
LteUeComponentCarrierManager::SetNumberOfComponentCarriers (uint8_t noOfComponentCarriers)
{
  NS_LOG_FUNCTION (this);
  NS_ABORT_MSG_IF (noOfComponentCarriers < MIN_NO_CC || noOfComponentCarriers > MAX_NO_CC, "Number of component carriers should be greater than 0 and less than 6");
  m_noOfComponentCarriers = noOfComponentCarriers;
  //Set the number of component carriers in UE RRC
  m_ccmRrcSapUser->SetNumberOfComponentCarriers(noOfComponentCarriers);
}


} // end of namespace ns3
