/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/log.h"
#include "ns3/epc-x2-header.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EpcX2Header");

NS_OBJECT_ENSURE_REGISTERED (EpcX2Header);

EpcX2Header::EpcX2Header ()
  : m_messageType (0xfa),
    m_procedureCode (0xfa),
    m_lengthOfIes (0xfa),
    m_numberOfIes (0xfa)
{
}

EpcX2Header::~EpcX2Header ()
{
  m_messageType = 0xfb;
  m_procedureCode = 0xfb;
  m_lengthOfIes = 0xfb;
  m_numberOfIes = 0xfb;
}

TypeId
EpcX2Header::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2Header")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2Header> ()
  ;
  return tid;
}

TypeId
EpcX2Header::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2Header::GetSerializedSize (void) const
{
  return 7;
}

void
EpcX2Header::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (m_messageType);
  i.WriteU8 (m_procedureCode);

  i.WriteU8 (0x00); // criticality = REJECT
  i.WriteU8 (m_lengthOfIes + 3);
  i.WriteHtonU16 (0);
  i.WriteU8 (m_numberOfIes);
}

uint32_t
EpcX2Header::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_messageType = i.ReadU8 ();
  m_procedureCode = i.ReadU8 ();

  i.ReadU8 ();
  m_lengthOfIes = i.ReadU8 () - 3;
  i.ReadNtohU16 ();
  m_numberOfIes = i.ReadU8 ();
  
  return GetSerializedSize ();
}

void
EpcX2Header::Print (std::ostream &os) const
{
  os << "MessageType=" << (uint32_t) m_messageType;
  os << " ProcedureCode=" << (uint32_t) m_procedureCode;
  os << " LengthOfIEs=" << (uint32_t) m_lengthOfIes;
  os << " NumberOfIEs=" << (uint32_t) m_numberOfIes;
}

uint8_t
EpcX2Header::GetMessageType () const
{
  return m_messageType;
}

void
EpcX2Header::SetMessageType (uint8_t messageType)
{
  m_messageType = messageType;
}

uint8_t
EpcX2Header::GetProcedureCode () const
{
  return m_procedureCode;
}

void
EpcX2Header::SetProcedureCode (uint8_t procedureCode)
{
  m_procedureCode = procedureCode;
}


void
EpcX2Header::SetLengthOfIes (uint32_t lengthOfIes)
{
  m_lengthOfIes = lengthOfIes;
}

void
EpcX2Header::SetNumberOfIes (uint32_t numberOfIes)
{
  m_numberOfIes = numberOfIes;
}

/////////////////////////////////////////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2HandoverRequestHeader);

EpcX2HandoverRequestHeader::EpcX2HandoverRequestHeader ()
  : m_numberOfIes (1 + 1 + 1 + 1),
    m_headerLength (6 + 5 + 12 + (3 + 4 + 8 + 8 + 4)),
    m_oldEnbUeX2apId (0xfffa),
    m_cause (0xfffa),
    m_targetCellId (0xfffa),
    m_mmeUeS1apId (0xfffffffa)
{
  m_erabsToBeSetupList.clear ();
}

EpcX2HandoverRequestHeader::~EpcX2HandoverRequestHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_oldEnbUeX2apId = 0xfffb;
  m_cause = 0xfffb;
  m_targetCellId = 0xfffb;
  m_mmeUeS1apId = 0xfffffffb;
  m_erabsToBeSetupList.clear ();
}

TypeId
EpcX2HandoverRequestHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2HandoverRequestHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2HandoverRequestHeader> ()
  ;
  return tid;
}

TypeId
EpcX2HandoverRequestHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2HandoverRequestHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2HandoverRequestHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (10);              // id = OLD_ENB_UE_X2AP_ID
  i.WriteU8 (0);                    // criticality = REJECT
  i.WriteU8 (2);                    // length of OLD_ENB_UE_X2AP_ID
  i.WriteHtonU16 (m_oldEnbUeX2apId);

  i.WriteHtonU16 (5);               // id = CAUSE
  i.WriteU8 (1 << 6);               // criticality = IGNORE
  i.WriteU8 (1);                    // length of CAUSE
  i.WriteU8 (m_cause);

  i.WriteHtonU16 (11);              // id = TARGET_CELLID
  i.WriteU8 (0);                    // criticality = REJECT
  i.WriteU8 (8);                    // length of TARGET_CELLID
  i.WriteHtonU32 (0x123456);        // fake PLMN
  i.WriteHtonU32 (m_targetCellId << 4);

  i.WriteHtonU16 (14);              // id = UE_CONTEXT_INFORMATION
  i.WriteU8 (0);                    // criticality = REJECT

  i.WriteHtonU32 (m_mmeUeS1apId);
  i.WriteHtonU64 (m_ueAggregateMaxBitRateDownlink);
  i.WriteHtonU64 (m_ueAggregateMaxBitRateUplink);

  std::vector <EpcX2Sap::ErabToBeSetupItem>::size_type sz = m_erabsToBeSetupList.size (); 
  i.WriteHtonU32 (sz);              // number of bearers
  for (int j = 0; j < (int) sz; j++)
    {
      i.WriteHtonU16 (m_erabsToBeSetupList [j].erabId);
      i.WriteHtonU16 (m_erabsToBeSetupList [j].erabLevelQosParameters.qci);
      i.WriteHtonU64 (m_erabsToBeSetupList [j].erabLevelQosParameters.gbrQosInfo.gbrDl);
      i.WriteHtonU64 (m_erabsToBeSetupList [j].erabLevelQosParameters.gbrQosInfo.gbrUl);
      i.WriteHtonU64 (m_erabsToBeSetupList [j].erabLevelQosParameters.gbrQosInfo.mbrDl);
      i.WriteHtonU64 (m_erabsToBeSetupList [j].erabLevelQosParameters.gbrQosInfo.mbrUl);
      i.WriteU8 (m_erabsToBeSetupList [j].erabLevelQosParameters.arp.priorityLevel);
      i.WriteU8 (m_erabsToBeSetupList [j].erabLevelQosParameters.arp.preemptionCapability);
      i.WriteU8 (m_erabsToBeSetupList [j].erabLevelQosParameters.arp.preemptionVulnerability);
      i.WriteU8 (m_erabsToBeSetupList [j].dlForwarding);
      i.WriteHtonU32 (m_erabsToBeSetupList [j].transportLayerAddress.Get ());
      i.WriteHtonU32 (m_erabsToBeSetupList [j].gtpTeid);
    }

}

uint32_t
EpcX2HandoverRequestHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_headerLength = 0;
  m_numberOfIes = 0;

  i.ReadNtohU16 ();
  i.ReadU8 ();
  i.ReadU8 ();
  m_oldEnbUeX2apId = i.ReadNtohU16 ();
  m_headerLength += 6;
  m_numberOfIes++;

  i.ReadNtohU16 ();
  i.ReadU8 ();
  i.ReadU8 ();
  m_cause = i.ReadU8 ();
  m_headerLength += 5;
  m_numberOfIes++;
  
  i.ReadNtohU16 ();
  i.ReadU8 ();
  i.ReadU8 ();
  i.ReadNtohU32 ();
  m_targetCellId = i.ReadNtohU32 () >> 4;
  m_headerLength += 12;
  m_numberOfIes++;

  i.ReadNtohU16 ();
  i.ReadU8 ();
  m_mmeUeS1apId = i.ReadNtohU32 ();
  m_ueAggregateMaxBitRateDownlink = i.ReadNtohU64 ();
  m_ueAggregateMaxBitRateUplink   = i.ReadNtohU64 ();
  int sz = i.ReadNtohU32 ();
  m_headerLength += 27;
  m_numberOfIes++;

  for (int j = 0; j < sz; j++)
    {
      EpcX2Sap::ErabToBeSetupItem erabItem;

      erabItem.erabId = i.ReadNtohU16 ();
 
      erabItem.erabLevelQosParameters = EpsBearer ((EpsBearer::Qci) i.ReadNtohU16 ());
      erabItem.erabLevelQosParameters.gbrQosInfo.gbrDl = i.ReadNtohU64 ();
      erabItem.erabLevelQosParameters.gbrQosInfo.gbrUl = i.ReadNtohU64 ();
      erabItem.erabLevelQosParameters.gbrQosInfo.mbrDl = i.ReadNtohU64 ();
      erabItem.erabLevelQosParameters.gbrQosInfo.mbrUl = i.ReadNtohU64 ();
      erabItem.erabLevelQosParameters.arp.priorityLevel = i.ReadU8 ();
      erabItem.erabLevelQosParameters.arp.preemptionCapability = i.ReadU8 ();
      erabItem.erabLevelQosParameters.arp.preemptionVulnerability = i.ReadU8 ();

      erabItem.dlForwarding = i.ReadU8 ();
      erabItem.transportLayerAddress = Ipv4Address (i.ReadNtohU32 ());
      erabItem.gtpTeid = i.ReadNtohU32 ();

      m_erabsToBeSetupList.push_back (erabItem);
      m_headerLength += 48;
    }

  return GetSerializedSize ();
}

void
EpcX2HandoverRequestHeader::Print (std::ostream &os) const
{
  os << "OldEnbUeX2apId = " << m_oldEnbUeX2apId;
  os << " Cause = " << m_cause;
  os << " TargetCellId = " << m_targetCellId;
  os << " MmeUeS1apId = " << m_mmeUeS1apId;
  os << " UeAggrMaxBitRateDownlink = " << m_ueAggregateMaxBitRateDownlink;
  os << " UeAggrMaxBitRateUplink = " << m_ueAggregateMaxBitRateUplink;
  os << " NumOfBearers = " << m_erabsToBeSetupList.size ();

  std::vector <EpcX2Sap::ErabToBeSetupItem>::size_type sz = m_erabsToBeSetupList.size ();
  if (sz > 0)
    {
      os << " [";
    }
  for (int j = 0; j < (int) sz; j++)
    {
      os << m_erabsToBeSetupList[j].erabId;
      if (j < (int) sz - 1)
        {
          os << ", ";
        }
      else
        {
          os << "]";
        }
    }
}

uint16_t
EpcX2HandoverRequestHeader::GetOldEnbUeX2apId () const
{
  return m_oldEnbUeX2apId;
}

void
EpcX2HandoverRequestHeader::SetOldEnbUeX2apId (uint16_t x2apId)
{
  m_oldEnbUeX2apId = x2apId;
}

uint16_t
EpcX2HandoverRequestHeader::GetCause () const
{
  return m_cause;
}

void
EpcX2HandoverRequestHeader::SetCause (uint16_t cause)
{
  m_cause = cause;
}

uint16_t
EpcX2HandoverRequestHeader::GetTargetCellId () const
{
  return m_targetCellId;
}

void
EpcX2HandoverRequestHeader::SetTargetCellId (uint16_t targetCellId)
{
  m_targetCellId = targetCellId;
}

uint32_t
EpcX2HandoverRequestHeader::GetMmeUeS1apId () const
{
  return m_mmeUeS1apId;
}

void
EpcX2HandoverRequestHeader::SetMmeUeS1apId (uint32_t mmeUeS1apId)
{
  m_mmeUeS1apId = mmeUeS1apId;
}

std::vector <EpcX2Sap::ErabToBeSetupItem>
EpcX2HandoverRequestHeader::GetBearers () const
{
  return m_erabsToBeSetupList;
}

void
EpcX2HandoverRequestHeader::SetBearers (std::vector <EpcX2Sap::ErabToBeSetupItem> bearers)
{
  m_headerLength += 48 * bearers.size ();
  m_erabsToBeSetupList = bearers;
}

uint64_t
EpcX2HandoverRequestHeader::GetUeAggregateMaxBitRateDownlink () const
{
  return m_ueAggregateMaxBitRateDownlink;
}

void
EpcX2HandoverRequestHeader::SetUeAggregateMaxBitRateDownlink (uint64_t bitRate)
{
  m_ueAggregateMaxBitRateDownlink = bitRate;
}

uint64_t
EpcX2HandoverRequestHeader::GetUeAggregateMaxBitRateUplink () const
{
  return m_ueAggregateMaxBitRateUplink;
}

void
EpcX2HandoverRequestHeader::SetUeAggregateMaxBitRateUplink (uint64_t bitRate)
{
  m_ueAggregateMaxBitRateUplink = bitRate;
}

uint32_t
EpcX2HandoverRequestHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2HandoverRequestHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

/////////////////////////////////////////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2HandoverRequestAckHeader);

EpcX2HandoverRequestAckHeader::EpcX2HandoverRequestAckHeader ()
  : m_numberOfIes (1 + 1 + 1 + 1),
    m_headerLength (2 + 2 + 4 + 4),
    m_oldEnbUeX2apId (0xfffa),
    m_newEnbUeX2apId (0xfffa)
{
}

EpcX2HandoverRequestAckHeader::~EpcX2HandoverRequestAckHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_oldEnbUeX2apId = 0xfffb;
  m_newEnbUeX2apId = 0xfffb;
  m_erabsAdmittedList.clear ();
  m_erabsNotAdmittedList.clear ();
}

TypeId
EpcX2HandoverRequestAckHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2HandoverRequestAckHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2HandoverRequestAckHeader> ()
  ;
  return tid;
}

TypeId
EpcX2HandoverRequestAckHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2HandoverRequestAckHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2HandoverRequestAckHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (m_oldEnbUeX2apId);
  i.WriteHtonU16 (m_newEnbUeX2apId);

  std::vector <EpcX2Sap::ErabAdmittedItem>::size_type sz = m_erabsAdmittedList.size (); 
  i.WriteHtonU32 (sz);
  for (int j = 0; j < (int) sz; j++)
    {
      i.WriteHtonU16 (m_erabsAdmittedList [j].erabId);
      i.WriteHtonU32 (m_erabsAdmittedList [j].ulGtpTeid);
      i.WriteHtonU32 (m_erabsAdmittedList [j].dlGtpTeid);
    }

  std::vector <EpcX2Sap::ErabNotAdmittedItem>::size_type sz2 = m_erabsNotAdmittedList.size (); 
  i.WriteHtonU32 (sz2);
  for (int j = 0; j < (int) sz2; j++)
    {
      i.WriteHtonU16 (m_erabsNotAdmittedList [j].erabId);
      i.WriteHtonU16 (m_erabsNotAdmittedList [j].cause);
    }
}

uint32_t
EpcX2HandoverRequestAckHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_headerLength = 0;
  m_numberOfIes = 0;

  m_oldEnbUeX2apId = i.ReadNtohU16 ();
  m_newEnbUeX2apId = i.ReadNtohU16 ();
  m_headerLength += 4;
  m_numberOfIes += 2;

  int sz = i.ReadNtohU32 ();
  m_headerLength += 4;
  m_numberOfIes++;

  for (int j = 0; j < sz; j++)
    {
      EpcX2Sap::ErabAdmittedItem erabItem;

      erabItem.erabId = i.ReadNtohU16 ();
      erabItem.ulGtpTeid = i.ReadNtohU32 ();
      erabItem.dlGtpTeid = i.ReadNtohU32 ();

      m_erabsAdmittedList.push_back (erabItem);
      m_headerLength += 10;
    }

  sz = i.ReadNtohU32 ();
  m_headerLength += 4;
  m_numberOfIes++;

  for (int j = 0; j < sz; j++)
    {
      EpcX2Sap::ErabNotAdmittedItem erabItem;

      erabItem.erabId = i.ReadNtohU16 ();
      erabItem.cause  = i.ReadNtohU16 ();

      m_erabsNotAdmittedList.push_back (erabItem);
      m_headerLength += 4;
    }

  return GetSerializedSize ();
}

void
EpcX2HandoverRequestAckHeader::Print (std::ostream &os) const
{
  os << "OldEnbUeX2apId=" << m_oldEnbUeX2apId;
  os << " NewEnbUeX2apId=" << m_newEnbUeX2apId;

  os << " AdmittedBearers=" << m_erabsAdmittedList.size ();
  std::vector <EpcX2Sap::ErabAdmittedItem>::size_type sz = m_erabsAdmittedList.size ();
  if (sz > 0)
    {
      os << " [";
    }
  for (int j = 0; j < (int) sz; j++)
    {
      os << m_erabsAdmittedList[j].erabId;
      if (j < (int) sz - 1)
        {
          os << ", ";
        }
      else
        {
          os << "]";
        }
    }
  
  os << " NotAdmittedBearers=" << m_erabsNotAdmittedList.size ();
  std::vector <EpcX2Sap::ErabNotAdmittedItem>::size_type sz2 = m_erabsNotAdmittedList.size ();
  if (sz2 > 0)
    {
      os << " [";
    }
  for (int j = 0; j < (int) sz2; j++)
    {
      os << m_erabsNotAdmittedList[j].erabId;
      if (j < (int) sz2 - 1)
        {
          os << ", ";
        }
      else
        {
          os << "]";
        }
    }

}

uint16_t
EpcX2HandoverRequestAckHeader::GetOldEnbUeX2apId () const
{
  return m_oldEnbUeX2apId;
}

void
EpcX2HandoverRequestAckHeader::SetOldEnbUeX2apId (uint16_t x2apId)
{
  m_oldEnbUeX2apId = x2apId;
}

uint16_t
EpcX2HandoverRequestAckHeader::GetNewEnbUeX2apId () const
{
  return m_newEnbUeX2apId;
}

void
EpcX2HandoverRequestAckHeader::SetNewEnbUeX2apId (uint16_t x2apId)
{
  m_newEnbUeX2apId = x2apId;
}

std::vector <EpcX2Sap::ErabAdmittedItem> 
EpcX2HandoverRequestAckHeader::GetAdmittedBearers () const
{
  return m_erabsAdmittedList;
}

void
EpcX2HandoverRequestAckHeader::SetAdmittedBearers (std::vector <EpcX2Sap::ErabAdmittedItem> bearers)
{
  m_headerLength += 10 * bearers.size ();
  m_erabsAdmittedList = bearers;
}

std::vector <EpcX2Sap::ErabNotAdmittedItem>
EpcX2HandoverRequestAckHeader::GetNotAdmittedBearers () const
{
  return m_erabsNotAdmittedList;
}

void
EpcX2HandoverRequestAckHeader::SetNotAdmittedBearers (std::vector <EpcX2Sap::ErabNotAdmittedItem> bearers)
{
  m_headerLength += 4 * bearers.size ();
  m_erabsNotAdmittedList = bearers;
}

uint32_t
EpcX2HandoverRequestAckHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2HandoverRequestAckHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

/////////////////////////////////////////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2HandoverPreparationFailureHeader);

EpcX2HandoverPreparationFailureHeader::EpcX2HandoverPreparationFailureHeader ()
  : m_numberOfIes (1 + 1 + 1),
    m_headerLength (2 + 2 + 2),
    m_oldEnbUeX2apId (0xfffa),
    m_cause (0xfffa),
    m_criticalityDiagnostics (0xfffa)
{
}

EpcX2HandoverPreparationFailureHeader::~EpcX2HandoverPreparationFailureHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_oldEnbUeX2apId = 0xfffb;
  m_cause = 0xfffb;
  m_criticalityDiagnostics = 0xfffb;
}

TypeId
EpcX2HandoverPreparationFailureHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2HandoverPreparationFailureHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2HandoverPreparationFailureHeader> ()
  ;
  return tid;
}

TypeId
EpcX2HandoverPreparationFailureHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2HandoverPreparationFailureHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2HandoverPreparationFailureHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (m_oldEnbUeX2apId);
  i.WriteHtonU16 (m_cause);
  i.WriteHtonU16 (m_criticalityDiagnostics);
}

uint32_t
EpcX2HandoverPreparationFailureHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_oldEnbUeX2apId = i.ReadNtohU16 ();
  m_cause = i.ReadNtohU16 ();
  m_criticalityDiagnostics = i.ReadNtohU16 ();

  m_headerLength = 6;
  m_numberOfIes = 3;

  return GetSerializedSize ();
}

void
EpcX2HandoverPreparationFailureHeader::Print (std::ostream &os) const
{
  os << "OldEnbUeX2apId = " << m_oldEnbUeX2apId;
  os << " Cause = " << m_cause;
  os << " CriticalityDiagnostics = " << m_criticalityDiagnostics;
}

uint16_t
EpcX2HandoverPreparationFailureHeader::GetOldEnbUeX2apId () const
{
  return m_oldEnbUeX2apId;
}

void
EpcX2HandoverPreparationFailureHeader::SetOldEnbUeX2apId (uint16_t x2apId)
{
  m_oldEnbUeX2apId = x2apId;
}

uint16_t
EpcX2HandoverPreparationFailureHeader::GetCause () const
{
  return m_cause;
}

void
EpcX2HandoverPreparationFailureHeader::SetCause (uint16_t cause)
{
  m_cause = cause;
}

uint16_t
EpcX2HandoverPreparationFailureHeader::GetCriticalityDiagnostics () const
{
  return m_criticalityDiagnostics;
}

void
EpcX2HandoverPreparationFailureHeader::SetCriticalityDiagnostics (uint16_t criticalityDiagnostics)
{
  m_criticalityDiagnostics = criticalityDiagnostics;
}

uint32_t
EpcX2HandoverPreparationFailureHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2HandoverPreparationFailureHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

/////////////////////////////////////////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2SnStatusTransferHeader);

EpcX2SnStatusTransferHeader::EpcX2SnStatusTransferHeader ()
  : m_numberOfIes (3),
    m_headerLength (6),
    m_oldEnbUeX2apId (0xfffa),
    m_newEnbUeX2apId (0xfffa)
{
  m_erabsSubjectToStatusTransferList.clear (); 
}

EpcX2SnStatusTransferHeader::~EpcX2SnStatusTransferHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_oldEnbUeX2apId = 0xfffb;
  m_newEnbUeX2apId = 0xfffb;
  m_erabsSubjectToStatusTransferList.clear (); 
}

TypeId
EpcX2SnStatusTransferHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2SnStatusTransferHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2SnStatusTransferHeader> ()
  ;
  return tid;
}

TypeId
EpcX2SnStatusTransferHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2SnStatusTransferHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2SnStatusTransferHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (m_oldEnbUeX2apId);
  i.WriteHtonU16 (m_newEnbUeX2apId);

  std::vector <EpcX2Sap::ErabsSubjectToStatusTransferItem>::size_type sz = m_erabsSubjectToStatusTransferList.size ();
  i.WriteHtonU16 (sz);              // number of ErabsSubjectToStatusTransferItems

  for (int j = 0; j < (int) sz; j++)
    {
      EpcX2Sap::ErabsSubjectToStatusTransferItem item = m_erabsSubjectToStatusTransferList [j];

      i.WriteHtonU16 (item.erabId);

      uint16_t bitsetSize = EpcX2Sap::m_maxPdcpSn / 64;
      for (int k = 0; k < bitsetSize; k++)
        {
          uint64_t statusValue = 0;
          for (int m = 0; m < 64; m++)
            {
              statusValue |= item.receiveStatusOfUlPdcpSdus[64 * k + m] << m;
            }
          i.WriteHtonU64 (statusValue);
        }

      i.WriteHtonU16 (item.ulPdcpSn);
      i.WriteHtonU32 (item.ulHfn);
      i.WriteHtonU16 (item.dlPdcpSn);
      i.WriteHtonU32 (item.dlHfn);
    }
}

uint32_t
EpcX2SnStatusTransferHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_oldEnbUeX2apId = i.ReadNtohU16 ();
  m_newEnbUeX2apId = i.ReadNtohU16 ();
  int sz = i.ReadNtohU16 ();

  m_numberOfIes = 3;
  m_headerLength = 6 + sz * (14 + (EpcX2Sap::m_maxPdcpSn / 64));

  for (int j = 0; j < sz; j++)
    {
      EpcX2Sap::ErabsSubjectToStatusTransferItem ErabItem;
      ErabItem.erabId = i.ReadNtohU16 ();

      uint16_t bitsetSize = EpcX2Sap::m_maxPdcpSn / 64;
      for (int k = 0; k < bitsetSize; k++)
        {
          uint64_t statusValue = i.ReadNtohU64 ();
          for (int m = 0; m < 64; m++)
            {
              ErabItem.receiveStatusOfUlPdcpSdus[64 * k + m] = (statusValue >> m) & 1;
            }
        }

      ErabItem.ulPdcpSn = i.ReadNtohU16 ();
      ErabItem.ulHfn    = i.ReadNtohU32 ();
      ErabItem.dlPdcpSn = i.ReadNtohU16 ();
      ErabItem.dlHfn    = i.ReadNtohU32 ();

      m_erabsSubjectToStatusTransferList.push_back (ErabItem);
    }

  return GetSerializedSize ();
}

void
EpcX2SnStatusTransferHeader::Print (std::ostream &os) const
{
  os << "OldEnbUeX2apId = " << m_oldEnbUeX2apId;
  os << " NewEnbUeX2apId = " << m_newEnbUeX2apId;
  os << " ErabsSubjectToStatusTransferList size = " << m_erabsSubjectToStatusTransferList.size ();

  std::vector <EpcX2Sap::ErabsSubjectToStatusTransferItem>::size_type sz = m_erabsSubjectToStatusTransferList.size ();
  if (sz > 0)
    {
      os << " [";
    }
  for (int j = 0; j < (int) sz; j++)
    {
      os << m_erabsSubjectToStatusTransferList[j].erabId;
      if (j < (int) sz - 1)
        {
          os << ", ";
        }
      else
        {
          os << "]";
        }
    }
}

uint16_t
EpcX2SnStatusTransferHeader::GetOldEnbUeX2apId () const
{
  return m_oldEnbUeX2apId;
}

void
EpcX2SnStatusTransferHeader::SetOldEnbUeX2apId (uint16_t x2apId)
{
  m_oldEnbUeX2apId = x2apId;
}

uint16_t
EpcX2SnStatusTransferHeader::GetNewEnbUeX2apId () const
{
  return m_newEnbUeX2apId;
}

void
EpcX2SnStatusTransferHeader::SetNewEnbUeX2apId (uint16_t x2apId)
{
  m_newEnbUeX2apId = x2apId;
}

std::vector <EpcX2Sap::ErabsSubjectToStatusTransferItem>
EpcX2SnStatusTransferHeader::GetErabsSubjectToStatusTransferList () const
{
  return m_erabsSubjectToStatusTransferList;
}

void
EpcX2SnStatusTransferHeader::SetErabsSubjectToStatusTransferList (std::vector <EpcX2Sap::ErabsSubjectToStatusTransferItem> erabs)
{
  m_headerLength += erabs.size () * (14 + (EpcX2Sap::m_maxPdcpSn / 8));
  m_erabsSubjectToStatusTransferList = erabs;
}

uint32_t
EpcX2SnStatusTransferHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2SnStatusTransferHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

/////////////////////////////////////////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2UeContextReleaseHeader);

EpcX2UeContextReleaseHeader::EpcX2UeContextReleaseHeader ()
  : m_numberOfIes (1 + 1),
    m_headerLength (2 + 2),
    m_oldEnbUeX2apId (0xfffa),
    m_newEnbUeX2apId (0xfffa)
{
}

EpcX2UeContextReleaseHeader::~EpcX2UeContextReleaseHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_oldEnbUeX2apId = 0xfffb;
  m_newEnbUeX2apId = 0xfffb;
}

TypeId
EpcX2UeContextReleaseHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2UeContextReleaseHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2UeContextReleaseHeader> ()
  ;
  return tid;
}

TypeId
EpcX2UeContextReleaseHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2UeContextReleaseHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2UeContextReleaseHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (m_oldEnbUeX2apId);
  i.WriteHtonU16 (m_newEnbUeX2apId);
}

uint32_t
EpcX2UeContextReleaseHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_oldEnbUeX2apId = i.ReadNtohU16 ();
  m_newEnbUeX2apId = i.ReadNtohU16 ();
  m_numberOfIes = 2;
  m_headerLength = 4;

  return GetSerializedSize ();
}

void
EpcX2UeContextReleaseHeader::Print (std::ostream &os) const
{
  os << "OldEnbUeX2apId=" << m_oldEnbUeX2apId;
  os << " NewEnbUeX2apId=" << m_newEnbUeX2apId;
}

uint16_t
EpcX2UeContextReleaseHeader::GetOldEnbUeX2apId () const
{
  return m_oldEnbUeX2apId;
}

void
EpcX2UeContextReleaseHeader::SetOldEnbUeX2apId (uint16_t x2apId)
{
  m_oldEnbUeX2apId = x2apId;
}

uint16_t
EpcX2UeContextReleaseHeader::GetNewEnbUeX2apId () const
{
  return m_newEnbUeX2apId;
}

void
EpcX2UeContextReleaseHeader::SetNewEnbUeX2apId (uint16_t x2apId)
{
  m_newEnbUeX2apId = x2apId;
}

uint32_t
EpcX2UeContextReleaseHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2UeContextReleaseHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

/////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2LoadInformationHeader);

EpcX2LoadInformationHeader::EpcX2LoadInformationHeader ()
  : m_numberOfIes (1),
    m_headerLength (6)
{
  m_cellInformationList.clear ();
}

EpcX2LoadInformationHeader::~EpcX2LoadInformationHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_cellInformationList.clear ();
}

TypeId
EpcX2LoadInformationHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2LoadInformationHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2LoadInformationHeader> ()
  ;
  return tid;
}

TypeId
EpcX2LoadInformationHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2LoadInformationHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2LoadInformationHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (6);               // id = CELL_INFORMATION
  i.WriteU8 (1 << 6);               // criticality = IGNORE
  i.WriteU8 (4);                    // length of CELL_INFORMATION_ID

  std::vector <EpcX2Sap::CellInformationItem>::size_type sz = m_cellInformationList.size ();
  i.WriteHtonU16 (sz);              // number of cellInformationItems

  for (int j = 0; j < (int) sz; j++)
    {
      i.WriteHtonU16 (m_cellInformationList [j].sourceCellId);

      std::vector <EpcX2Sap::UlInterferenceOverloadIndicationItem>::size_type sz2;
      sz2 = m_cellInformationList [j].ulInterferenceOverloadIndicationList.size ();
      i.WriteHtonU16 (sz2);         // number of UlInterferenceOverloadIndicationItem

      for (int k = 0; k < (int) sz2; k++)
        {
          i.WriteU8 (m_cellInformationList [j].ulInterferenceOverloadIndicationList [k]);
        }

      std::vector <EpcX2Sap::UlHighInterferenceInformationItem>::size_type sz3;
      sz3 = m_cellInformationList [j].ulHighInterferenceInformationList.size ();
      i.WriteHtonU16 (sz3);         // number of UlHighInterferenceInformationItem

      for (int k = 0; k < (int) sz3; k++)
        {
          i.WriteHtonU16 (m_cellInformationList [j].ulHighInterferenceInformationList [k].targetCellId);

          std::vector <bool>::size_type sz4;
          sz4 = m_cellInformationList [j].ulHighInterferenceInformationList [k].ulHighInterferenceIndicationList.size ();
          i.WriteHtonU16 (sz4);

          for (int m = 0; m < (int) sz4; m++)
            {
              i.WriteU8 (m_cellInformationList [j].ulHighInterferenceInformationList [k].ulHighInterferenceIndicationList [m]);
            }
        }

      std::vector <bool>::size_type sz5;
      sz5 = m_cellInformationList [j].relativeNarrowbandTxBand.rntpPerPrbList.size ();
      i.WriteHtonU16 (sz5);

      for (int k = 0; k < (int) sz5; k++)
        {
          i.WriteU8 (m_cellInformationList [j].relativeNarrowbandTxBand.rntpPerPrbList [k]);
        }

      i.WriteHtonU16 (m_cellInformationList [j].relativeNarrowbandTxBand.rntpThreshold);
      i.WriteHtonU16 (m_cellInformationList [j].relativeNarrowbandTxBand.antennaPorts);
      i.WriteHtonU16 (m_cellInformationList [j].relativeNarrowbandTxBand.pB);
      i.WriteHtonU16 (m_cellInformationList [j].relativeNarrowbandTxBand.pdcchInterferenceImpact);
    }
}

uint32_t
EpcX2LoadInformationHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_headerLength = 0;
  m_numberOfIes = 0;

  i.ReadNtohU16 ();
  i.ReadU8 ();
  i.ReadU8 ();
  int sz = i.ReadNtohU16 ();
  m_headerLength += 6;
  m_numberOfIes++;

  for (int j = 0; j < sz; j++)
    {
      EpcX2Sap::CellInformationItem cellInfoItem;
      cellInfoItem.sourceCellId = i.ReadNtohU16 ();
      m_headerLength += 2;

      int sz2 = i.ReadNtohU16 ();
      m_headerLength += 2;
      for (int k = 0; k < sz2; k++)
        {
          EpcX2Sap::UlInterferenceOverloadIndicationItem item = (EpcX2Sap::UlInterferenceOverloadIndicationItem) i.ReadU8 ();
          cellInfoItem.ulInterferenceOverloadIndicationList.push_back (item);
        }
      m_headerLength += sz2;

      int sz3 = i.ReadNtohU16 ();
      m_headerLength += 2;
      for (int k = 0; k < sz3; k++)
        {
          EpcX2Sap::UlHighInterferenceInformationItem item;
          item.targetCellId = i.ReadNtohU16 ();
          m_headerLength += 2;

          int sz4 = i.ReadNtohU16 ();
          m_headerLength += 2;
          for (int m = 0; m < sz4; m++)
            {
              item.ulHighInterferenceIndicationList.push_back (i.ReadU8 ());
            }
          m_headerLength += sz4;

          cellInfoItem.ulHighInterferenceInformationList.push_back (item);
        }

      int sz5 = i.ReadNtohU16 ();
      m_headerLength += 2;
      for (int k = 0; k < sz5; k++)
        {
          cellInfoItem.relativeNarrowbandTxBand.rntpPerPrbList.push_back (i.ReadU8 ());
        }
      m_headerLength += sz5;

      cellInfoItem.relativeNarrowbandTxBand.rntpThreshold = i.ReadNtohU16 ();
      cellInfoItem.relativeNarrowbandTxBand.antennaPorts = i.ReadNtohU16 ();
      cellInfoItem.relativeNarrowbandTxBand.pB = i.ReadNtohU16 ();
      cellInfoItem.relativeNarrowbandTxBand.pdcchInterferenceImpact = i.ReadNtohU16 ();
      m_headerLength += 8;

      m_cellInformationList.push_back (cellInfoItem);
    }

  return GetSerializedSize ();
}

void
EpcX2LoadInformationHeader::Print (std::ostream &os) const
{
  os << "NumOfCellInformationItems=" << m_cellInformationList.size ();
}

std::vector <EpcX2Sap::CellInformationItem>
EpcX2LoadInformationHeader::GetCellInformationList () const
{
  return m_cellInformationList;
}

void
EpcX2LoadInformationHeader::SetCellInformationList (std::vector <EpcX2Sap::CellInformationItem> cellInformationList)
{
  m_cellInformationList = cellInformationList;
  m_headerLength += 2;

  std::vector <EpcX2Sap::CellInformationItem>::size_type sz = m_cellInformationList.size ();
  for (int j = 0; j < (int) sz; j++)
    {
      m_headerLength += 2;

      std::vector <EpcX2Sap::UlInterferenceOverloadIndicationItem>::size_type sz2;
      sz2 = m_cellInformationList [j].ulInterferenceOverloadIndicationList.size ();
      m_headerLength += 2 + sz2;

      std::vector <EpcX2Sap::UlHighInterferenceInformationItem>::size_type sz3;
      sz3 = m_cellInformationList [j].ulHighInterferenceInformationList.size ();
      m_headerLength += 2;

      for (int k = 0; k < (int) sz3; k++)
        {
          std::vector <bool>::size_type sz4;
          sz4 = m_cellInformationList [j].ulHighInterferenceInformationList [k].ulHighInterferenceIndicationList.size ();
          m_headerLength += 2 + 2 + sz4;
        }

      std::vector <bool>::size_type sz5;
      sz5 = m_cellInformationList [j].relativeNarrowbandTxBand.rntpPerPrbList.size ();
      m_headerLength += 2 + sz5 + 8;
    }
}

uint32_t
EpcX2LoadInformationHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2LoadInformationHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

////////////////

NS_OBJECT_ENSURE_REGISTERED (EpcX2ResourceStatusUpdateHeader);

EpcX2ResourceStatusUpdateHeader::EpcX2ResourceStatusUpdateHeader ()
  : m_numberOfIes (3),
    m_headerLength (6),
    m_enb1MeasurementId (0xfffa),
    m_enb2MeasurementId (0xfffa)
{
  m_cellMeasurementResultList.clear ();
}

EpcX2ResourceStatusUpdateHeader::~EpcX2ResourceStatusUpdateHeader ()
{
  m_numberOfIes = 0;
  m_headerLength = 0;
  m_enb1MeasurementId = 0xfffb;
  m_enb2MeasurementId = 0xfffb;
  m_cellMeasurementResultList.clear ();
}

TypeId
EpcX2ResourceStatusUpdateHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcX2ResourceStatusUpdateHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<EpcX2ResourceStatusUpdateHeader> ()
  ;
  return tid;
}

TypeId
EpcX2ResourceStatusUpdateHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
EpcX2ResourceStatusUpdateHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void
EpcX2ResourceStatusUpdateHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (m_enb1MeasurementId);
  i.WriteHtonU16 (m_enb2MeasurementId);

  std::vector <EpcX2Sap::CellMeasurementResultItem>::size_type sz = m_cellMeasurementResultList.size ();
  i.WriteHtonU16 (sz);              // number of CellMeasurementResultItem

  for (int j = 0; j < (int) sz; j++)
    {
      EpcX2Sap::CellMeasurementResultItem item = m_cellMeasurementResultList [j];

      i.WriteHtonU16 (item.sourceCellId);
      i.WriteU8 (item.dlHardwareLoadIndicator);
      i.WriteU8 (item.ulHardwareLoadIndicator);
      i.WriteU8 (item.dlS1TnlLoadIndicator);
      i.WriteU8 (item.ulS1TnlLoadIndicator);

      i.WriteHtonU16 (item.dlGbrPrbUsage);
      i.WriteHtonU16 (item.ulGbrPrbUsage);
      i.WriteHtonU16 (item.dlNonGbrPrbUsage);
      i.WriteHtonU16 (item.ulNonGbrPrbUsage);
      i.WriteHtonU16 (item.dlTotalPrbUsage);
      i.WriteHtonU16 (item.ulTotalPrbUsage);

      i.WriteHtonU16 (item.dlCompositeAvailableCapacity.cellCapacityClassValue);
      i.WriteHtonU16 (item.dlCompositeAvailableCapacity.capacityValue);
      i.WriteHtonU16 (item.ulCompositeAvailableCapacity.cellCapacityClassValue);
      i.WriteHtonU16 (item.ulCompositeAvailableCapacity.capacityValue);
    }
}

uint32_t
EpcX2ResourceStatusUpdateHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_enb1MeasurementId = i.ReadNtohU16 ();
  m_enb2MeasurementId = i.ReadNtohU16 ();

  int sz = i.ReadNtohU16 ();
  for (int j = 0; j < sz; j++)
    {
      EpcX2Sap::CellMeasurementResultItem item;

      item.sourceCellId = i.ReadNtohU16 ();
      item.dlHardwareLoadIndicator = (EpcX2Sap::LoadIndicator) i.ReadU8 ();
      item.ulHardwareLoadIndicator = (EpcX2Sap::LoadIndicator) i.ReadU8 ();
      item.dlS1TnlLoadIndicator = (EpcX2Sap::LoadIndicator) i.ReadU8 ();
      item.ulS1TnlLoadIndicator = (EpcX2Sap::LoadIndicator) i.ReadU8 ();

      item.dlGbrPrbUsage = i.ReadNtohU16 ();
      item.ulGbrPrbUsage = i.ReadNtohU16 ();
      item.dlNonGbrPrbUsage = i.ReadNtohU16 ();
      item.ulNonGbrPrbUsage = i.ReadNtohU16 ();
      item.dlTotalPrbUsage = i.ReadNtohU16 ();
      item.ulTotalPrbUsage = i.ReadNtohU16 ();

      item.dlCompositeAvailableCapacity.cellCapacityClassValue = i.ReadNtohU16 ();
      item.dlCompositeAvailableCapacity.capacityValue = i.ReadNtohU16 ();
      item.ulCompositeAvailableCapacity.cellCapacityClassValue = i.ReadNtohU16 ();
      item.ulCompositeAvailableCapacity.capacityValue = i.ReadNtohU16 ();

      m_cellMeasurementResultList.push_back (item);
    }

  m_headerLength = 6 + sz * 26;
  m_numberOfIes = 3;

  return GetSerializedSize ();
}

void
EpcX2ResourceStatusUpdateHeader::Print (std::ostream &os) const
{
  os << "Enb1MeasurementId = " << m_enb1MeasurementId
     << " Enb2MeasurementId = " << m_enb2MeasurementId
     << " NumOfCellMeasurementResultItems = " << m_cellMeasurementResultList.size ();
}

uint16_t
EpcX2ResourceStatusUpdateHeader::GetEnb1MeasurementId () const
{
  return m_enb1MeasurementId;
}

void
EpcX2ResourceStatusUpdateHeader::SetEnb1MeasurementId (uint16_t enb1MeasurementId)
{
  m_enb1MeasurementId = enb1MeasurementId;
}

uint16_t
EpcX2ResourceStatusUpdateHeader::GetEnb2MeasurementId () const
{
  return m_enb2MeasurementId;
}

void
EpcX2ResourceStatusUpdateHeader::SetEnb2MeasurementId (uint16_t enb2MeasurementId)
{
  m_enb2MeasurementId = enb2MeasurementId;
}

std::vector <EpcX2Sap::CellMeasurementResultItem>
EpcX2ResourceStatusUpdateHeader::GetCellMeasurementResultList () const
{
  return m_cellMeasurementResultList;
}

void
EpcX2ResourceStatusUpdateHeader::SetCellMeasurementResultList (std::vector <EpcX2Sap::CellMeasurementResultItem> cellMeasurementResultList)
{
  m_cellMeasurementResultList = cellMeasurementResultList;

  std::vector <EpcX2Sap::CellMeasurementResultItem>::size_type sz = m_cellMeasurementResultList.size ();
  m_headerLength += sz * 26;
}

uint32_t
EpcX2ResourceStatusUpdateHeader::GetLengthOfIes () const
{
  return m_headerLength;
}

uint32_t
EpcX2ResourceStatusUpdateHeader::GetNumberOfIes () const
{
  return m_numberOfIes;
}

} // namespace ns3
