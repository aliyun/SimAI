/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#include "service-flow.h"
#include "service-flow-record.h"
#include "ns3/simulator.h"
#include "wimax-tlv.h"

namespace ns3 {

ServiceFlow::ServiceFlow (Direction direction)
{
  InitValues ();
  m_direction = direction;
  m_type = SF_TYPE_PROVISIONED;
  m_record = new ServiceFlowRecord ();
  m_sfid = 0;
  m_connection = 0;
  m_isEnabled = false;
  m_isMulticast = false;
  m_modulationType = WimaxPhy::MODULATION_TYPE_QPSK_12;
}

ServiceFlow::ServiceFlow ()
  : m_sfid (0),
    m_direction (SF_DIRECTION_DOWN),
    m_type (SF_TYPE_PROVISIONED),
    m_connection (0),
    m_isEnabled (false),
    m_record (new ServiceFlowRecord ())
{
  InitValues ();
  m_isMulticast = false;
  m_modulationType = WimaxPhy::MODULATION_TYPE_QPSK_12;

}

ServiceFlow::ServiceFlow (uint32_t sfid, Direction direction, Ptr<WimaxConnection> connection)
{
  InitValues ();
  m_record = new ServiceFlowRecord ();
  m_isEnabled = false;
  m_connection = connection;
  m_connection->SetServiceFlow (this);
  m_type = SF_TYPE_PROVISIONED;
  m_direction = direction;
  m_sfid = sfid;
  m_isMulticast = false;
  m_modulationType = WimaxPhy::MODULATION_TYPE_QPSK_12;
}

ServiceFlow::ServiceFlow (Tlv tlv)
{
  InitValues ();
  m_connection = 0;
  m_isEnabled = 0;
  m_record = new ServiceFlowRecord ();
  NS_ASSERT_MSG (tlv.GetType () == Tlv::UPLINK_SERVICE_FLOW || tlv.GetType () == Tlv::DOWNLINK_SERVICE_FLOW,
                 "Invalid TLV");

  SfVectorTlvValue * param;
  param = (SfVectorTlvValue*)(tlv.PeekValue ());

  if (tlv.GetType () == Tlv::UPLINK_SERVICE_FLOW)
    {
      m_direction = SF_DIRECTION_UP;
    }
  else
    {
      m_direction = SF_DIRECTION_DOWN;
    }

  for (std::vector<Tlv*>::const_iterator iter = param->Begin (); iter != param->End (); ++iter)
    {
      switch ((*iter)->GetType ())
        {
        case SfVectorTlvValue::SFID:
          {
            m_sfid = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::CID:
          {
            uint16_t cid = ((U16TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            m_connection = CreateObject<WimaxConnection> (cid, Cid::TRANSPORT);
            break;
          }
        case SfVectorTlvValue::QoS_Parameter_Set_Type:
          {
            m_qosParamSetType = ((U8TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Traffic_Priority:
          {
            m_trafficPriority = ((U8TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Maximum_Sustained_Traffic_Rate:
          {
            m_maxSustainedTrafficRate = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Maximum_Traffic_Burst:
          {
            m_maxTrafficBurst = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Minimum_Reserved_Traffic_Rate:
          {
            m_minReservedTrafficRate = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Minimum_Tolerable_Traffic_Rate:
          {
            m_minTolerableTrafficRate = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Service_Flow_Scheduling_Type:
          {
            m_schedulingType = (ServiceFlow::SchedulingType)((U8TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Request_Transmission_Policy:
          {
            m_requestTransmissionPolicy = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Tolerated_Jitter:
          {
            m_toleratedJitter = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Maximum_Latency:
          {
            m_maximumLatency = ((U32TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::Fixed_length_versus_Variable_length_SDU_Indicator:
          {
            m_fixedversusVariableSduIndicator = ((U8TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case SfVectorTlvValue::CS_Specification:
          {
            m_csSpecification = (enum CsSpecification)(((U8TlvValue*)((*iter)->PeekValue ()))->GetValue ());
            break;
          }

        case SfVectorTlvValue::IPV4_CS_Parameters:
          {
            m_convergenceSublayerParam = CsParameters (*(*iter));
            break;
          }

        }
    }
  m_isMulticast = false;
  m_modulationType = WimaxPhy::MODULATION_TYPE_QPSK_12;
}

ServiceFlow::~ServiceFlow (void)
{
  if (m_record != 0)
    {
      delete m_record;
      m_record = 0;
    }
  m_connection = 0;
}

void
ServiceFlow::SetDirection (Direction direction)
{
  m_direction = direction;
}

ServiceFlow::Direction
ServiceFlow::GetDirection (void) const
{
  return m_direction;
}

void
ServiceFlow::SetType (Type type)
{
  m_type = type;
}

ServiceFlow::Type
ServiceFlow::GetType (void) const
{
  return m_type;
}

void
ServiceFlow::SetConnection (Ptr<WimaxConnection> connection)
{
  m_connection = connection;
  m_connection->SetServiceFlow (this);
}

Ptr<WimaxConnection>
ServiceFlow::GetConnection (void) const
{
  return m_connection;
}

void
ServiceFlow::SetIsEnabled (bool isEnabled)
{
  m_isEnabled = isEnabled;
}

bool
ServiceFlow::GetIsEnabled (void) const
{
  return m_isEnabled;
}

void
ServiceFlow::SetRecord (ServiceFlowRecord *record)
{
  m_record = record;
}

ServiceFlowRecord*
ServiceFlow::GetRecord (void) const
{
  return m_record;
}

Ptr<WimaxMacQueue>
ServiceFlow::GetQueue (void) const
{
  if (!m_connection)
    {
      return 0;
    }
  return m_connection->GetQueue ();
}

enum ServiceFlow::SchedulingType
ServiceFlow::GetSchedulingType (void) const
{
  return m_schedulingType;
}

bool
ServiceFlow::HasPackets (void) const
{
  if (!m_connection)
    {
      return false;
    }
  return m_connection->HasPackets ();
}

bool
ServiceFlow::HasPackets (MacHeaderType::HeaderType packetType) const
{
  if (!m_connection)
    {
      return false;
    }
  return m_connection->HasPackets (packetType);
}

void
ServiceFlow::CleanUpQueue (void)
{
  GenericMacHeader hdr;
  Time timeStamp;
  Ptr<Packet> packet;
  Time currentTime = Simulator::Now ();
  if (m_connection)
    {
      while (m_connection->HasPackets ())
        {
          packet = m_connection->GetQueue ()->Peek (hdr, timeStamp);

          if (currentTime - timeStamp > MilliSeconds (GetMaximumLatency ()))
            {
              m_connection->Dequeue ();
            }
          else
            {
              break;
            }
        }
    }
}

void
ServiceFlow::PrintQoSParameters (void) const
{
}
// ==============================================================================


uint32_t
ServiceFlow::GetSfid (void) const
{
  return m_sfid;
}
uint16_t
ServiceFlow::GetCid (void) const
{
  if (m_connection == 0)
    {
      return 0;
    }
  return m_connection->GetCid ().GetIdentifier ();
}
std::string
ServiceFlow::GetServiceClassName () const
{
  return m_serviceClassName;
}
uint8_t
ServiceFlow::GetQosParamSetType (void) const
{
  return m_qosParamSetType;
}
uint8_t
ServiceFlow::GetTrafficPriority (void) const
{
  return m_trafficPriority;
}
uint32_t
ServiceFlow::GetMaxSustainedTrafficRate (void) const
{
  return m_maxSustainedTrafficRate;
}
uint32_t
ServiceFlow::GetMaxTrafficBurst (void) const
{
  return m_maxTrafficBurst;
}
uint32_t
ServiceFlow::GetMinReservedTrafficRate (void) const
{
  return m_minReservedTrafficRate;
}
uint32_t
ServiceFlow::GetMinTolerableTrafficRate (void) const
{
  return m_minTolerableTrafficRate;
}
enum
ServiceFlow::SchedulingType ServiceFlow::GetServiceSchedulingType (void) const
{
  return m_schedulingType;
}
uint32_t
ServiceFlow::GetRequestTransmissionPolicy (void) const
{
  return m_requestTransmissionPolicy;
}
uint32_t
ServiceFlow::GetToleratedJitter (void) const
{
  return m_toleratedJitter;
}
uint32_t
ServiceFlow::GetMaximumLatency (void) const
{
  return m_maximumLatency;
}
uint8_t
ServiceFlow::GetFixedversusVariableSduIndicator (void) const
{
  return m_fixedversusVariableSduIndicator;
}
uint8_t
ServiceFlow::GetSduSize (void) const
{
  return m_sduSize;
}
uint16_t
ServiceFlow::GetTargetSAID (void) const
{
  return m_targetSAID;
}
uint8_t
ServiceFlow::GetArqEnable (void) const
{
  return m_arqEnable;
}
uint16_t
ServiceFlow::GetArqWindowSize (void) const
{
  return m_arqWindowSize;
}
uint16_t
ServiceFlow::GetArqRetryTimeoutTx (void) const
{
  return m_arqRetryTimeoutTx;
}
uint16_t
ServiceFlow::GetArqRetryTimeoutRx (void) const
{
  return m_arqRetryTimeoutRx;
}

uint16_t
ServiceFlow::GetArqBlockLifeTime (void) const
{
  return m_arqBlockLifeTime;
}
uint16_t
ServiceFlow::GetArqSyncLoss (void) const
{
  return m_arqSyncLoss;
}
uint8_t
ServiceFlow::GetArqDeliverInOrder (void) const
{
  return m_arqDeliverInOrder;
}
uint16_t
ServiceFlow::GetArqPurgeTimeout (void) const
{
  return m_arqPurgeTimeout;
}
uint16_t
ServiceFlow::GetArqBlockSize (void) const
{
  return m_arqBlockSize;
}
enum
ServiceFlow::CsSpecification ServiceFlow::GetCsSpecification (void) const
{
  return m_csSpecification;
}
CsParameters
ServiceFlow::GetConvergenceSublayerParam (void) const
{
  return m_convergenceSublayerParam;
}
uint16_t
ServiceFlow::GetUnsolicitedGrantInterval (void) const
{
  return m_unsolicitedGrantInterval;
}
uint16_t
ServiceFlow::GetUnsolicitedPollingInterval (void) const
{
  return m_unsolicitedPollingInterval;
}

bool
ServiceFlow::GetIsMulticast (void) const
{
  return m_isMulticast;
}
enum WimaxPhy::ModulationType
ServiceFlow::GetModulation (void) const
{
  return m_modulationType;
}


// ==============================================================================

void
ServiceFlow::SetSfid (uint32_t sfid)
{
  m_sfid = sfid;
}
void
ServiceFlow::SetServiceClassName (std::string name)
{
  m_serviceClassName = name;
}
void
ServiceFlow::SetQosParamSetType (uint8_t type)
{
  m_qosParamSetType = type;
}
void
ServiceFlow::SetTrafficPriority (uint8_t priority)
{
  m_trafficPriority = priority;
}
void
ServiceFlow::SetMaxSustainedTrafficRate (uint32_t maxSustainedRate)
{
  m_maxSustainedTrafficRate = maxSustainedRate;
}
void
ServiceFlow::SetMaxTrafficBurst (uint32_t maxTrafficBurst)
{
  m_maxTrafficBurst = maxTrafficBurst;
}
void
ServiceFlow::SetMinReservedTrafficRate (uint32_t minResvRate)
{
  m_minReservedTrafficRate = minResvRate;
}
void
ServiceFlow::SetMinTolerableTrafficRate (uint32_t minJitter)
{
  m_minTolerableTrafficRate = minJitter;
}
void
ServiceFlow::SetServiceSchedulingType (enum ServiceFlow::SchedulingType schedType)
{
  m_schedulingType = schedType;
}
void
ServiceFlow::SetRequestTransmissionPolicy (uint32_t policy)
{
  m_requestTransmissionPolicy = policy;
}
void
ServiceFlow::SetToleratedJitter (uint32_t jitter)
{
  m_toleratedJitter = jitter;
}
void
ServiceFlow::SetMaximumLatency (uint32_t MaximumLatency)
{
  m_maximumLatency = MaximumLatency;
}
void
ServiceFlow::SetFixedversusVariableSduIndicator (uint8_t sduIndicator)
{
  m_fixedversusVariableSduIndicator = sduIndicator;
}
void
ServiceFlow::SetSduSize (uint8_t sduSize)
{
  m_sduSize = sduSize;
}
void
ServiceFlow::SetTargetSAID (uint16_t targetSaid)
{
  m_targetSAID = targetSaid;
}
void
ServiceFlow::SetArqEnable (uint8_t arqEnable)
{
  m_arqEnable = arqEnable;
}
void
ServiceFlow::SetArqWindowSize (uint16_t arqWindowSize)
{
  m_arqWindowSize = arqWindowSize;
}
void
ServiceFlow::SetArqRetryTimeoutTx (uint16_t timeout)
{
  m_arqRetryTimeoutTx = timeout;
}
void
ServiceFlow::SetArqRetryTimeoutRx (uint16_t timeout)
{
  m_arqRetryTimeoutRx = timeout;
}
void
ServiceFlow::SetArqBlockLifeTime (uint16_t lifeTime)
{
  m_arqBlockLifeTime = lifeTime;
}
void
ServiceFlow::SetArqSyncLoss (uint16_t syncLoss)
{
  m_arqSyncLoss = syncLoss;
}
void
ServiceFlow::SetArqDeliverInOrder (uint8_t inOrder)
{
  m_arqDeliverInOrder = inOrder;
}
void
ServiceFlow::SetArqPurgeTimeout (uint16_t timeout)
{
  m_arqPurgeTimeout = timeout;
}
void
ServiceFlow::SetArqBlockSize (uint16_t size)
{
  m_arqBlockSize = size;
}
void
ServiceFlow::SetCsSpecification (enum ServiceFlow::CsSpecification spec)
{
  m_csSpecification = spec;
}
void
ServiceFlow::SetConvergenceSublayerParam (CsParameters csparam)
{
  m_convergenceSublayerParam = csparam;
}
void
ServiceFlow::SetUnsolicitedGrantInterval (uint16_t unsolicitedGrantInterval)
{
  m_unsolicitedGrantInterval = unsolicitedGrantInterval;
}
void
ServiceFlow::SetUnsolicitedPollingInterval (uint16_t unsolicitedPollingInterval)
{
  m_unsolicitedPollingInterval = unsolicitedPollingInterval;
}
void
ServiceFlow::SetIsMulticast (bool isMulticast)
{
  m_isMulticast = isMulticast;
}
void
ServiceFlow::SetModulation (enum WimaxPhy::ModulationType modulationType)
{
  m_modulationType = modulationType;
}

void
ServiceFlow::InitValues (void)
{
  m_sfid = 0;
  m_serviceClassName = "";
  m_qosParamSetType = 0;
  m_trafficPriority = 0;
  m_maxSustainedTrafficRate = 0;
  m_maxTrafficBurst = 0;
  m_minReservedTrafficRate = 0;
  m_minTolerableTrafficRate = 0;
  m_schedulingType = ServiceFlow::SF_TYPE_NONE;
  m_requestTransmissionPolicy = 0;
  m_toleratedJitter = 0;
  m_maximumLatency = 0;
  m_fixedversusVariableSduIndicator = 0;
  m_sduSize = 0;
  m_targetSAID = 0;
  m_arqEnable = 0;
  m_arqWindowSize = 0;
  m_arqRetryTimeoutTx = 0;
  m_arqRetryTimeoutRx = 0;
  m_csSpecification = ServiceFlow::IPV4;
  m_unsolicitedGrantInterval = 0;
  m_unsolicitedPollingInterval = 0;
  m_arqBlockLifeTime = 0;
  m_arqSyncLoss = 0;
  m_arqDeliverInOrder = 0;
  m_arqPurgeTimeout = 0;
  m_arqBlockSize = 0;
  m_direction = ServiceFlow::SF_DIRECTION_DOWN;
  m_type = ServiceFlow::SF_TYPE_ACTIVE;
  m_isMulticast = false;
  m_modulationType = WimaxPhy::MODULATION_TYPE_QPSK_12;
}

void
ServiceFlow::CopyParametersFrom (ServiceFlow sf)
{
  m_serviceClassName = sf.GetServiceClassName ();
  m_qosParamSetType = sf.GetQosParamSetType ();
  m_trafficPriority = sf.GetTrafficPriority ();
  m_maxSustainedTrafficRate = sf.GetMaxSustainedTrafficRate ();
  m_maxTrafficBurst = sf.GetMaxTrafficBurst ();
  m_minReservedTrafficRate = sf.GetMinReservedTrafficRate ();
  m_minTolerableTrafficRate = sf.GetMinTolerableTrafficRate ();
  m_schedulingType = sf.GetServiceSchedulingType ();
  m_requestTransmissionPolicy = sf.GetRequestTransmissionPolicy ();
  m_toleratedJitter = sf.GetToleratedJitter ();
  m_maximumLatency = sf.GetMaximumLatency ();
  m_fixedversusVariableSduIndicator = sf.GetFixedversusVariableSduIndicator ();
  m_sduSize = sf.GetSduSize ();
  m_targetSAID = sf.GetTargetSAID ();
  m_arqEnable = sf.GetArqEnable ();
  m_arqWindowSize = sf.GetArqWindowSize ();
  m_arqRetryTimeoutTx = sf.GetArqRetryTimeoutTx ();
  m_arqRetryTimeoutRx = sf.GetArqRetryTimeoutRx ();
  m_csSpecification = sf.GetCsSpecification ();
  m_convergenceSublayerParam = sf.GetConvergenceSublayerParam ();
  m_unsolicitedGrantInterval = sf.GetUnsolicitedGrantInterval ();
  m_unsolicitedPollingInterval = sf.GetUnsolicitedPollingInterval ();
  m_direction = sf.GetDirection ();
  m_isMulticast = sf.GetIsMulticast ();
  m_modulationType = sf.GetModulation ();
}

ServiceFlow::ServiceFlow (const ServiceFlow & sf)
{
  m_sfid = sf.GetSfid ();
  m_serviceClassName = sf.GetServiceClassName ();
  m_qosParamSetType = sf.GetQosParamSetType ();
  m_trafficPriority = sf.GetTrafficPriority ();
  m_maxSustainedTrafficRate = sf.GetMaxSustainedTrafficRate ();
  m_maxTrafficBurst = sf.GetMaxTrafficBurst ();
  m_minReservedTrafficRate = sf.GetMinReservedTrafficRate ();
  m_minTolerableTrafficRate = sf.GetMinTolerableTrafficRate ();
  m_schedulingType = sf.GetServiceSchedulingType ();
  m_requestTransmissionPolicy = sf.GetRequestTransmissionPolicy ();
  m_toleratedJitter = sf.GetToleratedJitter ();
  m_maximumLatency = sf.GetMaximumLatency ();
  m_fixedversusVariableSduIndicator = sf.GetFixedversusVariableSduIndicator ();
  m_sduSize = sf.GetSduSize ();
  m_targetSAID = sf.GetTargetSAID ();
  m_arqEnable = sf.GetArqEnable ();
  m_arqWindowSize = sf.GetArqWindowSize ();
  m_arqRetryTimeoutTx = sf.GetArqRetryTimeoutTx ();
  m_arqRetryTimeoutRx = sf.GetArqRetryTimeoutRx ();
  m_csSpecification = sf.GetCsSpecification ();
  m_convergenceSublayerParam = sf.GetConvergenceSublayerParam ();
  m_unsolicitedGrantInterval = sf.GetUnsolicitedGrantInterval ();
  m_unsolicitedPollingInterval = sf.GetUnsolicitedPollingInterval ();
  m_direction = sf.GetDirection ();
  m_type = sf.GetType ();
  m_connection = sf.GetConnection ();
  m_isEnabled = sf.GetIsEnabled ();
  m_record = new ServiceFlowRecord ();
  (*m_record) = (*sf.GetRecord ());
  m_isMulticast = sf.GetIsMulticast ();
  m_modulationType = sf.GetModulation ();
}

ServiceFlow &
ServiceFlow::operator = (ServiceFlow const& o)
{

  m_sfid = o.GetSfid ();
  m_serviceClassName = o.GetServiceClassName ();
  m_qosParamSetType = o.GetQosParamSetType ();
  m_trafficPriority = o.GetTrafficPriority ();
  m_maxSustainedTrafficRate = o.GetMaxSustainedTrafficRate ();
  m_maxTrafficBurst = o.GetMaxTrafficBurst ();
  m_minReservedTrafficRate = o.GetMinReservedTrafficRate ();
  m_minTolerableTrafficRate = o.GetMinTolerableTrafficRate ();
  m_schedulingType = o.GetServiceSchedulingType ();
  m_requestTransmissionPolicy = o.GetRequestTransmissionPolicy ();
  m_toleratedJitter = o.GetToleratedJitter ();
  m_maximumLatency = o.GetMaximumLatency ();
  m_fixedversusVariableSduIndicator = o.GetFixedversusVariableSduIndicator ();
  m_sduSize = o.GetSduSize ();
  m_targetSAID = o.GetTargetSAID ();
  m_arqEnable = o.GetArqEnable ();
  m_arqWindowSize = o.GetArqWindowSize ();
  m_arqRetryTimeoutTx = o.GetArqRetryTimeoutTx ();
  m_arqRetryTimeoutRx = o.GetArqRetryTimeoutRx ();
  m_csSpecification = o.GetCsSpecification ();
  m_convergenceSublayerParam = o.GetConvergenceSublayerParam ();
  m_unsolicitedGrantInterval = o.GetUnsolicitedGrantInterval ();
  m_unsolicitedPollingInterval = o.GetUnsolicitedPollingInterval ();
  m_direction = o.GetDirection ();
  m_type = o.GetType ();
  m_connection = o.GetConnection ();
  m_isEnabled = o.GetIsEnabled ();
  m_isMulticast = o.GetIsMulticast ();
  m_modulationType = o.GetModulation ();
  if (m_record != 0)
    {
      delete m_record;
    }

  m_record = new ServiceFlowRecord ();

  (*m_record) = (*o.GetRecord ());
  return *this;
}

char*
ServiceFlow::GetSchedulingTypeStr (void) const
{
  switch (m_schedulingType)
    {
    case SF_TYPE_UGS:
      return (char*) "UGS";
      break;
    case SF_TYPE_RTPS:
      return (char*) "rtPS";
      break;
    case SF_TYPE_NRTPS:
      return (char*) "nrtPS";
      break;
    case SF_TYPE_BE:
      return (char*) "BE";
      break;
    default:
      NS_FATAL_ERROR ("Invalid scheduling type");
    }
  return 0;
}

Tlv
ServiceFlow::ToTlv (void) const
{
  SfVectorTlvValue tmpSfVector;
  tmpSfVector.Add (Tlv (SfVectorTlvValue::SFID, 4, U32TlvValue (m_sfid)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::CID, 2, U16TlvValue (GetCid ())));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::QoS_Parameter_Set_Type, 1, U8TlvValue (m_qosParamSetType)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Traffic_Priority, 1, U8TlvValue (m_trafficPriority)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Maximum_Sustained_Traffic_Rate, 4, U32TlvValue (m_maxSustainedTrafficRate)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Maximum_Traffic_Burst, 4, U32TlvValue (m_maxTrafficBurst)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Minimum_Reserved_Traffic_Rate, 4, U32TlvValue (m_minReservedTrafficRate)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Minimum_Tolerable_Traffic_Rate, 4, U32TlvValue (m_minTolerableTrafficRate)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Service_Flow_Scheduling_Type, 1, U8TlvValue (m_schedulingType)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Request_Transmission_Policy, 4, U32TlvValue (m_requestTransmissionPolicy)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Tolerated_Jitter, 4, U32TlvValue (m_toleratedJitter)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Maximum_Latency, 4, U32TlvValue (m_maximumLatency)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Fixed_length_versus_Variable_length_SDU_Indicator,
                        1,
                        U8TlvValue (m_fixedversusVariableSduIndicator)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::SDU_Size, 1, U8TlvValue (m_sduSize)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::Target_SAID, 2, U16TlvValue (m_targetSAID)));
  tmpSfVector.Add (Tlv (SfVectorTlvValue::CS_Specification, 1, U8TlvValue (m_csSpecification)));
  tmpSfVector.Add (m_convergenceSublayerParam.ToTlv ());
  if (m_direction == SF_DIRECTION_UP)
    {
      return Tlv (Tlv::UPLINK_SERVICE_FLOW, tmpSfVector.GetSerializedSize (), tmpSfVector);
    }
  else
    {
      return Tlv (Tlv::DOWNLINK_SERVICE_FLOW, tmpSfVector.GetSerializedSize (), tmpSfVector);
    }
}

bool
ServiceFlow::CheckClassifierMatch (Ipv4Address srcAddress,
                                   Ipv4Address dstAddress,
                                   uint16_t srcPort,
                                   uint16_t dstPort,
                                   uint8_t proto) const
{
  return m_convergenceSublayerParam.GetPacketClassifierRule ().CheckMatch (srcAddress,
                                                                           dstAddress,
                                                                           srcPort,
                                                                           dstPort,
                                                                           proto);
}
} // namespace ns3
