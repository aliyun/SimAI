/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012-2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/log.h"
#include "radio-bearer-stats-calculator.h"
#include "radio-bearer-stats-connector.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RadioBearerStatsConnector");

/**
  * Less than operator for CellIdRnti, because it is used as key in map
  */
bool
operator < (const RadioBearerStatsConnector::CellIdRnti& a, const RadioBearerStatsConnector::CellIdRnti& b)
{
  return ( (a.cellId < b.cellId) || ( (a.cellId == b.cellId) && (a.rnti < b.rnti) ) );
}

/**
 * This structure is used as interface between trace
 * sources and RadioBearerStatsCalculator. It stores
 * and provides calculators with cellId and IMSI,
 * because most trace sources do not provide it.
 */
struct BoundCallbackArgument : public SimpleRefCount<BoundCallbackArgument>
{
public:
  Ptr<RadioBearerStatsCalculator> stats;  //!< statistics calculator
  uint64_t imsi; //!< imsi
  uint16_t cellId; //!< cellId
};

/**
 * Callback function for DL TX statistics for both RLC and PDCP
 * \param arg
 * \param path
 * \param rnti
 * \param lcid
 * \param packetSize
 */
void
DlTxPduCallback (Ptr<BoundCallbackArgument> arg, std::string path,
                 uint16_t rnti, uint8_t lcid, uint32_t packetSize)
{
  NS_LOG_FUNCTION (path << rnti << (uint16_t)lcid << packetSize);
  arg->stats->DlTxPdu (arg->cellId, arg->imsi, rnti, lcid, packetSize);
}

/**
 * Callback function for DL RX statistics for both RLC and PDCP
 * \param arg
 * \param path
 * \param rnti
 * \param lcid
 * \param packetSize
 * \param delay
 */
void
DlRxPduCallback (Ptr<BoundCallbackArgument> arg, std::string path,
                 uint16_t rnti, uint8_t lcid, uint32_t packetSize, uint64_t delay)
{
  NS_LOG_FUNCTION (path << rnti << (uint16_t)lcid << packetSize << delay);
  arg->stats->DlRxPdu (arg->cellId, arg->imsi, rnti, lcid, packetSize, delay);
}

/**
 * Callback function for UL TX statistics for both RLC and PDCP
 * \param arg
 * \param path
 * \param rnti
 * \param lcid
 * \param packetSize
 */
void
UlTxPduCallback (Ptr<BoundCallbackArgument> arg, std::string path,
                 uint16_t rnti, uint8_t lcid, uint32_t packetSize)
{
  NS_LOG_FUNCTION (path << rnti << (uint16_t)lcid << packetSize);
  arg->stats->UlTxPdu (arg->cellId, arg->imsi, rnti, lcid, packetSize);
}

/**
 * Callback function for UL RX statistics for both RLC and PDCP
 * \param arg
 * \param path
 * \param rnti
 * \param lcid
 * \param packetSize
 * \param delay
 */
void
UlRxPduCallback (Ptr<BoundCallbackArgument> arg, std::string path,
                 uint16_t rnti, uint8_t lcid, uint32_t packetSize, uint64_t delay)
{
  NS_LOG_FUNCTION (path << rnti << (uint16_t)lcid << packetSize << delay);
  arg->stats->UlRxPdu (arg->cellId, arg->imsi, rnti, lcid, packetSize, delay);
}


RadioBearerStatsConnector::RadioBearerStatsConnector ()
  : m_connected (false)
{
}

void
RadioBearerStatsConnector::EnableRlcStats (Ptr<RadioBearerStatsCalculator> rlcStats)
{
  m_rlcStats = rlcStats;
  EnsureConnected ();
}

void
RadioBearerStatsConnector::EnablePdcpStats (Ptr<RadioBearerStatsCalculator> pdcpStats)
{
  m_pdcpStats = pdcpStats;
  EnsureConnected ();
}

void
RadioBearerStatsConnector::EnsureConnected ()
{
  NS_LOG_FUNCTION (this);
  if (!m_connected)
    {
      Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/NewUeContext",
                       MakeBoundCallback (&RadioBearerStatsConnector::NotifyNewUeContextEnb, this));

      Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/RandomAccessSuccessful",
                       MakeBoundCallback (&RadioBearerStatsConnector::NotifyRandomAccessSuccessfulUe, this));

      Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/Srb1Created",
                       MakeBoundCallback (&RadioBearerStatsConnector::CreatedSrb1Ue, this));

      Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/DrbCreated",
                       MakeBoundCallback (&RadioBearerStatsConnector::CreatedDrbUe, this));

      m_connected = true;
    }
}

void
RadioBearerStatsConnector::StoreUeManagerPath (std::string context, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << context << cellId << rnti);
  std::string ueManagerPath;
  ueManagerPath = context.substr (0, context.rfind ("/")) + "/UeMap/" + std::to_string (rnti);
  NS_LOG_DEBUG ("ueManagerPath = " << ueManagerPath);
  CellIdRnti key;
  key.cellId = cellId;
  key.rnti = rnti;
  m_ueManagerPathByCellIdRnti[key] = ueManagerPath;

  Config::Connect (ueManagerPath + "/DrbCreated",
                   MakeBoundCallback (&RadioBearerStatsConnector::CreatedDrbEnb, this));
}

void
RadioBearerStatsConnector::NotifyNewUeContextEnb (RadioBearerStatsConnector* c, std::string context, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (c << context << cellId << rnti);
  c->StoreUeManagerPath (context, cellId, rnti);
}

void
RadioBearerStatsConnector::NotifyRandomAccessSuccessfulUe (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (c << context << imsi << cellId << rnti);
  c->ConnectTracesSrb0 (context, imsi, cellId, rnti);
}

void
RadioBearerStatsConnector::CreatedDrbEnb (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
  NS_LOG_FUNCTION (c << context << imsi << cellId << rnti << (uint16_t)lcid);
  c->ConnectTracesDrbEnb (context, imsi, cellId, rnti, lcid);
}

void
RadioBearerStatsConnector::CreatedSrb1Ue (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (c << context << imsi << cellId << rnti);
  c->ConnectTracesSrb1 (context, imsi, cellId, rnti);
}

void
RadioBearerStatsConnector::CreatedDrbUe (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
  NS_LOG_FUNCTION (c << context << imsi << cellId << rnti << (uint16_t)lcid);
  c->ConnectTracesDrbUe (context, imsi, cellId, rnti, lcid);
}

void
RadioBearerStatsConnector::ConnectTracesSrb0 (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << context << imsi << cellId << rnti);
  std::string ueRrcPath = context.substr (0, context.rfind ("/"));
  NS_LOG_LOGIC ("ueRrcPath = " << ueRrcPath);
  CellIdRnti key;
  key.cellId = cellId;
  key.rnti = rnti;
  std::map<CellIdRnti, std::string>::iterator it = m_ueManagerPathByCellIdRnti.find (key);
  NS_ASSERT (it != m_ueManagerPathByCellIdRnti.end ());
  std::string ueManagerPath = it->second;
  NS_LOG_LOGIC ("ueManagerPath = " << ueManagerPath);
  if (m_rlcStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_rlcStats;
      Config::Connect (ueRrcPath + "/Srb0/LteRlc/TxPDU",
                       MakeBoundCallback (&UlTxPduCallback, arg));
      Config::Connect (ueRrcPath + "/Srb0/LteRlc/RxPDU",
                       MakeBoundCallback (&DlRxPduCallback, arg));
      Config::Connect (ueManagerPath + "/Srb0/LteRlc/TxPDU",
                       MakeBoundCallback (&DlTxPduCallback, arg));
      Config::Connect (ueManagerPath + "/Srb0/LteRlc/RxPDU",
                       MakeBoundCallback (&UlRxPduCallback, arg));
    }
}

void
RadioBearerStatsConnector::ConnectTracesSrb1 (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << context << imsi << cellId << rnti);
  std::string ueRrcPath = context.substr (0, context.rfind ("/"));
  NS_LOG_LOGIC ("ueRrcPath = " << ueRrcPath);
  CellIdRnti key;
  key.cellId = cellId;
  key.rnti = rnti;
  std::map<CellIdRnti, std::string>::iterator it = m_ueManagerPathByCellIdRnti.find (key);
  NS_ASSERT (it != m_ueManagerPathByCellIdRnti.end ());
  std::string ueManagerPath = it->second;
  NS_LOG_LOGIC ("ueManagerPath = " << ueManagerPath);
  if (m_rlcStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_rlcStats;
      Config::Connect (ueRrcPath + "/Srb1/LteRlc/TxPDU",
                       MakeBoundCallback (&UlTxPduCallback, arg));
      Config::Connect (ueRrcPath + "/Srb1/LteRlc/RxPDU",
                       MakeBoundCallback (&DlRxPduCallback, arg));
      Config::Connect (ueManagerPath + "/Srb1/LteRlc/TxPDU",
                       MakeBoundCallback (&DlTxPduCallback, arg));
      Config::Connect (ueManagerPath + "/Srb1/LteRlc/RxPDU",
                       MakeBoundCallback (&UlRxPduCallback, arg));
    }
  if (m_pdcpStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_pdcpStats;
      Config::Connect (ueRrcPath + "/Srb1/LtePdcp/TxPDU",
                       MakeBoundCallback (&UlTxPduCallback, arg));
      Config::Connect (ueRrcPath + "/Srb1/LtePdcp/RxPDU",
                       MakeBoundCallback (&DlRxPduCallback, arg));
      Config::Connect (ueManagerPath + "/Srb1/LtePdcp/TxPDU",
                       MakeBoundCallback (&DlTxPduCallback, arg));
      Config::Connect (ueManagerPath + "/Srb1/LtePdcp/RxPDU",
                       MakeBoundCallback (&UlRxPduCallback, arg));
    }
}

void
RadioBearerStatsConnector::ConnectTracesDrbEnb (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
  NS_LOG_FUNCTION (this << context << imsi << cellId << rnti << (uint16_t)lcid);
  NS_LOG_LOGIC ("expected context should match /NodeList/*/DeviceList/*/LteEnbRrc/");
  std::string basePath;
  basePath = context.substr (0, context.rfind ("/")) + "/DataRadioBearerMap/" + std::to_string (lcid - 2);
  NS_LOG_LOGIC ("basePath = " << basePath);
  if (m_rlcStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_rlcStats;
      Config::Connect (basePath + "/LteRlc/TxPDU",
                       MakeBoundCallback (&DlTxPduCallback, arg));
      Config::Connect (basePath + "/LteRlc/RxPDU",
                       MakeBoundCallback (&UlRxPduCallback, arg));
    }
  if (m_pdcpStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_pdcpStats;
      bool foundTxPdcp = Config::ConnectFailSafe (basePath + "/LtePdcp/TxPDU",
                       MakeBoundCallback (&DlTxPduCallback, arg));
      bool foundRxPdcp = Config::ConnectFailSafe (basePath + "/LtePdcp/RxPDU",
                       MakeBoundCallback (&UlRxPduCallback, arg));
      if (!foundTxPdcp && !foundRxPdcp)
        {
          NS_LOG_WARN ("Unable to connect PDCP traces. This may happen if RlcSm is used");
        }
    }
}

void
RadioBearerStatsConnector::ConnectTracesDrbUe (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
  NS_LOG_FUNCTION (this << context << imsi << cellId << rnti << (uint16_t)lcid);
  NS_LOG_LOGIC ("expected context should match /NodeList/*/DeviceList/*/LteUeRrc/");
  std::string basePath;
  basePath = context.substr (0, context.rfind ("/")) + "/DataRadioBearerMap/" + std::to_string (lcid);
  NS_LOG_LOGIC ("basePath = " << basePath);
  if (m_rlcStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_rlcStats;
      Config::Connect (basePath + "/LteRlc/TxPDU",
                       MakeBoundCallback (&UlTxPduCallback, arg));
      Config::Connect (basePath + "/LteRlc/RxPDU",
                       MakeBoundCallback (&DlRxPduCallback, arg));
    }
  if (m_pdcpStats)
    {
      Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument> ();
      arg->imsi = imsi;
      arg->cellId = cellId;
      arg->stats = m_pdcpStats;
      bool foundTxPdcp = Config::ConnectFailSafe (basePath + "/LtePdcp/TxPDU",
                       MakeBoundCallback (&UlTxPduCallback, arg));
      bool foundRxPdcp = Config::ConnectFailSafe (basePath + "/LtePdcp/RxPDU",
                       MakeBoundCallback (&DlRxPduCallback, arg));
      if (!foundTxPdcp && !foundRxPdcp)
        {
          NS_LOG_WARN ("Unable to connect PDCP traces. This may happen if RlcSm is used");
        }
    }
}

void
RadioBearerStatsConnector::DisconnectTracesEnb (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);
  /**
   * This method is left empty to be extended in future.
   * Note: Be aware, that each of the connect method uses
   * its own BoundCallbackArgument struct as an argument
   * of the callback. And, if the code to disconnect the
   * traces would also use their own struct, the traces
   * will not disconnect, since it changes the parameter
   * of the callback.
   */
}

void
RadioBearerStatsConnector::DisconnectTracesUe (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);
  /**
   * This method is left empty to be extended in future.
   * Note: Be aware, that each of the connect method uses
   * its own BoundCallbackArgument struct as an argument
   * of the callback. And, if the code to disconnect the
   * traces would also use their own struct, the traces
   * will not disconnect, since it changes the parameter
   * of the callback.
   */
}

} // namespace ns3
