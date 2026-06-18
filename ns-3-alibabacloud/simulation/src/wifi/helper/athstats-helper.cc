/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/wifi-mode.h"
#include "ns3/wifi-phy-common.h"
#include "ns3/wifi-phy-state.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "athstats-helper.h"
#include <iomanip>
#include <fstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Athstats");

AthstatsHelper::AthstatsHelper ()
  : m_interval (Seconds (1.0))
{
}

void
AthstatsHelper::EnableAthstats (std::string filename,  uint32_t nodeid, uint32_t deviceid)
{
  Ptr<AthstatsWifiTraceSink> athstats = CreateObject<AthstatsWifiTraceSink> ();
  std::ostringstream oss;
  oss << filename
      << "_" << std::setfill ('0') << std::setw (3) << std::right <<  nodeid
      << "_" << std::setfill ('0') << std::setw (3) << std::right << deviceid;
  athstats->Open (oss.str ());

  oss.str ("");
  oss << "/NodeList/" << nodeid << "/DeviceList/" << deviceid;
  std::string devicepath = oss.str ();

  Config::Connect (devicepath + "/Mac/MacTx", MakeCallback (&AthstatsWifiTraceSink::DevTxTrace, athstats));
  Config::Connect (devicepath + "/Mac/MacRx", MakeCallback (&AthstatsWifiTraceSink::DevRxTrace, athstats));

  Config::Connect (devicepath + "/RemoteStationManager/MacTxRtsFailed", MakeCallback (&AthstatsWifiTraceSink::TxRtsFailedTrace, athstats));
  Config::Connect (devicepath + "/RemoteStationManager/MacTxDataFailed", MakeCallback (&AthstatsWifiTraceSink::TxDataFailedTrace, athstats));
  Config::Connect (devicepath + "/RemoteStationManager/MacTxFinalRtsFailed", MakeCallback (&AthstatsWifiTraceSink::TxFinalRtsFailedTrace, athstats));
  Config::Connect (devicepath + "/RemoteStationManager/MacTxFinalDataFailed", MakeCallback (&AthstatsWifiTraceSink::TxFinalDataFailedTrace, athstats));

  Config::Connect (devicepath + "/Phy/State/RxOk", MakeCallback (&AthstatsWifiTraceSink::PhyRxOkTrace, athstats));
  Config::Connect (devicepath + "/Phy/State/RxError", MakeCallback (&AthstatsWifiTraceSink::PhyRxErrorTrace, athstats));
  Config::Connect (devicepath + "/Phy/State/Tx", MakeCallback (&AthstatsWifiTraceSink::PhyTxTrace, athstats));
  Config::Connect (devicepath + "/Phy/State/State", MakeCallback (&AthstatsWifiTraceSink::PhyStateTrace, athstats));
}

void
AthstatsHelper::EnableAthstats (std::string filename, Ptr<NetDevice> nd)
{
  EnableAthstats (filename, nd->GetNode ()->GetId (), nd->GetIfIndex ());
}

void
AthstatsHelper::EnableAthstats (std::string filename, NetDeviceContainer d)
{
  for (NetDeviceContainer::Iterator i = d.Begin (); i != d.End (); ++i)
    {
      Ptr<NetDevice> dev = *i;
      EnableAthstats (filename, dev->GetNode ()->GetId (), dev->GetIfIndex ());
    }
}

void
AthstatsHelper::EnableAthstats (std::string filename, NodeContainer n)
{
  NetDeviceContainer devs;
  for (NodeContainer::Iterator i = n.Begin (); i != n.End (); ++i)
    {
      Ptr<Node> node = *i;
      for (std::size_t j = 0; j < node->GetNDevices (); ++j)
        {
          devs.Add (node->GetDevice (j));
        }
    }
  EnableAthstats (filename, devs);
}

NS_OBJECT_ENSURE_REGISTERED (AthstatsWifiTraceSink);

TypeId
AthstatsWifiTraceSink::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AthstatsWifiTraceSink")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AthstatsWifiTraceSink> ()
    .AddAttribute ("Interval",
                   "Time interval between reports",
                   TimeValue (Seconds (1.0)),
                   MakeTimeAccessor (&AthstatsWifiTraceSink::m_interval),
                   MakeTimeChecker ())
  ;
  return tid;
}

AthstatsWifiTraceSink::AthstatsWifiTraceSink ()
  :   m_txCount (0),
    m_rxCount (0),
    m_shortRetryCount (0),
    m_longRetryCount (0),
    m_exceededRetryCount (0),
    m_phyRxOkCount (0),
    m_phyRxErrorCount (0),
    m_phyTxCount (0),
    m_writer (0)
{
  Simulator::ScheduleNow (&AthstatsWifiTraceSink::WriteStats, this);
}

AthstatsWifiTraceSink::~AthstatsWifiTraceSink ()
{
  NS_LOG_FUNCTION (this);

  if (m_writer != 0)
    {
      NS_LOG_LOGIC ("m_writer nonzero " << m_writer);
      if (m_writer->is_open ())
        {
          NS_LOG_LOGIC ("m_writer open.  Closing " << m_writer);
          m_writer->close ();
        }

      NS_LOG_LOGIC ("Deleting writer " << m_writer);
      delete m_writer;

      NS_LOG_LOGIC ("m_writer = 0");
      m_writer = 0;
    }
  else
    {
      NS_LOG_LOGIC ("m_writer == 0");
    }
}

void
AthstatsWifiTraceSink::ResetCounters ()
{
  m_txCount = 0;
  m_rxCount = 0;
  m_shortRetryCount = 0;
  m_longRetryCount = 0;
  m_exceededRetryCount = 0;
  m_phyRxOkCount = 0;
  m_phyRxErrorCount = 0;
  m_phyTxCount = 0;
}

void
AthstatsWifiTraceSink::DevTxTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this << context << p);
  ++m_txCount;
}

void
AthstatsWifiTraceSink::DevRxTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this << context << p);
  ++m_rxCount;
}

void
AthstatsWifiTraceSink::TxRtsFailedTrace (std::string context, Mac48Address address)
{
  NS_LOG_FUNCTION (this << context << address);
  ++m_shortRetryCount;
}

void
AthstatsWifiTraceSink::TxDataFailedTrace (std::string context, Mac48Address address)
{
  NS_LOG_FUNCTION (this << context << address);
  ++m_longRetryCount;
}

void
AthstatsWifiTraceSink::TxFinalRtsFailedTrace (std::string context, Mac48Address address)
{
  NS_LOG_FUNCTION (this << context << address);
  ++m_exceededRetryCount;
}

void
AthstatsWifiTraceSink::TxFinalDataFailedTrace (std::string context, Mac48Address address)
{
  NS_LOG_FUNCTION (this << context << address);
  ++m_exceededRetryCount;
}

void
AthstatsWifiTraceSink::PhyRxOkTrace (std::string context, Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble)
{
  NS_LOG_FUNCTION (this << context << packet << " mode=" << mode << " snr=" << snr << "preamble=" << preamble);
  ++m_phyRxOkCount;
}

void
AthstatsWifiTraceSink::PhyRxErrorTrace (std::string context, Ptr<const Packet> packet, double snr)
{
  NS_LOG_FUNCTION (this << context << packet << " snr=" << snr );
  ++m_phyRxErrorCount;
}

void
AthstatsWifiTraceSink::PhyTxTrace (std::string context, Ptr<const Packet> packet, WifiMode mode, WifiPreamble preamble, uint8_t txPower)
{
  NS_LOG_FUNCTION (this << context << packet << "PHYTX mode=" << mode << "Preamble=" << preamble << "Power=" << txPower);
  ++m_phyTxCount;
}

void
AthstatsWifiTraceSink::PhyStateTrace (std::string context, Time start, Time duration, WifiPhyState state)
{
  NS_LOG_FUNCTION (this << context << start << duration << state);
}

void
AthstatsWifiTraceSink::Open (std::string const &name)
{
  NS_LOG_FUNCTION (this << name);
  NS_ABORT_MSG_UNLESS (m_writer == 0, "AthstatsWifiTraceSink::Open (): m_writer already allocated (std::ofstream leak detected)");

  m_writer = new std::ofstream ();
  NS_ABORT_MSG_UNLESS (m_writer, "AthstatsWifiTraceSink::Open (): Cannot allocate m_writer");

  NS_LOG_LOGIC ("Created writer " << m_writer);

  m_writer->open (name.c_str (), std::ios_base::binary | std::ios_base::out);
  NS_ABORT_MSG_IF (m_writer->fail (), "AthstatsWifiTraceSink::Open (): m_writer->open (" << name.c_str () << ") failed");

  NS_ASSERT_MSG (m_writer->is_open (), "AthstatsWifiTraceSink::Open (): m_writer not open");

  NS_LOG_LOGIC ("Writer opened successfully");
}

void
AthstatsWifiTraceSink::WriteStats ()
{
  NS_LOG_FUNCTION (this);
  //The comments below refer to how each value maps to madwifi's athstats
  //I know C strings are ugly but that's the quickest way to use exactly the same format as in madwifi
  char str[200];
  snprintf (str, 200, "%8u %8u %7u %7u %7u %6u %6u %6u %7u %4u %3uM\n",
            (unsigned int) m_txCount, // /proc/net/dev transmitted packets to which we should subtract management frames
            (unsigned int) m_rxCount, // /proc/net/dev received packets but subtracts management frames from it
            (unsigned int) 0, // ast_tx_altrate
            (unsigned int) m_shortRetryCount, // ast_tx_shortretry
            (unsigned int) m_longRetryCount, // ast_tx_longretry
            (unsigned int) m_exceededRetryCount, // ast_tx_xretries
            (unsigned int) m_phyRxErrorCount,    // ast_rx_crcerr
            (unsigned int) 0, // ast_rx_badcrypt
            (unsigned int) 0, // ast_rx_phyerr
            (unsigned int) 0, // ast_rx_rssi
            (unsigned int) 0 // rate
            );

  if (m_writer)
    {

      *m_writer << str;

      ResetCounters ();
      Simulator::Schedule (m_interval, &AthstatsWifiTraceSink::WriteStats, this);
    }
}

} //namespace ns3
