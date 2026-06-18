/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/string.h"
#include "ns3/config.h"
#include "ns3/names.h"
#include "ns3/abort.h"
#include "ns3/wave-net-device.h"
#include "ns3/qos-txop.h"
#include "ns3/minstrel-wifi-manager.h"
#include "ns3/radiotap-header.h"
#include "wave-mac-helper.h"
#include "wave-helper.h"

NS_LOG_COMPONENT_DEFINE ("WaveHelper");

namespace ns3 {

/**
 * ASCII Phy transmit sink with context
 * \param stream the output stream
 * \param context the context
 * \param p the packet
 * \param mode the mode
 * \param preamble the preamble
 * \param txLevel transmit level
 */
static void
AsciiPhyTransmitSinkWithContext (
  Ptr<OutputStreamWrapper> stream,
  std::string context,
  Ptr<const Packet> p,
  WifiMode mode,
  WifiPreamble preamble,
  uint8_t txLevel)
{
  NS_LOG_FUNCTION (stream << context << p << mode << preamble << txLevel);
  *stream->GetStream () << "t " << Simulator::Now ().GetSeconds () << " " << context << " " << *p << std::endl;
}

/**
 * ASCII Phy transmit sink without context
 * \param stream the output stream
 * \param p the packet
 * \param mode the mode
 * \param preamble the preamble
 * \param txLevel transmit level
 */
static void
AsciiPhyTransmitSinkWithoutContext (
  Ptr<OutputStreamWrapper> stream,
  Ptr<const Packet> p,
  WifiMode mode,
  WifiPreamble preamble,
  uint8_t txLevel)
{
  NS_LOG_FUNCTION (stream << p << mode << preamble << txLevel);
  *stream->GetStream () << "t " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}

/**
 * ASCII Phy receive sink with context
 * \param stream the output stream
 * \param context the context
 * \param p the packet
 * \param snr the SNR
 * \param mode the mode
 * \param preamble the preamble
 */
static void
AsciiPhyReceiveSinkWithContext (
  Ptr<OutputStreamWrapper> stream,
  std::string context,
  Ptr<const Packet> p,
  double snr,
  WifiMode mode,
  enum WifiPreamble preamble)
{
  NS_LOG_FUNCTION (stream << context << p << snr << mode << preamble);
  *stream->GetStream () << "r " << Simulator::Now ().GetSeconds () << " " << context << " " << *p << std::endl;
}

/**
 * ASCII Phy receive sink without context
 * \param stream the output stream
 * \param p the packet
 * \param snr the SNR
 * \param mode the mode
 * \param preamble the preamble
 */
static void
AsciiPhyReceiveSinkWithoutContext (
  Ptr<OutputStreamWrapper> stream,
  Ptr<const Packet> p,
  double snr,
  WifiMode mode,
  enum WifiPreamble preamble)
{
  NS_LOG_FUNCTION (stream << p << snr << mode << preamble);
  *stream->GetStream () << "r " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}


/****************************** YansWavePhyHelper ***********************************/
YansWavePhyHelper
YansWavePhyHelper::Default (void)
{
  YansWavePhyHelper helper;
  helper.SetErrorRateModel ("ns3::NistErrorRateModel");
  return helper;
}

void
YansWavePhyHelper::EnablePcapInternal (std::string prefix, Ptr<NetDevice> nd, bool promiscuous, bool explicitFilename)
{
  //
  // All of the Pcap enable functions vector through here including the ones
  // that are wandering through all of devices on perhaps all of the nodes in
  // the system.  We can only deal with devices of type WaveNetDevice.
  //
  Ptr<WaveNetDevice> device = nd->GetObject<WaveNetDevice> ();
  if (device == 0)
    {
      NS_LOG_INFO ("YansWavePhyHelper::EnablePcapInternal(): Device " << &device << " not of type ns3::WaveNetDevice");
      return;
    }

  std::vector<Ptr<WifiPhy> > phys = device->GetPhys ();
  NS_ABORT_MSG_IF (phys.size () == 0, "EnablePcapInternal(): Phy layer in WaveNetDevice must be set");

  PcapHelper pcapHelper;

  std::string filename;
  if (explicitFilename)
    {
      filename = prefix;
    }
  else
    {
      filename = pcapHelper.GetFilenameFromDevice (prefix, device);
    }

  Ptr<PcapFileWrapper> file = pcapHelper.CreateFile (filename, std::ios::out, GetPcapDataLinkType ());

  std::vector<Ptr<WifiPhy> >::iterator i;
  for (i = phys.begin (); i != phys.end (); ++i)
    {
      Ptr<WifiPhy> phy = (*i);
      phy->TraceConnectWithoutContext ("MonitorSnifferTx", MakeBoundCallback (&YansWavePhyHelper::PcapSniffTxEvent, file));
      phy->TraceConnectWithoutContext ("MonitorSnifferRx", MakeBoundCallback (&YansWavePhyHelper::PcapSniffRxEvent, file));
    }
}

void
YansWavePhyHelper::EnableAsciiInternal (
  Ptr<OutputStreamWrapper> stream,
  std::string prefix,
  Ptr<NetDevice> nd,
  bool explicitFilename)
{
  //
  // All of the ascii enable functions vector through here including the ones
  // that are wandering through all of devices on perhaps all of the nodes in
  // the system.  We can only deal with devices of type WaveNetDevice.
  //
  Ptr<WaveNetDevice> device = nd->GetObject<WaveNetDevice> ();
  if (device == 0)
    {
      NS_LOG_INFO ("EnableAsciiInternal(): Device " << device << " not of type ns3::WaveNetDevice");
      return;
    }

  //
  // Our trace sinks are going to use packet printing, so we have to make sure
  // that is turned on.
  //
  Packet::EnablePrinting ();

  uint32_t nodeid = nd->GetNode ()->GetId ();
  uint32_t deviceid = nd->GetIfIndex ();
  std::ostringstream oss;

  //
  // If we are not provided an OutputStreamWrapper, we are expected to create
  // one using the usual trace filename conventions and write our traces
  // without a context since there will be one file per context and therefore
  // the context would be redundant.
  //
  if (stream == 0)
    {
      //
      // Set up an output stream object to deal with private ofstream copy
      // constructor and lifetime issues.  Let the helper decide the actual
      // name of the file given the prefix.
      //
      AsciiTraceHelper asciiTraceHelper;

      std::string filename;
      if (explicitFilename)
        {
          filename = prefix;
        }
      else
        {
          filename = asciiTraceHelper.GetFilenameFromDevice (prefix, device);
        }

      Ptr<OutputStreamWrapper> theStream = asciiTraceHelper.CreateFileStream (filename);
      //
      // We could go poking through the phy and the state looking for the
      // correct trace source, but we can let Config deal with that with
      // some search cost.  Since this is presumably happening at topology
      // creation time, it doesn't seem much of a price to pay.
      //
      oss.str ("");
      oss << "/NodeList/" << nodeid << "/DeviceList/" << deviceid << "/$ns3::WaveNetDevice/PhyEntities/*/$ns3::WifiPhy/State/RxOk";
      Config::ConnectWithoutContext (oss.str (), MakeBoundCallback (&AsciiPhyReceiveSinkWithoutContext, theStream));

      oss.str ("");
      oss << "/NodeList/" << nodeid << "/DeviceList/" << deviceid << "/$ns3::WaveNetDevice/PhyEntities/*/$ns3::WifiPhy/State/Tx";
      Config::ConnectWithoutContext (oss.str (), MakeBoundCallback (&AsciiPhyTransmitSinkWithoutContext, theStream));

      return;
    }

  //
  // If we are provided an OutputStreamWrapper, we are expected to use it, and
  // to provide a context.  We are free to come up with our own context if we
  // want, and use the AsciiTraceHelper Hook*WithContext functions, but for
  // compatibility and simplicity, we just use Config::Connect and let it deal
  // with coming up with a context.
  //
  oss.str ("");
  oss << "/NodeList/" << nodeid << "/DeviceList/" << deviceid << "/$ns3::WaveNetDevice/PhyEntities/*/$ns3::WifiPhy/RxOk";
  Config::Connect (oss.str (), MakeBoundCallback (&AsciiPhyReceiveSinkWithContext, stream));

  oss.str ("");
  oss << "/NodeList/" << nodeid << "/DeviceList/" << deviceid << "/$ns3::WaveNetDevice/PhyEntities/*/$ns3::WifiPhy/State/Tx";
  Config::Connect (oss.str (), MakeBoundCallback (&AsciiPhyTransmitSinkWithContext, stream));
}

/********************************** WaveHelper ******************************************/
WaveHelper::WaveHelper ()
{
}

WaveHelper::~WaveHelper ()
{
}

WaveHelper
WaveHelper::Default (void)
{
  WaveHelper helper;
  // default 7 MAC entities and single PHY device.
  helper.CreateMacForChannel (ChannelManager::GetWaveChannels ());
  helper.CreatePhys (1);
  helper.SetChannelScheduler ("ns3::DefaultChannelScheduler");
  helper.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                  "DataMode", StringValue ("OfdmRate6MbpsBW10MHz"),
                                  "ControlMode",StringValue ("OfdmRate6MbpsBW10MHz"),
                                  "NonUnicastMode", StringValue ("OfdmRate6MbpsBW10MHz"));
  return helper;
}

void
WaveHelper::CreateMacForChannel (std::vector<uint32_t> channelNumbers)
{
  if (channelNumbers.size () == 0)
    {
      NS_FATAL_ERROR ("the WAVE MAC entities is at least one");
    }
  for (std::vector<uint32_t>::iterator i = channelNumbers.begin (); i != channelNumbers.end (); ++i)
    {
      if (!ChannelManager::IsWaveChannel (*i))
        {
          NS_FATAL_ERROR ("the channel number " << (*i) << " is not a valid WAVE channel number");
        }
    }
  m_macsForChannelNumber = channelNumbers;
}

void
WaveHelper::CreatePhys (uint32_t phys)
{
  if (phys == 0)
    {
      NS_FATAL_ERROR ("the WAVE PHY entities is at least one");
    }
  if (phys > ChannelManager::GetNumberOfWaveChannels ())
    {
      NS_FATAL_ERROR ("the number of assigned WAVE PHY entities is more than the number of valid WAVE channels");
    }
  m_physNumber = phys;
}

void
WaveHelper::SetRemoteStationManager (std::string type,
                                     std::string n0, const AttributeValue &v0,
                                     std::string n1, const AttributeValue &v1,
                                     std::string n2, const AttributeValue &v2,
                                     std::string n3, const AttributeValue &v3,
                                     std::string n4, const AttributeValue &v4,
                                     std::string n5, const AttributeValue &v5,
                                     std::string n6, const AttributeValue &v6,
                                     std::string n7, const AttributeValue &v7)
{
  m_stationManager = ObjectFactory ();
  m_stationManager.SetTypeId (type);
  m_stationManager.Set (n0, v0);
  m_stationManager.Set (n1, v1);
  m_stationManager.Set (n2, v2);
  m_stationManager.Set (n3, v3);
  m_stationManager.Set (n4, v4);
  m_stationManager.Set (n5, v5);
  m_stationManager.Set (n6, v6);
  m_stationManager.Set (n7, v7);
}

void
WaveHelper::SetChannelScheduler (std::string type,
                                 std::string n0, const AttributeValue &v0,
                                 std::string n1, const AttributeValue &v1,
                                 std::string n2, const AttributeValue &v2,
                                 std::string n3, const AttributeValue &v3,
                                 std::string n4, const AttributeValue &v4,
                                 std::string n5, const AttributeValue &v5,
                                 std::string n6, const AttributeValue &v6,
                                 std::string n7, const AttributeValue &v7)
{
  m_channelScheduler = ObjectFactory ();
  m_channelScheduler.SetTypeId (type);
  m_channelScheduler.Set (n0, v0);
  m_channelScheduler.Set (n1, v1);
  m_channelScheduler.Set (n2, v2);
  m_channelScheduler.Set (n3, v3);
  m_channelScheduler.Set (n4, v4);
  m_channelScheduler.Set (n5, v5);
  m_channelScheduler.Set (n6, v6);
  m_channelScheduler.Set (n7, v7);
}

NetDeviceContainer
WaveHelper::Install (const WifiPhyHelper &phyHelper,  const WifiMacHelper &macHelper, NodeContainer c) const
{
  try
    {
      [[maybe_unused]] const QosWaveMacHelper& qosMac = dynamic_cast<const QosWaveMacHelper&> (macHelper);
    }
  catch (const std::bad_cast &)
    {
      NS_FATAL_ERROR ("WifiMacHelper should be the class or subclass of QosWaveMacHelper");
    }

  NetDeviceContainer devices;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Ptr<Node> node = *i;
      Ptr<WaveNetDevice> device = CreateObject<WaveNetDevice> ();

      device->SetChannelManager (CreateObject<ChannelManager> ());
      device->SetChannelCoordinator (CreateObject<ChannelCoordinator> ());
      device->SetVsaManager (CreateObject<VsaManager> ());
      device->SetChannelScheduler (m_channelScheduler.Create<ChannelScheduler> ());

      for (uint32_t j = 0; j != m_physNumber; ++j)
        {
          Ptr<WifiPhy> phy = phyHelper.Create (node, device);
          phy->ConfigureStandard (WIFI_STANDARD_80211p);
          phy->SetOperatingChannel (WifiPhy::ChannelTuple {ChannelManager::GetCch (), 0,
                                                           WIFI_PHY_BAND_5GHZ, 0});
          device->AddPhy (phy);
        }

      for (std::vector<uint32_t>::const_iterator k = m_macsForChannelNumber.begin ();
           k != m_macsForChannelNumber.end (); ++k)
        {
          Ptr<WifiMac> wifiMac = macHelper.Create (device, WIFI_STANDARD_80211p);
          Ptr<OcbWifiMac> ocbMac = DynamicCast<OcbWifiMac> (wifiMac);
          ocbMac->SetWifiRemoteStationManager (m_stationManager.Create<WifiRemoteStationManager> ());
          ocbMac->EnableForWave (device);
          device->AddMac (*k, ocbMac);
        }

      device->SetAddress (Mac48Address::Allocate ());

      node->AddDevice (device);
      devices.Add (device);
    }
  return devices;
}

NetDeviceContainer
WaveHelper::Install (const WifiPhyHelper &phy, const WifiMacHelper &mac, Ptr<Node> node) const
{
  return Install (phy, mac, NodeContainer (node));
}

NetDeviceContainer
WaveHelper::Install (const WifiPhyHelper &phy, const WifiMacHelper &mac, std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return Install (phy, mac, NodeContainer (node));
}

void
WaveHelper::EnableLogComponents (void)
{
  WifiHelper::EnableLogComponents ();

  LogComponentEnable ("WaveNetDevice", LOG_LEVEL_ALL);
  LogComponentEnable ("ChannelCoordinator", LOG_LEVEL_ALL);
  LogComponentEnable ("ChannelManager", LOG_LEVEL_ALL);
  LogComponentEnable ("ChannelScheduler", LOG_LEVEL_ALL);
  LogComponentEnable ("DefaultChannelScheduler", LOG_LEVEL_ALL);
  LogComponentEnable ("VsaManager", LOG_LEVEL_ALL);
  LogComponentEnable ("OcbWifiMac", LOG_LEVEL_ALL);
  LogComponentEnable ("VendorSpecificAction", LOG_LEVEL_ALL);
  LogComponentEnable ("WaveFrameExchangeManager", LOG_LEVEL_ALL);
  LogComponentEnable ("HigherLayerTxVectorTag", LOG_LEVEL_ALL);
}

int64_t
WaveHelper::AssignStreams (NetDeviceContainer c, int64_t stream)
{
  int64_t currentStream = stream;
  Ptr<NetDevice> netDevice;
  for (NetDeviceContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      netDevice = (*i);
      Ptr<WaveNetDevice> wave = DynamicCast<WaveNetDevice> (netDevice);
      if (wave)
        {
          // Handle any random numbers in the PHY objects.
          std::vector<Ptr<WifiPhy> > phys = wave->GetPhys ();
          for (std::vector<Ptr<WifiPhy> >::iterator j = phys.begin (); j != phys.end (); ++j)
            {
              currentStream += (*j)->AssignStreams (currentStream);
            }

          // Handle any random numbers in the MAC objects.
          std::map<uint32_t, Ptr<OcbWifiMac> > macs = wave->GetMacs ();
          for ( std::map<uint32_t, Ptr<OcbWifiMac> >::iterator k = macs.begin (); k != macs.end (); ++k)
            {
              // Handle any random numbers in the station managers.
              Ptr<WifiRemoteStationManager> manager = k->second->GetWifiRemoteStationManager ();
              Ptr<MinstrelWifiManager> minstrel = DynamicCast<MinstrelWifiManager> (manager);
              if (minstrel)
                {
                  currentStream += minstrel->AssignStreams (currentStream);
                }

              PointerValue ptr;
              k->second->GetAttribute ("Txop", ptr);
              Ptr<Txop> txop = ptr.Get<Txop> ();
              currentStream += txop->AssignStreams (currentStream);

              k->second->GetAttribute ("VO_Txop", ptr);
              Ptr<QosTxop> vo_txop = ptr.Get<QosTxop> ();
              currentStream += vo_txop->AssignStreams (currentStream);

              k->second->GetAttribute ("VI_Txop", ptr);
              Ptr<QosTxop> vi_txop = ptr.Get<QosTxop> ();
              currentStream += vi_txop->AssignStreams (currentStream);

              k->second->GetAttribute ("BE_Txop", ptr);
              Ptr<QosTxop> be_txop = ptr.Get<QosTxop> ();
              currentStream += be_txop->AssignStreams (currentStream);

              k->second->GetAttribute ("BK_Txop", ptr);
              Ptr<QosTxop> bk_txop = ptr.Get<QosTxop> ();
              currentStream += bk_txop->AssignStreams (currentStream);
            }
        }
    }
  return (currentStream - stream);
}
} // namespace ns3
