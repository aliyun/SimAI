/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 University of Washington
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
 */

#include <stdint.h>
#include <string>
#include <fstream>

#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/ptr.h"
#include "ns3/node.h"
#include "ns3/names.h"
#include "ns3/net-device.h"
#include "ns3/pcap-file-wrapper.h"

#include "trace-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TraceHelper");

PcapHelper::PcapHelper ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

PcapHelper::~PcapHelper ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

Ptr<PcapFileWrapper>
PcapHelper::CreateFile (
  std::string filename, 
  std::ios::openmode filemode,
  DataLinkType dataLinkType,
  uint32_t    snapLen, 
  int32_t     tzCorrection)
{
  NS_LOG_FUNCTION (filename << filemode << dataLinkType << snapLen << tzCorrection);

  Ptr<PcapFileWrapper> file = CreateObject<PcapFileWrapper> ();
  file->Open (filename, filemode);
  NS_ABORT_MSG_IF (file->Fail (), "Unable to Open " << filename << " for mode " << filemode);

  file->Init (dataLinkType, snapLen, tzCorrection);
  NS_ABORT_MSG_IF (file->Fail (), "Unable to Init " << filename);

  //
  // Note that the pcap helper promptly forgets all about the pcap file.  We
  // rely on the reference count of the file object which will soon be owned
  // by the caller to keep the object alive.  If the caller uses the file 
  // object to hook a trace source, ownership of the file object will be
  // implicitly transferred to the callback which keeps the object alive.
  // When the callback is destroyed (when either the trace is disconnected or
  // the object with the trace source is deleted) the callback will be destroyed
  // and the file object will be destroyed, releasing the pointer and closing
  // the file.
  //
  return file;
}

std::string
PcapHelper::GetFilenameFromDevice (std::string prefix, Ptr<NetDevice> device, bool useObjectNames)
{
  NS_LOG_FUNCTION (prefix << device << useObjectNames);
  NS_ABORT_MSG_UNLESS (prefix.size (), "Empty prefix string");

  std::ostringstream oss;
  oss << prefix << "-";

  std::string nodename;
  std::string devicename;

  Ptr<Node> node = device->GetNode ();

  if (useObjectNames)
    {
      nodename = Names::FindName (node);
      devicename = Names::FindName (device);
    }

  if (nodename.size ())
    {
      oss << nodename;
    }
  else
    {
      oss << node->GetId ();
    }

  oss << "-";

  if (devicename.size ())
    {
      oss << devicename;
    }
  else
    {
      oss << device->GetIfIndex ();
    }

  oss << ".pcap";

  return oss.str ();
}

std::string
PcapHelper::GetFilenameFromInterfacePair (std::string prefix, Ptr<Object> object, uint32_t interface, bool useObjectNames)
{
  NS_LOG_FUNCTION (prefix << object << interface << useObjectNames);
  NS_ABORT_MSG_UNLESS (prefix.size (), "Empty prefix string");

  std::ostringstream oss;
  oss << prefix << "-";

  std::string objname;
  std::string nodename;

  Ptr<Node> node = object->GetObject<Node> ();

  if (useObjectNames)
    {
      objname = Names::FindName (object);
      nodename = Names::FindName (node);
    }

  if (objname.size ())
    {
      oss << objname;
    }
  else if (nodename.size ())
    {
      oss << nodename;
    }
  else
    {
      oss << "n" << node->GetId ();
    }

  oss << "-i" << interface << ".pcap";

  return oss.str ();
}

//
// The basic default trace sink.  This one just writes the packet to the pcap
// file which is good enough for most kinds of captures.
//
void
PcapHelper::DefaultSink (Ptr<PcapFileWrapper> file, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (file << p);
  file->Write (Simulator::Now (), p);
}

void
PcapHelper::SinkWithHeader (Ptr<PcapFileWrapper> file, const Header &header, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (file << p);
  file->Write (Simulator::Now (), header, p);
}

AsciiTraceHelper::AsciiTraceHelper ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

AsciiTraceHelper::~AsciiTraceHelper ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

Ptr<OutputStreamWrapper>
AsciiTraceHelper::CreateFileStream (std::string filename, std::ios::openmode filemode)
{
  NS_LOG_FUNCTION (filename << filemode);

  Ptr<OutputStreamWrapper> StreamWrapper = Create<OutputStreamWrapper> (filename, filemode);

  //
  // Note that the ascii trace helper promptly forgets all about the trace file.
  // We rely on the reference count of the file object which will soon be owned
  // by the caller to keep the object alive.  If the caller uses the stream 
  // object to hook a trace source, ownership of the stream object will be
  // implicitly transferred to the callback which keeps the object alive.
  // When the callback is destroyed (when either the trace is disconnected or
  // the object with the trace source is deleted) the callback will be destroyed
  // and the stream object will be destroyed, releasing the pointer and closing
  // the underlying file.
  //
  return StreamWrapper;
}

std::string
AsciiTraceHelper::GetFilenameFromDevice (std::string prefix, Ptr<NetDevice> device, bool useObjectNames)
{
  NS_LOG_FUNCTION (prefix << device << useObjectNames);
  NS_ABORT_MSG_UNLESS (prefix.size (), "Empty prefix string");

  std::ostringstream oss;
  oss << prefix << "-";

  std::string nodename;
  std::string devicename;

  Ptr<Node> node = device->GetNode ();

  if (useObjectNames)
    {
      nodename = Names::FindName (node);
      devicename = Names::FindName (device);
    }

  if (nodename.size ())
    {
      oss << nodename;
    }
  else
    {
      oss << node->GetId ();
    }

  oss << "-";

  if (devicename.size ())
    {
      oss << devicename;
    }
  else
    {
      oss << device->GetIfIndex ();
    }

  oss << ".tr";

  return oss.str ();
}

std::string
AsciiTraceHelper::GetFilenameFromInterfacePair (
  std::string prefix, 
  Ptr<Object> object, 
  uint32_t interface, 
  bool useObjectNames)
{
  NS_LOG_FUNCTION (prefix << object << interface << useObjectNames);
  NS_ABORT_MSG_UNLESS (prefix.size (), "Empty prefix string");

  std::ostringstream oss;
  oss << prefix << "-";

  std::string objname;
  std::string nodename;

  Ptr<Node> node = object->GetObject<Node> ();

  if (useObjectNames)
    {
      objname = Names::FindName (object);
      nodename = Names::FindName (node);
    }

  if (objname.size ())
    {
      oss << objname;
    }
  else if (nodename.size ())
    {
      oss << nodename;
    }
  else
    {
      oss << "n" << node->GetId ();
    }

  oss << "-i" << interface << ".tr";

  return oss.str ();
}

//
// One of the basic default trace sink sets.  Enqueue:
//
//   When a packet has been sent to a device for transmission, the device is
//   expected to place the packet onto a transmit queue even if it does not
//   have to delay the packet at all, if only to trigger this event.  This 
//   event will eventually translate into a '+' operation in the trace file.
//
//   This is typically implemented by hooking the "TxQueue/Enqueue" trace hook
//   in the device (actually the Queue in the device).
//
void
AsciiTraceHelper::DefaultEnqueueSinkWithoutContext (Ptr<OutputStreamWrapper> stream, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "+ " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}

void
AsciiTraceHelper::DefaultEnqueueSinkWithContext (Ptr<OutputStreamWrapper> stream, std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "+ " << Simulator::Now ().GetSeconds () << " " << context << " " << *p << std::endl;
}

//
// One of the basic default trace sink sets.  Drop:
//
//   When a packet has been sent to a device for transmission, the device is
//   expected to place the packet onto a transmit queue.  If this queue is 
//   full the packet will be dropped.  The device is expected to trigger an
//   event to indicate that an outbound packet is being dropped.  This event
//   will eventually translate into a 'd' operation in the trace file.
//
//   This is typically implemented by hooking the "TxQueue/Drop" trace hook
//   in the device (actually the Queue in the device).
//
void
AsciiTraceHelper::DefaultDropSinkWithoutContext (Ptr<OutputStreamWrapper> stream, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "d " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}

void
AsciiTraceHelper::DefaultDropSinkWithContext (Ptr<OutputStreamWrapper> stream, std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "d " << Simulator::Now ().GetSeconds () << " " << context << " " << *p << std::endl;
}

//
// One of the basic default trace sink sets.  Dequeue:
//
//   When a packet has been sent to a device for transmission, the device is
//   expected to place the packet onto a transmit queue even if it does not
//   have to delay the packet at all.  The device removes the packet from the
//   transmit queue when the packet is ready to send, and this dequeue will 
//   fire a corresponding event.  This event will eventually translate into a 
//   '-' operation in the trace file.
//
//   This is typically implemented by hooking the "TxQueue/Dequeue" trace hook
//   in the device (actually the Queue in the device).
//
void
AsciiTraceHelper::DefaultDequeueSinkWithoutContext (Ptr<OutputStreamWrapper> stream, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "- " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}

void
AsciiTraceHelper::DefaultDequeueSinkWithContext (Ptr<OutputStreamWrapper> stream, std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "- " << Simulator::Now ().GetSeconds () << " " << context << " " << *p << std::endl;
}

//
// One of the basic default trace sink sets.  Receive:
//
//   When a packet is received by a device for transmission, the device is
//   expected to trigger this event to indicate the reception has occurred.
//   This event will eventually translate into an 'r' operation in the trace 
//   file.
//
//   This is typically implemented by hooking the "MacRx" trace hook in the
//   device.
void
AsciiTraceHelper::DefaultReceiveSinkWithoutContext (Ptr<OutputStreamWrapper> stream, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "r " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}

void
AsciiTraceHelper::DefaultReceiveSinkWithContext (Ptr<OutputStreamWrapper> stream, std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (stream << p);
  *stream->GetStream () << "r " << Simulator::Now ().GetSeconds () << " " << context << " " << *p << std::endl;
}

void 
PcapHelperForDevice::EnablePcap (std::string prefix, Ptr<NetDevice> nd, bool promiscuous, bool explicitFilename)
{
  EnablePcapInternal (prefix, nd, promiscuous, explicitFilename);
}

void 
PcapHelperForDevice::EnablePcap (std::string prefix, std::string ndName, bool promiscuous, bool explicitFilename)
{
  Ptr<NetDevice> nd = Names::Find<NetDevice> (ndName);
  EnablePcap (prefix, nd, promiscuous, explicitFilename);
}

void 
PcapHelperForDevice::EnablePcap (std::string prefix, NetDeviceContainer d, bool promiscuous)
{
  for (NetDeviceContainer::Iterator i = d.Begin (); i != d.End (); ++i)
    {
      Ptr<NetDevice> dev = *i;
      EnablePcap (prefix, dev, promiscuous);
    }
}

void
PcapHelperForDevice::EnablePcap (std::string prefix, NodeContainer n, bool promiscuous)
{
  NetDeviceContainer devs;
  for (NodeContainer::Iterator i = n.Begin (); i != n.End (); ++i)
    {
      Ptr<Node> node = *i;
      for (uint32_t j = 0; j < node->GetNDevices (); ++j)
        {
          devs.Add (node->GetDevice (j));
        }
    }
  EnablePcap (prefix, devs, promiscuous);
}

void
PcapHelperForDevice::EnablePcapAll (std::string prefix, bool promiscuous)
{
  EnablePcap (prefix, NodeContainer::GetGlobal (), promiscuous);
}

void 
PcapHelperForDevice::EnablePcap (std::string prefix, uint32_t nodeid, uint32_t deviceid, bool promiscuous)
{
  NodeContainer n = NodeContainer::GetGlobal ();

  for (NodeContainer::Iterator i = n.Begin (); i != n.End (); ++i)
    {
      Ptr<Node> node = *i;
      if (node->GetId () != nodeid) 
        {
          continue;
        }

      NS_ABORT_MSG_IF (deviceid >= node->GetNDevices (), "PcapHelperForDevice::EnablePcap(): Unknown deviceid = " 
                       << deviceid);
      Ptr<NetDevice> nd = node->GetDevice (deviceid);
      EnablePcap (prefix, nd, promiscuous);
      return;
    }
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (std::string prefix, Ptr<NetDevice> nd, bool explicitFilename)
{
  EnableAsciiInternal (Ptr<OutputStreamWrapper> (), prefix, nd, explicitFilename);
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (Ptr<OutputStreamWrapper> stream, Ptr<NetDevice> nd)
{
  EnableAsciiInternal (stream, std::string (), nd, false);
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (std::string prefix, std::string ndName, bool explicitFilename)
{
  EnableAsciiImpl (Ptr<OutputStreamWrapper> (), prefix, ndName, explicitFilename);
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (Ptr<OutputStreamWrapper> stream, std::string ndName)
{
  EnableAsciiImpl (stream, std::string (), ndName, false);
}

//
// Private API
//
void 
AsciiTraceHelperForDevice::EnableAsciiImpl (
  Ptr<OutputStreamWrapper> stream, 
  std::string prefix, 
  std::string ndName,
  bool explicitFilename)
{
  Ptr<NetDevice> nd = Names::Find<NetDevice> (ndName);
  EnableAsciiInternal (stream, prefix, nd, explicitFilename);
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (std::string prefix, NetDeviceContainer d)
{
  EnableAsciiImpl (Ptr<OutputStreamWrapper> (), prefix, d);
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (Ptr<OutputStreamWrapper> stream, NetDeviceContainer d)
{
  EnableAsciiImpl (stream, std::string (), d);
}

//
// Private API
//
void 
AsciiTraceHelperForDevice::EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, std::string prefix, NetDeviceContainer d)
{
  for (NetDeviceContainer::Iterator i = d.Begin (); i != d.End (); ++i)
    {
      Ptr<NetDevice> dev = *i;
      EnableAsciiInternal (stream, prefix, dev, false);
    }
}

//
// Public API
//
void
AsciiTraceHelperForDevice::EnableAscii (std::string prefix, NodeContainer n)
{
  EnableAsciiImpl (Ptr<OutputStreamWrapper> (), prefix, n);
}

//
// Public API
//
void
AsciiTraceHelperForDevice::EnableAscii (Ptr<OutputStreamWrapper> stream, NodeContainer n)
{
  EnableAsciiImpl (stream, std::string (), n);
}

//
// Private API
//
void
AsciiTraceHelperForDevice::EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, std::string prefix, NodeContainer n)
{
  NetDeviceContainer devs;
  for (NodeContainer::Iterator i = n.Begin (); i != n.End (); ++i)
    {
      Ptr<Node> node = *i;
      for (uint32_t j = 0; j < node->GetNDevices (); ++j)
        {
          devs.Add (node->GetDevice (j));
        }
    }
  EnableAsciiImpl (stream, prefix, devs);
}

//
// Public API
//
void
AsciiTraceHelperForDevice::EnableAsciiAll (std::string prefix)
{
  EnableAsciiImpl (Ptr<OutputStreamWrapper> (), prefix, NodeContainer::GetGlobal ());
}

//
// Public API
//
void
AsciiTraceHelperForDevice::EnableAsciiAll (Ptr<OutputStreamWrapper> stream)
{
  EnableAsciiImpl (stream, std::string (), NodeContainer::GetGlobal ());
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (Ptr<OutputStreamWrapper> stream, uint32_t nodeid, uint32_t deviceid)
{
  EnableAsciiImpl (stream, std::string (), nodeid, deviceid, false);
}

//
// Public API
//
void 
AsciiTraceHelperForDevice::EnableAscii (
  std::string prefix, 
  uint32_t nodeid, 
  uint32_t deviceid,
  bool explicitFilename)
{
  EnableAsciiImpl (Ptr<OutputStreamWrapper> (), prefix, nodeid, deviceid, explicitFilename);
}

//
// Private API
//
void 
AsciiTraceHelperForDevice::EnableAsciiImpl (
  Ptr<OutputStreamWrapper> stream, 
  std::string prefix, 
  uint32_t nodeid, 
  uint32_t deviceid,
  bool explicitFilename)
{
  NodeContainer n = NodeContainer::GetGlobal ();

  for (NodeContainer::Iterator i = n.Begin (); i != n.End (); ++i)
    {
      Ptr<Node> node = *i;
      if (node->GetId () != nodeid) 
        {
          continue;
        }

      NS_ABORT_MSG_IF (deviceid >= node->GetNDevices (), 
                       "AsciiTraceHelperForDevice::EnableAscii(): Unknown deviceid = " << deviceid);

      Ptr<NetDevice> nd = node->GetDevice (deviceid);

      EnableAsciiInternal (stream, prefix, nd, explicitFilename);
      return;
    }
}

} // namespace ns3

