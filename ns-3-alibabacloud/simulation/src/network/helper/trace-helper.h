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

#ifndef TRACE_HELPER_H
#define TRACE_HELPER_H

#include "ns3/assert.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"
#include "ns3/pcap-file-wrapper.h"
#include "ns3/output-stream-wrapper.h"

namespace ns3 {

/**
 * \brief Manage pcap files for device models
 *
 * Handling pcap files is a common operation for ns-3 devices.  It is useful to
 * provide a common base class for dealing with these ops.
 */

class PcapHelper
{
public:
  /**
   * This enumeration holds the data link types that will be written to the pcap file.
   *
   * We don't include pcap-bpf.h to avoid an explicit dependency on the real pcap
   * and we don't make an enumeration of all of the values to make it easy to
   * pass new values in.
   *
   * For a list of Data Link Types see http://www.tcpdump.org/linktypes.html
   */
  enum DataLinkType {
    DLT_NULL = 0,
    DLT_EN10MB = 1,
    DLT_PPP = 9,
    DLT_RAW = 101,
    DLT_IEEE802_11 = 105,
    DLT_LINUX_SLL = 113,
    DLT_PRISM_HEADER = 119,
    DLT_IEEE802_11_RADIO = 127,
    DLT_IEEE802_15_4 = 195,
    DLT_NETLINK = 253,
    DLT_LORATAP = 270
  };

  /**
   * @brief Create a pcap helper.
   */
  PcapHelper ();

  /**
   * @brief Destroy a pcap helper.
   */
  ~PcapHelper ();

  /**
   * @brief Let the pcap helper figure out a reasonable filename to use for a
   * pcap file associated with a device.
   * 
   * @param prefix prefix string
   * @param device NetDevice
   * @param useObjectNames use node and device names instead of indexes
   * @returns file name
   */
  std::string GetFilenameFromDevice (std::string prefix, Ptr<NetDevice> device, bool useObjectNames = true);

  /**
   * @brief Let the pcap helper figure out a reasonable filename to use for the
   * pcap file associated with a node.
   * 
   * @param prefix prefix string
   * @param object interface (such as Ipv4Interface or Ipv6Interface)
   * @param interface interface id
   * @param useObjectNames use node and device names instead of indexes
   * @returns file name
   */
  std::string GetFilenameFromInterfacePair (std::string prefix, Ptr<Object> object, 
                                            uint32_t interface, bool useObjectNames = true);

  /**
   * @brief Create and initialize a pcap file.
   * 
   * @param filename file name
   * @param filemode file mode
   * @param dataLinkType data link type of packet data
   * @param snapLen maximum length of packet data stored in records
   * @param tzCorrection time zone correction to be applied to timestamps of packets
   * @returns a smart pointer to the Pcap file
   */
  Ptr<PcapFileWrapper> CreateFile (std::string filename,
                                   std::ios::openmode filemode,
                                   DataLinkType dataLinkType,
                                   uint32_t snapLen = std::numeric_limits<uint32_t>::max (),
                                   int32_t tzCorrection = 0);
  /**
   * @brief Hook a trace source to the default trace sink
   * 
   * @param object object
   * @param traceName trace source name
   * @param file file wrapper
   */
  template <typename T> void HookDefaultSink (Ptr<T> object, std::string traceName, Ptr<PcapFileWrapper> file);

private:
  /**
   * The basic default trace sink.
   *
   * This one just writes the packet to the pcap
   * file which is good enough for most kinds of captures.
   *
   * @param file the file to write to
   * @param p the packet to write
   */
  static void DefaultSink (Ptr<PcapFileWrapper> file, Ptr<const Packet> p);

  /**
   * This trace sink passes a header separately from the packet to prevent creating a new packet
   * (for performance reasons)
   *
   * @param file the file to write to
   * @param header header of the packet
   * @param p the packet to write
   *
   * @see DefaultSink
   */
  static void SinkWithHeader (Ptr<PcapFileWrapper> file, const Header& header, Ptr<const Packet> p);
};

template <typename T> void
PcapHelper::HookDefaultSink (Ptr<T> object, std::string tracename, Ptr<PcapFileWrapper> file)
{
  bool result =
    object->TraceConnectWithoutContext (tracename.c_str (), MakeBoundCallback (&DefaultSink, file));
  NS_ASSERT_MSG (result == true, "PcapHelper::HookDefaultSink():  Unable to hook \"" << tracename << "\"");
}

/**
 * \brief Manage ASCII trace files for device models
 *
 * Handling ascii trace files is a common operation for ns-3 devices.  It is 
 * useful to provide a common base class for dealing with these ops.
 */

class AsciiTraceHelper
{
public:
  /**
   * @brief Create an ascii trace helper.
   */
  AsciiTraceHelper ();

  /**
   * @brief Destroy an ascii trace helper.
   */
  ~AsciiTraceHelper ();

  /**
   * @brief Let the ascii trace helper figure out a reasonable filename to use
   * for an ascii trace file associated with a device.
   * 
   * @param prefix prefix string
   * @param device NetDevice
   * @param useObjectNames use node and device names instead of indexes
   * @returns file name
   */
  std::string GetFilenameFromDevice (std::string prefix, Ptr<NetDevice> device, bool useObjectNames = true);

  /**
   * @brief Let the ascii trace helper figure out a reasonable filename to use
   * for an ascii trace file associated with a node.
   * 
   * @param prefix prefix string
   * @param object interface (such as Ipv4Interface or Ipv6Interface)
   * @param interface interface id
   * @param useObjectNames use node and device names instead of indexes
   * @returns file name
   */
  std::string GetFilenameFromInterfacePair (std::string prefix, Ptr<Object> object, 
                                            uint32_t interface, bool useObjectNames = true);

  /**
   * @brief Create and initialize an output stream object we'll use to write the 
   * traced bits.
   *
   * One of the common issues users run into when trying to use tracing in ns-3
   * is actually a design decision made in the C++ stream library.  It is not 
   * widely known that copy and assignment of iostreams is forbidden by 
   * std::basic_ios<>.  This is because it is not possible to predict the 
   * semantics of the stream desired by a user.
   *
   * The tempting ns-3 idiom when tracing to a file is to create a bound callback
   * with an ofstream as the bound object.  Unfortunately, this implies a copy 
   * construction in order to get the ofstream object into the callback.  This 
   * operation, as mentioned above, is forbidden by the STL.  You could use a 
   * global ostream and pass a pointer to it, but that is pretty ugly.  You 
   * could create an ostream on the stack and pass a pointer to it, but you may
   * run into object lifetime issues.  Ns-3 has a nice reference counted object
   * that can solve the problem so we use one of those to carry the stream
   * around and deal with the lifetime issues.
   * 
   * @param filename file name
   * @param filemode file mode
   * @returns a smart pointer to the output stream
   */
  Ptr<OutputStreamWrapper> CreateFileStream (std::string filename, 
                                             std::ios::openmode filemode = std::ios::out);

  /**
   * @brief Hook a trace source to the default enqueue operation trace sink that
   * does not accept nor log a trace context.
   * 
   * @param object object
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultEnqueueSinkWithoutContext (Ptr<T> object, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default enqueue operation trace sink that
   * does accept and log a trace context.
   * 
   * @param object object
   * @param context context string
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultEnqueueSinkWithContext (Ptr<T> object, 
                                          std::string context, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default drop operation trace sink that 
   * does not accept nor log a trace context.
   * 
   * @param object object
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultDropSinkWithoutContext (Ptr<T> object, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default drop operation trace sink that 
   * does accept and log a trace context.
   * 
   * @param object object
   * @param context context string
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultDropSinkWithContext (Ptr<T> object, 
                                       std::string context, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default dequeue operation trace sink
   * that does not accept nor log a trace context.
   * 
   * @param object object
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultDequeueSinkWithoutContext (Ptr<T> object, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default dequeue operation trace sink
   * that does accept and log a trace context.
   * 
   * @param object object
   * @param context context string
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultDequeueSinkWithContext (Ptr<T> object, 
                                          std::string context, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default receive operation trace sink
   * that does not accept nor log a trace context.
   * 
   * @param object object
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultReceiveSinkWithoutContext (Ptr<T> object, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Hook a trace source to the default receive operation trace sink
   * that does accept and log a trace context.
   * 
   * @param object object
   * @param context context string
   * @param traceName trace source name
   * @param stream output stream wrapper
   */
  template <typename T> 
  void HookDefaultReceiveSinkWithContext (Ptr<T> object, 
                                          std::string context, std::string traceName, Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Basic Enqueue default trace sink.
   *
   * When a packet has been sent to a device for transmission, the device is
   * expected to place the packet onto a transmit queue even if it does not
   * have to delay the packet at all, if only to trigger this event.  This
   * event will eventually translate into a '+' operation in the trace file.
   *
   * This is typically implemented by hooking the "TxQueue/Enqueue" trace hook
   * in the device (actually the Queue in the device).
   *
   * @param file the output file
   * @param p the packet
   */
  static void DefaultEnqueueSinkWithoutContext (Ptr<OutputStreamWrapper> file, Ptr<const Packet> p);

  /**
   * @brief Basic Enqueue default trace sink.
   *
   * When a packet has been sent to a device for transmission, the device is
   * expected to place the packet onto a transmit queue even if it does not
   * have to delay the packet at all, if only to trigger this event.  This
   * event will eventually translate into a '+' operation in the trace file.
   *
   * This is typically implemented by hooking the "TxQueue/Enqueue" trace hook
   * in the device (actually the Queue in the device).
   *
   * @param file the output file
   * @param context the context
   * @param p the packet
   */
  static void DefaultEnqueueSinkWithContext (Ptr<OutputStreamWrapper> file, std::string context, Ptr<const Packet> p);

  /**
   * @brief Basic Drop default trace sink.
   *
   * When a packet has been sent to a device for transmission, the device is
   * expected to place the packet onto a transmit queue.  If this queue is
   * full the packet will be dropped.  The device is expected to trigger an
   * event to indicate that an outbound packet is being dropped.  This event
   * will eventually translate into a 'd' operation in the trace file.
   *
   * This is typically implemented by hooking the "TxQueue/Drop" trace hook
   * in the device (actually the Queue in the device).
   *
   * @param file the output file
   * @param p the packet
   */
  static void DefaultDropSinkWithoutContext (Ptr<OutputStreamWrapper> file, Ptr<const Packet> p);

  /**
   * @brief Basic Drop default trace sink.
   *
   * When a packet has been sent to a device for transmission, the device is
   * expected to place the packet onto a transmit queue.  If this queue is
   * full the packet will be dropped.  The device is expected to trigger an
   * event to indicate that an outbound packet is being dropped.  This event
   * will eventually translate into a 'd' operation in the trace file.
   *
   * This is typically implemented by hooking the "TxQueue/Drop" trace hook
   * in the device (actually the Queue in the device).
   *
   * @param file the output file
   * @param context the context
   * @param p the packet
   */
  static void DefaultDropSinkWithContext (Ptr<OutputStreamWrapper> file, std::string context, Ptr<const Packet> p);

  /**
   * @brief Basic Dequeue default trace sink.
   *
   * When a packet has been sent to a device for transmission, the device is
   * expected to place the packet onto a transmit queue even if it does not
   * have to delay the packet at all.  The device removes the packet from the
   * transmit queue when the packet is ready to send, and this dequeue will
   * fire a corresponding event.  This event will eventually translate into a
   * '-' operation in the trace file.
   *
   * This is typically implemented by hooking the "TxQueue/Dequeue" trace hook
   * in the device (actually the Queue in the device).
   *
   * @param file the output file
   * @param p the packet
   */
  static void DefaultDequeueSinkWithoutContext (Ptr<OutputStreamWrapper> file, Ptr<const Packet> p);

  /**
   * @brief Basic Dequeue default trace sink.
   *
   * When a packet has been sent to a device for transmission, the device is
   * expected to place the packet onto a transmit queue even if it does not
   * have to delay the packet at all.  The device removes the packet from the
   * transmit queue when the packet is ready to send, and this dequeue will
   * fire a corresponding event.  This event will eventually translate into a
   * '-' operation in the trace file.
   *
   * This is typically implemented by hooking the "TxQueue/Dequeue" trace hook
   * in the device (actually the Queue in the device).
   *
   * @param file the output file
   * @param context the context
   * @param p the packet
   */
  static void DefaultDequeueSinkWithContext (Ptr<OutputStreamWrapper> file, std::string context, Ptr<const Packet> p);

  /**
   * @brief Basic Receive default trace sink.
   *
   * When a packet is received by a device for transmission, the device is
   * expected to trigger this event to indicate the reception has occurred.
   * This event will eventually translate into an 'r' operation in the trace
   * file.
   *
   * This is typically implemented by hooking the "MacRx" trace hook in the
   * device.
   *
   * @param file the output file
   * @param p the packet
   */
  static void DefaultReceiveSinkWithoutContext (Ptr<OutputStreamWrapper> file, Ptr<const Packet> p);

  /**
   * @brief Basic Receive default trace sink.
   *
   * When a packet is received by a device for transmission, the device is
   * expected to trigger this event to indicate the reception has occurred.
   * This event will eventually translate into an 'r' operation in the trace
   * file.
   *
   * This is typically implemented by hooking the "MacRx" trace hook in the
   * device.
   *
   * @param file the output file
   * @param context the context
   * @param p the packet
   */
  static void DefaultReceiveSinkWithContext (Ptr<OutputStreamWrapper> file, std::string context, Ptr<const Packet> p);
};

template <typename T> void
AsciiTraceHelper::HookDefaultEnqueueSinkWithoutContext (Ptr<T> object, std::string tracename, Ptr<OutputStreamWrapper> file)
{
  bool result =
    object->TraceConnectWithoutContext (tracename, MakeBoundCallback (&DefaultEnqueueSinkWithoutContext, file));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultEnqueueSinkWithoutContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultEnqueueSinkWithContext (
  Ptr<T> object, 
  std::string context, 
  std::string tracename, 
  Ptr<OutputStreamWrapper> stream)
{
  bool result =
    object->TraceConnect (tracename, context, MakeBoundCallback (&DefaultEnqueueSinkWithContext, stream));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultEnqueueSinkWithContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultDropSinkWithoutContext (Ptr<T> object, std::string tracename, Ptr<OutputStreamWrapper> file)
{
  bool result =
    object->TraceConnectWithoutContext (tracename, MakeBoundCallback (&DefaultDropSinkWithoutContext, file));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultDropSinkWithoutContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultDropSinkWithContext (
  Ptr<T> object, 
  std::string context,
  std::string tracename, 
  Ptr<OutputStreamWrapper> stream)
{
  bool result =
    object->TraceConnect (tracename, context, MakeBoundCallback (&DefaultDropSinkWithContext, stream));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultDropSinkWithContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultDequeueSinkWithoutContext (Ptr<T> object, std::string tracename, Ptr<OutputStreamWrapper> file)
{
  bool result =
    object->TraceConnectWithoutContext (tracename, MakeBoundCallback (&DefaultDequeueSinkWithoutContext, file));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultDequeueSinkWithoutContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultDequeueSinkWithContext (
  Ptr<T> object, 
  std::string context,
  std::string tracename, 
  Ptr<OutputStreamWrapper> stream)
{
  bool result =
    object->TraceConnect (tracename, context, MakeBoundCallback (&DefaultDequeueSinkWithContext, stream));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultDequeueSinkWithContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultReceiveSinkWithoutContext (Ptr<T> object, std::string tracename, Ptr<OutputStreamWrapper> file)
{
  bool result =
    object->TraceConnectWithoutContext (tracename, MakeBoundCallback (&DefaultReceiveSinkWithoutContext, file));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultReceiveSinkWithoutContext():  Unable to hook \"" 
                 << tracename << "\"");
}

template <typename T> void
AsciiTraceHelper::HookDefaultReceiveSinkWithContext (
  Ptr<T> object, 
  std::string context,
  std::string tracename, 
  Ptr<OutputStreamWrapper> stream)
{
  bool result =
    object->TraceConnect (tracename, context, MakeBoundCallback (&DefaultReceiveSinkWithContext, stream));
  NS_ASSERT_MSG (result == true, "AsciiTraceHelper::HookDefaultReceiveSinkWithContext():  Unable to hook \"" 
                 << tracename << "\"");
}

/**
 * \brief Base class providing common user-level pcap operations for helpers
 * representing net devices.
 */
class PcapHelperForDevice
{
public:
  /**
   * @brief Construct a PcapHelperForDevice
   */
  PcapHelperForDevice () {}

  /**
   * @brief Destroy a PcapHelperForDevice
   */
  virtual ~PcapHelperForDevice () {}

  /**
   * @brief Enable pcap output the indicated net device.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param nd Net device for which you want to enable tracing.
   * @param promiscuous If true capture all possible packets available at the device.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnablePcapInternal (std::string prefix, Ptr<NetDevice> nd, bool promiscuous, bool explicitFilename) = 0;

  /**
   * @brief Enable pcap output the indicated net device.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param nd Net device for which you want to enable tracing.
   * @param promiscuous If true capture all possible packets available at the device.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnablePcap (std::string prefix, Ptr<NetDevice> nd, bool promiscuous = false, bool explicitFilename = false);

  /**
   * @brief Enable pcap output the indicated net device using a device previously
   * named using the ns-3 object name service.
   *
   * @param prefix filename prefix to use for pcap files.
   * @param ndName The name of the net device in which you want to enable tracing.
   * @param promiscuous If true capture all possible packets available at the device.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnablePcap (std::string prefix, std::string ndName, bool promiscuous = false, bool explicitFilename = false);

  /**
   * @brief Enable pcap output on each device in the container which is of the 
   * appropriate type.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param d container of devices of type ns3::CsmaNetDevice
   * @param promiscuous If true capture all possible packets available at the device.
   */
  void EnablePcap (std::string prefix, NetDeviceContainer d, bool promiscuous = false);

  /**
   * @brief Enable pcap output on each device (which is of the appropriate type)
   * in the nodes provided in the container.
   *
   * \param prefix Filename prefix to use for pcap files.
   * \param n container of nodes.
   * \param promiscuous If true capture all possible packets available at the device.
   */
  void EnablePcap (std::string prefix, NodeContainer n, bool promiscuous = false);

  /**
   * @brief Enable pcap output on the device specified by a global node-id (of
   * a previously created node) and associated device-id.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param nodeid the node id
   * @param deviceid the device id
   * @param promiscuous If true capture all possible packets available at the device.
   */
  void EnablePcap (std::string prefix, uint32_t nodeid, uint32_t deviceid, bool promiscuous = false);

  /**
   * @brief Enable pcap output on each device (which is of the appropriate type)
   * in the set of all nodes created in the simulation.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param promiscuous If true capture all possible packets available at the device.
   */
  void EnablePcapAll (std::string prefix, bool promiscuous = false);
};

/**
 * \brief Base class providing common user-level ascii trace operations for helpers
 * representing net devices.
 */
class AsciiTraceHelperForDevice
{
public:
  /**
   * @brief Construct an AsciiTraceHelperForDevice.
   */
  AsciiTraceHelperForDevice () {}

  /**
   * @brief Destroy an AsciiTraceHelperForDevice.
   */
  virtual ~AsciiTraceHelperForDevice () {}

  /**
   * @brief Enable ascii trace output on the indicated net device.
   *
   * The implementation is expected to use a provided Ptr<OutputStreamWrapper>
   * if it is non-null.  If the OutputStreamWrapper is null, the implementation
   * is expected to use a provided prefix to construct a new file name for
   * each net device using the rules described in the class overview.
   *
   * If the prefix is provided, there will be one file per net device created.
   * In this case, adding a trace context to the file would be pointless, so
   * the device implementation is expected to TraceConnectWithoutContext.
   *
   * If the output stream object is provided, there may be many different 
   * devices writing to a single file.  In this case, the device adding a 
   * trace context could be important, so the device implementation is 
   * expected to TraceConnect.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii trace files.
   * @param nd Net device for which you want to enable tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnableAsciiInternal (Ptr<OutputStreamWrapper> stream, 
                                    std::string prefix, 
                                    Ptr<NetDevice> nd,
                                    bool explicitFilename) = 0;

  /**
   * @brief Enable ascii trace output on the indicated net device.
   *
   * @param prefix Filename prefix to use for ascii files.
   * @param nd Net device for which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAscii (std::string prefix, Ptr<NetDevice> nd, bool explicitFilename = false);

  /**
   * @brief Enable ascii trace output on the indicated net device.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param nd Net device for which you want to enable tracing.
   */
  void EnableAscii (Ptr<OutputStreamWrapper> stream, Ptr<NetDevice> nd);

  /**
   * @brief Enable ascii trace output the indicated net device using a device 
   * previously named using the ns-3 object name service.
   *
   * @param prefix filename prefix to use for ascii files.
   * @param ndName The name of the net device in which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAscii (std::string prefix, std::string ndName, bool explicitFilename = false);

  /**
   * @brief Enable ascii trace output the indicated net device using a device 
   * previously named using the ns-3 object name service.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param ndName The name of the net device in which you want to enable tracing.
   */
  void EnableAscii (Ptr<OutputStreamWrapper> stream, std::string ndName);

  /**
   * @brief Enable ascii trace output on each device in the container which is
   * of the appropriate type.
   *
   * @param prefix Filename prefix to use for ascii files.
   * @param d container of devices
   */
  void EnableAscii (std::string prefix, NetDeviceContainer d);

  /**
   * @brief Enable ascii trace output on each device in the container which is
   * of the appropriate type.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param d container of devices
   */
  void EnableAscii (Ptr<OutputStreamWrapper> stream, NetDeviceContainer d);

  /**
   * @brief Enable ascii trace output on each device (which is of the 
   * appropriate type) in the nodes provided in the container.
   *
   * \param prefix Filename prefix to use for ascii files.
   * \param n container of nodes.
   */
  void EnableAscii (std::string prefix, NodeContainer n);

  /**
   * @brief Enable ascii trace output on each device (which is of the 
   * appropriate type) in the nodes provided in the container.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * \param n container of nodes.
   */
  void EnableAscii (Ptr<OutputStreamWrapper> stream, NodeContainer n);

  /**
   * @brief Enable ascii trace output on each device (which is of the
   * appropriate type) in the set of all nodes created in the simulation.
   *
   * @param prefix Filename prefix to use for ascii files.
   */
  void EnableAsciiAll (std::string prefix);

  /**
   * @brief Enable ascii trace output on each device (which is of the
   * appropriate type) in the set of all nodes created in the simulation.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   */
  void EnableAsciiAll (Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Enable ascii trace output on the device specified by a global 
   * node-id (of a previously created node) and associated device-id.
   *
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param deviceid The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAscii (std::string prefix, uint32_t nodeid, uint32_t deviceid, bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the device specified by a global 
   * node-id (of a previously created node) and associated device-id.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param deviceid The device identifier/index of the device on which to enable
   *               ascii tracing
   */
  void EnableAscii (Ptr<OutputStreamWrapper> stream, uint32_t nodeid, uint32_t deviceid);

private:
  /**
   * @brief Enable ascii trace output on the device specified by a global
   * node-id (of a previously created node) and associated device-id (implementation)
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii trace files.
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param deviceid The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */

  void EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, 
                        std::string prefix, 
                        uint32_t nodeid, 
                        uint32_t deviceid,
                        bool explicitFilename);

  /**
   * @brief Enable ascii trace output on each device (which is of the
   * appropriate type) in the nodes provided in the container (implementation).
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii files.
   * @param n container of nodes.
   */
  void EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, std::string prefix, NodeContainer n);

  /**
   * @brief Enable ascii trace output on each device in the container which is
   * of the appropriate type (implementation).
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii files.
   * @param d container of devices
   */
  void EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, std::string prefix, NetDeviceContainer d);

  /**
   * @brief Enable ascii trace output the indicated net device using a device
   * previously named using the ns-3 object name service (implementation).
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix filename prefix to use for ascii files.
   * @param ndName The name of the net device in which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, std::string prefix, std::string ndName, bool explicitFilename);

  /**
   * @brief Enable ascii trace output the indicated net device (implementation).
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix filename prefix to use for ascii files.
   * @param nd Net device for which you want to enable tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAsciiImpl (Ptr<OutputStreamWrapper> stream, std::string prefix, Ptr<NetDevice> nd, bool explicitFilename);
};

} // namespace ns3

#endif /* TRACE_HELPER_H */
