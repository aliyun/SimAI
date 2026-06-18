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

#ifndef INTERNET_TRACE_HELPER_H
#define INTERNET_TRACE_HELPER_H

#include "ns3/assert.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/ipv6-interface-container.h"
#include "ns3/ipv4.h"
#include "ns3/ipv6.h"
#include "ns3/trace-helper.h"

namespace ns3 {

/**
 * \ingroup ipv4Helpers
 *
 * @brief Base class providing common user-level pcap operations for helpers
 * representing IPv4 protocols .
 */
class PcapHelperForIpv4
{
public:
  /**
   * @brief Construct a PcapHelperForIpv4.
   */
  PcapHelperForIpv4 () {}

  /**
   * @brief Destroy a PcapHelperForIpv4.
   */
  virtual ~PcapHelperForIpv4 () {}

  /**
   * @brief Enable pcap output the indicated Ipv4 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv4 Ptr<Ipv4> on which you want to enable tracing.
   * @param interface Interface on ipv4 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnablePcapIpv4Internal (std::string prefix, 
                                       Ptr<Ipv4> ipv4, 
                                       uint32_t interface,
                                       bool explicitFilename) = 0;

  /**
   * @brief Enable pcap output the indicated Ipv4 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv4 Ptr<Ipv4> on which you want to enable tracing.
   * @param interface Interface on ipv4 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnablePcapIpv4 (std::string prefix, Ptr<Ipv4> ipv4, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable pcap output the indicated Ipv4 and interface pair using a
   * Ptr<Ipv4> previously named using the ns-3 object name service.
   *
   * @param prefix filename prefix to use for pcap files.
   * @param ipv4Name Name of the Ptr<Ipv4> on which you want to enable tracing.
   * @param interface Interface on ipv4 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnablePcapIpv4 (std::string prefix, std::string ipv4Name, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable pcap output on each Ipv4 and interface pair in the container.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param c Ipv4InterfaceContainer of Ipv4 and interface pairs
   */
  void EnablePcapIpv4 (std::string prefix, Ipv4InterfaceContainer c);

  /**
   * @brief Enable pcap output on all Ipv4 and interface pairs existing in the
   * nodes provided in the container.
   *
   * \param prefix Filename prefix to use for pcap files.
   * \param n container of nodes.
   */
  void EnablePcapIpv4 (std::string prefix, NodeContainer n);

  /**
   * @brief Enable pcap output on the Ipv4 and interface pair specified by a 
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously 
   * determines the Ipv4.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param nodeid The node identifier/number of the node on which to enable tracing.
   * @param interface Interface on ipv4 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnablePcapIpv4 (std::string prefix, uint32_t nodeid, uint32_t interface, bool explicitFilename);

  /**
   * @brief Enable pcap output on all Ipv4 and interface pairs existing in the 
   * set of all nodes created in the simulation.
   *
   * @param prefix Filename prefix to use for pcap files.
   */
  void EnablePcapIpv4All (std::string prefix);

};

/**
 * \ingroup ipv4Helpers
 *
 * @brief Base class providing common user-level ascii trace operations for 
 * helpers representing IPv4 protocols .
 */
class AsciiTraceHelperForIpv4
{
public:
  /**
   * @brief Construct an AsciiTraceHelperForIpv4.
   */
  AsciiTraceHelperForIpv4 () {}

  /**
   * @brief Destroy an AsciiTraceHelperForIpv4
   */
  virtual ~AsciiTraceHelperForIpv4 () {}

  /**
   * @brief Enable ascii trace output on the indicated Ipv4 and interface pair.
   *
   * The implementation is expected to use a provided Ptr<OutputStreamWrapper>
   * if it is non-null.  If the OutputStreamWrapper is null, the implementation
   * is expected to use a provided prefix to construct a new file name for
   * each net device using the rules described in the class overview.
   *
   * If the prefix is provided, there will be one file per Ipv4 and interface pair
   * created.  In this case, adding a trace context to the file would be pointless,
   * so the helper implementation is expected to TraceConnectWithoutContext.
   *
   * If the output stream object is provided, there may be many different Ipv4 
   * and interface pairs writing to a single file.  In this case, the trace 
   * context could be important, so the helper implementation is expected to 
   * TraceConnect.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii trace files.
   * @param ipv4 Ptr<Ipv4> on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  virtual void EnableAsciiIpv4Internal (Ptr<OutputStreamWrapper> stream, 
                                        std::string prefix, 
                                        Ptr<Ipv4> ipv4, 
                                        uint32_t interface,
                                        bool explicitFilename) = 0;

  /**
   * @brief Enable ascii trace output on the indicated Ipv4 and interface pair.
   *
   * @param prefix Filename prefix to use for ascii files.
   * @param ipv4 Ptr<Ipv4> on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv4 (std::string prefix, Ptr<Ipv4> ipv4, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable ascii trace output on the indicated Ipv4 and interface pair.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param ipv4 Ptr<Ipv4> on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   */
  void EnableAsciiIpv4 (Ptr<OutputStreamWrapper> stream, Ptr<Ipv4> ipv4, uint32_t interface);

  /**
   * @brief Enable ascii trace output the indicated Ipv4 and interface pair
   * using an Ipv4 previously named using the ns-3 object name service.
   *
   * @param prefix filename prefix to use for ascii files.
   * @param ipv4Name The name of the Ipv4 on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv4 (std::string prefix, std::string ipv4Name, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable ascii trace output the indicated net device using a device 
   * previously named using the ns-3 object name service.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param ipv4Name The name of the Ipv4 on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   */
  void EnableAsciiIpv4 (Ptr<OutputStreamWrapper> stream, std::string ipv4Name, uint32_t interface);

  /**
   * @brief Enable ascii trace output on each Ipv4 and interface pair in the 
   * container
   *
   * @param prefix Filename prefix to use for ascii files.
   * @param c Ipv4InterfaceContainer of Ipv4 and interface pairs on which to 
   *          enable tracing.
   */
  void EnableAsciiIpv4 (std::string prefix, Ipv4InterfaceContainer c);

  /**
   * @brief Enable ascii trace output on each device in the container which is
   * of the appropriate type.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param c Ipv4InterfaceContainer of Ipv4 and interface pairs on which to 
   *          enable tracing.
   */
  void EnableAsciiIpv4 (Ptr<OutputStreamWrapper> stream, Ipv4InterfaceContainer c);

  /**
   * @brief Enable ascii trace output on all Ipv4 and interface pairs existing
   * in the nodes provided in the container.
   *
   * \param prefix Filename prefix to use for ascii files.
   * \param n container of nodes.
   */
  void EnableAsciiIpv4 (std::string prefix, NodeContainer n);

  /**
   * @brief Enable ascii trace output on all Ipv4 and interface pairs existing
   * in the nodes provided in the container.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * \param n container of nodes.
   */
  void EnableAsciiIpv4 (Ptr<OutputStreamWrapper> stream, NodeContainer n);

  /**
   * @brief Enable ascii trace output on all Ipv4 and interface pairs existing
   * in the set of all nodes created in the simulation.
   *
   * @param prefix Filename prefix to use for ascii files.
   */
  void EnableAsciiIpv4All (std::string prefix);

  /**
   * @brief Enable ascii trace output on each device (which is of the
   * appropriate type) in the set of all nodes created in the simulation.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   */
  void EnableAsciiIpv4All (Ptr<OutputStreamWrapper> stream);

  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously 
   * determines the Ipv4.
   *
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param deviceid The device identifier/index of the device on which to enable
   *                 ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAsciiIpv4 (std::string prefix, uint32_t nodeid, uint32_t deviceid, bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously 
   * determines the Ipv4.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnableAsciiIpv4 (Ptr<OutputStreamWrapper> stream, uint32_t nodeid, uint32_t interface, bool explicitFilename);

private:
  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously
   * determines the Ipv4.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv4Impl (Ptr<OutputStreamWrapper> stream, 
                            std::string prefix, 
                            uint32_t nodeid, 
                            uint32_t interface,
                            bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously
   * determines the Ipv4.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param n container of nodes.
   */
  void EnableAsciiIpv4Impl (Ptr<OutputStreamWrapper> stream, std::string prefix, NodeContainer n);

  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously
   * determines the Ipv4.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param c Ipv4InterfaceContainer of Ipv4 and interface pairs
   */
  void EnableAsciiIpv4Impl (Ptr<OutputStreamWrapper> stream, std::string prefix, Ipv4InterfaceContainer c);

  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously
   * determines the Ipv4.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param ipv4Name Name of the Ptr<Ipv4> on which you want to enable tracing.
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv4Impl (Ptr<OutputStreamWrapper> stream, 
                            std::string prefix, 
                            std::string ipv4Name, 
                            uint32_t interface,
                            bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the Ipv4 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv4 aggregated to a node, the node-id unambiguously
   * determines the Ipv4.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param ipv4 Ptr<Ipv4> on which you want to enable tracing.
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv4Impl (Ptr<OutputStreamWrapper> stream, 
                            std::string prefix, 
                            Ptr<Ipv4> ipv4, 
                            uint32_t interface,
                            bool explicitFilename);
};

/**
 * \ingroup ipv6Helpers
 *
 * @brief Base class providing common user-level pcap operations for helpers
 * representing IPv6 protocols .
 */
class PcapHelperForIpv6
{
public:
  /**
   * @brief Construct a PcapHelperForIpv6.
   */
  PcapHelperForIpv6 () {}

  /**
   * @brief Destroy a PcapHelperForIpv6
   */
  virtual ~PcapHelperForIpv6 () {}

  /**
   * @brief Enable pcap output the indicated Ipv6 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv6 Ptr<Ipv6> on which you want to enable tracing.
   * @param interface Interface on ipv6 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  virtual void EnablePcapIpv6Internal (std::string prefix, 
                                       Ptr<Ipv6> ipv6, 
                                       uint32_t interface,
                                       bool explicitFilename) = 0;

  /**
   * @brief Enable pcap output the indicated Ipv6 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv6 Ptr<Ipv6> on which you want to enable tracing.
   * @param interface Interface on ipv6 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnablePcapIpv6 (std::string prefix, Ptr<Ipv6> ipv6, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable pcap output the indicated Ipv6 and interface pair using a
   * Ptr<Ipv6> previously named using the ns-3 object name service.
   *
   * @param prefix filename prefix to use for pcap files.
   * @param ipv6Name Name of the Ptr<Ipv6> on which you want to enable tracing.
   * @param interface Interface on ipv6 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnablePcapIpv6 (std::string prefix, std::string ipv6Name, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable pcap output on each Ipv6 and interface pair in the container.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param c Ipv6InterfaceContainer of Ipv6 and interface pairs
   */
  void EnablePcapIpv6 (std::string prefix, Ipv6InterfaceContainer c);

  /**
   * @brief Enable pcap output on all Ipv6 and interface pairs existing in the
   * nodes provided in the container.
   *
   * \param prefix Filename prefix to use for pcap files.
   * \param n container of nodes.
   */
  void EnablePcapIpv6 (std::string prefix, NodeContainer n);

  /**
   * @brief Enable pcap output on the Ipv6 and interface pair specified by a 
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously 
   * determines the Ipv6.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param nodeid The node identifier/number of the node on which to enable tracing.
   * @param interface Interface on ipv6 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  void EnablePcapIpv6 (std::string prefix, uint32_t nodeid, uint32_t interface, bool explicitFilename);

  /**
   * @brief Enable pcap output on all Ipv6 and interface pairs existing in the 
   * set of all nodes created in the simulation.
   *
   * @param prefix Filename prefix to use for pcap files.
   */
  void EnablePcapIpv6All (std::string prefix);
};

/**
 * \ingroup ipv6Helpers
 *
 * @brief Base class providing common user-level ascii trace operations for
 * helpers representing IPv6 protocols .
 */
class AsciiTraceHelperForIpv6
{
public:
  /**
   * @brief Construct an AsciiTraceHelperForIpv6.
   */
  AsciiTraceHelperForIpv6 () {}

  /**
   * @brief Destroy an AsciiTraceHelperForIpv6
   */
  virtual ~AsciiTraceHelperForIpv6 () {}

  /**
   * @brief Enable ascii trace output on the indicated Ipv6 and interface pair.
   *
   * The implementation is expected to use a provided Ptr<OutputStreamWrapper>
   * if it is non-null.  If the OutputStreamWrapper is null, the implementation
   * is expected to use a provided prefix to construct a new file name for
   * each net device using the rules described in the class overview.
   *
   * If the prefix is provided, there will be one file per Ipv6 and interface pair
   * created.  In this case, adding a trace context to the file would be pointless,
   * so the helper implementation is expected to TraceConnectWithoutContext.
   *
   * If the output stream object is provided, there may be many different Ipv6 
   * and interface pairs writing to a single file.  In this case, the trace 
   * context could be important, so the helper implementation is expected to 
   * TraceConnect.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii trace files.
   * @param ipv6 Ptr<Ipv6> on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  virtual void EnableAsciiIpv6Internal (Ptr<OutputStreamWrapper> stream, 
                                        std::string prefix, 
                                        Ptr<Ipv6> ipv6, 
                                        uint32_t interface,
                                        bool explicitFilename) = 0;

  /**
   * @brief Enable ascii trace output on the indicated Ipv6 and interface pair.
   *
   * @param prefix Filename prefix to use for ascii files.
   * @param ipv6 Ptr<Ipv6> on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv6 (std::string prefix, Ptr<Ipv6> ipv6, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable ascii trace output on the indicated Ipv6 and interface pair.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param ipv6 Ptr<Ipv6> on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   */
  void EnableAsciiIpv6 (Ptr<OutputStreamWrapper> stream, Ptr<Ipv6> ipv6, uint32_t interface);

  /**
   * @brief Enable ascii trace output the indicated Ipv6 and interface pair
   * using an Ipv6 previously named using the ns-3 object name service.
   *
   * @param prefix filename prefix to use for ascii files.
   * @param ipv6Name The name of the Ipv6 on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv6 (std::string prefix, std::string ipv6Name, uint32_t interface, bool explicitFilename = false);

  /**
   * @brief Enable ascii trace output the indicated net device using a device 
   * previously named using the ns-3 object name service.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param ipv6Name The name of the Ipv6 on which you want to enable tracing.
   * @param interface The interface on which you want to enable tracing.
   */
  void EnableAsciiIpv6 (Ptr<OutputStreamWrapper> stream, std::string ipv6Name, uint32_t interface); 

  /**
   * @brief Enable ascii trace output on each Ipv6 and interface pair in the 
   * container
   *
   * @param prefix Filename prefix to use for ascii files.
   * @param c Ipv6InterfaceContainer of Ipv6 and interface pairs on which to 
   *          enable tracing.
   */
  void EnableAsciiIpv6 (std::string prefix, Ipv6InterfaceContainer c);

  /**
   * @brief Enable ascii trace output on each device in the container which is
   * of the appropriate type.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param c Ipv6InterfaceContainer of Ipv6 and interface pairs on which to 
   *          enable tracing.
   */
  void EnableAsciiIpv6 (Ptr<OutputStreamWrapper> stream, Ipv6InterfaceContainer c);

  /**
   * @brief Enable ascii trace output on all Ipv6 and interface pairs existing
   * in the nodes provided in the container.
   *
   * \param prefix Filename prefix to use for ascii files.
   * \param n container of nodes.
   */
  void EnableAsciiIpv6 (std::string prefix, NodeContainer n);

  /**
   * @brief Enable ascii trace output on all Ipv6 and interface pairs existing
   * in the nodes provided in the container.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * \param n container of nodes.
   */
  void EnableAsciiIpv6 (Ptr<OutputStreamWrapper> stream, NodeContainer n);

  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously 
   * determines the Ipv6.
   *
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv6 (std::string prefix, uint32_t nodeid, uint32_t interface, bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously 
   * determines the Ipv6.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param interface The interface on which you want to enable tracing.
   */
  void EnableAsciiIpv6 (Ptr<OutputStreamWrapper> stream, uint32_t nodeid, uint32_t interface);

  /**
   * @brief Enable ascii trace output on all Ipv6 and interface pairs existing
   * in the set of all nodes created in the simulation.
   *
   * @param prefix Filename prefix to use for ascii files.
   */
  void EnableAsciiIpv6All (std::string prefix);

  /**
   * @brief Enable ascii trace output on each device (which is of the
   * appropriate type) in the set of all nodes created in the simulation.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   */
  void EnableAsciiIpv6All (Ptr<OutputStreamWrapper> stream);

private:
  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously
   * determines the Ipv6.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param nodeid The node identifier/number of the node on which to enable
   *               ascii tracing
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv6Impl (Ptr<OutputStreamWrapper> stream, 
                            std::string prefix, 
                            uint32_t nodeid, 
                            uint32_t interface,
                            bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously
   * determines the Ipv6.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param n container of nodes.
   */
  void EnableAsciiIpv6Impl (Ptr<OutputStreamWrapper> stream, std::string prefix, NodeContainer n);

  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously
   * determines the Ipv6.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param c Ipv6InterfaceContainer of Ipv6 and interface pairs
   */
  void EnableAsciiIpv6Impl (Ptr<OutputStreamWrapper> stream, std::string prefix, Ipv6InterfaceContainer c);

  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously
   * determines the Ipv6.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param ipv6Name Name of the Ptr<Ipv6> on which you want to enable tracing.
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv6Impl (Ptr<OutputStreamWrapper> stream, 
                            std::string prefix, 
                            std::string ipv6Name, 
                            uint32_t interface,
                            bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the Ipv6 and interface pair specified by a
   * global node-id (of a previously created node) and interface.  Since there
   * can be only one Ipv6 aggregated to a node, the node-id unambiguously
   * determines the Ipv6.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use when creating ascii trace files
   * @param ipv6 Ptr<Ipv6> on which you want to enable tracing.
   * @param interface The device identifier/index of the device on which to enable
   *               ascii tracing
   * @param explicitFilename Treat the prefix as an explicit filename if true.
   */
  void EnableAsciiIpv6Impl (Ptr<OutputStreamWrapper> stream, 
                            std::string prefix, 
                            Ptr<Ipv6> ipv6, 
                            uint32_t interface,
                            bool explicitFilename);
};

} // namespace ns3

#endif /* INTERNET_TRACE_HELPER_H */
