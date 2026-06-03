// -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*-
//
// Copyright (c) 2009 INESC Porto
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt> <gjcarneiro@gmail.com>
//
#ifndef FLOW_MONITOR_HELPER_H
#define FLOW_MONITOR_HELPER_H

#include "ns3/node-container.h"
#include "ns3/object-factory.h"
#include "ns3/flow-monitor.h"
#include "ns3/flow-classifier.h"
#include <string>

namespace ns3 {

class AttributeValue;
class Ipv4FlowClassifier;
class Ipv6FlowClassifier;

/**
 * \ingroup flow-monitor
 * \brief Helper to enable IP flow monitoring on a set of Nodes
 */
class FlowMonitorHelper
{
public:

  FlowMonitorHelper ();
  ~FlowMonitorHelper ();

  // Delete copy constructor and assignment operator to avoid misuse
  FlowMonitorHelper (const FlowMonitorHelper &) = delete;
  FlowMonitorHelper& operator= (const FlowMonitorHelper &) = delete;

  /**
   * \brief Set an attribute for the to-be-created FlowMonitor object
   * \param n1 attribute name
   * \param v1 attribute value
   */
  void SetMonitorAttribute (std::string n1, const AttributeValue &v1);

  /**
   * \brief Enable flow monitoring on a set of nodes
   * \param nodes A NodeContainer holding the set of nodes to work with.
   * \returns a pointer to the FlowMonitor object
   */
  Ptr<FlowMonitor> Install (NodeContainer nodes);
  /**
   * \brief Enable flow monitoring on a single node
   * \param node A Ptr<Node> to the node on which to enable flow monitoring.
   * \returns a pointer to the FlowMonitor object
   */
  Ptr<FlowMonitor> Install (Ptr<Node> node);
  /**
   * \brief Enable flow monitoring on all nodes
   * \returns a pointer to the FlowMonitor object
   */
  Ptr<FlowMonitor> InstallAll ();

  /**
   * \brief Retrieve the FlowMonitor object created by the Install* methods
   * \returns a pointer to the FlowMonitor object
   */
  Ptr<FlowMonitor> GetMonitor ();

  /**
   * \brief Retrieve the FlowClassifier object for IPv4 created by the Install* methods
   * \returns a pointer to the FlowClassifier object
   */
  Ptr<FlowClassifier> GetClassifier ();

  /**
   * \brief Retrieve the FlowClassifier object for IPv6 created by the Install* methods
   * \returns a pointer to the FlowClassifier object
   */
  Ptr<FlowClassifier> GetClassifier6 ();

  /**
   * Serializes the results to an std::ostream in XML format
   * \param os the output stream
   * \param indent number of spaces to use as base indentation level
   * \param enableHistograms if true, include also the histograms in the output
   * \param enableProbes if true, include also the per-probe/flow pair statistics in the output
   */
  void SerializeToXmlStream (std::ostream &os, uint16_t indent, bool enableHistograms, bool enableProbes);

  /**
   * Same as SerializeToXmlStream, but returns the output as a std::string
   * \param indent number of spaces to use as base indentation level
   * \param enableHistograms if true, include also the histograms in the output
   * \param enableProbes if true, include also the per-probe/flow pair statistics in the output
   * \return the XML output as string
   */
  std::string SerializeToXmlString (uint16_t indent, bool enableHistograms, bool enableProbes);

  /**
   * Same as SerializeToXmlStream, but writes to a file instead
   * \param fileName name or path of the output file that will be created
   * \param enableHistograms if true, include also the histograms in the output
   * \param enableProbes if true, include also the per-probe/flow pair statistics in the output
   */
  void SerializeToXmlFile (std::string fileName, bool enableHistograms, bool enableProbes);

private:
  ObjectFactory m_monitorFactory;        //!< Object factory
  Ptr<FlowMonitor> m_flowMonitor;        //!< the FlowMonitor object
  Ptr<FlowClassifier> m_flowClassifier4; //!< the FlowClassifier object for IPv4
  Ptr<FlowClassifier> m_flowClassifier6; //!< the FlowClassifier object for IPv6
};

} // namespace ns3


#endif /* FLOW_MONITOR_HELPER_H */
