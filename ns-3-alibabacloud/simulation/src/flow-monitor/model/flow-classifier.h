/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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

#ifndef FLOW_CLASSIFIER_H
#define FLOW_CLASSIFIER_H

#include "ns3/simple-ref-count.h"
#include <ostream>

namespace ns3 {

/**
 * \ingroup flow-monitor
 * \brief Abstract identifier of a packet flow
 */
typedef uint32_t FlowId;

/**
 * \ingroup flow-monitor
 * \brief Abstract identifier of a packet within a flow
 */
typedef uint32_t FlowPacketId;


/// \ingroup flow-monitor
/// Provides a method to translate raw packet data into abstract
/// `flow identifier` and `packet identifier` parameters.  These
/// identifiers are unsigned 32-bit integers that uniquely identify a
/// flow and a packet within that flow, respectively, for the whole
/// simulation, regardless of the point in which the packet was
/// captured.  These abstract identifiers are used in the
/// communication between FlowProbe and FlowMonitor, and all collected
/// statistics reference only those abstract identifiers in order to
/// keep the core architecture generic and not tied down to any
/// particular flow capture method or classification system.
class FlowClassifier : public SimpleRefCount<FlowClassifier>
{
private:
  FlowId m_lastNewFlowId; //!< Last known Flow ID

public:

  FlowClassifier ();
  virtual ~FlowClassifier ();

  // Delete copy constructor and assignment operator to avoid misuse
  FlowClassifier (FlowClassifier const &) = delete;
  FlowClassifier& operator= (FlowClassifier const &) = delete;

  /// Serializes the results to an std::ostream in XML format
  /// \param os the output stream
  /// \param indent number of spaces to use as base indentation level
  virtual void SerializeToXmlStream (std::ostream &os, uint16_t indent) const = 0;

protected:
  /// Returns a new, unique Flow Identifier
  /// \returns a new FlowId
  FlowId GetNewFlowId ();

  ///
  /// \brief Add a number of spaces for indentation purposes.
  /// \param os The stream to write to.
  /// \param level The number of spaces to add.
  void Indent (std::ostream &os, uint16_t level) const;

};

inline void
FlowClassifier::Indent (std::ostream &os, uint16_t level) const
{
  for (uint16_t __xpto = 0; __xpto < level; __xpto++) os << ' ';
}


} // namespace ns3

#endif /* FLOW_CLASSIFIER_H */
