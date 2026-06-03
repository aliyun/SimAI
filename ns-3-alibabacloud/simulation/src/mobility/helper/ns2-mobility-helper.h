/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
 *               2009,2010 Contributors
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Contributors: Thomas Waldecker <twaldecker@rocketmail.com>
 *               Martín Giachino <martin.giachino@gmail.com>
 */
#ifndef NS2_MOBILITY_HELPER_H
#define NS2_MOBILITY_HELPER_H

#include <string>
#include <stdint.h>
#include "ns3/ptr.h"
#include "ns3/object.h"

namespace ns3 {

class ConstantVelocityMobilityModel;

/**
 * \ingroup mobility
 * \brief Helper class which can read ns-2 movement files and configure nodes mobility.
 * 
 * This implementation is based on the ns2 movement documentation of ns2
 * as described in http://www.isi.edu/nsnam/ns/doc/node172.html
 *
 * Valid trace files use the following ns2 statements:
 \verbatim
   $node set X_ x1
   $node set Y_ y1
   $node set Z_ z1
   $ns at $time $node setdest x2 y2 speed
   $ns at $time $node set X_ x1
   $ns at $time $node set Y_ Y1
   $ns at $time $node set Z_ Z1
 \endverbatim
 *
 * Note that initial position statements may also appear at the end of
 * the mobility file like this:
 \verbatim
   $ns at $time $node setdest x2 y2 speed
   $ns at $time $node set X_ x1
   $ns at $time $node set Y_ Y1
   $ns at $time $node set Z_ Z1
   $node set X_ x1
   $node set Y_ y1
   $node set Z_ z1
 \endverbatim
 *
 * The following tools are known to support this format:
 *  - BonnMotion http://net.cs.uni-bonn.de/wg/cs/applications/bonnmotion/
 *  - SUMO http://sourceforge.net/apps/mediawiki/sumo/index.php?title=Main_Page
 *  - TraNS http://trans.epfl.ch/ 
 *
 *  See usage example in examples/mobility/ns2-mobility-trace.cc
 *
 * \bug Rounding errors may cause movement to diverge from the mobility
 * pattern in ns-2 (using the same trace).
 * See https://www.nsnam.org/bugzilla/show_bug.cgi?id=1316
 */
class Ns2MobilityHelper
{
public:
  /**
   * \param filename filename of file which contains the
   *        ns2 movement trace.
   */
  Ns2MobilityHelper (std::string filename);

  /**
   * Read the ns2 trace file and configure the movement
   * patterns of all nodes contained in the global ns3::NodeList
   * whose nodeId is matches the nodeId of the nodes in the trace
   * file.
   */
  void Install (void) const;

  /**
   * \param begin an iterator which points to the start of the input
   *        object array.
   * \param end an iterator which points to the end of the input
   *        object array.
   *
   * Read the ns2 trace file and configure the movement
   * patterns of all input objects. Each input object
   * is identified by a unique node id which reflects
   * the index of the object in the input array.
   */
  template <typename T>
  void Install (T begin, T end) const;
private:
  /**
   * \brief a class to hold input objects internally
   */
  class ObjectStore
  {
public:
    virtual ~ObjectStore () {}
    /**
     * Return ith object in store
     * \param i index
     * \return pointer to object
     */
    virtual Ptr<Object> Get (uint32_t i) const = 0;
  };
  /**
   * Parses ns-2 mobility file to create ns-3 mobility events
   * \param store Object store containing ns-3 mobility models
   */
  void ConfigNodesMovements (const ObjectStore &store) const;
  /**
   * Get or create a ConstantVelocityMobilityModel corresponding to idString
   * \param idString string name for a node
   * \param store Object store containing ns-3 mobility models
   * \return pointer to a ConstantVelocityMobilityModel
   */
  Ptr<ConstantVelocityMobilityModel> GetMobilityModel (std::string idString, const ObjectStore &store) const;
  std::string m_filename; //!< filename of file containing ns-2 mobility trace 
};

} // namespace ns3

namespace ns3 {

template <typename T>
void 
Ns2MobilityHelper::Install (T begin, T end) const
{
  class MyObjectStore : public ObjectStore
  {
public:
    MyObjectStore (T begin, T end)
      : m_begin (begin),
        m_end (end)
    {}
    virtual Ptr<Object> Get (uint32_t i) const {
      T iterator = m_begin;
      iterator += i;
      if (iterator >= m_end)
        {
          return 0;
        }
      return *iterator;
    }
private:
    T m_begin;
    T m_end;
  };
  ConfigNodesMovements (MyObjectStore (begin, end));
}


} // namespace ns3

#endif /* NS2_MOBILITY_HELPER_H */
