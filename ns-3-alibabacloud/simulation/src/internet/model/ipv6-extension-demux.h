/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: David Gross <gdavid.devel@gmail.com>
 */

#ifndef IPV6_EXTENSION_DEMUX_H
#define IPV6_EXTENSION_DEMUX_H

#include <list>
#include "ns3/object.h"
#include "ns3/ptr.h"

namespace ns3
{

class Ipv6Extension;
class Node;

/**
 * \ingroup ipv6HeaderExt
 *
 * \brief Demultiplexes IPv6 extensions.
 */
class Ipv6ExtensionDemux : public Object
{
public:
  /**
   * \brief The interface ID.
   * \return type ID
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Constructor.
   */
  Ipv6ExtensionDemux ();

  /**
   * \brief Destructor.
   */
  virtual ~Ipv6ExtensionDemux ();

  /**
   * \brief Set the node.
   * \param node the node to set
   */
  void SetNode (Ptr<Node> node);

  /**
   * \brief Insert a new IPv6 Extension.
   * \param extension the extension to insert
   */
  void Insert (Ptr<Ipv6Extension> extension);

  /**
   * \brief Get the extension corresponding to extensionNumber.
   * \param extensionNumber extension number of the extension to retrieve
   * \return a matching IPv6 extension
   */
  Ptr<Ipv6Extension> GetExtension (uint8_t extensionNumber);

  /**
   * \brief Remove an extension from this demux.
   * \param extension pointer on the extension to remove
   */
  void Remove (Ptr<Ipv6Extension> extension);

protected:
  /**
   * \brief Dispose object.
   */
  virtual void DoDispose ();

private:

  /**
   * \brief Container of the IPv6 Extensions.
   */
  typedef std::list<Ptr<Ipv6Extension> > Ipv6ExtensionList_t;

  /**
   * \brief List of IPv6 Extensions supported.
   */
  Ipv6ExtensionList_t m_extensions;

  /**
   * \brief The node.
   */
  Ptr<Node> m_node;
};

} /* namespace ns3 */

#endif /* IPV6_EXTENSION_DEMUX_H */

