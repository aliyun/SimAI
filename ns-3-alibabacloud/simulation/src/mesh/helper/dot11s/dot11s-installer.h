/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef DOT11S_STACK_INSTALLER_H
#define DOT11S_STACK_INSTALLER_H

#include "ns3/mesh-stack-installer.h"

namespace ns3 {

/**
 * \ingroup dot11s
 * \brief Helper class to allow easy installation of 802.11s stack.
 */
class Dot11sStack : public MeshStack
{
public:
  
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * Create a Dot11sStack() installer helper.
   */
  Dot11sStack ();

  /**
   * Destroy a Dot11sStack() installer helper.
   */
  ~Dot11sStack ();

  /**
   * Break any reference cycles in the installer helper.  Required for ns-3
   * Object support.
   */
  void DoDispose ();

  /**
   * \brief Install an 802.11s stack.
   * \param mp The Ptr<MeshPointDevice> to use when setting up the PMP.
   * \return true if successful
   */
  bool InstallStack (Ptr<MeshPointDevice> mp);
 
  /**
   * \brief Iterate through the referenced devices and protocols and print
   * their statistics
   * \param mp The Ptr<MeshPointDevice> to use when setting up the PMP.
   * \param os The output stream
   */
  void Report (const Ptr<MeshPointDevice> mp, std::ostream&);

  /**
   * \brief Reset the statistics on the referenced devices and protocols.
   * \param mp The Ptr<MeshPointDevice> to use when setting up the PMP.
   */
  void ResetStats (const Ptr<MeshPointDevice> mp);
private:
  Mac48Address m_root; ///< root
};

} // namespace ns3

#endif

