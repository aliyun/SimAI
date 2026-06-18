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

#ifndef FLAME_INSTALLER_H
#define FLAME_INSTALLER_H

#include "ns3/mesh-stack-installer.h"

namespace ns3 {

/**
 * \ingroup flame
 *
 * \brief Helper class used to install FLAME mesh stack (actually single
 * protocol in this stack)
 */
class FlameStack : public MeshStack
{
public:

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * Construct a FlameStack helper class.
   */
  FlameStack ();

  /**
   * Destroy a FlameStack helper class.
   */
  ~FlameStack ();

  /**
   * Break any reference cycles in the installer helper.  Required for ns-3
   * Object support.
   */
  void DoDispose ();

  /**
   * \brief Install a flame stack on the given MeshPointDevice
   * \param mp The Ptr<MeshPointDevice> to use.
   * \return true if successful
   */
  bool InstallStack (Ptr<MeshPointDevice> mp);

  /**
   * \brief Print flame protocol statistics.
   * \param mp The Ptr<MeshPointDevice> to use.
   * \param os The output stream
   */
  void Report (const Ptr<MeshPointDevice> mp, std::ostream&);

  /**
   * \brief Reset the statistics.
   * \param mp The Ptr<MeshPointDevice> to use.
   */
  void ResetStats (const Ptr<MeshPointDevice> mp);
};

} // namespace ns3

#endif /* FLAME_INSTALLER_H */

