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


#ifndef MESH_STACK_INSTALLER_H
#define MESH_STACK_INSTALLER_H
#include "ns3/mesh-point-device.h"
namespace ns3 {
/**
 * \ingroup mesh
 *
 * \brief Prototype for class, which helps to install MAC-layer
 * routing stack to ns3::MeshPointDevice
 *
 * You need to create a MeshPointDevice and attach all
 * interfaces to it, than call Install method
 */
class MeshStack : public Object
{
public:
  /// \brief Register this type.
  /// \return The TypeId.
  static TypeId GetTypeId (void);
  
  /**
   * \brief Installs mesh stack. needed by helper only
   * \param mp the mesh point device
   * \returns true if successful
   */
  virtual bool InstallStack (Ptr<MeshPointDevice> mp) = 0;
  /**
   * Report statistics of a given mesh point
   * \param mp the mesh point device
   * \param os the output stream
   */
  virtual void Report (const Ptr<MeshPointDevice> mp, std::ostream& os) = 0;
  /**
   * Reset statistics of a given mesh point
   * \param mp the mesh point device
   */
  virtual void ResetStats (const Ptr<MeshPointDevice> mp) = 0;
};
}
#endif

