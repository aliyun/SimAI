/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Authors: Tiago G. Rodrigues (tgr002@bucknell.edu)
 */

#ifndef DATA_COLLECTION_OBJECT_H
#define DATA_COLLECTION_OBJECT_H

#include <string>
#include "ns3/object.h"

namespace ns3 {

/**
 * \ingroup aggregator
 *
 * Base class for data collection framework objects.
 *
 * All data collection objects have 1) a string name, and 2) enabled
 * or disabled status.
 */
class DataCollectionObject : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  DataCollectionObject ();
  virtual ~DataCollectionObject ();

  /// Set the status of an individual object.
  void Enable (void);
  /// Unset the status of an individual object.
  void Disable (void);

  /**
   * Check the status of an individual object.
   * \return true if the object is enabled
   */
  virtual bool IsEnabled (void) const;

  /**
   * Get the object's name.
   * \return the object's name
   */
  std::string GetName (void) const;

  /**
   * Set the object's name.
   * \param name the object's name
   */
  void SetName (std::string name);

protected:
  /// Object's activation state.
  bool m_enabled;

  /// Name of the object within the data collection framework
  std::string m_name;

};

} // namespace ns3

#endif // DATA_COLLECTION_OBJECT_H
