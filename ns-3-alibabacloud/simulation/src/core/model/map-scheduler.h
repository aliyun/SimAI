/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
 */

#ifndef MAP_SCHEDULER_H
#define MAP_SCHEDULER_H

#include "scheduler.h"
#include <stdint.h>
#include <map>
#include <utility>

/**
 * \file
 * \ingroup scheduler
 * ns3::MapScheduler declaration.
 */

namespace ns3 {

/**
 * \ingroup scheduler
 * \brief a std::map event scheduler
 *
 * This class implements the an event scheduler using an std::map
 * data structure.
 *
 * \par Time Complexity
 *
 * Operation    | Amortized %Time | Reason
 * :----------- | :-------------- | :-----
 * Insert()     | Logarithmic     | `std::map::insert()`
 * IsEmpty()    | Constant        | `std::map::empty()`
 * PeekNext()   | Constant        | `std::map::begin()`
 * Remove()     | Logarithmic     | `std::map::find()`
 * RemoveNext() | Constant        | `std::map::begin()`
 *
 * \par Memory Complexity
 *
 * Category  | Memory                           | Reason
 * :-------- | :------------------------------- | :-----
 * Overhead  | 3 x `sizeof (*)` + 2 x `size_t`<br/>(40 bytes) | red-black tree
 * Per Event | 3 x `sizeof (*)` + `int`<br/>(32 bytes)        | red-black tree
 *
 */
class MapScheduler : public Scheduler
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /** Constructor. */
  MapScheduler ();
  /** Destructor. */
  virtual ~MapScheduler ();

  // Inherited
  virtual void Insert (const Scheduler::Event &ev);
  virtual bool IsEmpty (void) const;
  virtual Scheduler::Event PeekNext (void) const;
  virtual Scheduler::Event RemoveNext (void);
  virtual void Remove (const Scheduler::Event &ev);

private:
  /** Event list type: a Map from EventKey to EventImpl. */
  typedef std::map<Scheduler::EventKey, EventImpl*> EventMap;
  /** EventMap iterator. */
  typedef std::map<Scheduler::EventKey, EventImpl*>::iterator EventMapI;
  /** EventMap const iterator. */
  typedef std::map<Scheduler::EventKey, EventImpl*>::const_iterator EventMapCI;

  /** The event list. */
  EventMap m_list;
};

} // namespace ns3

#endif /* MAP_SCHEDULER_H */
