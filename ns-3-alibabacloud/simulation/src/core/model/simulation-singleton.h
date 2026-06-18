/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
#ifndef SIMULATION_SINGLETON_H
#define SIMULATION_SINGLETON_H

/**
 * \file
 * \ingroup core
 * ns3::SimulationSingleton declaration and template implementation.
 */

namespace ns3 {

/**
 * \ingroup core
 * This singleton class template ensures that the type
 * for which we want a singleton has a lifetime bounded
 * by the simulation run lifetime. That it, the underlying
 * type will be automatically deleted upon a call
 * to Simulator::Destroy.
 *
 * For a singleton with a lifetime bounded by the process,
 * not the simulation run, see Singleton.
 */
template <typename T>
class SimulationSingleton
{
public:
  // Delete default constructor, copy constructor and assignment operator to avoid misuse
  SimulationSingleton<T> () = delete;
  SimulationSingleton<T> (const SimulationSingleton<T> &) = delete;
  SimulationSingleton<T> & operator = (const SimulationSingleton<T> &) = delete;

  /**
   * Get a pointer to the singleton instance.
   *
   * This instance will be automatically deleted when the
   * simulation is destroyed by a call to Simulator::Destroy.
   *
   * \returns A pointer to the singleton instance.
   */
  static T * Get (void);

private:

  /**
   * Get the singleton object, creating a new one if it doesn't exist yet.
   *
   * \internal
   * When a new object is created, this method schedules it's own
   * destruction using Simulator::ScheduleDestroy().
   *
   * \returns The address of the pointer holding the static instance.
   */
  static T ** GetObject (void);

  /** Delete the static instance. */
  static void DeleteObject (void);
};

} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

#include "simulator.h"

namespace ns3 {

template <typename T>
T *
SimulationSingleton<T>::Get (void)
{
  T ** ppobject = GetObject ();
  return *ppobject;
}

template <typename T>
T **
SimulationSingleton<T>::GetObject (void)
{
  static T *pobject = 0;
  if (pobject == 0)
    {
      pobject = new T ();
      Simulator::ScheduleDestroy (&SimulationSingleton<T>::DeleteObject);
    }
  return &pobject;
}

template <typename T>
void
SimulationSingleton<T>::DeleteObject (void)
{
  T **ppobject = GetObject ();
  delete (*ppobject);
  *ppobject = 0;
}

} // namespace ns3


#endif /* SIMULATION_SINGLETON_H */
