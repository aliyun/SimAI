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
#ifndef SINGLETON_H
#define SINGLETON_H

/**
 * \file
 * \ingroup access
 * ns3::Singleton declaration and template implementation.
 */

namespace ns3 {

/**
 * \ingroup access
 * \brief A template singleton
 *
 * This template class can be used to implement the singleton pattern.
 * The underlying object will be destroyed automatically when the process
 * exits.
 *
 * For a singleton whose lifetime is bounded by the simulation run,
 * not the process, see SimulationSingleton.
 *
 * To force your `class ExampleS` to be a singleton, inherit from Singleton:
 * \code
 *   class ExampleS : public Singleton<ExampleS> { ... };
 * \endcode
 *
 * Then, to reach the singleton instance, just do
 * \code
 *   ExampleS::Get ()->...;
 * \endcode
 *
 * \note
 * If you call Get() again after the object has
 * been destroyed, the object will be re-created which will result in a
 * memory leak as reported by most memory leak checkers. It is up to the
 * user to ensure that Get() is never called from a static variable
 * finalizer.
 */
template <typename T>
class Singleton
{
public:
  // Delete copy constructor and assignment operator to avoid misuse
  Singleton<T> (const Singleton<T> &) = delete;
  Singleton<T> & operator = (const Singleton<T> &) = delete;

  /**
   * Get a pointer to the singleton instance.
   *
   * The instance will be automatically deleted when
   * the process exits.
   *
   * \return A pointer to the singleton instance.
   */
  static T * Get (void);

protected:
  /** Constructor. */
  Singleton<T> ()
  {
  }

  /** Destructor. */
  virtual ~Singleton<T> ()
  {
  }
};

} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

namespace ns3 {

template <typename T>
T *
Singleton<T>::Get (void)
{
  static T object;
  return &object;
}


} // namespace ns3

#endif /* SINGLETON_H */
