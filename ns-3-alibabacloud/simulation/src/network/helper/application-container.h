/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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

#ifndef APPLICATION_CONTAINER_H
#define APPLICATION_CONTAINER_H

#include <stdint.h>
#include <vector>
#include "ns3/application.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

/**
 * \brief holds a vector of ns3::Application pointers.
 *
 * Typically ns-3 Applications are installed on nodes using an Application
 * helper.  The helper Install method takes a NodeContainer which holds 
 * some number of Ptr<Node>.  For each of the Nodes in the NodeContainer
 * the helper will instantiate an application, install it in a node and
 * add a Ptr<Application> to that application into a Container for use
 * by the caller.  This is that container used to hold the Ptr<Application>
 * which are instantiated by the Application helper.
 */
class ApplicationContainer
{
public:
  /**
   * Create an empty ApplicationContainer.
   */
  ApplicationContainer ();

  /**
   * Create an ApplicationContainer with exactly one application which has
   * been previously instantiated.  The single application is specified
   * by a smart pointer.
   *
   * \param application The Ptr<Application> to add to the container.
   */
  ApplicationContainer (Ptr<Application> application);

  /**
   * Create an ApplicationContainer with exactly one application which has
   * been previously instantiated and assigned a name using the Object Name
   * Service.  This Application is then specified by its assigned name. 
   *
   * \param name The name of the Application Object to add to the container.
   */
  ApplicationContainer (std::string name);

  /// Application container iterator
  typedef std::vector<Ptr<Application> >::const_iterator Iterator;

  /**
   * \brief Get an iterator which refers to the first Application in the 
   * container.
   *
   * Applications can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a 
   * for-loop to run through the Applications
   *
   * \code
   *   ApplicationContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some Application method
   *     }
   * \endcode
   *
   * \returns an iterator which refers to the first Application in the container.
   */
  Iterator Begin (void) const;

  /**
   * \brief Get an iterator which indicates past-the-last Application in the 
   * container.
   *
   * Applications can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a 
   * for-loop to run through the Applications
   *
   * \code
   *   ApplicationContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some Application method
   *     }
   * \endcode
   *
   * \returns an iterator which indicates an ending condition for a loop.
   */
  Iterator End (void) const;

  /**
   * \brief Get the number of Ptr<Application> stored in this container.
   *
   * Applications can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is typically used to
   * define an ending condition in a for-loop that runs through the stored
   * Applications
   *
   * \code
   *   uint32_t nApplications = container.GetN ();
   *   for (uint32_t i = 0 i < nApplications; ++i)
   *     {
   *       Ptr<Application> p = container.Get (i)
   *       i->method ();  // some Application method
   *     }
   * \endcode
   *
   * \returns the number of Ptr<Application> stored in this container.
   */
  uint32_t GetN (void) const;

  /**
   * \brief Get the Ptr<Application> stored in this container at a given
   * index.
   *
   * Applications can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is used to retrieve the
   * indexed Ptr<Appliation>.
   *
   * \code
   *   uint32_t nApplications = container.GetN ();
   *   for (uint32_t i = 0 i < nApplications; ++i)
   *     {
   *       Ptr<Application> p = container.Get (i)
   *       i->method ();  // some Application method
   *     }
   * \endcode
   *
   * \param i the index of the requested application pointer.
   * \returns the requested application pointer.
   */
  Ptr<Application> Get (uint32_t i) const;

  /**
   * \brief Append the contents of another ApplicationContainer to the end of
   * this container.
   *
   * \param other The ApplicationContainer to append.
   */
  void Add (ApplicationContainer other);

  /**
   * \brief Append a single Ptr<Application> to this container.
   *
   * \param application The Ptr<Application> to append.
   */
  void Add (Ptr<Application> application);

  /**
   * \brief Append to this container the single Ptr<Application> referred to
   * via its object name service registered name.
   *
   * \param name The name of the Application Object to add to the container.
   */
  void Add (std::string name);

  /**
   * \brief Arrange for all of the Applications in this container to Start()
   * at the Time given as a parameter.
   *
   * All Applications need to be provided with a starting simulation time and
   * a stopping simulation time.  The ApplicationContainer is a convenient 
   * place for allowing all of the contained Applications to be told to wake
   * up and start doing their thing (Start) at a common time.
   *
   * This method simply iterates through the contained Applications and calls
   * their Start() methods with the provided Time.
   *
   * \param start The Time at which each of the applications should start.
   */
  void Start (Time start);

  /**
   * \brief Start all of the Applications in this container at the start time 
   * given as a parameter, plus some jitter.
   *
   * This method iterates through the contained Applications and calls
   * their Start() methods with the provided start Time, plus a jitter value
   * drawn from the provided random variable.
   *
   * \param start The Time at which each of the applications should start.
   * \param rv The random variable that adds jitter (units of seconds)
   */
  void StartWithJitter (Time start, Ptr<RandomVariableStream> rv);

  /**
   * \brief Arrange for all of the Applications in this container to Stop()
   * at the Time given as a parameter.
   *
   * All Applications need to be provided with a starting simulation time and
   * a stopping simulation time.  The ApplicationContainer is a convenient 
   * place for allowing all of the contained Applications to be told to shut
   * down and stop doing their thing (Stop) at a common time.
   *
   * This method simply iterates through the contained Applications and calls
   * their Stop() methods with the provided Time.
   *
   * \param stop The Time at which each of the applications should stop.
   */
  void Stop (Time stop);

private:
  std::vector<Ptr<Application> > m_applications; //!< Applications smart pointers
};

} // namespace ns3

#endif /* APPLICATION_CONTAINER_H */
