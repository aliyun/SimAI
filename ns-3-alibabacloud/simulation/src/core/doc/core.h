/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Peter D. Barnes, Jr.
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
 * Authors:
 *   Tom Henderson <tomh@tomh.org>
 *   Mitch Watrous <watrous@u.washington.edu>
 *   Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *   Peter D. Barnes, Jr. <pdbarnes\llnl.gov>
 */

/**
 * \file
 * \ingroup core
 * This file provides some doxygen documentation for the core module.
 */


/**
 * \defgroup core Core
 * \brief The "core" module contains:
 *    - a time management class to hold a time and convert between various
 *      time units: ns3::Time 
 *    - a scheduler base class used to implement new simulation event
 *      schedulers:       
 *      ns3::Scheduler and ns3::SchedulerFactory 
 *    - a simulator class used to create, schedule and cancel events:
 *      ns3::Simulator
 *    - a Functor class: ns3::Callback  
 *    - an os-independent interface to get access to the elapsed wall clock
 *      time: ns3::SystemWallClockMs 
 *    - a class to register regression tests with the test manager: ns3::Test
 *      and ns3::TestManager
 *    - debugging facilities: \ref debugging
 *    - \ref randomvariable
 *    - a base class for objects which need to support per-instance
 *      "attributes" and trace sources: ns3::ObjectBase
 *    - a base class for objects which need to support reference counting
 *      and dynamic object aggregation: ns3::Object
 *    - a smart-pointer class ns3::Ptr designed to work together with
 *      ns3::Object
 *    - a configuration class used to set and control all attributes and
 *      trace sources in a simulation: ns3::Config.
 *
 * \todo Move all Private and Impl classes to namespace ns3::impl?
 */

/**
 * \ingroup core
 * \defgroup debugging Debugging tools
 *
 * Assertions, breakpoints, logging, and abnormal program termination
 */

/**
 * \ingroup core
 * \defgroup core-examples Core module examples
 *  Programs which illustrate use of core module functionality.
 */

/**
 * \ingroup core
 * \defgroup core-helpers Core module helper classes
 * Helper classes for the Core module
 */

/**
 * \ingroup tests
 * \ingroup core
 * \defgroup core-tests Core module tests
 * TestSuites for the Core module
 */
