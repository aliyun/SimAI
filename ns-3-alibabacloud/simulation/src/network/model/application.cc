/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 Georgia Tech Research Corporation
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
 * Author: George F. Riley<riley@ece.gatech.edu>
 */

// Implementation for ns3 Application base class.
// George F. Riley, Georgia Tech, Fall 2006

#include "application.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Application");

NS_OBJECT_ENSURE_REGISTERED (Application);

// Application Methods

TypeId 
Application::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Application")
    .SetParent<Object> ()
    .SetGroupName("Network")
    .AddAttribute ("StartTime", "Time at which the application will start",
                   TimeValue (Seconds (0.0)),
                   MakeTimeAccessor (&Application::m_startTime),
                   MakeTimeChecker ())
    .AddAttribute ("StopTime", "Time at which the application will stop",
                   TimeValue (TimeStep (0)),
                   MakeTimeAccessor (&Application::m_stopTime),
                   MakeTimeChecker ())
  ;
  return tid;
}

// \brief Application Constructor
Application::Application()
{
  NS_LOG_FUNCTION (this);
}

// \brief Application Destructor
Application::~Application()
{
  NS_LOG_FUNCTION (this);
}

void
Application::SetStartTime (Time start)
{
  NS_LOG_FUNCTION (this << start);
  m_startTime = start;
}
void
Application::SetStopTime (Time stop)
{
  NS_LOG_FUNCTION (this << stop);
  m_stopTime = stop;
}


void
Application::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  m_startEvent.Cancel ();
  m_stopEvent.Cancel ();
  Object::DoDispose ();
}

void
Application::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  m_startEvent = Simulator::Schedule (m_startTime, &Application::StartApplication, this);
  if (m_stopTime != TimeStep (0))
    {
      m_stopEvent = Simulator::Schedule (m_stopTime, &Application::StopApplication, this);
    }
  Object::DoInitialize ();
}

Ptr<Node> Application::GetNode () const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

void 
Application::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);
  m_node = node;
}

// Protected methods
// StartApp and StopApp will likely be overridden by application subclasses
void Application::StartApplication ()
{ // Provide null functionality in case subclass is not interested
  NS_LOG_FUNCTION (this);
}

void Application::StopApplication ()
{ // Provide null functionality in case subclass is not interested
  NS_LOG_FUNCTION (this);
}

} // namespace ns3


