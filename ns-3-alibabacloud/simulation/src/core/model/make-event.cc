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
 */

#include "make-event.h"
#include "log.h"

/**
 * \file
 * \ingroup events
 * ns3::MakeEvent(void(*f)(void)) implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MakeEvent");

// This is the only non-templated version of MakeEvent.
EventImpl * MakeEvent (void (*f)(void))
{
  NS_LOG_FUNCTION (f);
  // zero arg version
  class EventFunctionImpl0 : public EventImpl
  {
  public:
    typedef void (*F)(void);

    EventFunctionImpl0 (F function)
      : m_function (function)
    {}
    virtual ~EventFunctionImpl0 ()
    {}

  protected:
    virtual void Notify (void)
    {
      (*m_function)();
    }

  private:
    F m_function;
  } *ev = new EventFunctionImpl0 (f);
  return ev;
}

} // namespace ns3
