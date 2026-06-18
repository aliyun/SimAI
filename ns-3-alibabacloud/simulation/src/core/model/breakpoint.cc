/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INRIA, INESC Porto
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
 * Author: Gustavo Carneiro <gjc@inescporto.pt>
 */

#include "breakpoint.h"
#include "ns3/core-config.h"
#include "log.h"
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif

namespace ns3 {

/**
 * \file
 * \ingroup breakpoint
 * ns3::BreakpointFallback function implementation.
 */

NS_LOG_COMPONENT_DEFINE ("Breakpoint");

#if defined (HAVE_SIGNAL_H) && defined (SIGTRAP)

void
BreakpointFallback (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  raise (SIGTRAP);
}

#else

void
BreakpointFallback (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  int *a = 0;
  /**
   * we test here to allow a debugger to change the value of
   * the variable 'a' to allow the debugger to avoid the
   * subsequent segfault.
   */
  if (a == 0)
    {
      *a = 0;
    }
}

#endif // HAVE_SIGNAL_H

} // namespace ns3
