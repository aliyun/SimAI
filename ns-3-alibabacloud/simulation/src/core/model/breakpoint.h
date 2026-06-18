/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INESC Porto, INRIA
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
 * Author: Gustavo Carneiro <gjc@inescporto.pt>
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef BREAKPOINT_H
#define BREAKPOINT_H

/**
 * \file
 * \ingroup breakpoint
 * NS_BREAKPOINT() macro definition and ns3::BreakpointFallback
 * function declaration.
 */

namespace ns3 {

/* Hacker macro to place breakpoints for selected machines.
 * Actual use is strongly discouraged of course ;)
 * Copied from GLib 2.12.9.
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/.
 */

/**
 * \ingroup debugging
 * \defgroup breakpoint Breakpoints
 *
 * \brief Trigger a debugger breakpoint.
 */

/**
 * \ingroup breakpoint
 *
 * Inserts a breakpoint instruction (or equivalent system call) into
 * the code for selected machines.  When an NS_ASSERT cannot verify
 * its condition, this macro is used. Falls back to calling
 * ns3::BreakpointFallback() for architectures where breakpoint assembly
 * instructions are not supported.
 */
#if (defined (__i386__) || defined (__amd64__) || defined (__x86_64__)) && defined (__GNUC__) && __GNUC__ >= 2
#  define NS_BREAKPOINT() \
  do { __asm__ __volatile__ ("int $03"); } while(false)
#elif defined (_MSC_VER) && defined (_M_IX86)
#  define NS_BREAKPOINT() \
  do { __asm int 3h } while(false)
#elif defined (__alpha__) && !defined(__osf__) && defined (__GNUC__) && __GNUC__ >= 2
#  define NS_BREAKPOINT() \
  do { __asm__ __volatile__ ("bpt"); } while(false)
#else   /* !__i386__ && !__alpha__ */
#  define NS_BREAKPOINT()    ns3::BreakpointFallback ()
#endif

/**
 * \ingroup breakpoint
 *
 * \brief fallback breakpoint function
 *
 * This function is used by the NS_BREAKPOINT() macro as a fallback
 * for when breakpoint assembly instructions are not available.  It
 * attempts to halt program execution either by a raising SIGTRAP, on
 * unix systems, or by dereferencing a null pointer.
 *
 * Normally you should not call this function directly.
 */
void BreakpointFallback (void);


} // namespace ns3


#endif /* BREAKPOINT_H */
