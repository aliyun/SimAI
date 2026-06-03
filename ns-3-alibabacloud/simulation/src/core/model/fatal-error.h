/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA, 2010 NICTA
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
 *         Quincy Tse <quincy.tse@nicta.com.au>
 */
#ifndef NS3_FATAL_ERROR_H
#define NS3_FATAL_ERROR_H

#include <iostream>
#include <exception>
#include <cstdlib>

#include "fatal-impl.h"
#include "log.h"  // NS_LOG_APPEND...

/**
 * \file
 * \ingroup fatal
 * \brief \c NS_FATAL_x macro definitions.
 */

/**
 * \ingroup core
 * \defgroup fatal Fatal Error Handlers
 *
 * \brief Functions to help clean up when a fatal error
 * is encountered.
 *
 * The functions in this group are used to perform
 * limited clean up, like flushing active streams, when
 * fatal errors are encountered (through assertion fail,
 * calls to NS_ABORT_* or calls to NS_FATAL_ERROR).
 *
 * Currently, other than flushing active ostreams, these
 * functions does not interfere with outside memory.  There
 * is still a residual risk that invalid ostream
 * pointers may be present, and may corrupt the memory
 * on the attempt to execute the flush() function.
 */

/**
 * \ingroup fatal
 *
 * \brief Fatal error reporting with no message, implementation.
 *
 * When this macro is hit at runtime the error details will
 * printed to \c stderr, including the file name and line number.
 * Optionally, if \c fatal is true, the macro
 * will invoke \c std::terminate().  If \c fatal is false,
 * the invoking function should return an error code to its caller,
 * which is expected to call NS_FATAL_ERROR to cause termination.
 *
 * \param [in] fatal Call \c std::terminate() if true.
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_FATAL_ERROR_IMPL_NO_MSG(fatal) \
  do                                                      \
    {                                                     \
      NS_LOG_APPEND_TIME_PREFIX_IMPL;                     \
      NS_LOG_APPEND_NODE_PREFIX_IMPL;                     \
      std::cerr << "file=" << __FILE__ << ", line=" <<    \
        __LINE__ << std::endl;                            \
      ::ns3::FatalImpl::FlushStreams ();                  \
      if (fatal) std::terminate ();                       \
    }                                                     \
  while (false)


/**
 * \ingroup fatal
 *
 * \brief Fatal error reporting with a message, implementation.
 *
 * When this macro is hit at runtime the error details will
 * printed to \c stderr, including the message, file name and line number.
 * Optionally, if \c fatal is true, the macro
 * will invoke \c std::terminate().  If \c fatal is false,
 * the invoking function should return an error code to its caller,
 * which is expected to call NS_FATAL_ERROR to cause termination.
 *
 * \param [in] msg The error message to print, if not empty.
 * \param [in] fatal Call \c std::terminate() if true.
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_FATAL_ERROR_IMPL(msg,fatal) \
  do                                                    \
    {                                                   \
      std::cerr << "msg=\"" << msg << "\", ";           \
      NS_FATAL_ERROR_IMPL_NO_MSG (fatal);               \
    }                                                   \
  while (false)


/**
 * \ingroup fatal
 *
 * \brief Report a fatal error and terminate.
 *
 * When this macro is hit at runtime, details of filename
 * and line number are printed to \c stderr, and the program
 * is halted by calling \c std::terminate(). This will
 * trigger any clean up code registered by
 * \c std::set_terminate (NS3 default is a stream-flushing
 * code), but may be overridden.
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_FATAL_ERROR_NO_MSG()       NS_FATAL_ERROR_IMPL_NO_MSG (true)

/**
 * \ingroup fatal
 *
 * \brief Report a fatal error, deferring termination.
 *
 * When this macro is hit at runtime, details of filename
 * and line number are printed to \c stderr, however the program
 * is _not_ halted.  It is expected that the function using this
 * macro will return an error code, and its caller will
 * invoke NS_FATAL_ERROR(msg) triggering std::terminate().
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_FATAL_ERROR_NO_MSG_CONT()  NS_FATAL_ERROR_IMPL_NO_MSG (false)

/**
 * \ingroup fatal
 *
 * \brief Report a fatal error with a message and terminate.
 *
 * \param [in] msg message to output when this macro is hit.
 *
 * When this macro is hit at runtime, the user-specified
 * error message are printed to \c stderr, followed by a call
 * to the NS_FATAL_ERROR_NO_MSG() macro which prints the
 * details of filename and line number to \c stderr. The
 * program will be halted by calling \c std::terminate(),
 * triggering any clean up code registered by
 * \c std::set_terminate (NS3 default is a stream-flushing
 * code, but may be overridden).
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_FATAL_ERROR(msg)           NS_FATAL_ERROR_IMPL (msg, true)

/**
 * \ingroup fatal
 *
 * \brief Report a fatal error with a message, deferring termination.
 *
 * When this macro is hit at runtime, details of filename
 * and line number are printed to \c stderr, however the program
 * is _not_ halted.  It is expected that the function using this
 * macro will return an error code, and its caller will
 * invoke NS_FATAL_ERROR(msg) triggering \c std::terminate().
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_FATAL_ERROR_CONT(msg)      NS_FATAL_ERROR_IMPL (msg, false)


#endif /* FATAL_ERROR_H */
