/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA, 2010 NICTA
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
 * Author:  Original author unknown
 *          Quincy Tse <quincy.tse@nicta.com.au>
 */
#ifndef NS3_ABORT_H
#define NS3_ABORT_H

#include "fatal-error.h"

/**
 * \file
 * \ingroup fatal
 * \brief \c NS_ABORT_x macro definitions.
 */

/**
 * \ingroup fatal
 *
 * \brief Unconditional abnormal program termination with a message.
 *
 * \param [in] msg The message to output when this macro is hit.
 *
 * This macro is essentially equivalent to NS_FATAL_ERROR,
 * except it prepends the error message with the string
 * "aborted. ". When this macro is hit at runtime, the
 * program will be halted using \c std::terminate, which
 * triggers clean up code registered by \c std::set_terminate.
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 *
 * \see NS_FATAL_ERROR
 */
#define NS_ABORT_MSG(msg)                                              \
  do {                                                                 \
      std::cerr << "aborted. ";                                        \
      NS_FATAL_ERROR (msg);                                            \
    } while (false)


/**
 * \ingroup fatal
 *
 * \brief Abnormal program termination if a condition is \c true.
 *
 * \param [in] cond The condition to be evaluated.
 *
 * This is similar to \c NS_ASSERT(!(cond)), except this check
 * is enabled in all builds. If \c cond is evaluated to \c true,
 * the expression for \c cond is printed to \c stderr,
 * followed by a call to the NS_FATAL_ERROR_NO_MSG() macro
 * which prints the details of filename and line number to
 * \c stderr. The program will be halted by calling
 * \c std::terminate(), triggering any clean up code registered
 * by \c std::set_terminate.  The ns-3 default is a stream-flushing
 * code, but may be overridden.
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_ABORT_IF(cond)                                              \
  do {                                                                 \
      if (cond)                                                        \
        {                                                              \
          std::cerr << "aborted. cond=\"" << # cond << ", ";           \
          NS_FATAL_ERROR_NO_MSG ();                                    \
        }                                                              \
    } while (false)

/**
 * \ingroup fatal
 *
 * \brief Abnormal program termination if a condition is \c true,
 * with a message.
 *
 * \param [in] cond The condition to be evaluated.
 * \param [in] msg The message to output when cond is \c true.
 *
 * This is similar to NS_ASSERT_MSG(!(cond)), except this
 * check is enabled in all builds. If \c cond is evaluated to
 * \c true, the expression for \c cond is printed to
 * \c stderr, followed by a call to the NS_FATAL_ERROR() macro
 * which prints the user-specified error message, and details
 * of filename and line number to \c stderr. The program will
 * be halted by calling \c std::terminate(), triggering any
 * clean up code registered by \c std::set_terminate. The ns-3
 * default is a stream-flushing code, but may be overridden.
 *
 * This macro is enabled unconditionally in all builds,
 * including debug and optimized builds.
 */
#define NS_ABORT_MSG_IF(cond, msg)                                     \
  do {                                                                 \
      if (cond)                                                        \
        {                                                              \
          std::cerr << "aborted. cond=\"" << # cond << "\", ";         \
          NS_FATAL_ERROR (msg);                                        \
        }                                                              \
    } while (false)

/**
 * \ingroup fatal
 *
 * \brief Abnormal program termination if a condition is \c false.
 *
 * \param [in] cond The condition to be evaluated.
 *
 * This is an alias for NS_ABORT_IF(!(cond))
 *
 * \see NS_ABORT_IF
 */
#define NS_ABORT_UNLESS(cond)                                          \
  NS_ABORT_IF (!(cond))

/**
 * \ingroup fatal
 *
 * \brief Abnormal program termination if a condition is \c false,
 * with a message.
 *
 * \param [in] cond The condition to be evaluated.
 * \param [in] msg The message to output if cond is false.
 *
 * This is an alias for NS_ABORT_MSG_IF(!(cond))
 *
 * \see NS_ABORT_MSG_IF
 */
#define NS_ABORT_MSG_UNLESS(cond, msg)                                 \
  NS_ABORT_MSG_IF (!(cond),msg)

#endif /* NS3_ABORT_H */
