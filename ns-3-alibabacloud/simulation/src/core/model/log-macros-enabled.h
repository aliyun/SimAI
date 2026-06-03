/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INRIA
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

#ifndef NS3_LOG_MACROS_ENABLED_H
#define NS3_LOG_MACROS_ENABLED_H

/**
 * \file
 * \ingroup logging
 * NS_LOG and related logging macro definitions.
 */


// These two implementation macros
//   NS_LOG_APPEND_TIME_PREFIX_IMPL
//   NS_LOG_APPEND_NODE_PREFIX_IMPL
// need to be defined in all configurations (debug, release, optimized)
// for use by NS_FATAL_...

/**
 * \ingroup logging
 * Implementation details for NS_LOG_APPEND_TIME_PREFIX.
 * \internal
 * Logging implementation macro; should not be called directly.
 * We define this separately so we can reuse the definition
 * in NS_FATAL.
 */
#define NS_LOG_APPEND_TIME_PREFIX_IMPL                      \
  do {                                                      \
    ns3::TimePrinter printer = ns3::LogGetTimePrinter ();   \
    if (printer != 0)                                       \
      {                                                     \
        (*printer)(std::clog);                              \
        std::clog << " ";                                   \
      }                                                     \
  } while (false)

/**
 * \ingroup logging
 * Implementation details for NS_LOG_APPEND_NODE_PREFIX.
 * \internal
 * Logging implementation macro; should not be called directly.
 * We define this separately so we can reuse the definition
 * in NS_FATAL.
 */
#define NS_LOG_APPEND_NODE_PREFIX_IMPL                      \
  do {                                                      \
    ns3::NodePrinter printer = ns3::LogGetNodePrinter ();   \
    if (printer != 0)                                       \
      {                                                     \
        (*printer)(std::clog);                              \
        std::clog << " ";                                   \
      }                                                     \
  } while (false)




#ifdef NS3_LOG_ENABLE

/**
 * \ingroup logging
 * Append the simulation time to a log message.
 * \internal
 * Logging implementation macro; should not be called directly.
 */
#define NS_LOG_APPEND_TIME_PREFIX                           \
  if (g_log.IsEnabled (ns3::LOG_PREFIX_TIME))               \
    {                                                       \
      NS_LOG_APPEND_TIME_PREFIX_IMPL;                       \
    }

/**
 * \ingroup logging
 * Append the simulation node id to a log message.
 * \internal
 * Logging implementation macro; should not be called directly.
 */
#define NS_LOG_APPEND_NODE_PREFIX                          \
  if (g_log.IsEnabled (ns3::LOG_PREFIX_NODE))              \
    {                                                      \
      NS_LOG_APPEND_NODE_PREFIX_IMPL;                      \
    }

/**
 * \ingroup logging
 * Append the function name to a log message.
 * \internal
 * Logging implementation macro; should not be called directly.
 */
#define NS_LOG_APPEND_FUNC_PREFIX                          \
  if (g_log.IsEnabled (ns3::LOG_PREFIX_FUNC))              \
    {                                                      \
      std::clog << g_log.Name () << ":"                    \
                << __FUNCTION__ << "(): ";                 \
    }                                                      \

/**
 * \ingroup logging
 * Append the log severity level to a log message.
 * \internal
 * Logging implementation macro; should not be called directly.
 */
#define NS_LOG_APPEND_LEVEL_PREFIX(level)                       \
  if (g_log.IsEnabled (ns3::LOG_PREFIX_LEVEL))                  \
    {                                                           \
      std::clog << "[" << g_log.GetLevelLabel (level) << "] ";  \
    }                                                           \


#ifndef NS_LOG_APPEND_CONTEXT
/**
 * \ingroup logging
 * Append the node id (or other file-local programmatic context, such as
 * MPI rank) to a log message.
 *
 * This is implemented locally in `.cc` files because
 * the relevant variable is only known there.
 *
 * Preferred format is something like (assuming the node id is
 * accessible from `var`:
 * \code
 *   if (var)
 *     {
 *       std::clog << "[node " << var->GetObject<Node> ()->GetId () << "] ";
 *     }
 * \endcode
 */
#define NS_LOG_APPEND_CONTEXT
#endif /* NS_LOG_APPEND_CONTEXT */


#ifndef NS_LOG_CONDITION
/**
 * \ingroup logging
 * Limit logging output based on some file-local condition,
 * such as MPI rank.
 *
 * This is implemented locally in `.cc` files because
 * the relevant condition variable is only known there.
 *
 * Since this appears immediately before the `do { ... } while false`
 * construct of \c NS_LOG(level, msg), it must have the form
 * \code
 *   #define NS_LOG_CONDITION    if (condition)
 * \endcode
 */
#define NS_LOG_CONDITION
#endif

/**
 * \ingroup logging
 *
 * This macro allows you to log an arbitrary message at a specific
 * log level.
 *
 * The log message is expected to be a C++ ostream
 * message such as "my string" << aNumber << "my oth stream".
 *
 * Typical usage looks like:
 * \code
 * NS_LOG (LOG_DEBUG, "a number="<<aNumber<<", anotherNumber="<<anotherNumber);
 * \endcode
 *
 * \param [in] level The log level
 * \param [in] msg The message to log
 * \internal
 * Logging implementation macro; should not be called directly.
 */
#define NS_LOG(level, msg)                                      \
  NS_LOG_CONDITION                                              \
  do {                                                          \
      if (g_log.IsEnabled (level))                              \
        {                                                       \
          NS_LOG_APPEND_TIME_PREFIX;                            \
          NS_LOG_APPEND_NODE_PREFIX;                            \
          NS_LOG_APPEND_CONTEXT;                                \
          NS_LOG_APPEND_FUNC_PREFIX;                            \
          NS_LOG_APPEND_LEVEL_PREFIX (level);                   \
          std::clog << msg << std::endl;                        \
        }                                                       \
    } while (false)

/**
 * \ingroup logging
 *
 * Output the name of the function.
 *
 * This should be used only in static functions; most member functions
 * should instead use NS_LOG_FUNCTION().
 */
#define NS_LOG_FUNCTION_NOARGS()                                \
  NS_LOG_CONDITION                                              \
  do {                                                          \
      if (g_log.IsEnabled (ns3::LOG_FUNCTION))                  \
        {                                                       \
          NS_LOG_APPEND_TIME_PREFIX;                            \
          NS_LOG_APPEND_NODE_PREFIX;                            \
          NS_LOG_APPEND_CONTEXT;                                \
          std::clog << g_log.Name () << ":"                     \
                    << __FUNCTION__ << "()" << std::endl;       \
        }                                                       \
    } while (false)


/**
 * \ingroup logging
 *
 * If log level LOG_FUNCTION is enabled, this macro will output
 * all input parameters separated by ", ".
 *
 * Typical usage looks like:
 * \code
 * NS_LOG_FUNCTION (aNumber<<anotherNumber);
 * \endcode
 * And the output will look like:
 * \code
 * Component:Function (aNumber, anotherNumber)
 * \endcode
 *
 * To facilitate function tracing, most functions should begin with
 * (at least) NS_LOG_FUNCTION(this).  Static functions should use
 * NS_LOG_FUNCTION_NOARGS() instead.
 *
 * \param [in] parameters The parameters to output.
 */
#define NS_LOG_FUNCTION(parameters)                             \
  NS_LOG_CONDITION                                              \
  do                                                            \
    {                                                           \
      if (g_log.IsEnabled (ns3::LOG_FUNCTION))                  \
        {                                                       \
          NS_LOG_APPEND_TIME_PREFIX;                            \
          NS_LOG_APPEND_NODE_PREFIX;                            \
          NS_LOG_APPEND_CONTEXT;                                \
          std::clog << g_log.Name () << ":"                     \
                    << __FUNCTION__ << "(";                     \
          ns3::ParameterLogger (std::clog) << parameters;       \
          std::clog << ")" << std::endl;                        \
        }                                                       \
    }                                                           \
  while (false)


/**
 * \ingroup logging
 *
 * Output the requested message unconditionally.
 *
 * \param [in] msg The message to log
 */
#define NS_LOG_UNCOND(msg)              \
  NS_LOG_CONDITION                      \
  do {                                  \
      std::clog << msg << std::endl;    \
    } while (false)


#endif /* NS3_LOG_ENABLE */

#endif /* NS3_LOG_MACROS_ENABLED_H */
