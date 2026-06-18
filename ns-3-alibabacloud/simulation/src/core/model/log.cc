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
#include "log.h"

#include <list>
#include <utility>
#include <iostream>
#include "assert.h"
#include <stdexcept>
#include "ns3/core-config.h"
#include "fatal-error.h"

#include <cstdlib>    // getenv
#include <cstring>    // strlen

/**
 * \file
 * \ingroup logging
 * ns3::LogComponent and related implementations.
 */


namespace ns3 {

/**
 * \ingroup logging
 * The Log TimePrinter.
 * This is private to the logging implementation.
 */
static TimePrinter g_logTimePrinter = 0;
/**
 * \ingroup logging
 * The Log NodePrinter.
 */
static NodePrinter g_logNodePrinter = 0;

/**
 * \ingroup logging
 * Handler for \c print-list token in NS_LOG
 * to print the list of log components.
 * This is private to the logging implementation.
 */
class PrintList
{
public:
  PrintList ();  //<! Constructor, prints the list and exits.
};

/**
 * Invoke handler for \c print-list in NS_LOG environment variable.
 * This is private to the logging implementation.
 */
static PrintList g_printList;


/* static */
LogComponent::ComponentList *
LogComponent::GetComponentList (void)
{
  static LogComponent::ComponentList components;
  return &components;
}


PrintList::PrintList ()
{
  const char *envVar = std::getenv ("NS_LOG");
  if (envVar == 0 || std::strlen (envVar) == 0)
    {
      return;
    }
  std::string env = envVar;
  std::string::size_type cur = 0;
  std::string::size_type next = 0;
  while (next != std::string::npos)
    {
      next = env.find_first_of (":", cur);
      std::string tmp = std::string (env, cur, next - cur);
      if (tmp == "print-list")
        {
          LogComponentPrintList ();
          exit (0);
          break;
        }
      cur = next + 1;
    }
}


LogComponent::LogComponent (const std::string & name,
                            const std::string & file,
                            const enum LogLevel mask /* = 0 */)
  : m_levels (0), m_mask (mask), m_name (name), m_file (file)
{
  EnvVarCheck ();

  LogComponent::ComponentList *components = GetComponentList ();
  for (LogComponent::ComponentList::const_iterator i = components->begin ();
       i != components->end ();
       i++)
    {
      if (i->first == name)
        {
          NS_FATAL_ERROR ("Log component \"" << name << "\" has already been registered once.");
        }
    }
  components->insert (std::make_pair (name, this));
}

LogComponent &
GetLogComponent (const std::string name)
{
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  LogComponent* ret;

  try
    {
      ret = components->at (name);
    }
  catch (std::out_of_range&)
    {
      NS_FATAL_ERROR ("Log component \"" << name << "\" does not exist.");
    }
  return *ret;
}

void
LogComponent::EnvVarCheck (void)
{
  const char *envVar = std::getenv ("NS_LOG");
  if (envVar == 0 || std::strlen (envVar) == 0)
    {
      return;
    }
  std::string env = envVar;

  std::string::size_type cur = 0;
  std::string::size_type next = 0;
  while (next != std::string::npos)
    {
      next = env.find_first_of (":", cur);
      std::string tmp = std::string (env, cur, next - cur);
      std::string::size_type equal = tmp.find ("=");
      std::string component;
      if (equal == std::string::npos)
        {
          component = tmp;
          if (component == m_name || component == "*" || component == "***")
            {
              int level = LOG_LEVEL_ALL | LOG_PREFIX_ALL;
              Enable ((enum LogLevel)level);
              return;
            }
        }
      else
        {
          component = tmp.substr (0, equal);
          if (component == m_name || component == "*")
            {
              int level = 0;
              std::string::size_type cur_lev;
              std::string::size_type next_lev = equal;
              bool pre_pipe = true;  // before the first '|', enables positional 'all', '*'
              do
                {
                  cur_lev = next_lev + 1;
                  next_lev = tmp.find ("|", cur_lev);
                  std::string lev = tmp.substr (cur_lev, next_lev - cur_lev);
                  if (lev == "error")
                    {
                      level |= LOG_ERROR;
                    }
                  else if (lev == "warn")
                    {
                      level |= LOG_WARN;
                    }
                  else if (lev == "debug")
                    {
                      level |= LOG_DEBUG;
                    }
                  else if (lev == "info")
                    {
                      level |= LOG_INFO;
                    }
                  else if (lev == "function")
                    {
                      level |= LOG_FUNCTION;
                    }
                  else if (lev == "logic")
                    {
                      level |= LOG_LOGIC;
                    }
                  else if ( pre_pipe && ( (lev == "all") || (lev == "*") ) )
                    {
                      level |= LOG_LEVEL_ALL;
                    }
                  else if ( (lev == "prefix_func") || (lev == "func") )
                    {
                      level |= LOG_PREFIX_FUNC;
                    }
                  else if ( (lev == "prefix_time") || (lev == "time") )
                    {
                      level |= LOG_PREFIX_TIME;
                    }
                  else if ( (lev == "prefix_node") || (lev == "node") )
                    {
                      level |= LOG_PREFIX_NODE;
                    }
                  else if ( (lev == "prefix_level") || (lev == "level") )
                    {
                      level |= LOG_PREFIX_LEVEL;
                    }
                  else if ( (lev == "prefix_all")
                            || (!pre_pipe && ( (lev == "all") || (lev == "*") ) )
                            )
                    {
                      level |= LOG_PREFIX_ALL;
                    }
                  else if (lev == "level_error")
                    {
                      level |= LOG_LEVEL_ERROR;
                    }
                  else if (lev == "level_warn")
                    {
                      level |= LOG_LEVEL_WARN;
                    }
                  else if (lev == "level_debug")
                    {
                      level |= LOG_LEVEL_DEBUG;
                    }
                  else if (lev == "level_info")
                    {
                      level |= LOG_LEVEL_INFO;
                    }
                  else if (lev == "level_function")
                    {
                      level |= LOG_LEVEL_FUNCTION;
                    }
                  else if (lev == "level_logic")
                    {
                      level |= LOG_LEVEL_LOGIC;
                    }
                  else if (lev == "level_all")
                    {
                      level |= LOG_LEVEL_ALL;
                    }
                  else if (lev == "**")
                    {
                      level |= LOG_LEVEL_ALL | LOG_PREFIX_ALL;
                    }

                  pre_pipe = false;
                }
              while (next_lev != std::string::npos);

              Enable ((enum LogLevel)level);
            }
        }
      cur = next + 1;
    }
}


bool
LogComponent::IsEnabled (const enum LogLevel level) const
{
  //  LogComponentEnableEnvVar ();
  return (level & m_levels) ? 1 : 0;
}

bool
LogComponent::IsNoneEnabled (void) const
{
  return m_levels == 0;
}

void
LogComponent::SetMask (const enum LogLevel level)
{
  m_mask |= level;
}

void
LogComponent::Enable (const enum LogLevel level)
{
  m_levels |= (level & ~m_mask);
}

void
LogComponent::Disable (const enum LogLevel level)
{
  m_levels &= ~level;
}

char const *
LogComponent::Name (void) const
{
  return m_name.c_str ();
}

std::string
LogComponent::File (void) const
{
  return m_file;
}

/* static */
std::string
LogComponent::GetLevelLabel (const enum LogLevel level)
{
  if (level == LOG_ERROR)
    {
      return "ERROR";
    }
  else if (level == LOG_WARN)
    {
      // whitespace left at the end for alignment
      return "WARN ";
    }
  else if (level == LOG_DEBUG)
    {
      return "DEBUG";
    }
  else if (level == LOG_INFO)
    {
      // whitespace left at the end for alignment
      return "INFO ";
    }
  else if (level == LOG_FUNCTION)
    {
      return "FUNCT";
    }
  else if (level == LOG_LOGIC)
    {
      return "LOGIC";
    }
  else
    {
      return "unknown";
    }
}

void
LogComponentEnable (char const *name, enum LogLevel level)
{
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  LogComponent::ComponentList::const_iterator i;
  for (i = components->begin ();
       i != components->end ();
       i++)
    {
      if (i->first.compare (name) == 0)
        {
          i->second->Enable (level);
          return;
        }
    }
  if (i == components->end ())
    {
      // nothing matched
      LogComponentPrintList ();
      NS_FATAL_ERROR ("Logging component \"" << name <<
                      "\" not found. See above for a list of available log components");
    }
}

void
LogComponentEnableAll (enum LogLevel level)
{
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  for (LogComponent::ComponentList::const_iterator i = components->begin ();
       i != components->end ();
       i++)
    {
      i->second->Enable (level);
    }
}

void
LogComponentDisable (char const *name, enum LogLevel level)
{
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  for (LogComponent::ComponentList::const_iterator i = components->begin ();
       i != components->end ();
       i++)
    {
      if (i->first.compare (name) == 0)
        {
          i->second->Disable (level);
          break;
        }
    }
}

void
LogComponentDisableAll (enum LogLevel level)
{
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  for (LogComponent::ComponentList::const_iterator i = components->begin ();
       i != components->end ();
       i++)
    {
      i->second->Disable (level);
    }
}

void
LogComponentPrintList (void)
{
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  for (LogComponent::ComponentList::const_iterator i = components->begin ();
       i != components->end ();
       i++)
    {
      std::cout << i->first << "=";
      if (i->second->IsNoneEnabled ())
        {
          std::cout << "0" << std::endl;
          continue;
        }
      if (i->second->IsEnabled (LOG_LEVEL_ALL))
        {
          std::cout << "all";
        }
      else
        {
          if (i->second->IsEnabled (LOG_ERROR))
            {
              std::cout << "error";
            }
          if (i->second->IsEnabled (LOG_WARN))
            {
              std::cout << "|warn";
            }
          if (i->second->IsEnabled (LOG_DEBUG))
            {
              std::cout << "|debug";
            }
          if (i->second->IsEnabled (LOG_INFO))
            {
              std::cout << "|info";
            }
          if (i->second->IsEnabled (LOG_FUNCTION))
            {
              std::cout << "|function";
            }
          if (i->second->IsEnabled (LOG_LOGIC))
            {
              std::cout << "|logic";
            }
        }
      if (i->second->IsEnabled (LOG_PREFIX_ALL))
        {
          std::cout << "|prefix_all";
        }
      else
        {
          if (i->second->IsEnabled (LOG_PREFIX_FUNC))
            {
              std::cout << "|func";
            }
          if (i->second->IsEnabled (LOG_PREFIX_TIME))
            {
              std::cout << "|time";
            }
          if (i->second->IsEnabled (LOG_PREFIX_NODE))
            {
              std::cout << "|node";
            }
          if (i->second->IsEnabled (LOG_PREFIX_LEVEL))
            {
              std::cout << "|level";
            }
        }
      std::cout << std::endl;
    }
}

/**
 * \ingroup logging
 * Check if a log component exists.
 * This is private to the logging implementation.
 *
 * \param [in] componentName The putative log component name.
 * \returns \c true if \c componentName exists.
 */
static bool ComponentExists (std::string componentName)
{
  char const*name = componentName.c_str ();
  LogComponent::ComponentList *components = LogComponent::GetComponentList ();
  LogComponent::ComponentList::const_iterator i;
  for (i = components->begin ();
       i != components->end ();
       i++)
    {
      if (i->first.compare (name) == 0)
        {
          return true;
        }
    }
  NS_ASSERT (i == components->end ());
  // nothing matched
  return false;
}

/**
 * \ingroup logging
 * Parse the \c NS_LOG environment variable.
 * This is private to the logging implementation.
 */
static void CheckEnvironmentVariables (void)
{
  const char *envVar = std::getenv ("NS_LOG");
  if (envVar == 0 || std::strlen (envVar) == 0)
    {
      return;
    }

  std::string env = envVar;
  std::string::size_type cur = 0;
  std::string::size_type next = 0;

  while (next != std::string::npos)
    {
      next = env.find_first_of (":", cur);
      std::string tmp = std::string (env, cur, next - cur);
      std::string::size_type equal = tmp.find ("=");
      std::string component;
      if (equal == std::string::npos)
        {
          // ie no '=' characters found
          component = tmp;
          if (ComponentExists (component) || component == "*" || component == "***")
            {
              return;
            }
          else
            {
              LogComponentPrintList ();
              NS_FATAL_ERROR ("Invalid or unregistered component name \"" << component <<
                              "\" in env variable NS_LOG, see above for a list of valid components");
            }
        }
      else
        {
          component = tmp.substr (0, equal);
          if (ComponentExists (component) || component == "*")
            {
              std::string::size_type cur_lev;
              std::string::size_type next_lev = equal;
              do
                {
                  cur_lev = next_lev + 1;
                  next_lev = tmp.find ("|", cur_lev);
                  std::string lev = tmp.substr (cur_lev, next_lev - cur_lev);
                  if (lev == "error"
                      || lev == "warn"
                      || lev == "debug"
                      || lev == "info"
                      || lev == "function"
                      || lev == "logic"
                      || lev == "all"
                      || lev == "prefix_func"
                      || lev == "func"
                      || lev == "prefix_time"
                      || lev == "time"
                      || lev == "prefix_node"
                      || lev == "node"
                      || lev == "prefix_level"
                      || lev == "level"
                      || lev == "prefix_all"
                      || lev == "level_error"
                      || lev == "level_warn"
                      || lev == "level_debug"
                      || lev == "level_info"
                      || lev == "level_function"
                      || lev == "level_logic"
                      || lev == "level_all"
                      || lev == "*"
                      || lev == "**"
                      )
                    {
                      continue;
                    }
                  else
                    {
                      NS_FATAL_ERROR ("Invalid log level \"" << lev <<
                                      "\" in env variable NS_LOG for component name " << component);
                    }
                }
              while (next_lev != std::string::npos);
            }
          else
            {
              LogComponentPrintList ();
              NS_FATAL_ERROR ("Invalid or unregistered component name \"" << component <<
                              "\" in env variable NS_LOG, see above for a list of valid components");
            }
        }
      cur = next + 1;   // parse next component
    }
}
void LogSetTimePrinter (TimePrinter printer)
{
  g_logTimePrinter = printer;
  /** \internal
   *  This is the only place where we are more or less sure that all log variables
   * are registered. See \bugid{1082} for details.
   */
  CheckEnvironmentVariables ();
}
TimePrinter LogGetTimePrinter (void)
{
  return g_logTimePrinter;
}

void LogSetNodePrinter (NodePrinter printer)
{
  g_logNodePrinter = printer;
}
NodePrinter LogGetNodePrinter (void)
{
  return g_logNodePrinter;
}


ParameterLogger::ParameterLogger (std::ostream &os)
  : m_first (true),
    m_os (os)
{}

template<>
ParameterLogger &
ParameterLogger::operator<< <std::string> (const std::string param)
{
  if (m_first)
    {
      m_os << "\"" << param << "\"";
      m_first = false;
    }
  else
    {
      m_os << ", \"" << param << "\"";
    }
  return *this;
}

template<>
ParameterLogger &
ParameterLogger::operator<< <const char *> (const char * param)
{
  (*this) << std::string (param);
  return *this;
}

template<>
ParameterLogger &
ParameterLogger::operator<< <int8_t> (const int8_t param)
{
  if (m_first)
    {
      m_os << static_cast<int16_t> (param);
      m_first = false;
    }
  else
    {
      m_os << ", " << static_cast<int16_t> (param);
    }
  return *this;
}

template<>
ParameterLogger &
ParameterLogger::operator<< <uint8_t> (const uint8_t param)
{
  if (m_first)
    {
      m_os << static_cast<uint16_t> (param);
      m_first = false;
    }
  else
    {
      m_os << ", " << static_cast<uint16_t> (param);
    }
  return *this;
}

} // namespace ns3
