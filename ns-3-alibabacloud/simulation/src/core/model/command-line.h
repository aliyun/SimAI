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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <string>
#include <sstream>
#include <vector>

#include "callback.h"
#include "nstime.h"
#include "type-id.h"

/**
 * \file
 * \ingroup commandline
 * ns3::CommandLine declaration.
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup commandline Command Line Parsing
 *
 * A uniform way to specify program documentation,
 * allowed command line arguments and help strings,
 * and set any attribute or global value, all from
 * the command line directly.
 *
 * The main entry point is CommandLine
 */

/**
 * \ingroup commandline
 * \brief Parse command-line arguments
 *
 * Instances of this class can be used to parse command-line
 * arguments.  Programs can register a general usage message with
 * CommandLine::Usage, and arguments with CommandLine::AddValue.
 * Argument variable types with input streamers (`operator>>`)
 * can be set directly; more complex argument parsing
 * can be accomplished by providing a Callback.
 *
 * CommandLine also provides handlers for these standard arguments:
 * \verbatim
   --PrintGlobals:              Print the list of globals.
   --PrintGroups:               Print the list of groups.
   --PrintGroup=[group]:        Print all TypeIds of group.
   --PrintTypeIds:              Print all TypeIds.
   --PrintAttributes=[typeid]:  Print all attributes of typeid.
   --PrintVersion:              Print the ns-3 version.
   --PrintHelp:                 Print this help message. \endverbatim
 *
 * The more common \c \--version is a synonym for \c \--PrintVersion.
 *
 * The more common \c \--help is a synonym for \c \--PrintHelp; an example
 * is given below.
 *
 * CommandLine can also handle non-option arguments
 * (often called simply "positional" parameters: arguments which don't begin
 * with "-" or "--").  These can be parsed directly in to variables,
 * by registering arguments with AddNonOption in the order expected.
 * Additional non-option arguments encountered will be captured as strings.
 *
 * Finally, CommandLine processes Attribute and GlobalValue arguments.
 * Default values for specific attributes can be set using a shorthand
 * argument name.
 *
 * In use, arguments are given in the form
 * \verbatim
   --arg=value --toggle first-non-option\endverbatim
 * Most arguments expect a value, as in the first form, \c \--arg=value.
 * Toggles, corresponding to boolean arguments, can be given in any of
 * the forms
 * \verbatim
   --toggle1 --toggle2=1 --toggle3=t --toggle4=true \endverbatim
 * The first form changes the state of toggle1 from its default;
 * all the rest set the corresponding boolean variable to true.
 * \c 0, \c f and \c false are accepted to set the variable to false.
 * Option arguments can appear in any order on the command line,
 * even intermixed with non-option arguments.
 * The order of non-option arguments is preserved.
 *
 * Option arguments can be repeated on the command line; the last value given
 * will be the final value used.  For example,
 * \verbatim
   --arg=one --toggle=f --arg=another --toggle \endverbatim
 * The variable set by \c \--arg will end up with the value \c "another";
 * the boolean set by \c \--toggle will end up as \c true.
 *
 * Because arguments can be repeated it can be hard to decipher what
 * value each variable ended up with, especially when using boolean toggles.
 * Suggested best practice is for scripts to report the values of all items
 * settable through CommandLine, as done by the example below.
 *
 *
 * CommandLine can set the initial value of every attribute in the system
 * with the \c \--TypeIdName::AttributeName=value syntax, for example
 * \verbatim
   --Application::StartTime=3s \endverbatim
 * In some cases you may want to highlight the use of a particular
 * attribute for a simulation script.  For example, you might want
 * to make it easy to set the \c Application::StartTime using
 * the argument \c \--start, and have its help string show as part
 * of the help message.  This can be done using the
 * \link AddValue(const std::string&, const std::string&) AddValue (name, attributePath) \endlink
 * method.
 *
 * CommandLine can also set the value of every GlobalValue
 * in the system with the \c \--GlobalValueName=value syntax, for example
 * \verbatim
   --SchedulerType=HeapScheduler \endverbatim
 *
 * A simple example of CommandLine is in `src/core/example/``command-line-example.cc`
 * See that file for an example of handling non-option arguments.
 *
 * The heart of that example is this code:
 *
 * \code
 *    int         intArg  = 1;
 *    bool        boolArg = false;
 *    std::string strArg  = "strArg default";
 *
 *    CommandLine cmd (__FILE__);
 *    cmd.Usage ("CommandLine example program.\n"
 *               "\n"
 *               "This little program demonstrates how to use CommandLine.");
 *    cmd.AddValue ("intArg",  "an int argument",       intArg);
 *    cmd.AddValue ("boolArg", "a bool argument",       boolArg);
 *    cmd.AddValue ("strArg",  "a string argument",     strArg);
 *    cmd.AddValue ("anti",    "ns3::RandomVariableStream::Antithetic");
 *    cmd.AddValue ("cbArg",   "a string via callback", MakeCallback (SetCbArg));
 *    cmd.Parse (argc, argv);
 * \endcode
 * after which it prints the values of each variable.  (The \c SetCbArg function
 * is not shown here; see `src/core/example/``command-line-example.cc`)
 *
 * Here is the output from a few runs of that program:
 *
 * \verbatim
   $ ./ns3 run="command-line-example"
   intArg:   1
   boolArg:  false
   strArg:   "strArg default"
   cbArg:    "cbArg default"

   $ ./ns3 run="command-line-example --intArg=2 --boolArg --strArg=Hello --cbArg=World"
   intArg:   2
   boolArg:  true
   strArg:   "Hello"
   cbArg:    "World"

   $ ./ns3 run="command-line-example --help"
   ns3-dev-command-line-example-debug [Program Arguments] [General Arguments]

   CommandLine example program.

   This little program demonstrates how to use CommandLine.

   Program Arguments:
       --intArg:   an int argument [1]
       --boolArg:  a bool argument [false]
       --strArg:   a string argument [strArg default]
       --anti:     Set this RNG stream to generate antithetic values (ns3::RandomVariableStream::Antithetic) [false]
       --cbArg:    a string via callback

   General Arguments:
       --PrintGlobals:              Print the list of globals.
       --PrintGroups:               Print the list of groups.
       --PrintGroup=[group]:        Print all TypeIds of group.
       --PrintTypeIds:              Print all TypeIds.
       --PrintAttributes=[typeid]:  Print all attributes of typeid.
       --PrintVersion:              Print the ns-3 version.
       --PrintHelp:                 Print this help message. \endverbatim
 *
 * Having parsed the arguments, some programs will need to perform
 * some additional validation of the received values.  A common issue at this
 * point is to discover that the supplied arguments are incomplete or
 * incompatible.  Suggested best practice is to supply an error message
 * and the complete usage message.  For example,
 *
 * \code
 *   int value1;
 *   int value2;
 *
 *   CommandLine cmd (__FILE__);
 *   cmd.Usage ("...");
 *   cmd.AddValue ("value1", "first value", value1);
 *   cmd.AddValue ("value2", "second value", value1);
 *
 *   cmd.Parse (argc, argv);
 *
 *   if (value1 * value2 < 0)
 *     {
 *       std::cerr << "value1 and value2 must have the same sign!" << std::endl;
 *       std::cerr << cmd;
 *       exit (-1);
 *     }
 * \endcode
 *
 * Finally, note that for examples which will be run by \c test.py
 * the preferred declaration of a CommandLine instance is
 *
 * \code
 *     CommandLine cmd (__FILE__);
 * \endcode
 * This will ensure that the program usage and arguments can be added to
 * the Doxygen documentation automatically.
 */
class CommandLine
{
public:
  /** Constructor */
  CommandLine (void);
  /**
   * Construct and register the source file name.
   * This would typically be called by
   *     CommandLine cmd (__FILE__);
   *
   * This form is required to generate Doxygen documentation of the
   * arguments and options.
   *
   * \param [in] filename The source file name.
   */
  CommandLine (const std::string filename);
  /**
   * Copy constructor
   *
   * \param [in] cmd The CommandLine to copy from
   */
  CommandLine (const CommandLine &cmd);
  /**
   * Assignment
   *
   * \param [in] cmd The CommandLine to assign from
   * \return The CommandLine
   */
  CommandLine &operator = (const CommandLine &cmd);
  /** Destructor */
  ~CommandLine ();

  /**
   * Supply the program usage and documentation.
   *
   * \param [in] usage Program usage message to write with \c \--help.
   */
  void Usage (const std::string usage);

  /**
   * Add a program argument, assigning to POD
   *
   * \param [in] name The name of the program-supplied argument
   * \param [in] help The help text used by \c \--PrintHelp
   * \param [out] value A reference to the variable where the
   *        value parsed will be stored (if no value
   *        is parsed, this variable is not modified).
   */
  template <typename T>
  void AddValue (const std::string &name,
                 const std::string &help,
                 T &value);

  /**
   * Callback function signature for
   * AddValue(const std::string&,const std::string&,Callback<bool,const std::string>).
   *
   * \param [in] value The argument value.
   */
  typedef bool (* Callback) (const std::string value);

  /**
   * Add a program argument, using a Callback to parse the value
   *
   * \param [in] name The name of the program-supplied argument
   * \param [in] help The help text used by \c \--help
   * \param [in] callback A Callback function that will be invoked to parse and
   *   store the value.
   * \param [in] defaultValue Optional default value for argument.
   *
   * The callback should have the signature
   * CommandLine::Callback
   */
  void AddValue (const std::string &name,
                 const std::string &help,
                 ns3::Callback<bool, std::string> callback,
                 const std::string defaultValue = "");


  /**
   * Add a program argument as a shorthand for an Attribute.
   *
   * \param [in] name The name of the program-supplied argument.
   * \param [out] attributePath The fully-qualified name of the Attribute
   */
  void AddValue (const std::string &name,
                 const std::string &attributePath);

  /**
   * Add a non-option argument, assigning to POD
   *
   * \param [in] name The name of the program-supplied argument
   * \param [in] help The help text used by \c \--PrintHelp
   * \param [out] value A reference to the variable where the
   *        value parsed will be stored (if no value
   *        is parsed, this variable is not modified).
   */
  template <typename T>
  void AddNonOption (const std::string name, const std::string help, T & value);

  /**
   * Get extra non-option arguments by index.
   * This allows CommandLine to accept more non-option arguments than
   * have been configured explicitly with AddNonOption().
   *
   * This is only valid after calling Parse().
   *
   * \param [in] i The index of the non-option argument to return.
   * \return The i'th non-option argument, as a string.
   */
  std::string GetExtraNonOption (std::size_t i) const;

  /**
   * Get the total number of non-option arguments found,
   * including those configured with AddNonOption() and extra non-option
   * arguments.
   *
   * This is only valid after calling Parse().
   *
   * \returns the number of non-option arguments found.
   */
  std::size_t GetNExtraNonOptions (void) const;

  /**
   * Parse the program arguments
   *
   * \param [in] argc The 'argc' variable: number of arguments (including the
   *        main program name as first element).
   * \param [in] argv The 'argv' variable: a null-terminated array of strings,
   *        each of which identifies a command-line argument.
   *
   * Obviously, this method will parse the input command-line arguments and
   * will attempt to handle them all.
   *
   * As a side effect, this method saves the program basename, which
   * can be retrieved by GetName().
   */
  void Parse (int argc, char *argv[]);

  /**
   * Parse the program arguments.
   *
   * This version may be convenient when synthesizing arguments
   * programmatically.  Other than the type of argument this behaves
   * identically to Parse(int, char *)
   *
   * \param [in] args The vector of arguments.
   */
  void Parse (std::vector<std::string> args);

  /**
   * Get the program name
   *
   * \return The program name.  Only valid after calling Parse()
   */
  std::string GetName () const;

  /**
   * \brief Print program usage to the desired output stream
   *
   * Handler for \c \--PrintHelp and \c \--help:  print Usage(), argument names, and help strings
   *
   * Alternatively, an overloaded operator << can be used:
   * \code
   *       CommandLine cmd (__FILE__);
   *       cmd.Parse (argc, argv);
   *     ...
   *
   *       std::cerr << cmd;
   * \endcode
   *
   * \param [in,out] os The output stream to print on.
   */
  void PrintHelp (std::ostream &os) const;

  /**
   * Get the program version.
   *
   * \return The program version
   */
  std::string GetVersion () const;

  /**
   * Print ns-3 version to the desired output stream
   *
   * Handler for \c \--PrintVersion and \c \--version.
   *
   * \param [in,out] os The output stream to print on.
   */
  void PrintVersion (std::ostream &os) const;

private:

  /**
   * \ingroup commandline
   * \brief The argument abstract base class
   */
  class Item
  {
  public:
    std::string m_name;       /**< Argument label:  \c \--m_name=... */
    std::string m_help;       /**< Argument help string */
    virtual ~Item ();         /**< Destructor */
    /**
     * Parse from a string.
     *
     * \param [in] value The string representation
     * \return \c true if parsing the value succeeded
     */
    virtual bool Parse (const std::string value) = 0;
    /**
     * \return \c true if this item has a default value.
     */
    virtual bool HasDefault () const;
    /**
     * \return The default value
     */
    virtual std::string GetDefault () const = 0;
  };  // class Item

  /**
   * \ingroup commandline
   *\brief An argument Item assigning to POD
   */
  template <typename T>
  class UserItem : public Item
  {
  public:
    // Inherited
    virtual bool Parse (const std::string value);
    bool HasDefault () const;
    std::string GetDefault () const;

    T *m_valuePtr;            /**< Pointer to the POD location */
    std::string m_default;    /**< String representation of default value */
  };  // class UserItem

  /**
   * \ingroup commandline
   * \brief Extension of Item for strings.
   */
  class StringItem : public Item
  {
  public:
    // Inherited
    bool Parse (const std::string value);
    bool HasDefault (void) const;
    std::string GetDefault (void) const;

    std::string m_value;     /**< The argument value. */
  };  // class StringItem

  /**
   * \ingroup commandline
   * \brief An argument Item using a Callback to parse the input
   */
  class CallbackItem : public Item
  {
  public:
    // Inherited
    bool HasDefault (void) const;
    std::string GetDefault (void) const;

    /**
     * Parse from a string.
     *
     * \param [in] value The string representation
     * \return \c true if parsing the value succeeded
     */
    virtual bool Parse (const std::string value);
    ns3::Callback<bool, std::string> m_callback;  /**< The Callback */
    std::string m_default;  /**< The default value, as a string, if it exists. */
  };  // class CallbackItem


  /**
   * Handle an option in the form \c param=value.
   *
   * \param [in] param The option string.
   * \returns \c true if this was really an option.
   */
  bool HandleOption (const std::string & param) const;

  /**
   * Handle a non-option
   *
   * \param [in] value The command line non-option value.
   * \return \c true if \c value could be parsed correctly.
   */
  bool HandleNonOption (const std::string &value);

  /**
   * Match name against the program or general arguments,
   * and dispatch to the appropriate handler.
   *
   * \param [in] name The argument name
   * \param [in] value The command line value
   */
  void HandleArgument (const std::string &name, const std::string &value) const;
  /**
   * Callback function to handle attributes.
   *
   * \param [in] name The full name of the Attribute.
   * \param [in] value The value to assign to \pname{name}.
   * \return \c true if the value was set successfully, false otherwise.
   */
  static bool HandleAttribute (const std::string name, const std::string value);

  /**
   * Handler for \c \--PrintGlobals:  print all global variables and values
   * \param [in,out] os The output stream to print on.
   */
  void PrintGlobals (std::ostream &os) const;
  /**
   * Handler for \c \--PrintAttributes:  print the attributes for a given type
   * as well as its parents.
   *
   * \param [in,out] os the output stream.
   * \param [in] type The type name whose Attributes should be displayed,
   */
  void PrintAttributes (std::ostream &os, const std::string &type) const;
  /**
   * Print the Attributes for a single type.
   *
   * \param [in,out] os the output stream.
   * \param [in] tid The TypeId whose Attributes should be displayed,
   * \param [in] header A header line to print if \c tid has Attributes
   */
  void PrintAttributeList (std::ostream &os, const TypeId tid, std::stringstream & header) const;
  /**
   * Handler for \c \--PrintGroup:  print all types belonging to a given group.
   *
   * \param [in,out] os The output stream.
   * \param [in] group The name of the TypeId group to display
   */
  void PrintGroup (std::ostream &os, const std::string &group) const;
  /**
   * Handler for \c \--PrintTypeIds:  print all TypeId names.
   *
   * \param [in,out] os The output stream.
   */
  void PrintTypeIds (std::ostream &os) const;
  /**
   * Handler for \c \--PrintGroups:  print all TypeId group names
   *
   * \param [in,out] os The output stream.
   */
  void PrintGroups (std::ostream &os) const;
  /**
   * Copy constructor
   *
   * \param [in] cmd CommandLine to copy
   */
  void Copy (const CommandLine &cmd);
  /** Remove all arguments, Usage(), name */
  void Clear (void);
  /**
   * Append usage message in Doxygen format to the file indicated
   * by the NS_COMMANDLINE_INTROSPECTION environment variable.
   * This is typically only called once, by Parse().
   */
  void PrintDoxygenUsage (void) const;

  typedef std::vector<Item *> Items;    /**< Argument list container */
  Items m_options;                      /**< The list of option arguments */
  Items m_nonOptions;                   /**< The list of non-option arguments */
  std::size_t m_NNonOptions;            /**< The expected number of non-option arguments */
  std::size_t m_nonOptionCount;         /**< The number of actual non-option arguments seen so far. */
  std::string m_usage;                  /**< The Usage string */
  std::string m_shortName;              /**< The source file name (without `.cc`), as would be given to `ns3 run` */

};  // class CommandLine


/** \ingroup commandline
 *  \defgroup commandlinehelper Helpers to Specialize on bool
 */
/**
 * \ingroup commandlinehelper
 * \brief Helpers for CommandLine to specialize on bool
 */
namespace CommandLineHelper {

/**
 * \ingroup commandlinehelper
 * \brief Helpers to specialize CommandLine::UserItem::Parse()
 *
 * \param [in] value The argument name
 * \param [out] val The argument location
 * \tparam T \deduced The type being specialized
 * \return \c true if parsing was successful
 */
template <typename T>
bool UserItemParse (const std::string value, T & val);
/**
 * \brief Specialization of CommandLine::UserItem to \c bool
 *
 * \param [in] value The argument name
 * \param [out] val The boolean variable to set
 * \return \c true if parsing was successful
 */
template <>
bool UserItemParse<bool> (const std::string value, bool & val);
/**
 * \brief Specialization of CommandLine::UserItem to \c uint8_t
 * to distinguish from \c char
 *
 * \param [in] value The argument name
 * \param [out] val The \c uint8_t variable to set
 * \return \c true if parsing was successful
 */
template <>
bool UserItemParse<uint8_t> (const std::string value, uint8_t & val);

/**
 * \ingroup commandlinehelper
 * \brief Helper to specialize CommandLine::UserItem::GetDefault() on types
 * needing special handling.
 *
 * \param [in] val The argument value
 * \return The string representation of value
 * @{
 */
template <typename T> std::string GetDefault       (const T & val);
template <>           std::string GetDefault<bool> (const bool & val);
template <>           std::string GetDefault<Time> (const Time & val);
/**@}*/

}  // namespace CommandLineHelper


} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

namespace ns3 {

template <typename T>
void
CommandLine::AddValue (const std::string &name,
                       const std::string &help,
                       T &value)
{
  UserItem<T> *item = new UserItem<T> ();
  item->m_name = name;
  item->m_help = help;
  item->m_valuePtr = &value;

  std::stringstream ss;
  ss << value;
  ss >> item->m_default;

  m_options.push_back (item);
}

template <typename T>
void
CommandLine::AddNonOption (const std::string name,
                           const std::string help,
                           T & value)
{
  UserItem<T> *item = new UserItem<T> ();
  item->m_name = name;
  item->m_help = help;
  item->m_valuePtr = &value;

  std::stringstream ss;
  ss << value;
  ss >> item->m_default;
  m_nonOptions.push_back (item);
  ++m_NNonOptions;

}

template <typename T>
bool
CommandLine::UserItem<T>::HasDefault () const
{
  return (m_default.size () > 0);
}

template <typename T>
std::string
CommandLine::UserItem<T>::GetDefault () const
{
  return CommandLineHelper::GetDefault<T> (*m_valuePtr);
}

template <typename T>
std::string
CommandLineHelper::GetDefault (const T & val)
{
  std::ostringstream oss;
  oss << val;
  return oss.str ();
}

template <typename T>
bool
CommandLine::UserItem<T>::Parse (const std::string value)
{
  return CommandLineHelper::UserItemParse<T> (value, *m_valuePtr);
}

template <typename T>
bool
CommandLineHelper::UserItemParse (const std::string value, T & val)
{
  std::istringstream iss;
  iss.str (value);
  iss >> val;
  return !iss.bad () && !iss.fail ();
}

/**
 * Overloaded operator << to print program usage
 * (shortcut for CommandLine::PrintHelper)
 *
 * \see CommandLine::PrintHelper
 *
 * Example usage:
 * \code
 *    CommandLine cmd (__FILE__);
 *    cmd.Parse (argc, argv);
 *    ...
 *
 *    std::cerr << cmd;
 * \endcode
 *
 * \param [in,out] os The stream to print on.
 * \param [in] cmd The CommandLine describing the program.
 * \returns The stream.
 */
std::ostream & operator << (std::ostream & os, const CommandLine & cmd);

} // namespace ns3

#endif /* COMMAND_LINE_H */
