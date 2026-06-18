/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Natale Patriciello <natale.patriciello@gmail.com>
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
#ifndef SQLITE_OUTPUT_H
#define SQLITE_OUTPUT_H

#include "ns3/simple-ref-count.h"
#include <sqlite3.h>
#include <string>
#include <mutex>

namespace ns3 {

/**
 * \ingroup stats
 *
 * \brief A C++ interface towards an SQLITE database
 *
 * The class is able to execute commands, and retrieve results, from an SQLITE
 * database. The methods with the "Spin" prefix, in case of concurrent access
 * to the database, will spin until the operation is applied. The methods with
 * the "Wait" prefix will wait on a mutex.
 *
 * If you run multiple simulations that write on the same database, it is
 * recommended to use the "Wait" prefixed methods. Otherwise, if the access to
 * the database is unique, using "Spin" methods will speed up database access.
 *
 * The database is opened in the constructor, and closed in the deconstructor.
 */
class SQLiteOutput : public SimpleRefCount <SQLiteOutput>
{
public:
  /**
   * \brief SQLiteOutput constructor
   * \param name database name
   */
  SQLiteOutput (const std::string &name);

  /**
   * Destructor
   */
  ~SQLiteOutput ();

  /**
   * \brief Instruct SQLite to store the journal in memory. May lead to data losses
   * in case of unexpected program exits.
   */
  void SetJournalInMemory ();

  /**
   * \brief Execute a command until the return value is OK or an ERROR
   *
   * Ignore errors due to concurrency, the method will repeat the command instead
   *
   * \param cmd Command
   * \return true in case of success
   */
  bool SpinExec (const std::string &cmd) const;

  /**
   * \brief Execute a command until the return value is OK or an ERROR
   * \param stmt SQLite statement
   * \return true in case of success
   */
  bool SpinExec (sqlite3_stmt *stmt) const;

  /**
   * \brief Execute a command, waiting on a mutex
   * \param cmd Command to be executed
   * \return true in case of success
   */
  bool WaitExec (const std::string &cmd) const;

  /**
   * \brief Execute a command, waiting on a mutex
   * \param stmt Sqlite3 statement to be executed
   * \return true in case of success
   */
  bool WaitExec (sqlite3_stmt *stmt) const;

  /**
   * \brief Prepare a statement, waiting on a mutex
   * \param stmt Sqlite statement
   * \param cmd Command to prepare inside the statement
   * \return true in case of success
   */
  bool WaitPrepare (sqlite3_stmt **stmt, const std::string &cmd) const;

  /**
   * \brief Prepare a statement
   * \param stmt Sqlite statement
   * \param cmd Command to prepare inside the statement
   * \return true in case of success
   */
  bool SpinPrepare (sqlite3_stmt **stmt, const std::string &cmd) const;

  /**
   * \brief Bind a value to a sqlite statement
   * \param stmt Sqlite statement
   * \param pos Position of the bind argument inside the statement
   * \param value Value to bind
   * \return true on success, false otherwise
   */
  template<typename T>
  bool Bind (sqlite3_stmt *stmt, int pos, const T &value) const;

  /**
   * \brief Retrieve a value from an executed statement
   * \param stmt sqlite statement
   * \param pos Column position
   * \return the value from the executed statement
   */
  template<typename T>
  T RetrieveColumn (sqlite3_stmt *stmt, int pos) const;

  /**
   * \brief Execute a step operation on a statement until the result is ok or an error
   *
   * Ignores concurrency errors; it will retry instead of failing.
   *
   * \param stmt Statement
   * \return Sqlite error core
   */

  static int SpinStep (sqlite3_stmt *stmt);
  /**
   * \brief Finalize a statement until the result is ok or an error
   *
   * Ignores concurrency errors; it will retry instead of failing.
   *
   * \param stmt Statement
   * \return Sqlite error code
   */
  static int SpinFinalize (sqlite3_stmt *stmt);

  /**
   * \brief Reset a statement until the result is ok or an error
   *
   * Ignores concurrency errors: it will retry instead of failing.
   * \param stmt Statement
   * \return the return code of reset
   */
  static int SpinReset (sqlite3_stmt *stmt);

protected:
  /**
   * \brief Execute a command, waiting on a mutex
   * \param db Database
   * \param cmd Command
   * \return Sqlite error code
   */
  int WaitExec (sqlite3 *db, const std::string &cmd) const;

  /**
   * \brief Execute a statement, waiting on a mutex
   * \param db Database
   * \param stmt Statement
   * \return Sqlite error code
   */
  int WaitExec (sqlite3 *db, sqlite3_stmt *stmt) const;

  /**
   * \brief Prepare a statement, waiting on a mutex
   * \param db Database
   * \param stmt Statement
   * \param cmd Command to prepare
   * \return Sqlite error code
   */
  int WaitPrepare (sqlite3 *db, sqlite3_stmt **stmt, const std::string &cmd) const;

  /**
   * \brief Execute a command ignoring concurrency problems, retrying instead
   * \param db Database
   * \param cmd Command
   * \return Sqlite error code
   */
  static int SpinExec (sqlite3 *db, const std::string &cmd);

  /**
   * \brief Execute a Prepared Statement Object ignoring concurrency problems, retrying instead
   * \param db Database
   * \param stmt Prepared Statement Object
   * \return Sqlite error code
   */
  static int SpinExec (sqlite3 *db, sqlite3_stmt *stmt);

  /**
   * \brief Preparing a command ignoring concurrency problems, retrying instead
   * \param db Database
   * \param stmt Statement
   * \param cmd Command to prepare
   * \return Sqlite error code
   */
  static int SpinPrepare (sqlite3 *db, sqlite3_stmt **stmt, const std::string &cmd);

  /**
   * \brief Fail, printing an error message from sqlite
   * \param db Database
   * \param cmd Command
   */
  [[ noreturn ]] static void Error (sqlite3 *db, const std::string &cmd);

  /**
   * \brief Check any error in the db
   * \param db Database
   * \param rc Sqlite return code
   * \param cmd Command
   * \param hardExit if true, will exit the program
   * \return true in case of error, false otherwise
   */
  static bool CheckError (sqlite3 *db, int rc, const std::string &cmd,
                          bool hardExit);

private:
  std::string m_dBname;        //!< Database name
  mutable std::mutex m_mutex;  //!< Mutex
  sqlite3 *m_db {
    nullptr
  };                           //!< Database pointer
};

} // namespace ns3
#endif
