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
#include "sqlite-output.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "ns3/abort.h"
#include "ns3/log.h"
#include "ns3/nstime.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SQLiteOutput");

SQLiteOutput::SQLiteOutput (const std::string &name)
{
  int rc = sqlite3_open (name.c_str (), &m_db);
  NS_ABORT_MSG_UNLESS (rc == SQLITE_OK, "Failed to open DB");
}

SQLiteOutput::~SQLiteOutput ()
{
  int rc = SQLITE_FAIL;

  rc = sqlite3_close_v2 (m_db);
  NS_ABORT_MSG_UNLESS (rc == SQLITE_OK, "Failed to close DB");
}

void
SQLiteOutput::SetJournalInMemory ()
{
  NS_LOG_FUNCTION (this);
  SpinExec ("PRAGMA journal_mode = MEMORY");
}

bool
SQLiteOutput::SpinExec (const std::string &cmd) const
{
  return (SpinExec (m_db, cmd) == SQLITE_OK);
}

bool
SQLiteOutput::SpinExec (sqlite3_stmt *stmt) const
{
  int rc = SpinExec (m_db, stmt);
  return !CheckError (m_db, rc, "", false);
}

bool
SQLiteOutput::WaitExec (const std::string &cmd) const
{
  int rc = WaitExec (m_db, cmd);
  return !CheckError (m_db, rc, cmd, false);
}

bool
SQLiteOutput::WaitExec (sqlite3_stmt *stmt) const
{
  return (WaitExec (m_db, stmt) == SQLITE_OK);
}

bool
SQLiteOutput::WaitPrepare (sqlite3_stmt **stmt, const std::string &cmd) const
{
  return (WaitPrepare (m_db, stmt, cmd) == SQLITE_OK);
}

bool
SQLiteOutput::SpinPrepare (sqlite3_stmt **stmt, const std::string &cmd) const
{
  return (SpinPrepare (m_db, stmt, cmd) == SQLITE_OK);
}

template<typename T>
T
SQLiteOutput::RetrieveColumn ([[maybe_unused]] sqlite3_stmt *stmt, [[maybe_unused]] int pos) const
{
  NS_FATAL_ERROR ("Can't call generic fn");
}

/// \copydoc SQLiteOutput::RetrieveColumn
template<>
int
SQLiteOutput::RetrieveColumn (sqlite3_stmt *stmt, int pos) const
{
  return sqlite3_column_int (stmt, pos);
}

/// \copydoc SQLiteOutput::RetrieveColumn
template<>
uint32_t
SQLiteOutput::RetrieveColumn (sqlite3_stmt *stmt, int pos) const
{
  return static_cast<uint32_t> (sqlite3_column_int (stmt, pos));
}

/// \copydoc SQLiteOutput::RetrieveColumn
template<>
double
SQLiteOutput::RetrieveColumn (sqlite3_stmt *stmt, int pos) const
{
  return sqlite3_column_double (stmt, pos);
}

template<typename T>
bool
SQLiteOutput::Bind ([[maybe_unused]] sqlite3_stmt *stmt, [[maybe_unused]] int pos, [[maybe_unused]] const T &value) const
{
  NS_FATAL_ERROR ("Can't call generic fn");
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const Time &value) const
{
  if (sqlite3_bind_double (stmt, pos, value.GetSeconds ()) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const double &value) const
{
  if (sqlite3_bind_double (stmt, pos, value) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const uint32_t &value) const
{
  if (sqlite3_bind_int (stmt, pos, static_cast<int> (value)) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const long &value) const
{
  if (sqlite3_bind_int64 (stmt, pos, value) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const long long &value) const
{
  if (sqlite3_bind_int64 (stmt, pos, value) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const uint16_t &value) const
{
  if (sqlite3_bind_int (stmt, pos, static_cast<int> (value)) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const uint8_t &value) const
{
  if (sqlite3_bind_int (stmt, pos, static_cast<int> (value)) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const int &value) const
{
  if (sqlite3_bind_int (stmt, pos, value) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

//! \copydoc SQLiteOutput::Bind
template<>
bool
SQLiteOutput::Bind (sqlite3_stmt *stmt, int pos, const std::string &value) const
{
  if (sqlite3_bind_text (stmt, pos, value.c_str (), -1, SQLITE_STATIC) == SQLITE_OK)
    {
      return true;
    }
  return false;
}

int
SQLiteOutput::WaitPrepare (sqlite3 *db, sqlite3_stmt **stmt, const std::string &cmd) const
{
  int rc;
  bool ret;

  std::unique_lock lock {m_mutex};

  rc = sqlite3_prepare_v2 (db, cmd.c_str (),
                           static_cast<int> (cmd.size ()),
                           stmt, nullptr);

  ret = CheckError (db, rc, cmd, false);
  if (ret)
    {
      return rc;
    }

  return rc;
}

int
SQLiteOutput::SpinPrepare (sqlite3 *db, sqlite3_stmt **stmt, const std::string &cmd)
{
  int rc;

  do
    {
      rc = sqlite3_prepare_v2 (db, cmd.c_str (),
                               static_cast<int> (cmd.size ()),
                               stmt, nullptr);
    }
  while (rc == SQLITE_BUSY || rc == SQLITE_LOCKED);
  return rc;
}

int
SQLiteOutput::SpinStep (sqlite3_stmt *stmt)
{
  int rc;
  do
    {
      rc = sqlite3_step (stmt);
    }
  while (rc == SQLITE_BUSY || rc == SQLITE_LOCKED);

  return rc;
}

int
SQLiteOutput::SpinFinalize (sqlite3_stmt *stmt)
{
  int rc;
  do
    {
      rc = sqlite3_finalize (stmt);
    }
  while (rc == SQLITE_BUSY || rc == SQLITE_LOCKED);

  return rc;
}

int SQLiteOutput::SpinReset (sqlite3_stmt *stmt)
{
  int rc;

  do
    {
      rc = sqlite3_reset (stmt);
    }
  while (rc == SQLITE_BUSY || rc == SQLITE_LOCKED);

  return rc;
}

void
SQLiteOutput::Error (sqlite3 *db, const std::string &cmd)
{
  NS_ABORT_MSG (cmd << " error " << sqlite3_errmsg (db));
}

bool
SQLiteOutput::CheckError (sqlite3 *db, int rc, const std::string &cmd,
                          bool hardExit)
{
  if (rc != SQLITE_OK && rc != SQLITE_DONE)
    {
      if (hardExit)
        {
          Error (db, cmd);
        }
      else
        {
          std::cerr << sqlite3_errmsg (db) << std::endl;
        }
      return true;
    }
  return false;
}

int
SQLiteOutput::SpinExec (sqlite3 *db, const std::string &cmd)
{
  sqlite3_stmt *stmt;
  bool ret;

  int rc = SpinPrepare (db, &stmt, cmd);
  ret = CheckError (db, rc, cmd, false);
  if (ret)
    {
      return rc;
    }

  rc = SpinStep (stmt);
  ret = CheckError (db, rc, cmd, false);
  if (ret)
    {
      return rc;
    }

  rc = SpinFinalize (stmt);
  CheckError (db, rc, cmd, false);

  return rc;
}

int
SQLiteOutput::SpinExec (sqlite3 *db, sqlite3_stmt *stmt)
{
  bool ret;
  int rc = SpinStep (stmt);
  ret = CheckError (db, rc, "", false);
  if (ret)
    {
      return rc;
    }

  rc = SpinFinalize (stmt);
  return rc;
}

int
SQLiteOutput::WaitExec (sqlite3 *db, sqlite3_stmt *stmt) const
{
  bool ret;
  int rc = SQLITE_ERROR;

  std::unique_lock lock {m_mutex};

  rc = SpinStep (stmt);

  ret = CheckError (db, rc, "", false);
  if (ret)
    {
      return rc;
    }

  rc = SpinFinalize (stmt);

  return rc;
}

int
SQLiteOutput::WaitExec (sqlite3 *db, const std::string &cmd) const
{
  sqlite3_stmt *stmt;
  bool ret;
  int rc = SQLITE_ERROR;

  std::unique_lock lock {m_mutex};

  rc = SpinPrepare (db, &stmt, cmd);
  ret = CheckError (db, rc, cmd, false);
  if (ret)
    {
      return rc;
    }

  rc = SpinStep (stmt);

  ret = CheckError (db, rc, cmd, false);
  if (ret)
    {
      return rc;
    }

  rc = SpinFinalize (stmt);

  return rc;
}

} // namespace ns3;
