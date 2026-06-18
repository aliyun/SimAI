/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Drexel University
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
 * Author: Joe Kopena (tjkopena@cs.drexel.edu)
 */

#ifndef SQLITE_DATA_OUTPUT_H
#define SQLITE_DATA_OUTPUT_H

#include "ns3/nstime.h"

#include "data-output-interface.h"


struct sqlite3_stmt;

namespace ns3 {

class SQLiteOutput;
//------------------------------------------------------------
//--------------------------------------------
/**
 * \ingroup dataoutput
 * \class SqliteDataOutput
 * \brief Outputs data in a format compatible with SQLite
 */
class SqliteDataOutput : public DataOutputInterface
{
public:
  SqliteDataOutput ();
  virtual ~SqliteDataOutput () override;

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  virtual void Output (DataCollector &dc) override;

private:
  /**
   * \ingroup dataoutput
   *
   * \brief Class to generate OMNeT output
   */
  class SqliteOutputCallback : public DataOutputCallback
  {
public:
    /**
     * Constructor
     * \param db pointer to the instance this object belongs to
     * \param run experiment descriptor
     */
    SqliteOutputCallback (const Ptr<SQLiteOutput> &db, std::string run);

    /**
     * Destructor
     */
    ~SqliteOutputCallback ();

    /**
     * \brief Generates data statistics
     * \param key the SQL key to use
     * \param variable the variable name
     * \param statSum the stats to print
     */
    void OutputStatistic (std::string key,
                          std::string variable,
                          const StatisticalSummary *statSum);

    /**
     * \brief Generates a single data output
     * \param key the SQL key to use
     * \param variable the variable name
     * \param val the value
     */
    void OutputSingleton (std::string key,
                          std::string variable,
                          int val);

    /**
     * \brief Generates a single data output
     * \param key the SQL key to use
     * \param variable the variable name
     * \param val the value
     */
    void OutputSingleton (std::string key,
                          std::string variable,
                          uint32_t val);

    /**
     * \brief Generates a single data output
     * \param key the SQL key to use
     * \param variable the variable name
     * \param val the value
     */
    void OutputSingleton (std::string key,
                          std::string variable,
                          double val);

    /**
     * \brief Generates a single data output
     * \param key the SQL key to use
     * \param variable the variable name
     * \param val the value
     */
    void OutputSingleton (std::string key,
                          std::string variable,
                          std::string val);

    /**
     * \brief Generates a single data output
     * \param key the SQL key to use
     * \param variable the variable name
     * \param val the value
     */
    void OutputSingleton (std::string key,
                          std::string variable,
                          Time val);

private:
    Ptr<SQLiteOutput> m_db; //!< Db
    std::string m_runLabel; //!< Run label

    /// Pointer to a Sqlite3 singleton statement
    sqlite3_stmt *m_insertSingletonStatement;
  };

  Ptr<SQLiteOutput> m_sqliteOut; //!< Database
};

// end namespace ns3
}


#endif /* SQLITE_DATA_OUTPUT_H */
