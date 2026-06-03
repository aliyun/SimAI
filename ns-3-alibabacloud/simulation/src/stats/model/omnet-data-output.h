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

#ifndef OMNET_DATA_OUTPUT_H
#define OMNET_DATA_OUTPUT_H

#include "ns3/nstime.h"

#include "data-output-interface.h"

namespace ns3 {


//------------------------------------------------------------
//--------------------------------------------
/**
 * \ingroup dataoutput
 * \class OmnetDataOutput
 * \brief Outputs data in a format compatible with OMNeT library and framework
 *
 */
class OmnetDataOutput : public DataOutputInterface {
public:
  OmnetDataOutput();
  virtual ~OmnetDataOutput();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  virtual void Output (DataCollector &dc);

protected:
  virtual void DoDispose ();

private:
  /**
   * \ingroup dataoutput
   *
   * \brief Class to generate OMNeT output
   */
  class OmnetOutputCallback : public DataOutputCallback {
public:
    /**
     * Constructor
     * \param scalar the output stream
     */
    OmnetOutputCallback(std::ostream *scalar);

    /**
     * \brief Generates data statistics
     * \param context the output context
     * \param name the output name
     * \param statSum the stats to print
     */
    void OutputStatistic (std::string context,
                          std::string name,
                          const StatisticalSummary *statSum);

    /**
     * \brief Generates a single data output
     * \param context the output context
     * \param name the output name
     * \param val the value
     */
    void OutputSingleton (std::string context,
                          std::string name,
                          int val);

    /**
     * \brief Generates a single data output
     * \param context the output context
     * \param name the output name
     * \param val the value
     */
    void OutputSingleton (std::string context,
                          std::string name,
                          uint32_t val);

    /**
     * \brief Generates a single data output
     * \param context the output context
     * \param name the output name
     * \param val the value
     */
    void OutputSingleton (std::string context,
                          std::string name,
                          double val);

    /**
     * \brief Generates a single data output
     * \param context the output context
     * \param name the output name
     * \param val the value
     */
    void OutputSingleton (std::string context,
                          std::string name,
                          std::string val);

    /**
     * \brief Generates a single data output
     * \param context the output context
     * \param name the output name
     * \param val the value
     */
    void OutputSingleton (std::string context,
                          std::string name,
                          Time val);

private:
    std::ostream *m_scalar; //!< output stream
    // end class OmnetOutputCallback
  };

  // end class OmnetDataOutput
};

// end namespace ns3
};


#endif /* OMNET_DATA_OUTPUT_H */
