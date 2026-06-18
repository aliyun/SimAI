/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 University of Washington
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
 * Author: Mitch Watrous (watrous@u.washington.edu)
 */

#ifndef TIME_SERIES_ADAPTOR_H
#define TIME_SERIES_ADAPTOR_H

#include "ns3/data-collection-object.h"
#include "ns3/object.h"
#include "ns3/type-id.h"
#include "ns3/traced-value.h"

namespace ns3 {

/**
 * \ingroup aggregator
 *
 * \brief Takes probed values of different types and outputs the
 * current time plus the value with both converted to doubles.
 *
 * The role of the TimeSeriesAdaptor class is that of an adaptor
 * class, to take raw-valued probe data of different types, and output
 * a tuple of two double values.  The first is a timestamp which may
 * be set to different resolutions (e.g. Seconds, Milliseconds, etc.)
 * in the future, but which presently is hardcoded to Seconds.  The second
 * is the conversion of
 * a non-double value to a double value (possibly with loss of precision).
 *
 * It should be noted that time series adaptors convert
 * Simulation Time objects to double values in its output.
 */
class TimeSeriesAdaptor : public DataCollectionObject
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TimeSeriesAdaptor ();
  virtual ~TimeSeriesAdaptor ();

  /**
   * \brief Trace sink for receiving data from double valued trace
   * sources.
   * \param oldData the original value.
   * \param newData the new value.
   *
   * This method serves as a trace sink to double valued trace
   * sources.
   */
  void TraceSinkDouble (double oldData, double newData);

  /**
   * \brief Trace sink for receiving data from bool valued trace
   * sources.
   * \param oldData the original value.
   * \param newData the new value.
   *
   * This method serves as a trace sink to bool valued trace
   * sources.
   */
  void TraceSinkBoolean (bool oldData, bool newData);

  /**
   * \brief Trace sink for receiving data from uint8_t valued trace
   * sources.
   * \param oldData the original value.
   * \param newData the new value.
   *
   * This method serves as a trace sink to uint8_t valued trace
   * sources.
   */
  void TraceSinkUinteger8 (uint8_t oldData, uint8_t newData);

  /**
   * \brief Trace sink for receiving data from uint16_t valued trace
   * sources.
   * \param oldData the original value.
   * \param newData the new value.
   *
   * This method serves as a trace sink to uint16_t valued trace
   * sources.
   */
  void TraceSinkUinteger16 (uint16_t oldData, uint16_t newData);

  /**
   * \brief Trace sink for receiving data from uint32_t valued trace
   * sources.
   * \param oldData the original value.
   * \param newData the new value.
   *
   * This method serves as a trace sink to uint32_t valued trace
   * sources.
   */
  void TraceSinkUinteger32 (uint32_t oldData, uint32_t newData);

  /**
   * TracedCallback signature for output trace.
   *
   * \param [in] now The current time, in seconds.
   * \param [in] data The new data value.
   */
  typedef void (* OutputTracedCallback) (const double now, const double data);
  
private:
  TracedCallback<double, double> m_output; //!< output trace
};

} // namespace ns3

#endif // TIME_SERIES_ADAPTOR_H
