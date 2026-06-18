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

#ifndef DATA_CALCULATOR_H
#define DATA_CALCULATOR_H

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"

namespace ns3 {
extern const double NaN; //!< Stored representation of NaN

/**
 * \brief true if x is NaN
 * \param x
 * \return whether x is NaN
 */
inline bool isNaN (double x) { return x != x; }

class DataOutputCallback;

/**
 * \ingroup stats
 * \class StatisticalSummary
 * \brief Abstract class for calculating statistical data 
 *
 */
class StatisticalSummary {
public:
  /**
   * Destructor
   */
  virtual ~StatisticalSummary ()
  {
  }
  /**
   * Returns the number of observations.
   * \return Number of observations
   */
  virtual long getCount () const = 0;

  /**
   * \return Sum of values
   * @see getWeightedSum()
   */
  virtual double getSum () const = 0;

  /**
   * \return Sum of squared values
   * @see getWeightedSqrSum()
   */
  virtual double getSqrSum () const = 0;

  /**
   * Returns the minimum of the values.
   * \return Minimum of values
   */
  virtual double getMin () const = 0;

  /**
   * Returns the maximum of the values.
   * \return Maximum of values
   */
  virtual double getMax () const = 0;

  /**
   * Returns the mean of the (weighted) observations.
   * \return Mean of (weighted) observations
   */
  virtual double getMean () const = 0;

  /**
   * Returns the standard deviation of the (weighted) observations.
   * \return Standard deviation of (weighted) observations
   */
  virtual double getStddev () const = 0;

  /**
   * Returns the variance of the (weighted) observations.
   * \return Variance of (weighted) observations
   */
  virtual double getVariance () const = 0;
};

//------------------------------------------------------------
//--------------------------------------------
/**
 * \ingroup stats
 * \class DataCalculator
 * \brief Calculates data during a simulation
 *
 */
class DataCalculator : public Object {
public:
  DataCalculator();
  virtual ~DataCalculator();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * Returns whether the DataCalculator is enabled
   * \return true if DataCalculator is enabled
   */
  bool GetEnabled () const;
  /**
   * Enables DataCalculator when simulation starts
   */
  void Enable ();
  /**
   * Disables DataCalculator when simulation stops
   */
  void Disable ();
  /**
   * Sets the DataCalculator key to the provided key
   * \param key Key value as a string
   */
  void SetKey (const std::string key);
  /**
   * Gets the DataCalculator key
   * \return Key value as a string
   */
  std::string GetKey () const;

  /**
   * Sets the DataCalculator context to the provided context
   * \param context Context value as a string
   */
  void SetContext (const std::string context);
  /**
   * Gets the DataCalculator context
   * \return Context value as a string
   */
  std::string GetContext () const;

  /**
   * Starts DataCalculator at a given time in the simulation
   * \param startTime
   */
  virtual void Start (const Time& startTime);
  /**
   * Stops DataCalculator at a given time in the simulation
   * \param stopTime
   */
  virtual void Stop (const Time& stopTime);

  /**
   * Outputs data based on the provided callback
   * \param callback
   */
  virtual void Output (DataOutputCallback &callback) const = 0;

protected:
  bool m_enabled;    //!< Descendant classes *must* check & respect m_enabled!

  std::string m_key;      //!< Key value
  std::string m_context;  //!< Context value

  virtual void DoDispose (void);

private:
  EventId m_startEvent; //!< Start event
  EventId m_stopEvent;  //!< Stop event

  // end class DataCalculator
};


// end namespace ns3
};


#endif /* DATA_CALCULATOR_H */
