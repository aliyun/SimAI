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

#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <list>
#include <string>

#include "ns3/object.h"

namespace ns3 {

class DataCalculator;

//------------------------------------------------------------
//--------------------------------------------
/**
 * List of Ptrs to DataCalculator objects
 */
typedef std::list<Ptr<DataCalculator> > DataCalculatorList;
/**
 * List of pairs of strings representing metadata
 */
typedef std::list<std::pair<std::string, std::string> > MetadataList;

/**
 * \ingroup dataoutput
 * \class DataCollector
 * \brief Collects data 
 */
class DataCollector : public Object {
public:
  DataCollector();
  virtual ~DataCollector();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * Provide specific parameters to the DataCollector
   * \param experiment Label for the experiment
   * \param strategy Label for the strategy
   * \param input Label for the input
   * \param runID Label for the runID
   * \param description Description
   */
  void DescribeRun (std::string experiment,
                    std::string strategy,
                    std::string input,
                    std::string runID,
                    std::string description = "");

  /**
   * Return the experiment label
   * \return Experiment label
   */
  std::string GetExperimentLabel () const { return m_experimentLabel; }
  /**
   * Return the strategy label
   * \return Strategy label
   */
  std::string GetStrategyLabel () const { return m_strategyLabel; }
  /**
   * Return the input label
   * \return Input label
   */
  std::string GetInputLabel () const { return m_inputLabel; }
  /**
   * Return the runID label
   * \return Run label
   */
  std::string GetRunLabel () const { return m_runLabel; }
  /**
   * Return the description label
   * \return Description label
   */
  std::string GetDescription () const { return m_description; }

  /**
   * Add the key and the value as a pair of strings to the metadata list
   * \param key Key value to include
   * \param value Value to include of type string
   */
  void AddMetadata (std::string key, std::string value);
  /**
   * Add the key and the value as a pair of strings to the metadata list
   * \param key Key value to include
   * \param value Value to include of type double
   */
  void AddMetadata (std::string key, double value);
  /**
   * Add the key and the value as a pair of strings to the metadata list
   * \param key Key value to include
   * \param value Value to include of type uint32_t
   */
  void AddMetadata (std::string key, uint32_t value);
  /**
   * Returns an iterator to the beginning of the metadata list
   * \return Iterator pointing to the first value of the metadata list
   */
  MetadataList::iterator MetadataBegin ();
  /**
   * Returns an iterator to the past-the-end of the metadata list
   * \return Iterator pointing to the past-the-end element of the metadata list
   */
  MetadataList::iterator MetadataEnd ();

  /**
   * Add a DataCalculator object to the DataCollector
   * \param datac DataCalculator object to be added
   */
  void AddDataCalculator (Ptr<DataCalculator> datac);
  /**
   * Returns an iterator to the beginning of the DataCalculator list
   * \return Iterator pointing to the first value of the DataCalculator list
   */
  DataCalculatorList::iterator DataCalculatorBegin ();
  /**
   * Returns an iterator to the past-the-end of the DataCalculator list
   * \return Iterator pointing to the past-the-end element of the DataCalculator list
   */
  DataCalculatorList::iterator DataCalculatorEnd ();

protected:
  virtual void DoDispose ();

private:
  std::string m_experimentLabel; //!< Experiment label
  std::string m_strategyLabel;   //!< Strategy label
  std::string m_inputLabel;      //!< Input label
  std::string m_runLabel;        //!< Run label
  std::string m_description;     //!< Description label

  MetadataList m_metadata;       //!< List of experiment metadata
  DataCalculatorList m_calcList; //!< List of data calculators

  // end class DataCollector
};

// end namespace ns3
};

#endif /* DATA_COLLECTOR_H */
