/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 University of Washington
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef PREAMBLE_DETECTION_MODEL_H
#define PREAMBLE_DETECTION_MODEL_H

#include "ns3/object.h"

namespace ns3 {

/**
 * \ingroup wifi
 * \brief the interface for Wifi's preamble detection models
 *
 */
class PreambleDetectionModel : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * A pure virtual method that must be implemented in the subclass.
   * This method returns whether the preamble detection was successful.
   *
   * \param rssi the RSSI of the received signal (in Watts).
   * \param snr the SNR of the received signal in linear scale.
   * \param channelWidth the channel width of the received signal in MHz.
   *
   * \return true if the preamble has been detected,
   *         false otherwise
   */
  virtual bool IsPreambleDetected (double rssi, double snr, double channelWidth) const = 0;
};

} //namespace ns3

#endif /* PREAMBLE_DETECTION_MODEL_H */

