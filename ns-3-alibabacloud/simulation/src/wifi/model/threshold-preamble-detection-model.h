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

#ifndef THRESHOLD_PREAMBLE_DETECTION_MODEL_H
#define THRESHOLD_PREAMBLE_DETECTION_MODEL_H

#include "preamble-detection-model.h"

namespace ns3 {
/**
 * \ingroup wifi
 *
 * A threshold-based model for detecting PHY preamble.
 * This model assumes that a preamble is successfully detected if SNR is at or above a given threshold (set to 4 dB by default).
 * However, if RSSI is below a minimum RSSI (set to -82 dBm by default), the PHY preamble is not detected.
 */
class ThresholdPreambleDetectionModel : public PreambleDetectionModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  ThresholdPreambleDetectionModel ();
  ~ThresholdPreambleDetectionModel ();

  /**
   * This method returns whether the preamble detection was successful.
   *
   * \param rssi the RSSI of the received signal (in Watts).
   * \param snr the SNR ratio (linear scale) of the received signal.
   * \param channelWidth the channel width of the received signal in MHz.
   *
   * \return true if the preamble has been detected,
   *         false otherwise
   */
  bool IsPreambleDetected (double rssi, double snr, double channelWidth) const override;


private:
  double m_threshold; ///< SNR threshold in dB used to decide whether a preamble is successfully received
  double m_rssiMin;   ///< Minimum RSSI in dBm that shall be received to start the decision
};

} //namespace ns3

#endif /* THRESHOLD_PREAMBLE_DETECTION_MODEL_H */
