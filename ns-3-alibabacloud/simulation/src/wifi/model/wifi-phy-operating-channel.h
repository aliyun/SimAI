/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021
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
 * Authors: Stefano Avallone <stavallo@unina.it>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef WIFI_PHY_OPERATING_CHANNEL_H
#define WIFI_PHY_OPERATING_CHANNEL_H

#include "wifi-phy-band.h"
#include "wifi-standards.h"
#include <tuple>
#include <set>

namespace ns3 {

/**
  * A tuple (number, frequency, width, type, band) identifying a frequency channel
  */
typedef std::tuple<uint8_t, uint16_t, uint16_t, FrequencyChannelType, WifiPhyBand> FrequencyChannelInfo;


/**
 * \ingroup wifi
 *
 * Class that keeps track of all information about the current PHY operating channel.
 */
class WifiPhyOperatingChannel
{
public:
  /**
   * Create an uninitialized PHY operating channel.
   */
  WifiPhyOperatingChannel ();

  virtual ~WifiPhyOperatingChannel ();

  static const std::set<FrequencyChannelInfo> m_frequencyChannels;  //!< Available frequency channels

  /**
   * Return true if a valid channel has been set, false otherwise.
   *
   * \return true if a valid channel has been set, false otherwise
   */
  bool IsSet (void) const;
  /**
   * Set the channel according to the specified parameters if a unique
   * frequency channel matches the specified criteria, or abort the
   * simulation otherwise.
   * If the channel width is a multiple of 20 MHz, the primary 20 MHz channel
   * is set to the 20 MHz subchannel with the lowest center frequency.
   *
   * \param number the channel number (use 0 to leave it unspecified)
   * \param frequency the channel center frequency in MHz (use 0 to leave it unspecified)
   * \param width the channel width in MHz (use 0 to leave it unspecified)
   * \param standard the standard
   * \param band the PHY band
   */
  void Set (uint8_t number, uint16_t frequency, uint16_t width,
            WifiStandard standard, WifiPhyBand band);
  /**
   * Set the default channel of the given width and for the given standard and band.
   * If the channel width is a multiple of 20 MHz, the primary 20 MHz channel
   * is set to the 20 MHz subchannel with the lowest center frequency.
   *
   * \param width the channel width in MHz
   * \param standard the standard
   * \param band the PHY band
   */
  void SetDefault (uint16_t width, WifiStandard standard, WifiPhyBand band);

  /**
   * Get the default channel number of the given width and for the given standard
   * and band.
   *
   * \param width the channel width in MHz
   * \param standard the standard
   * \param band the PHY band
   * \return the default channel number
   */
  static uint8_t GetDefaultChannelNumber (uint16_t width, WifiStandard standard, WifiPhyBand band);

  /**
   * Return the channel number identifying the whole operating channel.
   *
   * \return the channel number identifying the whole operating channel
   */
  uint8_t GetNumber (void) const;
  /**
   * Return the center frequency of the operating channel (in MHz).
   *
   * \return the center frequency of the operating channel (in MHz)
   */
  uint16_t GetFrequency (void) const;
  /**
   * Return the width of the whole operating channel (in MHz).
   *
   * \return the width of the whole operating channel (in MHz)
   */
  uint16_t GetWidth (void) const;

  /**
   * If the operating channel width is a multiple of 20 MHz, return the index of the
   * primary channel of the given width within the operating channel (0 indicates
   * the 20 MHz subchannel with the lowest center frequency). Otherwise, return 0.
   *
   * \param primaryChannelWidth the width of the primary channel in MHz
   * \return the index of the requested primary channel within the operating channel
   */
  uint8_t GetPrimaryChannelIndex (uint16_t primaryChannelWidth) const;
  /**
   * Set the index of the primary 20 MHz channel (0 indicates the 20 MHz subchannel
   * with the lowest center frequency).
   *
   * \param index the index of the primary 20 MHz channel
   */
  void SetPrimary20Index (uint8_t index);
  /**
   * Get the center frequency of the primary channel of the given width.
   *
   * \param primaryChannelWidth the width of the primary channel in MHz
   * \return the center frequency of the primary channel of the given width
   */
  uint16_t GetPrimaryChannelCenterFrequency (uint16_t primaryChannelWidth) const;

  /// Typedef for a const iterator pointing to a channel in the set of available channels
  typedef std::set<FrequencyChannelInfo>::const_iterator ConstIterator;

  /**
   * Find the first channel matching the specified parameters.
   *
   * \param number the channel number (use 0 to leave it unspecified)
   * \param frequency the channel center frequency in MHz (use 0 to leave it unspecified)
   * \param width the channel width in MHz (use 0 to leave it unspecified)
   * \param standard the standard
   * \param band the PHY band
   * \param start an iterator pointing to the channel to start the search with
   * \return an iterator pointing to the found channel, if any, or to past-the-end
   *         of the set of available channels
   */
  static ConstIterator FindFirst (uint8_t number, uint16_t frequency, uint16_t width,
                                  WifiStandard standard, WifiPhyBand band,
                                  ConstIterator start = m_frequencyChannels.begin ());

  /**
   * Get channel number of the primary channel
   * \param primaryChannelWidth the width of the primary channel (MHz)
   * \param standard the standard
   *
   * \return channel number of the primary channel
   */
  uint8_t GetPrimaryChannelNumber (uint16_t primaryChannelWidth, WifiStandard standard) const;

private:
  ConstIterator m_channelIt;   //!< const iterator pointing to the configured frequency channel
  uint8_t m_primary20Index;    /**< index of the primary20 channel (0 indicates the 20 MHz
                                    subchannel with the lowest center frequency) */
};

} //namespace ns3

#endif /* WIFI_PHY_OPERATING_CHANNEL_H */
