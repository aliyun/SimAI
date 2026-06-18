/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef SUPPORTED_RATES_H
#define SUPPORTED_RATES_H

#include "wifi-information-element.h"

namespace ns3 {

class SupportedRates;

/**
 * \brief The Extended Supported Rates Information Element
 * \ingroup wifi
 *
 * This class knows how to serialise and deserialise the Extended
 * Supported Rates Element that holds (non-HT) rates beyond the 8 that
 * the original Supported Rates element can carry.
 *
 * The \c SupportedRates class still records all the rates, and an
 * instance of \c ExtendedSupportedRatesIE lies within \c
 * SupportedRates.
 */
class ExtendedSupportedRatesIE : public WifiInformationElement
{
public:
  ExtendedSupportedRatesIE ();
  /**
   * Create an extended supported rates information element
   * from the given rates.
   *
   * \param rates the extended supported rates
   */
  ExtendedSupportedRatesIE (SupportedRates *rates);

  // Implementations of pure virtual methods of WifiInformationElement
  WifiInformationElementId ElementId () const override;
  uint8_t GetInformationFieldSize () const override;
  void SerializeInformationField (Buffer::Iterator start) const override;
  uint8_t DeserializeInformationField (Buffer::Iterator start,
                                       uint8_t length) override;
  /* This information element is a bit special in that it is only
    included if there are more than 8 rates. To support this we
    override the Serialize and GetSerializedSize methods of
    WifiInformationElement. */
  Buffer::Iterator Serialize (Buffer::Iterator start) const override;
  uint16_t GetSerializedSize () const override;

  /**
   * Set supported rates.
   *
   * \param rates the supported rates
   */
  void SetSupportedRates (SupportedRates *rates);


private:
  /**
   * This member points to the SupportedRates object that contains the
   * actual rate details. This class is a friend of that, so we have
   * access to all the private data we need.
   */
  SupportedRates *m_supportedRates;
};


/**
 * \brief The Supported Rates Information Element
 * \ingroup wifi
 *
 * This class knows how to serialise and deserialise the Supported
 * Rates Element that holds the first 8 (non-HT) supported rates.
 *
 * The \c ExtendedSupportedRatesIE class (of which an instance exists
 * in objects of this class) deals with rates beyond the first 8.
 */
class SupportedRates : public WifiInformationElement
{
public:
  SupportedRates ();

  /**
   * Type conversion operator
   *
   * \param rates the reference const SupportedRates
   */
  SupportedRates (const SupportedRates & rates);

  // Implementations of pure virtual methods of WifiInformationElement
  WifiInformationElementId ElementId () const;
  uint8_t GetInformationFieldSize () const;
  void SerializeInformationField (Buffer::Iterator start) const;
  uint8_t DeserializeInformationField (Buffer::Iterator start,
                                       uint8_t length);

  /**
   * Assignment operator
   *
   * \param rates the rates to assign
   * \returns the assigned value
   */
  SupportedRates& operator= (const SupportedRates& rates);

  /**
   * This defines the maximum number of supported rates that a STA is
   * allowed to have. Currently this number is set for IEEE 802.11b/g and SISO IEEE 802.11n
   * stations which need 2 rates each from Clauses 15 and 18, and then 8
   * from Clause 19.
   */
  static const uint8_t MAX_SUPPORTED_RATES = 32;

  /**
   * Add the given rate to the supported rates.
   *
   * \param bs the rate to be added in bps
   */
  void AddSupportedRate (uint64_t bs);
  /**
   * Set the given rate to basic rates.
   *
   * \param bs the rate to be set in bps
   */
  void SetBasicRate (uint64_t bs);
  /**
   * Add a special value to the supported rate set, corresponding to
   * a BSS membership selector
   *
   * \param bs the special membership selector value (not a valid rate)
   */
  void AddBssMembershipSelectorRate (uint64_t bs);
  /**
   * Check if the given rate is supported.  The rate is encoded as it is
   * serialized to the Supported Rates Information Element (i.e. as a
   * multiple of 500 Kbits/sec, possibly with MSB set to 1).
   *
   * \param bs the rate to be checked in bps
   *
   * \return true if the rate is supported, false otherwise
   */
  bool IsSupportedRate (uint64_t bs) const;
  /**
   * Check if the given rate is a basic rate.  The rate is encoded as it is
   * serialized to the Supported Rates Information Element (i.e. as a
   * multiple of 500 Kbits/sec, with MSB set to 1).
   *
   * \param bs the rate to be checked in bps
   *
   * \return true if the rate is a basic rate, false otherwise
   */
  bool IsBasicRate (uint64_t bs) const;
  /**
   * Check if the given rate is a BSS membership selector value.  The rate
   * is encoded as it is serialized to the Supporting Rates Information
   * Element (i.e. with the MSB set to 1).
   *
   * \param bs the rate to be checked in bps
   *
   * \return true if the rate is a BSS membership selector, false otherwise
   */
  bool IsBssMembershipSelectorRate (uint64_t bs) const;
  /**
   * Return the number of supported rates.
   *
   * \return the number of supported rates
   */
  uint8_t GetNRates (void) const;
  /**
   * Return the rate at the given index.
   * Return the rate (converted back to raw value) at the given index.
   *
   * \param i the given index
   * \return the rate in bps
   */
  uint32_t GetRate (uint8_t i) const;

  /**
   * We support the Extended Supported Rates Information Element
   * through the ExtendedSupportedRatesIE object which is declared
   * above. We allow this class to be a friend so that it can access
   * our private data detailing the rates, and create an instance as
   * extended.
   */
  friend class ExtendedSupportedRatesIE;
  ExtendedSupportedRatesIE extended; //!< extended supported rates info element


private:
  uint8_t m_nRates;                      //!< Number of supported rates
  uint8_t m_rates[MAX_SUPPORTED_RATES];  //!< List of supported bit rates (divided by 500000)
};

/**
 * Serialize SupportedRates to the given ostream.
 *
 * \param os output stream
 * \param rates the SupportedRates
 *
 * \return std::ostream
 */
std::ostream &operator << (std::ostream &os, const SupportedRates &rates);

} //namespace ns3

#endif /* SUPPORTED_RATES_H */
