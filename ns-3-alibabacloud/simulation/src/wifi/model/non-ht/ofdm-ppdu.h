/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Orange Labs
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
 * Author: Rediet <getachew.redieteab@orange.com>
 *         Muhammad Iqbal Rochman <muhiqbalcr@uchicago.edu>
 *         Sébastien Deronne <sebastien.deronne@gmail.com> (LSigHeader)
 */

#ifndef OFDM_PPDU_H
#define OFDM_PPDU_H

#include "ns3/wifi-phy-band.h"
#include "ns3/wifi-ppdu.h"
#include "ns3/header.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::OfdmPpdu class.
 */

namespace ns3 {

class WifiPsdu;

/**
 * \brief OFDM PPDU (11a)
 * \ingroup wifi
 *
 * OfdmPpdu stores a preamble, PHY headers and a PSDU of a PPDU with non-HT header,
 * i.e., PPDU that uses OFDM modulation.
 */
class OfdmPpdu : public WifiPpdu
{
public:

  /**
   * OFDM and ERP OFDM L-SIG PHY header.
   * See section 17.3.4 in IEEE 802.11-2016.
   */
  class LSigHeader : public Header
  {
  public:
    LSigHeader ();
    virtual ~LSigHeader ();

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId (void);

    TypeId GetInstanceTypeId (void) const override;
    void Print (std::ostream &os) const override;
    uint32_t GetSerializedSize (void) const override;
    void Serialize (Buffer::Iterator start) const override;
    uint32_t Deserialize (Buffer::Iterator start) override;

    /**
     * Fill the RATE field of L-SIG (in bit/s).
     *
     * \param rate the RATE field of L-SIG expressed in bit/s
     * \param channelWidth the channel width (in MHz)
     */
    void SetRate (uint64_t rate, uint16_t channelWidth = 20);
    /**
     * Return the RATE field of L-SIG (in bit/s).
     *
     * \param channelWidth the channel width (in MHz)
     * \return the RATE field of L-SIG expressed in bit/s
     */
    uint64_t GetRate (uint16_t channelWidth = 20) const;
    /**
     * Fill the LENGTH field of L-SIG (in bytes).
     *
     * \param length the LENGTH field of L-SIG expressed in bytes
     */
    void SetLength (uint16_t length);
    /**
     * Return the LENGTH field of L-SIG (in bytes).
     *
     * \return the LENGTH field of L-SIG expressed in bytes
     */
    uint16_t GetLength (void) const;

  private:
    uint8_t m_rate;    ///< RATE field
    uint16_t m_length; ///< LENGTH field
  }; //class LSigHeader

  /**
   * Create an OFDM PPDU.
   *
   * \param psdu the PHY payload (PSDU)
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param band the WifiPhyBand used for the transmission of this PPDU
   * \param uid the unique ID of this PPDU
   * \param instantiateLSig flag used to instantiate LSigHeader (set LSigHeader's
   *                        rate and length), should be disabled by child classes
   */
  OfdmPpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, WifiPhyBand band, uint64_t uid,
            bool instantiateLSig = true);
  /**
   * Destructor for OfdmPpdu.
   */
  virtual ~OfdmPpdu ();

  Time GetTxDuration (void) const override;
  Ptr<WifiPpdu> Copy (void) const override;

protected:
  WifiPhyBand m_band;       //!< the WifiPhyBand used to transmit that PPDU
  uint16_t m_channelWidth;  //!< the channel width used to transmit that PPDU in MHz
  LSigHeader m_lSig;        //!< the L-SIG PHY header

private:
  WifiTxVector DoGetTxVector (void) const override;
}; //class OfdmPpdu

} //namespace ns3

#endif /* OFDM_PPDU_H */
