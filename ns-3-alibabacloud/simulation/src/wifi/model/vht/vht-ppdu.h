/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Orange Labs
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
 *         Sébastien Deronne <sebastien.deronne@gmail.com> (VhtSigHeader)
 */

#ifndef VHT_PPDU_H
#define VHT_PPDU_H

#include "ns3/ofdm-ppdu.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::VhtPpdu class.
 */

namespace ns3 {

class WifiPsdu;

/**
 * \brief VHT PPDU (11ac)
 * \ingroup wifi
 *
 * VhtPpdu stores a preamble, PHY headers and a PSDU of a PPDU with VHT header
 */
class VhtPpdu : public OfdmPpdu
{
public:

  /**
   * VHT PHY header (VHT-SIG-A1/A2/B).
   * See section 21.3.8 in IEEE 802.11-2016.
   */
  class VhtSigHeader : public Header
  {
  public:
    VhtSigHeader ();
    virtual ~VhtSigHeader ();

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
     * Set the Multi-User (MU) flag.
     *
     * \param mu the MU flag
     */
    void SetMuFlag (bool mu);

    /**
     * Fill the channel width field of VHT-SIG-A1 (in MHz).
     *
     * \param channelWidth the channel width (in MHz)
     */
    void SetChannelWidth (uint16_t channelWidth);
    /**
     * Return the channel width (in MHz).
     *
     * \return the channel width (in MHz)
     */
    uint16_t GetChannelWidth (void) const;
    /**
     * Fill the number of streams field of VHT-SIG-A1.
     *
     * \param nStreams the number of streams
     */
    void SetNStreams (uint8_t nStreams);
    /**
     * Return the number of streams.
     *
     * \return the number of streams
     */
    uint8_t GetNStreams (void) const;

    /**
     * Fill the short guard interval field of VHT-SIG-A2.
     *
     * \param sgi whether short guard interval is used or not
     */
    void SetShortGuardInterval (bool sgi);
    /**
     * Return the short GI field of VHT-SIG-A2.
     *
     * \return the short GI field of VHT-SIG-A2
     */
    bool GetShortGuardInterval (void) const;
    /**
     * Fill the short GI NSYM disambiguation field of VHT-SIG-A2.
     *
     * \param disambiguation whether short GI NSYM disambiguation is set or not
     */
    void SetShortGuardIntervalDisambiguation (bool disambiguation);
    /**
     * Return the short GI NSYM disambiguation field of VHT-SIG-A2.
     *
     * \return the short GI NSYM disambiguation field of VHT-SIG-A2
     */
    bool GetShortGuardIntervalDisambiguation (void) const;
    /**
     * Fill the SU VHT MCS field of VHT-SIG-A2.
     *
     * \param mcs the SU VHT MCS field of VHT-SIG-A2
     */
    void SetSuMcs (uint8_t mcs);
    /**
     * Return the SU VHT MCS field of VHT-SIG-A2.
     *
     * \return the SU VHT MCS field of VHT-SIG-A2
     */
    uint8_t GetSuMcs (void) const;

  private:
    //VHT-SIG-A1 fields
    uint8_t m_bw;   ///< BW
    uint8_t m_nsts; ///< NSTS

    //VHT-SIG-A2 fields
    uint8_t m_sgi;                ///< Short GI
    uint8_t m_sgi_disambiguation; ///< Short GI NSYM Disambiguation
    uint8_t m_suMcs;              ///< SU VHT MCS

    /// This is used to decide whether MU SIG-B should be added or not
    bool m_mu;
  }; //class VhtSigHeader

  /**
   * Create a VHT PPDU.
   *
   * \param psdu the PHY payload (PSDU)
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param ppduDuration the transmission duration of this PPDU
   * \param band the WifiPhyBand used for the transmission of this PPDU
   * \param uid the unique ID of this PPDU
   */
  VhtPpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, Time ppduDuration,
           WifiPhyBand band, uint64_t uid);
  /**
   * Destructor for VhtPpdu.
   */
  virtual ~VhtPpdu ();

  Time GetTxDuration (void) const override;
  Ptr<WifiPpdu> Copy (void) const override;
  WifiPpduType GetType (void) const override;

private:
  WifiTxVector DoGetTxVector (void) const override;

  VhtSigHeader m_vhtSig;  //!< the VHT-SIG PHY header
}; //class VhtPpdu

} //namespace ns3

#endif /* VHT_PPDU_H */
