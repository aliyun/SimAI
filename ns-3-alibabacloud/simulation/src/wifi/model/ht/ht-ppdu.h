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
 *         Sébastien Deronne <sebastien.deronne@gmail.com> (HtSigHeader)
 */

#ifndef HT_PPDU_H
#define HT_PPDU_H

#include "ns3/ofdm-ppdu.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::HtPpdu class.
 */

namespace ns3 {

class WifiPsdu;

/**
 * \brief HT  PPDU (11n)
 * \ingroup wifi
 *
 * HtPpdu stores a preamble, PHY headers and a PSDU of a PPDU with HT header
 */
class HtPpdu : public OfdmPpdu
{
public:

  /**
   * HT PHY header (HT-SIG1/2).
   * See section 19.3.9 in IEEE 802.11-2016.
   */
  class HtSigHeader : public Header
  {
  public:
    HtSigHeader ();
    virtual ~HtSigHeader ();

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
     * Fill the MCS field of HT-SIG.
     *
     * \param mcs the MCS field of HT-SIG
     */
    void SetMcs (uint8_t mcs);
    /**
     * Return the MCS field of HT-SIG.
     *
     * \return the MCS field of HT-SIG
     */
    uint8_t GetMcs (void) const;
    /**
     * Fill the channel width field of HT-SIG (in MHz).
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
     * Fill the aggregation field of HT-SIG.
     *
     * \param aggregation whether the PSDU contains A-MPDU or not
     */
    void SetAggregation (bool aggregation);
    /**
     * Return the aggregation field of HT-SIG.
     *
     * \return the aggregation field of HT-SIG
     */
    bool GetAggregation (void) const;
    /**
     * Fill the short guard interval field of HT-SIG.
     *
     * \param sgi whether short guard interval is used or not
     */
    void SetShortGuardInterval (bool sgi);
    /**
     * Return the short guard interval field of HT-SIG.
     *
     * \return the short guard interval field of HT-SIG
     */
    bool GetShortGuardInterval (void) const;
    /**
     * Fill the HT length field of HT-SIG (in bytes).
     *
     * \param length the HT length field of HT-SIG (in bytes)
     */
    void SetHtLength (uint16_t length);
    /**
     * Return the HT length field of HT-SIG (in bytes).
     *
     * \return the HT length field of HT-SIG (in bytes)
     */
    uint16_t GetHtLength (void) const;

  private:
    uint8_t m_mcs;         ///< Modulation and Coding Scheme index
    uint8_t m_cbw20_40;    ///< CBW 20/40
    uint16_t m_htLength;   ///< HT length
    uint8_t m_aggregation; ///< Aggregation
    uint8_t m_sgi;         ///< Short Guard Interval
  }; //HtSigHeader

  /**
   * Create an HT PPDU.
   *
   * \param psdu the PHY payload (PSDU)
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param ppduDuration the transmission duration of this PPDU
   * \param band the WifiPhyBand used for the transmission of this PPDU
   * \param uid the unique ID of this PPDU
   */
  HtPpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, Time ppduDuration,
          WifiPhyBand band, uint64_t uid);
  /**
   * Destructor for HtPpdu.
   */
  virtual ~HtPpdu ();

  Time GetTxDuration (void) const override;
  Ptr<WifiPpdu> Copy (void) const override;

private:
  WifiTxVector DoGetTxVector (void) const override;

  HtSigHeader m_htSig;  //!< the HT-SIG PHY header
}; //class HtPpdu

} //namespace ns3

#endif /* HT_PPDU_H */
