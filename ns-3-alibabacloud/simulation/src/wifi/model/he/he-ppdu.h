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
 *         Sébastien Deronne <sebastien.deronne@gmail.com> (HeSigHeader)
 */

#ifndef HE_PPDU_H
#define HE_PPDU_H

#include "ns3/ofdm-ppdu.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::HePpdu class.
 */

namespace ns3 {

class WifiPsdu;

/**
 * \brief HE PPDU (11ax)
 * \ingroup wifi
 *
 * HePpdu stores a preamble, PHY headers and a map of PSDUs of a PPDU with HE header
 */
class HePpdu : public OfdmPpdu
{
public:

  /**
   * HE-SIG PHY header (HE-SIG-A1/A2/B)
   */
  class HeSigHeader : public Header
  {
  public:
    HeSigHeader ();
    virtual ~HeSigHeader ();

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
     * Fill the MCS field of HE-SIG-A1.
     *
     * \param mcs the MCS field of HE-SIG-A1
     */
    void SetMcs (uint8_t mcs);
    /**
     * Return the MCS field of HE-SIG-A1.
     *
     * \return the MCS field of HE-SIG-A1
     */
    uint8_t GetMcs (void) const;
    /**
     * Fill the BSS Color field of HE-SIG-A1.
     *
     * \param bssColor the BSS Color value
     */
    void SetBssColor (uint8_t bssColor);
    /**
     * Return the BSS Color field in the HE-SIG-A1.
     *
     * \return the BSS Color field in the HE-SIG-A1
     */
    uint8_t GetBssColor (void) const;
    /**
     * Fill the channel width field of HE-SIG-A1 (in MHz).
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
     * Fill the GI + LTF size field of HE-SIG-A1.
     *
     * \param gi the guard interval (in nanoseconds)
     * \param ltf the sequence of HE-LTF
     */
    void SetGuardIntervalAndLtfSize (uint16_t gi, uint8_t ltf);
    /**
     * Return the guard interval (in nanoseconds).
     *
     * \return the guard interval (in nanoseconds)
     */
    uint16_t GetGuardInterval (void) const;
    /**
     * Fill the number of streams field of HE-SIG-A1.
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

  private:
    //HE-SIG-A1 fields
    uint8_t m_format;       ///< Format bit
    uint8_t m_bssColor;     ///< BSS color field
    uint8_t m_ul_dl;        ///< UL/DL bit
    uint8_t m_mcs;          ///< MCS field
    uint8_t m_spatialReuse; ///< Spatial Reuse field
    uint8_t m_bandwidth;    ///< Bandwidth field
    uint8_t m_gi_ltf_size;  ///< GI+LTF Size field
    uint8_t m_nsts;         ///< NSTS

    /// This is used to decide whether MU SIG-B should be added or not
    bool m_mu;
  }; //class HeSigHeader

  /**
   * The transmit power spectral density flag, namely used
   * to correctly build PSD for HE TB PPDU non-OFDMA and
   * OFDMA portions.
   */
  enum TxPsdFlag
  {
    PSD_NON_HE_TB = 0,           //!< non-HE TB PPDU transmissions
    PSD_HE_TB_NON_OFDMA_PORTION, //!< preamble of HE TB PPDU, which should only be sent on minimum subset of 20 MHz channels containing RU
    PSD_HE_TB_OFDMA_PORTION      //!< OFDMA portion of HE TB PPDU, which should only be sent on RU
  };

  /**
   * Create an SU HE PPDU, storing a PSDU.
   *
   * \param psdu the PHY payload (PSDU)
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param ppduDuration the transmission duration of this PPDU
   * \param band the WifiPhyBand used for the transmission of this PPDU
   * \param uid the unique ID of this PPDU
   */
  HePpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, Time ppduDuration,
          WifiPhyBand band, uint64_t uid);
  /**
   * Create an MU HE PPDU, storing a map of PSDUs.
   *
   * This PPDU can either be UL or DL.
   *
   * \param psdus the PHY payloads (PSDUs)
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param ppduDuration the transmission duration of this PPDU
   * \param band the WifiPhyBand used for the transmission of this PPDU
   * \param uid the unique ID of this PPDU or of the triggering PPDU if this is an HE TB PPDU
   * \param flag the flag indicating the type of Tx PSD to build
   * \param p20Index the index of the primary 20 MHz channel
   */
  HePpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration,
          WifiPhyBand band, uint64_t uid, TxPsdFlag flag, uint8_t p20Index);
  /**
   * Destructor for HePpdu.
   */
  virtual ~HePpdu ();

  Time GetTxDuration (void) const override;
  Ptr<WifiPpdu> Copy (void) const override;
  WifiPpduType GetType (void) const override;
  uint16_t GetStaId (void) const override;
  uint16_t GetTransmissionChannelWidth (void) const override;
  bool CanBeReceived (uint16_t txCenterFreq, uint16_t p20MinFreq,
                      uint16_t p20MaxFreq) const override;

  /**
   * Get the payload of the PPDU.
   *
   * \param bssColor the BSS color of the PHY calling this function.
   * \param staId the STA-ID of the PHY calling this function.
   * \return the PSDU
   */
  Ptr<const WifiPsdu> GetPsdu (uint8_t bssColor, uint16_t staId = SU_STA_ID) const;

  /**
   * \return the transmit PSD flag set for this PPDU
   *
   * \see TxPsdFlag
   */
  TxPsdFlag GetTxPsdFlag (void) const;

  /**
   * \param flag the transmit PSD flag set for this PPDU
   *
   * \see TxPsdFlag
   */
  void SetTxPsdFlag (TxPsdFlag flag);

protected:
  std::string PrintPayload (void) const override;

  /**
   * Return true if the PPDU is a MU PPDU
   * \return true if the PPDU is a MU PPDU
   */
  bool IsMu (void) const;
  /**
   * Return true if the PPDU is a DL MU PPDU
   * \return true if the PPDU is a DL MU PPDU
   */
  bool IsDlMu (void) const;
  /**
   * Return true if the PPDU is an UL MU PPDU
   * \return true if the PPDU is an UL MU PPDU
   */
  bool IsUlMu (void) const;

  WifiTxVector::HeMuUserInfoMap m_muUserInfos; //!< the HE MU specific per-user information (to be removed once HE-SIG-B headers are implemented)

private:
  WifiTxVector DoGetTxVector (void) const override;

  /**
   * Fill in the HE PHY headers.
   *
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param ppduDuration the transmission duration of this PPDU
   */
  void SetPhyHeaders (const WifiTxVector& txVector, Time ppduDuration);

  HeSigHeader m_heSig;   //!< the HE-SIG PHY header
  TxPsdFlag m_txPsdFlag; //!< the transmit power spectral density flag
}; //class HePpdu

/**
* \brief Stream insertion operator.
*
* \param os the stream
* \param flag the transmit power spectral density flag
* \returns a reference to the stream
*/
std::ostream& operator<< (std::ostream& os, const HePpdu::TxPsdFlag &flag);

} //namespace ns3

#endif /* HE_PPDU_H */
