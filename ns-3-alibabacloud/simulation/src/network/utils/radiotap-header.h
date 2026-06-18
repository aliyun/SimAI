/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Foundation, Include., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef RADIOTAP_HEADER_H
#define RADIOTAP_HEADER_H

#include <ns3/header.h>

namespace ns3 {

/**
 * @brief Radiotap header implementation
 *
 * Radiotap is a de facto standard for 802.11 frame injection and reception.
 * The radiotap header format is a mechanism to supply additional information
 * about frames, from the driver to userspace applications such as libpcap, and
 * from a userspace application to the driver for transmission.
 */
class RadiotapHeader : public Header
{
public:
  RadiotapHeader ();
  /**
   * @brief Get the type ID.
   * @returns the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * This method is used by Packet::AddHeader to store the header into the byte
   * buffer of a packet.  This method returns the number of bytes which are
   * needed to store the header data during a Serialize.
   *
   * @returns The expected size of the header.
   */
  virtual uint32_t GetSerializedSize (void) const;

  /**
   * This method is used by Packet::AddHeader to store the header into the byte
   * buffer of a packet.  The data written is expected to match bit-for-bit the
   * representation of this header in a real network.
   *
   * @param start An iterator which points to where the header should
   *              be written.
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * This method is used by Packet::RemoveHeader to re-create a header from the
   * byte buffer of a packet.  The data read is expected to match bit-for-bit
   * the representation of this header in real networks.
   *
   * @param start An iterator which points to where the header should
   *              written.
   * @returns The number of bytes read.
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

  /**
   * This method is used by Packet::Print to print the content of the header as
   * ascii data to a C++ output stream.  Although the header is free to format
   * its output as it wishes, it is recommended to follow a few rules to integrate
   * with the packet pretty printer: start with flags, small field
   * values located between a pair of parens. Values should be separated
   * by whitespace. Follow the parens with the important fields,
   * separated by whitespace.
   *
   * eg: (field1 val1 field2 val2 field3 val3) field4 val4 field5 val5
   *
   * @param os The output stream
   */
  virtual void Print (std::ostream &os) const;

  /**
   * @brief Set the Time Synchronization Function Timer (TSFT) value.  Valid for
   * received frames only.
   *
   * @param tsft Value in microseconds of the MAC's 64-bit 802.11 Time
   *             Synchronization Function timer when the first bit of the MPDU
   *             arrived at the MAC.
   */
  void SetTsft (uint64_t tsft);

  /**
   * @brief Frame flags.
   */
  enum FrameFlag
  {
    FRAME_FLAG_NONE           = 0x00, /**< No flags set */
    FRAME_FLAG_CFP            = 0x01, /**< Frame sent/received during CFP */
    FRAME_FLAG_SHORT_PREAMBLE = 0x02, /**< Frame sent/received with short preamble */
    FRAME_FLAG_WEP            = 0x04, /**< Frame sent/received with WEP encryption */
    FRAME_FLAG_FRAGMENTED     = 0x08, /**< Frame sent/received with fragmentation */
    FRAME_FLAG_FCS_INCLUDED   = 0x10, /**< Frame includes FCS */
    FRAME_FLAG_DATA_PADDING   = 0x20, /**< Frame has padding between 802.11 header and payload (to 32-bit boundary) */
    FRAME_FLAG_BAD_FCS        = 0x40, /**< Frame failed FCS check */
    FRAME_FLAG_SHORT_GUARD    = 0x80  /**< Frame used short guard interval (HT) */
  };

  /**
   * @brief Set the frame flags of the transmitted or received frame.
   * @param flags flags to set.
   */
  void SetFrameFlags (uint8_t flags);

  /**
   * @brief Set the transmit/receive channel frequency in units of megahertz
   * @param rate the transmit/receive channel frequency in units of megahertz.
   */
  void SetRate (uint8_t rate);

  /**
   * @brief Channel flags.
   */
  enum ChannelFlags
  {
    CHANNEL_FLAG_NONE          = 0x0000, /**< No flags set */
    CHANNEL_FLAG_TURBO         = 0x0010, /**< Turbo Channel */
    CHANNEL_FLAG_CCK           = 0x0020, /**< CCK channel */
    CHANNEL_FLAG_OFDM          = 0x0040, /**< OFDM channel */
    CHANNEL_FLAG_SPECTRUM_2GHZ = 0x0080, /**< 2 GHz spectrum channel */
    CHANNEL_FLAG_SPECTRUM_5GHZ = 0x0100, /**< 5 GHz spectrum channel */
    CHANNEL_FLAG_PASSIVE       = 0x0200, /**< Only passive scan allowed */
    CHANNEL_FLAG_DYNAMIC       = 0x0400, /**< Dynamic CCK-OFDM channel */
    CHANNEL_FLAG_GFSK          = 0x0800  /**< GFSK channel (FHSS PHY) */
  };

  /**
   * @brief Set the transmit/receive channel frequency and flags
   * @param frequency The transmit/receive data rate in units of 500 kbps.
   * @param flags The flags to set.
   * @see ChannelFlags
   */
  void SetChannelFrequencyAndFlags (uint16_t frequency, uint16_t flags);

  /**
   * @brief Set the RF signal power at the antenna as a decibel difference
   * from an arbitrary, fixed reference.
   *
   * @param signal The RF signal power at the antenna as a decibel difference
   *               from an arbitrary, fixed reference;
   */
  void SetAntennaSignalPower (double signal);

  /**
   * @brief Set the RF noise power at the antenna as a decibel difference
   * from an arbitrary, fixed reference.
   *
   * @param noise The RF noise power at the antenna as a decibel difference
   *              from an arbitrary, fixed reference.
   */
  void SetAntennaNoisePower (double noise);

  /**
   * @brief MCS known bits.
   */
  enum McsKnown
  {
    MCS_KNOWN_NONE           = 0x00, /**< No flags set */
    MCS_KNOWN_BANDWIDTH      = 0x01, /**< Bandwidth */
    MCS_KNOWN_INDEX          = 0x02, /**< MCS index known */
    MCS_KNOWN_GUARD_INTERVAL = 0x04, /**< Guard interval */
    MCS_KNOWN_HT_FORMAT      = 0x08, /**< HT format */
    MCS_KNOWN_FEC_TYPE       = 0x10, /**< FEC type */
    MCS_KNOWN_STBC           = 0x20, /**< STBC known */
    MCS_KNOWN_NESS           = 0x40, /**< Ness known (Number of extension spatial streams) */
    MCS_KNOWN_NESS_BIT_1     = 0x80, /**< Ness data - bit 1 (MSB) of Number of extension spatial streams */
  };

  /**
   * @brief MCS flags.
   */
  enum McsFlags
  {
    MCS_FLAGS_NONE           = 0x00, /**< Default: 20 MHz, long guard interval, mixed HT format and BCC FEC type */
    MCS_FLAGS_BANDWIDTH_40   = 0x01, /**< 40 MHz */
    MCS_FLAGS_BANDWIDTH_20L  = 0x02, /**< 20L (20 MHz in lower half of 40 MHz channel) */
    MCS_FLAGS_BANDWIDTH_20U  = 0x03, /**< 20U (20 MHz in upper half of 40 MHz channel) */
    MCS_FLAGS_GUARD_INTERVAL = 0x04, /**< Short guard interval */
    MCS_FLAGS_HT_GREENFIELD  = 0x08, /**< Greenfield HT format */
    MCS_FLAGS_FEC_TYPE       = 0x10, /**< LDPC FEC type */
    MCS_FLAGS_STBC_STREAMS   = 0x60, /**< STBC enabled */
    MCS_FLAGS_NESS_BIT_0     = 0x80, /**< Ness - bit 0 (LSB) of Number of extension spatial streams */
  };

  /**
   * @brief Set the MCS fields
   *
   * @param known The kwown flags.
   * @param flags The flags to set.
   * @param mcs The MCS index value.
   */
  void SetMcsFields (uint8_t known, uint8_t flags, uint8_t mcs);

  /**
   * @brief A-MPDU status flags.
   */
  enum AmpduFlags
  {
    A_MPDU_STATUS_NONE                = 0x00, /**< No flags set */
    A_MPDU_STATUS_REPORT_ZERO_LENGTH  = 0x01, /**< Driver reports 0-length subframes */
    A_MPDU_STATUS_IS_ZERO_LENGTH      = 0x02, /**< Frame is 0-length subframe (valid only if 0x0001 is set) */
    A_MPDU_STATUS_LAST_KNOWN          = 0x04, /**< Last subframe is known (should be set for all subframes in an A-MPDU) */
    A_MPDU_STATUS_LAST                = 0x08, /**< This frame is the last subframe */
    A_MPDU_STATUS_DELIMITER_CRC_ERROR = 0x10, /**< Delimiter CRC error */
    A_MPDU_STATUS_DELIMITER_CRC_KNOWN = 0x20  /**< Delimiter CRC value known: the delimiter CRC value field is valid */
  };

  /**
   * @brief Set the A-MPDU status fields
   *
   * @param referenceNumber The A-MPDU reference number to identify all subframes belonging to the same A-MPDU.
   * @param flags The flags to set.
   * @param crc The CRC value value.
   */
  void SetAmpduStatus (uint32_t referenceNumber, uint16_t flags, uint8_t crc);

  /**
   * @brief VHT known bits.
   */
  enum VhtKnown
  {
    VHT_KNOWN_NONE                         = 0x0000, /**< No flags set */
    VHT_KNOWN_STBC                         = 0x0001, /**< Space-time block coding (1 if all spatial streams of all users have STBC, 0 otherwise). */
    VHT_KNOWN_TXOP_PS_NOT_ALLOWED          = 0x0002, /**< TXOP_PS_NOT_ALLOWED known */
    VHT_KNOWN_GUARD_INTERVAL               = 0x0004, /**< Guard interval */
    VHT_KNOWN_SHORT_GI_NSYM_DISAMBIGUATION = 0x0008, /**< Short GI NSYM disambiguation known */
    VHT_KNOWN_LDPC_EXTRA_OFDM_SYMBOL       = 0x0010, /**< LDPC extra OFDM symbol known */
    VHT_KNOWN_BEAMFORMED                   = 0x0020, /**< Beamformed known/applicable (this flag should be set to zero for MU PPDUs). */
    VHT_KNOWN_BANDWIDTH                    = 0x0040, /**< Bandwidth known */
    VHT_KNOWN_GROUP_ID                     = 0x0080, /**< Group ID known */
    VHT_KNOWN_PARTIAL_AID                  = 0x0100, /**< Partial AID known/applicable */
  };

  /**
   * @brief VHT flags.
   */
  enum VhtFlags
  {
    VHT_FLAGS_NONE                         = 0x00, /**< No flags set */
    VHT_FLAGS_STBC                         = 0x01, /**< Set if all spatial streams of all users have space-time block coding */
    VHT_FLAGS_TXOP_PS_NOT_ALLOWED          = 0x02, /**< Set if STAs may not doze during TXOP (valid only for AP transmitters). */
    VHT_FLAGS_GUARD_INTERVAL               = 0x04, /**< Short guard interval */
    VHT_FLAGS_SHORT_GI_NSYM_DISAMBIGUATION = 0x08, /**< Set if NSYM mod 10 = 9 (valid only if short GI is used).*/
    VHT_FLAGS_LDPC_EXTRA_OFDM_SYMBOL       = 0x10, /**< Set if one or more users are using LDPC and the encoding process resulted in extra OFDM symbol(s) */
    VHT_FLAGS_BEAMFORMED                   = 0x20, /**< Set if beamforming is used (valid for SU PPDUs only). */
  };

  /**
   * @brief Set the VHT fields
   *
   * @param known The kwown flags.
   * @param flags The flags to set.
   * @param bandwidth The bandwidth value.
   * @param mcs_nss The mcs_nss value.
   * @param coding The coding value.
   * @param group_id The group_id value.
   * @param partial_aid The partial_aid value.
   */
  void SetVhtFields (uint16_t known, uint8_t flags,
                     uint8_t bandwidth, uint8_t mcs_nss [4],
                     uint8_t coding, uint8_t group_id,
                     uint16_t partial_aid);

  /**
   * @brief HE data1.
   */
  enum HeData1
  {
    HE_DATA1_FORMAT_EXT_SU      = 0x0001, /**< HE EXT SU PPDU format */
    HE_DATA1_FORMAT_MU          = 0x0002, /**< HE MU PPDU format */
    HE_DATA1_FORMAT_TRIG        = 0x0003, /**< HE TRIG PPDU format */
    HE_DATA1_BSS_COLOR_KNOWN    = 0x0004, /**< BSS Color known */
    HE_DATA1_BEAM_CHANGE_KNOWN  = 0x0008, /**< Beam Change known */
    HE_DATA1_UL_DL_KNOWN        = 0x0010, /**< UL/DL known */
    HE_DATA1_DATA_MCS_KNOWN     = 0x0020, /**< data MCS known */
    HE_DATA1_DATA_DCM_KNOWN     = 0x0040, /**< data DCM known */
    HE_DATA1_CODING_KNOWN       = 0x0080, /**< Coding known */
    HE_DATA1_LDPC_XSYMSEG_KNOWN = 0x0100, /**< LDPC extra symbol segment known */
    HE_DATA1_STBC_KNOWN         = 0x0200, /**< STBC known */
    HE_DATA1_SPTL_REUSE_KNOWN   = 0x0400, /**< Spatial Reuse known (Spatial Reuse 1 for HE TRIG PPDU format) */
    HE_DATA1_SPTL_REUSE2_KNOWN  = 0x0800, /**< Spatial Reuse 2 known (HE TRIG PPDU format), STA-ID known (HE MU PPDU format) */
    HE_DATA1_SPTL_REUSE3_KNOWN  = 0x1000, /**< Spatial Reuse 3 known (HE TRIG PPDU format) */
    HE_DATA1_SPTL_REUSE4_KNOWN  = 0x2000, /**< Spatial Reuse 4 known (HE TRIG PPDU format) */
    HE_DATA1_BW_RU_ALLOC_KNOWN  = 0x4000, /**< data BW/RU allocation known */
    HE_DATA1_DOPPLER_KNOWN      = 0x8000, /**< Doppler known */
  };

  /**
   * @brief HE data2.
   */
  enum HeData2
  {
    HE_DATA2_PRISEC_80_KNOWN    = 0x0001, /**< pri/sec 80 MHz known */
    HE_DATA2_GI_KNOWN           = 0x0002, /**< GI known */
    HE_DATA2_NUM_LTF_SYMS_KNOWN = 0x0004, /**< number of LTF symbols known */
    HE_DATA2_PRE_FEC_PAD_KNOWN  = 0x0008, /**< Pre-FEC Padding Factor known */
    HE_DATA2_TXBF_KNOWN         = 0x0010, /**< TxBF known */
    HE_DATA2_PE_DISAMBIG_KNOWN  = 0x0020, /**< PE Disambiguity known */
    HE_DATA2_TXOP_KNOWN         = 0x0040, /**< TXOP known */
    HE_DATA2_MIDAMBLE_KNOWN     = 0x0080, /**< midamble periodicity known */
    HE_DATA2_RU_OFFSET          = 0x3f00, /**< RU allocation offset */
    HE_DATA2_RU_OFFSET_KNOWN    = 0x4000, /**< RU allocation offset known */
    HE_DATA2_PRISEC_80_SEC      = 0x8000, /**< pri/sec 80 MHz */
  };

  /**
   * @brief HE data5.
   */
  enum HeData5
  {
    HE_DATA5_DATA_BW_RU_ALLOC_40MHZ  = 0x0001, /**< 40 MHz data Bandwidth */
    HE_DATA5_DATA_BW_RU_ALLOC_80MHZ  = 0x0002, /**< 80 MHz data Bandwidth */
    HE_DATA5_DATA_BW_RU_ALLOC_160MHZ = 0x0003, /**< 160 MHz data Bandwidth */
    HE_DATA5_DATA_BW_RU_ALLOC_26T    = 0x0004, /**< 26-tone RU allocation */
    HE_DATA5_DATA_BW_RU_ALLOC_52T    = 0x0005, /**< 52-tone RU allocation */
    HE_DATA5_DATA_BW_RU_ALLOC_106T   = 0x0006, /**< 106-tone RU allocation */
    HE_DATA5_DATA_BW_RU_ALLOC_242T   = 0x0007, /**< 242-tone RU allocation */
    HE_DATA5_DATA_BW_RU_ALLOC_484T   = 0x0008, /**< 484-tone RU allocation */
    HE_DATA5_DATA_BW_RU_ALLOC_996T   = 0x0009, /**< 996-tone RU allocation */
    HE_DATA5_DATA_BW_RU_ALLOC_2x996T = 0x000a, /**< 2x996-tone RU allocation */
    HE_DATA5_GI_1_6                  = 0x0010, /**< 1.6us GI */
    HE_DATA5_GI_3_2                  = 0x0020, /**< 3.2us GI */
    HE_DATA5_LTF_SYM_SIZE            = 0x00c0, /**< LTF symbol size */
    HE_DATA5_NUM_LTF_SYMS            = 0x0700, /**< number of LTF symbols */
    HE_DATA5_PRE_FEC_PAD             = 0x3000, /**< Pre-FEC Padding Factor */
    HE_DATA5_TXBF                    = 0x4000, /**< TxBF */
    HE_DATA5_PE_DISAMBIG             = 0x8000, /**< PE Disambiguity */
  };

  /**
   * @brief Set the HE fields
   *
   * @param data1 The data1 field.
   * @param data2 The data2 field.
   * @param data3 The data3 field.
   * @param data4 The data4 field.
   * @param data5 The data5 field.
   * @param data6 The data6 field.
   */
  void SetHeFields (uint16_t data1,
                    uint16_t data2,
                    uint16_t data3,
                    uint16_t data4,
                    uint16_t data5,
                    uint16_t data6);

  /**
   * @brief HE MU flags1.
   */
  enum HeMuFlags1
  {
    HE_MU_FLAGS1_SIGB_MCS                = 0x000f, /**< SIG-B MCS (from SIG-A) */
    HE_MU_FLAGS1_SIGB_MCS_KNOWN          = 0x0010, /**< SIG-B MCS known */
    HE_MU_FLAGS1_SIGB_DCM                = 0x0020, /**< SIG-B DCM (from SIG-A) */
    HE_MU_FLAGS1_SIGB_DCM_KNOWN          = 0x0040, /**< SIG-B DCM known */
    HE_MU_FLAGS1_CH2_CENTER_26T_RU_KNOWN = 0x0080, /**< (Channel 2) Center 26-tone RU bit known */
    HE_MU_FLAGS1_CH1_RUS_KNOWN           = 0x0100, /**< Channel 1 RUs known (which depends on BW) */
    HE_MU_FLAGS1_CH2_RUS_KNOWN           = 0x0200, /**< Channel 2 RUs known (which depends on BW) */
    HE_MU_FLAGS1_CH1_CENTER_26T_RU_KNOWN = 0x1000, /**< (Channel 1) Center 26-tone RU bit known */
    HE_MU_FLAGS1_CH1_CENTER_26T_RU       = 0x2000, /**< (Channel 1) Center 26-tone RU value */
    HE_MU_FLAGS1_SIGB_COMPRESSION_KNOWN  = 0x4000, /**< SIG-B Compression known */
    HE_MU_FLAGS1_NUM_SIGB_SYMBOLS_KNOWN  = 0x8000, /**< # of HE-SIG-B Symbols/MU-MIMO Users known */
  };

  /**
   * @brief HE MU flags2.
   */
  enum HeMuFlags2
  {
    HE_MU_FLAGS2_BW_FROM_SIGA                                 = 0x0003, /**< Bandwidth from Bandwidth field in HE-SIG-A */
    HE_MU_FLAGS2_BW_FROM_SIGA_KNOWN                           = 0x0004, /**< Bandwidth from Bandwidth field in HE-SIG-A known */
    HE_MU_FLAGS2_SIGB_COMPRESSION_FROM_SIGA                   = 0x0008, /**< SIG-B compression from SIG-A */
    HE_MU_FLAGS2_NUM_SIGB_SYMBOLS_FROM_SIGA                   = 0x00f0, /**< # of HE-SIG-B Symbols - 1 or # of MU-MIMO Users - 1 from SIG-A */
    HE_MU_FLAGS2_PREAMBLE_PUNCTURING_FROM_SIGA_BW_FIELD       = 0x0300, /**< Preamble puncturing from Bandwidth field in HE-SIG-A */
    HE_MU_FLAGS2_PREAMBLE_PUNCTURING_FROM_SIGA_BW_FIELD_KNOWN = 0x0400, /**< Preamble puncturing from Bandwidth field in HE-SIG-A known */
    HE_MU_FLAGS2_CH2_CENTER_26T_RU                            = 0x0800, /**< (Channel 2) Center 26-tone RU value */
  };

  /**
   * @brief Set the HE MU fields
   *
   * @param flags1 The flags1 field.
   * @param flags2 The flags2 field.
   * @param ruChannel1 The RU_channel1 field.
   * @param ruChannel2 The RU_channel2 field.
   */
  void SetHeMuFields (uint16_t flags1, uint16_t flags2, const std::array<uint8_t, 4> &ruChannel1, const std::array<uint8_t, 4> &ruChannel2);

  /**
   * @brief HE MU per_user_known.
   */
  enum HeMuPerUserKnown
  {
    HE_MU_PER_USER_POSITION_KNOWN              = 0x01, /**< User field position known */
    HE_MU_PER_USER_STA_ID_KNOWN                = 0x02, /**< STA-ID known */
    HE_MU_PER_USER_NSTS_KNOWN                  = 0x04, /**< NSTS known */
    HE_MU_PER_USER_TX_BF_KNOWN                 = 0x08, /**< Tx Beamforming known */
    HE_MU_PER_USER_SPATIAL_CONFIGURATION_KNOWN = 0x10, /**< Spatial Configuration known */
    HE_MU_PER_USER_MCS_KNOWN                   = 0x20, /**< MCS known */
    HE_MU_PER_USER_DCM_KNOWN                   = 0x40, /**< DCM known */
    HE_MU_PER_USER_CODING_KNOWN                = 0x80, /**< Coding known */
  };

  /**
   * @brief Set the HE MU per user fields
   *
   * @param perUser1 The per_user_1 field.
   * @param perUser2 The per_user_2 field.
   * @param perUserPosition The per_user_position field.
   * @param perUserKnown The per_user_known field.
   */
  void SetHeMuPerUserFields (uint16_t perUser1, uint16_t perUser2, uint8_t perUserPosition, uint8_t perUserKnown);

private:
  /**
   * @brief Radiotap flags.
   */
  enum RadiotapFlags
  {
    RADIOTAP_TSFT              = 0x00000001,
    RADIOTAP_FLAGS             = 0x00000002,
    RADIOTAP_RATE              = 0x00000004,
    RADIOTAP_CHANNEL           = 0x00000008,
    RADIOTAP_FHSS              = 0x00000010,
    RADIOTAP_DBM_ANTSIGNAL     = 0x00000020,
    RADIOTAP_DBM_ANTNOISE      = 0x00000040,
    RADIOTAP_LOCK_QUALITY      = 0x00000080,
    RADIOTAP_TX_ATTENUATION    = 0x00000100,
    RADIOTAP_DB_TX_ATTENUATION = 0x00000200,
    RADIOTAP_DBM_TX_POWER      = 0x00000400,
    RADIOTAP_ANTENNA           = 0x00000800,
    RADIOTAP_DB_ANTSIGNAL      = 0x00001000,
    RADIOTAP_DB_ANTNOISE       = 0x00002000,
    RADIOTAP_RX_FLAGS          = 0x00004000,
    RADIOTAP_MCS               = 0x00080000,
    RADIOTAP_AMPDU_STATUS      = 0x00100000,
    RADIOTAP_VHT               = 0x00200000,
    RADIOTAP_HE                = 0x00800000,
    RADIOTAP_HE_MU             = 0x01000000,
    RADIOTAP_HE_MU_OTHER_USER  = 0x02000000,
    RADIOTAP_ZERO_LEN_PSDU     = 0x04000000,
    RADIOTAP_LSIG              = 0x08000000,
    RADIOTAP_EXT               = 0x80000000
  };

  uint16_t m_length;        //!< entire length of radiotap data + header
  uint32_t m_present;       //!< bits describing which fields follow header

  uint64_t m_tsft;          //!< Time Synchronization Function Timer (when the first bit of the MPDU arrived at the MAC)
  uint8_t m_flags;          //!< Properties of transmitted and received frames.
  uint8_t m_rate;           //!< TX/RX data rate in units of 500 kbps
  uint8_t m_channelPad;     //!< Tx/Rx channel padding.
  uint16_t m_channelFreq;   //!< Tx/Rx frequency in MHz.
  uint16_t m_channelFlags;  //!< Tx/Rx channel flags.
  int8_t m_antennaSignal;   //!< RF signal power at the antenna, dB difference from an arbitrary, fixed reference.
  int8_t m_antennaNoise;    //!< RF noise power at the antenna, dB difference from an arbitrary, fixed reference.

  uint8_t m_mcsKnown; //!< MCS Flags, known information field.
  uint8_t m_mcsFlags; //!< MCS Flags, flags field.
  uint8_t m_mcsRate;  //!< MCS Flags, mcs rate index.

  uint8_t m_ampduStatusPad;       //!< A-MPDU Status Flags, padding before A-MPDU Status Field.
  uint32_t m_ampduStatusRef;      //!< A-MPDU Status Flags, reference number.
  uint16_t m_ampduStatusFlags;    //!< A-MPDU Status Flags, information about the received A-MPDU.
  uint8_t m_ampduStatusCRC;       //!< A-MPDU Status Flags, delimiter CRC value.

  uint8_t m_vhtPad;         //!< VHT padding.
  uint16_t m_vhtKnown;      //!< VHT known field.
  uint8_t m_vhtFlags;       //!< VHT flags field.
  uint8_t m_vhtBandwidth;   //!< VHT bandwidth field.
  uint8_t m_vhtMcsNss[4];   //!< VHT mcs_nss field.
  uint8_t m_vhtCoding;      //!< VHT coding field.
  uint8_t m_vhtGroupId;     //!< VHT group_id field.
  uint16_t m_vhtPartialAid; //!< VHT partial_aid field.

  uint8_t m_hePad;         //!< HE padding.
  uint16_t m_heData1;      //!< HE data1 field.
  uint16_t m_heData2;      //!< HE data2 field.
  uint16_t m_heData3;      //!< HE data3 field.
  uint16_t m_heData4;      //!< HE data4 field.
  uint16_t m_heData5;      //!< HE data5 field.
  uint16_t m_heData6;      //!< HE data6 field.

  uint8_t m_heMuPad;       //!< HE MU padding.
  uint16_t m_heMuFlags1;   //!< HE MU flags1 field.
  uint16_t m_heMuFlags2;   //!< HE MU flags2 field.

  uint8_t m_heMuOtherUserPad;    //!< HE MU other user padding.
  uint16_t m_heMuPerUser1;       //!< HE MU per_user_1 field.
  uint16_t m_heMuPerUser2;       //!< HE MU per_user_2 field.
  uint8_t m_heMuPerUserPosition; //!< HE MU per_user_position field.
  uint8_t m_heMuPerUserKnown;    //!< HE MU per_user_known field.
};

} // namespace ns3

#endif /*  RADIOTAP_HEADER_H */
