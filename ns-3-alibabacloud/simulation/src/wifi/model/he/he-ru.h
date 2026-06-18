/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef HE_RU_H
#define HE_RU_H

#include <map>
#include <vector>
#include <cstdint>
#include <ostream>

namespace ns3 {


/**
 * This class stores the subcarrier groups of all the available HE RUs.
 */
class HeRu
{
public:
  /**
   * The different HE Resource Unit (RU) types.
   */
  enum RuType
  {
    RU_26_TONE = 0,
    RU_52_TONE,
    RU_106_TONE,
    RU_242_TONE,
    RU_484_TONE,
    RU_996_TONE,
    RU_2x996_TONE
  };

  /// (lowest index, highest index) pair defining a subcarrier range
  typedef std::pair<int16_t, int16_t> SubcarrierRange;

  /// a vector of subcarrier ranges defining a subcarrier group
  typedef std::vector<SubcarrierRange> SubcarrierGroup;


  /**
   * RU Specification. Stores the information carried by the RU Allocation subfield
   * of the User Info field of Trigger frames (see 9.3.1.22.1 of 802.11ax D8.0).
   * Note that primary80MHz must be true if ruType is RU_2x996_TONE.
   * Internally, this class also stores the RU PHY index (ranging from 1 to the number
   * of RUs of the given type in a channel of the considered width), so that this class
   * contains all the information needed to locate the RU in a 160 MHz channel.
   */
  class RuSpec
  {
  public:
    /**
     * Default constructor
     */
    RuSpec ();
    /**
     * Constructor
     *
     * \param ruType the RU type
     * \param index the RU index (starting at 1)
     * \param primary80MHz whether the RU is allocated in the primary 80MHz channel
     */
    RuSpec (RuType ruType, std::size_t index, bool primary80MHz);

    /**
     * Get the RU type
     *
     * \return the RU type
     */
    RuType GetRuType (void) const;
    /**
     * Get the RU index
     *
     * \return the RU index
     */
    std::size_t GetIndex (void) const;
    /**
     * Get the primary 80 MHz flag
     *
     * \return true if the RU is in the primary 80 MHz channel and false otherwise
     */
    bool GetPrimary80MHz (void) const;
    /**
     * Set the RU PHY index
     *
     * \param bw the width of the channel of which the RU is part (in MHz)
     * \param p20Index the index of the primary20 channel
     */
    void SetPhyIndex (uint16_t bw, uint8_t p20Index);
    /**
     * Return true if the RU PHY index has been set, false otherwise
     *
     * \return true if the RU PHY index has been set, false otherwise
     */
    bool IsPhyIndexSet (void) const;
    /**
     * Get the RU PHY index
     *
     * \return the RU PHY index
     */
    std::size_t GetPhyIndex (void) const;

    /**
     * Compare this RU to the given RU.
     *
     * \param other the given RU
     * \return true if this RU compares equal to the given RU, false otherwise
     */
    bool operator== (const RuSpec& other) const;
    /**
     * Compare this RU to the given RU.
     *
     * \param other the given RU
     * \return true if this RU differs from the given RU, false otherwise
     */
    bool operator!= (const RuSpec& other) const;

  private:
    RuType m_ruType;         //!< RU type
    std::size_t m_index;     /**< RU index (starting at 1) as defined by Tables 27-7
                                  to 27-9 of 802.11ax D8.0 */
    bool m_primary80MHz;     //!< true if the RU is allocated in the primary 80MHz channel
    std::size_t m_phyIndex;  /**< the RU PHY index, which is used to indicate whether an
                                  RU is located in the lower half or the higher half of
                                  a 160MHz channel. For channel widths less than 160MHz,
                                  the RU PHY index equals the RU index */
  };


  /**
   * Get the number of distinct RUs of the given type (number of tones)
   * available in a HE PPDU of the given bandwidth.
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param ruType the RU type (number of tones)
   * \return the number of distinct RUs available
   */
  static std::size_t GetNRus (uint16_t bw, RuType ruType);

  /**
   * Get the set of distinct RUs of the given type (number of tones)
   * available in a HE PPDU of the given bandwidth.
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param ruType the RU type (number of tones)
   * \return the set of distinct RUs available
   */
  static std::vector<HeRu::RuSpec> GetRusOfType (uint16_t bw, HeRu::RuType ruType);

  /**
   * Get the set of 26-tone RUs that can be additionally allocated if the given
   * bandwidth is split in RUs of the given type.
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param ruType the RU type (number of tones)
   * \return the set of 26-tone RUs that can be additionally allocated
   */
  static std::vector<HeRu::RuSpec> GetCentral26TonesRus (uint16_t bw, HeRu::RuType ruType);

  /**
   * Get the subcarrier group of the RU having the given PHY index among all the
   * RUs of the given type (number of tones) available in a HE PPDU of the
   * given bandwidth. A subcarrier group is defined as one or more pairs
   * indicating the lowest frequency index and the highest frequency index.
   * Note that for channel width of 160 MHz the returned range is relative to
   * the 160 MHz channel (i.e. -1012 to 1012). The PHY index parameter is used to
   * distinguish between lower and higher 80 MHz subchannels.
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param ruType the RU type (number of tones)
   * \param phyIndex the PHY index (starting at 1) of the RU
   * \return the subcarrier range of the specified RU
   */
  static SubcarrierGroup GetSubcarrierGroup (uint16_t bw, RuType ruType, std::size_t phyIndex);

  /**
   * Check whether the given RU overlaps with the given set of RUs.
   * Note that for channel width of 160 MHz the returned range is relative to
   * the 160 MHz channel (i.e. -1012 to 1012).
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param ru the given RU allocation
   * \param v the given set of RUs
   * \return true if the given RU overlaps with the given set of RUs.
   */
  static bool DoesOverlap (uint16_t bw, RuSpec ru, const std::vector<RuSpec> &v);

  /**
   * Check whether the given RU overlaps with the given tone ranges.
   * Note that for channel width of 160 MHz the returned range is relative to
   * the 160 MHz channel (i.e. -1012 to 1012).
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param ru the given RU allocation
   * \param toneRanges the given set of tone ranges
   * \return true if the given RU overlaps with the given set of tone ranges.
   */
  static bool DoesOverlap (uint16_t bw, RuSpec ru, const SubcarrierGroup &toneRanges);

  /**
   * Find the RU allocation of the given RU type overlapping the given
   * reference RU allocation.
   * Note that an assert is generated if the RU allocation is not found.
   *
   * \param bw the bandwidth (MHz) of the HE PPDU (20, 40, 80, 160)
   * \param referenceRu the reference RU allocation
   * \param searchedRuType the searched RU type
   * \return the searched RU allocation.
   */
  static RuSpec FindOverlappingRu (uint16_t bw, RuSpec referenceRu, RuType searchedRuType);

  /**
   * Get the approximate bandwidth occupied by a RU.
   *
   * \param ruType the RU type
   * \return the approximate bandwidth (in MHz) occupied by the RU
   */
  static uint16_t GetBandwidth (RuType ruType);

  /**
   * Get the RU corresponding to the approximate bandwidth.
   *
   * \param bandwidth the approximate bandwidth (in MHz) occupied by the RU
   * \return the RU type
   */
  static RuType GetRuType (uint16_t bandwidth);

  /**
   * Given the channel bandwidth and the number of stations candidate for being
   * assigned an RU, maximize the number of candidate stations that can be assigned
   * an RU subject to the constraint that all the stations must be assigned an RU
   * of the same size (in terms of number of tones).
   *
   * \param bandwidth the channel bandwidth in MHz
   * \param nStations the number of candidate stations. On return, it is set to
   *                  the number of stations that are assigned an RU
   * \param[out] nCentral26TonesRus the number of additional 26-tone RUs that can be
   *                                allocated if the returned RU size is greater than 26 tones
   * \return the RU type
   */
  static RuType GetEqualSizedRusForStations (uint16_t bandwidth, std::size_t& nStations,
                                             std::size_t& nCentral26TonesRus);

  /// (bandwidth, number of tones) pair
  typedef std::pair<uint8_t, RuType> BwTonesPair;

  /// map (bandwidth, number of tones) pairs to the group of subcarrier ranges
  typedef std::map<BwTonesPair, std::vector<SubcarrierGroup> > SubcarrierGroups;

  /// Subcarrier groups for all RUs (with indices being applicable to primary 80 MHz channel)
  static const SubcarrierGroups  m_heRuSubcarrierGroups;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param ruType the RU type
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const HeRu::RuType &ruType);

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param ru the RU
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const HeRu::RuSpec &ru);

} //namespace ns3

#endif /* HE_RU_H */
