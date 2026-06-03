/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Duy Nguyen
 * Copyright (c) 2015 Ghada Badawy
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
 *Authors: Duy Nguyen <duy@soe.ucsc.edu>
 *         Ghada Badawy <gbadawy@gmail.com>
 *         Matias Richart <mrichart@fing.edu.uy>
 *
 * MinstrelHt is a rate adaptation algorithm for high-throughput (HT) 802.11
 */

#ifndef MINSTREL_HT_WIFI_MANAGER_H
#define MINSTREL_HT_WIFI_MANAGER_H

#include "ns3/wifi-remote-station-manager.h"
#include "minstrel-wifi-manager.h"
#include "ns3/wifi-mpdu-type.h"

namespace ns3 {

/**
 * Data structure to save transmission time calculations per rate.
 */
typedef std::map<WifiMode, Time> TxTime;

/**
 * \enum McsGroupType
 * \brief Available MCS group types
 */
enum McsGroupType
  {
    GROUP_HT = 0,
    GROUP_VHT,
    GROUP_HE
  };

/**
* \brief Stream insertion operator.
*
* \param os the stream
* \param type the MCS group type
* \returns a reference to the stream
*/
inline std::ostream& operator<< (std::ostream& os, McsGroupType type)
{
  switch (type)
    {
      case GROUP_HT:
        return (os << "HT");
      case GROUP_VHT:
        return (os << "VHT");
      case GROUP_HE:
        return (os << "HE");
      default:
        return (os << "INVALID");
    }
}

/**
 * Data structure to contain the information that defines a group.
 * It also contains the transmission times for all the MCS in the group.
 * A group is a collection of MCS defined by the number of spatial streams,
 * if it uses or not Short Guard Interval, and the channel width used.
 */
struct McsGroup
{
  uint8_t streams;   ///< number of spatial streams
  uint16_t gi;       ///< guard interval duration (nanoseconds)
  uint16_t chWidth;  ///< channel width (MHz)
  McsGroupType type; ///< identifies the group, \see McsGroupType
  bool isSupported;  ///< flag whether group is  supported
  // To accurately account for TX times, we separate the TX time of the first
  // MPDU in an A-MPDU from the rest of the MPDUs.
  TxTime ratesTxTimeTable;          ///< rates transmit time table
  TxTime ratesFirstMpduTxTimeTable; ///< rates MPDU transmit time table
};

/**
 * Data structure for a table of group definitions.
 * A vector of McsGroups.
 */
typedef std::vector<McsGroup> MinstrelMcsGroups;

struct MinstrelHtWifiRemoteStation;
/**
 * A struct to contain all statistics information related to a data rate.
 */
struct MinstrelHtRateInfo
{
  /**
   * Perfect transmission time calculation, or frame calculation.
   * Given a bit rate and a packet length n bytes.
   */
  Time perfectTxTime;
  bool supported;               //!< If the rate is supported.
  uint8_t mcsIndex;             //!< The index in the operationalMcsSet of the WifiRemoteStationManager.
  uint32_t retryCount;          //!< Retry limit.
  uint32_t adjustedRetryCount;  //!< Adjust the retry limit for this rate.
  uint32_t numRateAttempt;      //!< Number of transmission attempts so far.
  uint32_t numRateSuccess;      //!< Number of successful frames transmitted so far.
  double prob;                  //!< Current probability within last time interval. (# frame success )/(# total frames)
  bool retryUpdated;            //!< If number of retries was updated already.
  /**
   * Exponential weighted moving average of probability.
   * EWMA calculation:
   * ewma_prob =[prob *(100 - ewma_level) + (ewma_prob_old * ewma_level)]/100
   */
  double ewmaProb;
  double ewmsdProb;             //!< Exponential weighted moving standard deviation of probability.
  uint32_t prevNumRateAttempt;  //!< Number of transmission attempts with previous rate.
  uint32_t prevNumRateSuccess;  //!< Number of successful frames transmitted with previous rate.
  uint32_t numSamplesSkipped;   //!< Number of times this rate statistics were not updated because no attempts have been made.
  uint64_t successHist;         //!< Aggregate of all transmission successes.
  uint64_t attemptHist;         //!< Aggregate of all transmission attempts.
  double throughput;            //!< Throughput of this rate (in packets per second).
};

/**
 * Data structure for a Minstrel Rate table.
 * A vector of a struct MinstrelHtRateInfo.
 */
typedef std::vector<MinstrelHtRateInfo> MinstrelHtRate;

/**
 * A struct to contain information of a group.
 */
struct GroupInfo
{
  /**
   * MCS rates are divided into groups based on the number of streams and flags that they use.
   */
  uint8_t m_col;                  //!< Sample table column.
  uint8_t m_index;                //!< Sample table index.
  bool m_supported;               //!< If the rates of this group are supported by the station.
  uint16_t m_maxTpRate;           //!< The max throughput rate of this group in bps.
  uint16_t m_maxTpRate2;          //!< The second max throughput rate of this group in bps.
  uint16_t m_maxProbRate;         //!< The highest success probability rate of this group in bps.
  MinstrelHtRate m_ratesTable;    //!< Information about rates of this group.
};

/**
 * Data structure for a table of groups. Each group is of type GroupInfo.
 * A vector of a GroupInfo.
 */
typedef std::vector<struct GroupInfo> McsGroupData;

/**
 * Constants for maximum values.
 */

static const uint8_t MAX_HT_SUPPORTED_STREAMS = 4;  //!< Maximal number of streams supported by the HT PHY layer.
static const uint8_t MAX_VHT_SUPPORTED_STREAMS = 8; //!< Maximal number of streams supported by the VHT PHY layer.
static const uint8_t MAX_HE_SUPPORTED_STREAMS = 8;  //!< Maximal number of streams supported by the HE PHY layer.
static const uint8_t MAX_HT_STREAM_GROUPS = 4;      //!< Maximal number of groups per stream in HT (2 possible channel widths and 2 possible GI configurations).
static const uint8_t MAX_VHT_STREAM_GROUPS = 8;     //!< Maximal number of groups per stream in VHT (4 possible channel widths and 2 possible GI configurations).
static const uint8_t MAX_HE_STREAM_GROUPS = 12;     //!< Maximal number of groups per stream in HE (4 possible channel widths and 3 possible GI configurations).
static const uint8_t MAX_HT_GROUP_RATES = 8;        //!< Number of rates (or MCS) per HT group.
static const uint8_t MAX_VHT_GROUP_RATES = 10;      //!< Number of rates (or MCS) per VHT group.
static const uint8_t MAX_HE_GROUP_RATES = 12;       //!< Number of rates (or MCS) per HE group.
static const uint8_t MAX_HT_WIDTH = 40;             //!< Maximal channel width in MHz.
static const uint8_t MAX_VHT_WIDTH = 160;           //!< Maximal channel width in MHz.
static const uint8_t MAX_HE_WIDTH = 160;            //!< Maximal channel width in MHz.

/**
 * \brief Implementation of Minstrel-HT Rate Control Algorithm
 * \ingroup wifi
 *
 * Minstrel-HT is a rate adaptation mechanism for the 802.11n/ac/ax standards
 * based on Minstrel, and is based on the approach of probing the channel
 * to dynamically learn about working rates that can be supported.
 * Minstrel-HT is designed for high-latency devices that implement a
 * Multiple Rate Retry (MRR) chain. This kind of device does
 * not give feedback for every frame retransmission, but only when a frame
 * was correctly transmitted (an Ack is received) or a frame transmission
 * completely fails (all retransmission attempts fail).
 * The MRR chain is used to advise the hardware about which rate to use
 * when retransmitting a frame.
 *
 * Minstrel-HT adapts the MCS, channel width, number of streams, and
 * short guard interval (enabled or disabled).  For keeping statistics,
 * it arranges MCS in groups, where each group is defined by the
 * tuple (streams, GI, channel width).  There is a vector of all groups
 * supported by the PHY layer of the transmitter; for each group, the
 * capabilities and the estimated duration of its rates are maintained.
 *
 * Each station maintains a table of groups statistics. For each group, a flag
 * indicates if the group is supported by the station. Different stations
 * communicating with an AP can have different capabilities.
 *
 * Stats are updated per A-MPDU when receiving AmpduTxStatus. If the number
 * of successful or failed MPDUs is greater than zero (a BlockAck was
 * received), the rates are also updated.
 * If the number of successful and failed MPDUs is zero (BlockAck timeout),
 * then the rate selected is based on the MRR chain.
 *
 * On each update interval, it sets the maxThrRate, the secondmaxThrRate
 * and the maxProbRate for the MRR chain. These rates are only used when
 * an entire A-MPDU fails and is retried.
 *
 * Differently from legacy minstrel, sampling is not done based on
 * "lookaround ratio", but assuring all rates are sampled at least once
 * each interval. However, it samples less often the low rates and high
 * probability of error rates.
 *
 * When this rate control is configured but non-legacy modes are not supported,
 * Minstrel-HT uses legacy Minstrel (minstrel-wifi-manager) for rate control.
 */
class MinstrelHtWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  MinstrelHtWifiManager ();
  virtual ~MinstrelHtWifiManager ();

  void SetupPhy (const Ptr<WifiPhy> phy) override;
  void SetupMac (const Ptr<WifiMac> mac) override;
  int64_t AssignStreams (int64_t stream) override;

  /**
   * TracedCallback signature for rate change events.
   *
   * \param [in] rate The new rate.
   * \param [in] address The remote station MAC address.
   */
  typedef void (*RateChangeTracedCallback)(const uint64_t rate, const Mac48Address remoteAddress);


private:
  void DoInitialize (void) override;
  WifiRemoteStation * DoCreateStation (void) const override;
  void DoReportRxOk (WifiRemoteStation *station,
                     double rxSnr, WifiMode txMode) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsOk (WifiRemoteStation *station,
                      double ctsSnr, WifiMode ctsMode, double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode,
                       double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;
  void DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus,
                              double rxSnr, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
  bool DoNeedRetransmission (WifiRemoteStation *st, Ptr<const Packet> packet, bool normally) override;

  /**
   * Check the validity of a combination of number of streams, chWidth and mode.
   *
   * \param phy pointer to the wifi PHY
   * \param streams the number of streams
   * \param chWidth the channel width (MHz)
   * \param mode the wifi mode
   * \returns true if the combination is valid
   */
  bool IsValidMcs (Ptr<WifiPhy> phy, uint8_t streams, uint16_t chWidth, WifiMode mode);

  /**
   * Estimates the TxTime of a frame with a given mode and group (stream, guard interval and channel width).
   *
   * \param phy pointer to the wifi PHY
   * \param streams the number of streams
   * \param gi guard interval duration (nanoseconds)
   * \param chWidth the channel width (MHz)
   * \param mode the wifi mode
   * \param mpduType the type of the MPDU
   * \returns the transmit time
   */
  Time CalculateMpduTxDuration (Ptr<WifiPhy> phy, uint8_t streams, uint16_t gi,
                                uint16_t chWidth, WifiMode mode, MpduType mpduType);

  /**
   * Obtain the TxTime saved in the group information.
   *
   * \param groupId the group ID
   * \param mode the wifi mode
   * \returns the transmit time
   */
  Time GetMpduTxTime (uint8_t groupId, WifiMode mode) const;

  /**
   * Save a TxTime to the vector of groups.
   *
   * \param groupId the group ID
   * \param mode the wifi mode
   * \param t the transmit time
   */
  void AddMpduTxTime (uint8_t groupId, WifiMode mode, Time t);

  /**
   * Obtain the TxTime saved in the group information.
   *
   * \param groupId the group ID
   * \param mode the wifi mode
   * \returns the transmit time
   */
  Time GetFirstMpduTxTime (uint8_t groupId, WifiMode mode) const;

  /**
   * Save a TxTime to the vector of groups.
   *
   * \param groupId the group ID
   * \param mode the wifi mode
   * \param t the transmit time
   */
  void AddFirstMpduTxTime (uint8_t groupId, WifiMode mode, Time t);

  /**
   * Update the number of retries and reset accordingly.
   * \param station the wifi remote station
   */
  void UpdateRetry (MinstrelHtWifiRemoteStation *station);

  /**
   * Update the number of sample count variables.
   *
   * \param station the wifi remote station
   * \param nSuccessfulMpdus the number of successfully received MPDUs
   * \param nFailedMpdus the number of failed MPDUs
   */
  void UpdatePacketCounters (MinstrelHtWifiRemoteStation *station, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus);

  /**
   * Getting the next sample from Sample Table.
   *
   * \param station the wifi remote station
   * \returns the next sample
   */
  uint16_t GetNextSample (MinstrelHtWifiRemoteStation *station);

  /**
   * Set the next sample from Sample Table.
   *
   * \param station the wifi remote station
   */
  void SetNextSample (MinstrelHtWifiRemoteStation *station);

  /**
   * Find a rate to use from Minstrel Table.
   *
   * \param station the Minstrel-HT wifi remote station
   * \returns the rate in bps
   */
  uint16_t FindRate (MinstrelHtWifiRemoteStation *station);

  /**
   * Update the Minstrel Table.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void UpdateStats (MinstrelHtWifiRemoteStation *station);

  /**
   * Initialize Minstrel Table.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void RateInit (MinstrelHtWifiRemoteStation *station);

  /**
   * Return the average throughput of the MCS defined by groupId and rateId.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param groupId the group ID
   * \param rateId the rate ID
   * \param ewmaProb the EWMA probability
   * \returns the throughput in bps
   */
  double CalculateThroughput (MinstrelHtWifiRemoteStation *station, uint8_t groupId, uint8_t rateId, double ewmaProb);

  /**
   * Set index rate as maxTpRate or maxTp2Rate if is better than current values.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param index the index
   */
  void SetBestStationThRates (MinstrelHtWifiRemoteStation *station, uint16_t index);

  /**
   * Set index rate as maxProbRate if it is better than current value.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param index the index
   */
  void SetBestProbabilityRate (MinstrelHtWifiRemoteStation *station, uint16_t index);

  /**
   * Calculate the number of retransmissions to set for the index rate.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param index the index
   */
  void CalculateRetransmits (MinstrelHtWifiRemoteStation *station, uint16_t index);

  /**
   * Calculate the number of retransmissions to set for the (groupId, rateId) rate.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param groupId the group ID
   * \param rateId the rate ID
   */
  void CalculateRetransmits (MinstrelHtWifiRemoteStation *station, uint8_t groupId, uint8_t rateId);

  /**
   * Estimate the time to transmit the given packet with the given number of retries.
   * This function is "roughly" the function "calc_usecs_unicast_packet" in minstrel.c
   * in the madwifi implementation.
   *
   * The basic idea is that, we try to estimate the "average" time used to transmit the
   * packet for the given number of retries while also accounting for the 802.11 congestion
   * window change. The original code in the madwifi seems to estimate the number of backoff
   * slots as the half of the current CW size.
   *
   * There are four main parts:
   *  - wait for DIFS (sense idle channel)
   *  - Ack timeouts
   *  - Data transmission
   *  - backoffs according to CW
   *
   * \param dataTransmissionTime the data transmission time
   * \param shortRetries the short retries
   * \param longRetries the long retries
   * \returns the unicast packet time
   */
  Time CalculateTimeUnicastPacket (Time dataTransmissionTime, uint32_t shortRetries, uint32_t longRetries);

  /**
   * Perform EWMSD (Exponentially Weighted Moving Standard Deviation) calculation.
   *
   * \param oldEwmsd the old EWMSD
   * \param currentProb the current probability
   * \param ewmaProb the EWMA probability
   * \param weight the weight
   * \returns the EWMSD
   */
  double CalculateEwmsd (double oldEwmsd, double currentProb, double ewmaProb, double weight);

  /**
   * Initialize Sample Table.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void InitSampleTable (MinstrelHtWifiRemoteStation *station);

  /**
   * Printing Sample Table.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void PrintSampleTable (MinstrelHtWifiRemoteStation *station);

  /**
   * Printing Minstrel Table.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void PrintTable (MinstrelHtWifiRemoteStation *station);

  /**
   * Print group statistics.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param groupId the group ID
   * \param of the output file stream
   */
  void StatsDump (MinstrelHtWifiRemoteStation *station, uint8_t groupId, std::ofstream &of);

  /**
   * Check for initializations.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void CheckInit (MinstrelHtWifiRemoteStation *station);

  /**
   * Count retries.
   *
   * \param station the Minstrel-HT wifi remote station
   * \returns the count of retries
   */
  uint32_t CountRetries (MinstrelHtWifiRemoteStation * station);

  /**
   * Update rate.
   *
   * \param station the Minstrel-HT wifi remote station
   */
  void UpdateRate (MinstrelHtWifiRemoteStation *station);

  /**
   * Return the rateId inside a group, from the global index.
   *
   * \param index the index
   * \returns the rate ID
   */
  uint8_t GetRateId (uint16_t index);

  /**
   * Return the groupId from the global index.
   *
   * \param index the index
   * \returns the group ID
   */
  uint8_t GetGroupId (uint16_t index);

  /**
   * Returns the global index corresponding to the groupId and rateId.
   *
   * For managing rates from different groups, a global index for
   * all rates in all groups is used.
   * The group order is fixed by BW -> SGI -> streams.
   *
   * \param groupId the group ID
   * \param rateId the rate ID
   * \returns the index
   */
  uint16_t GetIndex (uint8_t groupId, uint8_t rateId);

  /**
   * Returns the groupId of an HT MCS with the given number of streams, GI and channel width used.
   *
   * \param txstreams the number of streams
   * \param gi guard interval duration (nanoseconds)
   * \param chWidth the channel width (MHz)
   * \returns the HT group ID
   */
  uint8_t GetHtGroupId (uint8_t txstreams, uint16_t gi, uint16_t chWidth);

  /**
   * Returns the groupId of a VHT MCS with the given number of streams, GI and channel width used.
   *
   * \param txstreams the number of streams
   * \param gi guard interval duration (nanoseconds)
   * \param chWidth the channel width (MHz)
   * \returns the VHT group ID
   */
  uint8_t GetVhtGroupId (uint8_t txstreams, uint16_t gi, uint16_t chWidth);

  /**
   * Returns the groupId of an HE MCS with the given number of streams, GI and channel width used.
   *
   * \param txstreams the number of streams
   * \param gi guard interval duration (nanoseconds)
   * \param chWidth the channel width (MHz)
   * \returns the HE group ID
   */
  uint8_t GetHeGroupId (uint8_t txstreams, uint16_t gi, uint16_t chWidth);

  /**
   * Returns the lowest global index of the rates supported by the station.
   *
   * \param station the Minstrel-HT wifi remote station
   * \returns the lowest global index
   */
  uint16_t GetLowestIndex (MinstrelHtWifiRemoteStation *station);

  /**
   * Returns the lowest global index of the rates supported by in the group.
   *
   * \param station the Minstrel-HT wifi remote station
   * \param groupId the group ID
   * \returns the lowest global index
   */
  uint16_t GetLowestIndex (MinstrelHtWifiRemoteStation *station, uint8_t groupId);

  /**
   * Returns a list of only the HE MCS supported by the device.
   * \returns the list of the HE MCS supported
   */
  WifiModeList GetHeDeviceMcsList () const;

  /**
   * Returns a list of only the VHT MCS supported by the device.
   * \returns the list of the VHT MCS supported
   */
  WifiModeList GetVhtDeviceMcsList (void) const;

  /**
   * Returns a list of only the HT MCS supported by the device.
   * \returns the list of the HT MCS supported
   */
  WifiModeList GetHtDeviceMcsList (void) const;

  Time m_updateStats;            //!< How frequent do we calculate the stats.
  Time m_legacyUpdateStats;      //!< How frequent do we calculate the stats for legacy MinstrelWifiManager.
  uint8_t m_lookAroundRate;      //!< The % to try other rates than our current rate.
  uint8_t m_ewmaLevel;           //!< Exponential weighted moving average level (or coefficient).
  uint8_t m_nSampleCol;          //!< Number of sample columns.
  uint32_t m_frameLength;        //!< Frame length used for calculate modes TxTime in bytes.
  uint8_t m_numGroups;           //!< Number of groups Minstrel should consider.
  uint8_t m_numRates;            //!< Number of rates per group Minstrel should consider.
  bool m_useLatestAmendmentOnly; //!< Flag if only the latest supported amendment by both peers should be used.
  bool m_printStats;             //!< If statistics table should be printed.

  MinstrelMcsGroups m_minstrelGroups;                 //!< Global array for groups information.

  Ptr<MinstrelWifiManager> m_legacyManager;           //!< Pointer to an instance of MinstrelWifiManager. Used when 802.11n/ac/ax not supported.

  Ptr<UniformRandomVariable> m_uniformRandomVariable; //!< Provides uniform random variables.

  TracedValue<uint64_t> m_currentRate; //!< Trace rate changes
};

} // namespace ns3

#endif /* MINSTREL_HT_WIFI_MANAGER_H */
