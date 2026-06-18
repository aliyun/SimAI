/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */


#ifndef EPS_BEARER
#define EPS_BEARER

#include <ns3/uinteger.h>
#include <ns3/object-base.h>
#include <unordered_map>

namespace ns3 {

/**
 * 3GPP TS 36.413 9.2.1.18 GBR QoS Information
 *
 */
struct GbrQosInformation
{
  /** 
   * Default constructor, initializes member variables to zero or equivalent
   */
  GbrQosInformation ();

  uint64_t gbrDl;  /**< Guaranteed Bit Rate (bit/s) in downlink */
  uint64_t gbrUl;  /**< Guaranteed Bit Rate (bit/s) in uplink */
  uint64_t mbrDl;  /**< Maximum Bit Rate (bit/s) in downlink */
  uint64_t mbrUl;  /**< Maximum Bit Rate (bit/s) in uplink */
};


/**
 * 3GPP 23.203 Section 6.1.7.3 Allocation and Retention Priority characteristics
 *
 */
struct AllocationRetentionPriority
{
  /** 
   * Default constructor, initializes member variables to zero or equivalent
   */
  AllocationRetentionPriority ();
  uint8_t priorityLevel;     ///< 1-15; 1 = highest
  bool preemptionCapability; ///< true if bearer can preempt others
  bool preemptionVulnerability; ///< true if bearer can be preempted by others
};

/**
 * \brief This class contains the specification of EPS Bearers.
 *
 * See the following references:
 * 3GPP TS 23.203, Section 4.7.2 The EPS bearer
 * 3GPP TS 23.203, Section 4.7.3 Bearer level QoS parameters
 * 3GPP TS 36.413 Section 9.2.1.15 E-RAB Level QoS Parameters
 *
 * It supports the selection of different specifications depending on the
 * release. To change the release, change the attribute "Release". Please remember
 * that we must expose to all releases the most recent Qci. Asking for Qci parameters
 * for a release in which it has not been created will result in a crash.
 *
 * For example, if you select Release 11 (or if you don't select anything, as
 * it is the default selection) and then ask for the packet error rate of
 * the NGBR_MC_DELAY_SIGNAL Qci, the program will crash.
 *
 * Please note that from Release 8 (the latest when the LENA project finished)
 * to Release 11, the bearers ID and requirements are the same. From Release 12,
 * they started to change, and the latest version is now Release 15. However,
 * we do not support intermediate types: in other words, you can select from
 * Release 8 to Release 11, or Release 15. Any other value will result in a
 * program crash.
 *
 * The release version only affect Bearer definitions. Other part of the LTE
 * module are not affected when changing the Release attribute.
 */
class EpsBearer : public ObjectBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  virtual TypeId GetInstanceTypeId (void) const override;

  /**
   * QoS Class Indicator. See 3GPP 23.203 Section 6.1.7.2 for standard values.
   * Updated to Release 15.
   */
  enum Qci : uint8_t
  {
    GBR_CONV_VOICE          = 1,    ///< GBR Conversational Voice
    GBR_CONV_VIDEO          = 2,    ///< GBR Conversational Video (Live streaming)
    GBR_GAMING              = 3,    ///< GBR Real Time Gaming
    GBR_NON_CONV_VIDEO      = 4,    ///< GBR Non-Conversational Video (Buffered Streaming)
    GBR_MC_PUSH_TO_TALK     = 65,   ///< GBR Mission Critical User Plane Push To Talk voice
    GBR_NMC_PUSH_TO_TALK    = 66,   ///< GBR Non-Mission-Critical User Plane Push To Talk voice
    GBR_MC_VIDEO            = 67,   ///< GBR Mission Critical Video User Plane
    GBR_V2X                 = 75,   ///< GBR V2X Messages
    NGBR_IMS                = 5,    ///< Non-GBR IMS Signalling
    NGBR_VIDEO_TCP_OPERATOR = 6,    ///< Non-GBR TCP-based Video (Buffered Streaming, e.g., www, e-mail...)
    NGBR_VOICE_VIDEO_GAMING = 7,    ///< Non-GBR Voice, Video, Interactive Streaming
    NGBR_VIDEO_TCP_PREMIUM  = 8,    ///< Non-GBR TCP-based Video (Buffered Streaming, e.g., www, e-mail...)
    NGBR_VIDEO_TCP_DEFAULT  = 9,    ///< Non-GBR TCP-based Video (Buffered Streaming, e.g., www, e-mail...)
    NGBR_MC_DELAY_SIGNAL    = 69,   ///< Non-GBR Mission Critical Delay Sensitive Signalling (e.g., MC-PTT)
    NGBR_MC_DATA            = 70,   ///< Non-GBR Mission Critical Data
    NGBR_V2X                = 79,   ///< Non-GBR V2X Messages
    NGBR_LOW_LAT_EMBB       = 80,   ///< Non-GBR Low Latency eMBB applications
    DGBR_DISCRETE_AUT_SMALL = 82,   ///< Delay-Critical GBR Discrete Automation Small Packets (TS 22.261)
    DGBR_DISCRETE_AUT_LARGE = 83,   ///< Delay-Critical GBR Discrete Automation Large Packets (TS 22.261)
    DGBR_ITS                = 84,   ///< Delay-Critical GBR Intelligent Transport Systems (TS 22.261)
    DGBR_ELECTRICITY        = 85,   ///< Delay-Critical GBR Electricity Distribution High Voltage (TS 22.261)
  } qci; ///< Qos class indicator

  GbrQosInformation gbrQosInfo; ///< GBR QOS information
  AllocationRetentionPriority arp; ///< allocation retention priority

  /**
   * Default constructor. QCI will be initialized to NGBR_VIDEO_TCP_DEFAULT
   * 
   */
  EpsBearer ();

  /**
   *
   * @param x the QoS Class Indicator
   *
   */
  EpsBearer (Qci x);

  /**
   *
   * @param x the QoS Class Indicator
   * @param y the GbrQosInformation
   *
   */
  EpsBearer (Qci x, GbrQosInformation y);

  /**
   * \brief EpsBearer copy constructor
   * \param o other instance
   */
  EpsBearer (const EpsBearer &o);

  /**
    * \brief Deconstructor
    */
  virtual ~EpsBearer () { }

  /**
   * \brief SetRelease
   * \param release The release the user want for this bearer
   *
   * Releases introduces new types, and change values for existing ones.
   * While we can't do much for the added type (we must expose them even
   * if the user want to work with older releases) by calling this method
   * we can, at least, select the specific parameters value the bearer returns.
   *
   * For instance, if the user select release 10 (the default) the priority
   * of CONV_VIDEO will be 2. With release 15, such priority will be 20.
   */
  void SetRelease (uint8_t release);

  /**
   * \brief GetRelease
   * \return The release currently set for this bearer type
   */
  uint8_t GetRelease () const { return m_release; }

  /**
   *
   * @return true if the EPS Bearer is a Guaranteed Bit Rate bearer, false otherwise
   */
  bool IsGbr () const;

  /**
   *
   * @return the priority associated with the QCI of this bearer as per 3GPP 23.203 Section 6.1.7.2
   */
  uint8_t GetPriority () const;

  /**
   *
   *
   *
   * @return the packet delay budget associated with the QCI of this bearer as per 3GPP 23.203 Section 6.1.7.2
   */
  uint16_t GetPacketDelayBudgetMs () const;

  /**
   *
   *
   *
   * @return the packet error loss rate associated with the QCI of this bearer as per 3GPP 23.203 Section 6.1.7.2
   */
  double  GetPacketErrorLossRate () const;

private:
  /**
   * \brief Hashing QCI
   *
   * Qci are just uint8_t, so that's how we calculate the hash. Unfortunately,
   * we have to provide this struct because gcc 4.9 would not compile otherwise.
   */
  struct QciHash
  {
    /**
     * \brief Hash the QCI like a normal uint8_t
     * \param s Qci to hash
     * \return Hash of Qci
     */
    std::size_t
    operator () (Qci const& s) const noexcept
      {
        return std::hash<uint8_t> {} (s);
      }
  };

  /**
    * \brief Map between QCI and requirements
    *
    * The tuple is formed by: isGbr, priority, packet delay budget, packet error rate,
    *  default maximum data burst, default averaging window (0 when does not apply)
    */
  typedef std::unordered_map<Qci, std::tuple<bool, uint8_t, uint16_t, double, uint32_t, uint32_t>, QciHash > BearerRequirementsMap;

  /**
   * \brief Is the selected QCI GBR?
   * \param map Map between QCI and requirements
   * \param qci QCI to look for
   * \return GBR flag for the selected CQI
   */
  static uint32_t
  IsGbr (const BearerRequirementsMap &map, Qci qci) {return std::get<0> (map.at (qci));}

  /**
   * \brief Get priority for the selected QCI
   * \param map Map between QCI and requirements
   * \param qci QCI to look for
   * \return priority for the selected QCI
   */
  static uint8_t
  GetPriority (const BearerRequirementsMap &map, Qci qci) {return std::get<1> (map.at (qci));}
  /**
   * \brief Get packet delay in ms for the selected QCI
   * \param map Map between QCI and requirements
   * \param qci QCI to look for
   * \return packet delay in ms for the selected QCI
   */
  static uint16_t
  GetPacketDelayBudgetMs (const BearerRequirementsMap &map, Qci qci) {return std::get<2> (map.at (qci));}
  /**
   * \brief Get packet error rate for the selected QCI
   * \param map Map between QCI and requirements
   * \param qci QCI to look for
   * \return packet error rate for the selected QCI
   */
  static double
  GetPacketErrorLossRate (const BearerRequirementsMap &map, Qci qci) {return std::get<3> (map.at (qci));}
  /**
   * \brief Get maximum data burst for the selected QCI
   * \param map Map between QCI and requirements
   * \param qci QCI to look for
   * \return maximum data burst for the selected QCI
   */
  static uint32_t
  GetMaxDataBurst (const BearerRequirementsMap &map, Qci qci) {return std::get<4> (map.at (qci));}
  /**
   * \brief Get default averaging window for the selected QCI
   * \param map Map between QCI and requirements
   * \param qci QCI to look for
   * \return default averaging window for the selected QCI
   */
  static uint32_t
  GetAvgWindow (const BearerRequirementsMap &map, Qci qci) {return std::get<5> (map.at (qci));}

  /**
   * \brief Retrieve requirements for Rel. 11
   * \return the BearerRequirementsMap for Release 11
   *
   * It returns a pointer to a non-const static data. That is not thread-safe,
   * nor safe to do in general. However, a const-correct version would have
   * to initialize two static maps, and then returning either one or the other.
   * But that's a huge memory increase, and EpsBearer is used everywhere.
   *
   * To be revisited when GCC 4.9 will not be supported anymore.
   */
  static BearerRequirementsMap * GetRequirementsRel11 ();

  /**
   * \brief Retrieve requirements for Rel. 15
   * \return the BearerRequirementsMap for Release 15
   *
   * It returns a pointer to a non-const static data. That is not thread-safe,
   * nor safe to do in general. However, a const-correct version would have
   * to initialize two static maps, and then returning either one or the other.
   * But that's a huge memory increase, and EpsBearer is used everywhere.
   *
   * To be revisited when GCC 4.9 will not be supported anymore.
   */
  static BearerRequirementsMap * GetRequirementsRel15 ();

  /**
    * \brief Requirements pointer per bearer
    *
    * It will point to a static map.
    */
  BearerRequirementsMap *m_requirements;

  uint8_t m_release {30}; //!< Release (10 or 15)
};

} // namespace ns3


#endif // EPS_BEARER
