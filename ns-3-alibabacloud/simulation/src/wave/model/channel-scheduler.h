/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef CHANNEL_SCHEDULER_H
#define CHANNEL_SCHEDULER_H

#include <map>
#include "wave-net-device.h"
namespace ns3 {
class WaveNetDevice;

/**
 * \ingroup wave
 * EdcaParameter structure
 */
struct EdcaParameter
{
  uint32_t cwmin; ///< minimum
  uint32_t cwmax; ///< maximum
  uint32_t aifsn; ///< AIFSN
};

/**
 * \ingroup wave
 * EDCA parameters typedef
 */
typedef std::map<AcIndex,EdcaParameter> EdcaParameters;

/**
 * \ingroup wave
 * EDCA parameters iterator typedef
 */
typedef std::map<AcIndex,EdcaParameter>::const_iterator EdcaParametersI;

#define EXTENDED_ALTERNATING 0x00
#define EXTENDED_CONTINUOUS 0xff
/**
 * \ingroup wave
 *
 * \param channelNumber channel number that the SCH service
 * can be made available for communications.
 * \param operationalRateSet OperationalRateSet if present, as specified in IEEE Std 802.11.
 * valid range: 1-127.
 * \param immediateAccess Indicates that the MLME should provide immediate
 * access to the SCH and not wait until the next SCH interval.
 * \param extendedAccess Indicates that the MLME should provide continuous
 * access (during both SCH interval and CCH interval) to the SCH for ExtendedAccess
 * control channel intervals. A value of 255 indicates indefinite access.
 * \param edcaParameters If present, as specified in IEEE Std 802.11.
 */
struct SchInfo
{
  uint32_t channelNumber; ///< channel number
  //OperationalRateSet  operationalRateSet;  // not supported
  bool immediateAccess; ///< immediate access
  uint8_t extendedAccess; ///< extended access
  EdcaParameters edcaParameters; ///< EDCA parameters
  /// Initializer
  SchInfo ()
    : channelNumber (SCH1),
      immediateAccess (false),
      extendedAccess (EXTENDED_ALTERNATING)
  {

  }
  /**
   * Initializer
   * \param channel the channel number
   * \param immediate true if immediate access
   * \param channelAccess
   */
  SchInfo (uint32_t channel, bool immediate, uint32_t channelAccess)
    : channelNumber (channel),
      immediateAccess (immediate),
      extendedAccess (channelAccess)
  {

  }
  /**
   * Initializer
   * \param channel the channel number
   * \param immediate true if immediate access
   * \param channelAccess
   * \param edca the EDCA parameters
   */
  SchInfo (uint32_t channel, bool immediate, uint32_t channelAccess, EdcaParameters edca)
    : channelNumber (channel),
      immediateAccess (immediate),
      extendedAccess (channelAccess),
      edcaParameters (edca)
  {

  }
};

/// ChannelAccess enumeration
enum ChannelAccess
{
  ContinuousAccess,      // continuous access for SCHs
  AlternatingAccess,       //alternating CCH and SCH access
  ExtendedAccess,         // extended access in SCHs
  DefaultCchAccess,     // default continuous CCH access
  NoAccess,                    // no channel access assigned
};

/**
 * \ingroup wave
 * \brief This class will assign channel access for requests from higher layers.
 * The channel access options include (1) continuous access for SCHs, (2) alternating
 * CCH and SCH access,  (3) extended access for SCHs and (4) default continuous CCH
 * access. The options except (4) have an immediate parameter to enable immediate
 * access assignment without the need for waiting.
 * Its sub-class can use different mechanism to assign the channel access.
 */
class ChannelScheduler : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  ChannelScheduler ();
  virtual ~ChannelScheduler ();

  /**
   * \param device enable channel scheduler associated with WaveNetDevice
   */
  virtual void SetWaveNetDevice (Ptr<WaveNetDevice> device);
  /**
   * \return whether CCH channel access is assigned.
   */
  bool IsCchAccessAssigned (void) const;
  /**
   * \return whether SCH channel access is assigned.
   */
  bool IsSchAccessAssigned (void) const;
  /**
   * \param channelNumber the specified channel number
   * \return  whether channel access is assigned for the channel.
   */
  bool IsChannelAccessAssigned (uint32_t channelNumber) const;
  /**
   * \param channelNumber the specified channel number
   * \return whether the continuous access is assigned for the specific channel.
   */
  bool IsContinuousAccessAssigned (uint32_t channelNumber) const;
  /**
   * \param channelNumber the specified channel number
   * \return whether the continuous access is assigned for the specific channel.
   */
  bool IsAlternatingAccessAssigned (uint32_t channelNumber) const;
  /**
   * \param channelNumber the specified channel number
   * \return whether the continuous access is assigned for the specific channel.
   */
  bool IsExtendedAccessAssigned (uint32_t channelNumber) const;
  /**
   * \return whether the continuous access is assigned for CCHl.
   */
  bool IsDefaultCchAccessAssigned (void) const;
  /**
   * \param channelNumber the specified channel number
   * \return  the type of current assigned channel access for the specific channel.
   */
  virtual enum ChannelAccess GetAssignedAccessType (uint32_t channelNumber) const = 0;

  /**
   * \param schInfo the request information for assigning SCH access.
   * \return whether the channel access is assigned successfully.
   *
   * This method is called to assign channel access for sending packets.
   */
  bool StartSch (const SchInfo & schInfo);
  /**
   * \param channelNumber indicating which channel should release
   * the assigned channel access resource.
   * \return true if successful.
   */
  bool StopSch (uint32_t channelNumber);

protected:
  virtual void DoInitialize (void);
  virtual void DoDispose (void);

  /**
     * \param channelNumber the specific channel
     * \param immediate indicate whether channel switch to channel
     * \return whether the channel access is assigned successfully
     *
     * This method will assign alternating access for SCHs and CCH.
     */
  virtual bool AssignAlternatingAccess (uint32_t channelNumber, bool immediate) = 0;
  /**
   * \param channelNumber the specific channel
   * \param immediate indicate whether channel switch to channel
   * \return whether the channel access is assigned successfully
   *
   * This method will assign continuous access CCH.
   */
  virtual bool AssignContinuousAccess (uint32_t channelNumber, bool immediate) = 0;
  /**
   * \param channelNumber the specific channel
   * \param extends extension duration
   * \param immediate indicate whether channel switch to channel
   * \return whether the channel access is assigned successfully
   *
   * This method will assign extended access for SCHs.
   */
  virtual bool AssignExtendedAccess (uint32_t channelNumber, uint32_t extends, bool immediate) = 0;
  /**
   * This method will assign default CCH access for CCH.
   * \return whether the channel access is assigned successfully
   */
  virtual bool AssignDefaultCchAccess (void) = 0;
  /**
   * \param channelNumber indicating for which channel should release
   * the assigned channel access resource.
   * \return whether the channel access is released successfully
   */
  virtual bool ReleaseAccess (uint32_t channelNumber) = 0;

  Ptr<WaveNetDevice> m_device; ///< the device
};

}
#endif /* CHANNEL_SCHEDULER_H */
