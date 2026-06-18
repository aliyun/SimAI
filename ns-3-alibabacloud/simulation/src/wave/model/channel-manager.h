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
#ifndef CHANNEL_MANAGER_H
#define CHANNEL_MANAGER_H
#include <map>
#include <vector>
#include "ns3/object.h"
#include "ns3/wifi-mode.h"
#include "ns3/wifi-phy-common.h"

namespace ns3 {

/**
 * \ingroup wave
 *
 * WAVE channels
 * channel number             |   172   |   174   |   176   |   178   |   180   |   182   |   184   |
 * channel bandwidth          10MHz  10MHz  10MHz  10MHz 10MHz  10MHz  10MHz
 * channel name                      SCH1     SCH2     SCH3     CCH       SCH4     SCH5     SCH6
 * another name                     CH172   CH174   CH176   CH178   CH180  CH182   CH184
 *
 * not support
 * channel 175 : combine channel 174 and 176
 * channel 181 : combine channel 180 and 182
 */
#define CH172 172
#define CH174 174
#define CH176 176
#define CH178 178
#define CH180 180
#define CH182 182
#define CH184 184

#define SCH1 172
#define SCH2 174
#define SCH3 176
#define CCH  178
#define SCH4 180
#define SCH5 182
#define SCH6 184

/**
 * \ingroup wave
 * \brief manage 7 WaveChannels and the tx information such as data rate and txPowerLevel.
 * for transmitting VSA management frames.
 */
class ChannelManager : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  ChannelManager ();
  virtual ~ChannelManager ();

  /**
   * \return the channel number of WAVE CCH.
   */
  static uint32_t GetCch (void);
  /**
   * \return the channel number set of WAVE SCHs.
   */
  static std::vector<uint32_t> GetSchs (void);
  /**
    * \return the channel number set of WAVE channels.
    *
    * The sequence is CCH, SCH1, SCH2, SCH3, SCH4, SCH5 and SCH6.
    */
  static std::vector<uint32_t> GetWaveChannels (void);
  /**
   * \return the number of WAVE channels.
   */
  static uint32_t GetNumberOfWaveChannels (void);
  /**
   * \param channelNumber the specific channel
   * \return whether channel is valid CCH channel
   */
  static bool IsCch (uint32_t channelNumber);
  /**
   * \param channelNumber the specific channel
   * \return whether channel is valid SCH channel
   */
  static bool IsSch (uint32_t channelNumber);
  /**
   * \param channelNumber the specific channel
   * \return whether channel is valid WAVE channel
   */
  static bool IsWaveChannel (uint32_t channelNumber);

  /**
   * \param channelNumber the specific channel
   * \return the operating class on this channel
   *
   * the operating class is unused in the simulation
   */
  uint32_t GetOperatingClass (uint32_t channelNumber);
  /**
   * \param channelNumber the specific channel
   * \return the adaptable mode for management frames
   */
  bool GetManagementAdaptable (uint32_t channelNumber);
  /**
   * \param channelNumber the specific channel
   * \return the data rate for management frames
   */
  WifiMode GetManagementDataRate (uint32_t channelNumber);
  /**
   * \param channelNumber the specific channel
   * \return the preamble for management frames
   */
  WifiPreamble GetManagementPreamble (uint32_t channelNumber);
  /**
   * \param channelNumber the specific channel
   * \return the tx power level for management frames
   */
  uint32_t GetManagementPowerLevel (uint32_t channelNumber);

protected:
  virtual void DoDispose (void);

private:
  /// 1609.4-2010 Annex H
  static const uint32_t  DEFAULT_OPERATING_CLASS = 17;

  /// WaveChannel structure
  struct WaveChannel
  {
    uint32_t channelNumber; ///< channel number
    uint32_t operatingClass; ///< operating class
    bool adaptable; ///< adaptable?
    WifiMode dataRate; ///< data rate
    WifiPreamble preamble; ///< preamble
    uint32_t txPowerLevel; ///< transmit power level

    /**
     * initializor
     * \param channel the channel number
     */
    WaveChannel (uint32_t channel)
      : channelNumber (channel),
        operatingClass (DEFAULT_OPERATING_CLASS),
        adaptable (true),
        dataRate (WifiMode ("OfdmRate6MbpsBW10MHz")),
        preamble (WIFI_PREAMBLE_LONG),
        txPowerLevel (4)
    {
    }
  };
  std::map<uint32_t, WaveChannel *> m_channels; ///< list of channels
};

}
#endif /* CHANNEL_MANAGER_H */
