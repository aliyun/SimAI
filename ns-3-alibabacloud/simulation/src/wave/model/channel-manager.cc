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
#include "channel-manager.h"
#include "ns3/log.h"
#include "ns3/assert.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ChannelManager");

NS_OBJECT_ENSURE_REGISTERED (ChannelManager);

TypeId
ChannelManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ChannelManager")
    .SetParent<Object> ()
    .SetGroupName ("Wave")
    .AddConstructor<ChannelManager> ()
  ;
  return tid;
}

ChannelManager::ChannelManager ()
{
  NS_LOG_FUNCTION (this);
  m_channels.insert (std::make_pair (CCH, new WaveChannel (CCH)));
  m_channels.insert (std::make_pair (SCH1, new WaveChannel (SCH1)));
  m_channels.insert (std::make_pair (SCH2, new WaveChannel (SCH2)));
  m_channels.insert (std::make_pair (SCH3, new WaveChannel (SCH3)));
  m_channels.insert (std::make_pair (SCH4, new WaveChannel (SCH4)));
  m_channels.insert (std::make_pair (SCH5, new WaveChannel (SCH5)));
  m_channels.insert (std::make_pair (SCH6, new WaveChannel (SCH6)));
}

ChannelManager::~ChannelManager ()
{
  NS_LOG_FUNCTION (this);
}

uint32_t
ChannelManager::GetCch (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return CCH;
}

std::vector<uint32_t>
ChannelManager::GetSchs (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  std::vector<uint32_t> schs;
  schs.push_back (SCH1);
  schs.push_back (SCH2);
  schs.push_back (SCH3);
  schs.push_back (SCH4);
  schs.push_back (SCH5);
  schs.push_back (SCH6);
  return schs;
}

std::vector<uint32_t>
ChannelManager::GetWaveChannels (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  std::vector<uint32_t> channels;
  channels.push_back (CCH);
  channels.push_back (SCH1);
  channels.push_back (SCH2);
  channels.push_back (SCH3);
  channels.push_back (SCH4);
  channels.push_back (SCH5);
  channels.push_back (SCH6);
  return channels;
}

uint32_t
ChannelManager::GetNumberOfWaveChannels (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static  uint32_t NumberOfWaveChannels  = GetWaveChannels ().size ();
  return NumberOfWaveChannels;
}

bool
ChannelManager::IsCch (uint32_t channelNumber)
{
  NS_LOG_FUNCTION_NOARGS ();
  return channelNumber == CCH;
}

bool
ChannelManager::IsSch (uint32_t channelNumber)
{
  NS_LOG_FUNCTION_NOARGS ();
  if (channelNumber < SCH1 || channelNumber > SCH6)
    {
      return false;
    }
  if (channelNumber % 2 == 1)
    {
      return false;
    }
  return (channelNumber != CCH);
}

bool
ChannelManager::IsWaveChannel (uint32_t channelNumber)
{
  NS_LOG_FUNCTION_NOARGS ();
  if (channelNumber < SCH1 || channelNumber > SCH6)
    {
      return false;
    }
  if (channelNumber % 2 == 1)
    {
      return false;
    }
  return true;
}

uint32_t
ChannelManager::GetOperatingClass (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  return m_channels[channelNumber]->operatingClass;
}

bool
ChannelManager::GetManagementAdaptable (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  return m_channels[channelNumber]->adaptable;
}

WifiMode
ChannelManager::GetManagementDataRate (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  return m_channels[channelNumber]->dataRate;
}

WifiPreamble
ChannelManager::GetManagementPreamble (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  return m_channels[channelNumber]->preamble;
}

uint32_t
ChannelManager::GetManagementPowerLevel (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  return m_channels[channelNumber]->txPowerLevel;
}

void
ChannelManager::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  std::map<uint32_t, WaveChannel *> ::iterator i;
  for (i = m_channels.begin (); i != m_channels.end (); ++i)
    {
      delete (i->second);
    }
  m_channels.clear ();
}

} // namespace ns3
