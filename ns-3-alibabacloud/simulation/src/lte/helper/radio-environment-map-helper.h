/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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


#ifndef RADIO_ENVIRONMENT_MAP_HELPER_H
#define RADIO_ENVIRONMENT_MAP_HELPER_H


#include <ns3/object.h>
#include <fstream>


namespace ns3 {

class RemSpectrumPhy;
class Node;
class NetDevice;
class SpectrumChannel;
//class BuildingsMobilityModel;
class MobilityModel;

/** 
 * \ingroup lte
 *
 * Generates a 2D map of the SINR from the strongest transmitter in the
 * downlink of an LTE FDD system. For instructions on usage, please refer to
 * the User Documentation.
 */
class RadioEnvironmentMapHelper : public Object
{
public:  

  RadioEnvironmentMapHelper ();
  virtual ~RadioEnvironmentMapHelper ();
  
  // inherited from Object
  virtual void DoDispose (void);
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /** 
   * \return the bandwidth (in num of RBs) over which SINR is calculated
   */
  uint16_t GetBandwidth () const;

  /** 
   *
   * \param bw  the bandwidth (in num of RBs) over which SINR is calculated
   */
  void SetBandwidth (uint16_t bw);

  /** 
   * Deploy the RemSpectrumPhy objects that generate the map according to the specified settings.
   * 
   */
  void Install ();

private:

  /**
   * Scheduled by Install() to perform the actual generation of map.
   *
   * If control channel is used for SINR calculation (the default), the delay
   * is 2.6 milliseconds from the start of simulation. Otherwise, if data
   * channel is used, the delay is 500.1 milliseconds from the start of
   * simulation.
   *
   * The method will divide the whole map into parts (each contains at most a
   * certain number of SINR listening points), and then call RunOneIteration()
   * on each part, one by one.
   */
  void DelayedInstall ();

  /**
   * Mobilize all the listeners to a specified area. Afterwards, schedule a
   * call to PrintAndReset() in 0.5 milliseconds.
   *
   * \param xMin X coordinate of the first SINR listening point to deploy.
   * \param xMax X coordinate of the last SINR listening point to deploy.
   * \param yMin Y coordinate of the first SINR listening point to deploy.
   * \param yMax Y coordinate of the last SINR listening point to deploy.
   */
  void RunOneIteration (double xMin, double xMax, double yMin, double yMax);

  /// Go through every listener, write the computed SINR, and then reset it.
  void PrintAndReset ();

  /// Called when the map generation procedure has been completed.
  void Finalize ();

  /// A complete Radio Environment Map is composed of many of this structure.
  struct RemPoint 
  {
    /// Simplified listener which compute SINR over the DL channel.
    Ptr<RemSpectrumPhy> phy;
    /// Position of the listener in the environment.
    Ptr<MobilityModel> bmm;
  };

  /// List of listeners in the environment.
  std::list<RemPoint> m_rem;

  double m_xMin;   ///< The `XMin` attribute.
  double m_xMax;   ///< The `XMax` attribute.
  uint16_t m_xRes; ///< The `XRes` attribute.
  double m_xStep;  ///< Distance along X axis between adjacent listening points.

  double m_yMin;   ///< The `YMin` attribute.
  double m_yMax;   ///< The `YMax` attribute.
  uint16_t m_yRes; ///< The `YRes` attribute.
  double m_yStep;  ///< Distance along Y axis between adjacent listening points.

  uint32_t m_maxPointsPerIteration;  ///< The `MaxPointsPerIteration` attribute.

  uint16_t m_earfcn;     ///< The `Earfcn` attribute.
  uint16_t m_bandwidth;  ///< The `Bandwidth` attribute.
 
  double m_z;  ///< The `Z` attribute.

  /**
   * The `ChannelPath` attribute. If `Channel` attribute is not set, then
   * `ChannelPath` will be used to determine the DL channel object for which
   * the REM will be created.
   */
  std::string m_channelPath;

  std::string m_outputFile;   ///< The `OutputFile` attribute.

  bool m_stopWhenDone;   ///< The `StopWhenDone` attribute.
  
  /**
   * The `Channel` attribute, which is a direct pointer to the DL channel
   * object for which will be created the REM. Alternatively, `ChannelPath`
   * attribute can be used. If `ChannelPath` attribute is being used then the
   * m_channel object is configured by using the `ChannelPath` attribute value.
   */
  Ptr<SpectrumChannel> m_channel;

  double m_noisePower;  ///< The `NoisePower` attribute.

  std::ofstream m_outFile;  ///< Stream the output to a file.

  bool m_useDataChannel;  ///< The `UseDataChannel` attribute.
  int32_t m_rbId;         ///< The `RbId` attribute.

}; // end of `class RadioEnvironmentMapHelper`


} // end of `namespace ns3`

#endif /* RADIO_ENVIRONMENT_MAP_HELPER_H */
