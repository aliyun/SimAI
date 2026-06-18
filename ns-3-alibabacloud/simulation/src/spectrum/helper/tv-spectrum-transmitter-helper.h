/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 * Author: Benjamin Cizdziel <ben.cizdziel@gmail.com>
 */

#ifndef TV_SPECTRUM_TRANSMITTER_HELPER_H
#define TV_SPECTRUM_TRANSMITTER_HELPER_H

#include <ns3/spectrum-value.h>
#include <ns3/spectrum-phy.h>
#include <ns3/spectrum-channel.h>
#include <ns3/antenna-model.h>
#include <ns3/spectrum-signal-parameters.h>
#include <ns3/mobility-model.h>
#include <ns3/net-device.h>
#include <ns3/net-device-container.h>
#include <ns3/node-container.h>
#include <ns3/non-communicating-net-device.h>
#include <ns3/random-variable-stream.h>
#include <ns3/tv-spectrum-transmitter.h>
#include "ns3/object-factory.h"

class TvHelperDistributionTestCase;

namespace ns3
{

/**
 * \ingroup spectrum
 *
 * Helper class which uses TvSpectrumTransmitter class to create customizable 
 * TV transmitter(s) that transmit PSD spectrum specified by user-set attributes.
 * Has functionality to create TV transmitter(s) with actual frequencies of 
 * specific geographic regions.
 * Provides method to create a random set of transmitters within a given region 
 * and location.
 *
 * Here is an example of how to use this class:
 * \code
   TvSpectrumTransmitterHelper tvTransHelper;
   tvTransHelper.SetChannel (channel); // provided that user has a Ptr<SpectrumChannel> ready.
   tvTransHelper.SetAttribute ("StartFrequency", DoubleValue (524e6));
   tvTransHelper.SetAttribute ("ChannelBandwidth", DoubleValue (6e6));
   tvTransHelper.SetAttribute ("StartingTime", TimeValue (Seconds (0)));
   tvTransHelper.SetAttribute ("TransmitDuration", TimeValue (Seconds (0.2)));
   tvTransHelper.SetAttribute ("BasePsd", DoubleValue (22.22));
   tvTransHelper.SetAttribute ("TvType", EnumValue (TvSpectrumTransmitter::TVTYPE_8VSB));
   tvTransHelper.SetAttribute ("Antenna", StringValue ("ns3::IsotropicAntennaModel"));
   tvTransHelper.Install (tvTransmitterNode); // provided that user has a NodeContainer ready.
   \endcode
 */
class TvSpectrumTransmitterHelper
{

public:
  friend class ::TvHelperDistributionTestCase;

  /**
   * geographical region that TV transmitters are being set up in
   */
  enum Region
  {
    REGION_NORTH_AMERICA,
    REGION_JAPAN,
    REGION_EUROPE
  };

  /**
   * density of location that TV transmitters are being set up in
   */
  enum Density
  {
    DENSITY_LOW,
    DENSITY_MEDIUM,
    DENSITY_HIGH
  };
  
  TvSpectrumTransmitterHelper (); //!< Default constructor
  virtual ~TvSpectrumTransmitterHelper (); //!< Destructor

  /** 
   * Set the spectrum channel for the device(s) to transmit on
   * 
   * @param c a pointer to the spectrum channel
   */
  void SetChannel (Ptr<SpectrumChannel> c);

  /** 
   * Set attribute for each TvSpectrumTransmitter instance to be created 
   * 
   * @param name the name of the attribute to set
   * @param val the value of the attribute to set
   */
  void SetAttribute (std::string name, const AttributeValue &val);

  /** 
   * Set up and start the TV Transmitter's transmission on the spectrum channel.
   * Creates one TV Transmitter for each node given as an input, with settings 
   * selected from attributes.
   * Use SetAttribute() to select settings for the TV Transmitter(s).
   * If multiple transmitters created, all will have same settings (frequency, 
   * PSD, etc.) except for position/location, which depends on the positions 
   * in the input NodeContainer.
   *
   * @param nodes a container containing the node(s) which the TV 
   * transmitter(s) will be attached to
   *
   * @return a device container which contains all the devices created by this 
   * method
   */
  NetDeviceContainer Install (NodeContainer nodes);

  /** 
   * Set up and start the TV Transmitter's transmission on the spectrum channel.
   * Creates one TV Transmitter for each node given as an input, with settings 
   * selected from attributes.
   * Use SetAttribute() to select settings for the TV Transmitter(s).
   * Transmitter(s) will be created with frequencies corresponding to the 
   * user-selected channel number of the user-selected region.
   * If multiple transmitters created, all will have same settings (frequency, 
   * PSD, etc.) except for position/location, which depends on the positions 
   * in the input NodeContainer.
   *
   * @param nodes a container containing the node(s) which the TV 
   * transmitter(s) will be attached to
   * @param region the region of which the transmitter(s) will be characterized
   * @param channelNumber the TV channel number in the selected region that the 
   * transmitter frequencies will be modeled after
   *
   * @return a device container which contains all the devices created by this 
   * method
   */
  NetDeviceContainer Install (NodeContainer nodes,
                              Region region,
                              uint16_t channelNumber);

  /** 
   * Set up and start the TV Transmitter's transmission on the spectrum channel.
   * Consecutively created transmitters (consecutive in input NodeContainer) 
   * will have adjacent channels, i.e. a transmitter's frequency spectrum will 
   * border (but not overlap) the frequency spectrum of the transmitter created 
   * directly before it and the transmitter created directly after it.
   * Creates one TV Transmitter for each node given as an input, with settings 
   * selected from attributes.
   * Use SetAttribute() to select settings for the TV Transmitter(s).
   * If multiple transmitters created, they will have same settings except for 
   * frequency and position/location.
   * For each node, start frequency will be incremented by the channel bandwidth 
   * that is set.
   * For example, if two nodes are given as inputs to InstallAdjacent with 
   * start frequency set to 500 MHz and channel bandwidth set to 6 MHz, the 
   * first node will be a transmitter ranging from 500 MHz to 506 MHz and the 
   * second node will be a transmitter ranging from 506 MHz to 512 MHz.
   *
   * @param nodes a container containing the node(s) which the TV 
   * transmitter(s) will be attached to
   *
   * @return a device container which contains all the devices created by this 
   * method
   */
  NetDeviceContainer InstallAdjacent (NodeContainer nodes);

  /** 
   * Set up and start the TV Transmitter's transmission on the spectrum channel.
   * Consecutively created transmitters (consecutive in input NodeContainer) 
   * will have adjacent channels, with the frequency spectrum and bandwidth of 
   * each channel determined by the user-selected region.
   * Creates one TV Transmitter for each node given as an input, with settings 
   * selected from attributes.
   * Use SetAttribute() to select settings for the TV Transmitter(s).
   * The first created transmitter will have frequencies corresponding to the 
   * user-selected channel number of the user-selected region.
   * Each subsequently created transmitter will have its frequencies modeled 
   * after the channel number (of the user-selected region) following the 
   * previous one; for example if the first created transmitter is modeled after 
   * channel 1, the next one created will be modeled after channel 2, and the 
   * following one will be modeled after channel 3.
   * If multiple transmitters created, they will have same settings except for 
   * frequency and position/location.
   *
   * @param nodes a container containing the node(s) which the TV 
   * transmitter(s) will be attached to
   * @param region the region of which the transmitter(s) will be characterized
   * @param channelNumber the TV channel number in the selected region that the 
   * first created transmitter's frequencies will be modeled after
   *
   * @return a device container which contains all the devices created by this 
   * method
   */
  NetDeviceContainer InstallAdjacent (NodeContainer nodes,
                                      Region region,
                                      uint16_t channelNumber);

  /**
   * Assigns the stream number for the uniform random number generator to use
   *
   * @param streamNum first stream index to use
   * @return the number of stream indices assigned by this helper
   */
  int64_t AssignStreams (int64_t streamNum);

  /**
   * Generates and installs (starts transmission on the spectrum channel) a 
   * random number of TV transmitters with channel frequencies corresponding 
   * to the given region at random locations on or above earth. 
   * The generated transmitters are located at randomly selected points within a 
   * given altitude above earth's surface centered around a given origin point 
   * (on earth's surface) within a given distance radius, corresponding to a 
   * uniform distribution.
   * Distance radius is measured as if all points are on earth's surface
   * (with altitude = 0).
   * Assumes earth is a perfect sphere.
   * The given region's channel numbers that the generated TV transmitters' 
   * frequency spectra are modeled after are uniformly selected at random.
   * These channel numbers are never repeated.
   * The number of transmitters generated is uniformly random based on given 
   * density.
   * Each transmitter has BasePsd and TvType set from SetAttribute(), which 
   * should be set before calling this method.
   * 
   * @param region the region that the transmitters are in
   * @param density the density (high, medium, or low) of the location being 
   * simulated, which determines how many transmitters are created and how many 
   * channels are occupied. Low density will generate between one and one third 
   * of the number of TV channels in the given region, medium density will 
   * generate between one third and two thirds, and high density will generate 
   * between two thirds and all of the channels.
   * @param originLatitude origin point latitude in degrees
   * @param originLongitude origin point longitude in degrees
   * @param maxAltitude maximum altitude in meters above earth's surface with
   * which random points can be generated
   * @param maxRadius max distance in meters from origin with which random 
   * transmitters can be generated (all transmitters are less than or equal to 
   * this distance from the origin, relative to points being on earth's surface)
   */
  void CreateRegionalTvTransmitters (Region region, 
                                     Density density, 
                                     double originLatitude, 
                                     double originLongitude, 
                                     double maxAltitude,
                                     double maxRadius);


private:
  Ptr<SpectrumChannel> m_channel; //!< Pointer to spectrum channel object

  /**
   * Generates random indices of given region frequency array (ignoring indices 
   * referring to invalid channels).
   * Number of indices generated (which is number of TV transmitters to be 
   * created) is random based on given density.
   * Indices generated refer to frequencies that TV transmitters will be 
   * created with.
   * 
   * @param startFrequencies array containing all channel start frequencies for 
   * a particular region
   * @param startFrequenciesLength number of elements in startFrequencies array
   * @param density the density (high, medium, or low) of the location being 
   * simulated, which determines how many transmitters are created
   *
   * @return a list containing the indices in startFrequencies that transmitters 
   * will be created for
   */
  std::list<int> GenerateRegionalTransmitterIndices (const double startFrequencies[], 
                                                     const int startFrequenciesLength, 
                                                     Density density);
  
  /**
   * Randomly generates the number of TV transmitters to be created based on 
   * given density and number of possible TV channels. 
   * Low density will generate a transmitter for between one (a single 
   * transmitter) and one third of the number of possible channels, medium 
   * density will generate a transmitter for between one third and two thirds, 
   * and high density will generate a transmitter for between two thirds and all 
   * of the possible channels. 
   * These ratios are approximated in the implementation, but there is no 
   * overlap possible in the number of transmitters generated between adjacent 
   * densities. 
   * For example, given 60 possible channels, for low density between 1 and 20 
   * transmitters can be created, for medium density between 21 and 40 
   * transmitters can be created, and for high density between 41 and 60 
   * transmitters can be created (all inclusive).
   * 
   * @param density the density (high, medium, or low) of the location being 
   * simulated
   * @param numChannels the number of possible TV channels in the region being 
   * simulated
   *
   * @return the number of TV transmitters that will be created
   */
  int GetRandomNumTransmitters (Density density, uint32_t numChannels);

  /**
   * Installs each randomly generated regional TV transmitter
   * 
   * @param region the region that the transmitters are in
   * @param transmitterIndicesToCreate a list containing the channel number 
   * indices (for region's start frequencies array) that transmitters will be 
   * created for; this is returned from GenerateRegionalTransmitterIndices()
   * @param transmitterLocations a list containing the vectors 
   * (x, y, z location) of each TV transmitter to be generated; this is 
   * returned from RandGeographicCoordsAroundPoint()
   */
  void InstallRandomRegionalTransmitters (Region region, 
                                          std::list<int> transmitterIndicesToCreate, 
                                          std::list<Vector> transmitterLocations);

  ObjectFactory m_factory; //!< Object factory for attribute setting
  Ptr<UniformRandomVariable> m_uniRand; //!< Object to generate uniform random numbers

};

} // namespace ns3

#endif /* TV_SPECTRUM_TRANSMITTER_HELPER_H */
