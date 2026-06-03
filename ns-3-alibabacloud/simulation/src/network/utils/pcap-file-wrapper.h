/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#ifndef PCAP_FILE_WRAPPER_H
#define PCAP_FILE_WRAPPER_H

#include <cstring>
#include <limits>
#include <fstream>
#include "ns3/ptr.h"
#include "ns3/packet.h"
#include "ns3/object.h"
#include "ns3/nstime.h"
#include "pcap-file.h"

namespace ns3 {

/**
 * A class that wraps a PcapFile as an ns3::Object and provides a higher-layer
 * ns-3 interface to the low-level public methods of PcapFile.  Users are
 * encouraged to use this object instead of class ns3::PcapFile in ns-3
 * public APIs.
 */
class PcapFileWrapper : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  PcapFileWrapper ();
  ~PcapFileWrapper ();


  /**
   * \return true if the 'fail' bit is set in the underlying iostream, false otherwise.
   */
  bool Fail (void) const;
  /**
   * \return true if the 'eof' bit is set in the underlying iostream, false otherwise.
   */
  bool Eof (void) const;
  /**
   * Clear all state bits of the underlying iostream.
   */
  void Clear (void);

  /**
   * Create a new pcap file or open an existing pcap file.  Semantics are
   * similar to the stdc++ io stream classes.
   *
   * Since a pcap file is always a binary file, the file type is automatically 
   * selected as a binary file (fstream::binary is automatically ored with the mode
   * field).
   *
   * \param filename String containing the name of the file.
   *
   * \param mode String containing the access mode for the file.
   *
   */
  void Open (std::string const &filename, std::ios::openmode mode);

  /**
   * Close the underlying pcap file.
   */
  void Close (void);

  /**
   * Initialize the pcap file associated with this wrapper.  This file must have
   * been previously opened with write permissions.
   *
   * \param dataLinkType A data link type as defined in the pcap library.  If
   * you want to make resulting pcap files visible in existing tools, the 
   * data link type must match existing definitions, such as PCAP_ETHERNET,
   * PCAP_PPP, PCAP_80211, etc.  If you are storing different kinds of packet
   * data, such as naked TCP headers, you are at liberty to locally define your
   * own data link types.  According to the pcap-linktype man page, "well-known"
   * pcap linktypes range from 0 to 177.  If you use a large random number for
   * your type, chances are small for a collision.
   *
   * \param snapLen An optional maximum size for packets written to the file.
   * Defaults to 65535.  If packets exceed this length they are truncated.
   *
   * \param tzCorrection An integer describing the offset of your local
   * time zone from UTC/GMT.  For example, Pacific Standard Time in the US is
   * GMT-8, so one would enter -8 for that correction.  Defaults to 0 (UTC).
   *
   * \warning Calling this method on an existing file will result in the loss
   * any existing data.
   */
  void Init (uint32_t dataLinkType, 
             uint32_t snapLen = std::numeric_limits<uint32_t>::max (), 
             int32_t tzCorrection = PcapFile::ZONE_DEFAULT);

  /**
   * \brief Write the next packet to file
   * 
   * \param t Packet timestamp as ns3::Time.
   * \param p Packet to write to the pcap file.
   * 
   */
  void Write (Time t, Ptr<const Packet> p);

  /**
   * \brief Write the provided header along with the packet to the pcap file.
   *
   * It is the case that adding a header to a packet prior to writing it to a
   * file must trigger a deep copy in the Packet.  By providing the header
   * separately, we can avoid that copy.
   * 
   * \param t Packet timestamp as ns3::Time.
   * \param header The Header to prepend to the packet.
   * \param p Packet to write to the pcap file.
   * 
   */
  void Write (Time t, const Header &header, Ptr<const Packet> p);

  /**
   * \brief Write the provided data buffer to the pcap file.
   *
   * \param t Packet timestamp as ns3::Time.
   * \param buffer The buffer to write.
   * \param length The size of the buffer.
   * 
   */
  void Write (Time t, uint8_t const *buffer, uint32_t length);

  /**
   * \brief Read the next packet from the file.
   * 
   * \param t Reference to packet timestamp as ns3::Time.
   * \returns a pointer to ns3::Packet.
   */
  Ptr<Packet> Read (Time &t);

/**
   * \brief Returns the magic number of the pcap file as defined by the magic_number
   * field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns magic number
   */ 
  uint32_t GetMagic (void);

  /**
   * \brief Returns the major version of the pcap file as defined by the version_major
   * field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns major version
   */ 
  uint16_t GetVersionMajor (void);

  /**
   * \brief Returns the minor version of the pcap file as defined by the version_minor
   * field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns minor version
   */ 
  uint16_t GetVersionMinor (void);

  /**
   * \brief Returns the time zone offset of the pcap file as defined by the thiszone
   * field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns time zone offset
   */ 
  int32_t GetTimeZoneOffset (void);

  /**
   * \brief Returns the accuracy of timestamps field of the pcap file as defined
   * by the sigfigs field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns accuracy of timestamps
   */ 
  uint32_t GetSigFigs (void);

  /**
   * \brief Returns the max length of saved packets field of the pcap file as 
   * defined by the snaplen field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns max length of saved packets field
   */ 
  uint32_t GetSnapLen (void);

  /**
   * \brief Returns the data link type field of the pcap file as defined by the 
   * network field in the pcap global header.
   *
   * See http://wiki.wireshark.org/Development/LibpcapFileFormat
   *
   * \returns data link type field
   */ 
  uint32_t GetDataLinkType (void);

private:
  PcapFile m_file; //!< Pcap file
  uint32_t m_snapLen; //!< max length of saved packets
  bool     m_nanosecMode; //!< Timestamps in nanosecond mode
};

} // namespace ns3

#endif /* PCAP_FILE_WRAPPER_H */
