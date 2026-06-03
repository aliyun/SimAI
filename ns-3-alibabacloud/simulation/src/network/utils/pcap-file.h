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
 * 
 * Author:  Craig Dowell (craigdo@ee.washington.edu)
 */

#ifndef PCAP_FILE_H
#define PCAP_FILE_H

#include <string>
#include <fstream>
#include <stdint.h>
#include "ns3/ptr.h"

namespace ns3 {

class Packet;
class Header;


/**
 * \brief A class representing a pcap file
 * 
 * A class representing a pcap file.  This allows easy creation, writing and 
 * reading of files composed of stored packets; which may be viewed using
 * standard tools.
 */
class PcapFile
{
public:
  static const int32_t  ZONE_DEFAULT    = 0;           /**< Time zone offset for current location */
  static const uint32_t SNAPLEN_DEFAULT = 65535;       /**< Default value for maximum octets to save per packet */

public:
  PcapFile ();
  ~PcapFile ();

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
   * similar to the stdc++ io stream classes, but differ in that
   * positions in the file are based on packets not characters.  For example
   * if the file is opened for reading, the file position indicator (seek
   * position) points to the beginning of the first packet in the file, not
   * zero (which would point to the start of the pcap header).
   *
   * Since a pcap file is always a binary file, the file type is automatically 
   * selected as a binary file (fstream::binary is automatically ored with the mode
   * field).
   *
   * \param filename String containing the name of the file.
   *
   * \param mode the access mode for the file.
   */
  void Open (std::string const &filename, std::ios::openmode mode);

  /**
   * Close the underlying file.
   */
  void Close (void);

  /**
   * Initialize the pcap file associated with this object.  This file must have
   * been previously opened with write permissions.
   *
   * \param dataLinkType A data link type as defined in the pcap library.  If
   * you want to make resulting pcap files visible in existing tools, the 
   * data link type must match existing definitions, such as PCAP_ETHERNET,
   * PCAP_PPP, PCAP_80211, etc.  If you are storing different kinds of packet
   * data, such as naked TCP headers, you are at liberty to locally define your
   * own data link types.  According to the pcap-linktype man page, "well-known"
   * pcap linktypes range from 0 to 263.  If you use a large random number for
   * your type, chances are small for a collision.
   *
   * \param snapLen An optional maximum size for packets written to the file.
   * Defaults to 65535.  If packets exceed this length they are truncated.
   *
   * \param timeZoneCorrection An integer describing the offset of your local
   * time zone from UTC/GMT.  For example, Pacific Standard Time in the US is
   * GMT-8, so one would enter -8 for that correction.  Defaults to 0 (UTC).
   *
   * \param swapMode Flag indicating a difference in endianness of the 
   * writing system. Defaults to false.
   *
   * \param nanosecMode Flag indicating the time resolution of the writing
   * system. Default to false.
   *
   * \warning Calling this method on an existing file will result in the loss
   * any existing data.
   */
  void Init (uint32_t dataLinkType, 
             uint32_t snapLen = SNAPLEN_DEFAULT, 
             int32_t timeZoneCorrection = ZONE_DEFAULT,
             bool swapMode = false,
             bool nanosecMode = false);

  /**
   * \brief Write next packet to file
   * 
   * \param tsSec       Packet timestamp, seconds 
   * \param tsUsec      Packet timestamp, microseconds
   * \param data        Data buffer
   * \param totalLen    Total packet length
   * 
   */
  void Write (uint32_t tsSec, uint32_t tsUsec, uint8_t const * const data, uint32_t totalLen);

  /**
   * \brief Write next packet to file
   * 
   * \param tsSec       Packet timestamp, seconds 
   * \param tsUsec      Packet timestamp, microseconds
   * \param p           Packet to write
   * 
   */
  void Write (uint32_t tsSec, uint32_t tsUsec, Ptr<const Packet> p);
  /**
   * \brief Write next packet to file
   * 
   * \param tsSec       Packet timestamp, seconds 
   * \param tsUsec      Packet timestamp, microseconds
   * \param header      Header to write, in front of packet
   * \param p           Packet to write
   * 
   */
  void Write (uint32_t tsSec, uint32_t tsUsec, const Header &header, Ptr<const Packet> p);


  /**
   * \brief Read next packet from file
   * 
   * \param data        [out] Data buffer
   * \param maxBytes    Allocated data buffer size
   * \param tsSec       [out] Packet timestamp, seconds
   * \param tsUsec      [out] Packet timestamp, microseconds
   * \param inclLen     [out] Included length
   * \param origLen     [out] Original length
   * \param readLen     [out] Number of bytes read
   * 
   */
  void Read (uint8_t * const data, 
             uint32_t maxBytes,
             uint32_t &tsSec, 
             uint32_t &tsUsec, 
             uint32_t &inclLen, 
             uint32_t &origLen, 
             uint32_t &readLen);

  /**
   * \brief Get the swap mode of the file.
   *
   * Pcap files use a magic number that is overloaded to identify both the 
   * format of the file itself and the byte ordering of the file.  The magic
   * number (and all data) is written into the file according to the native
   * byte ordering of the writing system.  If a reading application reads
   * the magic number identically (for example 0xa1b2c3d4) then no byte 
   * swapping is required to correctly interpret the file data.  If the reading
   * application sees the magic number is byte swapped (for example 0xd4c3b2a1)
   * then it knows that it needs to byteswap appropriate fields in the format.
   *
   * GetSWapMode returns a value indicating whether or not the fields are being
   * byteswapped.  Used primarily for testing the class itself, but may be 
   * useful as a flag indicating a difference in endianness of the writing 
   * system.
   *
   * \returns swap mode of the file
   */
  bool GetSwapMode (void);

  /**
   * \brief Get the nanosecond mode of the file.
   *
   * \returns true if the packet timestamps in the PCAP
   * file have nanosecond resolution.
   */
   bool IsNanoSecMode (void);
 
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

  /**
   * \brief Compare two PCAP files packet-by-packet
   * 
   * \return true if files are different, false otherwise
   * 
   * \param  f1         First PCAP file name
   * \param  f2         Second PCAP file name
   * \param  sec        [out] Time stamp of first different packet, seconds. Undefined if files doesn't differ.
   * \param  usec       [out] Time stamp of first different packet, microseconds. Undefined if files doesn't differ.
   * \param  packets    [out] Number of first different packet. Total number of parsed packets if files doesn't differ.
   * \param  snapLen    Snap length (if used)
   */
  static bool Diff (std::string const & f1, std::string const & f2, 
                    uint32_t & sec, uint32_t & usec, uint32_t & packets,
                    uint32_t snapLen = SNAPLEN_DEFAULT);

private:
  /**
   * \brief Pcap file header
   */
  typedef struct {
    uint32_t m_magicNumber;   /**< Magic number identifying this as a pcap file */
    uint16_t m_versionMajor;  /**< Major version identifying the version of pcap used in this file */
    uint16_t m_versionMinor;  /**< Minor version identifying the version of pcap used in this file */
    int32_t  m_zone;          /**< Time zone correction to be applied to timestamps of packets */
    uint32_t m_sigFigs;       /**< Unused by pretty much everybody */
    uint32_t m_snapLen;       /**< Maximum length of packet data stored in records */
    uint32_t m_type;          /**< Data link type of packet data */
  } PcapFileHeader;

  /**
   * \brief Pcap record header
   */
  typedef struct {
    uint32_t m_tsSec;         /**< seconds part of timestamp */
    uint32_t m_tsUsec;        /**< microseconds part of timestamp (nsecs for PCAP_NSEC_MAGIC) */
    uint32_t m_inclLen;       /**< number of octets of packet saved in file */
    uint32_t m_origLen;       /**< actual length of original packet */
  } PcapRecordHeader;

  /**
   * \brief Swap a value byte order
   * \param val the value
   * \returns the value with byte order swapped
   */
  uint8_t Swap (uint8_t val);
  /**
   * \brief Swap a value byte order
   * \param val the value
   * \returns the value with byte order swapped
   */
  uint16_t Swap (uint16_t val);
  /**
   * \brief Swap a value byte order
   * \param val the value
   * \returns the value with byte order swapped
   */
  uint32_t Swap (uint32_t val);
  /**
   * \brief Swap the byte order of a Pcap file header
   * \param from original file header
   * \param to swapped file header
   */
  void Swap (PcapFileHeader *from, PcapFileHeader *to);
  /**
   * \brief Swap the byte order of a Pcap record header
   * \param from original record header
   * \param to swapped record header
   */
  void Swap (PcapRecordHeader *from, PcapRecordHeader *to);

  /**
   * \brief Write a Pcap file header
   */
  void WriteFileHeader (void);
  /**
   * \brief Write a Pcap packet header
   *
   * The pcap header has a fixed length of 24 bytes. The last 4 bytes
   * represent the link-layer type
   *
   * \param tsSec Time stamp (seconds part)
   * \param tsUsec Time stamp (microseconds part)
   * \param totalLen total packet length
   * \returns the length of the packet to write in the Pcap file
   */
  uint32_t WritePacketHeader (uint32_t tsSec, uint32_t tsUsec, uint32_t totalLen);

  /**
   * \brief Read and verify a Pcap file header
   */
  void ReadAndVerifyFileHeader (void);

  std::string    m_filename;    //!< file name
  std::fstream   m_file;        //!< file stream
  PcapFileHeader m_fileHeader;  //!< file header
  bool m_swapMode;              //!< swap mode
  bool m_nanosecMode;           //!< nanosecond timestamp mode
};

} // namespace ns3

#endif /* PCAP_FILE_H */
