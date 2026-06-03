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

#include <iostream>
#include <cstring>
#include "ns3/assert.h"
#include "ns3/packet.h"
#include "ns3/fatal-error.h"
#include "ns3/fatal-impl.h"
#include "ns3/header.h"
#include "ns3/buffer.h"
#include "pcap-file.h"
#include "ns3/log.h"
#include "ns3/build-profile.h"
//
// This file is used as part of the ns-3 test framework, so please refrain from
// adding any ns-3 specific constructs such as Packet to this file.
//

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PcapFile");

const uint32_t MAGIC = 0xa1b2c3d4;            /**< Magic number identifying standard pcap file format */
const uint32_t SWAPPED_MAGIC = 0xd4c3b2a1;    /**< Looks this way if byte swapping is required */

const uint32_t NS_MAGIC = 0xa1b23c4d;         /**< Magic number identifying nanosec resolution pcap file format */
const uint32_t NS_SWAPPED_MAGIC = 0x4d3cb2a1; /**< Looks this way if byte swapping is required */

const uint16_t VERSION_MAJOR = 2;             /**< Major version of supported pcap file format */
const uint16_t VERSION_MINOR = 4;             /**< Minor version of supported pcap file format */

PcapFile::PcapFile ()
  : m_file (),
    m_swapMode (false),
    m_nanosecMode (false)
{
  NS_LOG_FUNCTION (this);
  FatalImpl::RegisterStream (&m_file);
}

PcapFile::~PcapFile ()
{
  NS_LOG_FUNCTION (this);
  FatalImpl::UnregisterStream (&m_file);
  Close ();
}


bool
PcapFile::Fail (void) const
{
  NS_LOG_FUNCTION (this);
  return m_file.fail ();
}
bool
PcapFile::Eof (void) const
{
  NS_LOG_FUNCTION (this);
  return m_file.eof ();
}
void
PcapFile::Clear (void)
{
  NS_LOG_FUNCTION (this);
  m_file.clear ();
}


void
PcapFile::Close (void)
{
  NS_LOG_FUNCTION (this);
  m_file.close ();
}

uint32_t
PcapFile::GetMagic (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_magicNumber;
}

uint16_t
PcapFile::GetVersionMajor (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_versionMajor;
}

uint16_t
PcapFile::GetVersionMinor (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_versionMinor;
}

int32_t
PcapFile::GetTimeZoneOffset (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_zone;
}

uint32_t
PcapFile::GetSigFigs (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_sigFigs;
}

uint32_t
PcapFile::GetSnapLen (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_snapLen;
}

uint32_t
PcapFile::GetDataLinkType (void)
{
  NS_LOG_FUNCTION (this);
  return m_fileHeader.m_type;
}

bool
PcapFile::GetSwapMode (void)
{
  NS_LOG_FUNCTION (this);
  return m_swapMode;
}

bool
PcapFile::IsNanoSecMode (void)
{
  NS_LOG_FUNCTION (this);
  return m_nanosecMode;
}

uint8_t
PcapFile::Swap (uint8_t val)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (val));
  return val;
}

uint16_t
PcapFile::Swap (uint16_t val)
{
  NS_LOG_FUNCTION (this << val);
  return ((val >> 8) & 0x00ff) | ((val << 8) & 0xff00);
}

uint32_t
PcapFile::Swap (uint32_t val)
{
  NS_LOG_FUNCTION (this << val);
  return ((val >> 24) & 0x000000ff) | ((val >> 8) & 0x0000ff00) | ((val << 8) & 0x00ff0000) | ((val << 24) & 0xff000000);
}

void
PcapFile::Swap (PcapFileHeader *from, PcapFileHeader *to)
{
  NS_LOG_FUNCTION (this << from << to);
  to->m_magicNumber = Swap (from->m_magicNumber);
  to->m_versionMajor = Swap (from->m_versionMajor);
  to->m_versionMinor = Swap (from->m_versionMinor);
  to->m_zone = Swap (uint32_t (from->m_zone));
  to->m_sigFigs = Swap (from->m_sigFigs);
  to->m_snapLen = Swap (from->m_snapLen);
  to->m_type = Swap (from->m_type);
}

void
PcapFile::Swap (PcapRecordHeader *from, PcapRecordHeader *to)
{
  NS_LOG_FUNCTION (this << from << to);
  to->m_tsSec = Swap (from->m_tsSec);
  to->m_tsUsec = Swap (from->m_tsUsec);
  to->m_inclLen = Swap (from->m_inclLen);
  to->m_origLen = Swap (from->m_origLen);
}

void
PcapFile::WriteFileHeader (void)
{
  NS_LOG_FUNCTION (this);
  //
  // If we're initializing the file, we need to write the pcap file header
  // at the start of the file.
  //
  m_file.seekp (0, std::ios::beg);

  //
  // We have the ability to write out the pcap file header in a foreign endian
  // format, so we need a temp place to swap on the way out.
  //
  PcapFileHeader header;

  //
  // the pointer headerOut selects either the swapped or non-swapped version of
  // the pcap file header.
  //
  PcapFileHeader *headerOut = 0;

  if (m_swapMode == false)
    {
      headerOut = &m_fileHeader;
    }
  else
    {
      Swap (&m_fileHeader, &header);
      headerOut = &header;
    }

  //
  // Watch out for memory alignment differences between machines, so write
  // them all individually.
  //
  m_file.write ((const char *)&headerOut->m_magicNumber, sizeof(headerOut->m_magicNumber));
  m_file.write ((const char *)&headerOut->m_versionMajor, sizeof(headerOut->m_versionMajor));
  m_file.write ((const char *)&headerOut->m_versionMinor, sizeof(headerOut->m_versionMinor));
  m_file.write ((const char *)&headerOut->m_zone, sizeof(headerOut->m_zone));
  m_file.write ((const char *)&headerOut->m_sigFigs, sizeof(headerOut->m_sigFigs));
  m_file.write ((const char *)&headerOut->m_snapLen, sizeof(headerOut->m_snapLen));
  m_file.write ((const char *)&headerOut->m_type, sizeof(headerOut->m_type));
}

void
PcapFile::ReadAndVerifyFileHeader (void)
{
  NS_LOG_FUNCTION (this);
  //
  // Pcap file header is always at the start of the file
  //
  m_file.seekg (0, std::ios::beg);

  //
  // Watch out for memory alignment differences between machines, so read
  // them all individually.
  //
  m_file.read ((char *)&m_fileHeader.m_magicNumber, sizeof(m_fileHeader.m_magicNumber));
  m_file.read ((char *)&m_fileHeader.m_versionMajor, sizeof(m_fileHeader.m_versionMajor));
  m_file.read ((char *)&m_fileHeader.m_versionMinor, sizeof(m_fileHeader.m_versionMinor));
  m_file.read ((char *)&m_fileHeader.m_zone, sizeof(m_fileHeader.m_zone));
  m_file.read ((char *)&m_fileHeader.m_sigFigs, sizeof(m_fileHeader.m_sigFigs));
  m_file.read ((char *)&m_fileHeader.m_snapLen, sizeof(m_fileHeader.m_snapLen));
  m_file.read ((char *)&m_fileHeader.m_type, sizeof(m_fileHeader.m_type));

  if (m_file.fail ())
    {
      return;
    }

  //
  // There are four possible magic numbers that can be there.  Normal and byte
  // swapped versions of the standard magic number, and normal and byte swapped
  // versions of the magic number indicating nanosecond resolution timestamps.
  //
  if (m_fileHeader.m_magicNumber != MAGIC && m_fileHeader.m_magicNumber != SWAPPED_MAGIC &&
      m_fileHeader.m_magicNumber != NS_MAGIC && m_fileHeader.m_magicNumber != NS_SWAPPED_MAGIC)
    {
      m_file.setstate (std::ios::failbit);
    }

  //
  // If the magic number is swapped, then we can assume that everything else we read
  // is swapped.
  //
  m_swapMode = (m_fileHeader.m_magicNumber == SWAPPED_MAGIC
                || m_fileHeader.m_magicNumber == NS_SWAPPED_MAGIC);

  if (m_swapMode)
    {
      Swap (&m_fileHeader, &m_fileHeader);
    }

  //
  // Timestamps can either be microsecond or nanosecond
  //
  m_nanosecMode = ((m_fileHeader.m_magicNumber == NS_MAGIC) ||
                   (m_fileHeader.m_magicNumber == NS_SWAPPED_MAGIC));

  //
  // We only deal with one version of the pcap file format.
  //
  if (m_fileHeader.m_versionMajor != VERSION_MAJOR || m_fileHeader.m_versionMinor != VERSION_MINOR)
    {
      m_file.setstate (std::ios::failbit);
    }

  //
  // A quick test of reasonablness for the time zone offset corresponding to
  // a real place on the planet.
  //
  if (m_fileHeader.m_zone < -12 || m_fileHeader.m_zone > 12)
    {
      m_file.setstate (std::ios::failbit);
    }

  if (m_file.fail ())
    {
      m_file.close ();
    }
}

void
PcapFile::Open (std::string const &filename, std::ios::openmode mode)
{
  NS_LOG_FUNCTION (this << filename << mode);
  NS_ASSERT ((mode & std::ios::app) == 0);
  NS_ASSERT (!m_file.fail ());
  //
  // All pcap files are binary files, so we just do this automatically.
  //
  mode |= std::ios::binary;

  m_filename=filename;
  m_file.open (filename.c_str (), mode);
  if (mode & std::ios::in)
    {
      // will set the fail bit if file header is invalid.
      ReadAndVerifyFileHeader ();
    }
}

void
PcapFile::Init (uint32_t dataLinkType, uint32_t snapLen, int32_t timeZoneCorrection, bool swapMode, bool nanosecMode)
{
  NS_LOG_FUNCTION (this << dataLinkType << snapLen << timeZoneCorrection << swapMode);

  //
  // Initialize the magic number and nanosecond mode flag
  //
  m_nanosecMode = nanosecMode;
  if (nanosecMode)
    {
      m_fileHeader.m_magicNumber = NS_MAGIC;
    }
  else
    {
      m_fileHeader.m_magicNumber = MAGIC;
    }

  //
  // Initialize remainder of the in-memory file header.
  //
  m_fileHeader.m_versionMajor = VERSION_MAJOR;
  m_fileHeader.m_versionMinor = VERSION_MINOR;
  m_fileHeader.m_zone = timeZoneCorrection;
  m_fileHeader.m_sigFigs = 0;
  m_fileHeader.m_snapLen = snapLen;
  m_fileHeader.m_type = dataLinkType;

  //
  // We use pcap files for regression testing.  We do byte-for-byte comparisons
  // in those tests to determine pass or fail.  If we allow big endian systems
  // to write big endian headers, they will end up byte-swapped and the
  // regression tests will fail.  Until we get rid of the regression tests, we
  // have to pick an endianness and stick with it.  The precedent is little
  // endian, so we set swap mode if required to pick little endian.
  //
  // We do want to allow a user or test suite to enable swapmode irrespective
  // of what we decide here, so we allow setting swapmode from formal parameter
  // as well.
  //
  // So, determine the endianness of the running system.
  //
  union {
    uint32_t a;
    uint8_t  b[4];
  } u;

  u.a = 1;
  bool bigEndian = u.b[3];

  //
  // And set swap mode if requested or we are on a big-endian system.
  //
  m_swapMode = swapMode | bigEndian;

  WriteFileHeader ();
}

uint32_t
PcapFile::WritePacketHeader (uint32_t tsSec, uint32_t tsUsec, uint32_t totalLen)
{
  NS_LOG_FUNCTION (this << tsSec << tsUsec << totalLen);
  NS_ASSERT (m_file.good ());

  uint32_t inclLen = totalLen > m_fileHeader.m_snapLen ? m_fileHeader.m_snapLen : totalLen;

  PcapRecordHeader header;
  header.m_tsSec = tsSec;
  header.m_tsUsec = tsUsec;
  header.m_inclLen = inclLen;
  header.m_origLen = totalLen;

  if (m_swapMode)
    {
      Swap (&header, &header);
    }

  //
  // Watch out for memory alignment differences between machines, so write
  // them all individually.
  //
  m_file.write ((const char *)&header.m_tsSec, sizeof(header.m_tsSec));
  m_file.write ((const char *)&header.m_tsUsec, sizeof(header.m_tsUsec));
  m_file.write ((const char *)&header.m_inclLen, sizeof(header.m_inclLen));
  m_file.write ((const char *)&header.m_origLen, sizeof(header.m_origLen));
  NS_BUILD_DEBUG(m_file.flush());
  return inclLen;
}

void
PcapFile::Write (uint32_t tsSec, uint32_t tsUsec, uint8_t const * const data, uint32_t totalLen)
{
  NS_LOG_FUNCTION (this << tsSec << tsUsec << &data << totalLen);
  uint32_t inclLen = WritePacketHeader (tsSec, tsUsec, totalLen);
  m_file.write ((const char *)data, inclLen);
  NS_BUILD_DEBUG(m_file.flush());
}

void
PcapFile::Write (uint32_t tsSec, uint32_t tsUsec, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this << tsSec << tsUsec << p);
  uint32_t inclLen = WritePacketHeader (tsSec, tsUsec, p->GetSize ());
  p->CopyData (&m_file, inclLen);
  NS_BUILD_DEBUG(m_file.flush());
}

void
PcapFile::Write (uint32_t tsSec, uint32_t tsUsec, const Header &header, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this << tsSec << tsUsec << &header << p);
  uint32_t headerSize = header.GetSerializedSize ();
  uint32_t totalSize = headerSize + p->GetSize ();
  uint32_t inclLen = WritePacketHeader (tsSec, tsUsec, totalSize);

  Buffer headerBuffer;
  headerBuffer.AddAtStart (headerSize);
  header.Serialize (headerBuffer.Begin ());
  uint32_t toCopy = std::min (headerSize, inclLen);
  headerBuffer.CopyData (&m_file, toCopy);
  inclLen -= toCopy;
  p->CopyData (&m_file, inclLen);
}

void
PcapFile::Read (
  uint8_t * const data,
  uint32_t maxBytes,
  uint32_t &tsSec,
  uint32_t &tsUsec,
  uint32_t &inclLen,
  uint32_t &origLen,
  uint32_t &readLen)
{
  NS_LOG_FUNCTION (this << &data <<maxBytes << tsSec << tsUsec << inclLen << origLen << readLen);
  NS_ASSERT (m_file.good ());

  PcapRecordHeader header;

  //
  // Watch out for memory alignment differences between machines, so read
  // them all individually.
  //
  m_file.read ((char *)&header.m_tsSec, sizeof(header.m_tsSec));
  m_file.read ((char *)&header.m_tsUsec, sizeof(header.m_tsUsec));
  m_file.read ((char *)&header.m_inclLen, sizeof(header.m_inclLen));
  m_file.read ((char *)&header.m_origLen, sizeof(header.m_origLen));

  if (m_file.fail ())
    {
      return;
    }

  if (m_swapMode)
    {
      Swap (&header, &header);
    }

  tsSec = header.m_tsSec;
  tsUsec = header.m_tsUsec;
  inclLen = header.m_inclLen;
  origLen = header.m_origLen;

  //
  // We don't always want to force the client to keep a maximum length buffer
  // around so we allow her to specify a minimum number of bytes to read.
  // Usually 64 bytes is enough information to print all of the headers, so
  // it isn't typically necessary to read all thousand bytes of an echo packet,
  // for example, to figure out what is going on.
  //
  readLen = maxBytes < header.m_inclLen ? maxBytes : header.m_inclLen;
  m_file.read ((char *)data, readLen);

  //
  // To keep the file pointer pointed in the right place, however, we always
  // need to account for the entire packet as stored originally.
  //
  if (readLen < header.m_inclLen)
    {
      m_file.seekg (header.m_inclLen - readLen, std::ios::cur);
    }
}

bool
PcapFile::Diff (std::string const & f1, std::string const & f2,
                uint32_t & sec, uint32_t & usec, uint32_t & packets,
                uint32_t snapLen)
{
  NS_LOG_FUNCTION (f1 << f2 << sec << usec << snapLen);
  PcapFile pcap1, pcap2;
  pcap1.Open (f1, std::ios::in);
  pcap2.Open (f2, std::ios::in);
  bool bad = pcap1.Fail () || pcap2.Fail ();
  if (bad)
    {
      return true;
    }

  uint8_t *data1 = new uint8_t [snapLen] ();
  uint8_t *data2 = new uint8_t [snapLen] ();
  uint32_t tsSec1 = 0;
  uint32_t tsSec2 = 0;
  uint32_t tsUsec1 = 0;
  uint32_t tsUsec2 = 0;
  uint32_t inclLen1 = 0;
  uint32_t inclLen2 = 0;
  uint32_t origLen1 = 0;
  uint32_t origLen2 = 0;
  uint32_t readLen1 = 0;
  uint32_t readLen2 = 0;
  bool diff = false;

  while (!pcap1.Eof () && !pcap2.Eof ())
    {
      pcap1.Read (data1, snapLen, tsSec1, tsUsec1, inclLen1, origLen1, readLen1);
      pcap2.Read (data2, snapLen, tsSec2, tsUsec2, inclLen2, origLen2, readLen2);

      bool same = pcap1.Fail () == pcap2.Fail ();
      if (!same)
        {
          diff = true;
          break;
        }
      if (pcap1.Eof ())
        {
          break;
        }

      ++packets;

      if (tsSec1 != tsSec2 || tsUsec1 != tsUsec2)
        {
          diff = true; // Next packet timestamps do not match
          break;
        }

      if (readLen1 != readLen2)
        {
          diff = true; // Packet lengths do not match
          break;
        }

      if (std::memcmp (data1, data2, readLen1) != 0)
        {
          diff = true; // Packet data do not match
          break;
        }
    }
  sec = tsSec1;
  usec = tsUsec1;

  bad = pcap1.Fail () || pcap2.Fail ();
  bool eof = pcap1.Eof () && pcap2.Eof ();
  if (bad && !eof)
    {
      diff = true;
    }

  delete[] data1;
  delete[] data2;

  return diff;
}

} // namespace ns3
