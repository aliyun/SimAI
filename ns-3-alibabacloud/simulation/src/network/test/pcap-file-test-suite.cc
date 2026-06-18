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
 * Author:  Craig Dowell (craigdo@ee.washington.edu)
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cstring>

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/pcap-file.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("pcap-file-test-suite");

// ===========================================================================
// Some utility functions for the tests.
// ===========================================================================

static uint16_t
Swap (uint16_t val)
{
  return ((val >> 8) & 0x00ff) | ((val << 8) & 0xff00);
}

static uint32_t 
Swap (uint32_t val)
{
  return ((val >> 24) & 0x000000ff) | ((val >> 8) & 0x0000ff00) | ((val << 8) & 0x00ff0000) | ((val << 24) & 0xff000000);
}

static bool
CheckFileExists (std::string filename)
{
  FILE * p = std::fopen (filename.c_str (), "rb");
  if (p == 0)
    {
      return false;
    }

  std::fclose (p);
  return true;
}


static bool
CheckFileLength (std::string filename, uint64_t sizeExpected)
{
  FILE * p = std::fopen (filename.c_str (), "rb");
  if (p == 0)
    {
      return false;
    }

  std::fseek (p, 0, SEEK_END);

  uint64_t sizeActual = std::ftell (p);
  std::fclose (p);

  return sizeActual == sizeExpected;
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test case to make sure that the Pcap File Object can do its
 * most basic job and create an empty pcap file.
 */
class WriteModeCreateTestCase : public TestCase
{
public:
  WriteModeCreateTestCase ();
  virtual ~WriteModeCreateTestCase ();

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  std::string m_testFilename; //!< File name
};

WriteModeCreateTestCase::WriteModeCreateTestCase ()
  : TestCase ("Check to see that PcapFile::Open with mode std::ios::out works")
{
}

WriteModeCreateTestCase::~WriteModeCreateTestCase ()
{
}

void
WriteModeCreateTestCase::DoSetup (void)
{
  std::stringstream filename;
  uint32_t n = rand ();
  filename << n;
  m_testFilename = CreateTempDirFilename (filename.str () + ".pcap");
}

void
WriteModeCreateTestCase::DoTeardown (void)
{
  if (remove (m_testFilename.c_str ()))
    {
      NS_LOG_ERROR ("Failed to delete file " << m_testFilename);
    }
}

void
WriteModeCreateTestCase::DoRun (void)
{
  PcapFile f;

  //
  // Opening a new file in write mode should result in an empty file of the
  // given name.
  //
  f.Open (m_testFilename, std::ios::out);

  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << ", \"w\") returns error");
  f.Close ();

  NS_TEST_ASSERT_MSG_EQ (CheckFileExists (m_testFilename), true, 
                         "Open (" << m_testFilename << ", \"std::ios::out\") does not create file");
  NS_TEST_ASSERT_MSG_EQ (CheckFileLength (m_testFilename, 0), true,
                         "Open (" << m_testFilename << ", \"std::ios::out\") does not result in an empty file");

  //
  // Calling Init() on a file created with "std::ios::out" should result in a file just 
  // long enough to contain the pcap file header.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  f.Init (1234, 5678, 7);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (1234, 5678, 7) returns error");

  f.Close ();

  NS_TEST_ASSERT_MSG_EQ (CheckFileLength (m_testFilename, 24), true, 
                         "Init () does not result in a file with a pcap file header");

  //
  // Opening an existing file in write mode should result in that file being
  // emptied.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  f.Close ();

  NS_TEST_ASSERT_MSG_EQ (CheckFileLength (m_testFilename, 0), true, 
                         "Open (" << m_testFilename << ", \"w\") does not result in an empty file");

  //
  // Initialize the file again.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, 
                         "Open (" << m_testFilename << ", \"w\") returns error");

  f.Init (1234, 5678, 7);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (1234, 5678, 7) returns error");

  //
  // Now we should be able to write to it since it was opened in std::ios::out mode.
  // This is just a permissions check so we don't actually look at the 
  // data.
  //
  uint8_t buffer[128];
  memset (buffer, 0, sizeof(buffer));
  f.Write (0, 0, buffer, 128);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Write (write-only-file " << m_testFilename << 
                         ") returns error");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test case to make sure that the Pcap File Object can open an
 * existing pcap file.
 */
class ReadModeCreateTestCase : public TestCase
{
public:
  ReadModeCreateTestCase ();
  virtual ~ReadModeCreateTestCase ();

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  std::string m_testFilename; //!< File name
};

ReadModeCreateTestCase::ReadModeCreateTestCase ()
  : TestCase ("Check to see that PcapFile::Open with mode std::ios::in works")
{
}

ReadModeCreateTestCase::~ReadModeCreateTestCase ()
{
}

void
ReadModeCreateTestCase::DoSetup (void)
{
  std::stringstream filename;
  uint32_t n = rand ();
  filename << n;
  m_testFilename = CreateTempDirFilename (filename.str () + ".pcap");
}

void
ReadModeCreateTestCase::DoTeardown (void)
{
  if (remove (m_testFilename.c_str ()))
    {
      NS_LOG_ERROR ("Failed to delete file " << m_testFilename);
    }
}

void
ReadModeCreateTestCase::DoRun (void)
{
  PcapFile f;

  //
  // Opening a non-existing file in read mode should result in an error.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), true, "Open (non-existing-filename " << m_testFilename << 
                         ", \"std::ios::in\") does not return error");
  f.Close ();
  f.Clear ();
  NS_TEST_ASSERT_MSG_EQ (CheckFileExists (m_testFilename), false, "Open (" << m_testFilename << 
                         ", \"std::ios::in\") unexpectedly created a file");

  //
  // Okay, now create an uninitialized file using previously tested operations
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (filename, \"std::ios::out\") returns error");
  f.Close ();

  //
  // Opening this file should result in an error since it has no pcap file header.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), true, "Open (non-initialized-filename " << m_testFilename << 
                         ", \"std::ios::in\") does not return error");
  f.Close ();
  f.Clear ();

  //
  // Okay, now open that non-initialized file in write mode and initialize it
  // Note that we open it in write mode to initialize it.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  f.Init (1234, 5678, 7);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (1234, 5678, 7) returns error");
  f.Close ();

  //
  // Opening this file should now work since it has a pcap file header.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (initialized-filename " << m_testFilename << 
                         ", \"std::ios::in\") returns error");

  //
  // Now we should not be able to write to it since it was opened in "r" mode
  // even if it has been initialized..
  //
  uint8_t buffer[128];
  f.Write (0, 0, buffer, 128);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), true, "Write (read-only-file " << m_testFilename << 
                         ") does not return error");
  f.Close ();
  f.Clear ();
}

#if 0
// ===========================================================================
// Test case to make sure that the Pcap File Object can open an existing pcap
// file for appending.
// ===========================================================================
class AppendModeCreateTestCase : public TestCase
{
public:
  AppendModeCreateTestCase ();
  virtual ~AppendModeCreateTestCase ();

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  std::string m_testFilename;
};

AppendModeCreateTestCase::AppendModeCreateTestCase ()
  : TestCase ("Check to see that PcapFile::Open with mode std::ios::app works")
{
}

AppendModeCreateTestCase::~AppendModeCreateTestCase ()
{
}

void
AppendModeCreateTestCase::DoSetup (void)
{
  std::stringstream filename;
  uint32_t n = rand ();
  filename << n;
  m_testFilename = CreateTempDirFilename (filename.str () + ".pcap");
}

void
AppendModeCreateTestCase::DoTeardown (void)
{
  if (remove (m_testFilename.c_str ()))
    {
      NS_LOG_ERROR ("Failed to delete file " << m_testFilename);
    }
}

void
AppendModeCreateTestCase::DoRun (void)
{
  PcapFile f;

  //
  // Opening a non-existing file in append mode should result in an error.
  //
  f.Open (m_testFilename, std::ios::out | std::ios::app);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), true, "Open (non-existing-filename " << m_testFilename << 
                         ", \"std::ios::app\") does not return error");
  f.Close ();
  f.Clear ();

  NS_TEST_ASSERT_MSG_EQ (CheckFileExists (m_testFilename), false, 
                         "Open (" << m_testFilename << ", \"std::ios::app\") unexpectedly created a file");

  //
  // Okay, now create an uninitialized file using previously tested operations
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");
  f.Close ();

  //
  // Opening this file should result in an error since it has no pcap file header.
  //
  f.Open (m_testFilename, std::ios::out | std::ios::app);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), true, "Open (non-initialized-filename " << m_testFilename << 
                         ", \"std::ios::app\") does not return error");
  f.Close ();
  f.Clear ();

  //
  // Okay, now open that non-initialized file in write mode and initialize it.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (non-initialized-filename " << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  f.Init (1234, 5678, 7);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (1234, 5678, 7) returns error");
  f.Close ();

  //
  // Opening this file should now work since it has a pcap file header.
  //
  f.Open (m_testFilename, std::ios::out | std::ios::app);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (initialized-filename " << m_testFilename << 
                         ", \"std::ios::app\") returns error");

  //
  // We should be able to write to it since it was opened in "std::ios::app" mode.
  //
  uint8_t buffer[128];
  memset (buffer, 0, sizeof(buffer));
  f.Write (0, 0, buffer, 128);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Write (append-mode-file " << m_testFilename << ") returns error");

  f.Close ();
}
#endif

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test case to make sure that the Pcap File Object can
 * write out correct pcap file headers in both endian cases,
 * and then read them in correctly.
 */
class FileHeaderTestCase : public TestCase
{
public:
  FileHeaderTestCase ();
  virtual ~FileHeaderTestCase ();

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  std::string m_testFilename; //!< File name
};

FileHeaderTestCase::FileHeaderTestCase ()
  : TestCase ("Check to see that PcapFileHeader is managed correctly")
{
}

FileHeaderTestCase::~FileHeaderTestCase ()
{
}

void
FileHeaderTestCase::DoSetup (void)
{
  std::stringstream filename;
  uint32_t n = rand ();
  filename << n;
  m_testFilename = CreateTempDirFilename (filename.str () + ".pcap");
}

void
FileHeaderTestCase::DoTeardown (void)
{
  if (remove (m_testFilename.c_str ()))
    {
      NS_LOG_ERROR ("Failed to delete file " << m_testFilename);
    }
}

void
FileHeaderTestCase::DoRun (void)
{
  PcapFile f;

  //
  // Create an uninitialized file using previously tested operations
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  //
  // Initialize the pcap file header.
  //
  f.Init (1234, 5678, 7);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, 
                         "Init (1234, 5678, 7) returns error");
  f.Close ();

  //
  // Take a look and see what was done to the file
  //
  FILE *p = std::fopen (m_testFilename.c_str (), "r+b");
  NS_TEST_ASSERT_MSG_NE (p, 0, "fopen(" << m_testFilename << ") should have been able to open a correctly created pcap file");

  uint32_t val32;
  uint16_t val16;

  //
  // Because the regression tests require that pcap file output be compared 
  // byte-by-byte, we had to decide on a single format for written pcap files.
  // This was little endian.  So we have to do something special with big-
  // endian machines here.
  //
  // When a big endian machine writes a pcap file, it is forced into swap
  // mode and actually writes little endian files.  This is automagically
  // fixed up when using a PcapFile to read the values, but when a big-
  // endian machine reads these values directly, they will be swapped.
  //
  // We can remove this nonsense when we get rid of the pcap-file-comparison
  // regression tests.
  //
  // So, determine the endian-ness of the running system, and if we're on
  // a big-endian machine, swap all of the results below before checking.
  //
  union {
    uint32_t a;
    uint8_t  b[4];
  } u;

  u.a = 1;
  bool bigEndian = u.b[3];

  size_t result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() magic number");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 0xa1b2c3d4, "Magic number written incorrectly");

  result = std::fread (&val16, sizeof(val16), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() version major");
  if (bigEndian) val16 = Swap (val16);
  NS_TEST_ASSERT_MSG_EQ (val16, 2, "Version major written incorrectly");

  result = std::fread (&val16, sizeof(val16), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() version minor");
  if (bigEndian) val16 = Swap (val16);
  NS_TEST_ASSERT_MSG_EQ (val16, 4, "Version minor written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() time zone correction");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 7, "Version minor written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() sig figs");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 0, "Sig figs written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() snap length");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 5678, "Snap length written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() data link type");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 1234, "Data length type written incorrectly");

  std::fclose (p);
  p = 0;

  //
  // We wrote a little-endian file out correctly, now let's see if we can read
  // it back in correctly.
  //
  // As mentioned above, when a big endian machine writes a pcap file, it is 
  // forced into swap mode and actually writes little endian files.  This is 
  // automagically fixed up when using a PcapFile to read the values, so we 
  // don't have to do anything special here.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (existing-initialized-file " << m_testFilename << 
                         ", \"std::ios::in\") returns error");

  NS_TEST_ASSERT_MSG_EQ (f.GetMagic (), 0xa1b2c3d4, "Read back magic number incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetVersionMajor (), 2, "Read back version major incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetVersionMinor (), 4, "Read back version minor incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetTimeZoneOffset (), 7, "Read back time zone offset incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetSigFigs (), 0, "Read back sig figs incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetSnapLen (), 5678, "Read back snap len incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetDataLinkType (), 1234, "Read back data link type incorrectly");
  f.Close ();

  //
  // Re-open the file to erase its contents.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  //
  // Initialize the pcap file header, turning on swap mode manually to force
  // the pcap file header to be written out in foreign-endian form, whichever
  // endian-ness that might be.  Since big-endian machines are automatically 
  // forced into swap mode, the <true> parameter to f.Init() below is actually
  // a no-op and we're always writing foreign-endian files.  In that case, 
  // this test case is really just a duplicate of the previous.
  //
  f.Init (1234, 5678, 7, true);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (1234, 5678, 7) returns error");
  f.Close ();

  //
  // Take a look and see what was done to the file.  Everything should now
  // appear byte-swapped.
  //
  p = std::fopen (m_testFilename.c_str (), "r+b");
  NS_TEST_ASSERT_MSG_NE (p, 0, "fopen(" << m_testFilename << ") should have been able to open a correctly created pcap file");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() magic number");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (0xa1b2c3d4)), "Magic number written incorrectly");

  result = std::fread (&val16, sizeof(val16), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() version major");
  NS_TEST_ASSERT_MSG_EQ (val16, Swap (uint16_t (2)), "Version major written incorrectly");

  result = std::fread (&val16, sizeof(val16), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() version minor");
  NS_TEST_ASSERT_MSG_EQ (val16, Swap (uint16_t (4)), "Version minor written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() time zone correction");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (7)), "Version minor written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() sig figs");
  NS_TEST_ASSERT_MSG_EQ (val32, 0, "Sig figs written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() snap length");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (5678)), "Snap length written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() data link type");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (1234)), "Data length type written incorrectly");

  std::fclose (p);
  p = 0;

  //
  // We wrote an opposite-endian file out correctly, now let's see if we can read
  // it back in correctly.  Again, in the case of a big-endian machine, we already
  // did this test and it is just a duplicate.  What we don't test on a big endian
  // machine is writing out a big-endian file by default, but we can't do that 
  // since it breaks regression testing.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (existing-initialized-file " << m_testFilename << 
                         ", \"std::ios::in\") returns error");

  NS_TEST_ASSERT_MSG_EQ (f.GetSwapMode (), true, "Byte-swapped file not correctly indicated");

  NS_TEST_ASSERT_MSG_EQ (f.GetMagic (), 0xa1b2c3d4, "Read back magic number incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetVersionMajor (), 2, "Read back version major incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetVersionMinor (), 4, "Read back version minor incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetTimeZoneOffset (), 7, "Read back time zone offset incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetSigFigs (), 0, "Read back sig figs incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetSnapLen (), 5678, "Read back snap len incorrectly");
  NS_TEST_ASSERT_MSG_EQ (f.GetDataLinkType (), 1234, "Read back data link type incorrectly");

  f.Close ();
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test case to make sure that the Pcap File Object can
 * write pcap packet records in both endian cases, and then read
 * them in correctly.
 */
class RecordHeaderTestCase : public TestCase
{
public:
  RecordHeaderTestCase ();
  virtual ~RecordHeaderTestCase ();

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  std::string m_testFilename; //!< File name
};

RecordHeaderTestCase::RecordHeaderTestCase ()
  : TestCase ("Check to see that PcapRecordHeader is managed correctly")
{
}

RecordHeaderTestCase::~RecordHeaderTestCase ()
{
}

void
RecordHeaderTestCase::DoSetup (void)
{
  std::stringstream filename;
  uint32_t n = rand ();
  filename << n;
  m_testFilename = CreateTempDirFilename (filename.str () + ".pcap");
}

void
RecordHeaderTestCase::DoTeardown (void)
{
  if (remove (m_testFilename.c_str ()))
    {
      NS_LOG_ERROR ("Failed to delete file " << m_testFilename);
    }
}

void
RecordHeaderTestCase::DoRun (void)
{
  PcapFile f;

  //
  // Create an uninitialized file using previously tested operations
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  //
  // Initialize the pcap file header.
  //
  f.Init (37, 43, -7);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (37, 43, -7) returns error");

  //
  // Initialize a buffer with a counting pattern to check the data later.
  //
  uint8_t bufferOut[128];
  for (uint32_t i = 0; i < 128; ++i)
    {
      bufferOut[i] = i;
    }

  //
  // Now we should be able to write a packet to it since it was opened in "w" 
  // mode.  The packet data written should be limited to 43 bytes in length 
  // by the Init() call above.
  //
  f.Write (1234, 5678, bufferOut, 128);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Write (write-only-file " << m_testFilename << ") returns error");
  f.Close ();

  //
  // Let's peek into the file and see what actually went out for that
  // packet.
  //
  FILE *p = std::fopen (m_testFilename.c_str (), "r+b");
  NS_TEST_ASSERT_MSG_NE (p, 0, "fopen() should have been able to open a correctly created pcap file");

  //
  // A pcap file header takes up 24 bytes, a pcap record header takes up 16 bytes
  // and we wrote in 43 bytes, so the file must be 83 bytes long.  Let's just
  // double check that this is exactly what happened.
  //
  std::fseek (p, 0, SEEK_END);
  uint64_t size = std::ftell (p);
  NS_TEST_ASSERT_MSG_EQ (size, 83, "Pcap file with one 43 byte packet is incorrect size");

  //
  // A pcap file header takes up 24 bytes, so we should see a pcap record header
  // starting there in the file.  We've tested this all before so we just assume
  // it's all right and just seek to just past that point..
  //
  std::fseek (p, 24, SEEK_SET);

  uint32_t val32;

  //
  // Because the regression tests require that pcap file output be compared 
  // byte-by-byte, we had to decide on a single format for written pcap files.
  // This was little endian.  So we have to do something special with big-
  // endian machines here.
  //
  // When a big endian machine writes a pcap file, it is forced into swap
  // mode and actually writes little endian files.  This is automagically
  // fixed up when using a PcapFile to read the values, but when a big-
  // endian machine reads these values directly, they will be swapped.
  //
  // We can remove this nonsense when we get rid of the pcap-file-comparison
  // regression tests.
  //
  // So, determine the endian-ness of the running system, and if we're on
  // a big-endian machine, swap all of the results below before checking.
  //
  union {
    uint32_t a;
    uint8_t  b[4];
  } u;

  u.a = 1;
  bool bigEndian = u.b[3];

  size_t result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() seconds timestamp");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 1234, "Seconds timestamp written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() microseconds timestamp");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 5678, "Microseconds timestamp written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() included length");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 43, "Included length written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() actual length");
  if (bigEndian) val32 = Swap (val32);
  NS_TEST_ASSERT_MSG_EQ (val32, 128, "Actual length written incorrectly");

  //
  // Take a look and see what went out into the file.  The packet data
  // should be unchanged (unswapped).
  //
  uint8_t bufferIn[128];

  result = std::fread (bufferIn, 1, 43, p);
  NS_TEST_ASSERT_MSG_EQ (result, 43, "Unable to fread() packet data of expected length");

  for (uint32_t i = 0; i < 43; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (bufferIn[i], bufferOut[i], "Incorrect packet data written");
    }

  std::fclose (p);
  p = 0;

  //
  // Let's see if the PcapFile object can figure out how to do the same thing and
  // correctly read in a packet.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::in\") of existing good file returns error");

  uint32_t tsSec, tsUsec, inclLen, origLen, readLen;

  f.Read (bufferIn, sizeof(bufferIn), tsSec, tsUsec, inclLen, origLen, readLen);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Read() of known good packet returns error");
  NS_TEST_ASSERT_MSG_EQ (tsSec, 1234, "Incorrectly read seconds timestamp from known good packet");
  NS_TEST_ASSERT_MSG_EQ (tsUsec, 5678, "Incorrectly read microseconds timestamp from known good packet");
  NS_TEST_ASSERT_MSG_EQ (inclLen, 43, "Incorrectly read included length from known good packet");
  NS_TEST_ASSERT_MSG_EQ (origLen, 128, "Incorrectly read original length from known good packet");
  NS_TEST_ASSERT_MSG_EQ (readLen, 43, "Incorrectly constructed actual read length from known good packet given buffer size");
  f.Close ();

  //
  // Did the data come back correctly?
  //
  for (uint32_t i = 0; i < 43; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (bufferIn[i], bufferOut[i], "Incorrect packet data read from known good packet");
    }

  //
  // We have to check to make sure that the pcap record header is swapped 
  // correctly.  Since big-endian machines are automatically forced into 
  // swap mode, the <true> parameter to f.Init() below is actually
  // a no-op and we're always writing foreign-endian files.  In that case, 
  // this test case is really just a duplicate of the previous.
  //
  // Open the file in write mode to clear the data.
  //
  f.Open (m_testFilename, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::out\") returns error");

  //
  // Initialize the pcap file header, forcing the object into swap mode.
  //
  f.Init (37, 43, -7, true);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (37, 43, -7) returns error");

  //
  // Now we should be able to write a packet to it since it was opened in "w" 
  // mode.  The packet data written should be limited to 43 bytes in length 
  // by the Init() call above.
  //
  f.Write (1234, 5678, bufferOut, 128);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Write (write-only-file " << m_testFilename << ") returns error");
  f.Close ();

  //
  // Let's peek into the file and see what actually went out for that
  // packet.
  //
  p = std::fopen (m_testFilename.c_str (), "r+b");
  NS_TEST_ASSERT_MSG_NE (p, 0, "fopen() should have been able to open a correctly created pcap file");

  //
  // A pcap file header takes up 24 bytes, a pcap record header takes up 16 bytes
  // and we wrote in 43 bytes, so the file must be 83 bytes long.  Let's just
  // double check that this is exactly what happened.
  //
  std::fseek (p, 0, SEEK_END);
  size = std::ftell (p);
  NS_TEST_ASSERT_MSG_EQ (size, 83, "Pcap file with one 43 byte packet is incorrect size");

  //
  // A pcap file header takes up 24 bytes, so we should see a pcap record header
  // starting there in the file.  We've tested this all before so we just assume
  // it's all right and just seek past it.
  //
  result = std::fseek (p, 24, SEEK_SET);
  NS_TEST_ASSERT_MSG_EQ (result, 0, "Failed seeking past pcap header");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() seconds timestamp");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (1234)), "Swapped seconds timestamp written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() microseconds timestamp");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (5678)), "Swapped microseconds timestamp written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() included length");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (43)), "Swapped included length written incorrectly");

  result = std::fread (&val32, sizeof(val32), 1, p);
  NS_TEST_ASSERT_MSG_EQ (result, 1, "Unable to fread() actual length");
  NS_TEST_ASSERT_MSG_EQ (val32, Swap (uint32_t (128)), "Swapped Actual length written incorrectly");

  //
  // Take a look and see what went out into the file.  The packet data
  // should be unchanged (unswapped).
  //
  result = std::fread (bufferIn, 1, 43, p);
  NS_TEST_ASSERT_MSG_EQ (result, 43, "Unable to fread() packet data of expected length");

  for (uint32_t i = 0; i < 43; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (bufferIn[i], bufferOut[i], "Incorrect packet data written");
    }

  std::fclose (p);
  p = 0;

  //
  // Let's see if the PcapFile object can figure out how to do the same thing and
  // correctly read in a packet.  The record header info should come back to us
  // swapped back into correct form.
  //
  f.Open (m_testFilename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << m_testFilename << 
                         ", \"std::ios::in\") of existing good file returns error");

  f.Read (bufferIn, sizeof(bufferIn), tsSec, tsUsec, inclLen, origLen, readLen);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Read() of known good packet returns error");
  NS_TEST_ASSERT_MSG_EQ (tsSec, 1234, "Incorrectly read seconds timestamp from known good packet");
  NS_TEST_ASSERT_MSG_EQ (tsUsec, 5678, "Incorrectly read microseconds timestamp from known good packet");
  NS_TEST_ASSERT_MSG_EQ (inclLen, 43, "Incorrectly read included length from known good packet");
  NS_TEST_ASSERT_MSG_EQ (origLen, 128, "Incorrectly read original length from known good packet");
  NS_TEST_ASSERT_MSG_EQ (readLen, 43, "Incorrectly constructed actual read length from known good packet given buffer size");

  //
  // Did the data come back correctly (unchanged / unswapped)?
  //
  for (uint32_t i = 0; i < 43; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (bufferIn[i], bufferOut[i], "Incorrect packet data read from known good packet");
    }

  f.Close ();
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test case to make sure that the Pcap File Object can read
 * out the contents of a known good pcap file.
 */
class ReadFileTestCase : public TestCase
{
public:
  ReadFileTestCase ();
  virtual ~ReadFileTestCase ();

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  std::string m_testFilename; //!< File name
};

ReadFileTestCase::ReadFileTestCase ()
  : TestCase ("Check to see that PcapFile can read out a known good pcap file")
{
}

ReadFileTestCase::~ReadFileTestCase ()
{
}

void
ReadFileTestCase::DoSetup (void)
{
}

void
ReadFileTestCase::DoTeardown (void)
{
}

static const uint32_t N_KNOWN_PACKETS = 6;
static const uint32_t N_PACKET_BYTES = 16;

/**
 * PCAP Packet structure
 */
typedef struct PACKET_ENTRY {
  uint32_t tsSec;   //!< Time (seconds part)
  uint32_t tsUsec;  //!< Time (micro seconds part)
  uint32_t inclLen; //!< Length of the entry in the PCAP
  uint32_t origLen; //!< length of the original packet
  uint16_t data[N_PACKET_BYTES];  //!< Packet data
} PacketEntry;

static const PacketEntry knownPackets[] = {
  { 2, 3696,   46,   46, { 0x0001, 0x0800, 0x0604, 0x0001, 0x0000, 0x0000, 0x0003, 0x0a01,
                           0x0201, 0xffff, 0xffff, 0xffff, 0x0a01, 0x0204, 0x0000, 0x0000}},
  { 2, 3707,   46,   46, { 0x0001, 0x0800, 0x0604, 0x0002, 0x0000, 0x0000, 0x0006, 0x0a01,
                           0x0204, 0x0000, 0x0000, 0x0003, 0x0a01, 0x0201, 0x0000, 0x0000}},
  { 2, 3801, 1070, 1070, { 0x4500, 0x041c, 0x0000, 0x0000, 0x3f11, 0x0000, 0x0a01, 0x0101,
                           0x0a01, 0x0204, 0xc001, 0x0009, 0x0408, 0x0000, 0x0000, 0x0000}},
  { 2, 3811,   46,   46, { 0x0001, 0x0800, 0x0604, 0x0001, 0x0000, 0x0000, 0x0006, 0x0a01,
                           0x0204, 0xffff, 0xffff, 0xffff, 0x0a01, 0x0201, 0x0000, 0x0000}},
  { 2, 3822,   46,   46, { 0x0001, 0x0800, 0x0604, 0x0002, 0x0000, 0x0000, 0x0003, 0x0a01,
                           0x0201, 0x0000, 0x0000, 0x0006, 0x0a01, 0x0204, 0x0000, 0x0000}},
  { 2, 3915, 1070, 1070, { 0x4500, 0x041c, 0x0000, 0x0000, 0x4011, 0x0000, 0x0a01, 0x0204,
                           0x0a01, 0x0101, 0x0009, 0xc001, 0x0408, 0x0000, 0x0000, 0x0000}}
};


void
ReadFileTestCase::DoRun (void)
{
  PcapFile f;

  //
  //
  std::string filename = CreateDataDirFilename ("known.pcap");
  f.Open (filename, std::ios::in);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << filename << 
                         ", \"std::ios::in\") returns error");
 
  //
  // We are going to read out the file header and all of the packets to make 
  // sure that we read what we know, a priori, to be there.
  //
  // The packet data was gotten using "tcpdump -nn -tt -r known.pcap -x"
  // and the timestamp and first 32 bytes of the resulting dump were 
  // duplicated in the structure above.
  //
  uint8_t data[N_PACKET_BYTES];
  uint32_t tsSec, tsUsec, inclLen, origLen, readLen;

  for (uint32_t i = 0; i < N_KNOWN_PACKETS; ++i)
    {
      PacketEntry const & p = knownPackets[i];

      f.Read (data, sizeof(data), tsSec, tsUsec, inclLen, origLen, readLen);
      NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Read() of known good pcap file returns error");
      NS_TEST_ASSERT_MSG_EQ (tsSec, p.tsSec, "Incorrectly read seconds timestap from known good pcap file");
      NS_TEST_ASSERT_MSG_EQ (tsUsec, p.tsUsec, "Incorrectly read microseconds timestap from known good pcap file");
      NS_TEST_ASSERT_MSG_EQ (inclLen, p.inclLen, "Incorrectly read included length from known good packet");
      NS_TEST_ASSERT_MSG_EQ (origLen, p.origLen, "Incorrectly read original length from known good packet");
      NS_TEST_ASSERT_MSG_EQ (readLen, N_PACKET_BYTES, "Incorrect actual read length from known good packet given buffer size");
    }

  //
  // The file should now be at EOF since we've read all of the packets.
  // Another packet read should return an error.
  //
  f.Read (data, 1, tsSec, tsUsec, inclLen, origLen, readLen);
  NS_TEST_ASSERT_MSG_EQ (f.Eof (), true, "Read() of known good pcap file at EOF does not return error");

  f.Close ();
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test case to make sure that the Pcap::Diff method works as expected.
 */
class DiffTestCase : public TestCase
{
public:
  DiffTestCase ();

private:
  virtual void DoRun (void);
};

DiffTestCase::DiffTestCase ()
  : TestCase ("Check that PcapFile::Diff works as expected")
{
}

void
DiffTestCase::DoRun (void)
{
  //
  // Check that PcapDiff(file, file) is false
  //
  std::string filename = CreateDataDirFilename ("known.pcap");
  uint32_t sec (0), usec (0), packets (0);
  bool diff = PcapFile::Diff (filename, filename, sec, usec, packets);
  NS_TEST_EXPECT_MSG_EQ (diff, false, "PcapDiff(file, file) must always be false");

  //
  // Create different PCAP file (with the same timestamps, but different packets) and check that it is indeed different 
  //
  std::string filename2 = CreateTempDirFilename ("different.pcap");
  PcapFile f;

  f.Open (filename2, std::ios::out);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Open (" << filename2 << ", \"std::ios::out\") returns error");
  f.Init (1, N_PACKET_BYTES);
  NS_TEST_ASSERT_MSG_EQ (f.Fail (), false, "Init (1, " << N_PACKET_BYTES << ") returns error");

  for (uint32_t i = 0; i < N_KNOWN_PACKETS; ++i)
    {
      PacketEntry const & p = knownPackets[i];

      f.Write (p.tsSec, p.tsUsec, (uint8_t const *)p.data, p.origLen);
      NS_TEST_EXPECT_MSG_EQ (f.Fail (), false, "Write must not fail");
    }
  f.Close ();

  packets = 0;
  diff = PcapFile::Diff (filename, filename2, sec, usec, packets);
  NS_TEST_EXPECT_MSG_EQ (diff, true, "PcapDiff(file, file2) must be true");
  NS_TEST_EXPECT_MSG_EQ (sec,  2, "Files are different from 2.3696 seconds");
  NS_TEST_EXPECT_MSG_EQ (usec, 3696, "Files are different from 2.3696 seconds");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief PCAP file utils TestSuite
 */
class PcapFileTestSuite : public TestSuite
{
public:
  PcapFileTestSuite ();
};

PcapFileTestSuite::PcapFileTestSuite ()
  : TestSuite ("pcap-file", UNIT)
{
  SetDataDir (NS_TEST_SOURCEDIR);
  AddTestCase (new WriteModeCreateTestCase, TestCase::QUICK);
  AddTestCase (new ReadModeCreateTestCase, TestCase::QUICK);
  //AddTestCase (new AppendModeCreateTestCase, TestCase::QUICK);
  AddTestCase (new FileHeaderTestCase, TestCase::QUICK);
  AddTestCase (new RecordHeaderTestCase, TestCase::QUICK);
  AddTestCase (new ReadFileTestCase, TestCase::QUICK);
  AddTestCase (new DiffTestCase, TestCase::QUICK);
}

static PcapFileTestSuite pcapFileTestSuite; //!< Static variable for test initialization
