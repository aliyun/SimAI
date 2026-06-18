/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@cutebugs.net>
 */

#include "ns3/buffer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/double.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup network
 * \defgroup network-test Network module unit tests
 */

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * Buffer unit tests.
 */
class BufferTest : public TestCase {
private:
  /**
   * Checks the buffer content
   * \param b The buffer to check
   * \param n The number of bytes to check
   * \param array The array of bytes that should be in the buffer
   */
  void EnsureWrittenBytes (Buffer b, uint32_t n, uint8_t array[]);
public:
  virtual void DoRun (void);
  BufferTest ();
};


BufferTest::BufferTest ()
  : TestCase ("Buffer") {
}

void
BufferTest::EnsureWrittenBytes (Buffer b, uint32_t n, uint8_t array[])
{
  bool success = true;
  uint8_t *expected = array;
  uint8_t const*got;
  got = b.PeekData ();
  for (uint32_t j = 0; j < n; j++)
    {
      if (got[j] != expected[j])
        {
          success = false;
        }
    }
  if (!success)
    {
      std::ostringstream failure;
      failure << "Buffer -- ";
      failure << "expected: n=";
      failure << n << ", ";
      failure.setf (std::ios::hex, std::ios::basefield);
      for (uint32_t j = 0; j < n; j++)
        {
          failure << (uint16_t)expected[j] << " ";
        }
      failure.setf (std::ios::dec, std::ios::basefield);
      failure << "got: ";
      failure.setf (std::ios::hex, std::ios::basefield);
      for (uint32_t j = 0; j < n; j++)
        {
          failure << (uint16_t)got[j] << " ";
        }
      failure << std::endl;
      NS_TEST_ASSERT_MSG_EQ (true, false, failure.str ());
    }
}

/*
 * Works only when variadic macros are
 * available which is the case for gcc.
 */
#define ENSURE_WRITTEN_BYTES(buffer, n, ...) \
  {                                          \
    uint8_t bytes[] = { __VA_ARGS__};        \
    EnsureWrittenBytes (buffer, n, bytes);   \
  }

void
BufferTest::DoRun (void)
{
  Buffer buffer;
  Buffer::Iterator i;
  buffer.AddAtStart (6);
  i = buffer.Begin ();
  i.WriteU8 (0x66);
  ENSURE_WRITTEN_BYTES (buffer, 1, 0x66);
  i = buffer.Begin ();
  i.WriteU8 (0x67);
  ENSURE_WRITTEN_BYTES (buffer, 1, 0x67);
  i.WriteHtonU16 (0x6568);
  i = buffer.Begin ();
  ENSURE_WRITTEN_BYTES (buffer, 3, 0x67, 0x65, 0x68);
  i.WriteHtonU16 (0x6369);
  ENSURE_WRITTEN_BYTES (buffer, 3, 0x63, 0x69, 0x68);
  i.WriteHtonU32 (0xdeadbeaf);
  ENSURE_WRITTEN_BYTES (buffer, 6, 0x63, 0x69, 0xde, 0xad, 0xbe, 0xaf);
  buffer.AddAtStart (2);
  i = buffer.Begin ();
  i.WriteU16 (0);
  ENSURE_WRITTEN_BYTES (buffer, 8, 0, 0, 0x63, 0x69, 0xde, 0xad, 0xbe, 0xaf);
  buffer.AddAtEnd (2);
  i = buffer.Begin ();
  i.Next (8);
  i.WriteU16 (0);
  ENSURE_WRITTEN_BYTES (buffer, 10, 0, 0, 0x63, 0x69, 0xde, 0xad, 0xbe, 0xaf, 0, 0);
  buffer.RemoveAtStart (3);
  i = buffer.Begin ();
  ENSURE_WRITTEN_BYTES (buffer, 7, 0x69, 0xde, 0xad, 0xbe, 0xaf, 0, 0);
  buffer.RemoveAtEnd (4);
  i = buffer.Begin ();
  ENSURE_WRITTEN_BYTES (buffer, 3, 0x69, 0xde, 0xad);
  buffer.AddAtStart (1);
  i = buffer.Begin ();
  i.WriteU8 (0xff);
  ENSURE_WRITTEN_BYTES (buffer, 4, 0xff, 0x69, 0xde, 0xad);
  buffer.AddAtEnd (1);
  i = buffer.Begin ();
  i.Next (4);
  i.WriteU8 (0xff);
  i.Prev (2);
  uint16_t saved = i.ReadU16 ();
  i.Prev (2);
  i.WriteHtonU16 (0xff00);
  i.Prev (2);
  NS_TEST_ASSERT_MSG_EQ (i.ReadNtohU16 (), 0xff00, "Could not read expected value");
  i.Prev (2);
  i.WriteU16 (saved);
  ENSURE_WRITTEN_BYTES (buffer, 5, 0xff, 0x69, 0xde, 0xad, 0xff);
  Buffer o = buffer;
  ENSURE_WRITTEN_BYTES (o, 5, 0xff, 0x69, 0xde, 0xad, 0xff);
  o.AddAtStart (1);
  i = o.Begin ();
  i.WriteU8 (0xfe);
  ENSURE_WRITTEN_BYTES (o, 6, 0xfe, 0xff, 0x69, 0xde, 0xad, 0xff);
  buffer.AddAtStart (2);
  i = buffer.Begin ();
  i.WriteU8 (0xfd);
  i.WriteU8 (0xfd);
  ENSURE_WRITTEN_BYTES (o, 6, 0xfe, 0xff, 0x69, 0xde, 0xad, 0xff);
  ENSURE_WRITTEN_BYTES (buffer, 7, 0xfd, 0xfd, 0xff, 0x69, 0xde, 0xad, 0xff);

  // test 64-bit read/write
  Buffer buff64;
  buff64.AddAtStart (8);
  i = buff64.Begin ();
  i.WriteU64 (0x0123456789ABCDEFllu);
  i = buff64.Begin ();
  NS_TEST_ASSERT_MSG_EQ (i.ReadU64 (), 0x0123456789abcdefllu, "Could not read expected value");
  i = buff64.Begin ();
  i.WriteHtolsbU64 (0x0123456789ABCDEFllu);
  ENSURE_WRITTEN_BYTES (buff64, 8, 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01);
  i = buff64.Begin ();
  NS_TEST_ASSERT_MSG_EQ (i.ReadLsbtohU64 (), 0x0123456789abcdefllu, "Could not read expected value");
  i = buff64.Begin ();
  i.WriteHtonU64 (0x0123456789ABCDEFllu);
  ENSURE_WRITTEN_BYTES (buff64, 8, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef);
  i = buff64.Begin ();
  NS_TEST_ASSERT_MSG_EQ (i.ReadNtohU64 (), 0x0123456789abcdefllu, "could not read expected value");

  // test self-assignment
  {
    Buffer a = o;
#if defined(__clang__)
  #if __has_warning("-Wself-assign-overloaded")
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wself-assign-overloaded"
  #endif
#endif
    a = a;
#if defined(__clang__)
  #if __has_warning("-Wself-assign-overloaded")
    #pragma clang diagnostic pop
  #endif
#endif
  }

  // test Remove start.
  buffer = Buffer (5);
  ENSURE_WRITTEN_BYTES (buffer, 5, 0, 0, 0, 0, 0);
  buffer.RemoveAtStart (1);
  ENSURE_WRITTEN_BYTES (buffer, 4, 0, 0, 0, 0);
  buffer.AddAtStart (1);
  buffer.Begin ().WriteU8 (0xff);
  ENSURE_WRITTEN_BYTES (buffer, 5, 0xff, 0, 0, 0, 0);
  buffer.RemoveAtStart (3);
  ENSURE_WRITTEN_BYTES (buffer, 2, 0, 0);
  buffer.AddAtStart (4);
  buffer.Begin ().WriteHtonU32 (0xdeadbeaf);
  ENSURE_WRITTEN_BYTES (buffer, 6,  0xde, 0xad, 0xbe, 0xaf, 0, 0);
  buffer.RemoveAtStart (2);
  ENSURE_WRITTEN_BYTES (buffer, 4,  0xbe, 0xaf, 0, 0);
  buffer.AddAtEnd (4);
  i = buffer.Begin ();
  i.Next (4);
  i.WriteHtonU32 (0xdeadbeaf);
  ENSURE_WRITTEN_BYTES (buffer, 8,  0xbe, 0xaf, 0, 0, 0xde, 0xad, 0xbe, 0xaf);
  buffer.RemoveAtStart (5);
  ENSURE_WRITTEN_BYTES (buffer, 3,  0xad, 0xbe, 0xaf);
  // test Remove end
  buffer = Buffer (5);
  ENSURE_WRITTEN_BYTES (buffer, 5, 0, 0, 0, 0, 0);
  buffer.RemoveAtEnd (1);
  ENSURE_WRITTEN_BYTES (buffer, 4, 0, 0, 0, 0);
  buffer.AddAtEnd (2);
  i = buffer.Begin ();
  i.Next (4);
  i.WriteU8 (0xab);
  i.WriteU8 (0xac);
  ENSURE_WRITTEN_BYTES (buffer, 6, 0, 0, 0, 0, 0xab, 0xac);
  buffer.RemoveAtEnd (1);
  ENSURE_WRITTEN_BYTES (buffer, 5, 0, 0, 0, 0, 0xab);
  buffer.RemoveAtEnd (3);
  ENSURE_WRITTEN_BYTES (buffer, 2, 0, 0);
  buffer.AddAtEnd (6);
  i = buffer.Begin ();
  i.Next (2);
  i.WriteU8 (0xac);
  i.WriteU8 (0xad);
  i.WriteU8 (0xae);
  i.WriteU8 (0xaf);
  i.WriteU8 (0xba);
  i.WriteU8 (0xbb);
  ENSURE_WRITTEN_BYTES (buffer, 8, 0, 0, 0xac, 0xad, 0xae, 0xaf, 0xba, 0xbb);
  buffer.AddAtStart (3);
  i = buffer.Begin ();
  i.WriteU8 (0x30);
  i.WriteU8 (0x31);
  i.WriteU8 (0x32);
  ENSURE_WRITTEN_BYTES (buffer, 11, 0x30, 0x31, 0x32, 0, 0, 0xac, 0xad, 0xae, 0xaf, 0xba, 0xbb);
  buffer.RemoveAtEnd (9);
  ENSURE_WRITTEN_BYTES (buffer, 2, 0x30, 0x31);
  buffer = Buffer (3);
  buffer.AddAtEnd (2);
  i = buffer.Begin ();
  i.Next (3);
  i.WriteHtonU16 (0xabcd);
  buffer.AddAtStart (1);
  buffer.Begin ().WriteU8 (0x21);
  ENSURE_WRITTEN_BYTES (buffer, 6, 0x21, 0, 0, 0, 0xab, 0xcd);
  buffer.RemoveAtEnd (8);
  NS_TEST_ASSERT_MSG_EQ (buffer.GetSize (), 0, "Buffer size not zero");

  buffer = Buffer (6);
  buffer.AddAtStart (9);
  buffer.AddAtEnd (3);
  i = buffer.End ();
  i.Prev (1);
  i.WriteU8 (1, 1);

  buffer = Buffer (6);
  buffer.AddAtStart (3);
  buffer.RemoveAtEnd (8);
  buffer.AddAtEnd (4);
  i = buffer.End ();
  i.Prev (4);
  i.WriteU8 (1, 4);

  buffer = Buffer (1);
  buffer.AddAtEnd (100);
  i = buffer.End ();
  i.Prev (100);
  i.WriteU8 (1, 100);

  /// \internal
  /// See \bugid{54}
  {
    const uint32_t actualSize = 72602;
    const uint32_t chunkSize = 67624;
    Ptr<UniformRandomVariable> bytesRng = CreateObject<UniformRandomVariable> ();
    bytesRng->SetAttribute ("Min", DoubleValue (0));
    bytesRng->SetAttribute ("Max", DoubleValue (256));

    Buffer inputBuffer;
    Buffer outputBuffer;

    inputBuffer.AddAtEnd (actualSize);
    {
      Buffer::Iterator iter = inputBuffer.Begin ();
      for (uint32_t i = 0; i < actualSize; i++)
        iter.WriteU8 (static_cast<uint8_t> (bytesRng->GetValue ()));
    }

    outputBuffer.AddAtEnd (chunkSize);
    Buffer::Iterator iter = outputBuffer.End ();
    iter.Prev (chunkSize);
    iter.Write (inputBuffer.PeekData (), chunkSize);

    NS_TEST_EXPECT_MSG_EQ (memcmp (inputBuffer.PeekData (), outputBuffer.PeekData (), chunkSize), 0, "memcp works");
  }

  buffer = Buffer (5);
  buffer.AddAtEnd (2);
  i = buffer.End ();
  i.Prev (2);
  i.WriteU8 (0);
  i.WriteU8 (0x66);
  ENSURE_WRITTEN_BYTES (buffer, 7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66);
  Buffer frag0 = buffer.CreateFragment (0, 2);
  ENSURE_WRITTEN_BYTES (frag0, 2, 0x00, 0x00);
  Buffer frag1 = buffer.CreateFragment (2, 5);
  ENSURE_WRITTEN_BYTES (frag1, 5, 0x00, 0x00, 0x00, 0x00, 0x66);
  frag0.AddAtEnd (frag1);
  ENSURE_WRITTEN_BYTES (buffer, 7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66);
  ENSURE_WRITTEN_BYTES (frag0, 7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66);

  buffer = Buffer (5);
  buffer.AddAtStart (2);
  i = buffer.Begin ();
  i.WriteU8 (0x1);
  i.WriteU8 (0x2);
  buffer.AddAtEnd (2);
  i = buffer.End ();
  i.Prev (2);
  i.WriteU8 (0x3);
  i.WriteU8 (0x4);
  ENSURE_WRITTEN_BYTES (buffer, 9, 0x1, 0x2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3, 0x4);
  Buffer other;
  other.AddAtStart (9);
  i = other.Begin ();
  i.Write (buffer.Begin (), buffer.End ());
  ENSURE_WRITTEN_BYTES (other, 9, 0x1, 0x2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3, 0x4);

  /// \internal See \bugid{1001}
  std::string ct ("This is the next content of the buffer.");
  buffer = Buffer ();
  buffer.AddAtStart (ct.size ());
  i = buffer.Begin ();
  i.Write ((const uint8_t*)ct.c_str (), ct.size ());
  uint32_t sizeBuffer = buffer.GetSize ();
  NS_TEST_ASSERT_MSG_EQ (sizeBuffer, ct.size(), "Buffer bad size");
  uint8_t const* evilBuffer = buffer.PeekData ();
  NS_TEST_ASSERT_MSG_NE( evilBuffer, 0, "Buffer PeekData failed");
  uint8_t *cBuf = (uint8_t*) malloc ( sizeBuffer );
  uint32_t copyLen = buffer.CopyData (cBuf, sizeBuffer);
  NS_TEST_ASSERT_MSG_EQ (copyLen, sizeBuffer, "CopyData return bad size");
  for (uint8_t i=0; i < sizeBuffer ; i++ )
    {
      NS_TEST_ASSERT_MSG_EQ ( cBuf [i], *(((const uint8_t*)ct.c_str ()) + i), "Bad buffer copied data");
      NS_TEST_ASSERT_MSG_EQ ( evilBuffer [i], cBuf [i] , "Bad buffer peeked");
    }
  free (cBuf);

  /// \internal See \bugid{2044}  Will not pass without bug2044 fix.
  buffer = Buffer (1);
  buffer.AddAtEnd (2);
  i = buffer.Begin ();
  i.Next (1);
  i.WriteU8 (0x77);
  i.WriteU8 (0x66);
  ENSURE_WRITTEN_BYTES (buffer, 3, 0x00, 0x77, 0x66);
  i = buffer.Begin ();
  i.ReadU8 ();
  uint16_t val1 = i.ReadNtohU16 ();
  i = buffer.Begin ();
  i.ReadU8 ();
  uint16_t val2 = 0;
  val2 |= i.ReadU8 ();
  val2 <<= 8;
  val2 |= i.ReadU8 ();
  NS_TEST_ASSERT_MSG_EQ (val1, val2, "Bad ReadNtohU16()");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Buffer TestSuite
 */
class BufferTestSuite : public TestSuite
{
public:
  BufferTestSuite ();
};

BufferTestSuite::BufferTestSuite ()
  : TestSuite ("buffer", UNIT)
{
  AddTestCase (new BufferTest, TestCase::QUICK);
}

static BufferTestSuite g_bufferTestSuite; //!< Static variable for test initialization
