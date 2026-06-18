/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* vim: set ts=2 sw=2 sta expandtab ai si cin: */
/* 
 * Copyright (c) 2009 Drexel University
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
 * Author: Tom Wambold <tom5760@gmail.com>
 */

#include <cstring>
#include <iostream>
#include "ns3/test.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/packetbb.h"

using namespace ns3;

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief PacketBb TestCase
 */
class PbbTestCase : public TestCase
{
public:
  /**
   * Constructor
   * \param name Test name.
   * \param packet Packet to test.
   * \param buffer Buffer to test.
   * \param size Buffer size.
   */
  PbbTestCase (std::string name, Ptr<PbbPacket> packet,
               uint8_t * buffer, uint32_t size);
  virtual ~PbbTestCase (void);

protected:
  virtual void DoRun (void);

private:
  /// Serialization
  void TestSerialize (void);
  /// Deserialization
  void TestDeserialize (void);

  Ptr<PbbPacket> m_refPacket; //!< Reference packet
  Buffer m_refBuffer; //!< Reference buffer
};

PbbTestCase::PbbTestCase (std::string name, Ptr<PbbPacket> packet,
                          uint8_t * buffer, uint32_t size)
  : TestCase (name)
{
  m_refPacket = packet;

  m_refBuffer.AddAtStart (size);
  m_refBuffer.Begin ().Write (buffer, size);
}

PbbTestCase::~PbbTestCase (void)
{
  return;
}

void
PbbTestCase::DoRun (void)
{
  TestSerialize ();
  TestDeserialize ();
}

void
PbbTestCase::TestSerialize (void)
{
  Buffer newBuffer;
  newBuffer.AddAtStart (m_refPacket->GetSerializedSize ());
  m_refPacket->Serialize (newBuffer.Begin ());

  NS_TEST_ASSERT_MSG_EQ (newBuffer.GetSize (), m_refBuffer.GetSize (),
                         "serialization failed, buffers have different sizes");

  int memrv = memcmp (newBuffer.PeekData (), m_refBuffer.PeekData (),
                      newBuffer.GetSize ());

  NS_TEST_ASSERT_MSG_EQ (memrv, 0,
                         "serialization faled, buffers differ");
}

void
PbbTestCase::TestDeserialize (void)
{
  Ptr<PbbPacket> newPacket = Create<PbbPacket> ();
  uint32_t numbytes = newPacket->Deserialize (m_refBuffer.Begin ());

  NS_TEST_ASSERT_MSG_EQ (numbytes, m_refBuffer.GetSize (),
                                      "deserialization failed, did not use all bytes");

  NS_TEST_ASSERT_MSG_EQ (*newPacket, *m_refPacket,
                                      "deserialization failed, objects do not match");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief PacketBb TestSuite
 */
class PbbTestSuite : public TestSuite
{
public:
  PbbTestSuite ();
};

PbbTestSuite::PbbTestSuite ()
  : TestSuite ("packetbb-test-suite", UNIT)
{
  /* Test 1
   *    ,------------------
   *    |  PACKET
   *    |------------------
   *    | * Packet version:    0
   *    | * Packet flags:  0
   *    `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    uint8_t buffer[] = { 0x00};
    AddTestCase (new PbbTestCase ("1", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 2
   * ,------------------
   * |  PACKET
   * |------------------
   * | * Packet version:    0
   * | * Packet flags:  8
   * | * Packet seq number: 2
   * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (2);
    uint8_t buffer[] = { 0x08, 0x00, 0x02};
    AddTestCase (new PbbTestCase ("2", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 3
   * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 3
         * `------------------
   * This test has the phastlv flag set to 1 with no tlvs.
   * I'll come back to this one later.
   */
#if 0
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (3);
    uint8_t buffer[] = { 0x0c, 0x00, 0x03, 0x00, 0x00};
    AddTestCase (new PbbTestCase ("3", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }
#endif

  /* Test 4
   * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 4
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (4);

    Ptr<PbbTlv> tlv = Create<PbbTlv>();
    tlv->SetType (1);

    packet->TlvPushBack (tlv);
    uint8_t buffer[] = {
      0x0c, 0x00, 0x04, 0x00,
      0x02, 0x01, 0x00
    };
    AddTestCase (new PbbTestCase ("4", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 5
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 5
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    |     - TLV
         * |    |         Flags = 128
         * |    |         Type = 2; Type ext. = 100; Value = (warning: parameter is NULL)
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (5);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv>();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbTlv> tlv2 = Create<PbbTlv>();
    tlv2->SetType (2);
    tlv2->SetTypeExt (100);
    packet->TlvPushBack (tlv2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x05, 0x00,
      0x05, 0x01, 0x00, 0x02,
      0x80, 0x64
    };
    AddTestCase (new PbbTestCase ("5", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 6
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 6
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    |     - TLV
         * |    |         Flags = 144
         * |    |         Type = 2; Type ext. = 100; Value = 01  02  03  04
         * |    |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (6);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv>();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbTlv> tlv2 = Create<PbbTlv>();
    tlv2->SetType (2);
    tlv2->SetTypeExt (100);

    uint8_t tlv2val[] = { 1, 2, 3, 4};
    tlv2->SetValue (tlv2val, sizeof(tlv2val));

    packet->TlvPushBack (tlv2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x06, 0x00,
      0x0a, 0x01, 0x00, 0x02,
      0x90, 0x64, 0x04, 0x01,
      0x02, 0x03, 0x04
    };
    AddTestCase (new PbbTestCase ("6", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 7
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 7
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    |     - TLV
         * |    |         Flags = 152
         * |    |         Type = 2; Type ext. = 100; Value = 00  01  02  03
         * |    |                                            04  05  06  07
         * |    |                                            08  09  0a  0b
         * |    |                                            0c  0d  0e  0f
         * |    |                                            10  11  12  13
         * |    |                                            14  15  16  17
         * |    |                                            18  19  1a  1b
         * |    |                                            1c  1d  1e  1f
         * |    |                                            20  21  22  23
         * |    |                                            24  25  26  27
         * |    |                                            28  29  2a  2b
         * |    |                                            2c  2d  2e  2f
         * |    |                                            30  31  32  33
         * |    |                                            34  35  36  37
         * |    |                                            38  39  3a  3b
         * |    |                                            3c  3d  3e  3f
         * |    |                                            40  41  42  43
         * |    |                                            44  45  46  47
         * |    |                                            48  49  4a  4b
         * |    |                                            4c  4d  4e  4f
         * |    |                                            50  51  52  53
         * |    |                                            54  55  56  57
         * |    |                                            58  59  5a  5b
         * |    |                                            5c  5d  5e  5f
         * |    |                                            60  61  62  63
         * |    |                                            64  65  66  67
         * |    |                                            68  69  6a  6b
         * |    |                                            6c  6d  6e  6f
         * |    |                                            70  71  72  73
         * |    |                                            74  75  76  77
         * |    |                                            78  79  7a  7b
         * |    |                                            7c  7d  7e  7f
         * |    |                                            80  81  82  83
         * |    |                                            84  85  86  87
         * |    |                                            88  89  8a  8b
         * |    |                                            8c  8d  8e  8f
         * |    |                                            90  91  92  93
         * |    |                                            94  95  96  97
         * |    |                                            98  99  9a  9b
         * |    |                                            9c  9d  9e  9f
         * |    |                                            a0  a1  a2  a3
         * |    |                                            a4  a5  a6  a7
         * |    |                                            a8  a9  aa  ab
         * |    |                                            ac  ad  ae  af
         * |    |                                            b0  b1  b2  b3
         * |    |                                            b4  b5  b6  b7
         * |    |                                            b8  b9  ba  bb
         * |    |                                            bc  bd  be  bf
         * |    |                                            c0  c1  c2  c3
         * |    |                                            c4  c5  c6  c7
         * |    |                                            c8  c9  ca  cb
         * |    |                                            cc  cd  ce  cf
         * |    |                                            d0  d1  d2  d3
         * |    |                                            d4  d5  d6  d7
         * |    |                                            d8  d9  da  db
         * |    |                                            dc  dd  de  df
         * |    |                                            e0  e1  e2  e3
         * |    |                                            e4  e5  e6  e7
         * |    |                                            e8  e9  ea  eb
         * |    |                                            ec  ed  ee  ef
         * |    |                                            f0  f1  f2  f3
         * |    |                                            f4  f5  f6  f7
         * |    |                                            f8  f9  fa  fb
         * |    |                                            fc  fd  fe  00
         * |    |                                            01  02  03  04
         * |    |                                            05  06  07  08
         * |    |                                            09  0a  0b  0c
         * |    |                                            0d  0e  0f  10
         * |    |                                            11  12  13  14
         * |    |                                            15  16  17  18
         * |    |                                            19  1a  1b  1c
         * |    |                                            1d  1e  1f  20
         * |    |                                            21  22  23  24
         * |    |                                            25  26  27  28
         * |    |                                            29  2a  2b  2c
         * |    |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (7);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv>();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbTlv> tlv2 = Create<PbbTlv>();
    tlv2->SetType (2);
    tlv2->SetTypeExt (100);

    uint8_t tlv2val[] = {
      0x00, 0x01, 0x02, 0x03,
      0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b,
      0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13,
      0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b,
      0x1c, 0x1d, 0x1e, 0x1f,
      0x20, 0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27,
      0x28, 0x29, 0x2a, 0x2b,
      0x2c, 0x2d, 0x2e, 0x2f,
      0x30, 0x31, 0x32, 0x33,
      0x34, 0x35, 0x36, 0x37,
      0x38, 0x39, 0x3a, 0x3b,
      0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43,
      0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b,
      0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53,
      0x54, 0x55, 0x56, 0x57,
      0x58, 0x59, 0x5a, 0x5b,
      0x5c, 0x5d, 0x5e, 0x5f,
      0x60, 0x61, 0x62, 0x63,
      0x64, 0x65, 0x66, 0x67,
      0x68, 0x69, 0x6a, 0x6b,
      0x6c, 0x6d, 0x6e, 0x6f,
      0x70, 0x71, 0x72, 0x73,
      0x74, 0x75, 0x76, 0x77,
      0x78, 0x79, 0x7a, 0x7b,
      0x7c, 0x7d, 0x7e, 0x7f,
      0x80, 0x81, 0x82, 0x83,
      0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8a, 0x8b,
      0x8c, 0x8d, 0x8e, 0x8f,
      0x90, 0x91, 0x92, 0x93,
      0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9a, 0x9b,
      0x9c, 0x9d, 0x9e, 0x9f,
      0xa0, 0xa1, 0xa2, 0xa3,
      0xa4, 0xa5, 0xa6, 0xa7,
      0xa8, 0xa9, 0xaa, 0xab,
      0xac, 0xad, 0xae, 0xaf,
      0xb0, 0xb1, 0xb2, 0xb3,
      0xb4, 0xb5, 0xb6, 0xb7,
      0xb8, 0xb9, 0xba, 0xbb,
      0xbc, 0xbd, 0xbe, 0xbf,
      0xc0, 0xc1, 0xc2, 0xc3,
      0xc4, 0xc5, 0xc6, 0xc7,
      0xc8, 0xc9, 0xca, 0xcb,
      0xcc, 0xcd, 0xce, 0xcf,
      0xd0, 0xd1, 0xd2, 0xd3,
      0xd4, 0xd5, 0xd6, 0xd7,
      0xd8, 0xd9, 0xda, 0xdb,
      0xdc, 0xdd, 0xde, 0xdf,
      0xe0, 0xe1, 0xe2, 0xe3,
      0xe4, 0xe5, 0xe6, 0xe7,
      0xe8, 0xe9, 0xea, 0xeb,
      0xec, 0xed, 0xee, 0xef,
      0xf0, 0xf1, 0xf2, 0xf3,
      0xf4, 0xf5, 0xf6, 0xf7,
      0xf8, 0xf9, 0xfa, 0xfb,
      0xfc, 0xfd, 0xfe, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c
    };
    tlv2->SetValue (tlv2val, sizeof(tlv2val));

    packet->TlvPushBack (tlv2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x07, 0x01,
      0x33, 0x01, 0x00, 0x02,
      0x98, 0x64, 0x01, 0x2c,
      0x00, 0x01, 0x02, 0x03,
      0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b,
      0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13,
      0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b,
      0x1c, 0x1d, 0x1e, 0x1f,
      0x20, 0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27,
      0x28, 0x29, 0x2a, 0x2b,
      0x2c, 0x2d, 0x2e, 0x2f,
      0x30, 0x31, 0x32, 0x33,
      0x34, 0x35, 0x36, 0x37,
      0x38, 0x39, 0x3a, 0x3b,
      0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43,
      0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b,
      0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53,
      0x54, 0x55, 0x56, 0x57,
      0x58, 0x59, 0x5a, 0x5b,
      0x5c, 0x5d, 0x5e, 0x5f,
      0x60, 0x61, 0x62, 0x63,
      0x64, 0x65, 0x66, 0x67,
      0x68, 0x69, 0x6a, 0x6b,
      0x6c, 0x6d, 0x6e, 0x6f,
      0x70, 0x71, 0x72, 0x73,
      0x74, 0x75, 0x76, 0x77,
      0x78, 0x79, 0x7a, 0x7b,
      0x7c, 0x7d, 0x7e, 0x7f,
      0x80, 0x81, 0x82, 0x83,
      0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8a, 0x8b,
      0x8c, 0x8d, 0x8e, 0x8f,
      0x90, 0x91, 0x92, 0x93,
      0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9a, 0x9b,
      0x9c, 0x9d, 0x9e, 0x9f,
      0xa0, 0xa1, 0xa2, 0xa3,
      0xa4, 0xa5, 0xa6, 0xa7,
      0xa8, 0xa9, 0xaa, 0xab,
      0xac, 0xad, 0xae, 0xaf,
      0xb0, 0xb1, 0xb2, 0xb3,
      0xb4, 0xb5, 0xb6, 0xb7,
      0xb8, 0xb9, 0xba, 0xbb,
      0xbc, 0xbd, 0xbe, 0xbf,
      0xc0, 0xc1, 0xc2, 0xc3,
      0xc4, 0xc5, 0xc6, 0xc7,
      0xc8, 0xc9, 0xca, 0xcb,
      0xcc, 0xcd, 0xce, 0xcf,
      0xd0, 0xd1, 0xd2, 0xd3,
      0xd4, 0xd5, 0xd6, 0xd7,
      0xd8, 0xd9, 0xda, 0xdb,
      0xdc, 0xdd, 0xde, 0xdf,
      0xe0, 0xe1, 0xe2, 0xe3,
      0xe4, 0xe5, 0xe6, 0xe7,
      0xe8, 0xe9, 0xea, 0xeb,
      0xec, 0xed, 0xee, 0xef,
      0xf0, 0xf1, 0xf2, 0xf3,
      0xf4, 0xf5, 0xf6, 0xf7,
      0xf8, 0xf9, 0xfa, 0xfb,
      0xfc, 0xfd, 0xfe, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c
    };
    AddTestCase (new PbbTestCase ("7", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 8
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 8
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    `-------------------
         * |
         * `------------------
  */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (8);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);
    packet->MessagePushBack (msg1);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x06, 0x00,
      0x00
    };
    AddTestCase (new PbbTestCase ("8", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 9
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 9
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  128
         * |    | * Originator address: 10.0.0.1
         * |    `-------------------
         * |
         * `------------------
  */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (9);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);
    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x09, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x06, 0x00,
      0x00, 0x02, 0x83, 0x00,   /* [14] used to be 0x80 */
      0x0a, 0x0a, 0x00, 0x00,
      0x01, 0x00, 0x00
    };
    AddTestCase (new PbbTestCase ("9", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 10
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 10
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  160
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop count:          1
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (10);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);
    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopCount (1);
    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x0a, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x06, 0x00,
      0x00, 0x02, 0xa3, 0x00,     /* [14] used to be 0xa0 */
      0x0b, 0x0a, 0x00, 0x00,
      0x01, 0x01, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("10", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 11
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 11
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  224
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (11);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);
    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x0b, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x06, 0x00,
      0x00, 0x02, 0xe3, 0x00,   /* [14] used to be 0xe0 */
      0x0c, 0x0a, 0x00, 0x00,
      0x01, 0xff, 0x01, 0x00,
      0x00
    };
    AddTestCase (new PbbTestCase ("11", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 12
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 12
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (12);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);
    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);
    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x0c, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x06, 0x00,
      0x00, 0x02, 0xf3, 0x00,   /* [14] - 0xf0 */
      0x0e, 0x0a, 0x00, 0x00,
      0x01, 0xff, 0x01, 0x30,
      0x39, 0x00, 0x00
    };
    AddTestCase (new PbbTestCase ("12", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 13
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 13
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (13);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);
    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);
    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x0d, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x06, 0x00,
      0x00, 0x02, 0xf3, 0x00,   /* [14] - 0xf0 */
      0x0e, 0x0a, 0x00, 0x00,
      0x01, 0xff, 0x01, 0x30,
      0x39, 0x00, 0x00
    };
    AddTestCase (new PbbTestCase ("13", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 14
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 14
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (14);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);
    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x0e, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x0e, 0x0a,  /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00
    };
    AddTestCase (new PbbTestCase ("14", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 15
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 15
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (1 addresses)
         * |    |     - 0.0.0.0/32
         * |    |     - Flags = 0
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (15);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("0.0.0.0"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x0f, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x16, 0x0a,    /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00
    };
    AddTestCase (new PbbTestCase ("15", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 16
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 16
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (1 addresses)
         * |    |     - 255.255.255.255/32
         * |    |     - Flags = 0
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (16);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("255.255.255.255"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x10, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x16, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x01, 0x00, 0xff,
      0xff, 0xff, 0xff, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("16", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 17
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 17
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (1 addresses)
         * |    |     - 0.0.0.1/32
         * |    |     - Flags = 0
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (17);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("0.0.0.1"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x11, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x16, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("17", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 18
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 18
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (1 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - Flags = 0
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (18);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x12, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x16, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x01, 0x00, 0x0a,
      0x00, 0x00, 0x00, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("18", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 19
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 19
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (1 addresses)
         * |    |     - 10.0.0.1/32
         * |    |     - Flags = 0
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (19);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.1"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x13, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x16, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x01, 0x00, 0x0a,
      0x00, 0x00, 0x01, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("19", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 20
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 20
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.1/32
         * |    |     - 10.0.0.2/32
         * |    |     - Flags = 128
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (20);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.1"));
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x14, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x18, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0x80, 0x03,
      0x0a, 0x00, 0x00, 0x01,
      0x02, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("20", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 21
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 21
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (21);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x15, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x1a, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("21", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 22
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 22
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - Flags = 32
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (22);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x16, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x21, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x02, 0x20, 0x03,
      0x0a, 0x0b, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("22", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 23
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 23
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (23);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));

    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (16);
    msg2a2->PrefixPushBack (24);

    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x17, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x32, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("23", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 24
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 24
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (24);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));

    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (16);
    msg2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> msg2a2tlv1 = Create<PbbAddressTlv> ();
    msg2a2tlv1->SetType (1);
    msg2a2->TlvPushBack (msg2a2tlv1);

    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x18, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x34, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x00,
      0x02, 0x01, 0x00,
    };
    AddTestCase (new PbbTestCase ("24", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 25
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 25
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 64
         * |    |         Index-start = 1
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (25);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));

    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (16);
    msg2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> msg2a2tlv1 = Create<PbbAddressTlv> ();
    msg2a2tlv1->SetType (1);
    msg2a2tlv1->SetIndexStart (1);
    msg2a2->TlvPushBack (msg2a2tlv1);

    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x19, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x35, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x00,
      0x03, 0x01, 0x40, 0x01,
    };
    AddTestCase (new PbbTestCase ("25", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 26
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 26
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 32
         * |    |         Index-start = 1
         * |    |         Index-stop = 3
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (26);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));

    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (16);
    msg2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> msg2a2tlv1 = Create<PbbAddressTlv> ();
    msg2a2tlv1->SetType (1);
    msg2a2tlv1->SetIndexStart (1);
    msg2a2tlv1->SetIndexStop (3);
    msg2a2->TlvPushBack (msg2a2tlv1);

    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x1a, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x36, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x00,
      0x04, 0x01, 0x20, 0x01,
      0x03,
    };
    AddTestCase (new PbbTestCase ("26", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 27
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 27
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 52
         * |    |         Index-start = 1
         * |    |         Index-stop = 3
         * |    |         Type = 1; Value = 01  02  03
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (27);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));

    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (16);
    msg2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> msg2a2tlv1 = Create<PbbAddressTlv> ();
    msg2a2tlv1->SetType (1);
    msg2a2tlv1->SetIndexStart (1);
    msg2a2tlv1->SetIndexStop (3);

    uint8_t value[] = { 1, 2, 3};
    msg2a2tlv1->SetValue (value, sizeof (value));
    msg2a2tlv1->SetMultivalue (true);

    msg2a2->TlvPushBack (msg2a2tlv1);

    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x1b, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x00, 0x3a, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x00,
      0x08, 0x01, 0x34, 0x01,
      0x03, 0x03, 0x01, 0x02,
      0x03,
    };
    AddTestCase (new PbbTestCase ("27", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 28
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 28
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 56
         * |    |         Index-start = 1
         * |    |         Index-stop = 3
         * |    |         Type = 1; Value = 00  01  02  03
         * |    |                           04  05  06  07
         * |    |                           08  09  0a  0b
         * |    |                           0c  0d  0e  0f
         * |    |                           10  11  12  13
         * |    |                           14  15  16  17
         * |    |                           18  19  1a  1b
         * |    |                           1c  1d  1e  1f
         * |    |                           20  21  22  23
         * |    |                           24  25  26  27
         * |    |                           28  29  2a  2b
         * |    |                           2c  2d  2e  2f
         * |    |                           30  31  32  33
         * |    |                           34  35  36  37
         * |    |                           38  39  3a  3b
         * |    |                           3c  3d  3e  3f
         * |    |                           40  41  42  43
         * |    |                           44  45  46  47
         * |    |                           48  49  4a  4b
         * |    |                           4c  4d  4e  4f
         * |    |                           50  51  52  53
         * |    |                           54  55  56  57
         * |    |                           58  59  5a  5b
         * |    |                           5c  5d  5e  5f
         * |    |                           60  61  62  63
         * |    |                           64  65  66  67
         * |    |                           68  69  6a  6b
         * |    |                           6c  6d  6e  6f
         * |    |                           70  71  72  73
         * |    |                           74  75  76  77
         * |    |                           78  79  7a  7b
         * |    |                           7c  7d  7e  7f
         * |    |                           80  81  82  83
         * |    |                           84  85  86  87
         * |    |                           88  89  8a  8b
         * |    |                           8c  8d  8e  8f
         * |    |                           90  91  92  93
         * |    |                           94  95  96  97
         * |    |                           98  99  9a  9b
         * |    |                           9c  9d  9e  9f
         * |    |                           a0  a1  a2  a3
         * |    |                           a4  a5  a6  a7
         * |    |                           a8  a9  aa  ab
         * |    |                           ac  ad  ae  af
         * |    |                           b0  b1  b2  b3
         * |    |                           b4  b5  b6  b7
         * |    |                           b8  b9  ba  bb
         * |    |                           bc  bd  be  bf
         * |    |                           c0  c1  c2  c3
         * |    |                           c4  c5  c6  c7
         * |    |                           c8  c9  ca  cb
         * |    |                           cc  cd  ce  cf
         * |    |                           d0  d1  d2  d3
         * |    |                           d4  d5  d6  d7
         * |    |                           d8  d9  da  db
         * |    |                           dc  dd  de  df
         * |    |                           e0  e1  e2  e3
         * |    |                           e4  e5  e6  e7
         * |    |                           e8  e9  ea  eb
         * |    |                           ec  ed  ee  ef
         * |    |                           f0  f1  f2  f3
         * |    |                           f4  f5  f6  f7
         * |    |                           f8  f9  fa  fb
         * |    |                           fc  fd  fe  00
         * |    |                           01  02  03  04
         * |    |                           05  06  07  08
         * |    |                           09  0a  0b  0c
         * |    |                           0d  0e  0f  10
         * |    |                           11  12  13  14
         * |    |                           15  16  17  18
         * |    |                           19  1a  1b  1c
         * |    |                           1d  1e  1f  20
         * |    |                           21  22  23  24
         * |    |                           25  26  27  28
         * |    |                           29  2a  2b  2c
         * |    |
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (28);

    Ptr<PbbTlv> tlv1 = Create<PbbTlv> ();
    tlv1->SetType (1);
    packet->TlvPushBack (tlv1);

    Ptr<PbbMessageIpv4> msg1 = Create<PbbMessageIpv4> ();
    msg1->SetType (1);

    Ptr<PbbTlv> msg1tlv1 = Create<PbbTlv> ();
    msg1tlv1->SetType (1);
    msg1->TlvPushBack (msg1tlv1);

    packet->MessagePushBack (msg1);

    Ptr<PbbMessageIpv4> msg2 = Create<PbbMessageIpv4> ();
    msg2->SetType (2);
    msg2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    msg2->SetHopLimit (255);
    msg2->SetHopCount (1);
    msg2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> msg2a1 = Create<PbbAddressBlockIpv4> ();
    msg2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    msg2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    msg2->AddressBlockPushBack (msg2a1);

    Ptr<PbbAddressBlockIpv4> msg2a2 = Create<PbbAddressBlockIpv4> ();
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    msg2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));

    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (32);
    msg2a2->PrefixPushBack (16);
    msg2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> msg2a2tlv1 = Create<PbbAddressTlv> ();
    msg2a2tlv1->SetType (1);
    msg2a2tlv1->SetIndexStart (1);
    msg2a2tlv1->SetIndexStop (3);

    uint8_t value[] = {
      0x00, 0x01, 0x02, 0x03,
      0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b,
      0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13,
      0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b,
      0x1c, 0x1d, 0x1e, 0x1f,
      0x20, 0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27,
      0x28, 0x29, 0x2a, 0x2b,
      0x2c, 0x2d, 0x2e, 0x2f,
      0x30, 0x31, 0x32, 0x33,
      0x34, 0x35, 0x36, 0x37,
      0x38, 0x39, 0x3a, 0x3b,
      0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43,
      0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b,
      0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53,
      0x54, 0x55, 0x56, 0x57,
      0x58, 0x59, 0x5a, 0x5b,
      0x5c, 0x5d, 0x5e, 0x5f,
      0x60, 0x61, 0x62, 0x63,
      0x64, 0x65, 0x66, 0x67,
      0x68, 0x69, 0x6a, 0x6b,
      0x6c, 0x6d, 0x6e, 0x6f,
      0x70, 0x71, 0x72, 0x73,
      0x74, 0x75, 0x76, 0x77,
      0x78, 0x79, 0x7a, 0x7b,
      0x7c, 0x7d, 0x7e, 0x7f,
      0x80, 0x81, 0x82, 0x83,
      0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8a, 0x8b,
      0x8c, 0x8d, 0x8e, 0x8f,
      0x90, 0x91, 0x92, 0x93,
      0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9a, 0x9b,
      0x9c, 0x9d, 0x9e, 0x9f,
      0xa0, 0xa1, 0xa2, 0xa3,
      0xa4, 0xa5, 0xa6, 0xa7,
      0xa8, 0xa9, 0xaa, 0xab,
      0xac, 0xad, 0xae, 0xaf,
      0xb0, 0xb1, 0xb2, 0xb3,
      0xb4, 0xb5, 0xb6, 0xb7,
      0xb8, 0xb9, 0xba, 0xbb,
      0xbc, 0xbd, 0xbe, 0xbf,
      0xc0, 0xc1, 0xc2, 0xc3,
      0xc4, 0xc5, 0xc6, 0xc7,
      0xc8, 0xc9, 0xca, 0xcb,
      0xcc, 0xcd, 0xce, 0xcf,
      0xd0, 0xd1, 0xd2, 0xd3,
      0xd4, 0xd5, 0xd6, 0xd7,
      0xd8, 0xd9, 0xda, 0xdb,
      0xdc, 0xdd, 0xde, 0xdf,
      0xe0, 0xe1, 0xe2, 0xe3,
      0xe4, 0xe5, 0xe6, 0xe7,
      0xe8, 0xe9, 0xea, 0xeb,
      0xec, 0xed, 0xee, 0xef,
      0xf0, 0xf1, 0xf2, 0xf3,
      0xf4, 0xf5, 0xf6, 0xf7,
      0xf8, 0xf9, 0xfa, 0xfb,
      0xfc, 0xfd, 0xfe, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c,
    };
    msg2a2tlv1->SetValue (value, sizeof (value));

    msg2a2->TlvPushBack (msg2a2tlv1);

    msg2->AddressBlockPushBack (msg2a2);

    packet->MessagePushBack (msg2);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x1c, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x03, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x01, 0x64, 0x0a,   /* [16] - 0xf0 */
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x01,
      0x32, 0x01, 0x38, 0x01,
      0x03, 0x01, 0x2c, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c,
      0x2d, 0x2e, 0x2f, 0x30,
      0x31, 0x32, 0x33, 0x34,
      0x35, 0x36, 0x37, 0x38,
      0x39, 0x3a, 0x3b, 0x3c,
      0x3d, 0x3e, 0x3f, 0x40,
      0x41, 0x42, 0x43, 0x44,
      0x45, 0x46, 0x47, 0x48,
      0x49, 0x4a, 0x4b, 0x4c,
      0x4d, 0x4e, 0x4f, 0x50,
      0x51, 0x52, 0x53, 0x54,
      0x55, 0x56, 0x57, 0x58,
      0x59, 0x5a, 0x5b, 0x5c,
      0x5d, 0x5e, 0x5f, 0x60,
      0x61, 0x62, 0x63, 0x64,
      0x65, 0x66, 0x67, 0x68,
      0x69, 0x6a, 0x6b, 0x6c,
      0x6d, 0x6e, 0x6f, 0x70,
      0x71, 0x72, 0x73, 0x74,
      0x75, 0x76, 0x77, 0x78,
      0x79, 0x7a, 0x7b, 0x7c,
      0x7d, 0x7e, 0x7f, 0x80,
      0x81, 0x82, 0x83, 0x84,
      0x85, 0x86, 0x87, 0x88,
      0x89, 0x8a, 0x8b, 0x8c,
      0x8d, 0x8e, 0x8f, 0x90,
      0x91, 0x92, 0x93, 0x94,
      0x95, 0x96, 0x97, 0x98,
      0x99, 0x9a, 0x9b, 0x9c,
      0x9d, 0x9e, 0x9f, 0xa0,
      0xa1, 0xa2, 0xa3, 0xa4,
      0xa5, 0xa6, 0xa7, 0xa8,
      0xa9, 0xaa, 0xab, 0xac,
      0xad, 0xae, 0xaf, 0xb0,
      0xb1, 0xb2, 0xb3, 0xb4,
      0xb5, 0xb6, 0xb7, 0xb8,
      0xb9, 0xba, 0xbb, 0xbc,
      0xbd, 0xbe, 0xbf, 0xc0,
      0xc1, 0xc2, 0xc3, 0xc4,
      0xc5, 0xc6, 0xc7, 0xc8,
      0xc9, 0xca, 0xcb, 0xcc,
      0xcd, 0xce, 0xcf, 0xd0,
      0xd1, 0xd2, 0xd3, 0xd4,
      0xd5, 0xd6, 0xd7, 0xd8,
      0xd9, 0xda, 0xdb, 0xdc,
      0xdd, 0xde, 0xdf, 0xe0,
      0xe1, 0xe2, 0xe3, 0xe4,
      0xe5, 0xe6, 0xe7, 0xe8,
      0xe9, 0xea, 0xeb, 0xec,
      0xed, 0xee, 0xef, 0xf0,
      0xf1, 0xf2, 0xf3, 0xf4,
      0xf5, 0xf6, 0xf7, 0xf8,
      0xf9, 0xfa, 0xfb, 0xfc,
      0xfd, 0xfe, 0x00, 0x01,
      0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09,
      0x0a, 0x0b, 0x0c, 0x0d,
      0x0e, 0x0f, 0x10, 0x11,
      0x12, 0x13, 0x14, 0x15,
      0x16, 0x17, 0x18, 0x19,
      0x1a, 0x1b, 0x1c, 0x1d,
      0x1e, 0x1f, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25,
      0x26, 0x27, 0x28, 0x29,
      0x2a, 0x2b, 0x2c
    };
    AddTestCase (new PbbTestCase ("28", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 29
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  1
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x0f, 0x00,
      0x06, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("29", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 30
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);
    m1->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x8f, 0x00,
      0x16, 0xab, 0xcd, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00
    };
    AddTestCase (new PbbTestCase ("30", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 31
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (1 addresses)
         * |    |     - 10::1/128
         * |    |     - Flags = 0
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);
    m1->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m1a1 = Create<PbbAddressBlockIpv6> ();
    m1a1->AddressPushBack (Ipv6Address ("10::1"));
    m1->AddressBlockPushBack (m1a1);

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x8f, 0x00,
      0x2a, 0xab, 0xcd, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x10, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("31", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 32
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (2 addresses)
         * |    |     - 10::1/128
         * |    |     - 10::2/128
         * |    |     - Flags = 128
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);
    m1->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m1a1 = Create<PbbAddressBlockIpv6> ();
    m1a1->AddressPushBack (Ipv6Address ("10::1"));
    m1a1->AddressPushBack (Ipv6Address ("10::2"));
    m1->AddressBlockPushBack (m1a1);

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x8f, 0x00,
      0x2c, 0xab, 0xcd, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x02,
      0x80, 0x0f, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x02, 0x00,
      0x00,
    };
    AddTestCase (new PbbTestCase ("32", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 33
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (2 addresses)
         * |    |     - 10::2/128
         * |    |     - 10::11:2/128
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);
    m1->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m1a1 = Create<PbbAddressBlockIpv6> ();
    m1a1->AddressPushBack (Ipv6Address ("10::2"));
    m1a1->AddressPushBack (Ipv6Address ("10::11:2"));
    m1->AddressBlockPushBack (m1a1);

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x8f, 0x00,
      0x2d, 0xab, 0xcd, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x02,
      0xc0, 0x0d, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x02,
      0x00, 0x02, 0x00, 0x11,
      0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("33", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 34
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (2 addresses)
         * |    |     - 10::2/128
         * |    |     - 10::11:2/128
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (2 addresses)
         * |    |     - 10::/128
         * |    |     - 11::/128
         * |    |     - Flags = 160
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);
    m1->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m1a1 = Create<PbbAddressBlockIpv6> ();
    m1a1->AddressPushBack (Ipv6Address ("10::2"));
    m1a1->AddressPushBack (Ipv6Address ("10::11:2"));
    m1->AddressBlockPushBack (m1a1);

    Ptr<PbbAddressBlockIpv6> m1a2 = Create<PbbAddressBlockIpv6> ();
    m1a2->AddressPushBack (Ipv6Address ("10::"));
    m1a2->AddressPushBack (Ipv6Address ("11::"));
    m1->AddressBlockPushBack (m1a2);

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x8f, 0x00,
      0x36, 0xab, 0xcd, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x02,
      0xc0, 0x0d, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x02,
      0x00, 0x02, 0x00, 0x11,
      0x00, 0x00, 0x02, 0xa0,
      0x01, 0x00, 0x0e, 0x10,
      0x11, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("34", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 35
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  0
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (2 addresses)
         * |    |     - 10::2/128
         * |    |     - 10::11:2/128
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10::/128
         * |    |     - 11::/128
         * |    |     - 10::5/64
         * |    |     - 10::6/48
         * |    |     - Flags = 136
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);
    m1->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m1a1 = Create<PbbAddressBlockIpv6> ();
    m1a1->AddressPushBack (Ipv6Address ("10::2"));
    m1a1->AddressPushBack (Ipv6Address ("10::11:2"));
    m1->AddressBlockPushBack (m1a1);

    Ptr<PbbAddressBlockIpv6> m1a2 = Create<PbbAddressBlockIpv6> ();
    m1a2->AddressPushBack (Ipv6Address ("10::"));
    m1a2->AddressPushBack (Ipv6Address ("11::"));
    m1a2->AddressPushBack (Ipv6Address ("10::5"));
    m1a2->AddressPushBack (Ipv6Address ("10::6"));
    m1a2->PrefixPushBack (128);
    m1a2->PrefixPushBack (128);
    m1a2->PrefixPushBack (64);
    m1a2->PrefixPushBack (48);
    m1->AddressBlockPushBack (m1a2);

    packet->MessagePushBack (m1);

    uint8_t buffer[] = {
      0x00, 0x01, 0x8f, 0x00,
      0x73, 0xab, 0xcd, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x02,
      0xc0, 0x0d, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x02,
      0x00, 0x02, 0x00, 0x11,
      0x00, 0x00, 0x04, 0x88,
      0x01, 0x00, 0x10, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x11, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x05, 0x10,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x06, 0x80, 0x80,
      0x40, 0x30, 0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("35", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 36
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 29
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 56
         * |    |         Index-start = 1
         * |    |         Index-stop = 3
         * |    |         Type = 1; Value = 00  01  02  03
         * |    |                           04  05  06  07
         * |    |                           08  09  0a  0b
         * |    |                           0c  0d  0e  0f
         * |    |                           10  11  12  13
         * |    |                           14  15  16  17
         * |    |                           18  19  1a  1b
         * |    |                           1c  1d  1e  1f
         * |    |                           20  21  22  23
         * |    |                           24  25  26  27
         * |    |                           28  29  2a  2b
         * |    |                           2c  2d  2e  2f
         * |    |                           30  31  32  33
         * |    |                           34  35  36  37
         * |    |                           38  39  3a  3b
         * |    |                           3c  3d  3e  3f
         * |    |                           40  41  42  43
         * |    |                           44  45  46  47
         * |    |                           48  49  4a  4b
         * |    |                           4c  4d  4e  4f
         * |    |                           50  51  52  53
         * |    |                           54  55  56  57
         * |    |                           58  59  5a  5b
         * |    |                           5c  5d  5e  5f
         * |    |                           60  61  62  63
         * |    |                           64  65  66  67
         * |    |                           68  69  6a  6b
         * |    |                           6c  6d  6e  6f
         * |    |                           70  71  72  73
         * |    |                           74  75  76  77
         * |    |                           78  79  7a  7b
         * |    |                           7c  7d  7e  7f
         * |    |                           80  81  82  83
         * |    |                           84  85  86  87
         * |    |                           88  89  8a  8b
         * |    |                           8c  8d  8e  8f
         * |    |                           90  91  92  93
         * |    |                           94  95  96  97
         * |    |                           98  99  9a  9b
         * |    |                           9c  9d  9e  9f
         * |    |                           a0  a1  a2  a3
         * |    |                           a4  a5  a6  a7
         * |    |                           a8  a9  aa  ab
         * |    |                           ac  ad  ae  af
         * |    |                           b0  b1  b2  b3
         * |    |                           b4  b5  b6  b7
         * |    |                           b8  b9  ba  bb
         * |    |                           bc  bd  be  bf
         * |    |                           c0  c1  c2  c3
         * |    |                           c4  c5  c6  c7
         * |    |                           c8  c9  ca  cb
         * |    |                           cc  cd  ce  cf
         * |    |                           d0  d1  d2  d3
         * |    |                           d4  d5  d6  d7
         * |    |                           d8  d9  da  db
         * |    |                           dc  dd  de  df
         * |    |                           e0  e1  e2  e3
         * |    |                           e4  e5  e6  e7
         * |    |                           e8  e9  ea  eb
         * |    |                           ec  ed  ee  ef
         * |    |                           f0  f1  f2  f3
         * |    |                           f4  f5  f6  f7
         * |    |                           f8  f9  fa  fb
         * |    |                           fc  fd  fe  00
         * |    |                           01  02  03  04
         * |    |                           05  06  07  08
         * |    |                           09  0a  0b  0c
         * |    |                           0d  0e  0f  10
         * |    |                           11  12  13  14
         * |    |                           15  16  17  18
         * |    |                           19  1a  1b  1c
         * |    |                           1d  1e  1f  20
         * |    |                           21  22  23  24
         * |    |                           25  26  27  28
         * |    |                           29  2a  2b  2c
         * |    |
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (2 addresses)
         * |    |     - 10::2/128
         * |    |     - 10::11:2/128
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10::/128
         * |    |     - 11::/128
         * |    |     - 10::5/64
         * |    |     - 10::6/48
         * |    |     - Flags = 136
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (29);

    Ptr<PbbTlv> ptlv1 = Create<PbbTlv> ();
    ptlv1->SetType (1);
    packet->TlvPushBack (ptlv1);

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);

    Ptr<PbbTlv> m1tlv1 = Create<PbbTlv> ();
    m1tlv1->SetType (1);
    m1->TlvPushBack (m1tlv1);
    packet->MessagePushBack (m1);

    Ptr<PbbMessageIpv4> m2 = Create<PbbMessageIpv4> ();
    m2->SetType (2);
    m2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    m2->SetHopLimit (255);
    m2->SetHopCount (1);
    m2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> m2a1 = Create<PbbAddressBlockIpv4> ();
    m2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    m2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    m2->AddressBlockPushBack (m2a1);

    Ptr<PbbAddressBlockIpv4> m2a2 = Create<PbbAddressBlockIpv4> ();
    m2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    m2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    m2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    m2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));
    m2a2->PrefixPushBack (32);
    m2a2->PrefixPushBack (32);
    m2a2->PrefixPushBack (16);
    m2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> m2a2tlv1 = Create<PbbAddressTlv> ();
    m2a2tlv1->SetType (1);
    m2a2tlv1->SetIndexStart (1);
    m2a2tlv1->SetIndexStop (3);

    uint8_t value[] = {
      0x00, 0x01, 0x02, 0x03,
      0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b,
      0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13,
      0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b,
      0x1c, 0x1d, 0x1e, 0x1f,
      0x20, 0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27,
      0x28, 0x29, 0x2a, 0x2b,
      0x2c, 0x2d, 0x2e, 0x2f,
      0x30, 0x31, 0x32, 0x33,
      0x34, 0x35, 0x36, 0x37,
      0x38, 0x39, 0x3a, 0x3b,
      0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43,
      0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b,
      0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53,
      0x54, 0x55, 0x56, 0x57,
      0x58, 0x59, 0x5a, 0x5b,
      0x5c, 0x5d, 0x5e, 0x5f,
      0x60, 0x61, 0x62, 0x63,
      0x64, 0x65, 0x66, 0x67,
      0x68, 0x69, 0x6a, 0x6b,
      0x6c, 0x6d, 0x6e, 0x6f,
      0x70, 0x71, 0x72, 0x73,
      0x74, 0x75, 0x76, 0x77,
      0x78, 0x79, 0x7a, 0x7b,
      0x7c, 0x7d, 0x7e, 0x7f,
      0x80, 0x81, 0x82, 0x83,
      0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8a, 0x8b,
      0x8c, 0x8d, 0x8e, 0x8f,
      0x90, 0x91, 0x92, 0x93,
      0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9a, 0x9b,
      0x9c, 0x9d, 0x9e, 0x9f,
      0xa0, 0xa1, 0xa2, 0xa3,
      0xa4, 0xa5, 0xa6, 0xa7,
      0xa8, 0xa9, 0xaa, 0xab,
      0xac, 0xad, 0xae, 0xaf,
      0xb0, 0xb1, 0xb2, 0xb3,
      0xb4, 0xb5, 0xb6, 0xb7,
      0xb8, 0xb9, 0xba, 0xbb,
      0xbc, 0xbd, 0xbe, 0xbf,
      0xc0, 0xc1, 0xc2, 0xc3,
      0xc4, 0xc5, 0xc6, 0xc7,
      0xc8, 0xc9, 0xca, 0xcb,
      0xcc, 0xcd, 0xce, 0xcf,
      0xd0, 0xd1, 0xd2, 0xd3,
      0xd4, 0xd5, 0xd6, 0xd7,
      0xd8, 0xd9, 0xda, 0xdb,
      0xdc, 0xdd, 0xde, 0xdf,
      0xe0, 0xe1, 0xe2, 0xe3,
      0xe4, 0xe5, 0xe6, 0xe7,
      0xe8, 0xe9, 0xea, 0xeb,
      0xec, 0xed, 0xee, 0xef,
      0xf0, 0xf1, 0xf2, 0xf3,
      0xf4, 0xf5, 0xf6, 0xf7,
      0xf8, 0xf9, 0xfa, 0xfb,
      0xfc, 0xfd, 0xfe, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c,
    };
    m2a2tlv1->SetValue (value, sizeof(value));
    m2a2->TlvPushBack (m2a2tlv1);

    m2->AddressBlockPushBack (m2a2);
    packet->MessagePushBack (m2);

    Ptr<PbbMessageIpv6> m3 = Create<PbbMessageIpv6> ();
    m3->SetType (1);
    m3->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m3a1 = Create<PbbAddressBlockIpv6> ();
    m3a1->AddressPushBack (Ipv6Address ("10::2"));
    m3a1->AddressPushBack (Ipv6Address ("10::11:2"));
    m3->AddressBlockPushBack (m3a1);

    Ptr<PbbAddressBlockIpv6> m3a2 = Create<PbbAddressBlockIpv6> ();
    m3a2->AddressPushBack (Ipv6Address ("10::"));
    m3a2->AddressPushBack (Ipv6Address ("11::"));
    m3a2->AddressPushBack (Ipv6Address ("10::5"));
    m3a2->AddressPushBack (Ipv6Address ("10::6"));
    m3a2->PrefixPushBack (128);
    m3a2->PrefixPushBack (128);
    m3a2->PrefixPushBack (64);
    m3a2->PrefixPushBack (48);

    m3->AddressBlockPushBack (m3a2);
    packet->MessagePushBack (m3);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x1d, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x0f, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x01, 0x64, 0x0a,
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x01,
      0x32, 0x01, 0x38, 0x01,
      0x03, 0x01, 0x2c, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c,
      0x2d, 0x2e, 0x2f, 0x30,
      0x31, 0x32, 0x33, 0x34,
      0x35, 0x36, 0x37, 0x38,
      0x39, 0x3a, 0x3b, 0x3c,
      0x3d, 0x3e, 0x3f, 0x40,
      0x41, 0x42, 0x43, 0x44,
      0x45, 0x46, 0x47, 0x48,
      0x49, 0x4a, 0x4b, 0x4c,
      0x4d, 0x4e, 0x4f, 0x50,
      0x51, 0x52, 0x53, 0x54,
      0x55, 0x56, 0x57, 0x58,
      0x59, 0x5a, 0x5b, 0x5c,
      0x5d, 0x5e, 0x5f, 0x60,
      0x61, 0x62, 0x63, 0x64,
      0x65, 0x66, 0x67, 0x68,
      0x69, 0x6a, 0x6b, 0x6c,
      0x6d, 0x6e, 0x6f, 0x70,
      0x71, 0x72, 0x73, 0x74,
      0x75, 0x76, 0x77, 0x78,
      0x79, 0x7a, 0x7b, 0x7c,
      0x7d, 0x7e, 0x7f, 0x80,
      0x81, 0x82, 0x83, 0x84,
      0x85, 0x86, 0x87, 0x88,
      0x89, 0x8a, 0x8b, 0x8c,
      0x8d, 0x8e, 0x8f, 0x90,
      0x91, 0x92, 0x93, 0x94,
      0x95, 0x96, 0x97, 0x98,
      0x99, 0x9a, 0x9b, 0x9c,
      0x9d, 0x9e, 0x9f, 0xa0,
      0xa1, 0xa2, 0xa3, 0xa4,
      0xa5, 0xa6, 0xa7, 0xa8,
      0xa9, 0xaa, 0xab, 0xac,
      0xad, 0xae, 0xaf, 0xb0,
      0xb1, 0xb2, 0xb3, 0xb4,
      0xb5, 0xb6, 0xb7, 0xb8,
      0xb9, 0xba, 0xbb, 0xbc,
      0xbd, 0xbe, 0xbf, 0xc0,
      0xc1, 0xc2, 0xc3, 0xc4,
      0xc5, 0xc6, 0xc7, 0xc8,
      0xc9, 0xca, 0xcb, 0xcc,
      0xcd, 0xce, 0xcf, 0xd0,
      0xd1, 0xd2, 0xd3, 0xd4,
      0xd5, 0xd6, 0xd7, 0xd8,
      0xd9, 0xda, 0xdb, 0xdc,
      0xdd, 0xde, 0xdf, 0xe0,
      0xe1, 0xe2, 0xe3, 0xe4,
      0xe5, 0xe6, 0xe7, 0xe8,
      0xe9, 0xea, 0xeb, 0xec,
      0xed, 0xee, 0xef, 0xf0,
      0xf1, 0xf2, 0xf3, 0xf4,
      0xf5, 0xf6, 0xf7, 0xf8,
      0xf9, 0xfa, 0xfb, 0xfc,
      0xfd, 0xfe, 0x00, 0x01,
      0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09,
      0x0a, 0x0b, 0x0c, 0x0d,
      0x0e, 0x0f, 0x10, 0x11,
      0x12, 0x13, 0x14, 0x15,
      0x16, 0x17, 0x18, 0x19,
      0x1a, 0x1b, 0x1c, 0x1d,
      0x1e, 0x1f, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25,
      0x26, 0x27, 0x28, 0x29,
      0x2a, 0x2b, 0x2c, 0x01,
      0x8f, 0x00, 0x73, 0xab,
      0xcd, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x00,
      0x00, 0x02, 0xc0, 0x0d,
      0x00, 0x10, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x02, 0x00, 0x02,
      0x00, 0x11, 0x00, 0x00,
      0x04, 0x88, 0x01, 0x00,
      0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x11,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x10, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x05, 0x10, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x06,
      0x80, 0x80, 0x40, 0x30,
      0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("36", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }

  /* Test 37
         * ,------------------
         * |  PACKET
         * |------------------
         * | * Packet version:    0
         * | * Packet flags:  12
         * | * Packet seq number: 30
         * |    | * Packet TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  0
         * |    | * Message TLV Block
         * |    |     - TLV
         * |    |         Flags = 0
         * |    |         Type = 1; Value = (warning: parameter is NULL)
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       2
         * |    | * Message flags:  240
         * |    | * Originator address: 10.0.0.1
         * |    | * Hop limit:          255
         * |    | * Hop count:          1
         * |    | * Message seq number: 12345
         * |    | - Address block (2 addresses)
         * |    |     - 10.0.0.2/32
         * |    |     - 10.1.1.2/32
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10.0.0.0/32
         * |    |     - 11.0.0.0/32
         * |    |     - 10.0.0.5/16
         * |    |     - 10.0.0.6/24
         * |    |     - Flags = 8
         * |    | - ADDRESS TLV block (1 TLVs)
         * |    |     - TLV
         * |    |         Flags = 56
         * |    |         Index-start = 1
         * |    |         Index-stop = 3
         * |    |         Type = 1; Value = 00  01  02  03
         * |    |                           04  05  06  07
         * |    |                           08  09  0a  0b
         * |    |                           0c  0d  0e  0f
         * |    |                           10  11  12  13
         * |    |                           14  15  16  17
         * |    |                           18  19  1a  1b
         * |    |                           1c  1d  1e  1f
         * |    |                           20  21  22  23
         * |    |                           24  25  26  27
         * |    |                           28  29  2a  2b
         * |    |                           2c  2d  2e  2f
         * |    |                           30  31  32  33
         * |    |                           34  35  36  37
         * |    |                           38  39  3a  3b
         * |    |                           3c  3d  3e  3f
         * |    |                           40  41  42  43
         * |    |                           44  45  46  47
         * |    |                           48  49  4a  4b
         * |    |                           4c  4d  4e  4f
         * |    |                           50  51  52  53
         * |    |                           54  55  56  57
         * |    |                           58  59  5a  5b
         * |    |                           5c  5d  5e  5f
         * |    |                           60  61  62  63
         * |    |                           64  65  66  67
         * |    |                           68  69  6a  6b
         * |    |                           6c  6d  6e  6f
         * |    |                           70  71  72  73
         * |    |                           74  75  76  77
         * |    |                           78  79  7a  7b
         * |    |                           7c  7d  7e  7f
         * |    |                           80  81  82  83
         * |    |                           84  85  86  87
         * |    |                           88  89  8a  8b
         * |    |                           8c  8d  8e  8f
         * |    |                           90  91  92  93
         * |    |                           94  95  96  97
         * |    |                           98  99  9a  9b
         * |    |                           9c  9d  9e  9f
         * |    |                           a0  a1  a2  a3
         * |    |                           a4  a5  a6  a7
         * |    |                           a8  a9  aa  ab
         * |    |                           ac  ad  ae  af
         * |    |                           b0  b1  b2  b3
         * |    |                           b4  b5  b6  b7
         * |    |                           b8  b9  ba  bb
         * |    |                           bc  bd  be  bf
         * |    |                           c0  c1  c2  c3
         * |    |                           c4  c5  c6  c7
         * |    |                           c8  c9  ca  cb
         * |    |                           cc  cd  ce  cf
         * |    |                           d0  d1  d2  d3
         * |    |                           d4  d5  d6  d7
         * |    |                           d8  d9  da  db
         * |    |                           dc  dd  de  df
         * |    |                           e0  e1  e2  e3
         * |    |                           e4  e5  e6  e7
         * |    |                           e8  e9  ea  eb
         * |    |                           ec  ed  ee  ef
         * |    |                           f0  f1  f2  f3
         * |    |                           f4  f5  f6  f7
         * |    |                           f8  f9  fa  fb
         * |    |                           fc  fd  fe  00
         * |    |                           01  02  03  04
         * |    |                           05  06  07  08
         * |    |                           09  0a  0b  0c
         * |    |                           0d  0e  0f  10
         * |    |                           11  12  13  14
         * |    |                           15  16  17  18
         * |    |                           19  1a  1b  1c
         * |    |                           1d  1e  1f  20
         * |    |                           21  22  23  24
         * |    |                           25  26  27  28
         * |    |                           29  2a  2b  2c
         * |    |
         * |    `-------------------
         * |
         * |    ,-------------------
         * |    |  MESSAGE
         * |    |-------------------
         * |    | * Message type:       1
         * |    | * Message flags:  129
         * |    | * Originator address: abcd::1
         * |    | - Address block (2 addresses)
         * |    |     - 10::2/128
         * |    |     - 10::11:2/128
         * |    |     - Flags = 192
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    | - Address block (4 addresses)
         * |    |     - 10::/128
         * |    |     - 11::/128
         * |    |     - 10::5/64
         * |    |     - 10::6/48
         * |    |     - Flags = 136
         * |    | - ADDRESS TLV block (0 TLVs)
         * |    `-------------------
         * |
         * `------------------
   */
  {
    Ptr<PbbPacket> packet = Create<PbbPacket> ();
    packet->SetSequenceNumber (30);

    Ptr<PbbTlv> ptlv1 = Create<PbbTlv> ();
    ptlv1->SetType (1);
    packet->TlvPushBack (ptlv1);

    Ptr<PbbMessageIpv6> m1 = Create<PbbMessageIpv6> ();
    m1->SetType (1);

    Ptr<PbbTlv> m1tlv1 = Create<PbbTlv> ();
    m1tlv1->SetType (1);
    m1->TlvPushBack (m1tlv1);
    packet->MessagePushBack (m1);

    Ptr<PbbMessageIpv4> m2 = Create<PbbMessageIpv4> ();
    m2->SetType (2);
    m2->SetOriginatorAddress (Ipv4Address ("10.0.0.1"));
    m2->SetHopLimit (255);
    m2->SetHopCount (1);
    m2->SetSequenceNumber (12345);

    Ptr<PbbAddressBlockIpv4> m2a1 = Create<PbbAddressBlockIpv4> ();
    m2a1->AddressPushBack (Ipv4Address ("10.0.0.2"));
    m2a1->AddressPushBack (Ipv4Address ("10.1.1.2"));
    m2->AddressBlockPushBack (m2a1);

    Ptr<PbbAddressBlockIpv4> m2a2 = Create<PbbAddressBlockIpv4> ();
    m2a2->AddressPushBack (Ipv4Address ("10.0.0.0"));
    m2a2->AddressPushBack (Ipv4Address ("11.0.0.0"));
    m2a2->AddressPushBack (Ipv4Address ("10.0.0.5"));
    m2a2->AddressPushBack (Ipv4Address ("10.0.0.6"));
    m2a2->PrefixPushBack (32);
    m2a2->PrefixPushBack (32);
    m2a2->PrefixPushBack (16);
    m2a2->PrefixPushBack (24);

    Ptr<PbbAddressTlv> m2a2tlv1 = Create<PbbAddressTlv> ();
    m2a2tlv1->SetType (1);
    m2a2tlv1->SetIndexStart (1);
    m2a2tlv1->SetIndexStop (3);

    uint8_t value[] = {
      0x00, 0x01, 0x02, 0x03,
      0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b,
      0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13,
      0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b,
      0x1c, 0x1d, 0x1e, 0x1f,
      0x20, 0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27,
      0x28, 0x29, 0x2a, 0x2b,
      0x2c, 0x2d, 0x2e, 0x2f,
      0x30, 0x31, 0x32, 0x33,
      0x34, 0x35, 0x36, 0x37,
      0x38, 0x39, 0x3a, 0x3b,
      0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43,
      0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b,
      0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53,
      0x54, 0x55, 0x56, 0x57,
      0x58, 0x59, 0x5a, 0x5b,
      0x5c, 0x5d, 0x5e, 0x5f,
      0x60, 0x61, 0x62, 0x63,
      0x64, 0x65, 0x66, 0x67,
      0x68, 0x69, 0x6a, 0x6b,
      0x6c, 0x6d, 0x6e, 0x6f,
      0x70, 0x71, 0x72, 0x73,
      0x74, 0x75, 0x76, 0x77,
      0x78, 0x79, 0x7a, 0x7b,
      0x7c, 0x7d, 0x7e, 0x7f,
      0x80, 0x81, 0x82, 0x83,
      0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8a, 0x8b,
      0x8c, 0x8d, 0x8e, 0x8f,
      0x90, 0x91, 0x92, 0x93,
      0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9a, 0x9b,
      0x9c, 0x9d, 0x9e, 0x9f,
      0xa0, 0xa1, 0xa2, 0xa3,
      0xa4, 0xa5, 0xa6, 0xa7,
      0xa8, 0xa9, 0xaa, 0xab,
      0xac, 0xad, 0xae, 0xaf,
      0xb0, 0xb1, 0xb2, 0xb3,
      0xb4, 0xb5, 0xb6, 0xb7,
      0xb8, 0xb9, 0xba, 0xbb,
      0xbc, 0xbd, 0xbe, 0xbf,
      0xc0, 0xc1, 0xc2, 0xc3,
      0xc4, 0xc5, 0xc6, 0xc7,
      0xc8, 0xc9, 0xca, 0xcb,
      0xcc, 0xcd, 0xce, 0xcf,
      0xd0, 0xd1, 0xd2, 0xd3,
      0xd4, 0xd5, 0xd6, 0xd7,
      0xd8, 0xd9, 0xda, 0xdb,
      0xdc, 0xdd, 0xde, 0xdf,
      0xe0, 0xe1, 0xe2, 0xe3,
      0xe4, 0xe5, 0xe6, 0xe7,
      0xe8, 0xe9, 0xea, 0xeb,
      0xec, 0xed, 0xee, 0xef,
      0xf0, 0xf1, 0xf2, 0xf3,
      0xf4, 0xf5, 0xf6, 0xf7,
      0xf8, 0xf9, 0xfa, 0xfb,
      0xfc, 0xfd, 0xfe, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c,
    };
    m2a2tlv1->SetValue (value, sizeof(value));
    m2a2->TlvPushBack (m2a2tlv1);

    m2->AddressBlockPushBack (m2a2);
    packet->MessagePushBack (m2);

    Ptr<PbbMessageIpv6> m3 = Create<PbbMessageIpv6> ();
    m3->SetType (1);
    m3->SetOriginatorAddress (Ipv6Address ("abcd::1"));

    Ptr<PbbAddressBlockIpv6> m3a1 = Create<PbbAddressBlockIpv6> ();
    m3a1->AddressPushBack (Ipv6Address ("10::2"));
    m3a1->AddressPushBack (Ipv6Address ("10::11:2"));
    m3->AddressBlockPushBack (m3a1);

    Ptr<PbbAddressBlockIpv6> m3a2 = Create<PbbAddressBlockIpv6> ();
    m3a2->AddressPushBack (Ipv6Address ("10::"));
    m3a2->AddressPushBack (Ipv6Address ("11::"));
    m3a2->AddressPushBack (Ipv6Address ("10::5"));
    m3a2->AddressPushBack (Ipv6Address ("10::6"));
    m3a2->PrefixPushBack (128);
    m3a2->PrefixPushBack (128);
    m3a2->PrefixPushBack (64);
    m3a2->PrefixPushBack (48);

    m3->AddressBlockPushBack (m3a2);
    packet->MessagePushBack (m3);

    uint8_t buffer[] = {
      0x0c, 0x00, 0x1e, 0x00,
      0x02, 0x01, 0x00, 0x01,
      0x0f, 0x00, 0x08, 0x00,
      0x02, 0x01, 0x00, 0x02,
      0xf3, 0x01, 0x64, 0x0a,
      0x00, 0x00, 0x01, 0xff,
      0x01, 0x30, 0x39, 0x00,
      0x00, 0x02, 0xc0, 0x01,
      0x0a, 0x01, 0x02, 0x00,
      0x00, 0x01, 0x01, 0x00,
      0x00, 0x04, 0x08, 0x0a,
      0x00, 0x00, 0x00, 0x0b,
      0x00, 0x00, 0x00, 0x0a,
      0x00, 0x00, 0x05, 0x0a,
      0x00, 0x00, 0x06, 0x20,
      0x20, 0x10, 0x18, 0x01,
      0x32, 0x01, 0x38, 0x01,
      0x03, 0x01, 0x2c, 0x00,
      0x01, 0x02, 0x03, 0x04,
      0x05, 0x06, 0x07, 0x08,
      0x09, 0x0a, 0x0b, 0x0c,
      0x0d, 0x0e, 0x0f, 0x10,
      0x11, 0x12, 0x13, 0x14,
      0x15, 0x16, 0x17, 0x18,
      0x19, 0x1a, 0x1b, 0x1c,
      0x1d, 0x1e, 0x1f, 0x20,
      0x21, 0x22, 0x23, 0x24,
      0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x2b, 0x2c,
      0x2d, 0x2e, 0x2f, 0x30,
      0x31, 0x32, 0x33, 0x34,
      0x35, 0x36, 0x37, 0x38,
      0x39, 0x3a, 0x3b, 0x3c,
      0x3d, 0x3e, 0x3f, 0x40,
      0x41, 0x42, 0x43, 0x44,
      0x45, 0x46, 0x47, 0x48,
      0x49, 0x4a, 0x4b, 0x4c,
      0x4d, 0x4e, 0x4f, 0x50,
      0x51, 0x52, 0x53, 0x54,
      0x55, 0x56, 0x57, 0x58,
      0x59, 0x5a, 0x5b, 0x5c,
      0x5d, 0x5e, 0x5f, 0x60,
      0x61, 0x62, 0x63, 0x64,
      0x65, 0x66, 0x67, 0x68,
      0x69, 0x6a, 0x6b, 0x6c,
      0x6d, 0x6e, 0x6f, 0x70,
      0x71, 0x72, 0x73, 0x74,
      0x75, 0x76, 0x77, 0x78,
      0x79, 0x7a, 0x7b, 0x7c,
      0x7d, 0x7e, 0x7f, 0x80,
      0x81, 0x82, 0x83, 0x84,
      0x85, 0x86, 0x87, 0x88,
      0x89, 0x8a, 0x8b, 0x8c,
      0x8d, 0x8e, 0x8f, 0x90,
      0x91, 0x92, 0x93, 0x94,
      0x95, 0x96, 0x97, 0x98,
      0x99, 0x9a, 0x9b, 0x9c,
      0x9d, 0x9e, 0x9f, 0xa0,
      0xa1, 0xa2, 0xa3, 0xa4,
      0xa5, 0xa6, 0xa7, 0xa8,
      0xa9, 0xaa, 0xab, 0xac,
      0xad, 0xae, 0xaf, 0xb0,
      0xb1, 0xb2, 0xb3, 0xb4,
      0xb5, 0xb6, 0xb7, 0xb8,
      0xb9, 0xba, 0xbb, 0xbc,
      0xbd, 0xbe, 0xbf, 0xc0,
      0xc1, 0xc2, 0xc3, 0xc4,
      0xc5, 0xc6, 0xc7, 0xc8,
      0xc9, 0xca, 0xcb, 0xcc,
      0xcd, 0xce, 0xcf, 0xd0,
      0xd1, 0xd2, 0xd3, 0xd4,
      0xd5, 0xd6, 0xd7, 0xd8,
      0xd9, 0xda, 0xdb, 0xdc,
      0xdd, 0xde, 0xdf, 0xe0,
      0xe1, 0xe2, 0xe3, 0xe4,
      0xe5, 0xe6, 0xe7, 0xe8,
      0xe9, 0xea, 0xeb, 0xec,
      0xed, 0xee, 0xef, 0xf0,
      0xf1, 0xf2, 0xf3, 0xf4,
      0xf5, 0xf6, 0xf7, 0xf8,
      0xf9, 0xfa, 0xfb, 0xfc,
      0xfd, 0xfe, 0x00, 0x01,
      0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09,
      0x0a, 0x0b, 0x0c, 0x0d,
      0x0e, 0x0f, 0x10, 0x11,
      0x12, 0x13, 0x14, 0x15,
      0x16, 0x17, 0x18, 0x19,
      0x1a, 0x1b, 0x1c, 0x1d,
      0x1e, 0x1f, 0x20, 0x21,
      0x22, 0x23, 0x24, 0x25,
      0x26, 0x27, 0x28, 0x29,
      0x2a, 0x2b, 0x2c, 0x01,
      0x8f, 0x00, 0x73, 0xab,
      0xcd, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x00,
      0x00, 0x02, 0xc0, 0x0d,
      0x00, 0x10, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x02, 0x00, 0x02,
      0x00, 0x11, 0x00, 0x00,
      0x04, 0x88, 0x01, 0x00,
      0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x11,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x10, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x05, 0x10, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x06,
      0x80, 0x80, 0x40, 0x30,
      0x00, 0x00,
    };
    AddTestCase (new PbbTestCase ("37", packet, buffer, sizeof(buffer)), TestCase::QUICK);
  }
}

static PbbTestSuite pbbTestSuite; //!< Static variable for test initialization
