/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
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
 */

#include "ns3/test.h"
#include "ns3/core-module.h"
#include "ns3/tcp-option.h"
#include "ns3/tcp-option-winscale.h"
#include "ns3/tcp-option-ts.h"

#include <string.h>

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP Window Scaling option Test
 */
class TcpOptionWSTestCase : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param name Test description.
   * \param scale Window scaling.
   */
  TcpOptionWSTestCase (std::string name, uint8_t scale);

  /**
   * \brief Serialization test.
   */
  void TestSerialize ();
  /**
   * \brief Deserialization test.
   */
  void TestDeserialize ();

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  uint8_t m_scale;  //!< Window scaling.
  Buffer m_buffer;  //!< Buffer.
};


TcpOptionWSTestCase::TcpOptionWSTestCase (std::string name, uint8_t scale)
  : TestCase (name)
{
  m_scale = scale;
}

void
TcpOptionWSTestCase::DoRun ()
{
  TestSerialize ();
  TestDeserialize ();
}

void
TcpOptionWSTestCase::TestSerialize ()
{
  TcpOptionWinScale opt;

  opt.SetScale (m_scale);
  NS_TEST_EXPECT_MSG_EQ (m_scale, opt.GetScale (), "Scale isn't saved correctly");

  m_buffer.AddAtStart (opt.GetSerializedSize ());

  opt.Serialize (m_buffer.Begin ());
}

void
TcpOptionWSTestCase::TestDeserialize ()
{
  TcpOptionWinScale opt;

  Buffer::Iterator start = m_buffer.Begin ();
  uint8_t kind = start.PeekU8 ();

  NS_TEST_EXPECT_MSG_EQ (kind, TcpOption::WINSCALE, "Different kind found");

  opt.Deserialize (start);

  NS_TEST_EXPECT_MSG_EQ (m_scale, opt.GetScale (), "Different scale found");
}

void
TcpOptionWSTestCase::DoTeardown ()
{
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP TimeStamp option Test
 */
class TcpOptionTSTestCase : public TestCase
{
public:

  /**
   * \brief Constructor.
   * \param name Test description.
   */
  TcpOptionTSTestCase (std::string name);

  /**
   * \brief Serialization test.
   */
  void TestSerialize ();
  /**
   * \brief Deserialization test.
   */
  void TestDeserialize ();

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  uint32_t m_timestamp; //!< TimeStamp.
  uint32_t m_echo;      //!< Echoed TimeStamp.
  Buffer m_buffer;      //!< Buffer.
};


TcpOptionTSTestCase::TcpOptionTSTestCase (std::string name)
  : TestCase (name)
{
  m_timestamp = 0;
  m_echo = 0;
}

void
TcpOptionTSTestCase::DoRun ()
{
  Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();

  for (uint32_t i = 0; i < 1000; ++i)
    {
      m_timestamp = x->GetInteger ();
      m_echo = x->GetInteger ();
      TestSerialize ();
      TestDeserialize ();
    }
}

void
TcpOptionTSTestCase::TestSerialize ()
{
  TcpOptionTS opt;

  opt.SetTimestamp (m_timestamp);
  opt.SetEcho (m_echo);

  NS_TEST_EXPECT_MSG_EQ (m_timestamp, opt.GetTimestamp (), "TS isn't saved correctly");
  NS_TEST_EXPECT_MSG_EQ (m_echo, opt.GetEcho (), "echo isn't saved correctly");

  m_buffer.AddAtStart (opt.GetSerializedSize ());

  opt.Serialize (m_buffer.Begin ());
}

void
TcpOptionTSTestCase::TestDeserialize ()
{
  TcpOptionTS opt;

  Buffer::Iterator start = m_buffer.Begin ();
  uint8_t kind = start.PeekU8 ();

  NS_TEST_EXPECT_MSG_EQ (kind, TcpOption::TS, "Different kind found");

  opt.Deserialize (start);

  NS_TEST_EXPECT_MSG_EQ (m_timestamp, opt.GetTimestamp (), "Different TS found");
  NS_TEST_EXPECT_MSG_EQ (m_echo, opt.GetEcho (), "Different echo found");
}

void
TcpOptionTSTestCase::DoTeardown ()
{
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP options TestSuite
 */
class TcpOptionTestSuite : public TestSuite
{
public:
  TcpOptionTestSuite ()
    : TestSuite ("tcp-option", UNIT)
  {
    for (uint8_t i = 0; i < 15; ++i)
      {
        AddTestCase (new TcpOptionWSTestCase ("Testing window scale value", i), TestCase::QUICK);
      }
    AddTestCase (new TcpOptionTSTestCase ("Testing serialization of random values for timestamp"), TestCase::QUICK);
  }

};

static TcpOptionTestSuite g_TcpOptionTestSuite; //!< Static variable for test initialization
