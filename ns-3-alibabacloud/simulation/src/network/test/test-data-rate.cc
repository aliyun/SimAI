/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
 * Author: Greg Steinbrecher <grs@fb.com>
 */

#include "ns3/data-rate.h"
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simulator.h"

using namespace ns3;

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test Data rate
 *
 */
class DataRateTestCase : public TestCase
{
public:
  /**
   * Constructor
   * \param name test name
   */
  DataRateTestCase (std::string name);
  virtual ~DataRateTestCase ();

  /**
   * Checks if two time values are equal
   * \param t1 first time to check
   * \param t2 second time to check
   * \param msg check output message
   */
  void CheckTimesEqual (Time t1, Time t2, const std::string msg);
  /**
   * Checks if two data rates values are equal
   * \param d1 first data rate to check
   * \param d2 second data rate to check
   * \param msg check output message
   */
  void CheckDataRateEqual (DataRate d1, DataRate d2, const std::string msg);

protected:
  virtual void DoRun (void) = 0;
};

DataRateTestCase::DataRateTestCase (std::string name) : TestCase (name)
{
}

DataRateTestCase::~DataRateTestCase ()
{
}

void
DataRateTestCase::CheckTimesEqual (Time actual, Time correct, const std::string msg)
{
  int64x64_t actualFemtos = actual.GetFemtoSeconds ();
  int64x64_t correctFemtos = correct.GetFemtoSeconds ();
  NS_TEST_EXPECT_MSG_EQ (actualFemtos, correctFemtos, msg);
}

void
DataRateTestCase::CheckDataRateEqual (DataRate d1, DataRate d2, const std::string msg)
{
  NS_TEST_EXPECT_MSG_EQ (d1, d2, msg);
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test Data rate
 *
 */
class DataRateTestCase1 : public DataRateTestCase
{
public:
  DataRateTestCase1 ();

  /**
   * Checks that a given number of bits, at a specified datarate, are
   * corresponding to a given time
   * \param rate the DataRate
   * \param nBits number of bits
   * \param correctTime expected time
   */
  void SingleTest (std::string rate, size_t nBits, Time correctTime);

private:
  virtual void DoRun (void);
};

DataRateTestCase1::DataRateTestCase1 ()
    : DataRateTestCase ("Test rounding of conversion from DataRate to time")
{
}

void
DataRateTestCase1::SingleTest (std::string rate, size_t nBits, Time correctTime)
{
  DataRate dr (rate);
  Time bitsTime = dr.CalculateBitsTxTime (nBits);
  CheckTimesEqual (bitsTime, correctTime, "CalculateBitsTxTime returned incorrect value");
  if ((nBits % 8) == 0)
    {
      Time bytesTime = dr.CalculateBytesTxTime (nBits / 8);
      CheckTimesEqual (bytesTime, correctTime, "CalculateBytesTxTime returned incorrect value");
    }
}

void
DataRateTestCase1::DoRun ()
{
  if (Time::GetResolution () != Time::FS)
    {
      Time::SetResolution (Time::FS);
    }
  SingleTest ("1GB/s", 512, Time (NanoSeconds (64)));
  SingleTest ("8Gb/s", 512, Time (NanoSeconds (64)));
  SingleTest ("1Gb/s", 512, Time (NanoSeconds (512)));
  SingleTest ("8GB/s", 512, Time (NanoSeconds (8)));
  size_t nBits;
  for (nBits = 0; nBits <= 512; nBits++)
    {
      SingleTest ("1Mb/s", nBits, Time (MicroSeconds (nBits)));
      SingleTest ("10Mb/s", nBits, Time (NanoSeconds (nBits * 100)));
      SingleTest ("100Mb/s", nBits, Time (NanoSeconds (nBits * 10)));
      SingleTest ("1Gb/s", nBits, Time (NanoSeconds (nBits)));
      SingleTest ("10Gb/s", nBits, Time (PicoSeconds (nBits * 100)));
      SingleTest ("25Gb/s", nBits, Time (PicoSeconds (nBits * 40)));
      SingleTest ("40Gb/s", nBits, Time (PicoSeconds (nBits * 25)));
      SingleTest ("100Gb/s", nBits, Time (PicoSeconds (nBits * 10)));
      SingleTest ("200Gb/s", nBits, Time (PicoSeconds (nBits * 5)));
      SingleTest ("400Gb/s", nBits, Time (FemtoSeconds (nBits * 2500)));
    }
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test Data rate
 *
 */
class DataRateTestCase2 : public DataRateTestCase
{
public:
  DataRateTestCase2 (); 
  /**
   * Checks data rate addition
   * \param rate1 first data rate
   * \param rate2 second data rate
   * \param rate3 third data rate (first plus second)
   */
  void AdditionTest (std::string rate1, std::string rate2, std::string rate3);
  /**
   * Checks data rate subtraction
   * \param rate1 first data rate
   * \param rate2 second data rate
   * \param rate3 third data rate (first minus second)
   */
  void SubtractionTest (std::string rate1, std::string rate2, std::string rate3);
  /**
   * Checks data rate integer multiplication
   * \param rate1 first data rate
   * \param factor multiplication factor
   * \param rate2 second data rate  (first multiplied by factor)
   */
  void MultiplicationIntTest (std::string rate1, uint64_t factor, std::string rate2);
  /**
   * Checks data rate floating point multiplication
   * \param rate1 first data rate
   * \param factor multiplication factor
   * \param rate2 second data rate  (first multiplied by factor)
   */
  void MultiplicationDoubleTest (std::string rate1, double factor, std::string rate2);

private:
  virtual void DoRun (void); 
};

DataRateTestCase2::DataRateTestCase2 ()
    : DataRateTestCase ("Test arithmatic on DateRate")
{
}

void
DataRateTestCase2::AdditionTest (std::string rate1, std::string rate2, std::string rate3)
{
  DataRate dr1 (rate1);
  DataRate dr2 (rate2);
  DataRate dr3 (rate3);

  CheckDataRateEqual(dr1 + dr2, dr3, "DataRate Addition returned incorrect value");
  
  dr1 += dr2;
  CheckDataRateEqual(dr1, dr3, "DataRate Addition returned incorrect value");
}

void
DataRateTestCase2::SubtractionTest (std::string rate1, std::string rate2, std::string rate3)
{
  DataRate dr1 (rate1);
  DataRate dr2 (rate2);
  DataRate dr3 (rate3);

  CheckDataRateEqual(dr1 - dr2, dr3, "DataRate Subtraction returned incorrect value");

  dr1 -= dr2;
  CheckDataRateEqual(dr1, dr3, "DataRate Subtraction returned incorrect value");
}

void
DataRateTestCase2::MultiplicationIntTest (std::string rate1, uint64_t factor, std::string rate2)
{
  DataRate dr1 (rate1);
  DataRate dr2 (rate2);

  CheckDataRateEqual(dr1 * factor, dr2, "DataRate Multiplication with Int returned incorrect value");

  dr1 *= factor;
  CheckDataRateEqual(dr1, dr2, "DataRate Multiplication with Int returned incorrect value");
}

void
DataRateTestCase2::MultiplicationDoubleTest (std::string rate1, double factor, std::string rate2)
{
  DataRate dr1 (rate1);
  DataRate dr2 (rate2);

  CheckDataRateEqual(dr1 * factor, dr2, "DataRate Multiplication with Double returned incorrect value");

  dr1 *= factor;
  CheckDataRateEqual(dr1, dr2, "DataRate Multiplication with Double returned incorrect value");
}

void
DataRateTestCase2::DoRun ()
{
  AdditionTest("1Mb/s", "3Mb/s", "4Mb/s");
  AdditionTest("1Gb/s", "1b/s", "1000000001b/s");
  SubtractionTest("1Mb/s", "1b/s", "999999b/s");
  SubtractionTest("2Gb/s", "2Gb/s", "0Gb/s");
  MultiplicationIntTest("5Gb/s", 2, "10Gb/s");
  MultiplicationIntTest("4Mb/s", 1000, "4Gb/s");
  MultiplicationDoubleTest("1Gb/s", 0.001, "1Mb/s");
  MultiplicationDoubleTest("6Gb/s", 1.0/7.0, "857142857.14b/s");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief DataRate TestSuite
 */
class DataRateTestSuite : public TestSuite
{
public:
  DataRateTestSuite ();
};

DataRateTestSuite::DataRateTestSuite () : TestSuite ("data-rate", UNIT)
{
  AddTestCase (new DataRateTestCase1 (), TestCase::QUICK);
  AddTestCase (new DataRateTestCase2 (), TestCase::QUICK);
}

static DataRateTestSuite sDataRateTestSuite; //!< Static variable for test initialization
