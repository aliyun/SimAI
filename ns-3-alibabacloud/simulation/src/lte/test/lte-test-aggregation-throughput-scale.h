/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Alexander Krotov
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 *
 */

#ifndef LTE_AGGREGATION_THROUGHPUT_SCALE_H
#define LTE_AGGREGATION_THROUGHPUT_SCALE_H

#include <ns3/test.h>

using namespace ns3;

/**
 * \brief Test suite for executing carrier aggregation throughput scaling test case.
 *
 * \sa ns3::LteAggregationThroughputScaleTestCase
 */
class LteAggregationThroughputScaleTestSuite : public TestSuite
{
public:
  LteAggregationThroughputScaleTestSuite ();
};

/**
 * \ingroup lte
 *
 * \brief Testing that UE throughput scales linearly with number of component carriers.
 *        Also attaches UE to last component carrier to make sure no code assumes
 *        that primary carrier is first.
 */
class LteAggregationThroughputScaleTestCase : public TestCase
{
public:
  /**
   * \brief Creates an instance of the carrier aggregation throughput scaling test case.
   * \param name name of this test
   */
  LteAggregationThroughputScaleTestCase (std::string name);

  virtual ~LteAggregationThroughputScaleTestCase ();

private:
  /**
   * \brief Setup the simulation, run it, and verify the result.
   */
  virtual void DoRun ();

  /**
   * \brief Get throughput function
   *
   * \param numberOfComponentCarriers Number of component carriers
   * \return The total data received (in Megabits)
   */
  double GetThroughput (uint8_t numberOfComponentCarriers);

  uint16_t m_expectedCellId; ///< Cell ID UE is expected to attach to
  uint16_t m_actualCellId; ///< Cell ID UE has attached to
};

#endif /* LTE_AGGREGATION_THROUGHPUT_SCALE_H */
