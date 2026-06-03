/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#include "box-line-intersection-test.h"
#include <ns3/simulator.h>

using namespace ns3;

/**
 * This TestSuite tests the intersection of a line segment
 * between two 3D positions with a 3D box. It generates two
 * positions from a set of predefined positions (see GeneratePosition method),
 * and then tests the intersection of a line segments between them with a box
 * of predefined dimensions.
 */


BoxLineIntersectionTestSuite::BoxLineIntersectionTestSuite ()
  : TestSuite ("box-line-intersection", UNIT)
{
  // Box in the positive x-plane to check the intersection with.
  Box box = Box (890.0, 990.0, 840.0, 870.0, 0.0, 6.0);
  bool intersect;
  // Test #1 :
  // pos1 (index 3) is outside the box and below the height of the box.
  // pos2 (index 6) is outside the box and above the height of the box.
  // Expected result: No intersection. The box is between the two position,
  // however, pos2 is above the height of the box.
  intersect = false;
  AddTestCase (new BoxLineIntersectionTestCase (3, 6, box, intersect), TestCase::QUICK);

  // Test #2 :
  // pos1 (index 1) is inside the box.
  // pos2 (index 2) is inside the box.
  // Expected result: Intersection.
  intersect = true;
  AddTestCase (new BoxLineIntersectionTestCase (1, 2, box, intersect), TestCase::QUICK);

  // Test #3 :
  // pos1 (index 3) is outside the box.
  // pos2 (index 1) is inside the box.
  // Expected result: Intersection.
  intersect = true;
  AddTestCase (new BoxLineIntersectionTestCase (3, 1, box, intersect), TestCase::QUICK);

  // Test #4:
  // pos1 (index 4) is outside the box.
  // pos2 (index 5) is outside the box.
  // Expected result: Intersection because box is in between the two positions.
  intersect = true;
  AddTestCase (new BoxLineIntersectionTestCase (4, 5, box, intersect), TestCase::QUICK);
}

static BoxLineIntersectionTestSuite boxLineIntersectionTestSuite; //!< boxLineIntersectionTestSuite


/**
 * TestCase
 */

BoxLineIntersectionTestCase::BoxLineIntersectionTestCase (uint16_t indexPos1, uint16_t indexPos2, Box box, bool intersect)
  : TestCase (BuildNameString (indexPos1, indexPos2, box, intersect)),
  m_indexPos1 (indexPos1),
  m_indexPos2 (indexPos2),
  m_box (box),
  m_intersect (intersect)
{
}

BoxLineIntersectionTestCase::~BoxLineIntersectionTestCase ()
{
}

std::string
BoxLineIntersectionTestCase::BuildNameString (uint16_t indexPos1, uint16_t indexPos2, Box box, bool intersect)
{
  std::ostringstream oss;
  oss << "Box line intersection test : checking"
      << " pos1 index " << indexPos1 << " and pos2 index " << indexPos2
      << " intersection with the box (" << box.xMin << ", " << box.xMax
      << ", " << box.yMin << ", " << box.yMax
      << ", " << box.zMin << ", " << box.zMax
      << "). The expected intersection flag = " << intersect << "  ";
  return oss.str ();
}

void
BoxLineIntersectionTestCase::DoRun (void)
{

  Vector pos1 = CreatePosition (m_indexPos1, (m_box.zMax - m_box.zMin));
  Vector pos2 = CreatePosition (m_indexPos2, (m_box.zMax - m_box.zMin));

  bool intersect = m_box.IsIntersect (pos1, pos2);

  NS_TEST_ASSERT_MSG_EQ (intersect, m_intersect, "Unexpected result of box and line segment intersection!");
  Simulator::Destroy ();
}

Vector
BoxLineIntersectionTestCase::CreatePosition (uint16_t index, double boxHeight)
{
  Vector pos;

  switch (index)
    {
    case 1:
      pos = Vector (934.0, 852.0, 1.5);
      break;
    case 2:
      pos = Vector (931.0,861.0, 1.5);
      break;
    case 3:
      pos = Vector (484.0, 298.0, 1.5);
      break;
    case 4:
      pos = Vector (1000.0, 850.0, 1.5);
      break;
    case 5:
      pos = Vector (850.0, 850.0, 1.5);
      break;
    case 6:
      pos = Vector (934.0, 852.0, boxHeight + 14.0);
      break;
    default:
      NS_FATAL_ERROR ("Unknown position index");
      break;
    }

  return pos;
}
