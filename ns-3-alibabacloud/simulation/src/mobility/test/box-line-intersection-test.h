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

#ifndef BOX_LINE_INTERSECTION_TEST_H
#define BOX_LINE_INTERSECTION_TEST_H

#include <ns3/test.h>
#include <ns3/vector.h>
#include <ns3/box.h>


using namespace ns3;

/**
 * \brief Test suite for
 *
 * \sa ns3::BoxLineIntersectionTestSuite
 */
class BoxLineIntersectionTestSuite : public TestSuite
{
public:
  /**
   * Constructor
   */
  BoxLineIntersectionTestSuite ();
};

/**
 * \brief TestCase to check the box line intersection
 */
class BoxLineIntersectionTestCase : public TestCase
{
public:
  /**
   * \brief Create BoxLineIntersectionTestCase
   * \param indexPos1 Index of the first position to generate
   * \param indexPos2 Index of the second position to generate
   * \param box The 3D box
   * \param intersect The expected result of the test. True for intersection,
   *        false otherwise
   */
  BoxLineIntersectionTestCase (uint16_t indexPos1, uint16_t indexPos2, Box box, bool intersect);
  /**
   * \brief Builds the test name string based on provided parameter values
   * \param indexPos1 Index of the first position to generate
   * \param indexPos2 Index of the second position to generate
   * \param box The 3D box
   * \param intersect The expected result of the test. True for intersection,
   *        false otherwise
   *
   * \return The name string
   */
  std::string BuildNameString (uint16_t indexPos1, uint16_t indexPos2, Box box, bool intersect);
  /**
   * Destructor
   */
  virtual ~BoxLineIntersectionTestCase ();

private:
  /**
   * \brief Setup the simulation according to the configuration set by the
   *        class constructor, run it, and verify the result.
   */
  virtual void DoRun (void);
  /**
   * \brief Create the position as per the given index
   *
   * \param index The index of the position to be created
   * \param boxHeight The height if the box. It is used to place the position
   *        above the height of the box
   * \returns The ns-3 3D Vector specifying the position
   */
  Vector CreatePosition (uint16_t index, double boxHeight);

  uint16_t m_indexPos1 {0}; //!< First position index
  uint16_t m_indexPos2 {0}; //!< Second position index
  Box m_box; //!< The box to check the intersection with
  bool m_intersect {false}; /**< Flag to indicate the intersection.
                             * True, for intersection, false otherwise.
                             */

};


#endif /* BOX_LINE_INTERSECTION_TEST_H */
