/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Lawrence Livermore National Laboratory
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
 * Author: Mathew Bielejeski <bielejeski1@llnl.gov>
 */

#include "ns3/core-module.h"
#include "ns3/length.h"

#include <iostream>

/**
 * \file
 * \ingroup core-examples
 * \ingroup length
 * Demonstrates usage of the ns3::Length class
 */

using namespace ns3;

/**
 * \ingroup core-examples
 * \ingroup length
 * \brief Demonstrates the use of ns3::Length constructors.
 */
void Constructors ()
{
    double input = 5;
    Length::Quantity quantity (input, Length::Unit::Meter);

    std::cout << "\nConstructors:"
              << "\nLength (" << input << ", Unit::Meter) = " << Length (input, Length::Unit::Meter)
              << "\nLength (" << input << ", \"m\") = " << Length (input, "m")
              << "\nLength (" << input << ", \"meter\") = " << Length (input, "meter")
              << "\nLength (Quantity(" << input << ", Unit::Meter)) = " << Length (quantity)
              << "\nLength (\"5m\") = " << Length ("5m")
              << "\nLength (\"5 m\") = " << Length ("5 m")
              << "\nLength (\"5meter\") = " << Length ("5meter")
              << "\nLength (\"5 meter\") = " << Length ("5 meter")
              << "\nLength (\"5meters\") = " << Length ("5meters")
              << "\nLength (\"5 meters\") = " << Length ("5 meters")
              << std::endl;
}

/**
 * \ingroup core-examples
 * \ingroup length
 * \brief Demonstrates the use of ns3::Length conversions.
 */
void Conversions ()
{
  //construct length using value and unit
  Length moonDistance (3.84402e8, Length::Unit::Meter);

  //Demonstrate conversion to various units
  std::cout << "\nConversions: "
            << "\nDistance to moon = " << moonDistance
            << "\nIn Feet: " << moonDistance.As (Length::Unit::Foot)
            << "\nIn Miles: " << moonDistance.As (Length::Unit::Mile)
            << "\nIn Kilometers: " << moonDistance.As (Length::Unit::Kilometer)
            << std::endl;
}

/**
 * \ingroup core-examples
 * \ingroup length
 * \brief Demonstrates the use of ns3::Length arithmetic operators.
 */
void ArithmeticOperators ()
{
  double scale = 10;

  //construct lengths using helper functions
  Length oneMeter = Meters (1);
  Length twoMeter = Meters (2);

  std::cout << "\nArithmetic Operations:"
            << "\nAddition: " << oneMeter << " + " << oneMeter << " = "
            << (oneMeter + oneMeter)
            << "\nSubtraction: " << oneMeter << " - " << oneMeter << " = "
            << (oneMeter - oneMeter)
            << "\nMultiplication By Scalar: " << oneMeter << " * " << scale << " = "
            << (oneMeter * scale)
            << "\nDivision: " << oneMeter << " / " << twoMeter << " = "
            << (oneMeter / twoMeter)
            << "\nDivision By Scalar: " << oneMeter << " / " << scale << " = "
            << (oneMeter / scale)
            << std::endl;

}

/**
 * \ingroup core-examples
 * \ingroup length
 * \brief Demonstrates the use of ns3::Length equality operators.
 */
void EqualityOperators ()
{
  Length oneMeter = Meters (1);
  Length twoMeter = Meters (2);
  Length threeMeter = Meters (3);

  std::cout << "\nComparison Operations:" << std::boolalpha
            << "\nEquality: " << oneMeter << " == " << oneMeter << " is "
            << (oneMeter == oneMeter)
            << "\nEquality: " << oneMeter << " == " << twoMeter << " is "
            << (oneMeter == twoMeter)
            << "\nInequality: " << oneMeter << " != " << oneMeter << " is "
            << (oneMeter != oneMeter)
            << "\nInequality: " << oneMeter << " != " << twoMeter << " is "
            << (oneMeter != twoMeter)
            << "\nLesser: "   << oneMeter << " < " << oneMeter << " is "
            << (oneMeter < oneMeter)
            << "\nLesser: "   << oneMeter << " < " << twoMeter << " is "
            << (oneMeter < twoMeter)
            << "\nLesser: "   << threeMeter << " < " << oneMeter << " is "
            << (threeMeter < oneMeter)
            << "\nGreater: "  << oneMeter << " > " << oneMeter << " is "
            << (oneMeter > oneMeter)
            << "\nGreater: "  << oneMeter << " > " << twoMeter << " is "
            << (oneMeter > twoMeter)
            << "\nGreater: "  << threeMeter << " > " << oneMeter << " is "
            << (threeMeter > oneMeter)
            << std::endl;
}

/**
 * \ingroup core-examples
 * \ingroup length
 * \brief Demonstrates the use of ns3::Length multiplications and divisions.
 */
void DivAndMod ()
{
  //construct length using helper function
  Length totalLen = Feet (20);
  Length pieceLen = Feet (3);
  Length remainder;

  int64_t count = Div (totalLen, pieceLen, &remainder);

  std::cout << "\nHow many times can a " << totalLen.As (Length::Unit::Foot) << " length "
            << "be split into " << pieceLen.As (Length::Unit::Foot) << " sized pieces? "
            << count
            << "\nremainder: " << remainder.As (Length::Unit::Foot)
            << std::endl;

  std::cout << "\nHow much remains after splitting a "
            << totalLen.As (Length::Unit::Foot) << " length into "
            << pieceLen.As (Length::Unit::Foot) << " sized pieces? "
            << Mod (totalLen, pieceLen).As (Length::Unit::Foot)
            << std::endl;
}

int main (int argc, char** argv)
{
  Constructors ();
  Conversions ();
  ArithmeticOperators ();
  EqualityOperators ();
  DivAndMod ();

  return 0;
}
