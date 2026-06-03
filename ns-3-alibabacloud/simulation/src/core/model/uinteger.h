/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef UINTEGER_H
#define UINTEGER_H

#include "attribute.h"
#include "attribute-helper.h"
#include <stdint.h>
#include <limits>

/**
 * \file
 * \ingroup attribute_Uinteger
 * ns3::UintegerValue attribute value declarations and template implementations.
 */

namespace ns3 {

//  Additional docs for class UintegerValue:
/**
 * Hold an unsigned integer type.
 *
 * This class can be used to hold variables of unsigned integer
 * type such as uint8_t, uint16_t, uint32_t, uint64_t, or,
 * unsigned int, etc.
 */
ATTRIBUTE_VALUE_DEFINE_WITH_NAME (uint64_t, Uinteger);
ATTRIBUTE_ACCESSOR_DEFINE (Uinteger);

template <typename T>
Ptr<const AttributeChecker> MakeUintegerChecker (void);

/**
 * Make a checker with a minimum value.
 *
 * The minimum value is included in the allowed range.
 *
 * \param [in] min The minimum value.
 * \returns The AttributeChecker.
 * \see AttributeChecker
 */
template <typename T>
Ptr<const AttributeChecker> MakeUintegerChecker (uint64_t min);

/**
 * Make a checker with a minimum and a maximum value.
 *
 * The minimum and maximum values are included in the allowed range.
 *
 * \param [in] min The minimum value.
 * \param [in] max The maximum value.
 * \returns The AttributeChecker.
 * \see AttributeChecker
 */
template <typename T>
Ptr<const AttributeChecker> MakeUintegerChecker (uint64_t min, uint64_t max);

} // namespace ns3


/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

#include "type-name.h"

namespace ns3 {

namespace internal {

Ptr<const AttributeChecker> MakeUintegerChecker (uint64_t min, uint64_t max, std::string name);

} // namespace internal


template <typename T>
Ptr<const AttributeChecker> MakeUintegerChecker (void)
{
  return internal::MakeUintegerChecker (std::numeric_limits<T>::min (),
                                        std::numeric_limits<T>::max (),
                                        TypeNameGet<T> ());
}

template <typename T>
Ptr<const AttributeChecker> MakeUintegerChecker (uint64_t min)
{
  return internal::MakeUintegerChecker (min,
                                        std::numeric_limits<T>::max (),
                                        TypeNameGet<T> ());
}

template <typename T>
Ptr<const AttributeChecker> MakeUintegerChecker (uint64_t min, uint64_t max)
{
  return internal::MakeUintegerChecker (min,
                                        max,
                                        TypeNameGet<T> ());
}

} // namespace ns3

#endif /* UINTEGER_H */
