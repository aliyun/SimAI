/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
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

#ifndef TYPE_NAME_H
#define TYPE_NAME_H

#include <stdint.h>
#include <string>

/**
 * \file
 * \ingroup attributeimpl
 * ns3::TypeNameGet() function declarations.
 */

namespace ns3 {

/**
 * \ingroup attributeimpl
 * Type name strings for numeric AttributeValue types.
 *
 * \tparam T \explicit The numeric type.
 * \returns The numeric type name as a string.
 */
template <typename T>
std::string TypeNameGet (void)
{
  return "unknown";
}

/**
 * \ingroup attributeimpl
 * ns3::TypeNameGet(void) specializaton.
 * \returns The numeric type name as a string.
 * @{
 */
// *NS_CHECK_STYLE_OFF*
template <> inline std::string TypeNameGet< int8_t  > (void) { return "int8_t"  ; };
template <> inline std::string TypeNameGet< int16_t > (void) { return "int16_t" ; };
template <> inline std::string TypeNameGet< int32_t > (void) { return "int32_t" ; };
template <> inline std::string TypeNameGet< int64_t > (void) { return "int64_t" ; };
template <> inline std::string TypeNameGet< uint8_t > (void) { return "uint8_t" ; };
template <> inline std::string TypeNameGet< uint16_t> (void) { return "uint16_t"; };
template <> inline std::string TypeNameGet< uint32_t> (void) { return "uint32_t"; };
template <> inline std::string TypeNameGet< uint64_t> (void) { return "uint64_t"; };
template <> inline std::string TypeNameGet< float   > (void) { return "float"   ; };
template <> inline std::string TypeNameGet< double  > (void) { return "double"  ; };
// *NS_CHECK_STYLE_ON*
/** @} */

} // namespace ns3

#endif /* TYPE_NAME_H */
