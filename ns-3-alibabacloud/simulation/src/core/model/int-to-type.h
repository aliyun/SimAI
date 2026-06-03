/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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

#ifndef INT_TO_TYPE_H
#define INT_TO_TYPE_H

/**
 * \file
 * ns3::IntToType template class.
 */

namespace ns3 {

/**
 * Convert an integer into a type.
 *
 * This trivial template is extremely useful, as explained in
 * "Modern C++ Design", p 29, section 2.4,
 * "Mapping Integral Constants to Types".
 *
 * For an example, see timer-impl.h
 *
 * \tparam v \explicit The integral constant value distinguishing this type
 *           from other values.
 */
template <int v>
struct IntToType
{
  /** Enumeration holding the type-specific value. */
  enum v_e
  {
    value = v  /**< The integer value distinguishing this type. */
  };
};

} // namespace ns3

#endif /* INT_TO_TYPE_H */
