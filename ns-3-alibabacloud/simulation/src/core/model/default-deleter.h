/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 INRIA
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

#ifndef DEFAULT_DELETER_H
#define DEFAULT_DELETER_H

/**
 * \file
 * \ingroup ptr
 * ns3::DefaultDeleter declaration and template implementation,
 * for reference-counted smart pointers.
 */

namespace ns3 {

/**
 * \ingroup ptr
 * \brief A template used to delete objects
 *        by the ns3::SimpleRefCount templates when the
 *        last reference to an object they manage
 *        disappears.
 *
 * \tparam T \deduced The object type being deleted.
 * \sa ns3::SimpleRefCount
 */
template <typename T>
struct DefaultDeleter
{
  /**
   * The default deleter implementation, which just does a normal
   * \code
   *   delete object;
   * \endcode
   * \tparam T \deduced The object type being deleted.
   * \param [in] object The object to delete.
   */
  inline static void Delete (T *object)
  {
    delete object;
  }
};

} // namespace ns3

#endif /* DEFAULT_DELETER_H */
