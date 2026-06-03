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

#ifndef NS3_DEPRECATED_H
#define NS3_DEPRECATED_H

/**
 * \file
 * \ingroup core
 * NS_DEPRECATED macro definition.
 */

/**
 * \ingroup core
 * \def NS_DEPRECATED
 * Mark a function as deprecated.
 *
 * Users should expect deprecated features to be removed eventually.
 *
 * When deprecating a feature, please update the documentation
 * with information for users on how to update their code.
 *
 * For example,
 * \snippet src/core/doc/deprecated-example.h doxygen snippet
 *
 * Please follow these two guidelines to ease future maintenance
 * (primarily the eventual removal of the deprecated code):
 *
 * 1.  Please use the versioned form `NS_DEPRECATED_3_XX`, 
 *     not the generic `NS_DEPRECATED`.
 *
 * 2.  Typically only the declaration needs to be deprecated, 
 *
 *     \code
 *     NS_DEPRECATED_3_XX ("see TheNewWay") void SomethingUseful ();
 *     \endcode
 *
 *     but it's helpful to put the same macro as a comment 
 *     at the site of the definition, to make it easier to find
 *     all the bits which eventually have to be removed:
 *
 *     \code
 *     \/\* NS_DEPRECATED_3_XX ("see TheNewWay") *\\/
 *     void SomethingUseful () { ... }
 *     \endcode.
 *
 * \param msg Optional message to add to the compiler warning.
 *
 */
#define NS_DEPRECATED(msg) [[deprecated (msg)]]

/**
 * \ingroup core
 * \def NS_DEPRECATED_3_36
 * Tag for things deprecated in version ns-3.36.
 */
#define NS_DEPRECATED_3_36(msg) NS_DEPRECATED(msg)

/**
 * \ingroup core
 * \def NS_DEPRECATED_3_35
 * Tag for things deprecated in version ns-3.35.
 */
#define NS_DEPRECATED_3_35 NS_DEPRECATED("")

/**
 * \ingroup core
 * \def NS_DEPRECATED_3_34
 * Tag for things deprecated in version ns-3.34.
 */
#define NS_DEPRECATED_3_34 NS_DEPRECATED("")

#endif /* NS3_DEPRECATED_H */
