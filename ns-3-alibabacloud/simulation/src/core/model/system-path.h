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
#ifndef SYSTEM_PATH_H
#define SYSTEM_PATH_H

#include <string>
#include <list>

/**
 * \file
 * \ingroup systempath
 * ns3::SystemPath declarations.
 */

namespace ns3 {

/**
 * \ingroup system
 * \defgroup systempath Host Filesystem
 * \brief Encapsulate OS-specific functions to manipulate file
 * and directory paths.
 *
 * The functions provided here are used mostly to implement
 * the ns-3 test framework.
 */

/**
 * \ingroup systempath
 * \brief Namespace for various file and directory path functions.
 */
namespace SystemPath {

/**
 * \ingroup systempath
 * Get the file system path to the current executable.
 *
 * \return The directory in which the currently-executing binary is located
 */
std::string FindSelfDirectory (void);

/**
 * \ingroup systempath
 * Join two file system path elements.
 *
 * \param [in] left A path element
 * \param [in] right A path element
 * \return A concatenation of the two input paths
 */
std::string Append (std::string left, std::string right);

/**
 * \ingroup systempath
 * Split a file system path into directories according to
 * the local path separator.
 *
 * This is the inverse of Join.
 *
 * \param [in] path A path
 * \return A list of path elements that can be joined together again with
 *         the Join function.
 * \sa ns3::SystemPath::Join
 */
std::list<std::string> Split (std::string path);

/**
 * Join a list of file system path directories into a single
 * file system path.
 *
 * This is the inverse of Split.
 *
 * \ingroup systempath
 * \param [in] begin Iterator to first element to join
 * \param [in] end Iterator to one past the last element to join
 * \return A path that is a concatenation of all the input elements.
 */
std::string Join (std::list<std::string>::const_iterator begin,
                  std::list<std::string>::const_iterator end);

/**
 * \ingroup systempath
 * Get the list of files located in a file system directory.
 *
 * \param [in] path A path which identifies a directory
 * \return A list of the filenames which are located in the input directory
 */
std::list<std::string> ReadFiles (std::string path);

/**
 * \ingroup systempath
 * Get the name of a temporary directory.
 *
 * The returned path identifies a directory which does not exist yet.
 * Call ns3::SystemPath::MakeDirectories to create it. Yes, there is a
 * well-known security race in this API but we don't care in ns-3.
 *
 * The final path to the directory is going to look something like
 *
 *     /tmp/ns3.14.30.29.32767
 *
 * The first part, "/tmp/" is the absolute path found by inspecting
 * the environment variables `TMP`and `TEMP`, in order. If neither
 * exists the hard-codes root path `/tmp/` is used.
 *
 * The directory name itself starts with the "ns3" identifier telling folks
 * who is making all of the temp directories.
 *
 * The next three number give the hour, minute and second, separated by
 * periods.
 *
 * The final number is randomly generated, to avoid name collisions.
 *
 * \return A path which identifies a temporary directory.
 */
std::string MakeTemporaryDirectoryName (void);

/**
 * \ingroup systempath
 * Create all the directories leading to path.
 *
 * \param [in] path A path to a directory
 */
void MakeDirectories (std::string path);

/**
 * \ingroup systempath
 * Check if a path exists.
 * Path can be a file or directory.
 * \param [in] path The path to check.
 * \returns \c true if the \pname{path} exists.
 */
bool Exists (const std::string path);

} // namespace SystemPath


} // namespace ns3


#endif /* SYSTEM_PATH_H */
