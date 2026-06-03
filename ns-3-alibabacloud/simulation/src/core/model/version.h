/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Lawrence Livermore National Laboratory
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
 * Authors: Mathew Bielejeski <bielejeski1@llnl.gov>
 */

#ifndef BUILD_VERSION_H_
#define BUILD_VERSION_H_

#include "ns3/int64x64.h" 
#include <string>

/**
 * \file
 * \ingroup buildversion
 * class ns3::Version definition
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup buildversion Build version reporting
 *
 * Version information is pulled from the local git repository during the build
 * process.  If a git repository is not found, the build system will search for 
 * a file named version.cache under the src/core directory.  The version.cache 
 * file must contain key/value pairs in the format key=value with one entry per 
 * line. The build system will use the data pulled from the git repository, or 
 * loaded from the version.cache file, to generate the file version-defines.h 
 *
 * The build will fail if a local git repository is not present and 
 * a version.cache file can not be found.
 */

/**
 * \ingroup buildversion
 *
 * Helper class providing functions to access various parts of the version
 * string, as well as functions for composing short and long variants of the
 * version string.
 *
 * See CommandLine::PrintVersion() for an example on how to
 * use Version to output a version string.  command-line-example has been updated
 * to include CommandLine::PrintVersion() in its output
 *
 * build-version-example.cc illustrates using each of these functions.
 *
 * Below is a partial view of a git branch:
 *
 * \note Square nodes represent tags
 * \note Circle nodes represent commits
 *
 * \dot
 * digraph {
 *   vt [label="ns-3.32", shape="box"]
 *   t  [label="mytag", shape="box"]
 *   h  [label="HEAD", shape="box"]
 *   c1 [label="6ad7f05"]
 *   c2 [label="05fc891"]
 *   c3 [label="bd9ffac"]
 *   c4 [label="70fa23b"]
 *
 *   h -> c1 -> c2 -> c3 -> c4
 *   t -> c2
 *   vt -> c4
 * }
 * \enddot
 *
 * Here are the values that will be assigned based on this example branch:
 *
 * | Component          | Value    | Notes |
 * |--------------------|----------|-------|
 * | VersionTag         | ns-3.32  |       |
 * | ClosestAncestorTag | mytag    |       |
 * | Major              | 3        |       |
 * | Minor              | 32       |       |
 * | Patch              | 0        | This version tag does not have a patch field |
 * | ReleaseCandidate   | ""       | This version tag does not have a release candidate field |
 * | TagDistance        | 1        |       |
 * | CommitHash         | g6ad7f05 | g at front of hash indicates a git hash |
 * | DirtyWorkingTree   | Variable | Depends on status of git working and stage areas |
 * | BuildProfile       | Variable | Depends on the value of --build-profile option of ns3 configure |
 */
class Version
{
public:
  /**
   * Returns the ns-3 version tag of the closest ancestor commit.
   *
   * The format of the tag is 
   * \verbatim ns3-<major>.<minor>[.patch] \endverbatim
   *
   * The patch field is optional and may not be present.  The value of
   * patch defaults to 0 if the tag does not have a patch field.
   *
   * \return ns-3 version tag
   */
  static std::string VersionTag (void);

  /**
   * Returns the closest tag that is attached to a commit that is an ancestor
   * of the current branch head.
   *
   * The value returned by this function may be the same as VersionTag()
   * if the ns-3 version tag is the closest ancestor tag.
   *
   * \return Closest tag attached to an ancestor of the current commit
   */
  static std::string ClosestAncestorTag (void);

  /**
   * Major component of the build version
   *
   * The format of the build version string is 
   * \verbatim ns-<major>.<minor>[.patch][-RC<digit>] \endverbatim
   *
   * The major component is the number before the first period
   *
   * \return The major component of the build version
   */
  static uint32_t Major (void);

  /**
   * Minor component of the build version
   *
   * The format of the build version string is 
   * \verbatim ns-<major>.<minor>[.patch][-RC<digit>] \endverbatim
   *
   * The minor component is the number after the first period
   *
   * \return The minor component of the build version
   */
  static uint32_t Minor (void);

  /**
   * Patch component of the build version
   *
   * A build version with a patch component will have the format
   * \verbatim ns-<major>.<minor>.<patch> \endverbatim
   *
   * The patch component is the number after the second period
   *
   * \return The patch component of the build version or 0 if the build version
   * does not have a patch component
   */
  static uint32_t Patch (void);

  /**
   * Release candidate component of the build version
   *
   * A build version with a release candidate will have the format
   * \verbatim ns-<major>.<minor>[.patch]-RC<digit> \endverbatim
   *
   * The string returned by this function will have the format RC<digit>
   *
   * \return The release candidate component of the build version or an empty
   * string if the build version does not have a release candidate component
   */
  static std::string ReleaseCandidate (void);

  /**
   * The number of commits between the current
   * commit and the tag returned by ClosestAncestorTag().
   *
   * \return The number of commits made since the last tagged commit
   */
  static uint32_t TagDistance (void);

  /**
   * Indicates whether there were uncommitted changes during the build
   *
   * \return \c true if the working tree had uncommitted changes.
   */
  static bool DirtyWorkingTree (void);

  /**
   * Hash of the most recent commit
   *
   * The hash component is the id of the most recent commit.
   * The returned value is a hexadecimal string with enough data to
   * uniquely identify the commit.
   *
   * The first character of the string is a letter indicating the type
   * of repository that was in use: g=git
   *
   * Example of hash output: g6bfb0c9
   *
   * \return hexadecimal representation of the most recent commit id
   */
  static std::string CommitHash (void);

  /**
   * Indicates the type of build that was performed (debug/release/optimized).
   *
   * This information is set by the --build-profile option of ns3 configure
   *
   * \return String containing the type of build
   */
  static std::string BuildProfile (void);

  /**
   * Constructs a string containing the ns-3 major and minor version components,
   * and indication of additional commits or dirty status.
   *
   * The format of the constructed string is 
   * \verbatim ns-<major>.<minor>[.patch][-rc]<flags> \endverbatim
   *
   *   * [patch] is included when Patch() > 0.
   *   * [-rc] is included when ReleaseCandidate() contains a non-empty string
   *   * \c flags will contain `+` when TagDistance() > 0
   *   * \c flags will contain `*` when DirtyWorkingTree() == true.
   *
   * [flags] will contain none, one, or both characters depending on the state
   * of the branch
   *
   * \return String containing the ns-3 major and minor components and flags.
   */
  static std::string ShortVersion (void);

  /**
   * Constructs a string containing the most recent tag and status flags.
   *
   * In the case where closest-ancestor-tag == version-tag, the output of this
   * function will be the same as ShortVersion()
   *
   * The format of the constructed string is `<closest-ancestor-tag>[flags]`.
   *
   *   * \c flags will contain `+` when TagDistance() > 0
   *   * \c flags will contain `*` when DirtyWorkingTree() == true.
   *
   * [flags] will contain none, one, or both characters depending on the state
   * of the branch
   *
   * \return String containing the closest ancestor tag and flags.
   */
  static std::string BuildSummary (void);

  /**
   * Constructs a string containing all of the build details
   *
   * The format of the constructed string is
   * \verbatim 
   * ns-<major>.<minor>[.patch][-rc][-closest-tag]-<tag-distance>@<hash>[-dirty]-<build-profile>
   * \endverbatim
   *
   * [patch], [rc], [closest-tag], and [dirty] will only be present under certain circumstances:
   *   * [patch] is included when Patch() > 0
   *   * [rc] is included when ReleaseCandidate() is not an empty string
   *   * [closest-tag] is included when ClosestTag() != VersionTag()
   *   * [dirty] is included when DirtyWorkingTree() is \c true
   *
   * \return String containing full version
   */
  static std::string LongVersion (void);

};  // class Version

}  // namespace ns3

#endif
