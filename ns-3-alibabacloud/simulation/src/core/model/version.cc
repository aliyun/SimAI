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

#include "version.h"
#include "ns3/version-defines.h"
#include <sstream>

/**
 * \file
 * \ingroup buildversion
 * ns3::Version implementation
 */

namespace ns3 {

std::string
Version::VersionTag (void)
{
  return NS3_VERSION_TAG;
}

std::string
Version::ClosestAncestorTag (void)
{
  return NS3_VERSION_CLOSEST_TAG;
}

uint32_t
Version::Major (void)
{
  return NS3_VERSION_MAJOR;
}

uint32_t
Version::Minor (void)
{
  return NS3_VERSION_MINOR;
}

uint32_t
Version::Patch (void)
{
  return NS3_VERSION_PATCH;
}

std::string
Version::ReleaseCandidate (void)
{
  return std::string{NS3_VERSION_RELEASE_CANDIDATE};
}

uint32_t
Version::TagDistance (void)
{
  return NS3_VERSION_TAG_DISTANCE;
}

bool
Version::DirtyWorkingTree (void)
{
  return static_cast<bool> (NS3_VERSION_DIRTY_FLAG);
}

std::string
Version::CommitHash (void)
{
  return std::string{NS3_VERSION_COMMIT_HASH};
}

std::string
Version::BuildProfile (void)
{
  return std::string{NS3_VERSION_BUILD_PROFILE};
}

std::string
Version::ShortVersion (void)
{
  std::ostringstream ostream;
  ostream << VersionTag (); 

  auto ancestorTag = ClosestAncestorTag ();
  if ( ( !ancestorTag.empty () && (ancestorTag != VersionTag ()) )
       || TagDistance () > 0)
    {
      ostream << "+";
    }
  if (DirtyWorkingTree ())
    {
      ostream << "*";
    }

  return ostream.str ();
}

std::string
Version::BuildSummary (void)
{
  std::ostringstream ostream;
  ostream << ClosestAncestorTag ();

  if (TagDistance () > 0)
    {
      ostream << "+";
    }
  if (DirtyWorkingTree ())
    {
      ostream << "*";
    }

  return ostream.str ();
}

std::string
Version::LongVersion (void)
{
  std::ostringstream ostream;
  ostream << VersionTag (); 

  auto ancestorTag = ClosestAncestorTag ();
  if ( !ancestorTag.empty () && (ancestorTag != VersionTag ()) )
    {
      ostream << '+' << ancestorTag;
    }

  auto tagDistance = TagDistance ();

  if ( tagDistance > 0 )
    {
      ostream << '+' << tagDistance;
    }

  ostream << '@' << CommitHash ();

  if ( DirtyWorkingTree () )
    {
      ostream << "-dirty";
    }

  ostream << '-' << BuildProfile ();

  return ostream.str ();
}

}   // namespace ns3

