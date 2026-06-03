/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington
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
 * Author: Mitch Watrous (watrous@u.washington.edu)
 *
 * This file is based on pcap-file.cc by Craig Dowell (craigdo@ee.washington.edu)
 */

#include "ascii-file.h"
#include "assert.h"
#include "fatal-error.h"
#include "fatal-impl.h"
#include <iostream>
#include <string>

//
// This file is used as part of the ns-3 test framework, so please refrain from
// adding any ns-3 specific constructs such as Packet to this file.
//
namespace ns3 {

AsciiFile::AsciiFile ()
  : m_file ()
{
  FatalImpl::RegisterStream (&m_file);
}

AsciiFile::~AsciiFile ()
{
  FatalImpl::UnregisterStream (&m_file);
  Close ();
}

bool
AsciiFile::Fail (void) const
{
  return m_file.fail ();
}
bool
AsciiFile::Eof (void) const
{
  return m_file.eof ();
}

void
AsciiFile::Close (void)
{
  m_file.close ();
}

void
AsciiFile::Open (std::string const &filename, std::ios::openmode mode)
{
  NS_ASSERT ((mode & std::ios::app) == 0);
  NS_ASSERT (!m_file.fail ());

  m_file.open (filename.c_str (), mode);
}

void
AsciiFile::Read (std::string& line)
{
  NS_ASSERT (m_file.good ());

  // Read the next line.
  getline (m_file, line);
}

bool
AsciiFile::Diff (std::string const & f1,
                 std::string const & f2,
                 uint64_t & lineNumber)
{
  AsciiFile ascii1, ascii2;
  ascii1.Open (f1, std::ios::in);
  ascii2.Open (f2, std::ios::in);
  bool bad = ascii1.Fail () || ascii2.Fail ();
  if (bad)
    {
      return true;
    }

  std::string line1;
  std::string line2;
  lineNumber = 0;
  bool diff = false;

  while (!ascii1.Eof () && !ascii2.Eof ())
    {
      ascii1.Read (line1);
      ascii2.Read (line2);

      lineNumber++;

      bool same = ascii1.Fail () == ascii2.Fail ();
      if (!same)
        {
          diff = true;
          break;
        }
      if (ascii1.Eof ())
        {
          break;
        }

      if (line1 != line2)
        {
          diff = true; // Lines do not match
          break;
        }
    }

  return diff;
}

} // namespace ns3
