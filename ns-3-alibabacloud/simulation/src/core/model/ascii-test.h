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
 * This file is based on pcap-test.h by Craig Dowell (craigdo@ee.washington.edu)
 */

#ifndef ASCII_TEST_H
#define ASCII_TEST_H

#include <stdint.h>
#include "ascii-file.h"
#include "ns3/test.h"

/**
 * \brief Test that a pair of new/reference ascii files are equal
 *
 * \param gotFilename The name of the new file to read in including
 * its path
 * \param expectedFilename The name of the reference file to read in
 * including its path
 */
#define NS_ASCII_TEST_EXPECT_EQ(gotFilename, expectedFilename)          \
  do {                                                                  \
      uint64_t line (0);                                                   \
      bool diff = AsciiFile::Diff (gotFilename, expectedFilename, line);  \
      NS_TEST_EXPECT_MSG_EQ (diff, false,                                 \
                             "ASCII traces " << gotFilename <<            \
                             " and " << expectedFilename <<               \
                             " differ starting from line " << line);      \
    } while (false)


#endif /* ASCII_TEST_H */
