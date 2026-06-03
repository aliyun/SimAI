/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@cutebugs.net>
 */

#ifndef PCAP_TEST_H
#define PCAP_TEST_H

#include <sstream>
#include <string>
#include <stdint.h>
#include "pcap-file.h"
#include "ns3/test.h"

/**
 * \brief Test that a pair of reference/new pcap files are equal
 *
 * The filename is interpreted as a stream.
 *
 * \param filename The name of the file to read in the reference/temporary
 *        directories
 */
#define NS_PCAP_TEST_EXPECT_EQ(filename)                                \
  do {                                                                  \
    std::ostringstream oss;                                             \
    oss << filename;                                                    \
    std::string expected = CreateDataDirFilename (oss.str());           \
    std::string got = CreateTempDirFilename (oss.str());                \
    uint32_t sec(0), usec(0), packets(0);				\
    /** \todo support default PcapWriter snap length here */		\
    bool diff = PcapFile::Diff (got, expected, sec, usec, packets);	\
    NS_TEST_EXPECT_MSG_EQ (diff, false,                                 \
                           "PCAP traces "				\
			   << got << " and " << expected		\
                           << " differ starting from packet "		\
			   << packets << " at " << sec << " s "		\
                           << usec << " us");                           \
  } while (false)


#endif /* PCAP_TEST_H */
