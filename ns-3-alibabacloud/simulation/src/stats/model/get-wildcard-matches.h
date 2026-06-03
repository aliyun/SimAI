/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 University of Washington
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
 */

#ifndef GET_WILDCARD_MATCHES_H
#define GET_WILDCARD_MATCHES_H

#include <string>

namespace ns3 {

/**
 * \param configPath Config path to access the probe.
 * \param matchedPath the path that matched the Config path.
 * \param wildcardSeparator the text to put between the wildcard
 * matches.  By default, a space is used.
 * \return String value of text matches
 *
 * \brief Returns the text matches from the matched path for each of
 * the wildcards in the Config path, separated by the wild card
 * separator.
 */
std::string GetWildcardMatches (const std::string &configPath,
                                const std::string &matchedPath,
                                const std::string &wildcardSeparator = " ");

} // namespace ns3

#endif // GET_WILDCARD_MATCHES_H
