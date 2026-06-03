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

#include <string>
#include <vector>
#include "get-wildcard-matches.h"
#include "ns3/assert.h"

namespace ns3 {

std::string
GetWildcardMatches (const std::string &configPath,
                    const std::string &matchedPath,
                    const std::string &wildcardSeparator)
{
  // If the Config path is just "*", return the whole matched path.
  if (configPath == "*")
    {
      return matchedPath;
    }

  std::vector<std::string> nonWildcardTokens;
  std::vector<std::size_t> nonWildcardTokenPositions;

  size_t nonWildcardTokenCount;
  size_t wildcardCount = 0;

  // Get the non-wildcard tokens from the Config path.
  size_t tokenStart;
  size_t asterisk = -1;
  do
    {
      // Find the non-wildcard token.
      tokenStart = asterisk + 1;
      asterisk = configPath.find ("*", tokenStart);

      // If a wildcard character was found, increment this counter.
      if (asterisk != std::string::npos)
        {
          wildcardCount++;
        }

      // Save this non-wildcard token.
      nonWildcardTokens.push_back (configPath.substr (tokenStart, asterisk - tokenStart));
    }
  while (asterisk != std::string::npos);

  // If there are no wildcards, return an empty string.
  if (wildcardCount == 0)
    {
      return "";
    }

  // Set the number of non-wildcard tokens in the Config path.
  nonWildcardTokenCount = nonWildcardTokens.size ();

  size_t i;

  // Find the positions of the non-wildcard tokens in the matched path.
  size_t token;
  tokenStart = 0;
  for (i = 0; i < nonWildcardTokenCount; i++)
    {
      // Look for the non-wilcard token.
      token = matchedPath.find (nonWildcardTokens[i], tokenStart);

      // Make sure that the token is found.
      if (token == std::string::npos)
        {
          NS_ASSERT_MSG (false, "Error: non-wildcard token not found in matched path");
        }

      // Save the position of this non-wildcard token.
      nonWildcardTokenPositions.push_back (token);

      // Start looking for the next non-wildcard token after the end of
      // this one.
      tokenStart = token + nonWildcardTokens[i].size ();
    }

  std::string wildcardMatches = "";

  // Put the text matches from the matched path for each of the
  // wildcards in the Config path into a string, separated by the
  // specified separator.
  size_t wildcardMatchesSet = 0;
  size_t matchStart;
  size_t matchEnd;
  for (i = 0; i < nonWildcardTokenCount; i++)
    {
      // Find the start and end of this wildcard match.
      matchStart = nonWildcardTokenPositions[i] + nonWildcardTokens[i].size ();
      if (i != nonWildcardTokenCount - 1)
        {
          matchEnd   = nonWildcardTokenPositions[i + 1] - 1;
        }
      else
        {
          matchEnd   = matchedPath.length () - 1;
        }

      // This algorithm gets confused by zero length non-wildcard
      // tokens.  So, only add this wildcard match and update the
      // counters if the match was calculated to start before it began.
      if (matchStart <= matchEnd)
        {
          // Add the wildcard match.
          wildcardMatches += matchedPath.substr (matchStart,
                                                 matchEnd - matchStart + 1);

          // See if all of the wildcard matches have been set.
          wildcardMatchesSet++;
          if (wildcardMatchesSet == wildcardCount)
            {
              break;
            }
          else
            {
              // If there are still more to set, add the separator to
              // the end of the one just added.
              wildcardMatches += wildcardSeparator;
            }
        }
    }

  // Return the wildcard matches.
  return wildcardMatches;
}

} // namespace ns3

