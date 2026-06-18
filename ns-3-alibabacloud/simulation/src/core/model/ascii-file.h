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
 * This file is based on pcap-file.h by Craig Dowell (craigdo@ee.washington.edu)
 */

#ifndef ASCII_FILE_H
#define ASCII_FILE_H

#include <string>
#include <fstream>
#include <stdint.h>

namespace ns3 {


/**
 * \brief A class representing an ascii file.
 *
 * This class represents an ascii file
 */
class AsciiFile
{
public:
  AsciiFile ();
  ~AsciiFile ();

  /**
   * \return true if the 'fail' bit is set in the underlying iostream, false otherwise.
   */
  bool Fail (void) const;
  /**
   * \return true if the 'eof' bit is set in the underlying iostream, false otherwise.
   */
  bool Eof (void) const;

  /**
   * Create a new ascii file or open an existing ascii file.
   *
   * \param filename String containing the name of the file.
   * \param mode the access mode for the file.
   */
  void Open (std::string const &filename, std::ios::openmode mode);

  /**
   * Close the underlying file.
   */
  void Close (void);

  /**
   * \brief Read next line from file
   *
   * \param line    [out] line from file
   *
   */
  void Read (std::string& line);

  /**
   * \brief Compare two ASCII files line-by-line
   *
   * \return true if files are different, false otherwise
   *
   * \param  f1         First ASCII file name
   * \param  f2         Second ASCII file name
   * \param  lineNumber   [out] Line number of first different line.
   */
  static bool Diff (std::string const & f1,
                    std::string const & f2,
                    uint64_t & lineNumber);

private:
  std::string    m_filename; //!< output file name
  std::fstream   m_file;     //!< output file
};

} // namespace ns3

#endif /* ASCII_FILE_H */
