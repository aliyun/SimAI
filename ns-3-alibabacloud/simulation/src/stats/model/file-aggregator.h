/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Author: L. Felipe Perrone (perrone@bucknell.edu)
 *
 * Modified by: Mitch Watrous (watrous@u.washington.edu)
 *
 */

#ifndef FILE_AGGREGATOR_H
#define FILE_AGGREGATOR_H

#include <fstream>
#include <map>
#include <string>
#include "ns3/data-collection-object.h"

namespace ns3 {

/**
 * \ingroup aggregator
 *
 * This aggregator sends values it receives to a file.
 **/
class FileAggregator : public DataCollectionObject
{
public:
  /// The type of file written by the aggregator.
  enum FileType
  {
    FORMATTED,
    SPACE_SEPARATED,
    COMMA_SEPARATED,
    TAB_SEPARATED
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * \param outputFileName name of the file to write.
   * \param fileType type of file to write.
   *
   * Constructs a file aggregator that will create a file named
   * outputFileName with values printed as specified by fileType.  The
   * default file type is space-separated.
   */
  FileAggregator (const std::string &outputFileName,
                  enum FileType fileType = SPACE_SEPARATED);

  virtual ~FileAggregator ();

  /**
   * \param fileType file type specifies the separator to use in
   * printing the file.
   *
   * \brief Set the file type to create, which determines the
   * separator to use when printing values to the file.
   */
  void SetFileType (enum FileType fileType);

  /**
   * \param heading the heading string.
   *
   * \brief Sets the heading string that will be printed on the first
   * line of the file.
   *
   * Note that the heading string will only be printed if it has been
   * set by calling this function.
   */
  void SetHeading (const std::string &heading);

  /**
   * \param format the 1D format string.
   *
   * \brief Sets the 1D format string for the C-style sprintf()
   * function.
   */
  void Set1dFormat (const std::string &format);

  /**
   * \param format the 2D format string.
   *
   * \brief Sets the 2D format string for the C-style sprintf()
   * function.
   */
  void Set2dFormat (const std::string &format);

  /**
   * \param format the 3D format string.
   *
   * \brief Sets the 3D format string for the C-style sprintf()
   * function.
   */
  void Set3dFormat (const std::string &format);

  /**
   * \param format the 4D format string.
   *
   * \brief Sets the 4D format string for the C-style sprintf()
   * function.
   */
  void Set4dFormat (const std::string &format);

  /**
   * \param format the 5D format string.
   *
   * \brief Sets the 5D format string for the C-style sprintf()
   * function.
   */
  void Set5dFormat (const std::string &format);

  /**
   * \param format the 6D format string.
   *
   * \brief Sets the 6D format string for the C-style sprintf()
   * function.
   */
  void Set6dFormat (const std::string &format);

  /**
   * \param format the 7D format string.
   *
   * \brief Sets the 7D format string for the C-style sprintf()
   * function.
   */
  void Set7dFormat (const std::string &format);

  /**
   * \param format the 8D format string.
   *
   * \brief Sets the 8D format string for the C-style sprintf()
   * function.
   */
  void Set8dFormat (const std::string &format);

  /**
   * \param format the 9D format string.
   *
   * \brief Sets the 9D format string for the C-style sprintf()
   * function.
   */
  void Set9dFormat (const std::string &format);

  /**
   * \param format the 10D format string.
   *
   * \brief Sets the 10D format string for the C-style sprintf()
   * function.
   */
  void Set10dFormat (const std::string &format);

  // Below are hooked to connectors exporting data
  // They are not overloaded since it confuses the compiler when made
  // into callbacks

  /**
   * \param context specifies the 1D dataset these values came from.
   * \param v1 value for the new data point.
   *
   * \brief Writes 1 value to the file.
   */
  void Write1d (std::string context,
                double v1);

  /**
   * \param context specifies the 2D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   *
   * \brief Writes 2 values to the file.
   */
  void Write2d (std::string context,
                double v1,
                double v2);

  /**
   * \param context specifies the 3D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   *
   * \brief Writes 3 values to the file.
   */
  void Write3d (std::string context,
                double v1,
                double v2,
                double v3);

  /**
   * \param context specifies the 4D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   *
   * \brief Writes 4 values to the file.
   */
  void Write4d (std::string context,
                double v1,
                double v2,
                double v3,
                double v4);

  /**
   * \param context specifies the 5D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   * \param v5 fifth value for the new data point.
   *
   * \brief Writes 5 values to the file.
   */
  void Write5d (std::string context,
                double v1,
                double v2,
                double v3,
                double v4,
                double v5);

  /**
   * \param context specifies the 6D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   * \param v5 fifth value for the new data point.
   * \param v6 sixth value for the new data point.
   *
   * \brief Writes 6 values to the file.
   */
  void Write6d (std::string context,
                double v1,
                double v2,
                double v3,
                double v4,
                double v5,
                double v6);

  /**
   * \param context specifies the 7D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   * \param v5 fifth value for the new data point.
   * \param v6 sixth value for the new data point.
   * \param v7 seventh value for the new data point.
   *
   * \brief Writes 7 values to the file.
   */
  void Write7d (std::string context,
                double v1,
                double v2,
                double v3,
                double v4,
                double v5,
                double v6,
                double v7);

  /**
   * \param context specifies the 8D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   * \param v5 fifth value for the new data point.
   * \param v6 sixth value for the new data point.
   * \param v7 seventh value for the new data point.
   * \param v8 eighth value for the new data point.
   *
   * \brief Writes 8 values to the file.
   */
  void Write8d (std::string context,
                double v1,
                double v2,
                double v3,
                double v4,
                double v5,
                double v6,
                double v7,
                double v8);

  /**
   * \param context specifies the 9D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   * \param v5 fifth value for the new data point.
   * \param v6 sixth value for the new data point.
   * \param v7 seventh value for the new data point.
   * \param v8 eighth value for the new data point.
   * \param v9 ninth value for the new data point.
   *
   * \brief Writes 9 values to the file.
   */
  void Write9d (std::string context,
                double v1,
                double v2,
                double v3,
                double v4,
                double v5,
                double v6,
                double v7,
                double v8,
                double v9);

  /**
   * \param context specifies the 10D dataset these values came from.
   * \param v1 first value for the new data point.
   * \param v2 second value for the new data point.
   * \param v3 third value for the new data point.
   * \param v4 fourth value for the new data point.
   * \param v5 fifth value for the new data point.
   * \param v6 sixth value for the new data point.
   * \param v7 seventh value for the new data point.
   * \param v8 eighth value for the new data point.
   * \param v9 ninth value for the new data point.
   * \param v10 tenth value for the new data point.
   *
   * \brief Writes 10 values to the file.
   */
  void Write10d (std::string context,
                 double v1,
                 double v2,
                 double v3,
                 double v4,
                 double v5,
                 double v6,
                 double v7,
                 double v8,
                 double v9,
                 double v10);

private:
  /// The file name.
  std::string m_outputFileName;

  /// Used to write values to the file.
  std::ofstream m_file;

  /// Determines the kind of file written by the aggregator.
  enum FileType m_fileType;

  /// Printed between values in the file.
  std::string m_separator;

  /// Indicates if the heading line for the file has been set.
  bool m_hasHeadingBeenSet;

  /// Heading line for the outputfile.
  std::string m_heading;

  std::string m_1dFormat;  //!< Format string for 1D C-style sprintf() function.
  std::string m_2dFormat;  //!< Format string for 2D C-style sprintf() function.
  std::string m_3dFormat;  //!< Format string for 3D C-style sprintf() function.
  std::string m_4dFormat;  //!< Format string for 4D C-style sprintf() function.
  std::string m_5dFormat;  //!< Format string for 5D C-style sprintf() function.
  std::string m_6dFormat;  //!< Format string for 6D C-style sprintf() function.
  std::string m_7dFormat;  //!< Format string for 7D C-style sprintf() function.
  std::string m_8dFormat;  //!< Format string for 8D C-style sprintf() function.
  std::string m_9dFormat;  //!< Format string for 9D C-style sprintf() function.
  std::string m_10dFormat; //!< Format string for 10D C-style sprintf() function.

}; // class FileAggregator


} // namespace ns3

#endif // FILE_AGGREGATOR_H
