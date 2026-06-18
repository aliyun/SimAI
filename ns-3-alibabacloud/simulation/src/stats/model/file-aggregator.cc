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

#include <iostream>
#include <fstream>
#include <string>

#include "file-aggregator.h"
#include "ns3/abort.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FileAggregator");

NS_OBJECT_ENSURE_REGISTERED (FileAggregator);

TypeId
FileAggregator::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::FileAggregator")
    .SetParent<DataCollectionObject> ()
    .SetGroupName ("Stats")
  ;

  return tid;
}

FileAggregator::FileAggregator (const std::string &outputFileName,
                                enum FileType fileType)
  : m_outputFileName    (outputFileName),
    m_fileType          (fileType),
    m_hasHeadingBeenSet (false),
    m_1dFormat          ("%e"),
    m_2dFormat          ("%e %e"),
    m_3dFormat          ("%e %e %e"),
    m_4dFormat          ("%e %e %e %e"),
    m_5dFormat          ("%e %e %e %e %e"),
    m_6dFormat          ("%e %e %e %e %e %e"),
    m_7dFormat          ("%e %e %e %e %e %e %e"),
    m_8dFormat          ("%e %e %e %e %e %e %e %e"),
    m_9dFormat          ("%e %e %e %e %e %e %e %e %e"),
    m_10dFormat         ("%e %e %e %e %e %e %e %e %e %e")
{
  NS_LOG_FUNCTION (this << outputFileName << fileType);

  // Set the values separator.
  switch (m_fileType)
    {
    case COMMA_SEPARATED:
      m_separator = ",";
      break;
    case TAB_SEPARATED:
      m_separator = "\t";
      break;
    default:
      // Space separated.
      m_separator = " ";
      break;
    }

  m_file.open (m_outputFileName.c_str ());
}

FileAggregator::~FileAggregator ()
{
  NS_LOG_FUNCTION (this);
  m_file.close ();
}

void
FileAggregator::SetFileType (enum FileType fileType)
{
  NS_LOG_FUNCTION (this << fileType);
  m_fileType = fileType;
}

void
FileAggregator::SetHeading (const std::string &heading)
{
  NS_LOG_FUNCTION (this << heading);
  if (!m_hasHeadingBeenSet)
    {
      m_heading = heading;
      m_hasHeadingBeenSet = true;

      // Print the heading to the file.
      m_file << m_heading << std::endl;
    }
}

void
FileAggregator::Set1dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_1dFormat = format;
}

void
FileAggregator::Set2dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_2dFormat = format;
}

void
FileAggregator::Set3dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_3dFormat = format;
}

void
FileAggregator::Set4dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_4dFormat = format;
}

void
FileAggregator::Set5dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_5dFormat = format;
}

void
FileAggregator::Set6dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_6dFormat = format;
}

void
FileAggregator::Set7dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_7dFormat = format;
}

void
FileAggregator::Set8dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_8dFormat = format;
}

void
FileAggregator::Set9dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_9dFormat = format;
}

void
FileAggregator::Set10dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);
  m_10dFormat = format;
}

void
FileAggregator::Write1d (std::string context,
                         double v1)
{
  NS_LOG_FUNCTION (this << context << v1);

  if (m_enabled)
    {
      // Write the 1D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the value.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_1dFormat.c_str (),
                                      v1);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing value to output file");
            }

          // Write the formatted value.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the value.
          m_file << v1 << std::endl;
        }
    }
}

void
FileAggregator::Write2d (std::string context,
                         double v1,
                         double v2)
{
  NS_LOG_FUNCTION (this << context << v1 << v2);

  if (m_enabled)
    {
      // Write the 2D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_2dFormat.c_str (),
                                      v1,
                                      v2);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << std::endl;
        }
    }
}

void
FileAggregator::Write3d (std::string context,
                         double v1,
                         double v2,
                         double v3)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3);

  if (m_enabled)
    {
      // Write the 3D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_3dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << std::endl;
        }
    }
}

void
FileAggregator::Write4d (std::string context,
                         double v1,
                         double v2,
                         double v3,
                         double v4)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4);

  if (m_enabled)
    {
      // Write the 4D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_4dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << std::endl;
        }
    }
}

void
FileAggregator::Write5d (std::string context,
                         double v1,
                         double v2,
                         double v3,
                         double v4,
                         double v5)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4 << v5);

  if (m_enabled)
    {
      // Write the 5D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_5dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4,
                                      v5);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << m_separator
                 << v5 << std::endl;
        }
    }
}

void
FileAggregator::Write6d (std::string context,
                         double v1,
                         double v2,
                         double v3,
                         double v4,
                         double v5,
                         double v6)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4 << v5 << v6);

  if (m_enabled)
    {
      // Write the 6D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_6dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4,
                                      v5,
                                      v6);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << m_separator
                 << v5 << m_separator
                 << v6 << std::endl;
        }
    }
}

void
FileAggregator::Write7d (std::string context,
                         double v1,
                         double v2,
                         double v3,
                         double v4,
                         double v5,
                         double v6,
                         double v7)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4 << v5 << v6 << v7);

  if (m_enabled)
    {
      // Write the 7D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_7dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4,
                                      v5,
                                      v6,
                                      v7);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << m_separator
                 << v5 << m_separator
                 << v6 << m_separator
                 << v7 << std::endl;
        }
    }
}

void
FileAggregator::Write8d (std::string context,
                         double v1,
                         double v2,
                         double v3,
                         double v4,
                         double v5,
                         double v6,
                         double v7,
                         double v8)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8);

  if (m_enabled)
    {
      // Write the 8D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_8dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4,
                                      v5,
                                      v6,
                                      v7,
                                      v8);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << m_separator
                 << v5 << m_separator
                 << v6 << m_separator
                 << v7 << m_separator
                 << v8 << std::endl;
        }
    }
}

void
FileAggregator::Write9d (std::string context,
                         double v1,
                         double v2,
                         double v3,
                         double v4,
                         double v5,
                         double v6,
                         double v7,
                         double v8,
                         double v9)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << v9);
  if (m_enabled)
    {
      // Write the 9D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_9dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4,
                                      v5,
                                      v6,
                                      v7,
                                      v8,
                                      v9);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << m_separator
                 << v5 << m_separator
                 << v6 << m_separator
                 << v7 << m_separator
                 << v8 << m_separator
                 << v9 << std::endl;
        }
    }
}

void
FileAggregator::Write10d (std::string context,
                          double v1,
                          double v2,
                          double v3,
                          double v4,
                          double v5,
                          double v6,
                          double v7,
                          double v8,
                          double v9,
                          double v10)
{
  NS_LOG_FUNCTION (this << context << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << v9 << v10);
  if (m_enabled)
    {
      // Write the 10D data point to the file.
      if (m_fileType == FORMATTED)
        {
          // Initially, have the C-style string in the buffer, which
          // is terminated by a null character, be of length zero.
          char buffer[500];
          int maxBufferSize = 500;
          buffer[0] = 0;

          // Format the values.
          int charWritten = snprintf (buffer,
                                      maxBufferSize,
                                      m_10dFormat.c_str (),
                                      v1,
                                      v2,
                                      v3,
                                      v4,
                                      v5,
                                      v6,
                                      v7,
                                      v8,
                                      v9,
                                      v10);
          if (charWritten < 0)
            {
              NS_LOG_DEBUG ("Error writing values to output file");
            }

          // Write the formatted values.
          m_file << buffer << std::endl;
        }
      else
        {
          // Write the values with the proper separator.
          m_file << v1 << m_separator
                 << v2 << m_separator
                 << v3 << m_separator
                 << v4 << m_separator
                 << v5 << m_separator
                 << v6 << m_separator
                 << v7 << m_separator
                 << v8 << m_separator
                 << v9 << m_separator
                 << v10 << std::endl;
        }
    }
}

} // namespace ns3

