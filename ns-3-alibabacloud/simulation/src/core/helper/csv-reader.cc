/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Lawrence Livermore National Laboratory
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
 * Author: Mathew Bielejeski <bielejeski1@llnl.gov>
 */

#include "csv-reader.h"

#include "ns3/log.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <limits>

/**
 * \file
 * \ingroup core
 * \ingroup csvreader
 *
 * ns3::CsvReader implementation
 */

NS_LOG_COMPONENT_DEFINE ("CsvReader");

namespace {

/**
 * Convert a string into another type.
 *
 * Uses a stringstream to deserialize the value stored in \p input
 * to a value of type T and writes the deserialized value to \p output.
 *
 * \tparam T Data type of output.
 * \param input String containing serialized data.
 * \param output Place to store deserialized value.
 *
 * \return \c true if deserialization was successful, \c false otherwise.
 */
template<typename T>
bool GenericTransform (std::string input, T& output)
{
  NS_LOG_FUNCTION (input);

  std::istringstream stream (input);

  stream >> output;

  return static_cast<bool> (stream);
}

}   // unnamed namespace

namespace ns3 {

CsvReader::CsvReader (const std::string& filepath, char delimiter /* =',' */)
  : m_delimiter (delimiter),
    m_rowsRead (0),
    m_fileStream (filepath),
    m_stream (&m_fileStream)
{
  NS_LOG_FUNCTION (this << filepath);
}

CsvReader::CsvReader (std::istream& stream, char delimiter /* =',' */)
  : m_delimiter (delimiter),
    m_rowsRead (0),
    m_fileStream (),
    m_stream (&stream)
{
  NS_LOG_FUNCTION (this);
}

CsvReader::~CsvReader ()
{}

std::size_t
CsvReader::ColumnCount () const
{
  NS_LOG_FUNCTION (this);

  return m_columns.size ();
}

std::size_t
CsvReader::RowNumber () const
{
  NS_LOG_FUNCTION (this);

  return m_rowsRead;
}

char
CsvReader::Delimiter () const
{
  NS_LOG_FUNCTION (this);

  return m_delimiter;
}

bool
CsvReader::FetchNextRow ()
{
  NS_LOG_FUNCTION (this);

  std::string line;

  if ( m_stream->eof () )
    {
      NS_LOG_LOGIC ("Reached end of stream");
      return false;
    }

  NS_LOG_LOGIC ("Reading line " << m_rowsRead + 1);

  std::getline (*m_stream, line);

  if ( m_stream->fail () )
    {
      NS_LOG_ERROR ("Reading line " << m_rowsRead + 1 << " failed");

      return false;
    }

  ++m_rowsRead;

  ParseLine (line);

  return true;
}

bool
CsvReader::IsBlankRow () const
{
  return m_blankRow;
}

bool
CsvReader::GetValueAs (std::string input, double& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, float& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, signed char& value) const
{
  typedef signed char byte_type;

  NS_LOG_FUNCTION (this << input);

  std::istringstream tempStream (input);

  std::int16_t tempOutput = 0;
  tempStream >> tempOutput;

  if (tempOutput >= std::numeric_limits<byte_type>::min ()
      || tempOutput <= std::numeric_limits<byte_type>::max () )
    {
      value = static_cast<byte_type> (tempOutput);
    }

  bool success = static_cast<bool> (tempStream);

  NS_LOG_DEBUG ("Input='" << input
                          << "', output=" << tempOutput
                          << ", result=" << success);

  return success;
}

bool
CsvReader::GetValueAs (std::string input, short& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, int& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, long& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, long long& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, std::string& value) const
{
  NS_LOG_FUNCTION (this << input);

  value = input;

  return true;
}

bool
CsvReader::GetValueAs (std::string input, unsigned char& value) const
{
  NS_LOG_FUNCTION (this << input);

  typedef unsigned char byte_type;

  NS_LOG_FUNCTION (this << input);

  std::istringstream tempStream (input);

  std::uint16_t tempOutput = 0;
  tempStream >> tempOutput;

  if (tempOutput >= std::numeric_limits<byte_type>::min ()
      || tempOutput <= std::numeric_limits<byte_type>::max () )
    {
      value = static_cast<byte_type> (tempOutput);
    }

  bool success = static_cast<bool> (tempStream);

  NS_LOG_DEBUG ("Input='" << input
                          << "', output=" << tempOutput
                          << ", result=" << success);

  return success;
}

bool
CsvReader::GetValueAs (std::string input, unsigned short& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, unsigned int& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, unsigned long& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::GetValueAs (std::string input, unsigned long long& value) const
{
  NS_LOG_FUNCTION (this << input);

  return GenericTransform (std::move (input), value);
}

bool
CsvReader::IsDelimiter (char c) const
{
  NS_LOG_FUNCTION (this << c);

  return c == m_delimiter;
}

void
CsvReader::ParseLine (const std::string& line)
{
  NS_LOG_FUNCTION (this << line);

  std::string value;
  m_columns.clear ();

  auto start_col = line.begin ();
  auto end_col = line.end ();

  while ( start_col != line.end () )
    {
      std::tie (value, end_col) = ParseColumn (start_col, line.end ());

      NS_LOG_DEBUG ("ParseColumn() returned: " << value);

      m_columns.push_back (std::move (value));

      if ( end_col != line.end () )
        {
          ++end_col;
        }

      start_col = end_col;
    }
  m_blankRow = (m_columns.size () == 1) && (m_columns[0] == "");
  NS_LOG_LOGIC ("blank row: " << m_blankRow);
}

std::tuple<std::string, std::string::const_iterator>
CsvReader::ParseColumn (std::string::const_iterator begin, std::string::const_iterator end)
{
  NS_LOG_FUNCTION (this << std::string (begin, end));

  enum class State
  {
    BEGIN,
    END_QUOTE,
    FIND_DELIMITER,
    QUOTED_STRING,
    UNQUOTED_STRING,
    END
  };

  State state = State::BEGIN;
  std::string buffer;
  auto iter = begin;

  while (state != State::END)
    {
      if (iter == end)
        {
          NS_LOG_DEBUG ("Found end iterator, switching to END state");

          state = State::END;
          continue;
        }

      auto c = *iter;

      NS_LOG_DEBUG ("Next character: '" << c << "'");

      //handle common cases here to avoid duplicating logic
      if (state != State::QUOTED_STRING)
        {
          if (IsDelimiter (c))
            {
              NS_LOG_DEBUG ("Found field delimiter, switching to END state");

              if ( state == State::UNQUOTED_STRING )
                {
                  NS_LOG_DEBUG ("Removing trailing whitespace from unquoted field: '" << buffer << "'");
                  auto len = buffer.size ();

                  //remove trailing whitespace from the field
                  while ( !buffer.empty ()
                          && std::isspace (static_cast<unsigned char> (buffer.back ())) )
                    {
                      buffer.pop_back ();
                    }

                  auto finalLen = buffer.size ();

                  NS_LOG_DEBUG ("Removed " << (len - finalLen) << " trailing whitespace characters");
                }

              state = State::END;

              continue;
            }
          else if (c == '#')
            {
              NS_LOG_DEBUG ("Found start of comment, switching to END state");

              //comments consume the rest of the line, set iter to end
              //to reflect that fact.
              iter = end;
              state = State::END;

              continue;
            }
        }

      switch (state)
        {
          case State::BEGIN:
            {
              if (c == '"')
                {
                  NS_LOG_DEBUG ("Switching state: BEGIN -> QUOTED_STRING");

                  state = State::QUOTED_STRING;
                }
              else if (!std::isspace (c))
                {
                  NS_LOG_DEBUG ("Switching state: BEGIN -> UNQUOTED_STRING");

                  state = State::UNQUOTED_STRING;
                  buffer.push_back (c);
                }

            } break;
          case State::QUOTED_STRING:
            {
              if (c == '"')
                {
                  NS_LOG_DEBUG ("Switching state: QUOTED_STRING -> END_QUOTE");
                  state = State::END_QUOTE;
                }
              else
                {
                  buffer.push_back (c);
                }

            } break;
          case State::END_QUOTE:
            {
              if (c == '"')
                {
                  NS_LOG_DEBUG ("Switching state: END_QUOTE -> QUOTED_STRING" );

                  //an escape quote instead of an end quote
                  state = State::QUOTED_STRING;
                  buffer.push_back (c);
                }
              else
                {
                  NS_LOG_DEBUG ("Switching state: END_QUOTE -> FIND_DELIMITER" );
                  state = State::FIND_DELIMITER;
                }

            } break;
          case State::UNQUOTED_STRING:
            {
              buffer.push_back (c);
            } break;
          case State::FIND_DELIMITER:
            break;
          case State::END:
            break;
        }

      ++iter;
    }

  NS_LOG_DEBUG ("Field value: " << buffer);

  return std::make_tuple (buffer, iter);
}

}   // namespace ns3

