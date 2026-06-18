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

#ifndef NS3_CSV_READER_H_
#define NS3_CSV_READER_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <string>
#include <vector>

/**
 * \file
 * \ingroup core
 * \ingroup csvreader
 *
 * ns3::CsvReader declaration
 *
 */
namespace ns3 {

/**
 * \ingroup core
 * \defgroup csvreader CSV File Reader
 *
 * A way to extract data from simple csv files.
 */

// *NS_CHECK_STYLE_OFF*  Style checker trims blank lines in code blocks

/**
 * \ingroup csvreader
 *
 * Provides functions for parsing and extracting data from
 * Comma Separated Value (CSV) formatted text files.
 * This parser is somewhat more relaxed than \RFC{4180};
 * see below for a list of the differences.
 * In particular it is possible to set the delimiting character at construction,
 * enabling parsing of tab-delimited streams or other formats with delimiters.
 *
 * \note Excel may generate "CSV" files with either ',' or ';' delimiter
 * depending on the locale: if ',' is the decimal mark then ';' is the list
 * separator and used to read/write "CSV" files.
 *
 * To use this facility, construct a CsvReader from either a file path
 * or \c std::istream, then FetchNextRow(), and finally GetValue()
 * to extract specific values from the row.
 *
 * For example:
 * \code
 *   CsvReader csv (filePath);
 *   while (csv.FetchNextRow ())
 *     {
 *       // Ignore blank lines
 *       if (csv.IsBlankRow ())
 *         {
 *           continue;
 *         }
 *
 *       // Expecting three values
 *       double x, y, z;
 *       bool ok = csv.GetValue (0, x);
 *       ok |= csv.GetValue (1, y);
 *       ok |= csv.GetValue (2, z);
 *       if (!ok)
 *         {
 *           // Handle error, then
 *           continue;
 *         }
 *       
 *       // Do something with values
 *
 *     }  // while FetchNextRow
 * \endcode
 *
 * As another example, supposing we need a vector from each row,
 * the middle of the previous example would become:
 * \code
 *       std::vector<double> v (n);
 *       bool ok = true;
 *       for (std::size_t i = 0; i < v.size (); ++i)
 *         {
 *           ok |= csv.GetValue (i, v[i]);
 *         }
 *       if (!ok) ...
 * \endcode
 *
 *
 * File Format
 * ===========
 *
 * This parser implements \RFC{4180}, but with several restrictions removed;
 * see below for differences.  All the formatting features described next
 * are illustrated in the examples which which follow.
 *
 * Comments
 * --------
 *
 * The hash character (#) is used to indicate the start of a comment.  Comments
 * are not parsed by the reader. Comments are treated as either an empty column
 * or part of an existing column depending on where the comment is located.
 * Comments that are found at the end of a line containing data are ignored.
 *
 *     1,2 # This comment ignored, leaving two data columns
 *
 * Lines that contain a comment and no data are treated as rows with a single
 * empty column, meaning that ColumnCount will return 1 and
 * GetValue() will return an empty string.
 *
 *     # This row treated as a single empty column, returning an empty string.
 *     "" # So is this
 *
 * IsBlankRow() will return \c true in either of these cases.
 *
 * Quoted Columns
 * --------------
 *
 * Columns with string data which contain the delimiter character or
 * the hash character can be wrapped in double quotes to prevent CsvReader
 * from treating them as special characters.
 *
 *     3,string without delimiter,"String with comma ',' delimiter"
 *
 * Double quotes can be escaped
 * by doubling up the quotes inside a quoted field. See example 6 below for
 * a demonstration.
 *
 * Whitespace
 * ----------
 *
 * Leading and trailing whitespace are ignored by the reader and are not
 * stored in the column data.
 *
 *     4,5 , 6     # Columns contain '4', '5', '6'
 *
 * If leading or trailing whitespace are important
 * for a column, wrap the column in double quotes as discussed above.
 *
 *     7,"8 "," 9" # Columns contain '7', '8 ', ' 9'
 *
 * Trailing Delimiter
 * ------------------
 *
 * Trailing delimiters are ignored; they do _not_ result in an empty column.
 *
 *
 * Differences from RFC 4180
 * -------------------------
 * Section 2.1
 *   - Line break can be LF or CRLF
 *
 * Section 2.3
 *   - Non-parsed lines are allowed anywhere, not just as a header.
 *   - Lines do not all have to contain the same number fields.
 *
 * Section 2.4
 *   - Characters other than comma can be used to separate fields.
 *   - Lines do not all have to contain the same number fields.
 *   - Leading/trailing spaces are stripped from the field
 *     unless the whitespace is wrapped in double quotes.
 *   - A trailing delimiter on a line is not an error.
 *
 * Section 2.6
 *   - Quoted fields cannot contain line breaks
 *
 * Examples
 * --------
 * \par Example 1: Basic
 * \code
 *     # Column 1: Product
 *     # Column 2: Price
 *     widget, 12.5
 * \endcode
 *
 * \par Example 2: Comment at end of line
 * \code
 *     # Column 1: Product
 *     # Column 2: Price
 *     broken widget, 12.5 # this widget is broken
 * \endcode
 *
 * \par Example 3: Delimiter in double quotes
 * \code
 *     # Column 1: Product
 *     # Column 2: Price
 *     # Column 3: Count
 *     # Column 4: Date
 *     widget, 12.5, 100, "November 6, 2018"
 * \endcode
 *
 * \par # Example 4: Hash character in double quotes
 * \code
 *     # Column 1: Key
 *     # Column 2: Value
 *     # Column 3: Description
 *     count, 5, "# of widgets currently in stock"
 * \endcode
 *
 * \par Example 5: Extra whitespace
 * \code
 *     # Column 1: Key
 *     # Column 2: Value
 *     # Column 3: Description
 *     count     ,     5    ,"# of widgets in stock"
 * \endcode
 *
 * \par Example 6: Escaped quotes
 * \code
 *     # Column 1: Key
 *     # Column 2: Description
 *     # The value returned for Column 2 will be: String with "embedded" quotes
 *     foo, "String with ""embedded"" quotes"
 * \endcode
 */
// *NS_CHECK_STYLE_ON*
class CsvReader
{
public:
  /**
   * Constructor
   *
   * Opens the file specified in the filepath argument and
   * reads data from it.
   *
   * \param filepath Path to a file containing CSV data.
   * \param delimiter Character used to separate fields in the data file.
   */
  CsvReader (const std::string& filepath, char delimiter = ',');

  /**
   * Constructor
   *
   * Reads csv data from the supplied input stream.
   *
   * \param stream Input stream containing csv data.
   * \param delimiter Character used to separate fields in the data stream.
   */
  CsvReader (std::istream& stream, char delimiter = ',');

  /**
   * Destructor
   */
  virtual ~CsvReader ();

  /**
   * Returns the number of columns in the csv data.
   *
   * \return Number of columns
   */
  std::size_t ColumnCount () const;

  /**
   * The number of lines that have been read.
   *
   * \return The number of lines that have been read.
   */
  std::size_t RowNumber () const;

  /**
   * Returns the delimiter character specified during object construction.
   *
   * \return Character used as the column separator.
   */
  char Delimiter () const;

  /**
   * Reads one line from the input until a new line is encountered.
   * The read data is stored in a cache which is accessed by the
   * GetValue functions to extract fields from the data.
   *
   * \return \c true if a line was read successfully or \c false if the
   * read failed or reached the end of the file.
   */
  bool FetchNextRow ();

  /**
   * Attempt to convert from the string data in the specified column
   * to the specified data type.
   *
   * \tparam T The data type of the output variable.
   *
   * \param [in] columnIndex Index of the column to fetch.
   * \param [out] value Location where the converted data will be stored.
   *
   * \return \c true if the specified column has data and the data
   * was converted to the specified data type.
   */
  template<class T>
  bool GetValue (std::size_t columnIndex, T& value) const;

  /**
   * Check if the current row is blank.
   * A blank row can consist of any combination of
   *
   * - Whitespace
   * - Comment
   * - Quoted empty string `""`
   *
   * \returns \c true if the input row is a blank line.
   */
  bool IsBlankRow () const;

private:
  /**
   * Attempt to convert from the string data stored at the specified column
   * index into the specified type.
   *
   * \param input [in] String value to be converted.
   * \param value [out] Location where the converted value will be stored.
   *
   * \return \c true if the column exists and the conversion succeeded,
   * \c false otherwise.
   */
  /** @{ */
  bool GetValueAs (std::string input, double& value) const;

  bool GetValueAs (std::string input, float& value) const;

  bool GetValueAs (std::string input, signed char& value) const;

  bool GetValueAs (std::string input, short& value) const;

  bool GetValueAs (std::string input, int& value) const;

  bool GetValueAs (std::string input, long& value) const;

  bool GetValueAs (std::string input, long long& value) const;

  bool GetValueAs (std::string input, std::string& value) const;

  bool GetValueAs (std::string input, unsigned char& value) const;

  bool GetValueAs (std::string input, unsigned short& value) const;

  bool GetValueAs (std::string input, unsigned int& value) const;

  bool GetValueAs (std::string input, unsigned long& value) const;

  bool GetValueAs (std::string input, unsigned long long& value) const;
  /** @} */

  /**
   * Returns \c true if the supplied character matches the delimiter.
   *
   * \param c Character to check.
   * \return \c true if \pname{c} is the delimiter character,
   * \c false otherwise.
   */
  bool IsDelimiter (char c) const;

  /**
   * Scans the string and splits it into individual columns based on the delimiter.
   *
   * \param [in] line String containing delimiter separated data.
   */
  void ParseLine (const std::string& line);

  /**
   * Extracts the data for one column in a csv row.
   *
   * \param begin Iterator to the first character in the row.
   * \param end Iterator to the last character in the row.
   * \return A tuple containing the content of the column and an iterator
   * pointing to the position in the row where the column ended.
   */
  std::tuple<std::string, std::string::const_iterator>
  ParseColumn (std::string::const_iterator begin, std::string::const_iterator end);

  /**
   * Container of CSV data.  Each entry represents one field in a row
   * of data.  The fields are stored in the same order that they are
   * encountered in the CSV data.
   */
  typedef std::vector<std::string> Columns;

  char m_delimiter;              //!< Character used to separate fields.
  std::size_t m_rowsRead;        //!< Number of lines processed.
  Columns m_columns;             //!< Fields extracted from the current line.
  bool m_blankRow;               //!< Line contains no data (blank line or comment only).
  std::ifstream m_fileStream;    //!< File stream containing the data.

  /**
   * Pointer to the input stream containing the data.
   */
  std::istream* m_stream;

};  // class CsvReader


/****************************************************
 *      Template implementations.
 ***************************************************/

template<class T>
bool
CsvReader::GetValue (std::size_t columnIndex, T& value) const
{
  if ( columnIndex >= ColumnCount () )
    {
      return false;
    }

  std::string cell = m_columns[columnIndex];

  return GetValueAs (std::move (cell), value);
}

}   //  namespace ns3

#endif  //  NS3_CSV_READER_H_
