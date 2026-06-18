/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA
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

#ifndef CONFIG_STORE_H
#define CONFIG_STORE_H

#include "ns3/object-base.h"
#include "file-config.h"

namespace ns3 {

/**
 * \defgroup configstore Configuration Store/Load
 *
 * \brief Store and load simulation attribute configuration
 *
 * ns-3 Objects and their attribute values (default, and per-instance values)
 * are stored in a specialized internal database.  The config-store module
 * permits these values to be imported and exported to formats of
 * different types (e.g. XML files, raw text files, or a GTK-based UI).
 *
 * While it is possible to generate a sample config file and lightly
 * edit it to change a couple of values, there are cases where this
 * process will not work because the same value on the same object
 * can appear multiple times in the same automatically-generated 
 * configuration file under different configuration paths.
 *
 * As such, the best way to use this class is to use it to generate
 * an initial configuration file, extract from that configuration
 * file only the strictly necessary elements, and move these minimal
 * elements to a new configuration file which can then safely
 * be edited. Another option is to use the ns3::GtkConfigStore class
 * which will allow you to edit the parameters and will generate 
 * configuration files where all the instances of the same parameter
 * are changed.
 */

/**
 * \ingroup configstore
 *
 */
class ConfigStore : public ObjectBase
{
public:
/**
 * \enum Mode for ConfigStore operation
 * \brief store / load mode
 */
  enum Mode {
    LOAD,
    SAVE,
    NONE
  };
/**
 * \enum FileFormat for ConfigStore operation
 * \brief file format
 */
  /// store format
  enum FileFormat {
    XML,
    RAW_TEXT
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  ConfigStore ();
  ~ConfigStore ();

  /**
   * Set the mode of operation
   * \param mode mode of operation
   */
  void SetMode (enum Mode mode);
  /**
   * Set the file format
   * \param format the file format
   */
  void SetFileFormat (enum FileFormat format);
  /**
   * Set the filename
   * \param filename the file name
   */
  void SetFilename (std::string filename);
  /**
   * Set if to save deprecated attributes
   * \param saveDeprecated the deprecated attributes save policy
   */
  void SetSaveDeprecated (bool saveDeprecated);

  /**
   * Configure the default values
   */
  void ConfigureDefaults (void);
  /**
   * Configure the attribute values
   */
  void ConfigureAttributes (void);

private:
  enum Mode m_mode; ///< store mode
  enum FileFormat m_fileFormat; ///< store format
  bool m_saveDeprecated; ///< save deprecated attributes
  std::string m_filename; ///< store file name
  FileConfig *m_file; ///< configuration file
};

/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] mode The configStore mode.
 * \returns The reference to the output stream.
 */
std::ostream & operator << (std::ostream & os, ConfigStore::Mode & mode);
/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] format The configStore file format.
 * \returns The reference to the output stream.
 */
std::ostream & operator << (std::ostream & os, ConfigStore::FileFormat & format);

}  // namespace ns3

#endif /* CONFIG_STORE_H */
