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

#ifndef FILE_CONFIG_H
#define FILE_CONFIG_H

#include <string>

namespace ns3 {

/**
 * \ingroup configstore
 * \brief base class for ConfigStore classes using files
 *
 */
class FileConfig
{
public:
  virtual ~FileConfig ();
  /**
   * Set the file name
   * \param filename the filename
   */
  virtual void SetFilename (std::string filename) = 0;
  /**
   * Set if to save deprecated attributes
   * \param saveDeprecated the deprecated attributes save policy
   */
  void SetSaveDeprecated (bool saveDeprecated);
  /**
   * Load or save the default values
   */
  virtual void Default (void) = 0;
  /**
   * Load or save the global values
   */
  virtual void Global (void) = 0;
  /**
   * Load or save the attributes values
   */
  virtual void Attributes (void) = 0;

protected:
  bool m_saveDeprecated; ///< save deprecated attributes
};

/**
 * \ingroup configstore
 * \brief A dummy class (does nothing)
 */
class NoneFileConfig : public FileConfig
{
public:
  NoneFileConfig ();
  virtual ~NoneFileConfig ();
  virtual void SetFilename (std::string filename);
  virtual void Default (void);
  virtual void Global (void);
  virtual void Attributes (void);
};

} // namespace ns3

#endif /* FILE_CONFIG_H */
