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

#ifndef XML_CONFIG_STORE_H
#define XML_CONFIG_STORE_H

#include <string>
#include <libxml/xmlwriter.h>
#include <libxml/xmlreader.h>
#include "file-config.h"

namespace ns3 {

/**
 * \ingroup configstore
 * \brief A class to enable saving of configuration store in an XML file
 *
 */
class XmlConfigSave : public FileConfig
{
public:
  XmlConfigSave ();
  virtual ~XmlConfigSave ();

  virtual void SetFilename (std::string filename);
  virtual void Default (void);
  virtual void Global (void);
  virtual void Attributes (void);
private:
  xmlTextWriterPtr m_writer; ///< XML writer
};

/**
 * \ingroup configstore
 * \brief A class to enable loading of configuration store from an XML file
 */
class XmlConfigLoad : public FileConfig
{
public:
  XmlConfigLoad ();
  virtual ~XmlConfigLoad ();

  virtual void SetFilename (std::string filename);
  virtual void Default (void);
  virtual void Global (void);
  virtual void Attributes (void);
private:
  std::string m_filename; ///< the file name
};

} // namespace ns3

#endif /* XML_CONFIG_STORE_H */
