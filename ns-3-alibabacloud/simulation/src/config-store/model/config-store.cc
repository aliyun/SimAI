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

#include "config-store.h"
#include "raw-text-config.h"
#include "ns3/abort.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/attribute-construction-list.h"
#include "ns3/enum.h"
#include "ns3/boolean.h"
#include "ns3/config-store-config.h"
#ifdef HAVE_LIBXML2
#include "xml-config.h"
#endif

#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <cstdlib>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ConfigStore");

NS_OBJECT_ENSURE_REGISTERED (ConfigStore);

TypeId 
ConfigStore::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ConfigStore")
    .SetParent<ObjectBase> ()
    .SetGroupName ("ConfigStore")
    .AddAttribute ("Mode", 
                   "Configuration mode",
                   EnumValue (ConfigStore::NONE),
                   MakeEnumAccessor (&ConfigStore::SetMode),
                   MakeEnumChecker (ConfigStore::NONE, "None",
                                    ConfigStore::LOAD, "Load",
                                    ConfigStore::SAVE, "Save"))
    .AddAttribute ("Filename", 
                   "The file where the configuration should be saved to or loaded from.",
                   StringValue (""),
                   MakeStringAccessor (&ConfigStore::SetFilename),
                   MakeStringChecker ())
    .AddAttribute ("FileFormat",
                   "Type of file format",
                   EnumValue (ConfigStore::RAW_TEXT),
                   MakeEnumAccessor (&ConfigStore::SetFileFormat),
                   MakeEnumChecker (ConfigStore::RAW_TEXT, "RawText",
                                    ConfigStore::XML, "Xml"))
    .AddAttribute ("SaveDeprecated",
                   "Save DEPRECATED attributes",
                   BooleanValue (true),
                   MakeBooleanAccessor (&ConfigStore::SetSaveDeprecated),
                   MakeBooleanChecker ())
  ;
  return tid;
}
TypeId 
ConfigStore::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}


ConfigStore::ConfigStore ()
{
  NS_LOG_FUNCTION (this);
  ObjectBase::ConstructSelf (AttributeConstructionList ());

#ifdef HAVE_LIBXML2
  if (m_fileFormat == ConfigStore::XML)
    {
      if (m_mode == ConfigStore::SAVE)
        {
          m_file = new XmlConfigSave ();
        }
      else if (m_mode == ConfigStore::LOAD)
        {
          m_file = new XmlConfigLoad ();
        }
      else 
        {
          m_file = new NoneFileConfig ();
        }
    }
#else
  if (m_fileFormat == ConfigStore::XML)
    {
      if (m_mode == ConfigStore::SAVE || m_mode == ConfigStore::LOAD)
        {
	  NS_ABORT_MSG ("ConfigStore tried to read or write an XML file but XML is not supported.");
        }
      else
        {
          m_file = new NoneFileConfig ();
        }
    }
#endif /* HAVE_LIBXML2 */

  if (m_fileFormat == ConfigStore::RAW_TEXT)
    {
      if (m_mode == ConfigStore::SAVE)
        {
          m_file = new RawTextConfigSave ();
        }
      else if (m_mode == ConfigStore::LOAD)
        {
          m_file = new RawTextConfigLoad ();
        }
      else
        {
          m_file = new NoneFileConfig ();
        }
    }
  m_file->SetFilename (m_filename);
  m_file->SetSaveDeprecated (m_saveDeprecated);

  NS_LOG_FUNCTION (this << ": format: " << m_fileFormat
                << ", mode: " << m_mode
                << ", file name: " << m_filename);
}

ConfigStore::~ConfigStore ()
{
  NS_LOG_FUNCTION (this);
  delete m_file;
  m_file = 0;
}

void 
ConfigStore::SetMode (enum Mode mode)
{
  NS_LOG_FUNCTION (this << mode);
  m_mode = mode;
}
void 
ConfigStore::SetFileFormat (enum FileFormat format)
{
  NS_LOG_FUNCTION (this << format);
  m_fileFormat = format;
}
void 
ConfigStore::SetFilename (std::string filename)
{
  NS_LOG_FUNCTION (this << filename);
  m_filename = filename;
}
void
ConfigStore::SetSaveDeprecated (bool saveDeprecated)
{
  NS_LOG_FUNCTION (this << saveDeprecated);
  m_saveDeprecated = saveDeprecated;
}

void 
ConfigStore::ConfigureAttributes (void)
{
  NS_LOG_FUNCTION (this);
  m_file->Attributes ();
}

void 
ConfigStore::ConfigureDefaults (void)
{
  NS_LOG_FUNCTION (this);
  m_file->Default ();
  m_file->Global ();
}

std::ostream &
operator << (std::ostream & os, ConfigStore::Mode & mode)
{
  switch (mode)
    {
    case ConfigStore::LOAD:      os << "LOAD";      break;
    case ConfigStore::SAVE:      os << "SAVE";      break;
    case ConfigStore::NONE:      os << "NONE";      break;
    default:                     os << "UNKNOWN";
    }
  return os;
}

std::ostream &
operator << (std::ostream & os, ConfigStore::FileFormat & format)
{
  switch (format)
    {
    case ConfigStore::XML:       os << "XML";       break;
    case ConfigStore::RAW_TEXT:  os << "RAW_TEXT";  break;
    }
  return os;
}

} // namespace ns3
