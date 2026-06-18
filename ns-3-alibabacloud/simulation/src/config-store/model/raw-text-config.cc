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

#include "raw-text-config.h"
#include "attribute-iterator.h"
#include "attribute-default-iterator.h"
#include "ns3/global-value.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/config.h"

#include <istream>
#include <sstream>
#include <algorithm>
#include <functional>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RawTextConfig");

RawTextConfigSave::RawTextConfigSave ()
  : m_os (0)
{
  NS_LOG_FUNCTION (this);
}
RawTextConfigSave::~RawTextConfigSave ()
{
  NS_LOG_FUNCTION (this);
  if (m_os != 0)
    {
      m_os->close ();
    }
  delete m_os;
  m_os = 0;
}
void
RawTextConfigSave::SetFilename (std::string filename)
{
  NS_LOG_FUNCTION (this << filename);
  m_os = new std::ofstream ();
  m_os->open (filename.c_str (), std::ios::out);
}
void
RawTextConfigSave::Default (void)
{
  NS_LOG_FUNCTION (this);
  class RawTextDefaultIterator : public AttributeDefaultIterator
  {
public:
    RawTextDefaultIterator (std::ostream *os) {
      m_os = os;
    }
    void SetSaveDeprecated (bool saveDeprecated) {
      m_saveDeprecated = saveDeprecated;
    }

private:
    virtual void StartVisitTypeId (std::string name) {
      m_typeId = name;
    }
    virtual void DoVisitAttribute (std::string name, std::string defaultValue) {
      NS_LOG_DEBUG ("Saving " << m_typeId << "::" << name);
      TypeId tid = TypeId::LookupByName (m_typeId);
      ns3::TypeId::SupportLevel supportLevel = TypeId::SupportLevel::SUPPORTED;
      for (std::size_t i = 0; i < tid.GetAttributeN (); i++)
        {
          struct TypeId::AttributeInformation tmp = tid.GetAttribute (i);
          if (tmp.name == name)
            {
              supportLevel = tmp.supportLevel;
              break;
            }
        }
      if (supportLevel == TypeId::SupportLevel::OBSOLETE)
        {
          NS_LOG_WARN ("Global attribute "
                       << m_typeId << "::" << name
                       << " was not saved because it is OBSOLETE");
        }
      else if ((supportLevel == TypeId::SupportLevel::DEPRECATED) && (m_saveDeprecated == false))
        {
          NS_LOG_WARN ("Global attribute " << m_typeId << "::" << name
                                           << " was not saved because it is DEPRECATED");
        }
      else
        {
          *m_os << "default " << m_typeId << "::" << name << " \"" << defaultValue << "\""
                << std::endl;
        }
      }
    std::string m_typeId;
    std::ostream *m_os;
    bool m_saveDeprecated;
  };

  RawTextDefaultIterator iterator = RawTextDefaultIterator (m_os);
  iterator.SetSaveDeprecated (m_saveDeprecated);
  iterator.Iterate ();
}
void
RawTextConfigSave::Global (void)
{
  NS_LOG_FUNCTION (this);
  for (GlobalValue::Iterator i = GlobalValue::Begin (); i != GlobalValue::End (); ++i)
    {
      StringValue value;
      (*i)->GetValue (value);
      NS_LOG_LOGIC ("Saving " << (*i)->GetName ());
      *m_os << "global " << (*i)->GetName () << " \"" << value.Get () << "\"" << std::endl;
    }
}
void
RawTextConfigSave::Attributes (void)
{
  NS_LOG_FUNCTION (this);
  class RawTextAttributeIterator : public AttributeIterator
  {
public:
    RawTextAttributeIterator (std::ostream *os)
      : m_os (os) {}
    void
    SetSaveDeprecated (bool saveDeprecated) {
      m_saveDeprecated = saveDeprecated;
    }

private:
    virtual void DoVisitAttribute (Ptr<Object> object, std::string name) {
      StringValue str;

      ns3::TypeId::SupportLevel supportLevel = TypeId::SupportLevel::SUPPORTED;
      TypeId tid = object->GetInstanceTypeId ();

      for (std::size_t i = 0; i < tid.GetAttributeN (); i++)
        {
          struct TypeId::AttributeInformation tmp = tid.GetAttribute (i);
          if (tmp.name == name)
            {
              supportLevel = tmp.supportLevel;
              break;
            }
        }
      if (supportLevel == TypeId::SupportLevel::OBSOLETE)
        {
          NS_LOG_WARN ("Attribute " << GetCurrentPath ()
                                    << " was not saved because it is OBSOLETE");
        }
      else if ((supportLevel == TypeId::SupportLevel::DEPRECATED) && (m_saveDeprecated == false))
        {
          NS_LOG_WARN ("Attribute " << GetCurrentPath ()
                                    << " was not saved because it is DEPRECATED");
        }
      else
        {
          object->GetAttribute (name, str);
          NS_LOG_DEBUG ("Saving " << GetCurrentPath ());
          *m_os << "value " << GetCurrentPath () << " \"" << str.Get () << "\"" << std::endl;
        }
    }
    std::ostream *m_os;
    bool m_saveDeprecated;
  };

  RawTextAttributeIterator iter = RawTextAttributeIterator (m_os);
  iter.SetSaveDeprecated (m_saveDeprecated);
  iter.Iterate ();
}

RawTextConfigLoad::RawTextConfigLoad ()
  : m_is (0)
{
  NS_LOG_FUNCTION (this);
}
RawTextConfigLoad::~RawTextConfigLoad ()
{
  NS_LOG_FUNCTION (this);
  if (m_is != 0)
    {
      m_is->close ();
      delete m_is;
      m_is = 0;
    }
}
void
RawTextConfigLoad::SetFilename (std::string filename)
{
  NS_LOG_FUNCTION (this << filename);
  m_is = new std::ifstream ();
  m_is->open (filename.c_str (), std::ios::in);
}
std::string
RawTextConfigLoad::Strip (std::string value)
{
  NS_LOG_FUNCTION (this << value);
  std::string::size_type start = value.find ("\"");
  std::string::size_type end = value.find ("\"", 1);
  NS_ABORT_MSG_IF (start != 0, "Ill-formed attribute value: " << value);
  NS_ABORT_MSG_IF (end != value.size () - 1, "Ill-formed attribute value: " << value);
  return value.substr (start+1, end-start-1);
}

void
RawTextConfigLoad::Default (void)
{
  NS_LOG_FUNCTION (this);
  m_is->clear ();
  m_is->seekg (0);
  std::string type, name, value;
  for (std::string line; std::getline (*m_is, line);)
    {
      if (!ParseLine (line, type, name, value)) 
        {
          continue;
        }

      NS_LOG_DEBUG ("type=" << type << ", name=" << name << ", value=" << value);
      value = Strip (value);
      if (type == "default")
        {
          Config::SetDefault (name, StringValue (value));
        }
      name.clear ();
      type.clear ();
      value.clear ();
    }
}
void
RawTextConfigLoad::Global (void)
{
  NS_LOG_FUNCTION (this);
  m_is->clear ();
  m_is->seekg (0);
  std::string type, name, value;
  for (std::string line; std::getline (*m_is, line);)
    {
      if (!ParseLine (line, type, name, value)) 
        {
          continue;
        }

      NS_LOG_DEBUG ("type=" << type << ", name=" << name << ", value=" << value);
      value = Strip (value);
      if (type == "global")
        {
          Config::SetGlobal (name, StringValue (value));
        }
      name.clear ();
      type.clear ();
      value.clear ();
    }
}
void
RawTextConfigLoad::Attributes (void)
{
  NS_LOG_FUNCTION (this);
  m_is->clear ();
  m_is->seekg (0);
  std::string type, name, value;
  for (std::string line; std::getline (*m_is, line);)
    {
      if (!ParseLine (line, type, name, value)) 
        {
          continue;
        }

      NS_LOG_DEBUG ("type=" << type << ", name=" << name << ", value=" << value);
      value = Strip (value);
      if (type == "value")
        {
          Config::Set (name, StringValue (value));
        }
      name.clear ();
      type.clear ();
      value.clear ();
    }
}

bool
RawTextConfigLoad::ParseLine (const std::string &line, std::string &type, std::string &name, std::string &value)
{
  NS_LOG_FUNCTION (this << line << type << name << value);

  // check for blank line
  {
    std::istringstream iss (line);
    iss >> std::ws;     // remove all blanks line
    if (!iss.good ())   // eofbit set if no non-blanks
      {
        return false;
      }
  }

  if (line.front () == '#')
    {
      return false;     // comment line
    }

  // for multiline values, append line to value if type and name not empty
  if (type.empty () && name.empty ())
    {
      std::istringstream iss (line);
      iss >> type >> name >> std::ws;
      std::getline (iss, value);   // remaining line, includes embedded spaces
    }
  else
    {
      value.append (line);
    }

  // two quotes in value signifies a completed (possibly multi-line)
  // config-store entry, return True to signal load function to
  // validate value (see Strip method) and set attribute
  return std::count (value.begin (), value.end (), '"') == 2;
}


} // namespace ns3
