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

#include "file-config.h"

namespace ns3 {

FileConfig::~FileConfig ()
{
}

void
FileConfig::SetSaveDeprecated (bool saveDeprecated)
{
  m_saveDeprecated = saveDeprecated;
}

NoneFileConfig::NoneFileConfig ()
{
}
NoneFileConfig::~NoneFileConfig ()
{
}
void 
NoneFileConfig::SetFilename (std::string filename)
{}
void 
NoneFileConfig::Default (void)
{}
void 
NoneFileConfig::Global (void)
{}
void
NoneFileConfig::Attributes (void)
{}

} // namespace ns3
