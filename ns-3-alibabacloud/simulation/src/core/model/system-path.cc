/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "system-path.h"
#include "fatal-error.h"
#include "assert.h"
#include "log.h"
#include "ns3/core-config.h"

#include <algorithm>
#include <cstdlib>  // getenv
#include <cerrno>
#include <cstring>  // strlen
#include <tuple>

#if defined (HAVE_DIRENT_H) && defined (HAVE_SYS_TYPES_H)
/** Do we have an \c opendir function? */
#define HAVE_OPENDIR
#include <sys/types.h>
#include <dirent.h>
#endif
#if defined (HAVE_SYS_STAT_H) && defined (HAVE_SYS_TYPES_H)
/** Do we have a \c makedir function? */
#define HAVE_MKDIR_H
#include <sys/types.h>
#include <sys/stat.h>
#endif
#include <sstream>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif /* __APPLE__ */

#ifdef __FreeBSD__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

/**
 * \def SYSTEM_PATH_SEP
 * System-specific path separator used between directory names.
 */
#if defined (__win32__)
#define SYSTEM_PATH_SEP "\\"
#else
#define SYSTEM_PATH_SEP "/"
#endif

/**
 * \file
 * \ingroup systempath
 * ns3::SystemPath implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SystemPath");

// unnamed namespace for internal linkage
namespace {
/**
 * \ingroup systempath
 * Get the list of files located in a file system directory with error.
 *
 * \param [in] path A path which identifies a directory
 * \return Tuple with a list of the filenames which are located in the input directory or error flag \c true if directory doesn't exist.
 */
std::tuple<std::list<std::string>, bool> ReadFilesNoThrow (std::string path)
{
  NS_LOG_FUNCTION (path);
  std::list<std::string> files;

#if defined HAVE_OPENDIR
  DIR *dp = opendir (path.c_str ());
  if (dp == NULL)
    {
      return std::make_tuple (files, true);
    }
  struct dirent *de = readdir (dp);
  while (de != 0)
    {
      files.push_back (de->d_name);
      de = readdir (dp);
    }
  closedir (dp);
#elif defined (HAVE_FIND_FIRST_FILE)
  /** \todo untested */
  HANDLE hFind;
  WIN32_FIND_DATA fileData;

  hFind = FindFirstFile (path.c_str (), &FindFileData);
  if (hFind == INVALID_HANDLE_VALUE)
    {
      return std::make_tuple (files, true);
    }
  do
    {
      files.push_back (fileData.cFileName);
    }
  while (FindNextFile (hFind, &fileData));
  FindClose (hFind);
#else
#error "No support for reading a directory on this platform"
#endif
  return std::make_tuple (files, false);
}

} // unnamed namespace

namespace SystemPath {

/**
 * \ingroup systempath
 * \brief Get the directory path for a file.
 *
 * This is an internal function (by virtue of not being
 * declared in a \c .h file); the public API is FindSelfDirectory().
 *
 * \param [in] path The full path to a file.
 * \returns The full path to the containing directory.
 */
std::string Dirname (std::string path)
{
  NS_LOG_FUNCTION (path);
  std::list<std::string> elements = Split (path);
  std::list<std::string>::const_iterator last = elements.end ();
  last--;
  return Join (elements.begin (), last);
}

std::string FindSelfDirectory (void)
{
  /**
   * This function returns the path to the running $PREFIX.
   * Mac OS X: _NSGetExecutablePath() (man 3 dyld)
   * Linux: readlink /proc/self/exe
   * Solaris: getexecname()
   * FreeBSD: sysctl CTL_KERN KERN_PROC KERN_PROC_PATHNAME -1
   * BSD with procfs: readlink /proc/curproc/file
   * Windows: GetModuleFileName() with hModule = NULL
   */
  NS_LOG_FUNCTION_NOARGS ();
  std::string filename;
#if defined(__linux__)
  {
    ssize_t size = 1024;
    char *buffer = (char*)malloc (size);
    memset (buffer, 0, size);
    int status;
    while (true)
      {
        status = readlink ("/proc/self/exe", buffer, size);
        if (status != 1 || (status == -1 && errno != ENAMETOOLONG))
          {
            break;
          }
        size *= 2;
        free (buffer);
        buffer = (char*)malloc (size);
        memset (buffer, 0, size);
      }
    if (status == -1)
      {
        NS_FATAL_ERROR ("Oops, could not find self directory.");
      }
    filename = buffer;
    free (buffer);
  }
#elif defined (__win32__)
  {
    /** \todo untested. it should work if code is compiled with
     *  LPTSTR = char *
     */
    DWORD size = 1024;
    LPTSTR lpFilename = (LPTSTR) malloc (sizeof(TCHAR) * size);
    DWORD status = GetModuleFilename (0, lpFilename, size);
    while (status == size)
      {
        size = size * 2;
        free (lpFilename);
        lpFilename = (LPTSTR) malloc (sizeof(TCHAR) * size);
        status = GetModuleFilename (0, lpFilename, size);
      }
    NS_ASSERT (status != 0);
    filename = lpFilename;
    free (lpFilename);
  }
#elif defined (__APPLE__)
  {
    uint32_t bufsize = 1024;
    char *buffer = (char *) malloc (bufsize);
    NS_ASSERT (buffer != 0);
    int status = _NSGetExecutablePath (buffer, &bufsize);
    if (status == -1)
      {
        free (buffer);
        buffer = (char *) malloc (bufsize);
        status = _NSGetExecutablePath (buffer, &bufsize);
      }
    NS_ASSERT (status == 0);
    filename = buffer;
    free (buffer);
  }
#elif defined (__FreeBSD__)
  {
    int     mib[4];
    std::size_t  bufSize = 1024;
    char   *buf = (char *) malloc (bufSize);

    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PATHNAME;
    mib[3] = -1;

    sysctl (mib, 4, buf, &bufSize, NULL, 0);
    filename = buf;
  }
#endif
  return Dirname (filename);
}

std::string Append (std::string left, std::string right)
{
  // removing trailing separators from 'left'
  NS_LOG_FUNCTION (left << right);
  while (true)
    {
      std::string::size_type lastSep = left.rfind (SYSTEM_PATH_SEP);
      if (lastSep != left.size () - 1)
        {
          break;
        }
      left = left.substr (0, left.size () - 1);
    }
  std::string retval = left + SYSTEM_PATH_SEP + right;
  return retval;
}

std::list<std::string> Split (std::string path)
{
  NS_LOG_FUNCTION (path);
  std::list<std::string> retval;
  std::string::size_type current = 0, next = 0;
  next = path.find (SYSTEM_PATH_SEP, current);
  while (next != std::string::npos)
    {
      std::string item = path.substr (current, next - current);
      retval.push_back (item);
      current = next + 1;
      next = path.find (SYSTEM_PATH_SEP, current);
    }
  std::string item = path.substr (current, next - current);
  retval.push_back (item);
  return retval;
}

std::string Join (std::list<std::string>::const_iterator begin,
                  std::list<std::string>::const_iterator end)
{
  NS_LOG_FUNCTION (*begin << *end);
  std::string retval = "";
  for (std::list<std::string>::const_iterator i = begin; i != end; i++)
    {
      if (*i == "")
        {
          // skip empty strings in the path list
          continue;
        }
      else if (i == begin)
        {
          retval = *i;
        }
      else
        {
          retval = retval + SYSTEM_PATH_SEP + *i;
        }
    }
  return retval;
}

std::list<std::string> ReadFiles (std::string path)
{
  NS_LOG_FUNCTION (path);
  bool err;
  std::list<std::string> files;
  std::tie (files, err) = ReadFilesNoThrow (path);
  if (err)
    {
      NS_FATAL_ERROR ("Could not open directory=" << path);
    }
  return files;
}

std::string
MakeTemporaryDirectoryName (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  char *path = NULL;

  path = std::getenv ("TMP");
  if (path == NULL || std::strlen (path) == 0)
    {
      path = std::getenv ("TEMP");
      if (path == NULL || std::strlen (path) == 0)
        {
          path = const_cast<char *> ("/tmp");
        }
    }

  //
  // Just in case the user wants to go back and find the output, we give
  // a hint as to which dir we created by including a time hint.
  //
  time_t now = time (NULL);
  struct tm *tm_now = localtime (&now);
  //
  // But we also randomize the name in case there are multiple users doing
  // this at the same time
  //
  srand (time (0));
  long int n = rand ();

  //
  // The final path to the directory is going to look something like
  //
  //   /tmp/ns3.14.30.29.32767
  //
  // The first segment comes from one of the temporary directory env
  // variables or /tmp if not found.  The directory name starts with an
  // identifier telling folks who is making all of the temp directories
  // and then the local time (in this case 14.30.29 -- which is 2:30 and
  // 29 seconds PM).
  //
  std::ostringstream oss;
  oss << path << SYSTEM_PATH_SEP << "ns-3." << tm_now->tm_hour << "."
      << tm_now->tm_min << "." << tm_now->tm_sec << "." << n;

  return oss.str ();
}

void
MakeDirectories (std::string path)
{
  NS_LOG_FUNCTION (path);

  // Make sure all directories on the path exist
  std::list<std::string> elements = Split (path);
  auto i = elements.begin ();
  while (i != elements.end ())
    {
      if (*i == "")
        {
          NS_LOG_LOGIC ("skipping empty directory name");
          ++i;
          continue;
        }
      NS_LOG_LOGIC ("creating directory " << *i);
      ++i;  // Now points to one past the directory we want to create
      std::string tmp = Join (elements.begin (), i);
      bool makeDirErr = false;

#if defined(HAVE_MKDIR_H)
      makeDirErr = mkdir (tmp.c_str (), S_IRWXU);
#endif

      if (makeDirErr)
        {
          NS_LOG_ERROR ("failed creating directory " << tmp);
        }
    }
}

bool
Exists (const std::string path)
{
  NS_LOG_FUNCTION (path);

  bool err;
  auto dirpath = Dirname (path);
  std::list<std::string> files;
  tie (files, err) = ReadFilesNoThrow (dirpath);
  if (err)
    {
      // Directory doesn't exist
      NS_LOG_LOGIC ("directory doesn't exist: " << dirpath);
      return false;
    }
  NS_LOG_LOGIC ("directory exists: " << dirpath);

  // Check if the file itself exists
  auto tokens = Split (path);
  std::string file = tokens.back ();

  if (file == "")
    {
      // Last component was a directory, not a file name
      // We already checked that the directory exists,
      // so return true
      NS_LOG_LOGIC ("directory path exists: " << path);
      return true;
    }

  files = ReadFiles (dirpath);

  auto it = std::find (files.begin (), files.end (), file);
  if (it == files.end ())
    {
      // File itself doesn't exist
      NS_LOG_LOGIC ("file itself doesn't exist: " << file);
      return false;
    }

  NS_LOG_LOGIC ("file itself exists: " << file);
  return true;

}  // Exists()


} // namespace SystemPath

} // namespace ns3
