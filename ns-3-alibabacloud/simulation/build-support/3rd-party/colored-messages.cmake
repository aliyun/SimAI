# Copyright (c) 2013 Fraser Hutchison
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation;
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA  02111-1307 USA
# cmake-format: off
#
# Author: Fraser Hutchison <fraser@casperlabs.io>
# Modified by: Gabriel Ferreira <gabrielcarvfer@gmail.com>
#
# Original code:
#  https://stackoverflow.com/a/19578320
# Used with permission of the author:
#  https://gitlab.com/nsnam/ns-3-dev/-/merge_requests/913#note_884829396
# cmake-format: on

# Set empty values for colors
set(ColourReset)
set(Red)
set(Green)
set(Yellow)
set(Blue)
set(Magenta)
set(Cyan)
set(White)
set(BoldRed)
set(BoldGreen)
set(BoldYellow)
set(BoldBlue)
set(BoldMagenta)
set(BoldCyan)
set(BoldWhite)

# Custom message types fallback when not colorized
set(HIGHLIGHTED_STATUS STATUS)

if(${NS3_COLORED_OUTPUT} OR "$ENV{CLICOLOR}")
  if(NOT WIN32)
    # When colorized, set color values
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[0m")
    set(ColourBold "${Esc}[1m")
    set(Red "${Esc}[31m")
    set(Green "${Esc}[32m")
    set(Yellow "${Esc}[33m")
    set(Blue "${Esc}[34m")
    set(Magenta "${Esc}[35m")
    set(Cyan "${Esc}[36m")
    set(White "${Esc}[37m")
    set(BoldRed "${Esc}[1;31m")
    set(BoldGreen "${Esc}[1;32m")
    set(BoldYellow "${Esc}[1;33m")
    set(BoldBlue "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan "${Esc}[1;36m")
    set(BoldWhite "${Esc}[1;37m")
  endif()

  # Set custom message type when colorized
  set(HIGHLIGHTED_STATUS HIGHLIGHTED_STATUS)

  # Replace the default message macro with a custom colored one
  function(message)
    list(GET ARGV 0 MessageType)
    if((${MessageType} STREQUAL FATAL_ERROR) OR (${MessageType} STREQUAL
                                                 SEND_ERROR)
    )
      list(REMOVE_AT ARGV 0)
      _message(${MessageType} "${BoldRed}${ARGV}${ColourReset}")
    elseif(MessageType STREQUAL WARNING)
      list(REMOVE_AT ARGV 0)
      _message(${MessageType} "${Yellow}${ARGV}${ColourReset}")
    elseif(MessageType STREQUAL AUTHOR_WARNING)
      list(REMOVE_AT ARGV 0)
      _message(${MessageType} "${BoldCyan}${ARGV}${ColourReset}")
    elseif(MessageType STREQUAL HIGHLIGHTED_STATUS) # Custom message type
      list(REMOVE_AT ARGV 0)
      _message(STATUS "${Yellow}${ARGV}${ColourReset}")
    elseif(MessageType STREQUAL STATUS)
      list(REMOVE_AT ARGV 0)
      _message(${MessageType} "${ARGV}")
    else()
      _message(${ARGV})
    endif()
  endfunction()
endif()
