#
#   Copyright 2014 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
#

# Try to find CUEW library and include path.
# Once done this will define
#
# CUEW_FOUND
# CUEW_INCLUDE_DIR
# CUEW_LIBRARY
#

include (FindPackageHandleStandardArgs)

if(WIN32)
  set(_cuew_SEARCH_DIRS
    "${CUEW_LOCATION}/include"
    "$ENV{CUEW_LOCATION}/include"
    "$ENV{PROGRAMFILES}/CUEW/include"
    "${PROJECT_SOURCE_DIR}/extern/cuew/include"
  )
else()
  set(_cuew_SEARCH_DIRS
      "${CUEW_LOCATION}"
      "$ENV{CUEW_LOCATION}"
      /usr
      /usr/local
      /sw
      /opt/local
      /opt/lib/cuew
  )
endif()

find_path(CUEW_INCLUDE_DIR
  NAMES
    cuew.h
  HINTS
    ${_cuew_SEARCH_DIRS}
  PATH_SUFFIXES
    include
  NO_DEFAULT_PATH
  DOC "The directory where cuew.h resides")

find_library(CUEW_LIBRARY
  NAMES
    CUEW cuew
  PATHS
    ${_cuew_SEARCH_DIRS}
  PATH_SUFFIXES
    lib lib64
  NO_DEFAULT_PATH
  DOC "The CUEW library")

find_package_handle_standard_args(CUEW
    REQUIRED_VARS
        CUEW_INCLUDE_DIR
        CUEW_LIBRARY
)
