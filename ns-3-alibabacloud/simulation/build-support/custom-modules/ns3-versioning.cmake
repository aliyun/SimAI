# Determine if the git repository is an ns-3 repository
#
# A repository is considered an ns-3 repository if it has at least one tag that
# matches the regex ns-3*

function(check_git_repo_has_ns3_tags HAS_TAGS GIT_VERSION_TAG)
  execute_process(
    COMMAND ${GIT} describe --tags --abbrev=0 --match ns-3.[0-9]*
    OUTPUT_VARIABLE GIT_TAG_OUTPUT ERROR_QUIET
  )

  # Result will be empty in case of a shallow clone or no git repo
  if(NOT GIT_TAG_OUTPUT)
    return()
  endif()

  string(REPLACE "\r" "" GIT_TAG_OUTPUT ${GIT_TAG_OUTPUT}) # remove CR (carriage
                                                           # return)
  string(REPLACE "\n" "" GIT_TAG_OUTPUT ${GIT_TAG_OUTPUT}) # remove LF (line
                                                           # feed)

  # Check if tag exists and return values to the caller
  string(LENGTH GIT_TAG_OUTPUT GIT_TAG_OUTPUT_LEN)
  set(${HAS_TAGS} FALSE PARENT_SCOPE)
  set(${GIT_VERSION_TAG} "" PARENT_SCOPE)
  if(${GIT_TAG_OUTPUT_LEN} GREATER "0")
    set(${HAS_TAGS} TRUE PARENT_SCOPE)
    set(${GIT_VERSION_TAG} ${GIT_TAG_OUTPUT} PARENT_SCOPE)
  endif()
endfunction()

# Function to generate version fields from an ns-3 git repository
function(check_ns3_closest_tags CLOSEST_TAG VERSION_TAG_DISTANCE
         VERSION_COMMIT_HASH VERSION_DIRTY_FLAG
)
  execute_process(
    COMMAND ${GIT} describe --tags --dirty --long
    OUTPUT_VARIABLE GIT_TAG_OUTPUT
  )

  string(REPLACE "\r" "" GIT_TAG_OUTPUT ${GIT_TAG_OUTPUT}) # remove CR (carriage
                                                           # return)
  string(REPLACE "\n" "" GIT_TAG_OUTPUT ${GIT_TAG_OUTPUT}) # remove LF (line
                                                           # feed)

  # Split ns-3.<version>.<patch>(-RC<digit>)-distance-commit(-dirty) into a list
  # ns;3.<version>.<patch>;(RC<digit);distance;commit(;dirty)
  string(REPLACE "-" ";" TAG_LIST "${GIT_TAG_OUTPUT}")

  list(GET TAG_LIST 0 NS)
  list(GET TAG_LIST 1 VERSION)

  if(${GIT_TAG_OUTPUT} MATCHES "RC")
    list(GET TAG_LIST 2 RC)
    set(${CLOSEST_TAG} "${NS}-${VERSION}-${RC}" PARENT_SCOPE)
    list(GET TAG_LIST 3 DISTANCE)
    list(GET TAG_LIST 4 HASH)
  else()
    set(${CLOSEST_TAG} "${NS}-${VERSION}" PARENT_SCOPE)
    list(GET TAG_LIST 2 DISTANCE)
    list(GET TAG_LIST 3 HASH)
  endif()

  set(${VERSION_TAG_DISTANCE} ${DISTANCE} PARENT_SCOPE)
  set(${VERSION_COMMIT_HASH} "${HASH}" PARENT_SCOPE)

  set(${VERSION_DIRTY_FLAG} 0 PARENT_SCOPE)
  if(${GIT_TAG_OUTPUT} MATCHES "dirty")
    set(${VERSION_DIRTY_FLAG} 1 PARENT_SCOPE)
  endif()
endfunction()

function(configure_embedded_version)
  if(NOT ${NS3_ENABLE_BUILD_VERSION})
    add_custom_target(
      check-version COMMAND echo Build version feature disabled. Reconfigure
                            ns-3 with NS3_ENABLE_BUILD_VERSION=ON
    )
    return()
  endif()

  mark_as_advanced(GIT)
  find_program(GIT git)
  if("${GIT}" STREQUAL "GIT-NOTFOUND")
    message(
      STATUS "Git was not found. Version related targets won't be enabled"
    )
    return()
  endif()

  check_git_repo_has_ns3_tags(HAS_NS3_TAGS NS3_VERSION_TAG)

  set(version_cache_file_template
      ${PROJECT_SOURCE_DIR}/build-support/version.cache.in
  )
  set(version_cache_file ${PROJECT_SOURCE_DIR}/src/core/model/version.cache)

  # Check if ns-3 git tags were found or at least version.cache file exists in
  # the src/core/model
  if((NOT HAS_NS3_TAGS) AND (NOT (EXISTS ${version_cache_file})))
    message(
      FATAL_ERROR
        "The ns-3 git commit history or the version.cache file are required to embed build version into libraries."
    )
    return()
  endif()

  # If git tags were found, extract the information
  if(HAS_NS3_TAGS)
    check_ns3_closest_tags(
      NS3_VERSION_CLOSEST_TAG NS3_VERSION_TAG_DISTANCE NS3_VERSION_COMMIT_HASH
      NS3_VERSION_DIRTY_FLAG
    )
    # Split commit tag (ns-3.<minor>[.patch][-RC<digit>]) into
    # (ns;3.<minor>[.patch];[-RC<digit>]):
    string(REPLACE "-" ";" NS3_VER_LIST ${NS3_VERSION_TAG})
    list(LENGTH NS3_VER_LIST NS3_VER_LIST_LEN)

    # Get last version tag fragment (RC<digit>)
    set(RELEASE_CANDIDATE " ")
    if(${NS3_VER_LIST_LEN} GREATER 2)
      list(GET NS3_VER_LIST 2 RELEASE_CANDIDATE)
    endif()

    # Get 3.<minor>[.patch]
    list(GET NS3_VER_LIST 1 VERSION_STRING)
    # Split into a list 3;<minor>[;patch]
    string(REPLACE "." ";" VERSION_LIST ${VERSION_STRING})
    list(LENGTH VERSION_LIST VER_LIST_LEN)

    list(GET VERSION_LIST 0 NS3_VERSION_MAJOR)
    if(${VER_LIST_LEN} GREATER 1)
      list(GET VERSION_LIST 1 NS3_VERSION_MINOR)
      if(${VER_LIST_LEN} GREATER 2)
        list(GET VERSION_LIST 2 NS3_VERSION_PATCH)
      else()
        set(NS3_VERSION_PATCH "00")
      endif()
    endif()

    # Transform list with 1 entry into strings
    set(NS3_VERSION_MAJOR "${NS3_VERSION_MAJOR}")
    set(NS3_VERSION_MINOR "${NS3_VERSION_MINOR}")
    set(NS3_VERSION_PATCH "${NS3_VERSION_PATCH}")
    set(NS3_VERSION_TAG "${NS3_VERSION_TAG}")
    set(NS3_VERSION_RELEASE_CANDIDATE "${RELEASE_CANDIDATE}")
    if(${NS3_VERSION_RELEASE_CANDIDATE} STREQUAL " ")
      set(NS3_VERSION_RELEASE_CANDIDATE \"\")
    endif()
    set(NS3_VERSION_BUILD_PROFILE ${cmakeBuildType})

    # Create version.cache file
    configure_file(${version_cache_file_template} ${version_cache_file} @ONLY)
  else()
    # Consume version.cache created previously
    file(STRINGS ${version_cache_file} version_cache_contents)
    foreach(line ${version_cache_contents})
      # Remove white spaces e.g. CLOSEST_TAG = '"ns-3.35"' =>
      # CLOSEST_TAG='"ns-3.35"'
      string(REPLACE " " "" line "${line}")

      # Transform line into list  CLOSEST_TAG='"ns-3.35"' =>
      # CLOSEST_TAG;'"ns-3.35"'
      string(REPLACE "=" ";" line "${line}")

      # Replace single and double quotes used by python
      string(REPLACE "'" "" line "${line}")
      string(REPLACE "\"" "" line "${line}")

      # Get key and value
      list(GET line 0 varname)
      list(GET line 1 varvalue)

      # If value is empty, replace with an empty string, assume its the release
      # candidate string
      if(NOT varvalue)
        set(varvalue "\"\"")
      endif()

      # Define version variables with the NS3_ prefix to configure
      # version-defines.h header
      set(NS3_${varname} ${varvalue})
    endforeach()
    set(NS3_VERSION_CLOSEST_TAG ${NS3_CLOSEST_TAG})
  endif()

  set(DIRTY)
  if(${NS3_VERSION_DIRTY_FLAG})
    set(DIRTY "-dirty")
  endif()

  # Add check-version target
  set(version
      ${NS3_VERSION_TAG}+${NS3_VERSION_TAG_DISTANCE}@${NS3_VERSION_COMMIT_HASH}${DIRTY}-${build_profile}
  )
  string(REPLACE "\"" "" version "${version}")
  add_custom_target(check-version COMMAND echo ns-3 version: ${version})

  # Enable embedding build version
  add_definitions(-DENABLE_BUILD_VERSION=1)
  configure_file(
    build-support/version-defines-template.h
    ${CMAKE_HEADER_OUTPUT_DIRECTORY}/version-defines.h
  )
  set(ENABLE_BUILD_VERSION True PARENT_SCOPE)
endfunction()
