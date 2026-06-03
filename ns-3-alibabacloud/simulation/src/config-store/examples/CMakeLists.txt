build_lib_example(
  NAME config-store-save
  SOURCE_FILES config-store-save.cc
  LIBRARIES_TO_LINK ${libcore}
                    ${libconfig-store}
)
