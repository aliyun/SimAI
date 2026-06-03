build_lib(
  LIBNAME point-to-point-layout
  SOURCE_FILES
    model/point-to-point-dumbbell.cc
    model/point-to-point-grid.cc
    model/point-to-point-star.cc
  HEADER_FILES
    model/point-to-point-dumbbell.h
    model/point-to-point-grid.h
    model/point-to-point-star.h
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libmobility}
)
