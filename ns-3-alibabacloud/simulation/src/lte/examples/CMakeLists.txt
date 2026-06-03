set(base_examples
    lena-cc-helper
    lena-cqi-threshold
    lena-deactivate-bearer
    lena-distributed-ffr
    lena-dual-stripe
    lena-fading
    lena-frequency-reuse
    lena-intercell-interference
    lena-ipv6-addr-conf
    lena-ipv6-ue-rh
    lena-ipv6-ue-ue
    lena-pathloss-traces
    lena-profiling
    lena-radio-link-failure
    lena-rem
    lena-rem-sector-antenna
    lena-rlc-traces
    lena-simple
    lena-simple-epc
    lena-simple-epc-backhaul
    lena-uplink-power-control
    lena-x2-handover
    lena-x2-handover-measures
)

foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK ${liblte}
  )
endforeach()

if(${ENABLE_EMU})
  build_lib_example(
    NAME lena-simple-epc-emu
    SOURCE_FILES lena-simple-epc-emu.cc
    LIBRARIES_TO_LINK ${liblte}
                      ${fd-net-device}
  )
endif()
