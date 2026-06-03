build_example(
  NAME mixed-wired-wireless
  SOURCE_FILES mixed-wired-wireless.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libapplications}
    ${libolsr}
    ${libnetanim}
)

build_example(
  NAME wifi-80211e-txop
  SOURCE_FILES wifi-80211e-txop.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-80211n-mimo
  SOURCE_FILES wifi-80211n-mimo.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-adhoc
  SOURCE_FILES wifi-adhoc.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-aggregation
  SOURCE_FILES wifi-aggregation.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-ap
  SOURCE_FILES wifi-ap.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-backward-compatibility
  SOURCE_FILES wifi-backward-compatibility.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-blockack
  SOURCE_FILES wifi-blockack.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-clear-channel-cmu
  SOURCE_FILES wifi-clear-channel-cmu.cc
  LIBRARIES_TO_LINK ${libinternet}
                    ${libwifi}
)

build_example(
  NAME wifi-dsss-validation
  SOURCE_FILES wifi-dsss-validation.cc
  LIBRARIES_TO_LINK ${libwifi}
)

build_example(
  NAME wifi-he-network
  SOURCE_FILES wifi-he-network.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-hidden-terminal
  SOURCE_FILES wifi-hidden-terminal.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libapplications}
    ${libflow-monitor}
)

build_example(
  NAME wifi-ht-network
  SOURCE_FILES wifi-ht-network.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-mixed-network
  SOURCE_FILES wifi-mixed-network.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-multi-tos
  SOURCE_FILES wifi-multi-tos.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-multirate
  SOURCE_FILES wifi-multirate.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libwifi}
    ${libolsr}
    ${libapplications}
    ${libflow-monitor}
)

build_example(
  NAME wifi-ofdm-he-validation
  SOURCE_FILES wifi-ofdm-he-validation.cc
  LIBRARIES_TO_LINK ${libwifi}
)

build_example(
  NAME wifi-ofdm-ht-validation
  SOURCE_FILES wifi-ofdm-ht-validation.cc
  LIBRARIES_TO_LINK ${libwifi}
)

build_example(
  NAME wifi-ofdm-validation
  SOURCE_FILES wifi-ofdm-validation.cc
  LIBRARIES_TO_LINK ${libwifi}
)

build_example(
  NAME wifi-ofdm-vht-validation
  SOURCE_FILES wifi-ofdm-vht-validation.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libwifi}
    ${libconfig-store}
    ${libstats}
)

build_example(
  NAME wifi-error-models-comparison
  SOURCE_FILES wifi-error-models-comparison.cc
  LIBRARIES_TO_LINK ${libwifi}
)

build_example(
  NAME wifi-power-adaptation-distance
  SOURCE_FILES wifi-power-adaptation-distance.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-power-adaptation-interference
  SOURCE_FILES wifi-power-adaptation-interference.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libapplications}
    ${libflow-monitor}
)

build_example(
  NAME wifi-rate-adaptation-distance
  SOURCE_FILES wifi-rate-adaptation-distance.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-simple-adhoc
  SOURCE_FILES wifi-simple-adhoc.cc
  LIBRARIES_TO_LINK ${libinternet}
                    ${libwifi}
)

build_example(
  NAME wifi-simple-adhoc-grid
  SOURCE_FILES wifi-simple-adhoc-grid.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libwifi}
    ${libolsr}
)

build_example(
  NAME wifi-simple-ht-hidden-stations
  SOURCE_FILES wifi-simple-ht-hidden-stations.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-simple-infra
  SOURCE_FILES wifi-simple-infra.cc
  LIBRARIES_TO_LINK ${libinternet}
                    ${libwifi}
)

build_example(
  NAME wifi-simple-interference
  SOURCE_FILES wifi-simple-interference.cc
  LIBRARIES_TO_LINK ${libinternet}
                    ${libwifi}
)

build_example(
  NAME wifi-sleep
  SOURCE_FILES wifi-sleep.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-spatial-reuse
  SOURCE_FILES wifi-spatial-reuse.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-spectrum-per-example
  SOURCE_FILES wifi-spectrum-per-example.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-spectrum-per-interference
  SOURCE_FILES wifi-spectrum-per-interference.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-spectrum-saturation-example
  SOURCE_FILES wifi-spectrum-saturation-example.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-tcp
  SOURCE_FILES wifi-tcp.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-timing-attributes
  SOURCE_FILES wifi-timing-attributes.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-txop-aggregation
  SOURCE_FILES wifi-txop-aggregation.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-vht-network
  SOURCE_FILES wifi-vht-network.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libapplications}
)

build_example(
  NAME wifi-wired-bridging
  SOURCE_FILES wifi-wired-bridging.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libcsma}
    ${libbridge}
    ${libapplications}
)
