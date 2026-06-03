#!/bin/bash
# -*- indent-tabs-mode:nil; -*-


# Process doxygen log to generate sorted list of top offenders.
#

me=$(basename $0)
DIR="$(dirname $0)"
# Trick to get the absolute path, since doxygen prefixes errors that way
ROOT=$(cd "$DIR/.."; pwd -P)

# Known log files
STANDARDLOGFILE=doxygen.log
WARNINGSLOGFILE=doxygen.warnings.log
# Default choice:  generate it
LOG="$DIR/$WARNINGSLOGFILE"
# Verbose log
VERBLOG="$DIR/doxygen.verbose.log"


# Options ------------------------------
#

# One line sysnopsis, continue
function synopsis_short
{
    echo "Usage: $me [-beithv] [-s <log-file> | -l | -w] [-m <module>] [-f <regex>] [-F <regex>]"
}

# Two line synopsis, then exit with error
function synopsis
{
    synopsis_short
    echo "    -h  For detailed usage"
    exit 1
}

# Full help, then exit with no error
function usage
{
    synopsis_short
    cat <<-EOF

    Run doxygen to generate all errors; report error counts
    by module and file.

    -i  Skip the build, try print-introspected-doxygen anyway.

    -s  Skip doxygen run; use existing <log-file>.
    -w  Skip doxygen run; use existing warnings log doc/$WARNINGSLOGFILE
    -l  Skip doxygen run; use the normal doxygen log doc/$STANDARDLOGFILE

    -b  Omit the blacklist filter of files whose warnings we ignore
    -e  Filter out warnings from */examples/*
    -t  Filter out warnings from */test/*
    -m  Only include files matching src/<module>
    -f  Only include files matching the <regex>
    -F  Exclude files matching the <regex>

    -v  Show detailed output from each step.
    -h  Print this usage message

    The default behavior is to modify doxygen.conf temporarily to
    report all undocumented elements, and to reduce the run time.
    The output of this special run is kept in doc/$WARNINGSLOGFILE.
    To further reduce the run time, the -i option also skips
    print-introspected-doxygen, so ns3 doesn\'t have to compile
    any modified files at all.

    The -f, -l, and -s options skip the doxygen run altogether.
    The first two use a specified or the standard log file;
    the -s option uses the warnings log from a prior run.
    Only the first of -f <log-file>, -s, or -l will have effect.

    The -e and -t options exclude examples and test directories
    from the counts.  The -m option only includes a specific module.
    The -F option only includes files (or warnings) matching the <regex>.
    The -m and -F options append the relevant warnings after the
    numerical report.  These can be used in any combination.

EOF
    exit 0
}

# Messaging ----------------------------
#

# Arg -v Verbosity level
verbosity=0

function verbose
{
    if [ "$1" == "-n" ]; then
        if [ $verbosity -eq 1 ]; then
            echo "$me:  ${2}..."
        else
            echo -n "${2}..."
        fi
    elif [ $verbosity -eq 1 ]; then
        echo "$me:  $1 $2"
    else
        echo "$2"
    fi
}

# Use file handle 6 for verbose output
rm -f $VERBLOG
exec 6>$VERBLOG

function status_report
{
    local status="$1"
    local long_msg="$2"
    local exitonerr="${3:-yes}"
    if [ $status -eq 0 ]; then
        [[ $verbosity -eq 1 && -e $VERBLOG ]] && cat $VERBLOG
        verbose "$long_msg "  "done."
        [[ -e $VERBLOG ]] && rm -f $VERBLOG
    else
        if [ $exitonerr == "yes" ]; then
            verbose "$long_msg "  "FAILED.  Details:"
            [[ -e $VERBLOG ]] && cat $VERBLOG && rm -f $VERBLOG
            exit 1
        else
            verbose "$long_msg "  "FAILED, continuing"
            [[ -e $VERBLOG ]] && cat $VERBLOG && rm -f $VERBLOG
        fi
    fi
}


# Argument processing ------------------
#

# -f argument
use_filearg=0
logfile_arg=
# -l
use_standard=0
# skip doxygen run; using existing log file
skip_doxy=0
# skip print-introspected-doxygen, avoiding a build
skip_intro=0

# Filtering flags
filter_blacklist=1
filter_examples=0
filter_test=0
explicit_m_option=0
filter_module=""
explicit_f_option=0
filter_in=""
filter_out=""

while getopts :bef:F:hilm:s:tvw option ; do

    case $option in
    (b)  filter_blacklist=0       ;;

    (e)  filter_examples=1        ;;

    (f)  filter_in="$OPTARG"
         explicit_f_option=1
         ;;

    (F)  filter_out="$OPTARG"     ;;

    (h)  usage                    ;;

    (i)  skip_intro=1             ;;

    (l)  use_standard=1           ;;

    (m)  filter_module="$OPTARG"
         explicit_m_option=1
         ;;

    (s)  use_filearg=1
         logfile_arg="$OPTARG"
         ;;

    (t)  filter_test=1            ;;

    (v)  verbosity=1
         exec 6>&1
         ;;

    (w)  use_filearg=1
         logfile_arg="$DIR/$WARNINGSLOGFILE"
         ;;

    (:)  echo "$me: Missing argument to -$OPTARG" ; synopsis ;;

    (\?) echo "$me: Invalid option: -$OPTARG"     ; synopsis ;;

    esac
done

function checklogfile
{
    if [ -e "$1" ] ; then
        skip_doxy=1
        LOG="$1"
    else
        echo "$me: log file $1 does not exist."
        synopsis
    fi
}

# Log file -----------------------------
#

if [[ $use_filearg -eq 1 && "${logfile_arg:-}" != "" ]] ; then
    checklogfile "$logfile_arg"
elif [ $use_standard -eq 1 ]; then
    checklogfile "$DIR/$STANDARDLOGFILE"
fi

# Log filters --------------------------
#

# Append a regular expression to a parameter
#  with '\|' alternation operator if the parameter wasn't empty to begin with.
function REappend
{
    local param="$1"
    local token="$2"

    eval "${param}=\"${!param:-}${!param:+\\|}$token\""
}

# Explicit -f or -m with empty args should filter out all, not pass all
[[ $explicit_f_option -eq 1 && "${filter_in:-}" == "" ]] && filter_out=".*"
[[ $explicit_m_option -eq 1 && "${filter_module:-}" == "" ]] && filter_out=".*"

# Filter in regular expression for -m and -f
filter_inRE=""
[[ "$filter_module" != "" ]] && REappend filter_inRE src/$filter_module
[[ "$filter_in"     != "" ]] && REappend filter_inRE "$filter_in"

# Blacklist filter of files whose warnings we ignore
filter_blacklistRE=""

#   External files: adding our own doxygen makes diffs with upstream very hard
#     cairo-wideint
REappend filter_blacklistRE "cairo-wideint"

#   Functions with varying numbers of arguments
#   This is temporary until we move to C++-14
REappend filter_blacklistRE "Schedule(Time"
REappend filter_blacklistRE "ScheduleWithContext(uint32_t"
REappend filter_blacklistRE "Schedule\\(Now\\|Destroy\\)(\\(MEM\\|void\\)"

#   ATTRIBUTE_HELPER_CPP( and _HEADER(
REappend filter_blacklistRE "ATTRIBUTE_HELPER_\\(CPP\\|HEADER\\)"

# Filter out regular expression for black list, -e, -t and -F
filter_outRE=""
[[ $filter_blacklist -eq 1 ]] && REappend filter_outRE "$filter_blacklistRE"
[[ $filter_examples  -eq 1 ]] && REappend filter_outRE "/examples/"
[[ $filter_test      -eq 1 ]] && REappend filter_outRE "/test/"
[[ "$filter_out"     != "" ]] && REappend filter_outRE "$filter_out"


# Configuration ------------------------
#

function on_off
{
    if [[ "${!1:-}" != "" && "${!1}" != "0" ]] ; then
        echo "ON"
    else
        echo "off"
    fi
}

if [ $verbosity -eq 1 ]; then
    echo
    echo "$me:"
    echo "    Verbose:          $(on_off verbosity)"
    echo "    Skip build:       $(on_off skip_intro)"
    echo "    Log file to use:  $LOG"
    echo "    Module filter:    $(on_off filter_module)  $filter_module"
    echo "    Examples filter:  $(on_off filter_examples)"
    echo "    Tests filter:     $(on_off filter_test)"
    echo "    Blacklist filter: $(on_off filter_blacklist)"
    echo "    Filter in:        $(on_off filter_in)  $filter_in"
    echo "    Filter out:       $(on_off filter_out)  $filter_out"
    echo

    #  Show the resulting filters here, in addition to below
    echo "    Net result of all filters:"
    [[ "${filter_inRE:-}"  != "" ]] && echo "      Filtering in:   \"$filter_inRE\""
    [[ "${filter_outRE:-}" != "" ]] && echo "      Filtering out:  \"$filter_outRE\""

    echo
fi


#  Run doxygen -------------------------
#

if [ $skip_doxy -eq 1 ]; then
    echo
    echo "Skipping doxygen run, using existing log file $LOG"

else

    # Modify doxygen.conf to generate all the warnings
    # We keep dot active to generate graphs in the documentaion
    # (see for example PacketTagList) and warn about ill-formed
    # graphs, but we disable all the doxygen-generated diagrams
    # to shorten the run time.

    conf=doc/doxygen.conf
    cp $conf ${conf}.bak
    cat <<-EOF >> $conf

    # doxygen.warnings.report.sh:
    EXTRACT_ALL = no
    WARNINGS = no
    WARN_LOGFILE = doc/$WARNINGSLOGFILE
    SOURCE_BROWSER = no
    HTML_OUTPUT = html-warn
    CLASS_DIAGRAMS = no
    CLASS_GRAPH = no
    COLLABORATION_GRAPH = no
    GROUP_GRAPHS = no
    INCLUDE_GRAPH = no
    INCLUDED_BY_GRAPH = no
    CALL_GRAPH = no
    CALLER_GRAPH = no
    GRAPHICAL_HIERARCHY = no
    DIRECTORY_GRAPH = no
EOF


    intro_h="introspected-doxygen.h"
    if [ $skip_intro -eq 1 ]; then
        verbose "" "Skipping ./ns3 build"
        verbose -n "Trying print-introspected-doxygen with doxygen build"
        (cd "$ROOT" && ./ns3 run print-introspected-doxygen --no-build >doc/$intro_h 2>&6 )
        status_report $? "./ns3 run print-introspected-doxygen" noexit
    else
        # Run introspection, which may require a build
        verbose -n "Building"
        (cd "$ROOT" && ./ns3 build >&6 2>&6 )
        status_report $? "./ns3 build"
        verbose -n "Running print-introspected-doxygen with doxygen build"
        (cd "$ROOT" && ./ns3 run print-introspected-doxygen --no-build >doc/$intro_h 2>&6 )
        status_report $? "./ns3 run print-introspected-doxygen"
    fi

    # Waf insists on writing cruft to stdout
    sed -i.bak -E '/^Waf:/d' doc/$intro_h
    rm doc/$intro_h.bak

    verbose -n "Rebuilding doxygen docs with full errors"
    (cd "$ROOT" && ./ns3 docs doxygen-no-build >&6 2>&6 )
    status_report $? "./ns3 docs doxygen-no-build"

    # Swap back to original config
    rm -f $conf
    mv -f $conf.bak $conf
fi

# Filter log file
function filter_log
{
    local flog;
    flog=$( cat "$LOG" | grep "^$ROOT" )

    [[ "${filter_inRE:-}"  != "" ]] && flog=$( echo "$flog" | grep "$filter_inRE" )
    [[ "${filter_outRE:-}" != "" ]] && flog=$( echo "$flog" | grep -v "$filter_outRE" )

    flog=$(                         \
        echo "$flog"              | \
        sort -t ':' -k1,1 -k2,2n  | \
        uniq                        \
        )

    echo "$flog"
}

# Analyze the log ----------------------
#
#  Show the resulting filters
echo
echo "Net result of all filters:"
[[ "${filter_inRE:-}"  != "" ]] && echo "Filtering in \"$filter_inRE\""
[[ "${filter_outRE:-}" != "" ]] && echo "Filtering out \"$filter_outRE\""

verbose -n "Filtering the doxygen log"

filter_log_results=$(filter_log)

# List of module directories (e.g, "src/core/model")
if [ ! -z "$filter_log_results" ]
then
    undocmods=$(                \
        filter_log            | \
        cut -d ':' -f 1       | \
        sed "s|$ROOT/||g"     | \
        cut -d '/' -f 1-3     | \
        sort                  | \
        uniq -c               | \
        sort -nr                \
        )
    modcount=$(                         \
        echo "$undocmods"             | \
        wc -l                         | \
        sed 's/^[ \t]*//;s/[ \t]*$//'   \
        )
    modwarncount=$(                               \
        echo "$undocmods"                       | \
        awk '{total += $1}; END {print total}'    \
        )
else
    undocmods=""
    modcount=0
    modwarncount=0
fi

# For a function with multiple undocumented parameters,
# Doxygen prints the additional parameters on separate lines,
# so they don't show up in the totals above.
# Rather than work too hard to get the exact number for each file,
# we just list the total here.
addlparam=$(                                  \
    grep "^  parameter '" "$LOG"            | \
    wc -l                                   | \
    sed 's/^[ \t]*//;s/[ \t]*$//'             \
    )

# Total number of warnings
warncount=$((modwarncount + addlparam))

# List of files appearing in the log
if [ ! -z "$filter_log_results" ]
then
    undocfiles=$(               \
        filter_log            | \
        cut -d ':' -f 1       | \
        sed "s|$ROOT||g"      | \
        cut -d '/' -f 2-      | \
        sort                  | \
        uniq -c               | \
        sort -k 2               \
        )
else
    undocfiles=""
fi

# Sorted by number, decreasing
undocsort=$(echo "$undocfiles" | sort -k1nr,2 )

# Total number of files
if [ ! -z "$undocfiles" ]
then
    filecount=$(                        \
        echo "$undocfiles"            | \
        wc -l                         | \
        sed 's/^[ \t]*//;s/[ \t]*$//'   \
        )
else
    filecount=0
fi

# Filtered in warnings
filterin=
if [ "${filter_inRE:-}" != "" ] ; then
    filterin=$(              \
        filter_log         | \
        sed "s|$ROOT/||g"    \
        )
fi

status_report 0 "Filter"
echo


# Summarize the log --------------------
#

echo
echo "Report of Doxygen warnings"
echo "----------------------------------------"
echo
echo "(All counts are lower bounds.)"
echo
echo "Warnings by module/directory:"
echo
echo "Count Directory"
echo "----- ----------------------------------"
echo "$undocmods"
echo " $addlparam additional undocumented parameters."
echo "----------------------------------------"
printf "%6d total warnings\n" $warncount
printf "%6d directories with warnings\n" $modcount
echo
echo
echo "Warnings by file (alphabetical)"
echo
echo "Count File"
echo "----- ----------------------------------"
echo "$undocfiles"
echo "----------------------------------------"
printf "%6d files with warnings\n" $filecount
echo
echo
echo "Warnings by file (numerical)"
echo
echo "Count File"
echo "----- ----------------------------------"
echo "$undocsort"
echo "----------------------------------------"
printf "%6d files with warnings\n" $filecount
echo
echo
echo "Doxygen Warnings Summary"
echo "----------------------------------------"
printf "%6d directories\n" $modcount
printf "%6d files\n" $filecount
printf "%6d warnings\n" $warncount

# Return status based on warnings
exit_status=$((warncount > 0))

# if [ "${filter_inRE:-}" != "" ] ; then
#     if [ "$filterin" != "" ] ; then
#         echo
#         echo
#         echo "Filtered Warnings"
#         echo "========================================"
#         echo "$filterin"
#         exit_status=1
#     else
#         exit_status=0
#     fi
# fi

if [ "$filterin" != "" ] ; then
    echo
    echo
    echo "Filtered Warnings"
    echo "========================================"
    echo "$filterin"
fi

status_report 0 $me

exit $exit_status
