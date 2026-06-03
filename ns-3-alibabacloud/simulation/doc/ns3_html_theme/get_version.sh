#!/bin/bash

# Get the current repo name and version
# to format urls appropriately via Javascript
# variables for inclusion in html files.

# Use cases:
# 1.  Hosted on nsnam.org, tagged release.
# 2.  Hosted on nsnam.org, ns-3-dev.
# 3.  User repo, modified from a tagged release (or ns-3-dev).
# 4.  User repo, at a release tag.
# 5.  User repo, at a private tag.
# 6.  Private web host, at a tag (or ns-3-dev, or local mod).
#
# For case 1 and 2, we want the links to point to the nsnam.org
# publicly hosted pages.  For all other cases, we want to point
# to the built pages in the repo itself.
#
# For robustness, we attempt to identify cases 1 & 2
# automatically.  (There is also an explicit switch '-p'
# to force us into the public case.)
#
# The approach to identify cases 1 & 2 is to test:
# a.  We're on nsnam.org (actually, nsnam-www.coe-hosted.gatech.edu), and
# b.  We're in the tmp build directory, /tmp/daily-nsnam/
#     (This is the directory used by the update-* scripts
#     run by cron jobs.)
#
# If both a and b are true, we're building for public urls.
# (The newer update-docs script (through ns3) sets
# NS3_WWW_URLS=public explicitly.)
# 
# The repo version is either a tag name or a commit (short) id.
#
# If we're building for nsnam.org, and at a tag, we use just
# the tag as the repo name/version string, e.g. 'ns-3.14'.
# Otherwise, we're building for ns-3-dev, and we use, e.g,
# 'ns-3-dev @ fd0f27a10eff'.
#
# If we're *not* building for nsnam.org, we use the repo
# directory name as the repo name.  (This will typically be
# a name meaningful to the user doing the build, perhaps a
# shorthand for the feature they are working on.)  For
# example, this script was developed in a repo named
# 'docs'.  We always use the repo version, resulting
# in document version strings like 'docs @ ns-3.15' or
# 'docs @ fd0f27a10eff'
#

me=`basename $0`
function say
{
    echo "$me: $*"
}

function usage
{
    cat <<-EOF
	Usage:  $me [-p]              normal versioning
	        $me [-n] [-d] [-t]    test options
	  -p  build public urls, NS3_WWW_URLS=public is an alternative

	Testing options:
	  -n  pretend we are on nsnam.org
	  -d  pretend we are in the automated build directory
	  -t  pretend we are at a repo tag
	    
EOF
    exit 1
}

# script arguments
say
say using doxygen: $(which doxygen) $(doxygen --version)
public=0
nsnam=0
daily=0
tag=0

while getopts :pndth option ; do
    case $option in
	(p)  public=1 ;;
	(n)  nsnam=1  ;;

	(d)  daily=1  ;;

	(t)  tag=1    ;;

	(h)  usage ;;
	(:)  say "Missing argument to -$OPTARG" ; usage ;;
	(\?) say "Invalid option: -$OPTARG"     ; usage ;;
    esac
done

# Hostname, fully qualified, e.g. nsnam-www.coe-hosted.gatech.edu
HOST=`hostname`
NSNAM="nsnam-www.coe-hosted.gatech.edu"

# Build directory
DAILY="^/tmp/daily-nsnam/"

if [ $nsnam -eq 1 ]; then
    HOST=$NSNAM
    say "-n forcing HOST = $HOST"
fi

if [ $daily -eq 1 ] ; then
    OLDPWD=$PWD
    PWD=/tmp/daily-nsnam/foo
    say "-d forcing PWD = $PWD"
fi

if [ $tag -eq 1 ]; then
    version="ns-3.14"
    say "-t forcing tagged version = $version"
fi

if  ((nsnam + daily + tag > 0)) ; then
    say
fi

if [[ ($public == 1) || \
    ("${NS3_WWW_URLS:-}" == "public") || \
    ( ($HOST == $NSNAM) && ($PWD =~ $DAILY) ) ]] ; then
    PUBLIC=1
    say "building public docs for nsnam.org"
else
    PUBLIC=0
    say "building private docs"
fi

if [ $daily -eq 1 ]; then
    PWD=$OLDPWD
fi

# Destination javascript file
outf="doc/ns3_html_theme/static/ns3_version.js"

# Distance from last tag
# 'git describe HEAD --tags' will return a string such as either
# 'ns-3.29-159-xxx' (if we are at a distance from the last release tag) or
# 'ns-3.29' if we are at the tag.
distance=`git describe HEAD --tags | cut -d '-' -f 3- | cut -d '-' -f 1`

if [ $distance -eq 1 ]; then
    version=`git describe HEAD --tags | cut -d '-' -f 2`
    say "at version $version"

elif [ $tag -eq 1 ]; then
    distance=1
    # version previously set
    vers_href=

else
    version=`git describe HEAD --tags | cut -d '-' -f 3- | cut -d '-' -f 2 | cut -c 2-`
    # Check for uncommitted changes
    changes=0
    if [ "$(git status --porcelain)" ]; then
        changes=1
    fi
    if [ $changes ] ; then
	say "beyond latest tag, last commit: $version, dirty"
	dirty="(+)"
    else
	say "beyond latest tag, last commit: $version, clean"
	dirty=
    fi
fi

if [ $PUBLIC -eq 1 ]; then
    echo "// ns3_version.js:  automatically generated"       >  $outf
    echo "//  public urls"                                   >> $outf
    # Generate URL relative to server root
    echo "var ns3_host = \"/\";"                             >> $outf
    
    if [ $distance -eq 1 ]; then
	# Like "http://www.nsnam.org/ns-3-14"
	vers_href="https://www.nsnam.org/ns-3-${version#ns-3.}"
	vers_href="<a href=\\\"$vers_href\\\">$version$dirty</a>"
	
	echo "var ns3_version = \"Release $vers_href\";"     >> $outf
	echo "var ns3_release = \"docs/release/${version#ns-}/\";" >> $outf
    else
	vers_href="https://gitlab.com/nsnam/ns-3-dev/commits/$version"
	version="<a href=\\\"$vers_href\\\">$version$dirty</a>"
	
	echo "var ns3_version = \"ns-3-dev @ $version\";"    >> $outf
	echo "var ns3_release = \"docs/\";" >> $outf
    fi
    echo "var ns3_local = \"\";"                             >> $outf
    echo "var ns3_doxy  = \"doxygen/\";"                     >> $outf
    
else
    repo=`basename $PWD`
    echo "// ns3_version.js:  automatically generated"       >  $outf
    echo "//  private urls"                                  >> $outf
    echo "var ns3_host = \"file://$PWD/\";"                  >  $outf
    echo "var ns3_version = \"$repo @ $version$dirty\";"     >> $outf
    echo "var ns3_release = \"doc/\";"                       >> $outf
    echo "var ns3_local = \"build/\";"                       >> $outf
    echo "var ns3_doxy  = \"html/\";"                        >> $outf
fi

# Copy to html directories
# This seems not always done automatically
# by Sphinx when rebuilding
cd doc 2>&1 >/dev/null
for d in {manual,models,tutorial}/build/{single,}html/_static/ ; do
    if [ ! -d $d ]; then
	mkdir -p $d
    fi
    cp ns3_html_theme/static/ns3_version.js $d
done
cd - 2>&1 >/dev/null

# Show what was done
say
say "outf = $outf:"
cat -n $outf
say Done.
