EPSTOPDF = epstopdf
DIA = dia
CONVERT = convert

SRC = ../../src
# Temporary source directory, for build
SOURCETEMP = source-temp
FIGURES = $(SOURCETEMP)/figures
#VPATH = $(FIGURES)

# list all manual .rst files that need to be copied to $SOURCETEMP
SOURCES = \
	source/conf.py \
	source/_static \
	source/index.rst \
	source/replace.txt \
	source/attributes.rst \
	source/callbacks.rst \
	source/develop.rst \
	source/documentation.rst \
	source/enable-modules.rst \
	source/enable-tests.rst \
	source/events.rst \
	source/features.rst \
	source/gnuplot.rst \
	source/hash-functions.rst \
	source/helpers.rst \
	source/how-to-write-tests.rst \
	source/logging.rst \
	source/new-models.rst \
	source/new-modules.rst \
	source/object-model.rst \
	source/object-names.rst \
	source/organization.rst \
	source/profiling.rst \
	source/python.rst \
	source/random-variables.rst \
	source/realtime.rst \
	source/simulator.rst \
	source/support.rst \
	source/test-background.rst \
	source/test-framework.rst \
	source/test-overview.rst \
	source/tests.rst \
	source/tracing.rst \
	source/troubleshoot.rst \
	source/utilities.rst \
	source/working-with-cmake.rst \
	source/working-with-git.rst \
	source/working-with-gitlab-ci-local.rst \
	${SRC}/stats/doc/data-collection.rst \
	${SRC}/stats/doc/data-collection-overview.rst \
	${SRC}/stats/doc/statistics.rst \
	${SRC}/stats/doc/data-collection-helpers.rst \
	${SRC}/stats/doc/probe.rst \
	${SRC}/stats/doc/collector.rst \
	${SRC}/stats/doc/aggregator.rst \
	${SRC}/stats/doc/adaptor.rst \
	${SRC}/stats/doc/scope-and-limitations.rst \

# list all manual figure files that need to be copied to 
# $SOURCETEMP/figures.  For each figure to be included in all
# documentation formats (html, latex...) the following formats are supported: 
#  1) a single .dia file (preferred option, because it can be edited)
#  2) a single .eps file
#  3) both a .pdf and .png file

SOURCEFIGS = \
    figures/software-organization.dia \
	figures/plot-2d.png \
	figures/plot-2d-with-error-bars.png \
	figures/plot-3d.png \
	figures/gtk-config-lena-dual-stripe-device-view.png \
	figures/gtk-config-lena-dual-stripe-eNB-tx-power.png \
	figures/heaptrack.png \
	figures/hotspot-cycles.png \
	figures/hotspot-setup.png \
	figures/hotspot-top-down.png \
	figures/perfetto-trace-cmake.png \
	figures/uprof-collect-callstack.png \
	figures/uprof-collect-callstack.png \
	figures/uprof-profile-application.png \
	figures/uprof-select-events.png \
	figures/uprof-start.png \
	figures/uprof-stats.png \
	figures/uprof-strace-lte.png \
	figures/vtune-configure.png \
	figures/vtune-landing.png \
	figures/vtune-perf-snapshot.png \
	figures/vtune-select-uarch-profiling.png \
	figures/vtune-uarch-core-stats.png \
	figures/vtune-uarch-profiling-summary.png \
	figures/vtune-uarch-wifi-stats.png \
	${SRC}/stats/doc/Stat-framework-arch.png \
	${SRC}/stats/doc/Wifi-default.png \
	${SRC}/stats/doc/dcf-overview.dia \
	${SRC}/stats/doc/dcf-overview-with-aggregation.dia \
	${SRC}/stats/doc/seventh-packet-byte-count.png \
	${SRC}/stats/doc/gnuplot-helper-example.png \
	${SRC}/stats/doc/gnuplot-aggregator.png \

# specify figures from which .png and .pdf figures need to be
# generated (all dia and eps figures)
IMAGES_EPS = \
	$(FIGURES)/software-organization.eps \
	$(FIGURES)/dcf-overview.eps \
	$(FIGURES)/dcf-overview-with-aggregation.eps \

# rescale pdf figures as necessary
$(FIGURES)/software-organization.pdf_width = 5in

IMAGES_PNG = $(IMAGES_EPS:.eps=.png)

IMAGES_PDF = ${IMAGES_EPS:.eps=.pdf}

IMAGES = $(IMAGES_EPS) $(IMAGES_PNG) $(IMAGES_PDF)

RESCALE = ../../utils/rescale-pdf.sh

%.eps : %.dia
	@echo dia $(notdir $<)
	@$(DIA) -t eps $< -e $@ >/dev/null

%.png : %.dia
	@echo dia $(notdir $<)
	@$(DIA) -t png $< -e $@ >/dev/null

%.png : %.eps
	@echo convert $(notdir $<)
	@$(CONVERT) $< $@ >/dev/null

%.pdf : %.eps
	@echo epstopdf $(notdir $<)
	@$(EPSTOPDF) $< -o=$@ >/dev/null
	@if test x$($@_width) != x; then $(RESCALE) $($@_width) $@ ; fi

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(SOURCETEMP)

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest

.NOTPARALLEL:

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"

copy-sources:  $(SOURCES)
	@rm -rf $(SOURCETEMP)
	@mkdir -p $(SOURCETEMP) 
	@mkdir -p $(FIGURES)
	@cp -r $(SOURCES) $(SOURCETEMP)
	@cp -r $(SOURCEFIGS) $(FIGURES)

clean:
	-rm -rf $(BUILDDIR)
	-rm -rf $(SOURCETEMP)

frag: pickle
	@if test ! -d $(BUILDDIR)/frag; then mkdir $(BUILDDIR)/frag; fi
	pushd $(BUILDDIR)/frag && ../../pickle-to-xml.py ../pickle/index.fpickle  > navigation.xml && popd
	cp -r $(BUILDDIR)/pickle/_images $(BUILDDIR)/frag

html: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/ns-3.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/ns-3.qhc"

devhelp: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/ns-3"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/ns-3"
	@echo "# devhelp"

epub: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

changes: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest: copy-sources $(IMAGES)
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."


test: $(IMAGES)
