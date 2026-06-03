
SEQDIAG = seqdiag
RESCALE = ../../../utils/rescale-pdf.sh

IMAGES_SEQDIAG = \
	spectrum-channel-phy-interface.seqdiag

spectrum-channel-phy-interface.pdf_width = 10cm

seqdiags: 	${IMAGES_SEQDIAG:.seqdiag=.png} \
		${IMAGES_SEQDIAG:.seqdiag=.pdf}


%.png : %.seqdiag; $(SEQDIAG) -Tpng --no-transparency -o $@ $< 

%.pdf : %.seqdiag
	$(SEQDIAG) -Tpdf -o $@ $<
	if test x$($@_width) != x; then $(RESCALE) $($@_width) $@ ; fi

clean:
	rm ${IMAGES_SEQDIAG:.seqdiag=.png} \
	   ${IMAGES_SEQDIAG:.seqdiag=.pdf}

