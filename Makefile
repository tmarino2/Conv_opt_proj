all:	write_up.pdf

write_up.pdf:	*.bib *.tex
	pdflatex -halt-on-error write_up
	bibtex write_up
	pdflatex -halt-on-error write_up
	pdflatex -halt-on-error write_up
	pdflatex -halt-on-error write_up

clean:
	rm -f *.lot *.lof *.toc *.aux *.bbl *.log *.dvi *.blg *.out *fls *fdb_latexmk *~ write_up.pdf

