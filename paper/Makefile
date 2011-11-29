all: paper.pdf

paper.pdf: paper.tex diagrams/*
	pdflatex paper.tex
	pdflatex paper.tex
	gnome-open paper.pdf

clean:
	rm -f paper.pdf paper.aux paper.log
