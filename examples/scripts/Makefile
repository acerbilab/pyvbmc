# Makefile for extracting code from Jupyter Notebooks

# Set target path one directory up
VPATH = $(realpath $(CURDIR)/..)

# Jupyter Notebook numbers:
EXAMPLES = 1 2 3 4 5 6

all:
	@for i in $(EXAMPLES); do\
		make --no-print-directory pyvbmc_example_"$$i"_full_code.py;\
	done

pyvbmc_example_%_full_code.py: $(VPATH)/pyvbmc_example_%*.ipynb
  # convert NB to script | remove leading lines | remove trailing lines | fix paths to .pkl files > output
	@jupyter nbconvert $< --no-prompt --stdout --to script --PythonExporter.exclude_markdown=True | sed -e '1,3d' | head -n -1 | sed -e 's/open("\(.*\.pkl\)"/open("..\/\1"/g' > $@
	@python -m isort $@
	@python -m black $@

clean:
	@rm pyvbmc_example_*.py
