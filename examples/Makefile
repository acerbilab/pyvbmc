# Makefile for extracting code from Jupyter Notebooks

EXAMPLES = 1 2 3 4 5
all:
	@for i in $(EXAMPLES); do\
		make --no-print-directory pyvbmc_example_"$$i"_full_code.py;\
	done

pyvbmc_example_%_full_code.py: pyvbmc_example_%*.ipynb
	@jupyter nbconvert $< --no-prompt --stdout --to script --PythonExporter.exclude_markdown=True | sed -e '1,3d' | head -n -1 >> $@
	@black $@

clean:
	@rm pyvbmc_example_*.py
