.PHONY: report csv clean

PYTHON ?= python3

report:
	$(PYTHON) conv2d_packed_prepare_perf.py

csv:
	$(PYTHON) conv2d_packed_prepare_perf.py --csv

clean:
	rm -f *.csv
