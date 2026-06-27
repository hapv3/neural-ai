.PHONY: report csv clean

PYTHON ?= python3

report:
	$(PYTHON) conv2d_feeder_p4_perf.py

csv:
	$(PYTHON) conv2d_feeder_p4_perf.py --csv

clean:
	rm -f *.csv
