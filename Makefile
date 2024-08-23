all:

test:
	python -m unittest discover -s ./src/


.PHONY: all clean test
