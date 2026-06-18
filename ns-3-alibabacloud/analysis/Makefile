all : trace_reader

trace_reader : trace_reader.cpp trace-format.h trace_filter.hpp utils.hpp sim-setting.h
	g++ trace_reader.cpp -o trace_reader -O3 -std=gnu++11

fct_analysis: fct_analysis.cpp
	g++ fct_analysis.cpp -o fct_analysis -O3 -std=gnu++11
