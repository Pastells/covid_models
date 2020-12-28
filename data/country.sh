#!/bin/bash
# Country time series
country=$1
folder="csse_covid_19_time_series/"
file_i="time_series_covid19_confirmed_global.csv"
# file_r="time_series_covid19_recovered_global.csv"
# file_d="time_series_covid19_deaths_global.csv"

grep -i $country $folder$file_i | sed 's/,//' > ${country}_i.csv
# grep -i $country $folder$file_r | sed 's/,//' > ${country}_r.csv
# grep -i $country $folder$file_d | sed 's/,//' > ${country}_d.csv

# change , to \n
sed 's/,/\n/g' ${country}_i.csv > ${country}_i_plot.dat
# grep -i $country $folder$file | sed 's/,//' | sed 's/,/\n/g' > ${country}_r_plot.dat
# grep -i $country $folder$file | sed 's/,//' | sed 's/,/\n/g' > ${country}_d_plot.dat
