#!/bin/bash

: ${1?"Usage: $0 country"}
country=$1

folder="csse_covid_19_time_series/"
file_c="time_series_covid19_confirmed_global.csv"
file_r="time_series_covid19_recovered_global.csv"
file_d="time_series_covid19_deaths_global.csv"

grep -i $country $folder$file_c | sed 's/,//' > ${country}_c.csv
grep -i $country $folder$file_r | sed 's/,//' > ${country}_r.csv
grep -i $country $folder$file_d | sed 's/,//' > ${country}_d.csv

# change , to \n
sed 's/,/\n/g' ${country}_c.csv | tail -n +4 > ${country}_c.dat
sed 's/,/\n/g' ${country}_r.csv | tail -n +4 > ${country}_r.dat
sed 's/,/\n/g' ${country}_d.csv | tail -n +4 > ${country}_d.dat

rm ${country}_*.csv
python country.py ${country}
