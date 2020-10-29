#!/bin/bash
# Country time series
country=$1

# change , to \n
#grep -i $country time_series_covid19_confirmed_global_oct15.csv | sed 's/,//' | sed 's/,/\n/g' > ${country}_i.dat
#grep -i $country time_series_covid19_recovered_global_oct15.csv | sed 's/,//' | sed 's/,/\n/g' > ${country}_r.dat
#grep -i $country time_series_covid19_deaths_global_oct15.csv | sed 's/,//' | sed 's/,/\n/g' > ${country}_d.dat

grep -i $country time_series_covid19_confirmed_global_oct15.csv | sed 's/,//' > ${country}_i.csv
grep -i $country time_series_covid19_recovered_global_oct15.csv | sed 's/,//' > ${country}_r.csv
grep -i $country time_series_covid19_deaths_global_oct15.csv | sed 's/,//' > ${country}_d.csv
