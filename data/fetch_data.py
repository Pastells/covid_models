#!/usr/bin/env python
import datetime
import logging
import pandas
import sys

BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
GLOBAL_CONFIRMED_URL = f"{BASE_URL}/time_series_covid19_confirmed_global.csv"
GLOBAL_RECOVERED_URL = f"{BASE_URL}/time_series_covid19_recovered_global.csv"
GLOBAL_DEATHS_URL = f"{BASE_URL}/time_series_covid19_deaths_global.csv"


global_confirmed = pandas.read_csv(GLOBAL_CONFIRMED_URL)
available_countries = global_confirmed["Country/Region"].unique()

countries = sys.argv[1:]
if len(countries) == 0:
    print("At least one country should be specified", file=sys.stderr)
    print(f"Usage: {sys.argv[0]} country1 country2 ...", file=sys.stderr)
    print("\nAvailable countries are:", file=sys.stderr)
    print(', '.join(available_countries))
    sys.exit(-1)

logging.basicConfig(level=logging.INFO)
logging.info("Parsing countries %s", ', '.join(countries))

global_confirmed = pandas.read_csv(GLOBAL_CONFIRMED_URL)
global_recovered = pandas.read_csv(GLOBAL_RECOVERED_URL)
global_deaths = pandas.read_csv(GLOBAL_DEATHS_URL)

logging.debug("Available countries: %s", ', '.join(available_countries))


def filter_data(df, country):
    return df[df["Country/Region"] == country]


def change_dateformat(day):
    date = datetime.datetime.strptime(day, "%m/%d/%y")
    return date.strftime("%Y/%m/%d")
    


def parse_country(country):
    n_headers = 4  # Province/State, Country/Region, Lat, Long
    days = global_confirmed.columns[n_headers:]
    first_day = change_dateformat(days[0])
    last_day = change_dateformat(days[-1])

    logging.debug("Day 0 is %s", first_day)
    logging.debug("Last day is %s", last_day)

    confirmed = filter_data(global_confirmed, country).sum(axis=0)[days]
    recovered = filter_data(global_recovered, country).sum(axis=0)[days]
    deaths = filter_data(global_deaths, country).sum(axis=0)[days]
    
    infected = confirmed - recovered - deaths

    df = pandas.DataFrame(
        [infected, recovered, deaths, confirmed],
        index=["#infected", "recovered", "dead", "cumulative"]
    ).transpose()
    df["date"] = list(map(change_dateformat, days))

    first_day_str = first_day.replace("/", "_")
    last_day_str = last_day.replace("/", "_")
    out_file = f"countries/{country}__{first_day_str}__{last_day_str}.csv"
    df.to_csv(out_file, index=False)
    logging.info("Stored data at %s", out_file)


for country in countries:
    if country not in available_countries:
        logging.warning("Country %s not found, skip...", country)
        continue
    parse_country(country)
