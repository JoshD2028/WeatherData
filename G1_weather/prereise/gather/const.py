import pandas as pd
from datetime import timedelta

# Map state abbreviations to state name
abv2state = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

# wind for wind farms
SELECTORS = {
    "UWind80": "U component of wind",
    "VWind80": "V component of wind",
    "UWind10": "10 metre U wind component",
    "VWind10": "10 metre V wind component",
    "rad": "Downward short-wave radiation flux",
    "vbd": "Visible Beam Downward Solar Flux",
    "vdd": "Visible Diffuse Downward Solar Flux",
    "2tmp": "2 metre temperature",
}

# OUTDIR = "../psse/grg-pssedata/"
OUTDIR = "./"

# study year; which year to study
YEAR = 2023
START = pd.to_datetime(f"{YEAR - 1}-12-31 01:00")
END = pd.to_datetime(f"{YEAR + 1}-01-02 00:00")
# START = pd.to_datetime("2019-12-31 01:00")
# END = pd.to_datetime("2021-01-02 00:00")
DATADIR = r"/research/alij/"
# START = pd.Timestamp(YEAR, 1, 1) - timedelta(days=1, hours=1)
# END = pd.Timestamp(YEAR, 1, 2) + timedelta(days=1)
print(f"start time is {START}")
print(f"end time is {END}")
SEARCHSTRING = "V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m"
TZ = 6
