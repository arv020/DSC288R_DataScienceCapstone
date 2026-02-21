# Flight Delay and Cancellation Prediction in the United States (2018–2022)

Capstone project for **DSC288R** — UC San Diego  
Master of Science in Data Science

Authors: Sahra Ranjbar, Arely Vasquez, Tatianna Sanchez

## Project Purpose

Predict whether a U.S. domestic flight will be **on time**, **delayed**, or **cancelled** using historical flight operations and weather data from **2018–2022**.

This project:
- Uses a **leakage-free feature set** (only information available before departure)
- Explores multiple modeling approaches (Logistic Regression, Random Forest, XGBoost)
- Applies **multi-stage sampling** to manage dataset scale and class imbalance
- Emphasizes **time-aware evaluation** and class-balanced metrics

---

## Project Structure

```text
├── data/
│   ├── raw/                  # Raw downloaded data (not tracked)
│   └── cleansed/             # Cleaned / joined datasets
│
├── scripts/
│   ├── 1_download_raw_data.py            # Download raw and cleansed flight/weather data
│   ├── 2_download_merged_data.py         # Download merged flight + weather data
│   └── build_flights_weather_merged      # Script on how merged data is processed
│
├── notebooks/
│   ├── EDA/
│   │   ├── 1_raw_eda.ipynb
│   │   └── 2_combined_eda.ipynb
│   │
│   └── modeling/
│       ├── Logistic_regression_v2.ipynb
│       ├── XGBoost.ipynb
│       ├── logistic_regression_baseline.ipynb
│       ├── random_forest_baseline.ipynb
│       ├── random_forest_v2.ipynb
│       ├── random_forest_v3.ipynb
│    ├── XGB_tuning_BEST_PARAM.ipynb
│    └── hyperparam_tuning_results.ipynb
│
│
├── samples/
│   ├── cancel_tuning_results.csv        # Hypertuning results on cancelled model performance
│   ├── delay_tuning_results.csv         # Hypertuning results on delayed model performance
│
│
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Workflow Overview
	1.	Run 1_download_raw_data.py to download raw flight and weather data locally
	2.	Run 2_download_merged_data.py to download merged data of flight + weather locally
	3.	Explore notebooks in EDA to understand data
  4.  Explore notebooks in modeling to see baseline models
  5.  Explore XGBoost models to see how hypertuning took place in evaluating the best model


## Evaluation Strategy
	1. Time-based splits to prevent future leakage
	2. Accuracy reported for reference only
	3. Primary metrics:
  	•	Macro F1
  	•	Balanced accuracy
  	•	Class-specific recall (Delayed, Cancelled)


<HTML>
<BODY>
<TABLE>
  <TR>
    <TD COLSPAN=2><H4>Airport Flights – On Time, Delayed, or Cancelled?</H4></TD>
  </TR>

  <TR><TD COLSPAN=2>&nbsp;</TD></TR>

  <TR><TD COLSPAN=2><H4>Data Dictionary (Leakage-Free Features)</H4></TD></TR>

  <!-- Time / Calendar -->
  <TR><TD>Year</TD><TD>Year</TD></TR>
  <TR><TD>Quarter</TD><TD>Quarter (1–4)</TD></TR>
  <TR><TD>Month</TD><TD>Month</TD></TR>
  <TR><TD>DayofMonth</TD><TD>Day of Month</TD></TR>
  <TR><TD>DayOfWeek</TD><TD>Day of Week</TD></TR>
  <TR><TD>FlightDate</TD><TD>Flight Date (yyyymmdd)</TD></TR>

  <!-- Airline -->
  <TR><TD>Marketing_Airline_Network</TD><TD>Unique marketing carrier code.</TD></TR>
  <TR><TD>Operated_or_Branded_Code_Share_Partners</TD><TD>Reporting carrier operated or branded code share partners.</TD></TR>
  <TR><TD>DOT_ID_Marketing_Airline</TD><TD>DOT identifier for marketing airline.</TD></TR>
  <TR><TD>IATA_Code_Marketing_Airline</TD><TD>IATA code for marketing airline.</TD></TR>
  <TR><TD>Flight_Number_Marketing_Airline</TD><TD>Marketing flight number.</TD></TR>
  <TR><TD>Operating_Airline</TD><TD>Operating carrier code.</TD></TR>
  <TR><TD>DOT_ID_Operating_Airline</TD><TD>DOT identifier for operating airline.</TD></TR>
  <TR><TD>IATA_Code_Operating_Airline</TD><TD>IATA code for operating airline.</TD></TR>

  <!-- Geography -->
  <TR><TD>Origin</TD><TD>Origin airport code.</TD></TR>
  <TR><TD>OriginCityName</TD><TD>Origin airport city name.</TD></TR>
  <TR><TD>OriginState</TD><TD>Origin airport state code.</TD></TR>
  <TR><TD>OriginStateName</TD><TD>Origin airport state name.</TD></TR>
  <TR><TD>OriginWac</TD><TD>Origin airport world area code.</TD></TR>

  <TR><TD>Dest</TD><TD>Destination airport code.</TD></TR>
  <TR><TD>DestCityName</TD><TD>Destination airport city name.</TD></TR>
  <TR><TD>DestState</TD><TD>Destination airport state code.</TD></TR>
  <TR><TD>DestStateName</TD><TD>Destination airport state name.</TD></TR>
  <TR><TD>DestWac</TD><TD>Destination airport world area code.</TD></TR>

  <!-- Scheduled timing -->
  <TR><TD>CRSDepTime</TD><TD>Scheduled departure time (local time: hhmm).</TD></TR>
  <TR><TD>DepTimeBlk</TD><TD>Scheduled departure time block.</TD></TR>
  <TR><TD>CRSArrTime</TD><TD>Scheduled arrival time (local time: hhmm).</TD></TR>
  <TR><TD>ArrTimeBlk</TD><TD>Scheduled arrival time block.</TD></TR>

  <!-- Flight characteristics -->
  <TR><TD>CRSElapsedTime</TD><TD>Scheduled elapsed flight time in minutes.</TD></TR>
  <TR><TD>Distance</TD><TD>Distance between airports in miles.</TD></TR>
  <TR><TD>DistanceGroup</TD><TD>Distance intervals in 250-mile groups.</TD></TR>

<!-- Origin Airport Weather conditions -->
  <TR><TD>p01i</TD><TD>Total precipitation accumulated over the previous one-hour period (inches). Includes: rain, snow, or other measurable precipitation</TD></TR>
  <TR><TD>gust</TD><TD>Max wind gust speed observed during the reporting period (in knots)</TD></TR>
  <TR><TD>vsby</TD><TD>Horizontal visibility measured in statute miles at reporting station (in miles)</TD></TR>
  <TR><TD>relh</TD><TD>Relative humidity expressed as a percentage. Variable is not directly measured but derived from temp and dew point.</TD></TR>
  <TR><TD>tempf</TD><TD>Air temperature in F</TD></TR>
  <TR><TD>sknt</TD><TD>Wind Speed in knots</TD></TR>

</TABLE>

</BODY>
</HTML>
