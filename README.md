# Flight Delay and Cancellation Prediction in the United States (2018–2022)

Capstone project for **DSC288R** — UC San Diego  
Master of Science in Data Science

Authors: Sahra Ranjbar, Arely Vasquez, Tatianna Sanchez

### Requirements
- Python 3.9+
- Conda (for environment setup)
- ~30 GB disk space (raw + processed datasets)
- 12+ GB RAM recommended (24M-row dataset)

---
## **Quickly Run the Model on Sample Data**

> Simulate using the model as a user would: see predictions and visualizations without running the full workflow.

1. Open the `3_tests/` directory.  
2. Run the notebook **`Model_Tests.ipynb`**.  

This will execute the model on sample data and generate all visualizations immediately.  
Otherwise, continue with the full workflow below.

---

### Reproduce Final Test Results
To reproduce the exact test macro-F1 of 0.4477 reported in the paper:

1. Complete Steps 1-5 below (download datasets)
2. Open `2_notebooks/2c_modeling/XGBoost/xgboost_v9_final_model.ipynb`
3. Run all cells

The notebook loads the pre-trained cascade model from `3_tests/cascade_final.pkl`, applies thresholds (τₐ = 0.5, τᵇ = 0.6) to the 2022 test set, and outputs the classification report and confusion matrix.

## Project Purpose

Predict whether a U.S. domestic flight will be **on time**, **delayed**, or **cancelled** using historical flight operations and weather data from **2018–2022**. This project:
- Uses a **leakage-free feature set** (only information available before departure)
- Explores multiple modeling approaches (Logistic Regression, Random Forest, XGBoost)
- Applies **multi-stage sampling** to manage dataset scale and class imbalance
- Emphasizes **time-aware evaluation** and class-balanced metrics

## Workflow Overview
Follow the steps below to reproduce the full project workflow.
### 1. SSH into UCSD dsmlp to use computing resources.

a. Connect to UCSD VPN, [UCSD VPN Instructions](https://blink.ucsd.edu/technology/network/connections/off-campus/VPN/index.html)

b. Open terminal and copy:

  ``` ssh <ucsd username>@dsmlp-login.ucsd.edu ```

c. Once successfully logged in, copy to terminal:

  ```launch-scipy-ml.sh -c 12 -m 76 -g 0 ```

d. Copy the link from the terminal onto a browser to access 

### 2. Clone the Repository locally by running the cells below on the terminal.

  ``` git clone https://github.com/arv020/DSC288R_DataScienceCapstone.git ```

  ```cd DSC288R_DataScienceCapstone ```
  
### 3. Set up environment Conda Enviornment. 
Ensure Conda is installed on your system. If using Windows, it is recommended to run the project using WSL.

Create the environment from the configuration file:

  ``` conda env create -f environment.yml ```

Initiate the environment:

``` conda init ```

***Close and open new terminal if directed to***

Activate the environment:


  ``` conda activate flight-status-ml ```

	
### 4. Download Raw Datasets
Run the following script to download the raw flight and weather datasets locally. The datasets will be saved to: 1_download_data/raw/

``` python 1_download_data/1_download_raw_data.py ```

### 5. Download Cleansed Dataset
Run the following script to download the pre-merged flight + weather dataset. The dataset will be saved to: 1_download_data/cleansed/

``` python 1_download_data/2_download_merged_data.py ```


### 6. Explore notebooks to understand EDA and Feature Engineering
Open the notebooks in the 2_notebooks/2a_EDA/ directory to explore and understand the dataset.
Open and run notebooks in the 2_notebooks/2b_feature_engineering/ explore feature engineering and prepare to run models in step 7.

### 6b. (Optional) Run the Data Pipeline from Scratch
If you want to regenerate the merged and feature-engineered datasets used by the XGBoost notebooks:
```bash
cd 4_scripts/
python 1_build_dataset.py
python 2_build_features.py
```

This produces:
- `1_download_data/cleansed/final_flights_model_dataset.parquet` (merged dataset)
- `1_download_data/cleansed/modeling_dataset.parquet` (feature-engineered dataset with 49 features)
- `4_scripts/feature_cols.py` (auto-generated feature list used by the notebooks)

**Note:** This requires the raw datasets from Step 4 and takes approximately 30-40 minutes on DSMLP due to the 24M-row dataset size.

### 7. Explore notebooks in modeling 
Explore models in the 2_notebooks/2c_modeling/ to see baseline models, hypertuning, and feature exploration. Final model can be found in 2_notebooks/2c_modeling/XGBoost/xgboost_v9_final_model.ipynb

### 8. Explore the tests to run final model
Open the notebook in the 3_tests/ directory to run the model on sample data as if you were a user and additional visualizations and tests.

---

## Project Structure

```text
├── 1_download_data/
│   ├── 1_download_raw_data.py        # Downloads raw datasets
│   └── 2_download_merged_data.py     # Downloads Cleaned/Joined datasets
│
├── 2_notebooks/
│   ├── 2a_EDA/
│   │   ├── 1_raw_eda.ipynb           # Raw EDA with datasets
│   │   └── 2_combined_eda.ipynb      # EDA once merged datasets
│   │
│   ├── 2b_feature_engineering/
│   │   ├── RF_Feature_Engineering.ipynb          # Feature Engineering used for Logistic Regression + RF versions 1-3
│   │   ├── RF_Initial_FeatureEngineering.ipynb   # Feature Engineering used for RF version 4
│   │   └── XGB_Feature_Engineering.ipynb         # Feature Engineering used for XGBoost
│   │
│   ├── 2c_modeling/
│   │   ├── Logistic Regression  # Logistic Regression Models
│   │   ├── Random Forest        # Random Forest Models
│   │   └── XGBoost              # XGBoost Models & Final Model
│
├── 3_tests/
│   ├── cascade_final.pkl     # Final trained model selected
│   ├── Model_Tests.ipynb     # Tests for the user to read in their own data
│   └── sample_data.csv       # Sample data in for model prediction
│
│
├── 4_scripts/
│   ├── exploration/
│   │   ├── bayesian_hyperparameter_search.py
│   │   ├── README.md         # Explanation for bayesian_hyperparameter_search
│   ├── 1_build_dataset.py        # Preprocessing for XGBoost
│   ├── 2_build_features.py       # Preprocessing for XGBoost
│   ├── 3_tune_hyperparams.py     # Tuning for XGBoost
│   ├── flight_weather_setup.py
│   └── resampling_experiments 
│
├── samples/
│   ├── cancel_tuning_results.csv        # Hypertuning results on cancelled model performance
│   └──  delay_tuning_results.csv         # Hypertuning results on delayed model performance
│
├── README.md                  # Read at beginning of project
├── environment.yml            # Creates the conda environment
└── .gitignore
```

---


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

  <TR><TD COLSPAN=2><H3>Data Dictionary (Leakage-Free Features)</H3></TD></TR>

  <!-- weather and flights data -->
  <TR>
    <TD COLSPAN=2><H4>Weather and Flights Dataset</H4></TD>
  </TR>
  
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

  <!-- OPSNET data (Airport Operations)-->
  <TR>
    <TD COLSPAN=2><H4>OPSNET Dataset (Airport Operations)</H4></TD>
  </TR>
  <TR><TD>LOCID</TD><TD>FAA location identifier for the airport or facility.</TD></TR>
  <TR><TD>YYYYMMDD</TD><TD>Date of record in YYYYMMDD format (Year, Month, Day).</TD></TR>
  <TR><TD>STATE</TD><TD>Two-letter U.S. state identifier for the facility location.</TD></TR>
  <TR><TD>REGION</TD><TD>FAA administrative region code representing the geographic region of the facility.</TD></TR>
  <TR><TD>SAREA</TD><TD>Service area classification indicating operational coverage (Eastern, Central, or Western Enroute/Terminal).</TD></TR>
  <TR><TD>CLASS_ID</TD><TD>Facility classification identifier describing the type of air traffic control facility.</TD></TR>
  <TR><TD>FTYPE</TD><TD>Facility type code indicating operational facility category (e.g., radar tower, TRACON, ARTCC).</TD></TR>
  <TR><TD>IFR_AC</TD><TD>IFR itinerant air carrier operations.</TD></TR>
  <TR><TD>IFR_AT</TD><TD>IFR itinerant air taxi operations.</TD></TR>
  <TR><TD>IFR_GA</TD><TD>IFR itinerant general aviation operations.</TD></TR>
  <TR><TD>IFR_MI</TD><TD>IFR itinerant military operations.</TD></TR>
  <TR><TD>VFR_AC</TD><TD>VFR itinerant air carrier operations.</TD></TR>
  <TR><TD>VFR_AT</TD><TD>VFR itinerant air taxi operations.</TD></TR>
  <TR><TD>VFR_GA</TD><TD>VFR itinerant general aviation operations.</TD></TR>
  <TR><TD>VFR_MI</TD><TD>VFR itinerant military operations.</TD></TR>
  <TR><TD>LOCAL_GA</TD><TD>Local civil general aviation operations.</TD></TR>
  <TR><TD>LOCAL_MI</TD><TD>Local military operations.</TD></TR>
  <TR><TD>AC</TD><TD>Total air carrier operations (IFR + VFR air carrier flights).</TD></TR>
  <TR><TD>ATAXI</TD><TD>Total air taxi operations (IFR + VFR air taxi flights).</TD></TR>
  <TR><TD>GA</TD><TD>Total general aviation operations (IFR + VFR general aviation + civil local operations).</TD></TR>
  <TR><TD>MIL</TD><TD>Total military operations (IFR + VFR military + local military operations).</TD></TR>
  <TR><TD>TOTAL</TD><TD>Total airport operations across all categories (AC + ATAXI + GA + MIL).</TD></TR>

  <!-- airports data-->
  <TR>
    <TD COLSPAN=2><H4>Airports Dataset</H4></TD>
  </TR>
  <TR><TD>id</TD><TD>Internal OurAirports numeric identifier for the airport.</TD></TR>
  <TR><TD>ident</TD><TD>Primary airport identifier used in the OurAirports system, typically the ICAO code.</TD></TR>
  <TR><TD>type</TD><TD>Airport type classification (e.g., large_airport, medium_airport, small_airport, heliport).</TD></TR>
  <TR><TD>name</TD><TD>Official name of the airport.</TD></TR>
  <TR><TD>latitude_deg</TD><TD>Latitude of the airport in decimal degrees.</TD></TR>
  <TR><TD>longitude_deg</TD><TD>Longitude of the airport in decimal degrees.</TD></TR>
  <TR><TD>elevation_ft</TD><TD>Elevation of the airport above mean sea level in feet.</TD></TR>
  <TR><TD>continent</TD><TD>Continent code indicating the geographic region of the airport.</TD></TR>
  <TR><TD>iso_country</TD><TD>ISO 3166-1 alpha-2 country code where the airport is located.</TD></TR>
  <TR><TD>iso_region</TD><TD>ISO 3166-2 regional subdivision code within the country.</TD></TR>
  <TR><TD>municipality</TD><TD>Primary municipality or city served by the airport.</TD></TR>
  <TR><TD>scheduled_service</TD><TD>Indicates whether the airport currently supports scheduled airline service (yes/no).</TD></TR>
  <TR><TD>gps_code</TD><TD>Airport code used in aviation GPS databases.</TD></TR>
  <TR><TD>icao_code</TD><TD>Four-letter ICAO airport identifier.</TD></TR>
  <TR><TD>iata_code</TD><TD>Three-letter IATA airport code used by airlines.</TD></TR>
  <TR><TD>local_code</TD><TD>Local airport code used primarily for U.S. airports.</TD></TR>
  <TR><TD>home_link</TD><TD>URL of the airport’s official website.</TD></TR>
  <TR><TD>wikipedia_link</TD><TD>URL of the airport’s Wikipedia page.</TD></TR>
  <TR><TD>keywords</TD><TD>Additional search keywords or alternate names associated with the airport.</TD></TR>

  
  <!-- runways data -->
  <TR>
    <TD COLSPAN=2><H4>Runways Dataset</H4></TD>
  </TR>
  <TR><TD>id</TD><TD>Internal OurAirports identifier for the runway.</TD></TR>
  <TR><TD>airport_ref</TD><TD>Internal foreign key linking the runway to the associated airport record.</TD></TR>
  <TR><TD>airport_ident</TD><TD>Airport identifier corresponding to the associated airport in airports.csv.</TD></TR>
  <TR><TD>length_ft</TD><TD>Total runway length in feet, including displaced thresholds and overruns.</TD></TR>
  <TR><TD>width_ft</TD><TD>Width of the runway in feet.</TD></TR>
  <TR><TD>surface</TD><TD>Runway surface type (e.g., asphalt, concrete, turf, gravel, water, or unknown).</TD></TR>
  <TR><TD>lighted</TD><TD>Indicates whether the runway has lighting (1 = lighted, 0 = not lighted).</TD></TR>
  <TR><TD>closed</TD><TD>Indicates whether the runway is currently closed (1 = closed, 0 = open).</TD></TR>
  <TR><TD>le_ident</TD><TD>Identifier for the low-numbered end of the runway.</TD></TR>
  <TR><TD>le_latitude_deg</TD><TD>Latitude of the low-numbered runway end in decimal degrees.</TD></TR>
  <TR><TD>le_longitude_deg</TD><TD>Longitude of the low-numbered runway end in decimal degrees.</TD></TR>
  <TR><TD>le_elevation_ft</TD><TD>Elevation of the low-numbered runway end in feet above mean sea level.</TD></TR>
  <TR><TD>le_heading_degT</TD><TD>True heading of the low-numbered runway end in degrees.</TD></TR>
  <TR><TD>le_displaced_threshold_ft</TD><TD>Length of displaced threshold at the low-numbered runway end in feet.</TD></TR>
  <TR><TD>he_ident</TD><TD>Identifier for the high-numbered end of the runway.</TD></TR>
  <TR><TD>he_latitude_deg</TD><TD>Latitude of the high-numbered runway end in decimal degrees.</TD></TR>
  <TR><TD>he_longitude_deg</TD><TD>Longitude of the high-numbered runway end in decimal degrees.</TD></TR>
  <TR><TD>he_elevation_ft</TD><TD>Elevation of the high-numbered runway end in feet above mean sea level.</TD></TR>
  <TR><TD>he_heading_degT</TD><TD>True heading of the high-numbered runway end in degrees.</TD></TR>
  <TR><TD>he_displaced_threshold_ft</TD><TD>Length of displaced threshold at the high-numbered runway end in feet.</TD></TR>
  
  <!-- Airline Delay Cause Data-->
  <TR>
    <TD COLSPAN=2><H4>Airline Delay Cause Dataset</H4></TD>
  </TR>
  <TR><TD>year</TD><TD>Year in YYYY format.</TD></TR>
  <TR><TD>month</TD><TD>Month in MM format (1–12).</TD></TR>
  <TR><TD>carrier</TD><TD>Code assigned by the U.S. DOT to uniquely identify an airline carrier.</TD></TR>
  <TR><TD>carrier_name</TD><TD>Unique airline name as defined by a single holding and reporting certificate.</TD></TR>
  <TR><TD>airport</TD><TD>Three-character alphanumeric airport code issued by the U.S. Department of Transportation.</TD></TR>
  <TR><TD>airport_name</TD><TD>Name of the airport.</TD></TR>
  <TR><TD>arr_flights</TD><TD>Total number of arrival flights.</TD></TR>
  <TR><TD>arr_del15</TD><TD>Number of arrivals delayed by 15 minutes or more.</TD></TR>
  <TR><TD>carrier_ct</TD><TD>Count of delays attributed to the carrier.</TD></TR>
  <TR><TD>weather_ct</TD><TD>Count of delays attributed to weather.</TD></TR>
  <TR><TD>nas_ct</TD><TD>Count of delays attributed to the National Air System (NAS).</TD></TR>
  <TR><TD>security_ct</TD><TD>Count of delays attributed to security issues.</TD></TR>
  <TR><TD>late_aircraft_ct</TD><TD>Count of delays attributed to late incoming aircraft.</TD></TR>
  <TR><TD>arr_cancelled</TD><TD>Number of cancelled flights.</TD></TR>
  <TR><TD>arr_diverted</TD><TD>Number of diverted flights.</TD></TR>
  <TR><TD>arr_delay</TD><TD>Total arrival delay in minutes, relative to scheduled arrival time.</TD></TR>
  <TR><TD>carrier_delay</TD><TD>Total carrier delay in minutes.</TD></TR>
  <TR><TD>weather_delay</TD><TD>Total weather delay in minutes.</TD></TR>
  <TR><TD>nas_delay</TD><TD>Total National Air System delay in minutes.</TD></TR>
  <TR><TD>security_delay</TD><TD>Total security delay in minutes.</TD></TR>
  <TR><TD>late_aircraft_delay</TD><TD>Total late aircraft delay in minutes.</TD></TR>

</TABLE>

</BODY>
</HTML>
