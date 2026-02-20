import os
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Paths
FLIGHTS_PATH = 'data/raw/all_flights_2018-2022_raw.parquet'
WEATHER_PATH = 'data/raw/flightsweather.parquet'

OUT_DIR = 'data/cleansed'
OUT_PATH = os.path.join(OUT_DIR, 'flight_weather_merged.parquet')
os.makedirs(OUT_DIR, exist_ok=True)

# batch size for streaming flights parquet (big file)
BATCH_SIZE = 600_000

print('Flights:', FLIGHTS_PATH)
print('Weather:', WEATHER_PATH)
print('Merged Clean Dataset:', OUT_PATH)

# Region mapping the (U.S. Census regions)
CENSUS_REGIONS_TO_STATE = {
    'Northeast': ['CT','ME','MA','NH','RI','VT','NJ','NY','PA'],
    'Midwest': ['IL','IN','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD'],
    'South': ['DE','DC','FL','GA','MD','NC','SC','VA','WV','AL','KY','MS','TN','AR','LA','OK','TX'],
    'West': ['AZ','CO','ID','MT','NV','NM','UT','WY','AK','CA','HI','OR','WA']
}

STATE_TO_REGION = {state: region for region, states in CENSUS_REGIONS_TO_STATE.items() for state in states}

# Add extra U.S. territories (not included in Cesus regions list) 
STATE_TO_REGION['PR'] = 'South' # puerto rico 
STATE_TO_REGION['VI'] = 'South' # virgin islands

# Top 30 airports and their states 
# used to build region-hour weather averages from weather dataset 
AIRPORT_TO_STATE = {
    'ATL':'GA','DFW':'TX','DEN':'CO','ORD':'IL','LAX':'CA','JFK':'NY','LAS':'NV','MCO':'FL','MIA':'FL','CLT':'NC',
    'SEA':'WA','PHX':'AZ','EWR':'NJ','SFO':'CA','IAH':'TX','BOS':'MA','FLL':'FL','MSP':'MN','LGA':'NY','DTW':'MI',
    'PHL':'PA','SLC':'UT','DCA':'DC','SAN':'CA','BWI':'MD','TPA':'FL','PDX':'OR','MDW':'IL','BNA':'TN','AUS':'TX', 'IAD':'VA','DAL':'TX', 'STL':'MO','HOU':'TX','SJC':'CA','RDU':'NC', 'HNL':'HI', 'SMF':'CA','MSY':'LA','OAK':'CA','MCI':'MO','IND':'IN','CLE':'OH','PIT':'PA','SNA':'CA', 'CVG':'KY','CMH':'OH','RSW':'FL','SAT':'TX','MKE':'WI'
}

# Read weather and group by hour per airport
weather_cols = ['airport_code','valid','tmpf','vsby','sknt','p01i','relh','gust']

weather_schema_cols = pq.read_schema(WEATHER_PATH).names
weather_cols_use = [col for col in weather_cols if col in weather_schema_cols]

# error if wrong file is downloaded
missing_required = [col for col in ['airport_code', 'valid'] if col not in weather_cols_use]
if missing_required:
    raise KeyError(f'Weather parquet missing required cols: {missing_required}')
    
weather_table = pq.read_table(WEATHER_PATH, columns=weather_cols_use)

# Get valid column from raw weather data
# pull out date and hour for merging with flights departure hour 
valid_ts = pc.cast(weather_table['valid'], pa.timestamp('us'))
# extracting date
w_date = pc.strftime(valid_ts, format='%Y-%m-%d')
# extracting hour 
w_hour = pc.hour(valid_ts)

weather_table = weather_table.append_column('w_date', w_date)
weather_table = weather_table.append_column('w_hour', w_hour)

# Group weather to hourly per airport 
weather_group_rules = []
for col in ['tmpf','vsby','sknt','p01i','relh','gust']:
    if col not in weather_table.column_names:
        continue
    if col == 'p01i':
        weather_group_rules.append((col, 'sum'))
    elif col == 'gust':
        weather_group_rules.append((col, 'max'))
    else:
        weather_group_rules.append((col, 'mean'))

weather_hourly = weather_table.group_by(['airport_code', 'w_date', 'w_hour']).aggregate(weather_group_rules)
weather_hourly = weather_hourly.to_pandas()

# Convert date/hour into pandas dtypes
weather_hourly['w_date'] = pd.to_datetime(weather_hourly['w_date'], errors='coerce').dt.normalize()
weather_hourly['w_hour'] = pd.to_numeric(weather_hourly['w_hour'], errors='coerce').astype('int16')

# Rename group by cols back to OG names
rename_map = {f'{col}_{fn}': col for col, fn in weather_group_rules}
weather_hourly = weather_hourly.rename(columns=rename_map)

weather_features = list(rename_map.values())
if not weather_features:
    raise ValueError('No weather features available after grouping')

# Use temperature to check if a weather match exists
anchor_feature = 'tmpf' if 'tmpf' in weather_features else weather_features[0]

# FALLBACK OPTION: Build region-hour weather averages
weather_region = weather_hourly.copy()
weather_region['state'] = weather_region['airport_code'].map(AIRPORT_TO_STATE)
weather_region['region'] = weather_region['state'].map(STATE_TO_REGION)

region_hourly = (
    weather_region.dropna(subset=['region'])
    .groupby(['region', 'w_date', 'w_hour'], as_index=False)[weather_features]
    .mean()
)
region_hourly['w_date'] = pd.to_datetime(region_hourly['w_date'])

del weather_table, weather_region
gc.collect()

# Read flights in batches, build keys, and merge weather
flight_cols_needed = [
    'FlightDate', 'Airline', 'Origin', 'OriginState',
    'CRSDepTime', 'Cancelled', 'DepDelay', 'Distance',
    'Month', 'DayOfWeek'
]

flight_schema_cols = pq.read_schema(FLIGHTS_PATH).names
flight_cols_use = [col for col in flight_cols_needed if col in flight_schema_cols]

# Must have columns
must_have = ['FlightDate','Origin','OriginState','CRSDepTime']
missing = [col for col in must_have if col not in flight_cols_use]
if missing:
    raise KeyError(f'Flights parquet missing required cols: {missing}')

pf = pq.ParquetFile(FLIGHTS_PATH)

# delete old output to not interfere
if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)

writer = None
rows_written=0

for batch_idx , batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=flight_cols_use)):
    df = batch.to_pandas()
    if df.empty:
        continue

    # Converting to datetime 
    df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')

    # Remove 2020 (COVID year) due to anomalies 
    df = df[df['FlightDate'].dt.year != 2020]
    
    # Convert datatypes + clean strings
    df['CRSDepTime'] = pd.to_numeric(df['CRSDepTime'], errors='coerce')
    df['Origin'] = df['Origin'].astype(str).str.upper().str.strip()
    # Normalize state abbreviations to capture all 
    df['OriginState'] = df['OriginState'].astype(str).str.upper().str.strip()

    df = df.dropna(subset=['FlightDate','Origin','OriginState','CRSDepTime']).copy()

    # Remove non-US territories (TT: Trinidad and Tobago)
    df = df[df['OriginState'] != 'TT']

    # Make merge keys for flights
    # CRSDepTime is HHMM, so //100 to get hour (0-23)
    df['dep_hour'] = (df['CRSDepTime'] // 100).astype('int16').clip(0, 23)
    
    # have to normalize so parquet schema stays stable 
    df['f_date'] = df['FlightDate'].dt.normalize()

    # Create region feature from OriginState
    df['region'] = df['OriginState'].map(STATE_TO_REGION)

    # Merge airport-hour weather
    m1 = df.merge(
        weather_hourly,
        left_on=['Origin', 'f_date', 'dep_hour'],
        right_on=['airport_code','w_date','w_hour'],
        how='left',
        copy=False
    )
    has_airport_weather = m1[anchor_feature].notna()

    # Merge region-hour weather (fallback-option 2)
    m2 = m1.merge(
        region_hourly,
        left_on=['region', 'f_date', 'dep_hour'],
        right_on=['region', 'w_date', 'w_hour'],
        how='left',
        suffixes=('', '_reg'),
        copy=False
    )

    # Fill missing airport-hour weather with region-hour averages
    for col in weather_features:
        reg_col = f'{col}_reg'
        if reg_col in m2.columns:
            m2[col] = m2[col].where(m2[col].notna(), m2[reg_col])
    has_after = m2[anchor_feature].notna()

    # Track where the weather values came from 
    m2['weather_source'] = np.select(
        [has_airport_weather, (~has_airport_weather) & has_after],
        ['airport_hour_match', 'region_hour_match',],
        default='still_missing'
    )
    # Keep only the columns we need
    keep_cols = [
        'FlightDate', 'Airline', 'Origin', 'OriginState',
        'CRSDepTime', 'Cancelled', 'DepDelay','Distance', 
        'Month', 'DayOfWeek', 'dep_hour', 'f_date', 'region', 'weather_source'
    ] + weather_features

    keep_cols = [col for col in keep_cols if col in m2.columns]

    out_chunk = m2[keep_cols].copy()

    
    # Force stable dtypes so schema stays consistent between batches
    for col in ['Airline', 'Origin','OriginState','region','weather_source']:
        if col in out_chunk.columns:
            out_chunk[col] = out_chunk[col].astype('string')

    if 'f_date' in out_chunk.columns:
        out_chunk['f_date'] = pd.to_datetime(out_chunk['f_date'], errors='coerce').dt.date
        

    table = pa.Table.from_pandas(out_chunk, preserve_index=False)

    if writer is None:
        fixed_schema = table.schema
        writer = pq.ParquetWriter(OUT_PATH, table.schema, compression='snappy')

    table = table.cast(fixed_schema)

    writer.write_table(table)
    rows_written += len(out_chunk)

    # Delete for memory purposes
    del df, m1, m2, out_chunk, table, has_airport_weather, has_after
    gc.collect()

if writer is not None:
    writer.close()


print('Saved:', OUT_PATH)