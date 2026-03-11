import os
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)

# Paths
FLIGHTS_PATH = BASE_DIR / '1_download_data' / 'raw' / 'all_flights_2018-2022_raw.parquet'
WEATHER_PATH = BASE_DIR / '1_download_data' / 'raw' / 'weather_airports_2018_2022_CLEAN.parquet'
OPSNET_XLS   = BASE_DIR / '1_download_data' / 'raw' / 'WEB-Report-74040.xls'
DELAY_CSV    = BASE_DIR / '1_download_data' / 'raw' / 'Airline_Delay_Cause.csv'
AIRPORTS_CSV = BASE_DIR / '1_download_data' / 'raw' / 'airports.csv'
RUNWAYS_CSV  = BASE_DIR / '1_download_data' / 'raw' / 'runways.csv'

OUT_DIR    = BASE_DIR / '1_download_data' / 'cleansed'
os.makedirs(OUT_DIR, exist_ok=True)

MERGED_OUT = OUT_DIR / 'flight_weather_merged.parquet'
FINAL_OUT  = OUT_DIR / 'final_flights_model_dataset.parquet'

# batch size for streaming flights parquet (big file)
BATCH_SIZE = 600_000

print('Flights:', FLIGHTS_PATH)
print('Weather:', WEATHER_PATH)
print('Merged Clean Dataset:', MERGED_OUT)
print('Final Dataset:', FINAL_OUT)


# Stage 1: merge flights + weather 

    # Region mapping (U.S. Census regions)
    CENSUS_REGIONS_TO_STATE = {
        'Northeast': ['CT','ME','MA','NH','RI','VT','NJ','NY','PA'],
        'Midwest': ['IL','IN','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD'],
        'South': ['DE','DC','FL','GA','MD','NC','SC','VA','WV','AL','KY','MS','TN','AR','LA','OK','TX'],
        'West': ['AZ','CO','ID','MT','NV','NM','UT','WY','AK','CA','HI','OR','WA']
    }

    STATE_TO_REGION = {state: region for region, states in CENSUS_REGIONS_TO_STATE.items() for state in states}

    # Add extra U.S. territories (not included in Census regions list)
    STATE_TO_REGION['PR'] = 'South' # puerto rico
    STATE_TO_REGION['VI'] = 'South' # virgin islands

    # Top airports and their states
    # used to build region-hour weather averages from weather dataset
    AIRPORT_TO_STATE = {
        'ATL':'GA','DFW':'TX','DEN':'CO','ORD':'IL','LAX':'CA','JFK':'NY','LAS':'NV','MCO':'FL','MIA':'FL','CLT':'NC',
        'SEA':'WA','PHX':'AZ','EWR':'NJ','SFO':'CA','IAH':'TX','BOS':'MA','FLL':'FL','MSP':'MN','LGA':'NY','DTW':'MI',
        'PHL':'PA','SLC':'UT','DCA':'DC','SAN':'CA','BWI':'MD','TPA':'FL','PDX':'OR','MDW':'IL','BNA':'TN','AUS':'TX',
        'IAD':'VA','DAL':'TX','STL':'MO','HOU':'TX','SJC':'CA','RDU':'NC','HNL':'HI','SMF':'CA','MSY':'LA','OAK':'CA',
        'MCI':'MO','IND':'IN','CLE':'OH','PIT':'PA','SNA':'CA','CVG':'KY','CMH':'OH','RSW':'FL','SAT':'TX','MKE':'WI'
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
    if os.path.exists(MERGED_OUT):
        os.remove(MERGED_OUT)

    writer = None
    rows_written = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=flight_cols_use)):
        df = batch.to_pandas()
        if df.empty:
            continue

        # Converting to datetime
        df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')

        # Remove 2020 (COVID year) due to anomalies
        df = df[df['FlightDate'].dt.year != 2020]

        # Convert datatypes and clean strings
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

        # normalize date
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
            'CRSDepTime', 'Cancelled', 'DepDelay', 'Distance',
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
            writer = pq.ParquetWriter(str(MERGED_OUT), table.schema, compression='snappy')

        table = table.cast(fixed_schema)
        writer.write_table(table)
        rows_written += len(out_chunk)

        # Delete for memory purposes
        del df, m1, m2, out_chunk, table, has_airport_weather, has_after
        gc.collect()

    if writer is not None:
        writer.close()

    print('Saved:', MERGED_OUT)
    del weather_hourly, region_hourly
    gc.collect()


# Stage 2: merge with extra datasets 

flights = pd.read_parquet(MERGED_OUT)
flights['date']    = pd.to_datetime(flights['FlightDate'])
flights['airport'] = flights['Origin'].astype(str).str.strip()

# Dataset 1: parse OPSNET xls (multi-level header)
opsnet_raw = pd.read_html(str(OPSNET_XLS))[0]
opsnet_raw.columns = [
    '_'.join([str(x) for x in col if pd.notna(x)]).strip()
    for col in opsnet_raw.columns
]
date_col     = [c for c in opsnet_raw.columns if c.endswith('_Date')][0]
facility_col = [c for c in opsnet_raw.columns if c.endswith('_Facility')][0]
total_col    = [c for c in opsnet_raw.columns if 'Total Operations' in c][0]

opsnet = opsnet_raw[[date_col, facility_col, total_col]].copy()
opsnet.columns = ['date', 'airport', 'airport_operations']
opsnet['date'] = opsnet['date'].astype(str).str.strip()
opsnet = opsnet[~opsnet['date'].str.contains('Sub-Total|Total', na=False)]
opsnet['date'] = pd.to_datetime(opsnet['date'], format='%m/%d/%Y', errors='coerce')
opsnet = opsnet.dropna(subset=['date'])
opsnet['airport'] = opsnet['airport'].astype(str).str.strip()
opsnet['airport_operations'] = pd.to_numeric(opsnet['airport_operations'], errors='coerce')

flights = flights.merge(opsnet, on=['airport', 'date'], how='left')
flights['has_opsnet'] = flights['airport_operations'].notna().astype(int)
print(f'  OPSNET matched: {flights["airport_operations"].notna().mean()*100:.1f}% of flights')
del opsnet, opsnet_raw
gc.collect()


# Dataset 2: BTS delay causes (monthly, per airport)

delay = pd.read_csv(DELAY_CSV)
delay = delay[(delay['year'] >= 2018) & (delay['year'] <= 2022)]
delay = delay[delay['year'] != 2020]

delay_cols = ['carrier_delay', 'weather_delay', 'nas_delay',
              'security_delay', 'late_aircraft_delay']
have = [c for c in delay_cols if c in delay.columns]

delay_agg = delay.groupby(['airport', 'year', 'month']).agg(
    {c: 'sum' for c in have}
).reset_index()

flights['year']  = pd.to_datetime(flights['FlightDate']).dt.year
flights['month'] = pd.to_datetime(flights['FlightDate']).dt.month
flights = flights.merge(delay_agg, on=['airport', 'year', 'month'], how='left')
print(f'  delay causes matched: {flights[have[0]].notna().mean()*100:.1f}% of flights')
del delay, delay_agg
gc.collect()


# Dataset 3: airport metadata (elevation, lat/lon)
airports = pd.read_csv(AIRPORTS_CSV, usecols=[
    'iata_code', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'type'
]).copy()
airports = airports.rename(columns={
    'iata_code':      'airport',
    'latitude_deg':   'airport_lat',
    'longitude_deg':  'airport_lon',
    'elevation_ft':   'airport_elevation',
    'type':           'airport_type',
})
airports = airports.dropna(subset=['airport'])
airports['airport'] = airports['airport'].astype(str).str.strip()
airports = airports.drop_duplicates(subset=['airport'])

flights = flights.merge(airports, on='airport', how='left')
print(f'  airport metadata matched: {flights["airport_lat"].notna().mean()*100:.1f}% of flights')

# save before runway stream join
tmp_path = OUT_DIR / '_tmp_prefinal.parquet'
flights.to_parquet(tmp_path, index=False)
del flights, airports
gc.collect()


# Dataset 4: runway features

# build runway lookup (runways use ICAO codes, need to map to IATA)
air = pd.read_csv(AIRPORTS_CSV, usecols=['ident', 'iata_code']).dropna()
air['ident']     = air['ident'].astype(str).str.strip()
air['iata_code'] = air['iata_code'].astype(str).str.strip()
icao_to_iata = dict(zip(air['ident'], air['iata_code']))

runways = pd.read_csv(RUNWAYS_CSV, usecols=['airport_ident', 'length_ft', 'closed'])
runways['airport_ident'] = runways['airport_ident'].astype(str).str.strip()
runways['airport'] = runways['airport_ident'].map(icao_to_iata)
runways = runways.dropna(subset=['airport'])
runways = runways[runways['closed'] != 1].copy()
runways['length_ft'] = pd.to_numeric(runways['length_ft'], errors='coerce')
runways = runways.dropna(subset=['length_ft'])

runway_feats = runways.groupby('airport').agg(
    num_runways          = ('length_ft', 'size'),
    avg_runway_length_ft = ('length_ft', 'mean'),
    max_runway_length_ft = ('length_ft', 'max'),
).reset_index()

runway_tbl = pa.Table.from_pandas(runway_feats, preserve_index=False)

# stream-join to avoid loading the full file at once
fl_ds  = ds.dataset(str(tmp_path), format='parquet')
writer = None
n_written = 0

for batch in fl_ds.scanner(batch_size=200_000).to_batches():
    left   = pa.Table.from_batches([batch])
    joined = left.join(runway_tbl, keys='airport', join_type='left outer')
    if writer is None:
        schema = joined.schema
        writer = pq.ParquetWriter(str(FINAL_OUT), schema, compression='snappy')
    writer.write_table(joined.cast(schema))
    n_written += len(joined)

if writer:
    writer.close()

os.remove(tmp_path)

print(f'Saved: {FINAL_OUT} ({n_written:,} rows)')
