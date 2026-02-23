# features for models 
# Model A
model_a_feature_cols = [
    # departure time
    "dep_hour", "dep_hour_sin", "dep_hour_cos",
    "is_early_morning", "is_evening", "is_red_eye",

    # seasonal 
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_summer", "is_holiday_season",

    # raw weather
    "tmpf", "vsby", "sknt", "p01i", "relh", "gust",

    # weather thresholds
    "high_wind", "low_visibility",
    "precip_any", "precip_light", "precip_moderate", "precip_heavy",

    # weather interactions 
    "precip_x_lowvis", "precip_x_highwind", "wind_precip",

    # weather source and severity
    "weather_severity", "weather_source_enc",

    # encoded features
    "Airline_enc", "Origin_enc", "Dest_enc", "region_enc",

    # cancellation rate/risk features
    "airline_cancel_rate", "airport_cancel_rate",
    "destination_cancel_rate", "route_cancel_risk",

    # lag features 
    "lag1_cancel_rate", "cancelled_yesterday",
    "lag1_delay_rate", "lag1_volume",  

    # congestion
    "hourly_flights", "hourly_flights_log",

    # distance
    "Distance", "log_distance",
]


# Model B
model_b_feature_cols = [
    # departure time
    "dep_hour", "dep_hour_sin", "dep_hour_cos",
    "is_early_morning", "is_evening", "is_midday", "is_red_eye",

    # seasonal
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_summer", "is_holiday_season",

    # raw weather
    "tmpf", "vsby", "sknt", "p01i", "relh", "gust",

    # weather thresholds
    "high_wind", "low_visibility",
    "precip_any", "precip_light", "precip_moderate", "precip_heavy",

    # weather interactions
    "precip_x_lowvis", "precip_x_highwind", "wind_precip",

    # weather source and severity
    "weather_severity", "weather_source_enc",

    # encoded features
    "Airline_enc", "Origin_enc", "Dest_enc", "region_enc",

    # delay rate/risk features
    "airline_delay_rate", "airport_delay_rate",
    "destination_delay_rate", "route_delay_risk",

    # lag features
    "lag1_delay_rate", "lag1_cancel_rate", "lag1_volume",

    # congestion
    "hourly_flights", "hourly_flights_log",

    # time and congestion interactions
    "evening_x_congestion", "early_x_congestion", "hour_x_lag_delay",

    # distance
    "Distance", "log_distance",
]
