from __future__ import annotations

RAW_CSV_PATH = "Tema Data.csv"

# Canonical columns (as in the CSV)
COL_LOCATION_ID = "Location ID"
COL_LOCATION_NAME = "Location Name"
COL_PLACE_OPEN = "Place Open"
COL_LOCAL_DT = "Local Date/Time"
COL_UTC_DT = "UTC Date/Time"
COL_AGG_RECORDS = "# of aggregated records"

COL_PM25_RAW = "PM2.5 (μg/m³) raw"
COL_PM25_CORR = "PM2.5 (μg/m³) corrected"
COL_PARTICLE_03 = "0.3μm particle count"
COL_CO2_CORR = "CO2 (ppm) corrected"
COL_TEMP_CORR = "Temperature (°C) corrected"
COL_HUMIDITY_CORR = "Humidity (%) corrected"
COL_HEAT_INDEX = "Heat Index (°C)"
COL_TVOC = "TVOC (ppb)"
COL_TVOC_INDEX = "TVOC index"
COL_NOX_INDEX = "NOX index"
COL_PM1 = "PM1 (μg/m³)"
COL_PM10 = "PM10 (μg/m³)"

TARGET_COL = COL_PM25_CORR

