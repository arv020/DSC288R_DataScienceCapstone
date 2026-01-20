
# DSC288R_DataScienceCapstone
Capstone project for DSC288R course for UCSD Data Science Masters Program

<HTML>
<BODY>
	<TABLE><TR><TD COLSPAN=2><H4>Airport Flights - On Time, Delayed, or Cancelled?</H4></TD></TR>
	<TR><TD COLSPAN=2>In order to download the data, run the script "1_download_data.py". After this you will be able to access the data locally as a parquet file. </TD></TR>
	<TR><TD COLSPAN=2>&nbsp;</TD></TR>
	<TR><TD COLSPAN=2><H4>Data Dictionary</H4></TD></TR>
    <TR><TD>Year</TD><TD>Year</TD></TR>
    <TR><TD>Quarter</TD><TD>Quarter (1-4)</TD></TR>
    <TR><TD>Month</TD><TD>Month</TD></TR>
    <TR><TD>DayofMonth</TD><TD>Day of Month</TD></TR>
    <TR><TD>DayOfWeek</TD><TD>Day of Week</TD></TR>
    <TR><TD>FlightDate</TD><TD>Flight Date (yyyymmdd)</TD></TR>

    <TR><TD>Marketing_Airline_Network</TD><TD>Unique Marketing Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users.</TD></TR>
    <TR><TD>Operated_or_Branded_Code_Share_Partners</TD><TD>Reporting Carrier Operated or Branded Code Share Partners.</TD></TR>
    <TR><TD>DOT_ID_Marketing_Airline</TD><TD>Identification number assigned by US DOT to identify a unique airline (carrier).</TD></TR>
    <TR><TD>IATA_Code_Marketing_Airline</TD><TD>Code assigned by IATA and commonly used to identify a carrier.</TD></TR>
    <TR><TD>Flight_Number_Marketing_Airline</TD><TD>Flight Number.</TD></TR>

    <TR><TD>Operating_Airline</TD><TD>Unique Carrier Code used for analysis across a range of years.</TD></TR>
    <TR><TD>DOT_ID_Operating_Airline</TD><TD>Identification number assigned by US DOT to identify a unique operating airline.</TD></TR>
    <TR><TD>IATA_Code_Operating_Airline</TD><TD>IATA code identifying the operating carrier.</TD></TR>
    <TR><TD>Tail_Number</TD><TD>Aircraft Tail Number.</TD></TR>
    <TR><TD>Flight_Number_Operating_Airline</TD><TD>Operating Flight Number.</TD></TR>

    <TR><TD>OriginAirportID</TD><TD>Origin Airport ID assigned by US DOT.</TD></TR>
    <TR><TD>OriginAirportSeqID</TD><TD>Origin Airport Sequence ID for time-specific airport information.</TD></TR>
    <TR><TD>OriginCityMarketID</TD><TD>Origin City Market ID.</TD></TR>
    <TR><TD>Origin</TD><TD>Origin Airport Code.</TD></TR>
    <TR><TD>OriginCityName</TD><TD>Origin Airport City Name.</TD></TR>
    <TR><TD>OriginState</TD><TD>Origin Airport State Code.</TD></TR>
    <TR><TD>OriginStateFips</TD><TD>Origin Airport State FIPS Code.</TD></TR>
    <TR><TD>OriginStateName</TD><TD>Origin Airport State Name.</TD></TR>
    <TR><TD>OriginWac</TD><TD>Origin Airport World Area Code.</TD></TR>

    <TR><TD>DestAirportID</TD><TD>Destination Airport ID assigned by US DOT.</TD></TR>
    <TR><TD>DestAirportSeqID</TD><TD>Destination Airport Sequence ID.</TD></TR>
    <TR><TD>DestCityMarketID</TD><TD>Destination City Market ID.</TD></TR>
    <TR><TD>Dest</TD><TD>Destination Airport Code.</TD></TR>
    <TR><TD>DestCityName</TD><TD>Destination Airport City Name.</TD></TR>
    <TR><TD>DestState</TD><TD>Destination Airport State Code.</TD></TR>
    <TR><TD>DestStateFips</TD><TD>Destination Airport State FIPS Code.</TD></TR>
    <TR><TD>DestStateName</TD><TD>Destination Airport State Name.</TD></TR>
    <TR><TD>DestWac</TD><TD>Destination Airport World Area Code.</TD></TR>

    <TR><TD>CRSDepTime</TD><TD>Scheduled Departure Time (local time: hhmm).</TD></TR>
    <TR><TD>DepTime</TD><TD>Actual Departure Time (local time: hhmm).</TD></TR>
    <TR><TD>DepDelay</TD><TD>Difference in minutes between scheduled and actual departure time.</TD></TR>
    <TR><TD>DepDelayMinutes</TD><TD>Departure delay in minutes (early departures set to 0).</TD></TR>
    <TR><TD>DepDel15</TD><TD>Departure Delay Indicator (15 minutes or more).</TD></TR>
    <TR><TD>DepartureDelayGroups</TD><TD>Departure delay grouped in 15-minute intervals.</TD></TR>
    <TR><TD>DepTimeBlk</TD><TD>Scheduled Departure Time Block.</TD></TR>
    <TR><TD>TaxiOut</TD><TD>Taxi-out time in minutes.</TD></TR>
    <TR><TD>WheelsOff</TD><TD>Time aircraft wheels left the ground.</TD></TR>

    <TR><TD>WheelsOn</TD><TD>Time aircraft wheels touched the ground.</TD></TR>
    <TR><TD>TaxiIn</TD><TD>Taxi-in time in minutes.</TD></TR>
    <TR><TD>CRSArrTime</TD><TD>Scheduled Arrival Time (local time: hhmm).</TD></TR>
    <TR><TD>ArrTime</TD><TD>Actual Arrival Time (local time: hhmm).</TD></TR>
    <TR><TD>ArrDelay</TD><TD>Difference in minutes between scheduled and actual arrival time.</TD></TR>
    <TR><TD>ArrDelayMinutes</TD><TD>Arrival delay in minutes (early arrivals set to 0).</TD></TR>
    <TR><TD>ArrDel15</TD><TD>Arrival Delay Indicator (15 minutes or more).</TD></TR>
    <TR><TD>ArrivalDelayGroups</TD><TD>Arrival delay grouped in 15-minute intervals.</TD></TR>
    <TR><TD>ArrTimeBlk</TD><TD>Scheduled Arrival Time Block.</TD></TR>

    <TR><TD>Cancelled</TD><TD>Cancelled Flight Indicator (1=Yes).</TD></TR>
    <TR><TD>Diverted</TD><TD>Diverted Flight Indicator (1=Yes).</TD></TR>

    <TR><TD>CRSElapsedTime</TD><TD>Scheduled elapsed flight time in minutes.</TD></TR>
    <TR><TD>ActualElapsedTime</TD><TD>Actual elapsed flight time in minutes.</TD></TR>
    <TR><TD>AirTime</TD><TD>Flight time in minutes.</TD></TR>

    <TR><TD>Distance</TD><TD>Distance between airports in miles.</TD></TR>
    <TR><TD>DistanceGroup</TD><TD>Distance intervals in 250-mile groups.</TD></TR>

    <TR><TD>DivAirportLandings</TD><TD>Number of diverted airport landings.</TD></TR>
</TABLE>
</BODY>
</HTML>