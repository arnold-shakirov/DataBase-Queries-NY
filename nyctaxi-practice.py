import pandas as pd
import datetime

df = pd.read_parquet("/data/nyctaxi/yellow_tripdata_2023-11.parquet")
print(df['tpep_pickup_datetime'])
df['date'] = df['tpep_pickup_datetime'].dt.date
df = df[(df['date'] > datetime.date(2023,10,31)) & (df['date'] < datetime.date(2023,12,1))]
print(df.columns)
df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

# Find average trip distance
print("Avg trip distance:", df['trip_distance'].mean())

# Find min and max fare amount
print("Min fare amount:", df['fare_amount'].min())
print("Max fare amount:", df['fare_amount'].max())

# Find average fare amount per # of passengers
print("Average fare amount per # of passengers:", df.groupby('passenger_count')['fare_amount'].mean())

# Find average fare amount for trips from the airport
print("Airport rides (avg fare):", df.query('Airport_fee > 0')['fare_amount'].mean())

"""
Tasks:

- Find average trip distance
- Find min and max fare amount
- Find average fare amount per # of passengers
- Find average fare amount for trips from the airport
- Find average congestion surcharge for each hour of the day; and for each day of the week

- Find most frequent pick up and drop off locations
- Find most frequent pick up/drop off pair
- Find most frequent pick up locations for night hours on weekends

- It's 3:35pm on a Saturday. I am at the Met. How much will it cost me and my two friends to get to the World Trade Center?
  - Met: #236
  - WTC: #261

Graphs:

- Trips per day
- Average trip distance per hour of the day
- Average fare amount per # of passengers
- Average fare amount for trips from the airport vs. non-airport
- Average congestion surcharge per hour of the day; and per day of the week (grid)
- Overlay on map: most frequent pick up and drop off locations

"""


import matplotlib.pyplot as plt

# Trips per day
#xy = df['tpep_pickup_datetime'].dt.date.value_counts().sort_index()
#x = xy.index
#y = xy.values
#print(x)
#print(y)

xy = df.groupby('date')['VendorID'].count()
x = xy.index
y = xy.values
print(x)
print(y)

df['week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
xy_byweek = df.groupby('week')['VendorID'].count()
x_byweek = xy_byweek.index
y_byweek = xy_byweek.values
print(x_byweek)
print(y_byweek)

plt.plot(x, y)
plt.axvline(datetime.date(2023,11,23), color='r', linestyle='--')
# add a vertical at each Monday
for i in range(len(x)):
    if x[i].weekday() == 0:
        plt.axvline(x[i], color='g', linestyle=':')
# plot mean
plt.axhline(y.mean(), color='k', linestyle='-.')
# plot rectangle of largest week
max_week = y_byweek.argmax()
plt.axvspan(datetime.date.fromisocalendar(2023, x_byweek[max_week], 1), datetime.date.fromisocalendar(2023, x_byweek[max_week], 7), alpha=0.3, color='y')
plt.xlabel('Date')
plt.ylabel('Trips')
plt.title('Trips per day')
plt.xticks(rotation=45)
plt.ylim(0, y.max() + 10000)
# vertical line at thanksgiving
plt.tight_layout()
plt.savefig('trips_per_day.png')

