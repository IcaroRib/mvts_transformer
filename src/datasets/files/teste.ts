# Dataset Information
# The goal of this dataset is to predict PM2.5 and PM10 air quality in the city of Beijing.
# This dataset contains 17532 time series with 9 dimensions.
# This includes hourly air pollutants measurments (SO2, NO2, CO and O3), temperature, pressure, dew point, rainfall and windspeed measurments from 12 nationally controlled air quality monitoring sites.
# The air-quality data are from the Beijing Municipal Environmental Monitoring Center.
# The meteorological data in each air-quality site are matched with the nearest weather station from the China Meteorological Administration.
# The time period is from March 1st, 2013 to February 28th, 2017.
#
# Please refer to https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data for more details
#
# Relevant Papers
# Zhang, S., Guo, B., Dong, A., He, J., Xu, Z. and Chen, S.X. (2017) Cautionary Tales on Air-Quality Improvement in Beijing. Proceedings of the Royal Society A, Volume 473, No. 2205, Pages 20170457
#
# Citation Request
# Zhang, S., Guo, B., Dong, A., He, J., Xu, Z. and Chen, S.X. (2017) Cautionary Tales on Air-Quality Improvement in Beijing. Proceedings of the Royal Society A, Volume 473, No. 2205, Pages 20170457
@problemname BeijingPM25Quality
@timestamps true
@missing true
@univariate true
@dimension 1
@equallength true
@serieslength 24
@targetlabel true
@data
(2016-01-01 00:00:00,1.1),(2016-01-01 01:00:00,1.0),(2016-01-01 02:00:00,0.8),2016-01-01 03:00:00,1.5),(2016-01-01 04:00:00,1.0),(2016-01-01 05:00:00,1.2),(2016-01-01 06:00:00,0.7),(2016-01-01 07:00:00,0.7),(2016-01-01 08:00:00,0.9),(2016-01-01 09:00:00,1.2),(2016-01-01 10:00:00,1.4),(2016-01-01 11:00:00,1.2),(2016-01-01 12:00:00,0.8),(2016-01-01 13:00:00,1.0),(2016-01-01 14:00:00,0.5),(2016-01-01 15:00:00,1.1),(2016-01-01 16:00:00,1.0),(2016-01-01 17:00:00,1.1),(2016-01-01 18:00:00,1.2),(2016-01-01 19:00:00,1.2),(2016-01-01 20:00:00,0.7),(2016-01-01 21:00:00,0.9),(2016-01-01 22:00:00,0.1),(2016-01-01 23:00:00,1.4):324.0