# MSS data forecast system

## Introduction

This is a system for forecasting the Mobile Spatial Statistic (MSS) data. The system is based on the [Mobaku](https://mobaku.jp/) data.

## Requirements

Python 3.11.6

## Usage

1. Download the MSS data and store unipped in `HOME_DIR` directory.
2. Create an `.env` file with the following contents:  

    ```text
    HOME_DIR = <path to the directory where the data is stored> / Folder path
    JAPAN_MESH4 = <path to the Japan Mesh data> / GeoJSON file address
    ```  
  
3. In `main.py` modify the parameters:  

    ```text
    MESH_ID = <`int` with MESH4 code> (e.g. 503324732) (not in used) 
    AOI_NAME = <`str` name of project> (folder name of project)
    AOI_POLYGON = <`str` path of a `shp` polygon file to extract MSS data in mesh4.>
    AOI_MESH = <`str` path to a `geojson` file of mesh4 data.>
    EVENT_NAME = <`str` name of event>
    EVENT_DATE_START = <`date string`> (e.g. "2023-11-01")
    EVENT_DATE_MAIN = <`date string`> (e.g. "2023-11-10")
    EVENT_DATE_END = <`date string`> (e.g. "2023-11-20")
    FILE_PREFIX = <`str` prefix for files. No spaces.>
    SEASONALITY = <`int` parameter for SARIMA>
    HOURS_TO_FORECAST = <`int` hours to forecast.> (currently 3 is only possible)
    PERCENTAGE_OF_DATA_FOR_TRAINING = <`float` number to split training and test data> (e.g. 0.7)
    ```

4. Run `python main.py` to start the system.

## Output

The system will output the following files:
`AOI_NAME` is used to create the folder of the project. Inside this folder each simulation will be stored in a subfolder with the current date and time as folder name (i.e. 20231201105613 is 2023-12-01 10:56:13).
Within the datetime folder, the following folders will be created:  

- data : contains the data used for the simulation.
  - GeoJSON of the area of interest.
  - Pickle file with the MSS data and the Class Object.
- figures : contains the figures and plot animation of the simulation.
- output : contains a csv file with the forecasted data. (`meshid`,`1h`,`2h`,`3h`)
- plots : contains the plots of the simulation (i.e. data, forecast errors, etc.).
