# Planet_Labs_Technical_Assessment
Take Home Test for Planet Labs Interview

# Task

The data folder includes GeoTIFFs and metadata files for a number of PlanetScope acquisitions over Sumatra between June and September, 2020. The AnalyticMS files are analytic assets from PSScene4Band imagery; the udm2 files are Usable Data Masks; and the metadata.json files include the full imagery footprint, acquisition times, and additional metadata. All raster data is clipped to the same extent (which is different than the full imagery footprints).

Your task is to write a program that analyzes the imagery and produces an approximate measure indicating the rate of change from green vegetation to bare soil over the time period represented by the image series.

In submitting your task, please provide instructions on running your program and include any additional detail describing the choices you made or issues you encountered.

# Instructions

# Additional Details About Program

% Challenges

1) This was my first remote sensing project using Python and Github. As such, it took some time to shift gears from MATLAB to Python Syntax. 
2) The Term "green vegetation" was difficult to interprete (i.e., what is considered green vegetation?). This challenge is handled by defined green vegetation based on the green absoption feature.


