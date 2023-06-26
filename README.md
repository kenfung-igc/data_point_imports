# data_point_imports

This repository is dedicated to importing data points from external systems:

## SHK BMS System

Under shk/bms, the main.py script takes as input: 
* a list of asset groups from Planon mapped to asset types in BMS (asset_groups.csv)
* a list of BMS points manually mapped to Planon asset types (points_manually_mapped.csv)
* a full list of BMS points to be mapped (points.py)

It employs machine learning model to map each point to Planon asset group and outputs:
* a mapping of points to Planon asset codes (point_asset_code_<timestamp>.csv)
* a list of point types for each asset group (point_asset_group_<timestamp>.csv)
* a summary of model performance and important phrases used (sample below)

Fit model on 533 points
Predicted asset type with 0.991 precision

Fit model on 28325 points
Predicted asset type with 0.991 precision

Predicted 29471 points
Fail to predict asset type for 1251 points.

Top 30 features (out of 761):
VAV-#H (0.2669)
AHU (0.1459)
VAV (0.1338)
LTG (0.0749)
AI (0.0399)
WCC (0.0308)
CAV (0.03)
MFDAMPER#-S (0.0244)
PLTG-Z#-S (0.0168)
PLTG-Z#-M (0.0122)
PLTG-Z#-C (0.0111)
EAFT#L#-PS (0.0058)
FCU-G-C (0.0056)
EAFT#L#-M (0.0053)
MODAMPER-S (0.0052)
FCU-G-S (0.0049)
EAFT#L#-DS (0.0048)
EAFT#L#-AFS (0.0043)
MODAMPER-C (0.0042)
EAFT#L#-C (0.0039)
EAFT#L#-S (0.0037)
EAFT#L#-FT (0.0036)
EAFT#L#-EMS (0.0035)
EAFT#L#-MA (0.0034)
AHU-T#-L#-#-FLOW-L (0.0024)
MFDAMPER-S (0.0021)
CAV-SA-TOTAL-FLOW-SP (0.0013)
PAU-SA-FR (0.0011)
FAF-T#-URF-#-C (0.0011)
FAF-T#-URF-#-MA (0.001)



