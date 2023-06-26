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

Fitting model on 533 points
Predicted asset type with 1.0 precision

Fitting model on 28325 points
Predicted asset type with 0.993 precision

Predicting 29471 points
Failed to predict asset type for 1219 points

Top 30 features (out of 761):
VAV-#H (0.3032)
AHU (0.1665)
VAV (0.0939)
LTG (0.0562)
BACNET (0.0383)
CAV (0.0317)
WCC (0.0273)
MFDAMPER#-S (0.0231)
PLTG-Z#-M (0.0166)
PLTG-Z#-C (0.0123)
PLTG-Z#-S (0.0103)
EAFT#L#-S (0.006)
FCU-G-S (0.0059)
MODAMPER-S (0.0058)
EAFT#L#-DS (0.0055)
MODAMPER-C (0.0051)
EAFT#L#-M (0.0049)
FCU-G-C (0.0048)
EAFT#L#-C (0.0044)
AI (0.0042)
EAFT#L#-AFS (0.004)
EAFT#L#-MA (0.0037)
EAFT#L#-FT (0.0036)
EAFT#L#-PS (0.0036)
EAFT#L#-EMS (0.0035)
AHU-T#-L#-#-FLOW-L (0.0026)
MFDAMPER-S (0.0019)
CAV-SA-TOTAL-FLOW-SP (0.0013)
FAF-T#-URF-#-AM (0.0012)
FAF-T#-URF-#-S (0.0012)
