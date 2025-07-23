import json
import ast

if __name__ == "__main__":
    test_str = """[
  {
    "question": "What data was used to calculate the number of people potentially exposed to flooding in Somalia for the 2024 HNRP?",
    "options": [
      {
        "answer": "Daily FloodScan (1998-2022) & WorldPop (2020 UN Adjusted) raster data",
        "is_correct": true
      },
      {
        "answer": "ECMWF seasonal forecast data only",
        "is_correct": false
      },
      {
        "answer": "UNFPA Methodology data only",
        "is_correct": false
      },
      {
        "answer": "Somalia ICCG and HCT data only",
        "is_correct": false
      }
    ]
  },
  {
    "question": "What threshold was used to reclassify the FloodScan daily flood fraction Standard Flood Exposure Depiction (SFED) to binary?",
    "options": [
      {
        "answer": "10 percent flood fraction threshold",
        "is_correct": false
      },
      {
        "answer": "20 percent flood fraction threshold",
        "is_correct": true
      },
      {
        "answer": "30 percent flood fraction threshold",
        "is_correct": false
      },
      {
        "answer": "40 percent flood fraction threshold",
        "is_correct": false
      }
    ]
  },
  {
    "question": "How were the yearly seasonal flood exposure rasters aggregated to obtain the estimated population exposure per district?",
    "options": [
      {
        "answer": "Via zonal statistics (mean)",
        "is_correct": false
      },
      {
        "answer": "Via zonal statistics (sum)",
        "is_correct": true
      },
      {
        "answer": "Via zonal statistics (median)",
        "is_correct": false
      },
      {
        "answer": "Via zonal statistics (mode)",
        "is_correct": false
      }
    ]
  },
  {
    "question": "What percentiles were used to estimate the range of population exposed for the MAM 2024 season?",
    "options": [
      {
        "answer": "25th-50th percentile levels",
        "is_correct": false
      },
      {
        "answer": "50th-95th percentile levels",
        "is_correct": true
      },
      {
        "answer": "25th-75th percentile levels",
        "is_correct": false
      },
      {
        "answer": "10th-90th percentile levels",
        "is_correct": false
      }
    ]
  },
  {
    "question": "Who should be contacted for questions or feedback on the Somalia Flood Exposure Methodology Note?",
    "options": [
      {
        "answer": "Leonardo Milano, Team Lead for Data Science at leonardo.milano@un.org",
        "is_correct": true
      },
      {
        "answer": "Somalia ICCG and HCT",
        "is_correct": false
      },
      {
        "answer": "UNFPA Methodology team",
        "is_correct": false
      },
      {
        "answer": "ECMWF seasonal forecast team",
        "is_correct": false
      }
    ]
  }
]
"""

dict_data = json.loads(test_str)
print(dict_data)