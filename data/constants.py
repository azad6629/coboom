import os
from pathlib import Path

DATA_BASE_DIR = Path("/workspace/DATASETS/XRAY_datasets/")


# #############################################
# NIH constants
# #############################################
NIH_CXR_DATA_DIR       = DATA_BASE_DIR / "NIH_Chest-Xray-14"
NIH_DATA_ENTRY_CSV     = NIH_CXR_DATA_DIR / "Data_Entry_2017.csv"
NIH_ORIGINAL_TRAIN_TXT = NIH_CXR_DATA_DIR / "train_val_list.txt"
NIH_ORIGINAL_TEST_TXT  = NIH_CXR_DATA_DIR / "test_list.txt"
NIH_TRAIN_CSV          = NIH_CXR_DATA_DIR / "train.csv"
NIH_TEST_CSV           = NIH_CXR_DATA_DIR / "test.csv"

# #############################################
# NIH constants from PCRL/CAiD/DiRA Distribution
# #############################################
CXR14_DATA_DIR       = DATA_BASE_DIR / "NIH_Chest-Xray-14/images"
NIH_TRAIN_TXT          = "./Xray14_train_official.txt"
NIH_VAL_TXT            = "./Xray14_val_official.txt"
NIH_TEST_TXT           = "./Xray14_test_official.txt"


NIH_PATH_COL = "Path"
NIH_TASKS = [
             "Atelectasis" ,
             "Cardiomegaly", 
             "Consolidation", 
             "Infiltration", 
             "Pneumothorax", 
             "Edema", 
             "Emphysema", 
             "Fibrosis", 
             "Effusion", 
             "Pneumonia", 
             "Pleural_Thickening", 
             "Nodule", 
             "Mass", 
             "Hernia",
             "No Finding"
            ]

Chex14_TASKS = [
             "Atelectasis" ,
             "Cardiomegaly", 
             "Consolidation", 
             "Infiltration", 
             "Pneumothorax", 
             "Edema", 
             "Emphysema", 
             "Fibrosis", 
             "Effusion", 
             "Pneumonia", 
             "Pleural_Thickening", 
             "Nodule", 
             "Mass", 
             "Hernia",
            ]


