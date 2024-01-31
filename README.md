# Credit Card Analysis

---

# Data Analysis Project: Exploratory Data Analysis and Clustering

This repository contains code and analysis for performing Exploratory Data Analysis (EDA) and clustering on a German credit dataset. The project aims to understand the dataset's characteristics, identify patterns, and apply clustering techniques to group similar instances.

## Table of Contents

- [Project Aim](#project-aim)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Clustering](#clustering)
- [Conclusion](#conclusion)

## Project Aim

The main objectives of this project are as follows:

1. Perform EDA and necessary data cleaning.
2. Apply one-hot encoding to categorical variables.
3. Visualize histograms of numerical features to identify skewness. Apply log transformation if needed.
4. Apply feature scaling to prepare the data for clustering.
5. Utilize the elbow method to determine the optimal number of clusters.
6. Visualize the chosen number of clusters using PCA.
7. Implement K-Fold Cross Validation with a selected classifier and report evaluation metrics.
8. Draw conclusions based on the analysis.

## Libraries Used

The following libraries were used for this project:

- pandas
- numpy
- seaborn
- matplotlib
- plotly
- scikit-learn

## Dataset

The dataset used for this project is the "German Dataset.csv," which contains information about credit applicants. It includes attributes like age, sex, job, housing, credit amount, duration, purpose, and risk. The dataset has both numerical and categorical features.

## Data Preprocessing

The data preprocessing steps included handling missing values in columns like "Saving accounts" and "Checking account," converting categorical variables into numerical format using one-hot encoding, and performing feature scaling to ensure comparable scales for clustering.

## Exploratory Data Analysis

EDA involved visualizing the distributions of numerical features such as age, credit amount, and duration using histograms. The analysis aimed to identify any patterns or trends in the data and determine if transformations were needed.

## Clustering

The clustering process began with determining the optimal number of clusters using the elbow method. The chosen number of clusters was visualized using Principal Component Analysis (PCA) for dimensionality reduction. A selected clustering algorithm was then applied to group similar instances.

## Conclusion

The project successfully explored the German credit dataset through EDA, performed clustering analysis, and reported findings. By implementing K-Fold Cross Validation with a chosen classifier, the project also assessed the performance of the clustering approach. The README file provides an overview of the steps taken, the tools used, and the insights gained from the analysis.

---
/////////////////

SAS CODE CONVERTED TO PYTHON S5

import pandas as pd

# Define constants
add_yr = 22

# Load data from CSV files
matched_los_reg_22 = pd.read_csv('matched.los_reg_22.csv')
matched_tpia_reg_22 = pd.read_csv('matched.tpia_reg_22.csv')
supp_los_supp_reg_22 = pd.read_csv('supp.los_supp_reg_22.csv')
supp_tpia_supp_reg_22 = pd.read_csv('supp.tpia_supp_reg_22.csv')

# Rename columns
matched_los_reg_22.rename(columns={'new_region_id': 'region_id'}, inplace=True)
matched_tpia_reg_22.rename(columns={'new_region_id': 'region_id'}, inplace=True)
supp_los_supp_reg_22.rename(columns={'new_region_id': 'region_id'}, inplace=True)
supp_tpia_supp_reg_22.rename(columns={'new_region_id': 'region_id'}, inplace=True)

# Compile function
def compile(yr, ind, report, id):
    # Load and filter data
    ed_facility_org = pd.read_csv(f'fac.ed_facility_org_{add_yr}.csv')
    ed_nacrs_flg = ed_facility_org[(ed_facility_org['nacrs_ed_flg'] == 1) & (ed_facility_org['corp_cnt'] == 1)]

    ind_report_com_trd_a = pd.read_csv(f'matched.{ind}.{report}.{yr}.csv')
    ind_report_cmp_a = pd.read_csv(f'com.{ind}.{report}.cmp_a.csv')
    ind_report_trend_b = pd.read_csv(f'trend.{ind}.{report}.trend_b.csv')

    # Left join dataframes
    merged_data = pd.merge(ind_report_com_trd_a, ind_report_cmp_a, on=id, how='left')
    merged_data = pd.merge(merged_data, ind_report_trend_b, on=id, how='left')

    # Sort data
    merged_data.sort_values(by=id, inplace=True)
    ed_nacrs_flg.sort_values(by=id, inplace=True)

    # Merge dataframes
    result_data = pd.merge_asof(merged_data, ed_nacrs_flg, on=id)

    # Filter rows
    result_data = result_data[~result_data.index.duplicated(keep='first')]

    # Save result to CSV
    result_data.to_csv(f'{ind}.{report}.com.trd.{yr}.csv', index=False)

# Call compile function
compile(add_yr, 'tpia', 'org', 'corp_id')
compile(add_yr, 'tpia', 'reg', 'region_id')
compile(add_yr, 'tpia', 'prov', 'province_id')

compile(add_yr, 'los', 'org', 'corp_id')
compile(add_yr, 'los', 'reg', 'region_id')
compile(add_yr, 'los', 'prov', 'province_id')

# Update previous fiscal year data function
def remove_prev_year_data(yr, ind_id):
    hsp_ind_organization_fact = pd.read_csv(f'hsp_last.hsp_ind_organization_fact{ind_id}.csv')
    hsp_ind_organization_fact = hsp_ind_organization_fact[hsp_ind_organization_fact['FISCAL_YEAR_WH_ID'] != yr - 5]
    hsp_ind_organization_fact.to_csv(f'hsp_last.hsp_ind_organization_fact{ind_id}.csv', index=False)

# Call remove_prev_year_data function
remove_prev_year_data(add_yr, 33)
remove_prev_year_data(add_yr, 34)

# Filter and save data
test1 = pd.read_csv('hsp_ind_organization_fact33.csv')
test1 = test1[test1['organization_id'].isin([1019, 10038, 5085, 5049])]
test1.to_csv('test1.csv', index=False)

test2 = pd.read_csv('hsp_ind_organization_fact34.csv')
test2 = test2[test2['organization_id'].isin([1048, 5160, 81118, 81170])]
test2.to_csv('test2.csv', index=False)

# Change corp_id as per README document
hsp_ind_organization_fact33 = pd.read_csv('hsp_ind_organization_fact33.csv')
hsp_ind_organization_fact34 = pd.read_csv('hsp_ind_organization_fact34.csv')

hsp_ind_organization_fact33['organization_id'].replace({5085: 81180, 5049: 81263}, inplace=True)
hsp_ind_organization_fact34['organization_id'].replace({81118: 81118}, inplace=True)  # You can add more replacements here

hsp_ind_organization_fact33.to_csv('hsp_ind_organization_fact33.csv', index=False)
hsp_ind_organization_fact34.to_csv('hsp_ind_organization_fact34.csv', index=False)

# Append new fiscal year data function
def append_new_year_data(yr, ind, ind_id, report, id):
    # Load and filter data
    ind_report_com_trd_a = pd.read_csv(f'{ind}.{report}.com.trd.{yr}.csv')
    ed_nacrs_flg = pd.read_csv(f'fac.ed_nacrs_flg_1_{yr}.csv')

    # SQL-like insert into statement
    ind_report_com_trd_a['FISCAL_YEAR_WH_ID'] = yr
    ind_report_com_trd_a['SEX_WH_ID'] = 3
    ind_report_com_trd_a['INDICATOR_SUPPRESSION_CODE'] = '007'
    ind_report_com_trd_a['INDICATOR_VALUE'] = ind_report_com_trd_a['percentile_90'].round(1)
    ind_report_com_trd_a['DATA_PERIOD_CODE'] = f'0{yr}'
    ind_report_com_trd_a['DATA_PERIOD_TYPE_CODE'] = 'FY'

    # Province both IMPROVEMENT_IND_CODE and COMPARE_IND_CODE must be 999
    if report == 'prov':
        ind_report_com_trd_a['IMPROVEMENT_IND_CODE'] = '999'
        ind_report_com_trd_a['COMPARE_IND_CODE'] = '999'
    # Region and Corp
    elif report in ['reg', 'org']:
        ind_report_com_trd_a['IMPROVEMENT_IND_CODE'] = ind_report_com_trd_a['IMPROVEMENT_IND_CODE'].fillna('999')
        ind_report_com_trd_a['COMPARE_IND_CODE'] = ind_report_com_trd_a['COMPARE_IND_CODE'].fillna('999')
    # Peer
    elif report == 'peer':
        ind_report_com_trd_a['IMPROVEMENT_IND_CODE'] = '999'
        ind_report_com_trd_a['COMPARE_IND_CODE'] = '999'

    # Merge dataframes
    merged_data = pd.merge_asof(ind_report_com_trd_a, ed_nacrs_flg, left_on=id, right_on=id)

    # Save result to CSV
    merged_data.to_csv(f'{ind}.{report}.com.trd.{yr}.csv', index=False)

# Call append_new_year_data function
append_new_year_data(add_yr, 'LOS', 33, 'nt', 'organization_id')
append_new_year_data(add_yr, 'TPIA', 34, 'nt', 'organization_id')

append_new_year_data(add_yr, 'LOS', 33, 'peer', 'Peer_group_ID')
append_new_year_data(add_yr, 'TPIA', 34, 'peer', 'Peer_group_ID')

append_new_year_data(add_yr, 'LOS', 33, 'prov', 'province_id')
append_new_year_data(add_yr, 'TPIA', 34, 'prov', 'province_id')

append_new_year_data(add_yr, 'LOS', 33, 'reg', 'region_id')
append_new_year_data(add_yr, 'TPIA', 34, 'reg', 'region_id')

append_new_year_data(add_yr, 'LOS', 33, 'org', 'corp_id')
append_new_year_data(add_yr, 'TPIA', 34, 'org', 'corp_id')

# Sort and save dataframes if needed
hsp_ind_organization_fact34 = pd.read_csv('hsp_ind_organization_fact34.csv')
hsp_ind_organization_fact34.sort_values(by=['organization_id', 'data_period_code'], inplace=True)
hsp_ind_organization_fact34.to_csv('hsp_ind_organization_fact34.csv', index=False)

hsp_ind_organization_fact33 = pd.read_csv('hsp_ind_organization_fact33.csv')
hsp_ind_organization_fact33.sort_values(by=['organization_id', 'data_period_code'], inplace=True)
hsp_ind_organization_fact33.to_csv('hsp_ind_organization_fact33.csv', index=False)

# Filter and save data if needed
test33 = pd.read_csv('hsp_ind_organization_fact33.csv')
test33 = test33[test33['organization_id'].isin([81096, 5049, 5085, 5035, 1048, 5065, 5041, 81118, 81124, 81263, 81180, 81170])]
test33.to_csv('test33.csv', index=False)

/////NEXT 

# Add dummy data if found less than five years function
def add_dummy(yr, ind_id):
    # Load data
    hsp_ind_organization_fact = pd.read_csv(f'hsp_ind_organization_fact{ind_id}.csv')

    # Count the number of fiscal years for each organization
    organization_counts = hsp_ind_organization_fact.groupby('organization_id')['FISCAL_YEAR_WH_ID'].count().reset_index()

    # Select organizations with less than five fiscal years
    less_than_five_yrs = organization_counts[organization_counts['FISCAL_YEAR_WH_ID'] < 5]

    # Select organizations with exactly four fiscal years
    four_yrs = organization_counts[organization_counts['FISCAL_YEAR_WH_ID'] == 4]

    # Select organizations with exactly one fiscal year
    new_yr = organization_counts[organization_counts['FISCAL_YEAR_WH_ID'] == 1]

    # Add new year dummy data
    new_year_data = pd.DataFrame({
        'ORGANIZATION_ID': four_yrs['organization_id'],
        'INDICATOR_CODE': f'0{ind_id}',
        'FISCAL_YEAR_WH_ID': yr,
        'SEX_WH_ID': 3,
        'INDICATOR_SUPPRESSION_CODE': '999',
        'TOP_PERFORMER_IND_CODE': '999',
        'IMPROVEMENT_IND_CODE': '999',
        'COMPARE_IND_CODE': '999',
        'INDICATOR_VALUE': None,
        'DATA_PERIOD_CODE': f'0{yr}',
        'DATA_PERIOD_TYPE_CODE': 'FY'
    })

    # Append new year dummy data
    hsp_ind_organization_fact = pd.concat([hsp_ind_organization_fact, new_year_data], ignore_index=True)

    # Add previous four years dummy data
    for i in range(1, 5):
        prev_year_data = pd.DataFrame({
            'ORGANIZATION_ID': new_yr['organization_id'],
            'INDICATOR_CODE': f'0{ind_id}',
            'FISCAL_YEAR_WH_ID': yr - i,
            'SEX_WH_ID': 3,
            'INDICATOR_SUPPRESSION_CODE': '999',
            'TOP_PERFORMER_IND_CODE': '999',
            'IMPROVEMENT_IND_CODE': '999',
            'COMPARE_IND_CODE': '999',
            'INDICATOR_VALUE': None,
            'DATA_PERIOD_CODE': f'0{yr - i}',
            'DATA_PERIOD_TYPE_CODE': 'FY'
        })

        # Append previous year dummy data
        hsp_ind_organization_fact = pd.concat([hsp_ind_organization_fact, prev_year_data], ignore_index=True)

    # Save the updated data
    hsp_ind_organization_fact.to_csv(f'hsp_ind_organization_fact{ind_id}.csv', index=False)

# Call add_dummy function
add_dummy(add_yr, 33)
add_dummy(add_yr, 34)


////NEXT


# Continue adding dummy data function
def add_dummy(yr, ind_id):
    # Load data
    hsp_ind_organization_fact = pd.read_csv(f'hsp_ind_organization_fact{ind_id}.csv')

    # Count the number of fiscal years for each organization
    organization_counts = hsp_ind_organization_fact.groupby('organization_id')['FISCAL_YEAR_WH_ID'].count().reset_index()

    # Select organizations with less than five fiscal years
    less_than_five_yrs = organization_counts[organization_counts['FISCAL_YEAR_WH_ID'] < 5]

    # Select organizations with exactly four fiscal years
    four_yrs = organization_counts[organization_counts['FISCAL_YEAR_WH_ID'] == 4]

    # Select organizations with exactly one fiscal year
    new_yr = organization_counts[organization_counts['FISCAL_YEAR_WH_ID'] == 1]

    # Add new year dummy data
    new_year_data = pd.DataFrame({
        'ORGANIZATION_ID': four_yrs['organization_id'],
        'INDICATOR_CODE': f'0{ind_id}',
        'FISCAL_YEAR_WH_ID': yr,
        'SEX_WH_ID': 3,
        'INDICATOR_SUPPRESSION_CODE': '999',
        'TOP_PERFORMER_IND_CODE': '999',
        'IMPROVEMENT_IND_CODE': '999',
        'COMPARE_IND_CODE': '999',
        'INDICATOR_VALUE': None,
        'DATA_PERIOD_CODE': f'0{yr}',
        'DATA_PERIOD_TYPE_CODE': 'FY'
    })

    # Append new year dummy data
    hsp_ind_organization_fact = pd.concat([hsp_ind_organization_fact, new_year_data], ignore_index=True)

    # Add previous four years dummy data
    for i in range(1, 5):
        prev_year_data = pd.DataFrame({
            'ORGANIZATION_ID': new_yr['organization_id'],
            'INDICATOR_CODE': f'0{ind_id}',
            'FISCAL_YEAR_WH_ID': yr - i,
            'SEX_WH_ID': 3,
            'INDICATOR_SUPPRESSION_CODE': '999',
            'TOP_PERFORMER_IND_CODE': '999',
            'IMPROVEMENT_IND_CODE': '999',
            'COMPARE_IND_CODE': '999',
            'INDICATOR_VALUE': None,
            'DATA_PERIOD_CODE': f'0{yr - i}',
            'DATA_PERIOD_TYPE_CODE': 'FY'
        })

        # Append previous year dummy data
        hsp_ind_organization_fact = pd.concat([hsp_ind_organization_fact, prev_year_data], ignore_index=True)

    # Save the updated data
    hsp_ind_organization_fact.to_csv(f'hsp_ind_organization_fact{ind_id}.csv', index=False)

# Call add_dummy function for ind_id=33 and ind_id=34
add_dummy(add_yr, 33)
add_dummy(add_yr, 34)


//// 

MIGHT USE BELOW OR ONE AFTER

# Continue with Python code for the provided SAS code

# Load HSP lookup table
hsp_organization_ext = pd.read_csv('lookup/hsp_organization_ext.csv')

# Remove organizations not present in HSP lookup table for ind_id=33
hsp_ind_organization_fact33 = hsp_ind_organization_fact33[hsp_ind_organization_fact33['organization_id'].isin(hsp_organization_ext['organization_id'])]

# Remove organizations not present in HSP lookup table for ind_id=34
hsp_ind_organization_fact34 = hsp_ind_organization_fact34[hsp_ind_organization_fact34['organization_id'].isin(hsp_organization_ext['organization_id'])]

# To see what is removed in the above step
organization_ids_removed = hsp_ind_organization_fact33[~hsp_ind_organization_fact33['organization_id'].isin(hsp_organization_ext['organization_id'])]['organization_id']

# Create reporting_corp DataFrame with distinct organization IDs
reporting_corp = pd.DataFrame({'organization_id': organization_ids_removed}).drop_duplicates()

# Save the updated data
hsp_ind_organization_fact33.to_csv('hsp_ind_organization_fact33.csv', index=False)
hsp_ind_organization_fact34.to_csv('hsp_ind_organization_fact34.csv', index=False)

# Now you have updated DataFrames hsp_ind_organization_fact33, hsp_ind_organization_fact34,
# and reporting_corp according to the provided SAS code.


////THIS ONE 

# Continue with Python code for the provided SAS code

# Create a list of organization IDs to remove (Python equivalent of the SAS code)
org_ids_to_remove = [1048, 81118, 81102, 7042, 7028, 7018, 7004, 81124, 1048, 81180, 81263]

# Filter the data to remove specific organization IDs
filtered_hsp_ind_org_fact33 = hsp_ind_organization_fact33[~hsp_ind_organization_fact33['organization_id'].isin(org_ids_to_remove)]
filtered_hsp_ind_org_fact34 = hsp_ind_organization_fact34[~hsp_ind_organization_fact34['organization_id'].isin(org_ids_to_remove)]

# Sort the filtered data
sorted_filtered_hsp_ind_org_fact33 = filtered_hsp_ind_org_fact33.sort_values(by=['organization_id', 'fiscal_year_wh_id'])
sorted_filtered_hsp_ind_org_fact34 = filtered_hsp_ind_org_fact34.sort_values(by=['organization_id', 'fiscal_year_wh_id'])

# Create DataFrames for the test data
test_data = sorted_filtered_hsp_ind_org_fact33[sorted_filtered_hsp_ind_org_fact33['organization_id'].isin(org_ids_to_remove)]
check33_data = sorted_filtered_hsp_ind_org_fact33.copy()
check34_data = sorted_filtered_hsp_ind_org_fact34.copy()

# Define a function to remove organizations not in the HSP lookup table
def remove_org_not_in_hsp_org_ext(yr, ind_id):
    # Get the organization IDs from the HSP lookup table
    org_ids_in_lookup = lookup_data['organization_id'].tolist()
    
    # Filter the data to keep only organizations in the lookup table
    filtered_hsp_ind_org_fact = globals()[f'sorted_filtered_hsp_ind_org_fact{ind_id}']
    filtered_hsp_ind_org_fact = filtered_hsp_ind_org_fact[filtered_hsp_ind_org_fact['organization_id'].isin(org_ids_in_lookup)]
    
    return filtered_hsp_ind_org_fact

# Call the function to remove organizations not in the HSP lookup table
sorted_filtered_hsp_ind_org_fact33 = remove_org_not_in_hsp_org_ext(add_yr, 33)
sorted_filtered_hsp_ind_org_fact34 = remove_org_not_in_hsp_org_ext(add_yr, 34)

# Create DataFrames for test1 and test2
test1_data = lookup_data[['organization_id', 'organization_name_e_desc']]
test2_data = check34_data[['organization_id']]

# Define a function to compare DataFrames and find removed organizations
def find_removed_organizations(df1, df2):
    removed_organizations = df1[~df1['organization_id'].isin(df2['organization_id'])]
    return removed_organizations

# Find removed organizations
removed_organizations_data = find_removed_organizations(test1_data, test2_data)

# Create DataFrames for reporting_org and reporting_corp
reporting_org_data = pd.concat([test1_data[test1_data['organization_id'].isin(test2_data['organization_id'])], removed_organizations_data])
reporting_corp_data = pd.DataFrame({'organization_id': reporting_org_data['organization_id'].unique()})

# Print the resulting DataFrames (you can save them to files if needed)
print(reporting_org_data)
print(reporting_corp_data)

/..........>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
CODEESSSS TO CORRECT ON G

Correct this error on the code below: Error is 
AttributeError: 'numpy.float64' object has no attribute 'rename'

# Constant values for org, reg
constant_org_regLOS = {
    "INDICATOR_CODE": "033",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "DATA_PERIOD_CODE": "033",
    "DATA_PERIOD_TYPE_CODE": 'FY'
}
constant_org_regTPIA = {
    "INDICATOR_CODE": "034",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "DATA_PERIOD_CODE": "034",
    "DATA_PERIOD_TYPE_CODE": 'FY'
}
# Function to prepare DataFrame
def prepare_org_regLOS(df, id_col):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_org_regLOS.items():
        df[col] = value
    df = df.reindex(columns=hsp_ind_organization_fact_los.columns)
    return df

# Prepare data
los_org_prepared = prepare_org_regLOS(los_org_com_trd, 'CORP_ID')
los_reg_prepared = prepare_org_regLOS(los_reg_com_trd, 'REGION_ID')

# Function to prepare DataFrame
def prepare_org_regTPIA(df, id_col):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_org_regTPIA.items():
        df[col] = value
    df = df.reindex(columns=hsp_ind_organization_fact_tpia.columns)
    return df
tpia_org_prepared = prepare_org_regTPIA(tpia_org_com_trd, 'CORP_ID')
tpia_reg_prepared = prepare_org_regTPIA(tpia_reg_com_trd, 'REGION_ID')

constant_los = {
    "INDICATOR_CODE": "033",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "IMPROVEMENT_IND_CODE": '999',
    "COMPARE_IND_CODE": '999',
    "DATA_PERIOD_CODE": "033",
    "DATA_PERIOD_TYPE_CODE": 'FY'
}
constant_tpia = {
    "INDICATOR_CODE": "034",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "IMPROVEMENT_IND_CODE": '999',
    "COMPARE_IND_CODE": '999',
    "DATA_PERIOD_CODE": "034",
    "DATA_PERIOD_TYPE_CODE": 'FY'
}

def prepare_los(df, id_col):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_los.items():
        df[col] = value
    df = df.reindex(columns=hsp_ind_organization_fact_los.columns)
    return df

los_prov_prepared = prepare_los(los_prov, 'PROVINCE_ID')
los_peer_prepared = prepare_los(los_peer, 'peer_id')
los_nat_prepared = prepare_los(LOS_nt, 'NATIONAL_ID')

def prepare_tpia(df, id_col):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_tpia.items():
        df[col] = value
    df = df.reindex(columns=hsp_ind_organization_fact_tpia.columns)
    return df
tpia_prov_prepared = prepare_tpia(tpia_prov, 'PROVINCE_ID')
tpia_peer_prepared =prepare_tpia(tpia_peer, 'peer_id')
tpia_nat_prepared =prepare_tpia(TPIA_nt, 'NATIONAL_ID')

# Concatenate all DataFrames
hsp_ind_organization_fact_los_final_a = pd.concat([hsp_ind_organization_fact_los, los_org_prepared,los_reg_prepared, los_prov_prepared, los_peer_prepared, los_nat_prepared ], ignore_index=True)
hsp_ind_organization_fact_los_final=hsp_ind_organization_fact_los_final_a.sort_values(['ORGANIZATION_ID','FISCAL_YEAR_WH_ID'])
display(hsp_ind_organization_fact_los_final)

hsp_ind_organization_fact_tpia_final_a = pd.concat([hsp_ind_organization_fact_tpia, tpia_org_prepared,tpia_reg_prepared, tpia_prov_prepared, tpia_peer_prepared, tpia_nat_prepared ], ignore_index=True)
hsp_ind_organization_fact_tpia_final=hsp_ind_organization_fact_tpia_final_a.sort_values(['ORGANIZATION_ID','FISCAL_YEAR_WH_ID'])
display(hsp_ind_organization_fact_tpia_final)


corect codr 



import pandas as pd  # Import the Pandas library

# Constant values for org, reg
constant_org_regLOS = {
    "INDICATOR_CODE": "033",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "DATA_PERIOD_CODE": "033",
    "DATA_PERIOD_TYPE_CODE": 'FY'
}
constant_org_regTPIA = {
    "INDICATOR_CODE": "034",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "DATA_PERIOD_CODE": "034",
    "DATA_PERIOD_TYPE_CODE": 'FY'
}

# Function to prepare DataFrame
def prepare_org_reg(df, id_col, constant_dict, hsp_ind_organization_fact_columns):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_dict.items():
        df[col] = value
    df = df.reindex(columns=hsp_ind_organization_fact_columns)
    return df

# Prepare data for LOS and TPIA
def prepare_los(df, id_col):
    return prepare_org_reg(df, id_col, constant_org_regLOS, hsp_ind_organization_fact_los.columns)

def prepare_tpia(df, id_col):
    return prepare_org_reg(df, id_col, constant_org_regTPIA, hsp_ind_organization_fact_tpia.columns)

# Concatenate all DataFrames for LOS and TPIA
hsp_ind_organization_fact_los_final_a = pd.concat([hsp_ind_organization_fact_los, prepare_los(los_org_com_trd, 'CORP_ID'), prepare_los(los_reg_com_trd, 'REGION_ID'), prepare_los(los_prov, 'PROVINCE_ID'), prepare_los(los_peer, 'peer_id'), prepare_los(LOS_nt, 'NATIONAL_ID')], ignore_index=True)
hsp_ind_organization_fact_los_final = hsp_ind_organization_fact_los_final_a.sort_values(['ORGANIZATION_ID', 'FISCAL_YEAR_WH_ID'])

hsp_ind_organization_fact_tpia_final_a = pd.concat([hsp_ind_organization_fact_tpia, prepare_tpia(tpia_org_com_trd, 'CORP_ID'), prepare_tpia(tpia_reg_com_trd, 'REGION_ID'), prepare_tpia(tpia_prov, 'PROVINCE_ID'), prepare_tpia(tpia_peer, 'peer_id'), prepare_tpia(TPIA_nt, 'NATIONAL_ID')], ignore_index=True)
hsp_ind_organization_fact_tpia_final = hsp_ind_organization_fact_tpia_final_a.sort_values(['ORGANIZATION_ID', 'FISCAL_YEAR_WH_ID'])

# Display the final DataFrames
display(hsp_ind_organization_fact_los_final)
display(hsp_ind_organization_fact_tpia_final)

This code should work correctly if your input DataFrames (los_org_com_trd, los_reg_com_trd, tpia_org_com_trd, tpia_reg_com_trd, los_prov, los_peer, LOS_nt, tpia_prov, tpia_peer, TPIA_nt) are Pandas DataFrames and all the required columns and variables are defined correctly.


ok rwbs code redefined 

import pandas as pd

# Constants for org, reg
constants = {
    "org_reg_LOS": {
        "INDICATOR_CODE": "033",
        "FISCAL_YEAR_WH_ID": 22,
        "SEX_WH_ID": 3,
        "INDICATOR_SUPPRESSION_CODE": '007',
        "TOP_PERFORMER_IND_CODE": '999',
        "DATA_PERIOD_CODE": "033",
        "DATA_PERIOD_TYPE_CODE": 'FY'
    },
    "org_reg_TPIA": {
        "INDICATOR_CODE": "034",
        "FISCAL_YEAR_WH_ID": 22,
        "SEX_WH_ID": 3,
        "INDICATOR_SUPPRESSION_CODE": '007',
        "TOP_PERFORMER_IND_CODE": '999',
        "DATA_PERIOD_CODE": "034",
        "DATA_PERIOD_TYPE_CODE": 'FY'
    }
}

# Function to prepare DataFrame
def prepare_df(df, id_col, constant_dict, columns):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_dict.items():
        df[col] = value
    df = df.reindex(columns=columns)
    return df

# Prepare data for LOS and TPIA
def prepare_los(df, id_col):
    return prepare_df(df, id_col, constants["org_reg_LOS"], hsp_ind_organization_fact_los.columns)

def prepare_tpia(df, id_col):
    return prepare_df(df, id_col, constants["org_reg_TPIA"], hsp_ind_organization_fact_tpia.columns)

# Concatenate all DataFrames for LOS and TPIA
def concat_dataframes(base_df, *dfs):
    return pd.concat([base_df] + list(dfs), ignore_index=True).sort_values(['ORGANIZATION_ID', 'FISCAL_YEAR_WH_ID'])

# Prepare and concatenate LOS DataFrames
hsp_ind_organization_fact_los_final = concat_dataframes(
    hsp_ind_organization_fact_los,
    prepare_los(los_org_com_trd, 'CORP_ID'),
    prepare_los(los_reg_com_trd, 'REGION_ID'),
    prepare_los(los_prov, 'PROVINCE_ID'),
    prepare_los(los_peer, 'peer_id'),
    prepare_los(LOS_nt, 'NATIONAL_ID')
)

# Prepare and concatenate TPIA DataFrames
hsp_ind_organization_fact_tpia_final = concat_dataframes(
    hsp_ind_organization_fact_tpia,
    prepare_tpia(tpia_org_com_trd, 'CORP_ID'),
    prepare_tpia(tpia_reg_com_trd, 'REGION_ID'),
    prepare_tpia(tpia_prov, 'PROVINCE_ID'),
    prepare_tpia(tpia_peer, 'peer_id'),
    prepare_tpia(TPIA_nt, 'NATIONAL_ID')
)

# Display the final DataFrames
display(hsp_ind_organization_fact_los_final)
display(hsp_ind_organization_fact_tpia_final)


anothr is 
import pandas as pd

# Constants for org, reg
constants = {
    "INDICATOR_CODE": "033",
    "FISCAL_YEAR_WH_ID": 22,
    "SEX_WH_ID": 3,
    "INDICATOR_SUPPRESSION_CODE": '007',
    "TOP_PERFORMER_IND_CODE": '999',
    "DATA_PERIOD_TYPE_CODE": 'FY'
}

# Define a dictionary of DataFrames and their associated id_col
dataframes = {
    "los_org": {"df": los_org_com_trd, "id_col": 'CORP_ID'},
    "los_reg": {"df": los_reg_com_trd, "id_col": 'REGION_ID'},
    "los_prov": {"df": los_prov, "id_col": 'PROVINCE_ID'},
    "los_peer": {"df": los_peer, "id_col": 'peer_id'},
    "los_nat": {"df": LOS_nt, "id_col": 'NATIONAL_ID'},
    "tpia_org": {"df": tpia_org_com_trd, "id_col": 'CORP_ID'},
    "tpia_reg": {"df": tpia_reg_com_trd, "id_col": 'REGION_ID'},
    "tpia_prov": {"df": tpia_prov, "id_col": 'PROVINCE_ID'},
    "tpia_peer": {"df": tpia_peer, "id_col": 'peer_id'},
    "tpia_nat": {"df": TPIA_nt, "id_col": 'NATIONAL_ID'},
}

# Function to prepare DataFrame
def prepare_df(df, id_col, constant_dict, columns):
    df = df.rename(columns={id_col: 'ORGANIZATION_ID', 'PERCENTILE_90': 'INDICATOR_VALUE'})
    for col, value in constant_dict.items():
        df[col] = value
    df = df.reindex(columns=columns)
    return df

# Initialize empty DataFrames
final_los_df = hsp_ind_organization_fact_los.copy()
final_tpia_df = hsp_ind_organization_fact_tpia.copy()

# Prepare and concatenate DataFrames for LOS and TPIA
for key, value in dataframes.items():
    df = value["df"]
    id_col = value["id_col"]
    constant_dict = constants.copy()
    constant_dict["DATA_PERIOD_CODE"] = key  # Customize DATA_PERIOD_CODE
    prepared_df = prepare_df(df, id_col, constant_dict, final_los_df.columns)
    final_los_df = pd.concat([final_los_df, prepared_df], ignore_index=True).sort_values(['ORGANIZATION_ID', 'FISCAL_YEAR_WH_ID'])
    final_tpia_df = pd.concat([final_tpia_df, prepared_df], ignore_index=True).sort_values(['ORGANIZATION_ID', 'FISCAL_YEAR_WH_ID'])

# Display the final DataFrames
display(final_los_df)
display(final_tpia_df)

