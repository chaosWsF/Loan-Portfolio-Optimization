import numpy as np
import pandas as pd

threshold = 0.01 # threshold to drop merge columns

sums_df = pd.read_csv('transaction_types.csv')

# Extract columns that meet threshold to keep
keep_list = list(sums_df[sums_df['sum'] >= threshold ]['c'])

# Extract columns to pool together
misc_list = list(sums_df[sums_df['sum'] < threshold]['c'])


# Create query to pool misc columns
sql_combine = ' + '.join(misc_list)
sql_combine = '( ' + sql_combine + ' )'
keep_sql = ', '.join(keep_list)

DDL = 'create or replace table pd-ai-notebooks-rd.loan_loss_revisit.root_borrower_train_set_clean_reduced as\n'
columns_before = 'SELECT bus_ptnr_group, cal_day, naics_id, BRR, BEACON, dunning_level_code, dunning_level, days_in_arrears, impaired, Oustanding_principle_on_posting_date, percentage_rate, transactions, abs_transactions, transactions_db, transactions_cr, n_transactions, SUB_SYSTEM_DP, SUB_SYSTEM_FD, SUB_SYSTEM_IN, SUB_SYSTEM_LN, SUB_SYSTEM_RF, SUB_SYSTEM_RP, SUB_SYSTEM_SP, SUB_SYSTEM_TF,'
keep_cols = keep_sql + ','
pooled_cols = sql_combine + ' transaction_type_misc,'
columns_after = 'mth_since_brr_update, defaults_3_months, defaults_6_months, defaults_9_months, defaults_12_months, has_loan'
from_clause = 'FROM `pd-ai-notebooks-rd.loan_loss_revisit.root_borrower_train_set_clean`'

query = '\n'.join([DDL, columns_before, keep_cols, pooled_cols, columns_after, from_clause])


# Write query to text file
with open('merge_trxn_query.sql','w') as f:
    f.write(query)