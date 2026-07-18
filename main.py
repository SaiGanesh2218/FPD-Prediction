from sqlalchemy import create_engine
username = "analyst"
password = "analyst123"
host = "49.207.11.47"
port = "6432"
database = "tenant_prabodh"
engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")
from sqlalchemy import inspect
inspector = inspect(engine)
tables = inspector.get_table_names()
import pandas as pd
import numpy as np
from datetime import timedelta
df_members           = pd.read_sql("SELECT * FROM members",                    engine)
df_savings_acc       = pd.read_sql("SELECT * FROM savings_accounts",           engine)
df_savings_txn       = pd.read_sql("SELECT * FROM savings_transactions",       engine)
df_groups            = pd.read_sql("SELECT * FROM groups",                     engine)
df_attendance        = pd.read_sql("SELECT * FROM attendance_records",         engine)
df_meeting           = pd.read_sql("SELECT * FROM meeting_records",            engine)
df_kyc               = pd.read_sql("SELECT * FROM kyc_submissions",            engine)
df_penalty           = pd.read_sql("SELECT * FROM penalty_applied",            engine)
df_guarantors        = pd.read_sql("SELECT * FROM loan_guarantors",            engine)
df_guarantor_exp     = pd.read_sql("SELECT * FROM guarantor_exposure_snapshots", engine)
df_branches          = pd.read_sql("SELECT * FROM branches",                   engine)
df_loan_products     = pd.read_sql("SELECT * FROM loan_products",              engine)
df_ageing            = pd.read_sql("SELECT * FROM loan_ageing_snapshot",       engine)
df_loans             = pd.read_sql("SELECT * FROM loan_accounts",              engine)
df_installments      = pd.read_sql("SELECT * FROM loan_installments",          engine)
print("All tables loaded!")
print("\nloan_accounts - status values:")
print(df_loans["status"].value_counts(dropna=False))

print("\nloan_installments - status values:")
print(df_installments["status"].value_counts(dropna=False))
df_first = df_installments[df_installments["installment_number"] == 1].copy()
df_first["due_date"] = (pd.to_datetime(df_first["due_date"], errors="coerce", utc=True).dt.tz_localize(None))

df_first["paid_at"] = (pd.to_datetime(df_first["paid_at"], errors="coerce", utc=True).dt.tz_localize(None))

df_first["fpd_deadline"] = (df_first["due_date"] + timedelta(days=5))

print(f"\nTotal first installment records : {len(df_first)}")
today = pd.Timestamp.now().tz_localize(None)

df_first = df_first[df_first["fpd_deadline"] < today].copy()

print(f"Loans with completed window : {len(df_first)}")
def assign_fpd(row):

    if pd.isna(row["paid_at"]):
        return 1

    elif row["paid_at"] > row["fpd_deadline"]:
        return 1

    else:
        return 0

df_first["FPD"] = df_first.apply(assign_fpd, axis=1)
print("\nFPD Distribution:")
print(df_first["FPD"].value_counts())

print("\nFPD Distribution (%):")
print(df_first["FPD"].value_counts(normalize=True).mul(100).round(2))
fpd_1 = df_first[df_first['FPD'] == 1]
fpd_0 = df_first[df_first['FPD'] == 0]

print(f"\nBefore sampling:")
print(f"FPD = 1 count : {len(fpd_1)}")
print(f"FPD = 0 count : {len(fpd_0)}")

sample_size = min(len(fpd_1), len(fpd_0))
fpd_1_sample = fpd_1.sample(n=len(fpd_1), random_state=42)
fpd_0_sample = fpd_0.sample(n=len(fpd_0), random_state=42)
df_sampled = pd.DataFrame(columns=[])
df_sampled = (pd.concat([fpd_1_sample, fpd_0_sample]).sample(frac=1, random_state=42).reset_index(drop=True))
print(df_sampled['FPD'].value_counts())
print(df_sampled['FPD'].value_counts(normalize=True).mul(100).round(2))
print(f"\nFinal sampled shape : {df_sampled.shape}")


df_sampled = df_first.copy().reset_index(drop=True)

print("\nUsing complete dataset (No Sampling)")
print(df_sampled['FPD'].value_counts())

print("\nFPD Distribution (%)")
print(df_sampled['FPD'].value_counts(normalize=True).mul(100).round(2))

print(f"\nDataset Shape : {df_sampled.shape}")
loan_cols = ['id','member_id','branch_id','loan_product_id','loan_cycle','loan_number','sanctioned_amount','interest_rate',
    'outstanding_principal','disbursed_at','status','purpose','num_installments','repayment_frequency','processing_fee','insurance_fee','created_at','updated_at','emi_calculation_method','outstanding_interest',
    'is_collateral_based','collateral_value','collateral_proof_urls','repayment_start_date','principal_per_installment','interest_per_installment','collection_frequency_cd']

df_sampled = df_sampled.merge(df_loans[loan_cols].rename(columns={'id': 'loan_id'}),on='loan_id', how='left')
import numpy as np
import pandas as pd
today = pd.Timestamp.today().normalize()

# Convert dates
df_members['date_of_birth'] = pd.to_datetime(df_members['date_of_birth'], errors='coerce', utc=True).dt.tz_localize(None)

df_members['created_at'] = pd.to_datetime(df_members['created_at'], errors='coerce', utc=True).dt.tz_localize(None)

df_loans['disbursed_at'] = pd.to_datetime(df_loans['disbursed_at'], errors='coerce', utc=True).dt.tz_localize(None)

# Basic member features
df_members['member_age'] = (today - df_members['date_of_birth']).dt.days
df_members['member_tenure'] = (today - df_members['created_at']).dt.days
df_members['is_aadhar_verified'] = (df_members['aadhar_number'].notna().astype(int))
member_features = df_members[['id','gender','occupation','caste_category','marital_status','member_age','member_tenure','is_aadhar_verified']].rename(columns={'id': 'member_id'})
df_sampled = df_sampled.merge(member_features,on='member_id',how='left')

if 'disbursed_at' not in df_sampled.columns:
    df_sampled = df_sampled.merge(df_loans[['id', 'disbursed_at']].rename(columns={'id': 'loan_id'}),on='loan_id',how='left')
df_sampled['disbursed_at'] = pd.to_datetime(df_sampled['disbursed_at'],errors='coerce',utc=True).dt.tz_localize(None)

# Sort loans
df_loans_sorted = (df_loans[df_loans['disbursed_at'].notna()].sort_values(['member_id', 'disbursed_at']).copy())

# Days to first loan
first_loan = (df_loans_sorted.groupby('member_id')['disbursed_at'].min().reset_index().rename(columns={'disbursed_at': 'first_loan_date'}))

member_reg = df_members[['id', 'created_at']].rename(columns={'id': 'member_id','created_at': 'member_registered_at'})
first_loan = first_loan.merge(member_reg,on='member_id',how='left')
first_loan['days_to_first_loan'] = (first_loan['first_loan_date']- first_loan['member_registered_at']).dt.days

# Prior loan count
prior_loan_counts = []
for _, row in df_sampled[['id', 'member_id', 'disbursed_at']].iterrows():
    count = df_loans_sorted[(df_loans_sorted['member_id'] == row['member_id']) &(df_loans_sorted['disbursed_at'] < row['disbursed_at'])].shape[0]
    prior_loan_counts.append({'id': row['id'],'prior_loan_count': count})
df_prior = pd.DataFrame(prior_loan_counts)
df_sampled.drop(columns=['prior_loan_count'],errors='ignore',inplace=True)
df_sampled = df_sampled.merge(df_prior,on='id',how='left')

# Average loan gap
def avg_gap(member):
    dates = (df_loans_sorted[df_loans_sorted['member_id'] == member]['disbursed_at'].sort_values().tolist())
    if len(dates) < 2:
        return np.nan
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    return np.mean(gaps)

member_gap = pd.DataFrame({'member_id': df_loans_sorted['member_id'].unique()})
member_gap['avg_loan_gap_days'] = (member_gap['member_id'].apply(avg_gap))
df_sampled.drop(columns=['avg_loan_gap_days'],errors='ignore',inplace=True)

df_sampled = df_sampled.merge(member_gap,on='member_id',how='left')

# Loan frequency
loan_freq = (df_loans_sorted.groupby('member_id').agg(total_loans=('id', 'count')).reset_index())
loan_freq = loan_freq.merge(member_reg,on='member_id',how='left')
loan_freq['tenure_years'] = ((today - loan_freq['member_registered_at']).dt.days / 365)
loan_freq['loan_frequency_per_year'] = (loan_freq['total_loans'] /loan_freq['tenure_years'].replace(0, np.nan))
df_sampled.drop(columns=['loan_frequency_per_year'],errors='ignore',inplace=True)
df_sampled = df_sampled.merge(loan_freq[['member_id', 'loan_frequency_per_year']],on='member_id',how='left')

# Days to first loan
df_sampled.drop(columns=['days_to_first_loan'],errors='ignore',inplace=True)
df_sampled = df_sampled.merge(first_loan[['member_id', 'days_to_first_loan']],on='member_id',how='left')

print("After member features :", df_sampled.shape)
df_sampled = df_sampled.drop(columns=['loan_frequency_per_year'], errors='ignore')
df_sampled = df_sampled.merge(loan_freq[['member_id', 'loan_frequency_per_year']],on='member_id',how='left')
print(f"After member features : {df_sampled.shape}")
# Remove previously merged loan product columns if they already exist
loan_cols = ['roi_tier_1_rate','min_loan_amount','max_loan_amount','thrift_eligibility_multiplier','repayment_due_day']
df_sampled = df_sampled.drop(columns=loan_cols, errors='ignore')
# Loan product features
loan_product_features = df_loan_products[['id','roi_tier_1_rate','min_loan_amount','max_loan_amount','thrift_eligibility_multiplier','repayment_due_day']].rename(columns={'id': 'loan_product_id'})
df_sampled = df_sampled.merge(loan_product_features,on='loan_product_id',how='left')
df_sampled['loan_to_max_ratio'] = (df_sampled['sanctioned_amount'] /df_sampled['max_loan_amount']).round(4)

# 1. Difference between sanctioned amount and maximum loan amount
df_sampled['sanctioned_vs_max_diff'] = (df_sampled['max_loan_amount'] -df_sampled['sanctioned_amount']).round(2)

# 2. Interest burden
df_sampled['interest_burden'] = (df_sampled['roi_tier_1_rate'] *df_sampled['sanctioned_amount']).round(2)

# 3. Collateral coverage ratio
df_sampled['collateral_coverage_ratio'] = (df_sampled['collateral_value'] /df_sampled['sanctioned_amount']).round(4)
df_sampled['collateral_coverage_ratio'] = (df_sampled['collateral_coverage_ratio'].fillna(0))

# Distribution
print("Collateral Coverage Ratio > 1:",
      (df_sampled['collateral_coverage_ratio'] > 1).sum())

print("Collateral Coverage Ratio = 1:",
      (df_sampled['collateral_coverage_ratio'] == 1).sum())

print("Collateral Coverage Ratio < 1:",
      (df_sampled['collateral_coverage_ratio'] < 1).sum())

print("\nPercentage Distribution")
print((df_sampled['collateral_coverage_ratio'].gt(1).value_counts(normalize=True) * 100))

print(f"\nAfter loan product features: {df_sampled.shape}")
df_savings_acc['opening_date'] = pd.to_datetime(df_savings_acc['opening_date'],errors='coerce')
savings_features = (df_savings_acc.groupby('member_id').agg( savings_balance=('balance', 'sum'), savings_tenure_months=('opening_date',lambda x: ((today - x.min()).days // 30)),savings_account_count=('id', 'count')).reset_index())
df_sampled = df_sampled.drop(columns=['savings_balance','savings_tenure_months','savings_account_count'],errors='ignore')
df_sampled = df_sampled.merge(savings_features,on='member_id',how='left')

# Savings to loan ratio
df_sampled['savings_to_loan_ratio'] = (df_sampled['savings_balance'] /df_sampled['sanctioned_amount']).round(4)

# Penalty count per member
penalty_count = df_penalty.groupby('member_id').agg(penalty_count=('id', 'count')).reset_index()
df_sampled = df_sampled.drop(columns=['penalty_count'], errors='ignore')
df_sampled = df_sampled.merge(penalty_count,on='member_id',how='left')
df_sampled['penalty_count'] = df_sampled['penalty_count'].fillna(0)

# Recent penalty flag (last 3 months)
df_penalty['cycle_month'] = pd.to_datetime(df_penalty['cycle_month'], errors='coerce')
cutoff_3m = today - timedelta(days=90)
recent_pen = (df_penalty[df_penalty['cycle_month'] >= cutoff_3m].groupby('member_id').agg(recent_penalty_flag=('id', 'count')).reset_index())
recent_pen['recent_penalty_flag'] = 1
df_sampled = df_sampled.drop(columns=['recent_penalty_flag'], errors='ignore')
df_sampled = df_sampled.merge(recent_pen[['member_id', 'recent_penalty_flag']],on='member_id',how='left')
df_sampled['recent_penalty_flag'] = (df_sampled['recent_penalty_flag'].fillna(0).astype(int))

# Average monthly savings deposit (last 6 months)
df_savings_txn['fin_st_dt'] = pd.to_datetime(df_savings_txn['fin_st_dt'], errors='coerce')
savings_cutoff = today - timedelta(days=180)
savings_txn_recent = df_savings_txn[df_savings_txn['fin_st_dt'] >= savings_cutoff].copy()
savings_acc_map = df_savings_acc[['id', 'member_id']].rename(columns={'id': 'savings_account_id'})
savings_txn_recent = savings_txn_recent.merge(savings_acc_map, on='savings_account_id', how='left')
avg_deposit = savings_txn_recent.groupby('member_id').agg(avg_monthly_deposit = ('sav_cr', 'mean')).reset_index()
df_sampled = df_sampled.merge(avg_deposit, on='member_id', how='left')

# Map savings transactions to members
df_savings_txn_full = df_savings_txn.merge(savings_acc_map, on='savings_account_id', how='left')
df_savings_txn_full['fin_st_dt'] = pd.to_datetime(df_savings_txn_full['fin_st_dt'], errors='coerce')
df_savings_txn_full['txn_month'] = df_savings_txn_full['fin_st_dt'].dt.to_period('M')

# 1. Deposit consistency — months with at least one deposit before loan sanction
deposit_months = (df_savings_txn_full[df_savings_txn_full['sav_cr'] > 0].groupby(['member_id', 'txn_month']).agg(has_deposit=('sav_cr', 'sum')).reset_index().groupby('member_id').agg(deposit_consistency_months=('has_deposit', 'count')).reset_index())
df_sampled = df_sampled.merge(deposit_months, on='member_id', how='left')
df_sampled['deposit_consistency_months'] = df_sampled['deposit_consistency_months'].fillna(0)

# 2. Days since last savings transaction before loan sanction
last_txn = (df_savings_txn_full.groupby('member_id')['fin_st_dt'].max().reset_index().rename(columns={'fin_st_dt': 'last_savings_txn_date'}))
df_sampled = df_sampled.merge(last_txn, on='member_id', how='left')
df_sampled['days_since_last_savings_txn'] = (today - df_sampled['last_savings_txn_date']).dt.days

# 3. Deposit growth trend
cutoff_3m_savings = today - timedelta(days=90)
recent_dep = (df_savings_txn_full[(df_savings_txn_full['fin_st_dt'] >= cutoff_3m_savings) &(df_savings_txn_full['sav_cr'] > 0)].groupby('member_id').agg(avg_recent_deposit=('sav_cr', 'mean')).reset_index())
older_dep = (df_savings_txn_full[(df_savings_txn_full['fin_st_dt'] < cutoff_3m_savings) &(df_savings_txn_full['sav_cr'] > 0)].groupby('member_id').agg(avg_older_deposit=('sav_cr', 'mean')).reset_index())
dep_growth = recent_dep.merge(older_dep, on='member_id', how='outer')
dep_growth['deposit_growth_trend'] = (dep_growth['avg_recent_deposit'] - dep_growth['avg_older_deposit']).round(2)

df_sampled = df_sampled.merge(dep_growth[['member_id', 'deposit_growth_trend']], on='member_id', how='left')

# 4. Deposit-to-withdrawal ratio
dep_wd = (df_savings_txn_full.groupby('member_id').agg(total_deposits    = ('sav_cr', 'sum'),total_withdrawals = ('sav_dr', 'sum')).reset_index())
dep_wd['deposit_to_withdrawal_ratio'] = (dep_wd['total_deposits'] /dep_wd['total_withdrawals'].replace(0, np.nan)).round(4)
df_sampled = df_sampled.merge(dep_wd[['member_id', 'deposit_to_withdrawal_ratio']], on='member_id', how='left')

# 5. Months with no savings activity before loan sanction
all_months_count = (df_savings_txn_full.groupby(['member_id', 'txn_month']).size().reset_index().groupby('member_id').agg(active_months=('txn_month', 'count')).reset_index())
savings_tenure_months_df = df_savings_acc.groupby('member_id').agg(sav_tenure_months=('opening_date', lambda x: ((today - x.min()).days // 30))).reset_index()
inactive_months = all_months_count.merge(savings_tenure_months_df, on='member_id', how='left')
inactive_months['inactive_savings_months'] = (inactive_months['sav_tenure_months'] - inactive_months['active_months']).clip(lower=0)
df_sampled = df_sampled.merge(inactive_months[['member_id', 'inactive_savings_months']], on='member_id', how='left')
print(f"After savings features : {df_sampled.shape}")
print(df_sampled.columns.tolist())

required = ['member_id', 'branch_id']
for col in required:
    print(col, col in df_sampled.columns)
member_group = (df_members[['id', 'group_id']].rename(columns={'id': 'member_id'}))
df_sampled = df_sampled.merge(member_group,on='member_id',how='left')
# 1. Attendance in last 5 meetings
df_meeting_sorted = df_meeting.sort_values(['group_id', 'meeting_date'])
last5_meeting_ids = (df_meeting_sorted.groupby('group_id').tail(5)['id'].tolist())
last5_att = (df_attendance[df_attendance['meeting_id'].isin(last5_meeting_ids)].groupby('member_id').agg(last5_meetings_attendance=('present', 'sum')).reset_index())
df_sampled.drop(columns=['last5_meetings_attendance'], errors='ignore', inplace=True)
df_sampled = df_sampled.merge(last5_att, on='member_id', how='left')
df_sampled['last5_meetings_attendance'] = (df_sampled['last5_meetings_attendance'].fillna(0).astype(int))

# 2. Maximum consecutive meetings missed
def max_consecutive_missed(member_id):
    records = (df_attendance[df_attendance['member_id'] == member_id].merge(df_meeting[['id', 'meeting_date']],left_on='meeting_id',right_on='id').sort_values('meeting_date')['present'].tolist())
    max_miss = 0
    curr = 0
    for p in records:
        if p == 0:
            curr += 1
            max_miss = max(max_miss, curr)
        else:
            curr = 0
    return max_miss


consec_miss = pd.DataFrame({'member_id': df_sampled['member_id'].drop_duplicates()})
consec_miss['max_consecutive_missed'] = (consec_miss['member_id'].apply(max_consecutive_missed))
df_sampled.drop(columns=['max_consecutive_missed'], errors='ignore', inplace=True)
df_sampled = df_sampled.merge(consec_miss, on='member_id', how='left')

# 3. Group default rate
loans_with_npa = (df_ageing[df_ageing['npa_classification'] != 'standard'][['member_id']].drop_duplicates())
loans_with_npa['ever_defaulted'] = 1
members_with_group = (df_members[['id', 'group_id']].rename(columns={'id': 'member_id'}))
group_defaults = members_with_group.merge(loans_with_npa,on='member_id',how='left')
group_defaults['ever_defaulted'] = (group_defaults['ever_defaulted'].fillna(0))
group_default_rate = (group_defaults.groupby('group_id').agg(group_default_rate=('ever_defaulted', 'mean')).reset_index())
df_sampled.drop(columns=['group_default_rate'], errors='ignore', inplace=True)
df_sampled = df_sampled.merge(group_default_rate,on='group_id',how='left')

# 4. Group average repayment rate
df_installments['due_date'] = pd.to_datetime(df_installments['due_date'],errors='coerce',utc=True).dt.tz_localize(None)
df_installments['paid_at'] = pd.to_datetime(df_installments['paid_at'],errors='coerce',utc=True).dt.tz_localize(None)
df_installments['paid_on_time'] = ((df_installments['paid_at'].notna()) &(df_installments['paid_at'] <= df_installments['due_date'])).astype(int)
member_repayment = (df_installments[df_installments['paid_at'].notna()].groupby('loan_id').agg(ontime_rate=('paid_on_time', 'mean')).reset_index())
member_repayment = member_repayment.merge(df_loans[['id', 'member_id']].rename(columns={'id': 'loan_id'}),on='loan_id',how='left')
member_avg_repayment = (member_repayment.groupby('member_id').agg(member_ontime_rate=('ontime_rate', 'mean')).reset_index())
group_repayment = (member_avg_repayment.merge(members_with_group, on='member_id', how='left').groupby('group_id').agg(group_avg_repayment_rate=('member_ontime_rate', 'mean')).reset_index())
df_sampled.drop(columns=['group_avg_repayment_rate'], errors='ignore', inplace=True)

df_sampled = df_sampled.merge(group_repayment,on='group_id',how='left')

# 5. Group stability ratio
total_members_grp = (df_members.groupby('group_id').agg(total_members_ever=('id', 'count')).reset_index())
active_members_grp = (df_members[df_members['status'].str.lower() == 'active'].groupby('group_id').agg(active_members_now=('id', 'count')).reset_index())
group_stability = total_members_grp.merge(active_members_grp,on='group_id',how='left')
group_stability['group_stability_ratio'] = (group_stability['active_members_now']/ group_stability['total_members_ever'])
df_sampled.drop(columns=['group_stability_ratio'], errors='ignore', inplace=True)
df_sampled = df_sampled.merge(group_stability[['group_id', 'group_stability_ratio']],on='group_id',how='left')

print(f"After group features : {df_sampled.shape}")
df_kyc['created_at'] = pd.to_datetime(df_kyc['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
df_kyc['reviewed_at'] = pd.to_datetime(df_kyc['reviewed_at'], errors='coerce', utc=True).dt.tz_localize(None)
df_kyc['kyc_days_to_review'] = (df_kyc['reviewed_at'] - df_kyc['created_at']).dt.days
kyc_latest = (df_kyc.sort_values('created_at').groupby('member_id').last().reset_index())
kyc_attempt = (df_kyc.groupby('member_id').agg(kyc_attempt_count=('id', 'count')).reset_index())
kyc_features = (kyc_latest[['member_id', 'status', 'kyc_days_to_review', 'created_at', 'reviewed_at']].rename(columns={'status': 'kyc_status','created_at': 'kyc_submitted_at','reviewed_at': 'kyc_approved_at'}).merge(kyc_attempt, on='member_id', how='left'))

# 1. Time from member registration to KYC approval
member_registration = (df_members[['id', 'created_at']].rename(columns={'id': 'member_id','created_at': 'member_registered_at'}))
member_registration['member_registered_at'] = pd.to_datetime(member_registration['member_registered_at'],errors='coerce',utc=True).dt.tz_localize(None)
kyc_features = kyc_features.merge(member_registration,on='member_id',how='left')
kyc_features['days_registration_to_kyc_approval'] = (kyc_features['kyc_approved_at'] -kyc_features['member_registered_at']).dt.days

# 2. Number of rejected KYC submissions
rejected_kyc = (df_kyc[df_kyc['status'].str.lower() == 'rejected'].groupby('member_id').agg(kyc_rejected_count=('id', 'count')).reset_index())
kyc_features = kyc_features.merge(rejected_kyc,on='member_id',how='left')
kyc_features['kyc_rejected_count'] = (kyc_features['kyc_rejected_count'].fillna(0).astype(int))

# Merge KYC features
df_sampled = df_sampled.merge(kyc_features[['member_id','kyc_status','kyc_days_to_review','kyc_attempt_count','kyc_approved_at','days_registration_to_kyc_approval','kyc_rejected_count']], on='member_id',how='left')
df_sampled['kyc_approved_at'] = pd.to_datetime(df_sampled['kyc_approved_at'],errors='coerce',utc=True).dt.tz_localize(None)
df_sampled['disbursed_at'] = pd.to_datetime(df_sampled['disbursed_at'],errors='coerce',utc=True).dt.tz_localize(None)

# 3. Days between KYC approval and loan sanction
df_sampled['days_kyc_to_loan_sanction'] = (df_sampled['disbursed_at'] -df_sampled['kyc_approved_at']).dt.days

# 4. KYC completed before loan sanction
df_sampled['kyc_completed_before_loan'] = ((df_sampled['kyc_approved_at'] < df_sampled['disbursed_at']) &df_sampled['kyc_approved_at'].notna() &df_sampled['disbursed_at'].notna()).astype(int)

print(f"After KYC features : {df_sampled.shape}")
df_sampled = df_sampled.drop(columns=['loan_account_id','guarantor_count','avg_guarantor_burden_tier','total_contingent_liability','guarantors_with_overdue','has_circular_guarantee','avg_active_guarantees_per_guar'],errors='ignore')
guarantor_count = (df_guarantors.groupby('loan_account_id').agg(guarantor_count=('id', 'count')).reset_index())
df_sampled = df_sampled.merge(guarantor_count,left_on='id',right_on='loan_account_id',how='left')
df_sampled = df_sampled.drop(columns='loan_account_id', errors='ignore')
df_sampled['guarantor_count'] = df_sampled['guarantor_count'].fillna(0)
guarantor_exp_latest = (df_guarantor_exp.sort_values('snapshot_date').groupby('guarantor_member_id').last().reset_index())
loan_guarantor_map = df_guarantors.merge(guarantor_exp_latest[['guarantor_member_id','active_guarantee_count','borrowers_overdue_count','circular_guarantee_flag','guarantor_burden_tier','contingent_principal']],on='guarantor_member_id',how='left')
tier_map = {'LOW': 1,'MEDIUM': 2,'HIGH': 3,'VERY_HIGH': 4}
loan_guarantor_map['guarantor_burden_tier_num'] = (loan_guarantor_map['guarantor_burden_tier'].astype(str).str.upper().map(tier_map))
guarantor_agg = (loan_guarantor_map.groupby('loan_account_id').agg(
    # 1. Average guarantor burden tier
    avg_guarantor_burden_tier=('guarantor_burden_tier_num', 'mean'),
    # 2. Total contingent liability
    total_contingent_liability=('contingent_principal', 'sum'),
    # 3. Count of guarantors with overdue borrowers
    guarantors_with_overdue=('borrowers_overdue_count',lambda x: (x.fillna(0) > 0).sum()),
    # 4. Circular guarantee flag
    has_circular_guarantee=('circular_guarantee_flag', 'max'),
    # 5. Average number of active guarantees held
    avg_active_guarantees_per_guar=('active_guarantee_count', 'mean')).reset_index())
df_sampled = df_sampled.merge(guarantor_agg,left_on='id',right_on='loan_account_id',how='left')
df_sampled = df_sampled.drop(columns='loan_account_id', errors='ignore')

print(f"After guarantor features : {df_sampled.shape}")
branch_base = df_branches[['id', 'branch_type']].rename(columns={'id': 'branch_id'})
df_sampled  = df_sampled.merge(branch_base, on='branch_id', how='left')

# 1. Historical branch NPA rate
branch_npa = df_ageing.groupby('branch_id').agg(branch_npa_rate=('npa_classification', lambda x: (x != 'standard').mean())).reset_index()

# 2. Average sanctioned loan amount per branch
branch_avg_amount = df_loans.groupby('branch_id').agg(branch_avg_sanctioned_amount=('sanctioned_amount', 'mean')).reset_index()

# 3. Average repayment performance of loans from each branch
branch_loan_repayment = (member_repayment.merge(df_loans[['id', 'branch_id']].rename(columns={'id': 'loan_id'}),on='loan_id', how='left').groupby('branch_id').agg(branch_avg_repayment_rate=('ontime_rate', 'mean')).reset_index())

# 4. Branch loan approval rate
df_loans['is_approved'] = df_loans['disbursed_at'].notnull().astype(int)
branch_approval = df_loans.groupby('branch_id').agg(branch_approval_rate=('is_approved', 'mean')).reset_index()

# 5. Branch portfolio composition
branch_portfolio = df_loans.groupby('branch_id').agg(branch_product_diversity=('loan_product_id', 'nunique')).reset_index()

# Merge all branch features
df_sampled = (df_sampled.merge(branch_npa,on='branch_id', how='left').merge(branch_avg_amount,on='branch_id', how='left').merge(branch_loan_repayment,on='branch_id', how='left').merge(branch_approval,on='branch_id', how='left').merge(branch_portfolio,on='branch_id', how='left'))

print(f"After branch features : {df_sampled.shape}")
ageing_features = df_ageing.groupby('member_id').agg(ever_npa = ('npa_classification', lambda x: int((x != 'standard').any())),max_overdue_0_30  = ('overdue_0_30',  'max'),max_overdue_31_60 = ('overdue_31_60', 'max'),max_par30_amount  = ('par_30_amount', 'max'),max_par90_amount  = ('par_90_amount', 'max')).reset_index()
df_sampled = df_sampled.merge(ageing_features, on='member_id', how='left')
prior_loan_features = []
for _, row in df_sampled[['id', 'member_id', 'disbursed_at']].iterrows():
    prior_loans = df_loans[(df_loans['member_id'] == row['member_id']) &(df_loans['disbursed_at'] <  row['disbursed_at']) &(df_loans['disbursed_at'].notnull())].copy()
    if len(prior_loans) == 0:
      prior_loan_features.append({
            'id'                         : row['id'],
            'avg_installment_delay_days' : np.nan,
            'pct_ontime_installments'    : np.nan,
            'prior_npa_loan_count'       : 0,
            'max_overdue_amount_ever'    : 0,
            'days_since_last_loan_closed': np.nan,
            'avg_prior_loan_duration'    : np.nan,
            'completed_loans_count'      : 0,})
      continue
    prior_loan_ids = prior_loans['id'].tolist()
    prior_inst = df_installments[df_installments['loan_id'].isin(prior_loan_ids)].copy()

    # 1. Average delay in installment payments
    paid_inst = prior_inst[prior_inst['paid_at'].notnull()].copy()
    paid_inst['delay_days'] = (paid_inst['paid_at'] - paid_inst['due_date']).dt.days.clip(lower=0)
    avg_delay = paid_inst['delay_days'].mean() if len(paid_inst) > 0 else np.nan

    # 2. Percentage of installments paid on or before due date
    if len(paid_inst) > 0:
        pct_ontime = (paid_inst['paid_at'] <= paid_inst['due_date']).mean()
    else:
        pct_ontime = np.nan

    # 3. Number of prior loans classified as NPA
    prior_npa = df_ageing[(df_ageing['member_id']  == row['member_id']) &(df_ageing['npa_classification']  != 'standard') &(df_ageing['loan_account_id'].isin(prior_loan_ids))]['loan_account_id'].nunique()

    # 4. Maximum overdue amount in prior loans
    prior_ageing = df_ageing[df_ageing['loan_account_id'].isin(prior_loan_ids)]
    max_overdue  = prior_ageing[['overdue_0_30', 'overdue_31_60', 'overdue_61_90', 'overdue_91_180', 'overdue_181_365', 'overdue_365_plus']].sum(axis=1).max()
    max_overdue  = max_overdue if not pd.isnull(max_overdue) else 0

    # 5. Gap between closure of previous loan and current loan sanction
    prior_loans['updated_at'] = pd.to_datetime(prior_loans.get('updated_at'), errors='coerce')
    closed_loans = prior_loans[prior_loans['status'].str.lower().isin(['closed'])]
    if len(closed_loans) > 0:
        last_closed  = closed_loans['updated_at'].max()
        days_gap     = (row['disbursed_at'] - last_closed).days
    else:
        days_gap = np.nan

    # 6. Average loan cycle duration for prior completed loans
    prior_loans['disbursed_at'] = pd.to_datetime(prior_loans['disbursed_at'], errors='coerce')
    completed = prior_loans[prior_loans['status'].str.lower() == 'closed'].copy()
    if len(completed) > 0:
        completed['updated_at']    = pd.to_datetime(completed['updated_at'], errors='coerce')
        completed['loan_duration'] = (completed['updated_at'] - completed['disbursed_at']).dt.days
        avg_duration = completed['loan_duration'].mean()
    else:
        avg_duration = np.nan

    # 7. Count of successfully completed loans
    completed_count = len(completed)
    prior_loan_features.append({
        'id'                         : row['id'],
        'avg_installment_delay_days' : round(avg_delay, 2) if not pd.isnull(avg_delay) else np.nan,
        'pct_ontime_installments'    : round(pct_ontime, 4) if not pd.isnull(pct_ontime) else np.nan,
        'prior_npa_loan_count'       : prior_npa,
        'max_overdue_amount_ever'    : max_overdue,
        'days_since_last_loan_closed': days_gap,
        'avg_prior_loan_duration'    : round(avg_duration, 2) if not pd.isnull(avg_duration) else np.nan,
        'completed_loans_count'      : completed_count,
    })

df_prior_history = pd.DataFrame(prior_loan_features)
df_sampled       = df_sampled.merge(df_prior_history, on='id', how='left')

print(f"After prior loan history features : {df_sampled.shape}")
feature_cols = [
    # Loan level
    'loan_cycle', 'sanctioned_amount', 'num_installments','is_collateral_based', 'collateral_value', 'loan_to_max_ratio','roi_tier_1_rate', 'thrift_eligibility_multiplier','sanctioned_vs_max_diff', 'interest_burden','collateral_coverage_ratio', 'loan_limit_utilization_pct',

    # Member level
    'member_age', 'member_tenure', 'is_aadhar_verified','gender', 'occupation', 'caste_category', 'marital_status','days_to_first_loan', 'prior_loan_count','avg_loan_gap_days', 'loan_frequency_per_year',

    # Savings level
    'savings_balance', 'savings_tenure_months', 'savings_to_loan_ratio','penalty_count', 'recent_penalty_flag', 'avg_monthly_deposit','deposit_consistency_months', 'days_since_last_savings_txn','deposit_growth_trend', 'deposit_to_withdrawal_ratio','inactive_savings_months',

    # Group level
    'group_age_months', 'group_size', 'group_is_active','attendance_rate', 'recent_attendance_rate', 'attendance_trend','last5_meetings_attendance', 'max_consecutive_missed','group_default_rate', 'group_avg_repayment_rate','group_stability_ratio',

    # KYC level
    'kyc_status', 'kyc_attempt_count', 'kyc_days_to_review','days_registration_to_kyc_approval', 'kyc_rejected_count','days_kyc_to_loan_sanction', 'kyc_completed_before_loan',

    # Guarantor level
    'guarantor_count','avg_guarantor_burden_tier','total_contingent_liability','guarantors_with_overdue','has_circular_guarantee','avg_active_guarantees_per_guar',

    # Branch level
    'branch_type', 'branch_npa_rate','branch_avg_sanctioned_amount','branch_avg_repayment_rate','branch_approval_rate','branch_product_diversity',

    # Prior loan history
    'ever_npa', 'max_overdue_0_30', 'max_overdue_31_60','max_par30_amount', 'max_par90_amount','avg_installment_delay_days','pct_ontime_installments','prior_npa_loan_count','max_overdue_amount_ever','days_since_last_loan_closed','avg_prior_loan_duration','completed_loans_count',]
target_col   = 'FPD'
feature_cols = [c for c in feature_cols if c in df_sampled.columns]
df_model     = df_sampled[feature_cols + [target_col]].copy()

print(f"\nModel dataset shape : {df_model.shape}")
print(f"Total features      : {len(feature_cols)}")
print(f"\nMissing values:")
print(df_model.isnull().sum()[df_model.isnull().sum() > 0])
numeric_cols = df_model.drop(columns=['FPD']).select_dtypes(include=np.number).columns
std_report = (df_model[numeric_cols].std().sort_values(ascending=False, na_position='last').reset_index())
std_report.columns = ['Feature', 'Standard Deviation']
print(f"Total Numerical Features : {len(std_report)}")
print(std_report.to_string(index=False))
def std_category(x):
    if pd.isna(x):
        return "NaN Std (Drop)"
    elif x == 0:
        return "Zero Std (Drop)"
    elif x < 0.01:
        return "Near Zero Std"
    elif x < 1:
        return "Low Std"
    elif x < 100:
        return "Medium Std"
    elif x < 10000:
        return "High Std"
    else:
        return "Very High Std"
std_report["Category"] = std_report["Standard Deviation"].apply(std_category)
print("\nFeature Categories")
print(std_report)
print("\nCategory Counts")
print(std_report["Category"].value_counts())
nan_std = std_report[std_report["Standard Deviation"].isna()]
zero_std = std_report[std_report["Standard Deviation"] == 0]
near_zero = std_report[(std_report["Standard Deviation"] > 0) &(std_report["Standard Deviation"] < 0.01)]
high_std = std_report[std_report["Standard Deviation"] > 10000]
print("\nNaN Std Features")
print(nan_std if len(nan_std) else "None")
print("\nZero Std Features")
print(zero_std if len(zero_std) else "None")
print("\nNear Zero Std Features")
print(near_zero if len(near_zero) else "None")
print("\nVery High Std Features")
print(high_std if len(high_std) else "None")
drop_cols = nan_std["Feature"].tolist() + zero_std["Feature"].tolist()
if drop_cols:
    print("\nDropping NaN/Zero Std Features:")
    print(drop_cols)
    df_model.drop(columns=drop_cols, inplace=True)
else:
    print("\nNo NaN or Zero Std Features Found.")
log_cols = high_std["Feature"].tolist()
if log_cols:
    print("\nApplying Log Transformation:")
    for col in log_cols:
        if col in df_model.columns:
            df_model[col + "_log"] = np.log1p(df_model[col].clip(lower=0))
            df_model.drop(columns=[col], inplace=True)
            print(f"{col}  -->  {col}_log")
else:
    print("\nNo Features Require Log Transformation.")
final_std = (
    df_model.drop(columns=['FPD']).select_dtypes(include=np.number).std())
print("\nFinal Dataset Summary")
print(f"Dataset Shape : {df_model.shape}")
print(f"Features      : {len(df_model.columns)-1}")
print(f"Minimum Std   : {final_std.min():.6f}")
print(f"Maximum Std   : {final_std.max():.2f}")
print(f"Average Std   : {final_std.mean():.2f}")
import pandas as pd
import numpy as np
feature_cols = [c for c in feature_cols if c in df_model.columns]
all_nan_cols = df_model.columns[df_model.isna().all()].tolist()
print(f"Columns with all NaN values ({len(all_nan_cols)}):")
print(all_nan_cols)
df_model.drop(columns=all_nan_cols, inplace=True)
print(f"\nDataset shape after dropping: {df_model.shape}")

import pandas as pd
import numpy as np
feature_cols = [c for c in feature_cols if c in df_model.columns]
categorical_cols = df_model[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print("Categorical Features")
for col in categorical_cols:
    print(f"\n{col}")
    print(df_model[col].value_counts(dropna=False))

# ONE-HOT ENCODING
one_hot_cols = [
    col for col in categorical_cols]
print("\nOne-Hot Encoding Columns:")
print(one_hot_cols)
df_model = pd.get_dummies(df_model,columns=one_hot_cols,drop_first=True,dtype=int)
print("\nEncoding Completed Successfully!")
print(f"Dataset Shape : {df_model.shape}")
print("\nRemaining Categorical Columns:")
print(df_model.select_dtypes(include=['object','category','bool']).columns.tolist())

print(f"\nMissing values:")
print(df_model.isnull().sum()[df_model.isnull().sum() > 0])
# Columns with missing values
missing_cols = ['max_overdue_0_30','max_overdue_31_60','max_par30_amount','max_par90_amount']

# Check missing values before filling
print("Missing values before filling:")
print(df_model[missing_cols].isnull().sum())

# Fill missing values with 0
df_model[missing_cols] = df_model[missing_cols].fillna(0)

print("\nMissing values after filling:")
print(df_model[missing_cols].isnull().sum())

print("\nTotal missing values in dataset:")
print(df_model.isnull().sum().sum())
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X = df_model.drop(columns=[target_col])
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nX_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"\nTrain FPD distribution:\n{y_train.value_counts()}")
print(f"\nTest  FPD distribution:\n{y_test.value_counts()}")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score,classification_report,confusion_matrix,ConfusionMatrixDisplay)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100,random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=5,subsample=0.8,colsample_bytree=0.8,random_state=42,eval_metric='logloss')
}
results = {}
print("TRAINING MACHINE LEARNING MODELS")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        "model": model,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    print(name)
    print("ROC-AUC :", round(auc,4))
    print(classification_report(y_test,y_pred,target_names=["No Default","Default"]))

# ANN
print("TRAINING ARTIFICIAL NEURAL NETWORK")
scaler = StandardScaler()
X_train_ann = scaler.fit_transform(X_train)
X_test_ann = scaler.transform(X_test)
ann = Sequential()
ann.add(Dense(64, activation='relu', input_shape=(X_train_ann.shape[1],)))
ann.add(Dropout(0.30))
ann.add(Dense(32, activation='relu'))
ann.add(Dropout(0.20))
ann.add(Dense(16, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))
ann.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
history = ann.fit(X_train_ann,y_train,validation_split=0.20,epochs=30,batch_size=32,verbose=1)
ann_prob = ann.predict(X_test_ann).flatten()
ann_pred = (ann_prob >= 0.5).astype(int)
ann_auc = roc_auc_score(y_test, ann_prob)
results["ANN"] = {
    "model": ann,
    "auc": ann_auc,
    "y_pred": ann_pred,
    "y_prob": ann_prob
}

print("\nROC-AUC :", round(ann_auc,4))
print(classification_report( y_test,ann_pred,target_names=["No Default","Default"]))

# MODEL COMPARISON

comparison = pd.DataFrame({"Model": list(results.keys()),"ROC-AUC": [results[m]["auc"] for m in results]})
comparison = comparison.sort_values(by="ROC-AUC",ascending=False).reset_index(drop=True)
print("MODEL COMPARISON")
print(comparison)
plt.figure(figsize=(8,5))
plt.bar(comparison["Model"],comparison["ROC-AUC"])
plt.ylabel("ROC-AUC")
plt.xlabel("Models")
plt.title("ROC-AUC Comparison of Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
# BEST MODEL
best_model = comparison.iloc[0]["Model"]
print("\nBest Model :", best_model)
# CONFUSION MATRIX
cm = confusion_matrix(y_test,results[best_model]["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["No Default","Default"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model}")
plt.tight_layout()
plt.show()
# FEATURE IMPORTANCE
if best_model in ["Random Forest","XGBoost"]:
    feat_imp = (pd.Series(results[best_model]["model"].feature_importances_,index=X_train.columns).sort_values(ascending=False).head(20))
    plt.figure(figsize=(10,6))
    feat_imp.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Top 20 Important Features - {best_model}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

print("\nFinished Model Comparison Successfully!")
from sklearn.metrics import accuracy_score

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    somers_d = 2 * auc - 1

    print(f"\n{name}")
    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"Somers D      : {somers_d:4f}")
default_pct = df_model["FPD"].mean() * 100

print(f"Total Records : {len(df_model)}")
print(f"Defaults      : {df_model['FPD'].sum()}")
print(f"Non-defaults  : {(df_model['FPD']==0).sum()}")
print(f"Default Rate  : {default_pct:.2f}%")
print("MODEL COMPARISON")
comparison = []
for name, res in results.items():
    auc = res['auc']
    somers_d = 2 * auc - 1
    comparison.append({'Model': name,'ROC-AUC': round(auc, 4),'Somers D': round(somers_d, 4)})
comparison_df = pd.DataFrame(comparison)
comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
print(comparison_df.to_string(index=False))
