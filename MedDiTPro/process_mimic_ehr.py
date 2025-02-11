import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ICD-9 and ICD-10 conversion functions
def convert_to_3digit_icd9(dxStr):
    if not isinstance(dxStr, str):
        dxStr = str(dxStr)
    if dxStr.startswith('E'):
        return dxStr[:4] if len(dxStr) > 4 else dxStr
    return dxStr[:3] if len(dxStr) > 3 else dxStr


def convert_to_3digit_icd10(dxStr):
    if not isinstance(dxStr, str):
        dxStr = str(dxStr)
    return dxStr[:3] if len(dxStr) > 3 else dxStr


def convert_to_3digit_diag_icd(dxStr, icd_version=9):
    if icd_version == 9:
        return convert_to_3digit_icd9(dxStr)
    elif icd_version == 10:
        return convert_to_3digit_icd10(dxStr)
    else:
        raise ValueError("Invalid ICD version. Supported versions: 9 or 10.")


# Procedure code conversion functions
def convert_proc_to_shorter_icd9(dxStr, length=3):
    if not isinstance(dxStr, str):
        dxStr = str(dxStr)
    if '.' in dxStr:
        return dxStr[:length + 1] if len(dxStr) > length + 1 else dxStr
    else:
        return dxStr[:length] if len(dxStr) > length else dxStr


def convert_proc_to_shorter_icd10(dxStr, length=3):
    if not isinstance(dxStr, str):
        dxStr = str(dxStr)
    return dxStr[:length] if len(dxStr) > length else dxStr


def convert_proc_to_shorter_icd(dxStr, icd_version=9, length=3):
    if icd_version == 9:
        return convert_proc_to_shorter_icd9(dxStr, length)
    elif icd_version == 10:
        return convert_proc_to_shorter_icd10(dxStr, length)
    else:
        raise ValueError("Invalid ICD version. Supported versions: 9 or 10.")


# Drug code processing
def chop_formulary_code(row):
    if 'd5w' in row['formulary_drug_cd'] or 'd5w' in str(row['drug']):
        return 'D5W'
    if 'ns' in row['formulary_drug_cd']:
        return 'NS'
    if 'bag' in row['formulary_drug_cd']:
        return 'BAG'
    if 'aa5d' in row['formulary_drug_cd']:
        return 'AA5D'
    if '5fu' in row['formulary_drug_cd']:
        return '5FU'
    chopped_code = row['formulary_drug_cd'][:next((i for i, c in enumerate(row['formulary_drug_cd']) if c.isdigit()),
                                                  len(row['formulary_drug_cd']))]
    return str(chopped_code)


# Load dataset function with proper column names handling and merging strategy
def load_data(dataset_name):
    if dataset_name == "mimic_iii":
        path = 'mimiciii/1.4/'
        print("Loading MIMIC-III dataset files...")
        raw_admission = pd.read_csv(path + 'ADMISSIONS.csv')
        print('admission loaded')
        raw_patient = pd.read_csv(path + 'PATIENTS.csv')
        print('patient loaded')
        raw_diag = pd.read_csv(path + 'DIAGNOSES_ICD.csv')
        print('diagnoses loaded')
        raw_drug = pd.read_csv(path + 'PRESCRIPTIONS.csv')
        print('prescriptions loaded')
        raw_lab = pd.read_csv(path + 'LABEVENTS.csv')
        print('labevents loaded')
        raw_proc = pd.read_csv(path + 'PROCEDURES_ICD.csv')
        print('procedures loaded')
    elif dataset_name == "mimic_iv":
        path = 'mimiciv/3.0/hosp/'
        print("Loading MIMIC-IV dataset files...")
        raw_admission = pd.read_csv(path + 'admissions.csv')
        print('admission loaded')
        raw_patient = pd.read_csv(path + 'patients.csv')
        print('patient loaded')
        raw_diag = pd.read_csv(path + 'diagnoses_icd.csv')
        print('diagnoses loaded')
        raw_drug = pd.read_csv(path + 'prescriptions.csv')
        print('prescriptions loaded')
        raw_lab = pd.read_csv(path + 'labevents.csv')
        print('labevents loaded')
        raw_proc = pd.read_csv(path + 'procedures_icd.csv')
        print('procedures loaded')
    else:
        raise ValueError("Invalid dataset name. Supported values: mimic_iii, mimic_iv")

    # Convert column names to lowercase for consistency
    raw_admission.columns = raw_admission.columns.str.lower()
    raw_patient.columns = raw_patient.columns.str.lower()
    raw_diag.columns = raw_diag.columns.str.lower()
    raw_drug.columns = raw_drug.columns.str.lower()
    raw_lab.columns = raw_lab.columns.str.lower()
    raw_proc.columns = raw_proc.columns.str.lower()

    raw_drug_cleaned = raw_drug.dropna(subset=['formulary_drug_cd', 'subject_id'])
    print(f'Dropped {raw_drug.shape[0] - raw_drug_cleaned.shape[0]} rows with missing formulary_drug_cd or subject_id.')

    # cast raw_lab's itemid to string
    raw_lab['itemid'] = raw_lab['itemid'].astype(str)
    print('lab itemid casted to string')

    print(f"Data for {dataset_name} loaded successfully with lowercase columns.")
    return raw_admission, raw_patient, raw_diag, raw_drug_cleaned, raw_lab, raw_proc


# Function to aggregate codes for each modality (e.g., diagnoses, drugs, procedures)
def aggregate_modality(modality_df, admissions_df, code_column):
    # Merge modality data with admissions data to get 'admittime' and 'dischtime'
    merged_df = pd.merge(modality_df, admissions_df[['subject_id', 'hadm_id', 'admittime', 'dischtime']],
                         on=['subject_id', 'hadm_id'], how='left')

    # Group by 'subject_id' and 'hadm_id' to aggregate visits
    grouped_visits = merged_df.groupby(['subject_id', 'hadm_id']).agg({
        code_column: lambda x: list(x.drop_duplicates()),  # Unique codes for each visit
        'admittime': 'first',
        'dischtime': 'first'
    }).reset_index()

    return grouped_visits



def build_visit_level_representation(grouped_diag, grouped_drug, grouped_lab, grouped_proc, patient_data):
    visit_data = {}
    
    # Initialize global indexer with special tokens
    code_to_index = {
        # General special tokens
        'pad': 0,         # Padding token
        'no_record': 1,   # No record for a modality
        'bos': 2,         # Beginning of sequence
        'eos': 3,         # End of sequence
        'sep': 4,         # Separator between different parts
        
        # Modality-specific markers
        'diag_start': 5,  # Start of diagnosis codes
        'diag_end': 6,    # End of diagnosis codes
        'drug_start': 7,   # Start of drug codes
        'drug_end': 8,     # End of drug codes
        'proc_start': 9,  # Start of procedure codes
        'proc_end': 10,   # End of procedure codes
        'lab_start': 11,  # Start of lab codes
        'lab_end': 12,     # End of lab codes
        'mask': 13,       # Masking token for MLM
        'visit_start': 14,  # Start of visit marker
        'visit_end': 15,    # End of visit marker
        'no_next_visit': 16 # No next visit marker
    }
    current_idx = max(code_to_index.values()) + 1  # Start from after special tokens
    
    # Create modality markers for actual codes
    MODALITY_MARKERS = {
        'diag': 'D_',
        'drug': 'M_',
        'proc': 'P_',
        'lab': 'L_'
    }
    
    # Create mapping for modality start/end tokens
    MODALITY_SPECIAL_TOKENS = {
        'diag_code': (code_to_index['diag_start'], code_to_index['diag_end']),
        'drug_code': (code_to_index['drug_start'], code_to_index['drug_end']),
        'proc_code': (code_to_index['proc_start'], code_to_index['proc_end']),
        'lab_code': (code_to_index['lab_start'], code_to_index['lab_end'])
    }
    
    patient_data['dod'] = pd.to_datetime(patient_data['dod'], errors='coerce')
    mortality_flag = {row['subject_id']: not pd.isna(row['dod']) for _, row in patient_data.iterrows()}

    def initialize_visit(subject_id, hadm_id, admittime, dischtime):
        if subject_id not in visit_data:
            visit_data[subject_id] = {
                'visits': {},
                'visit_order': []
            }
        
        # Initialize each modality with start token, no_record token, and end token
        visit_data[subject_id]['visits'][hadm_id] = {
            'admittime': pd.to_datetime(admittime),
            'dischtime': pd.to_datetime(dischtime),
            # Use lists with correct order for each modality
            'diag_code': [code_to_index['diag_start'], code_to_index['no_record'], code_to_index['diag_end']],
            'drug_code': [code_to_index['drug_start'], code_to_index['no_record'], code_to_index['drug_end']],
            'lab_code': [code_to_index['lab_start'], code_to_index['no_record'], code_to_index['lab_end']],
            'proc_code': [code_to_index['proc_start'], code_to_index['no_record'], code_to_index['proc_end']],
            'event': mortality_flag.get(subject_id, False)
        }

    def process_codes(row, code_field, modality_marker):
        nonlocal current_idx
        subject_id, hadm_id = row['subject_id'], row['hadm_id']
        
        if subject_id not in visit_data:
            visit_data[subject_id] = {
                'visits': {},
                'visit_order': []
            }
        if hadm_id not in visit_data[subject_id]['visits']:
            initialize_visit(subject_id, hadm_id, row['admittime'], row['dischtime'])
            visit_data[subject_id]['visit_order'].append(hadm_id)
        
        # Create indices for actual codes
        code_indices = []  # Use list instead of set to maintain order
        for code in row[code_field]:
            marked_code = f"{modality_marker}{code}"
            if marked_code not in code_to_index:
                code_to_index[marked_code] = current_idx
                current_idx += 1
            code_indices.append(code_to_index[marked_code])
        
        # Map input fields to storage fields
        field_mapping = {
            'diag_code': ('diag_code', 'diag_start', 'diag_end'),
            'drug': ('drug_code', 'drug_start', 'drug_end'),
            'proc_code': ('proc_code', 'proc_start', 'proc_end'),
            'itemid': ('lab_code', 'lab_start', 'lab_end')
        }
        
        # Get the storage field and tokens
        storage_field, start_token, end_token = field_mapping[code_field]
        start_idx = code_to_index[start_token]
        end_idx = code_to_index[end_token]

        if code_indices:
            visit_data[subject_id]['visits'][hadm_id][storage_field] = [start_idx] + sorted(code_indices) + [end_idx]
        else:
            # No codes found, add placeholder with no_record token
            visit_data[subject_id]['visits'][hadm_id][storage_field] = [start_idx, code_to_index['no_record'], end_idx]
        
        return subject_id, hadm_id, code_indices

    # Process each modality
    print("Processing diagnosis data...")
    for _, row in tqdm(grouped_diag.iterrows(), total=len(grouped_diag)):
        subject_id, hadm_id, _ = process_codes(row, 'diag_code', MODALITY_MARKERS['diag'])

    print("Processing drug data...")
    for _, row in tqdm(grouped_drug.iterrows(), total=len(grouped_drug)):
        subject_id, hadm_id, _ = process_codes(row, 'drug', MODALITY_MARKERS['drug'])

    print("Processing procedure data...")
    for _, row in tqdm(grouped_proc.iterrows(), total=len(grouped_proc)):
        subject_id, hadm_id, _ = process_codes(row, 'proc_code', MODALITY_MARKERS['proc'])

    print("Processing lab data...")
    for _, row in tqdm(grouped_lab.iterrows(), total=len(grouped_lab)):
        subject_id, hadm_id, _ = process_codes(row, 'itemid', MODALITY_MARKERS['lab'])

    # Sort visits and calculate time gaps
    print("Sorting visits chronologically...")
    for subject_id in tqdm(visit_data.keys()):
        visits = visit_data[subject_id]['visits']
        sorted_hadm_ids = sorted(visits.keys(), key=lambda x: visits[x]['admittime'])
        visit_data[subject_id]['visit_order'] = sorted_hadm_ids
        
        # Calculate time gaps
        for i, hadm_id in enumerate(sorted_hadm_ids):
            if i < len(sorted_hadm_ids) - 1:
                next_hadm_id = sorted_hadm_ids[i + 1]
                time_gap = (visits[next_hadm_id]['admittime'] - visits[hadm_id]['admittime']).days
            else:
                time_gap = 0
            visits[hadm_id]['time_gap'] = time_gap

    # Convert sets to lists
    print("Finalizing visit data...")
    for subject_id in tqdm(visit_data.keys()):
        for hadm_id in visit_data[subject_id]['visits']:
            visit = visit_data[subject_id]['visits'][hadm_id]

    return visit_data, code_to_index


# Function to calculate time gaps between visits for a patient
def calculate_time_gaps(visits):
    """
    This function calculates the time gaps between visits for a patient.
    Args:
    - visits (list of dict): List of visits sorted by admittime

    Returns:
    - list of int: Time gaps in days between visits (0 for the last visit)
    """
    time_gaps = []
    for i in range(len(visits) - 1):
        admittime_current = visits[i]['admittime']
        admittime_next = visits[i + 1]['admittime']
        gap = (admittime_next - admittime_current).days
        time_gaps.append(gap)

    # The last visit has a gap of 0 days since there's no future visit
    time_gaps.append(0)

    return time_gaps


# Function to aggregate patient-level data and calculate time gaps between visits
def aggregate_patient_level(visit_data):
    patient_data = {}

    # Use tqdm to show progress for processing each subject_id
    print("Aggregating patient-level data...")
    for subject_id, visits in tqdm(visit_data.items(), total=len(visit_data)):
        visit_list = []

        # Process each visit for the subject
        for hadm_id, visit_info in visits.items():
            visit_info['hadm_id'] = hadm_id
            visit_list.append(visit_info)

        # Sort visits by admittime
        visit_list.sort(key=lambda x: x['admittime'])

        # Calculate time gaps between visits
        time_gaps = calculate_time_gaps(visit_list)

        # Add time gaps to each visit's data
        for i, visit in enumerate(visit_list):
            visit['time_gap'] = time_gaps[i]

        # Store sorted visits for this patient
        patient_data[subject_id] = visit_list

    return patient_data


def main(dataset_name, short_ICD=True, proc_digits=3, drug_agg=True, save=False, icd_version=None):
    """
    Main function to process MIMIC datasets
    Args:
        dataset_name: 'mimic_iii' or 'mimic_iv'
        short_ICD: Whether to convert ICD codes to 3-digit format
        proc_digits: Number of digits to keep for procedure codes
        drug_agg: Whether to aggregate drug codes
        save: Whether to save processed data
        icd_version: For MIMIC-IV only - specify '9' or '10' to filter by ICD version, or None to include both
    """
    raw_admission, raw_patient, raw_diag, raw_drug, raw_lab, raw_proc = load_data(dataset_name)

    # Filter by ICD version for MIMIC-IV
    if dataset_name == "mimic_iv" and icd_version is not None:
        if icd_version not in [9, 10]:
            raise ValueError("icd_version must be 9 or 10 for MIMIC-IV")
        
        # Filter diagnoses
        raw_diag = raw_diag[raw_diag['icd_version'] == icd_version]
        if len(raw_diag) == 0:
            raise ValueError(f"No ICD-{icd_version} diagnosis codes found in the dataset")
        
        # Filter procedures
        raw_proc = raw_proc[raw_proc['icd_version'] == icd_version]
        if len(raw_proc) == 0:
            raise ValueError(f"No ICD-{icd_version} procedure codes found in the dataset")
        
        print(f"Filtered to ICD-{icd_version} codes only:")
        print(f"  Diagnoses: {len(raw_diag)} records")
        print(f"  Procedures: {len(raw_proc)} records")

    # Process diagnosis codes
    if short_ICD:
        if dataset_name == "mimic_iii":
            raw_diag['diag_code'] = raw_diag['icd9_code'].apply(lambda x: convert_to_3digit_diag_icd(x, icd_version=9))
        elif dataset_name == "mimic_iv":
            if icd_version is not None:
                raw_diag['diag_code'] = raw_diag['icd_code'].apply(
                    lambda x: convert_to_3digit_diag_icd(x, icd_version=icd_version))
            else:
                raw_diag['diag_code'] = raw_diag.apply(
                    lambda row: convert_to_3digit_diag_icd(row['icd_code'], icd_version=row['icd_version']), axis=1)
    else:
        if dataset_name == "mimic_iii":
            raw_diag['diag_code'] = raw_diag['icd9_code']
        elif dataset_name == "mimic_iv":
            raw_diag['diag_code'] = raw_diag['icd_code']
    print("Diagnoses processed successfully!")

    # Process procedure codes
    if dataset_name == "mimic_iii":
        raw_proc['proc_code'] = raw_proc['icd9_code'].apply(
            lambda x: convert_proc_to_shorter_icd9(x, length=proc_digits))
    elif dataset_name == "mimic_iv":
        if icd_version is not None:
            raw_proc['proc_code'] = raw_proc['icd_code'].apply(
                lambda x: convert_proc_to_shorter_icd(x, icd_version=icd_version, length=proc_digits))
        else:
            raw_proc['proc_code'] = raw_proc.apply(
                lambda row: convert_proc_to_shorter_icd(row['icd_code'], icd_version=row['icd_version'],
                                                      length=proc_digits), axis=1)
    print("Procedures processed successfully!")

    # Process drug codes (unchanged)
    if drug_agg:
        raw_drug['drug'] = raw_drug.apply(chop_formulary_code, axis=1)
    else:
        raw_drug['drug'] = raw_drug['formulary_drug_cd']
    print("Drugs processed successfully!")

    print("Aggregating data...")
    # Aggregate each modality
    grouped_diag = aggregate_modality(raw_diag, raw_admission, 'diag_code')
    grouped_drug = aggregate_modality(raw_drug, raw_admission, 'drug')
    grouped_lab = aggregate_modality(raw_lab, raw_admission, 'itemid')
    grouped_proc = aggregate_modality(raw_proc, raw_admission, 'proc_code')

    # Build visit-level representation with global indexer
    print("Building visit-level representation...")
    patient_data, code_to_index = build_visit_level_representation(
        grouped_diag, grouped_drug, grouped_lab, grouped_proc, raw_patient)

    if save:
        # Add ICD version to filename for MIMIC-IV
        icd_suffix = f"_icd{icd_version}" if dataset_name == "mimic_iv" and icd_version is not None else ""
        pickle.dump(patient_data, open(f"{dataset_name}{icd_suffix}_ehr.pkl", "wb"))
        pickle.dump(code_to_index, open(f"{dataset_name}{icd_suffix}_code_to_index.pkl", "wb"))

    print(f"Data from {dataset_name} processed successfully!")
    if dataset_name == "mimic_iv" and icd_version is not None:
        print(f"Processed using ICD-{icd_version} codes only")
    return patient_data, code_to_index

def take_a_look(dataset_name='mimic_iii', num_patients: int = 3, icd_version=None):
    """
    Inspect processed EHR data
    Args:
        dataset_name: 'mimic_iii' or 'mimic_iv'
        num_patients: Number of patients to display
        icd_version: For MIMIC-IV only - specify '9' or '10' to view specific ICD version data
    """
    count = 0
    # Construct filename based on dataset and ICD version
    icd_suffix = f"_icd{icd_version}" if dataset_name == "mimic_iv" and icd_version is not None else ""
    
    with open(f"{dataset_name}{icd_suffix}_ehr.pkl", 'rb') as f:
        data = pickle.load(f)
    with open(f"{dataset_name}{icd_suffix}_code_to_index.pkl", 'rb') as f:
        code_to_index = pickle.load(f)
        
    # Create reverse mapping
    index_to_code = {v: k for k, v in code_to_index.items()}
    special_tokens = {'pad', 'no_record', 'bos', 'eos', 'sep', 
                     'diag_start', 'diag_end', 'drug_start', 'drug_end',
                     'proc_start', 'proc_end', 'lab_start', 'lab_end',
                     'mask', 'visit_start', 'visit_end', 'no_next_visit'}
    
    print(f"Loaded {dataset_name} EHR data with {len(data)} patients.")
    if dataset_name == "mimic_iv" and icd_version is not None:
        print(f"Using ICD-{icd_version} codes only")
    
    for subject_id, patient_info in data.items():
        if len(patient_info['visits']) <= 1:
            continue
            
        print(f"\nPatient ID: {subject_id}")
        print(f"Visit order: {patient_info['visit_order']}")
        
        for hadm_id in patient_info['visit_order']:
            visit = patient_info['visits'][hadm_id]
            print(f"\n  Visit ID: {hadm_id}")
            print(f"  Admission time: {visit['admittime']}")
            print(f"  Time Gap: {visit['time_gap']} days")
            
            # Print each modality with decoded special tokens
            for modality in ['diag_code', 'drug_code', 'proc_code', 'lab_code']:
                codes = visit[modality]
                decoded = []
                for code in codes:
                    token = index_to_code.get(code, str(code))
                    if token in special_tokens:
                        decoded.append(f"<{token}>")
                    else:
                        if isinstance(token, str) and any(token.startswith(p) for p in ['D_', 'M_', 'P_', 'L_']):
                            decoded.append(f"{token}({code})")
                        else:
                            decoded.append(str(code))
                print(f"    {modality.replace('_code', '').capitalize()}: {' '.join(decoded)}")
            
            print(f"    Event: {'Died' if visit['event'] else 'Survived'}")
            
        print("-" * 50)
        count += 1
        if count >= num_patients:
            break

def take_a_look_at_mapping(dataset_name='mimic_iii', icd_version=None):
    """
    Inspect code-to-index mapping
    Args:
        dataset_name: 'mimic_iii' or 'mimic_iv'
        icd_version: For MIMIC-IV only - specify '9' or '10' to view specific ICD version mapping
    """
    # Construct filename based on dataset and ICD version
    icd_suffix = f"_icd{icd_version}" if dataset_name == "mimic_iv" and icd_version is not None else ""
    
    with open(f"{dataset_name}{icd_suffix}_code_to_index.pkl", 'rb') as f:
        mapping = pickle.load(f)
    
    print(f"Loaded {dataset_name} code to index mapping with {len(mapping)} unique codes.")
    if dataset_name == "mimic_iv" and icd_version is not None:
        print(f"Using ICD-{icd_version} codes only")
    
    # Group special tokens by type
    special_tokens = {
        'Basic Tokens': ['pad', 'no_record', 'bos', 'eos', 'sep', 'mask'],
        'Visit Tokens': ['visit_start', 'visit_end', 'no_next_visit'],
        'Diagnosis Tokens': ['diag_start', 'diag_end'],
        'Medication Tokens': ['drug_start', 'drug_end'],
        'Procedure Tokens': ['proc_start', 'proc_end'],
        'Lab Tokens': ['lab_start', 'lab_end']
    }
    
    print("\nSpecial Tokens:")
    for group, tokens in special_tokens.items():
        print(f"\n{group}:")
        for token in tokens:
            if token in mapping:
                print(f"  {token}: {mapping[token]}")
    
    print("\nMedical Codes by Modality:")
    for prefix, modality in [('D_', 'Diagnosis'), ('M_', 'Medication'), 
                           ('P_', 'Procedure'), ('L_', 'Lab')]:
        codes = {k: v for k, v in mapping.items() if k.startswith(prefix)}
        if codes:
            print(f"\n{modality} Codes:")
            print(f"  Total unique codes: {len(codes)}")
            print(f"  Index range: {min(codes.values())} - {max(codes.values())}")
            
            # For diagnosis codes in MIMIC-IV, show ICD version distribution
            if dataset_name == "mimic_iv" and prefix == 'D_' and icd_version is None:
                icd9_codes = sum(1 for k in codes if len(k.split('_')[1]) == 3)
                icd10_codes = sum(1 for k in codes if len(k.split('_')[1]) > 3)
                print(f"  ICD-9 codes: {icd9_codes}")
                print(f"  ICD-10 codes: {icd10_codes}")
            
            print(f"  First 5 examples:")
            for code, index in sorted(codes.items(), key=lambda x: x[1])[:5]:
                print(f"    {code}: {index}")

def split_data_by_reference(patient_data, code_to_index, dataset_name='mimic_iii', icd_version=None, save=True):
    """
    Split the processed EHR data. For MIMIC-III, uses reference files.
    For MIMIC-IV, performs random split with 75:10:15 ratio.
    
    Args:
        patient_data: The processed patient data dictionary
        code_to_index: The code to index mapping dictionary
        dataset_name: The name of the dataset ('mimic_iii' or 'mimic_iv')
        icd_version: For MIMIC-IV only - specify '9' or '10' to process specific ICD version data
        save: Whether to save the split datasets
    """
    if dataset_name == 'mimic_iii':
        # Use reference files for MIMIC-III
        splits = {
            'train': pd.read_csv('train_3digmimic.csv')['SUBJECT_ID'].unique(),
            'test': pd.read_csv('test_3digmimic.csv')['SUBJECT_ID'].unique(),
            'val': pd.read_csv('val_3digmimic.csv')['SUBJECT_ID'].unique(),
            'toy': pd.read_csv('toy_3digmimic.csv')['SUBJECT_ID'].unique()
        }
        
        split_data = {
            split_name: {
                subject_id: patient_data[subject_id]
                for subject_id in subject_ids
                if subject_id in patient_data
            }
            for split_name, subject_ids in splits.items()
        }
        
    else:  # mimic_iv
        # Random split for MIMIC-IV
        np.random.seed(42)  # Set seed for reproducibility
        
        # Get all patient IDs
        all_ids = list(patient_data.keys())
        np.random.shuffle(all_ids)
        
        # Calculate split sizes
        n_total = len(all_ids)
        n_train = int(0.75 * n_total)
        n_val = int(0.10 * n_total)
        # n_test will be the remainder (approximately 15%)
        
        # Split the IDs
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:n_train + n_val]
        test_ids = all_ids[n_train + n_val:]
        
        # Create toy set (e.g., 100 patients from train set)
        toy_ids = train_ids[:100] if len(train_ids) > 100 else train_ids
        
        # Create the split datasets
        split_data = {
            'train': {id: patient_data[id] for id in train_ids},
            'val': {id: patient_data[id] for id in val_ids},
            'test': {id: patient_data[id] for id in test_ids},
            'toy': {id: patient_data[id] for id in toy_ids}
        }
    
    # Print statistics for each split
    for split_name, data in split_data.items():
        print(f"{split_name} set: {len(data)} patients "
              f"({len(data)/len(patient_data)*100:.1f}% of total)")
    
    if save:
        # Construct filename suffix based on dataset and ICD version
        icd_suffix = f"_icd{icd_version}" if dataset_name == "mimic_iv" and icd_version is not None else ""
        
        for split_name, data in split_data.items():
            filename = f"{dataset_name}{icd_suffix}_{split_name}_ehr.pkl"
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            
            # Also save the patient IDs for each split if it's MIMIC-IV
            if dataset_name == 'mimic_iv':
                id_filename = f"{dataset_name}{icd_suffix}_{split_name}_ids.txt"
                with open(id_filename, "w") as f:
                    for patient_id in data.keys():
                        f.write(f"{patient_id}\n")
        
        print(f"\nSaved {len(split_data)} splits to files:")
        for split_name in split_data.keys():
            print(f"  - {dataset_name}{icd_suffix}_{split_name}_ehr.pkl")
            if dataset_name == 'mimic_iv':
                print(f"  - {dataset_name}{icd_suffix}_{split_name}_ids.txt")
        print(f"\nNote: Using global {dataset_name}{icd_suffix}_code_to_index.pkl for all splits")
    
    return split_data

def check_splits(split_data, dataset_name='mimic_iii', icd_version=None):
    """
    Print statistics about the splits
    Args:
        split_data: Dictionary containing the split datasets
        dataset_name: 'mimic_iii' or 'mimic_iv'
        icd_version: For MIMIC-IV only - specify '9' or '10' to indicate which ICD version was used
    """
    print("\nSplit Statistics:")
    if dataset_name == "mimic_iv" and icd_version is not None:
        print(f"Using ICD-{icd_version} codes only")
        
    for split_name, data in split_data.items():
        # Calculate basic statistics
        total_visits = sum(len(patient_info['visits']) for patient_info in data.values())
        avg_visits = total_visits / len(data) if data else 0
        
        # Count visits with diagnoses
        visits_with_diag = 0
        total_diag_codes = 0
        
        for patient in data.values():
            for visit_id in patient['visits']:
                visit = patient['visits'][visit_id]
                diag_codes = visit['diag_code']
                if len(diag_codes) > 2:  # More than just start/end tokens
                    visits_with_diag += 1
                    total_diag_codes += len(diag_codes) - 2  # Subtract start/end tokens
        
        print(f"\n{split_name} set:")
        print(f"  Number of patients: {len(data)}")
        print(f"  Total visits: {total_visits}")
        print(f"  Average visits per patient: {avg_visits:.2f}")
        print(f"  Visits with diagnoses: {visits_with_diag}")
        if visits_with_diag > 0:
            print(f"  Average diagnoses per visit: {total_diag_codes/visits_with_diag:.2f}")
        
        # Sample a few patients
        if data:
            sample_patient_id = next(iter(data))
            sample_patient = data[sample_patient_id]
            print(f"\n  Sample patient (ID: {sample_patient_id}):")
            print(f"    Number of visits: {len(sample_patient['visits'])}")
            print(f"    Visit IDs: {sample_patient['visit_order']}")
            
            # Show diagnosis statistics for sample visit
            if len(sample_patient['visits']) > 0:
                sample_visit_id = sample_patient['visit_order'][0]
                sample_visit = sample_patient['visits'][sample_visit_id]
                diag_codes = sample_visit['diag_code']
                if len(diag_codes) > 2:  # More than just start/end tokens
                    print(f"    Sample visit (ID: {sample_visit_id}) has {len(diag_codes)-2} diagnoses")

def split_and_see_data(dataset_name='mimic_iii', save=True, icd_version=None):
    """
    Split and analyze the processed EHR data
    Args:
        dataset_name: 'mimic_iii' or 'mimic_iv'
        save: Whether to save the split datasets
        icd_version: For MIMIC-IV only - specify '9' or '10' to process specific ICD version data
    """
    # Construct filename based on dataset and ICD version
    icd_suffix = f"_icd{icd_version}" if dataset_name == "mimic_iv" and icd_version is not None else ""
    
    with open(f"{dataset_name}{icd_suffix}_ehr.pkl", "rb") as f:
        patient_data = pickle.load(f)
    with open(f"{dataset_name}{icd_suffix}_code_to_index.pkl", "rb") as f:
        code_to_index = pickle.load(f)
    
    print("Loaded existing processed data")
    if dataset_name == "mimic_iv" and icd_version is not None:
        print(f"Using ICD-{icd_version} codes only")
    
    # Split the data
    split_data = split_data_by_reference(
        patient_data, 
        code_to_index, 
        dataset_name=dataset_name,
        icd_version=icd_version,
        save=save
    )
    
    # Check the splits
    check_splits(split_data, dataset_name=dataset_name, icd_version=icd_version)


def analyze_hadm_lists(file_name='train_3digmimic.csv'):
    """
    Analyze the length of HADM_ID lists in the CSV file.
    
    Args:
        file_name: Name of the CSV file to analyze
    """
    # Read the CSV file
    df = pd.read_csv(file_name)
    
    # Convert string representation of list to actual list
    hadm_lists = df['HADM_ID'].apply(eval)  # This assumes the lists are stored as strings
    
    # Calculate statistics
    lengths = hadm_lists.apply(len)
    stats = {
        'average_length': lengths.mean(),
        'median_length': lengths.median(),
        'min_length': lengths.min(),
        'max_length': lengths.max(),
        'std_length': lengths.std(),
        'total_patients': len(df),
        'length_distribution': lengths.value_counts().sort_index()
    }
    
    # Print results
    print(f"Analysis of HADM_ID lists in {file_name}:")
    print(f"Total number of patients: {stats['total_patients']}")
    print(f"Average visits per patient: {stats['average_length']:.2f}")
    print(f"Median visits per patient: {stats['median_length']:.2f}")
    print(f"Min visits: {stats['min_length']}")
    print(f"Max visits: {stats['max_length']}")
    print(f"Standard deviation: {stats['std_length']:.2f}")
    
    print("\nDistribution of visit counts:")
    for visits, count in stats['length_distribution'].items():
        print(f"{visits} visits: {count} patients ({count/len(df)*100:.1f}%)")

def analyze_mimic_admissions(file_path='/mimiciii/1.4/ADMISSIONS.CSV'):
    """
    Analyze the number of visits (HADM_ID) per patient (SUBJECT_ID) in MIMIC-III admissions.
    """
    # Read the admissions file
    df = pd.read_csv(file_path)
    
    # Group by SUBJECT_ID and count HADM_IDs
    visit_counts = df.groupby('SUBJECT_ID')['HADM_ID'].count()
    
    # Calculate statistics
    stats = {
        'total_patients': len(visit_counts),
        'total_visits': len(df),
        'average_visits': visit_counts.mean(),
        'median_visits': visit_counts.median(),
        'min_visits': visit_counts.min(),
        'max_visits': visit_counts.max(),
        'std_visits': visit_counts.std(),
        'visit_distribution': visit_counts.value_counts().sort_index()
    }
    
    # Print results
    print(f"\nAnalysis of MIMIC-III Admissions:")
    print(f"Total number of patients: {stats['total_patients']}")
    print(f"Total number of visits: {stats['total_visits']}")
    print(f"Average visits per patient: {stats['average_visits']:.2f}")
    print(f"Median visits per patient: {stats['median_visits']:.2f}")
    print(f"Min visits: {stats['min_visits']}")
    print(f"Max visits: {stats['max_visits']}")
    print(f"Standard deviation: {stats['std_visits']:.2f}")
    
    print("\nDistribution of visit counts:")
    for visits, count in stats['visit_distribution'].items():
        percentage = (count/stats['total_patients'])*100
        print(f"{visits} visit{'s' if visits > 1 else ''}: {count} patients ({percentage:.1f}%)")

def take_a_look_at_userid(dataset_name='mimic_iii', id=21016, icd_version=None):
    """
    Look at the data for a specific user
    Args:
        dataset_name: 'mimic_iii' or 'mimic_iv'
        id: The user ID to look up
        icd_version: For MIMIC-IV only - specify '9' or '10' to view specific ICD version data
    """
    # Construct filename based on dataset and ICD version
    icd_suffix = f"_icd{icd_version}" if dataset_name == "mimic_iv" and icd_version is not None else ""
    
    with open(f"{dataset_name}{icd_suffix}_ehr.pkl", 'rb') as f:
        data = pickle.load(f)
    with open(f"{dataset_name}{icd_suffix}_code_to_index.pkl", 'rb') as f:
        code_to_index = pickle.load(f)
    
    if id not in data:
        print(f"Patient ID {id} not found in the dataset!")
        return
        
    print(f"Data for patient {id}:")
    if dataset_name == "mimic_iv" and icd_version is not None:
        print(f"Using ICD-{icd_version} codes only")
    
    # Create reverse mapping for better readability
    index_to_code = {v: k for k, v in code_to_index.items()}
    
    patient_info = data[id]
    print("\nVisit Summary:")
    print(f"Number of visits: {len(patient_info['visits'])}")
    print(f"Visit order: {patient_info['visit_order']}")
    
    print("\nDetailed Visit Information:")
    for hadm_id in patient_info['visit_order']:
        visit = patient_info['visits'][hadm_id]
        print(f"\nVisit ID: {hadm_id}")
        print(f"Admission time: {visit['admittime']}")
        print(f"Time Gap: {visit['time_gap']} days")
        
        # Print codes for each modality
        for modality in ['diag_code', 'drug_code', 'proc_code', 'lab_code']:
            codes = visit[modality]
            decoded = [index_to_code.get(code, str(code)) for code in codes]
            print(f"{modality.replace('_code', '').capitalize()}: {decoded}")
        
        print(f"Event: {'Died' if visit['event'] else 'Survived'}")

if __name__ == '__main__':


    patient_data, code_to_index = main(dataset_name='mimic_iii', short_ICD=True, proc_digits=3, drug_agg=True, save=True)
    take_a_look(dataset_name='mimic_iii', num_patients=5)
    take_a_look_at_mapping(dataset_name='mimic_iii')
    split_and_see_data(dataset_name='mimic_iii', save=False)
    take_a_look_at_userid(dataset_name='mimic_iii',id=19166)

    patient_data, code_to_index = main(dataset_name='mimic_iv', short_ICD=True, proc_digits=3, drug_agg=True, save=True, icd_version=9)
    take_a_look('mimic_iv', icd_version=9)
    take_a_look_at_mapping('mimic_iv', icd_version=9)
    split_and_see_data(dataset_name='mimic_iv', save=True)
    take_a_look_at_userid(dataset_name='mimic_iv',id=19166)
    split_and_see_data(dataset_name='mimic_iv', save=True, icd_version=9)
    take_a_look_at_userid(dataset_name='mimic_iv',id=18939460, icd_version=9)



