import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle
import random
from functools import lru_cache
from open_clip import create_model_from_pretrained, get_tokenizer
import pandas as pd


class EHRDataloader(Dataset):
    def __init__(self, data_path, code_to_index_path, max_seq_length=512, mask_prob=0.15):
        """
        Args:
            data_path: Path to the split data (e.g., 'mimic_iii_train_ehr.pkl')
            code_to_index_path: Path to the code_to_index mapping file
            max_seq_length: Maximum sequence length to use
            mask_prob: Probability of masking a token for MLM
        """
        self.data = pickle.load(open(data_path, 'rb'))
        with open(code_to_index_path, 'rb') as f:
            self.code_to_index = pickle.load(f)
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob

        # Special tokens - consolidating all special tokens in one place
        self.special_tokens = {
            'pad': 0, 'no_record': 1, 'bos': 2, 'eos': 3, 'sep': 4,
            'diag_start': 5, 'diag_end': 6,
            'drug_start': 7, 'drug_end': 8,
            'proc_start': 9, 'proc_end': 10,
            'lab_start': 11, 'lab_end': 12,
            'mask': 13, 'visit_start': 14, 'visit_end': 15, 'no_next_visit': 16,
        }

        # Pre-filter valid patient IDs based on max sequence length
        self.valid_patient_ids = [
            patient_id for patient_id in self.data.keys()
            if self._is_valid_patient(self.data[patient_id])
        ]

        # Reverse mapping for special tokens
        self.index_to_special = {v: k for k, v in self.special_tokens.items()}

    def __len__(self):
        return len(self.valid_patient_ids)

    def _is_valid_patient(self, patient_data):
        """Check if a patient's data results in a valid sequence within the max length."""
        sequence, _, _ = self.convert_patient_to_sequence(patient_data)
        return len(sequence) <= self.max_seq_length

    def convert_patient_to_sequence(self, patient_data):
        """Convert patient data into a single sequence with visit boundaries."""
        sequence = [self.special_tokens['bos']]
        visit_positions = []  # Track start position of each visit
        visit_times = []     # Track admission times

        for idx, hadm_id in enumerate(patient_data['visit_order']):
            visit = patient_data['visits'][hadm_id]
            visit_positions.append(len(sequence))
            visit_times.append(visit['admittime'])  # Store admission time
            sequence.append(self.special_tokens['visit_start'])
            sequence.extend(visit['diag_code'])
            sequence.extend(visit['drug_code'])
            sequence.extend(visit['proc_code'])
            sequence.extend(visit['lab_code'])
            sequence.append(self.special_tokens['visit_end'])
            if idx < len(patient_data['visit_order']) - 1:
                sequence.append(self.special_tokens['sep'])

        sequence.append(self.special_tokens['eos'])
        return sequence, visit_positions, visit_times

    def create_modality_mask(self, sequence_key):
        """Create mask for intra-modality attention."""
        sequence = torch.tensor(sequence_key)
        seq_len = len(sequence)
        modality_mask = torch.zeros(seq_len, seq_len)

        for i, token in enumerate(sequence):
            if token in [self.special_tokens['diag_start'], self.special_tokens['drug_start'],
                         self.special_tokens['proc_start'], self.special_tokens['lab_start']]:
                end_token = self.special_tokens[f"{list(self.special_tokens.keys())[list(self.special_tokens.values()).index(token)].split('_')[0]}_end"]
                start_idx = i + 1
                end_idx = (sequence[i:] == end_token).nonzero(as_tuple=True)[0][0].item() + i
                modality_mask[start_idx:end_idx, start_idx:end_idx] = 1

        return modality_mask
    
    def create_mlm_labels(self, sequence, visit_positions):
        """Create MLM labels by randomly masking tokens."""
        mlm_labels = torch.full_like(sequence, -100)  # -100 is PyTorch's ignore index
        sequence = sequence.clone()
        
        # Only mask actual medical codes, not special tokens
        special_token_ids = set(self.special_tokens.values())
        maskable_indices = []
        
        # Find maskable indices (actual medical codes between start/end tokens)
        for pos in visit_positions:
            for token in [self.special_tokens['diag_start'],
                         self.special_tokens['drug_start'],
                         self.special_tokens['proc_start'],
                         self.special_tokens['lab_start']]:
                try:
                    start_idx = sequence[pos:].tolist().index(token) + pos + 1
                    end_idx = sequence[start_idx:].tolist().index(
                        token + 1) + start_idx  # Assuming end token is start + 1
                    maskable_indices.extend(range(start_idx, end_idx))
                except:
                    continue
        
        # Randomly mask tokens
        num_to_mask = int(len(maskable_indices) * self.mask_prob)
        masked_indices = random.sample(maskable_indices, num_to_mask)
        
        for idx in masked_indices:
            mlm_labels[idx] = sequence[idx]  # Record original token
            sequence[idx] = self.special_tokens['mask']  # Replace with mask token
            
        return sequence, mlm_labels
    
    def create_next_visit_labels(self, sequence, visit_positions):
        """Create labels for next visit prediction."""
        next_visit_labels = torch.full_like(sequence, -100)  # Initialize with ignore index
        
        # If there's only one visit, label the start of the sequence with <visit_end>
        if len(visit_positions) <= 1:
            next_visit_labels[0] = self.special_tokens['no_next_visit']
            return next_visit_labels
        
        # Otherwise, label tokens for the next visit
        for i in range(len(visit_positions) - 1):
            next_visit_start = visit_positions[i + 1]
            next_visit_end = (visit_positions[i + 2] if i + 2 < len(visit_positions) 
                            else len(sequence))
            next_visit_labels[next_visit_start:next_visit_end] = sequence[next_visit_start:next_visit_end]
        
        return next_visit_labels
    
    def create_modality_indices(self, input_ids):
        """
        Create modality indices based on modality-specific start and end tokens.
        """
        modality_indices = torch.zeros_like(input_ids)
        
        # Define modality IDs
        modality_map = {
            'diag': 1,
            'drug': 2,
            'proc': 3,
            'lab': 4
        }
        
        for modality, start_token in [
            ('diag', self.special_tokens['diag_start']),
            ('drug', self.special_tokens['drug_start']),
            ('proc', self.special_tokens['proc_start']),
            ('lab', self.special_tokens['lab_start']),
        ]:
            end_token = self.special_tokens[f'{modality}_end']
            modality_id = modality_map[modality]
            
            # Find positions of start and end tokens
            # For 1D tensor, nonzero returns a 1D tensor of positions
            starts = (input_ids == start_token).nonzero().squeeze(-1)
            ends = (input_ids == end_token).nonzero().squeeze(-1)
            
            # Ensure we have matching pairs of starts and ends
            for start, end in zip(starts, ends):
                # Fill the range with modality ID
                modality_indices[start+1:end] = modality_id

        return modality_indices

    def __getitem__(self, idx):
        patient_id = self.valid_patient_ids[idx]
        patient_data = self.data[patient_id]

        # sequence, visit_positions = self.convert_patient_to_sequence(patient_data)
        sequence, visit_positions, visit_times = self.convert_patient_to_sequence(patient_data)
        sequence = torch.tensor(sequence)

        time_gaps = torch.zeros(len(sequence))
        for i, pos in enumerate(visit_positions[:-1]):  # Note: iterate until second-to-last position
            # Convert timestamps to days and calculate gaps
            current_time = pd.to_datetime(visit_times[i])
            next_time = pd.to_datetime(visit_times[i+1])
            gap_days = (next_time - current_time).days
                
            # Fill the gap at the start position of the current visit
            time_gaps[pos] = gap_days

        # Create attention masks
        attention_mask = torch.ones(len(sequence))  # For padding
        modality_mask = self.create_modality_mask(tuple(sequence.tolist()))

        # Create MLM and next-visit labels
        mlm_sequence, mlm_labels = self.create_mlm_labels(sequence, visit_positions)
        next_visit_labels = self.create_next_visit_labels(sequence, visit_positions)

        # Create modality indices for positional embeddings
        modality_indices = self.create_modality_indices(mlm_sequence)

        last_visit_id = patient_data['visit_order'][-1]
        event_flag = int(patient_data['visits'][last_visit_id]['event'])

        # Pad all tensors
        padding_length = self.max_seq_length - len(sequence)
        if padding_length > 0:
            mlm_sequence = F.pad(mlm_sequence, (0, padding_length), value=self.special_tokens['pad'])
            attention_mask = F.pad(attention_mask, (0, padding_length), value=0)
            modality_mask = F.pad(modality_mask, (0, padding_length, 0, padding_length), value=0)
            mlm_labels = F.pad(mlm_labels, (0, padding_length), value=-100)
            next_visit_labels = F.pad(next_visit_labels, (0, padding_length), value=-100)
            modality_indices = F.pad(modality_indices, (0, padding_length), value=0)
            time_gaps = F.pad(time_gaps, (0, padding_length), value=0)


        return {
            'input_ids': mlm_sequence,
            'attention_mask': attention_mask,
            'modality_mask': modality_mask,
            'mlm_labels': mlm_labels,
            'next_visit_labels': next_visit_labels,
            'modality_indices': modality_indices,
            'patient_id': patient_id,
            'event_flag': torch.tensor(event_flag),
            'time_gaps': time_gaps
        }


def create_dataloader(data_path, code_to_index_path, batch_size=32, max_seq_length=512, 
                     num_workers=4, mask_prob=0.0):
    """Create a DataLoader with the specified parameters."""
    dataset = EHRDataloader(data_path, code_to_index_path, max_seq_length, mask_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                     num_workers=num_workers, pin_memory=True)

############################################################################################################
# Test for EHR dataloader
############################################################################################################

if __name__ == '__main__':
    
    # Create dataloader
    loader = create_dataloader(
        data_path='path_to_your_data.pkl',
        code_to_index_path='path_to_your_index.pkl',
        batch_size=32,
        max_seq_length=512,
        num_workers=4,
        mask_prob=0.15
    )

    # Get a batch
    batch = next(iter(loader))
    print("\nBatch Shapes:")
    print("Input shape:", batch['input_ids'].shape)
    print("Attention mask shape:", batch['attention_mask'].shape)
    print("Modality mask shape:", batch['modality_mask'].shape)
    print("MLM labels shape:", batch['mlm_labels'].shape)
    print("Next visit labels shape:", batch['next_visit_labels'].shape)
    print("Modality indices shape:", batch['modality_indices'].shape)

    # Print detailed sample for first patient in batch
    print("\nDetailed Sample for First Patient:")
    print("Patient ID:", batch['patient_id'][2])
    patient_seq = batch['input_ids'][2]
    attention_mask = batch['attention_mask'][2]
    
    # Only look at non-padded tokens
    seq_len = attention_mask.sum().int().item()
    patient_seq = patient_seq[:seq_len]
    
    print("\nTokenized Sequence with Special Token Labels:")
    for token_id in patient_seq:
        token_id = token_id.item()
        # Check if it's a special token
        if token_id in loader.dataset.index_to_special:
            print(f"<{loader.dataset.index_to_special[token_id]}>", end=' ')
        else:
            # It's a medical code
            code = loader.dataset.code_to_index.get(token_id, f"CODE_{token_id}")
            print(code, end=' ')
    print("\n")

    # Print sample of actual medical codes between special tokens
    print("\nVisit Structure:")
    current_visit = 0
    current_modality = None
    medical_codes = []

    for i, token_id in enumerate(patient_seq):
        token_id = token_id.item()
        
        if token_id == loader.dataset.special_tokens['visit_start']:
            current_visit += 1
            print(f"\nVisit {current_visit}:")
            
        elif token_id == loader.dataset.special_tokens['diag_start']:
            print("  Diagnoses:", end=' ')
            current_modality = 'diag'
            medical_codes = []
            
        elif token_id == loader.dataset.special_tokens['drug_start']:
            print("\n  Medications:", end=' ')
            current_modality = 'med'
            medical_codes = []
            
        elif token_id == loader.dataset.special_tokens['proc_start']:
            print("\n  Procedures:", end=' ')
            current_modality = 'proc'
            medical_codes = []
            
        elif token_id == loader.dataset.special_tokens['lab_start']:
            print("\n  Lab Tests:", end=' ')
            current_modality = 'lab'
            medical_codes = []
            
        elif token_id == loader.dataset.special_tokens['mask']:
            # Handle masked tokens
            medical_codes.append('<mask>')
            
        elif token_id not in loader.dataset.special_tokens.values():
            # This is an actual medical code
            code = {v: k for k, v in loader.dataset.code_to_index.items()}.get(token_id, f"CODE_{token_id}")
            medical_codes.append(code)
            
        elif token_id in [loader.dataset.special_tokens['diag_end'], 
                        loader.dataset.special_tokens['drug_end'],
                        loader.dataset.special_tokens['proc_end'], 
                        loader.dataset.special_tokens['lab_end']]:
            if not medical_codes:  # If no codes (and no masks), it's a no_record
                print("<no_record>", end='')
            else:
                print(", ".join(medical_codes), end='')
            current_modality = None

    print("*"*50)
    #print out the 2nd patient
    print("\nDetailed Sample for 2nd Patient:")
    print("Patient ID:", batch['patient_id'][1])
    patient_seq = batch['input_ids'][1]
    attention_mask = batch['attention_mask'][1]

    # Only look at non-padded tokens
    seq_len = attention_mask.sum().int().item()
    patient_seq = patient_seq[:seq_len]

    print("\nTokenized Sequence with Special Token Labels:")

    for token_id in patient_seq:
        token_id = token_id.item()
        # Check if it's a special token
        if token_id in loader.dataset.index_to_special:
            print(f"<{loader.dataset.index_to_special[token_id]}>", end=' ')
        else:
            # It's a medical code
            code = loader.dataset.code_to_index.get(token_id, f"CODE_{token_id}")
            print(code, end=' ')
    print("\n")

    # Print sample of actual medical codes between special tokens

    print("\nVisit Structure:")

    current_visit = 0   
    current_modality = None
    medical_codes = []

    for i, token_id in enumerate(patient_seq):
        token_id = token_id.item()

        if token_id == loader.dataset.special_tokens['visit_start']:
            current_visit += 1
            print(f"\nVisit {current_visit}:")

        elif token_id == loader.dataset.special_tokens['diag_start']:
            print("  Diagnoses:", end=' ')
            current_modality = 'diag'
            medical_codes = []

        elif token_id == loader.dataset.special_tokens['drug_start']:
            print("\n  Medications:", end=' ')
            current_modality = 'med'
            medical_codes = []

        elif token_id == loader.dataset.special_tokens['proc_start']:
            print("\n  Procedures:", end=' ')
            current_modality = 'proc'
            medical_codes = []

        elif token_id == loader.dataset.special_tokens['lab_start']:
            print("\n  Lab Tests:", end=' ')
            current_modality = 'lab'
            medical_codes = []

        elif token_id == loader.dataset.special_tokens['mask']:
            # Handle masked tokens
            medical_codes.append('<mask>')

        elif token_id not in loader.dataset.special_tokens.values():
            # This is an actual medical code
            code = {v: k for k, v in loader.dataset.code_to_index.items()}.get(token_id, f"CODE_{token_id}")
            medical_codes.append(code)

        elif token_id in [loader.dataset.special_tokens['diag_end'],
                        loader.dataset.special_tokens['drug_end'],
                        loader.dataset.special_tokens['proc_end'],
                        loader.dataset.special_tokens['lab_end']]:
            if not medical_codes:
                print("<no_record>", end='')
            else:
                print(", ".join(medical_codes), end='')
            current_modality = None
    print("*"*50)