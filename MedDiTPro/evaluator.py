"""LPL / MPL evaluation for EHRModel (used by runner.py for validation and test).

Both metrics are imputation perplexities, computed per modality on the evaluation split:

- MPL (cross-modality imputation): for each visit t and each modality with recorded codes, the codes of
  that modality in visit t are replaced by the [mask] token and the patient's later visits are removed.
  The model sees the earlier visits and the other modalities of visit t.
- LPL (longitudinal imputation): for each visit t after the first, all codes of visit t are replaced by
  [mask] and the later visits are removed. The model sees the earlier visits and the layout of visit t
  (one [mask] per code, and which modalities have no record), but none of its codes.

For every hidden code, the model's output at its position gives a softmax over the codes of that
modality, and NLL = -log p(code).
    LPL_m = median over the hidden codes of exp(NLL)   (per-code perplexity)
    MPL_m = exp(mean over the hidden codes of NLL)
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

MODALITIES = ['diag', 'drug', 'proc', 'lab']


class InVisitEvaluator:
    def __init__(self, model, device=None, batch_size=None, max_patients=None):
        """
        Args:
            model: EHRModel
            device: Device the model runs on (default: the model's device)
            batch_size: Number of imputation tasks per forward pass (default: the dataloader's batch size)
            max_patients: Evaluate only the first max_patients patients of the split (default: all)
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.batch_size = batch_size
        self.max_patients = max_patients

        config = model.config
        self.special_tokens = config.special_tokens
        num_special = len(config.special_tokens)
        self.vocab_offset = {
            'diag': num_special,
            'drug': num_special + config.diag_vocab_size,
            'proc': num_special + config.diag_vocab_size + config.drug_vocab_size,
            'lab': num_special + config.diag_vocab_size + config.drug_vocab_size + config.proc_vocab_size,
        }

    def _tasks(self, patient_data):
        """List the (kind, target visit, hidden modalities) imputation tasks of one patient."""
        tasks = []
        for t, hadm_id in enumerate(patient_data['visit_order']):
            visit = patient_data['visits'][hadm_id]
            recorded = [m for m in MODALITIES if self._codes(visit, m)]
            for modality in recorded:
                tasks.append(('mpl', t, (modality,)))
            if t > 0 and recorded:
                tasks.append(('lpl', t, tuple(recorded)))
        return tasks

    def _codes(self, visit, modality):
        """Codes of one modality block, without the start/end markers and the no_record token."""
        block = visit[f'{modality}_code'][1:-1]
        return [code for code in block if code != self.special_tokens['no_record']]

    def _encode(self, dataset, patient_data, target_visit, hidden_modalities):
        """Build one model input with the hidden blocks masked; returns the tensors and the hidden codes."""
        visit_order = patient_data['visit_order'][:target_visit + 1]
        sequence, visit_positions, _ = dataset.convert_patient_to_sequence(
            {'visits': patient_data['visits'], 'visit_order': visit_order})

        # Walk the target visit: [visit_start] diag_block drug_block proc_block lab_block [visit_end]
        visit = patient_data['visits'][visit_order[target_visit]]
        position = visit_positions[target_visit] + 1
        hidden = []  # (position, modality, code)
        for modality in MODALITIES:
            block = visit[f'{modality}_code']
            if modality in hidden_modalities:
                for offset, code in enumerate(block[1:-1]):
                    if code != self.special_tokens['no_record']:
                        hidden.append((position + 1 + offset, modality, code))
            position += len(block)
        for pos, _, _ in hidden:
            sequence[pos] = self.special_tokens['mask']

        input_ids = torch.tensor(sequence)
        attention_mask = torch.ones(len(sequence))
        modality_mask = dataset.create_modality_mask(tuple(sequence))
        modality_indices = dataset.create_modality_indices(input_ids)

        padding_length = dataset.max_seq_length - len(sequence)
        if padding_length > 0:
            input_ids = F.pad(input_ids, (0, padding_length), value=self.special_tokens['pad'])
            attention_mask = F.pad(attention_mask, (0, padding_length), value=0)
            modality_mask = F.pad(modality_mask, (0, padding_length, 0, padding_length), value=0)
            modality_indices = F.pad(modality_indices, (0, padding_length), value=0)
        return (input_ids, attention_mask, modality_mask, modality_indices), hidden

    def _score(self, batch, nll):
        """Run the model on a batch of tasks and append the NLL of every hidden code to nll[kind][modality]."""
        inputs, hidden = zip(*[(item['inputs'], item['hidden']) for item in batch])
        input_ids, attention_mask, modality_mask, modality_indices = (
            torch.stack(tensors).to(self.device) for tensors in zip(*inputs))
        predictions, _ = self.model(
            input_ids=input_ids,
            seq_mask=attention_mask,
            modality_mask=modality_mask,
            modality_indices=modality_indices,
            timesteps=None
        )
        for modality in MODALITIES:
            rows, positions, codes, kinds = [], [], [], []
            for b, (item, item_hidden) in enumerate(zip(batch, hidden)):
                for pos, m, code in item_hidden:
                    if m == modality:
                        rows.append(b)
                        positions.append(pos)
                        codes.append(code - self.vocab_offset[modality])
                        kinds.append(item['kind'])
            if not rows:
                continue
            logits = predictions[modality]['logits'][rows, positions].float()
            local_codes = torch.tensor(codes, device=logits.device)
            assert (local_codes >= 0).all() and (local_codes < logits.size(-1)).all(), \
                f"Code ids outside the {modality} vocabulary range."
            code_nll = -F.log_softmax(logits, dim=-1).gather(1, local_codes.unsqueeze(1)).squeeze(1)
            for kind, value in zip(kinds, code_nll.tolist()):
                nll[kind][modality].append(value)

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Args:
            dataloader: DataLoader (or Dataset) over an EHRDataloader split
        Returns:
            dict with lpl_{diag,drug,proc,lab} and mpl_{diag,drug,proc,lab}; inf when a modality has no codes
        """
        dataset = getattr(dataloader, 'dataset', dataloader)
        batch_size = self.batch_size or getattr(dataloader, 'batch_size', None) or 16
        patient_ids = dataset.valid_patient_ids
        if self.max_patients is not None:
            patient_ids = patient_ids[:self.max_patients]

        was_training = self.model.training
        self.model.eval()
        nll = {kind: {m: [] for m in MODALITIES} for kind in ['lpl', 'mpl']}
        try:
            batch = []
            for patient_id in tqdm(patient_ids, desc='Evaluating'):
                patient_data = dataset.data[patient_id]
                for kind, target_visit, hidden_modalities in self._tasks(patient_data):
                    inputs, hidden = self._encode(dataset, patient_data, target_visit, hidden_modalities)
                    batch.append({'kind': kind, 'inputs': inputs, 'hidden': hidden})
                    if len(batch) == batch_size:
                        self._score(batch, nll)
                        batch = []
            if batch:
                self._score(batch, nll)
        finally:
            self.model.train(was_training)

        metrics = {}
        for modality in MODALITIES:
            lpl_nll = np.array(nll['lpl'][modality])
            mpl_nll = np.array(nll['mpl'][modality])
            metrics[f'lpl_{modality}'] = float(np.median(np.exp(lpl_nll))) if lpl_nll.size else float('inf')
            metrics[f'mpl_{modality}'] = float(np.exp(mpl_nll.mean())) if mpl_nll.size else float('inf')
        return metrics
