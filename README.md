# MedDiTPro

Code for **MedDiTPro: A Prompt-Guided Diffusion Transformer for Multimodal Longitudinal Medical Data Synthesis**
(Yuan Zhong, Xiaochen Wang, Jiaqi Wang, Xiaokun Zhang, Fenglong Ma; KDD 2025,
[doi:10.1145/3711896.3737045](https://doi.org/10.1145/3711896.3737045)).

## Repository layout

```
README.md
requirements.txt
MedDiTPro/
├── process_mimic_ehr.py    MIMIC-III / MIMIC-IV preprocessing
├── seqStyleDataset.py      dataset and dataloader
├── model.py                MedDiTPro model (EHRModel)
├── runner.py               training and evaluation entry point
├── evaluator.py            LPL / MPL evaluation
├── config_mimic3.py        configuration, MIMIC-III
└── config_mimic4_icd9.py   configuration, MIMIC-IV (ICD-9)
```

## Installation

```bash
git clone https://github.com/zhongyuan1996/MedDiTPro.git
cd MedDiTPro
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
# For GPU training, first install a CUDA build of PyTorch: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Data

MIMIC-III v1.4 and MIMIC-IV are distributed by PhysioNet under credentialed access: you need a credentialed
PhysioNet account, the required human-subjects training, and a signed data use agreement.

- MIMIC-III v1.4: https://physionet.org/content/mimiciii/1.4/
- MIMIC-IV: https://physionet.org/content/mimiciv/ (`hosp` module)

No data is included in this repository. Keep the raw and processed data out of version control.

The preprocessing reads these tables (`.csv` or `.csv.gz`):

```
MIMIC-III                     MIMIC-IV (hosp)
├── ADMISSIONS.csv            ├── admissions.csv
├── PATIENTS.csv              ├── patients.csv
├── DIAGNOSES_ICD.csv         ├── diagnoses_icd.csv
├── PROCEDURES_ICD.csv        ├── procedures_icd.csv
├── PRESCRIPTIONS.csv         ├── prescriptions.csv
└── LABEVENTS.csv             └── labevents.csv
```

## Preprocessing

All commands below are run from the `MedDiTPro/` package folder.

```bash
cd MedDiTPro    # from the repository root

# MIMIC-III v1.4
python process_mimic_ehr.py --dataset mimic_iii --raw_dir /path/to/mimiciii/1.4 --out_dir ../processed

# MIMIC-IV
python process_mimic_ehr.py --dataset mimic_iv --raw_dir /path/to/mimiciv/3.0/hosp --out_dir ../processed
```

| Flag | Meaning |
|---|---|
| `--dataset` | `mimic_iii` or `mimic_iv` (required) |
| `--raw_dir` | folder with the raw CSV files (default: `mimiciii/1.4/` or `mimiciv/3.0/hosp/`) |
| `--out_dir` | output folder (default: `.`) |
| `--split_dir` | MIMIC-III only: folder with predefined split files `{train,val,test,toy}_3digmimic.csv` (optional) |
| `--icd_version` | MIMIC-IV only: ICD version to keep (default: `9`) |

The output folder then holds `mimic_iii_*` / `mimic_iv_icd9_*` files: the code-to-index mapping and the
train / val / test / toy splits that `runner.py` reads.

## Training

```bash
python runner.py --dataset mimic3      --data_dir ../processed --main_dir ../mainRes
python runner.py --dataset mimic4_icd9 --data_dir ../processed --main_dir ../mainRes

# Short run on the toy split
python runner.py --dataset mimic3 --data_dir ../processed --main_dir ../debugRes --debug

# Ablation run
python runner.py --dataset mimic3 --data_dir ../processed --main_dir ../mainRes --ablation ab1_shared
```

| Flag | Meaning |
|---|---|
| `--dataset` | `mimic3` (default) or `mimic4_icd9` |
| `--data_dir` | folder with the preprocessed files |
| `--main_dir` | output folder (default: `./mainRes`) |
| `--device` | for example `cuda`, `cuda:1` or `cpu` (default: `cuda`) |
| `--num_workers` | DataLoader workers (default: 4) |
| `--debug` | short run on the toy split |
| `--resume_from` | absolute path of a checkpoint to resume from |
| `--ablation` | `ab1_shared`, `ab2_modality`, `ab3_cross` or `ab4_diffusion` |

| `--ablation` | Paper (Table 6) |
|---|---|
| `ab1_shared` | AS-1 |
| `ab2_modality` | AS-2 |
| `ab3_cross` | AS-3 |
| `ab4_diffusion` | AS-4 |

Hyper-parameters are in `config_<dataset>.py`.

### Outputs

```
<main_dir>/
├── <dataset>_checkpoints/   best_model.pt, test_results.pt
├── <dataset>_logs/          training log
└── <dataset>_runs/          TensorBoard event files
```

With `--ablation`, results go to an `ablation_<dataset>_<name>/` subfolder and the test results to
`ablation_test_results.pt`. Load result files with `torch.load(path)`.

## Evaluation

Validation and test LPL / MPL are computed by `runner.py` during training. To evaluate a saved checkpoint on its
own (run from `MedDiTPro/`):

```python
import torch
from runner import get_config, set_data_dir, set_vocab_sizes
from model import EHRModel
from seqStyleDataset import create_dataloader
from evaluator import InVisitEvaluator

model_config, training_config = get_config('mimic3', device='cuda')
set_data_dir(training_config, '../processed')
set_vocab_sizes(model_config, training_config)

model = EHRModel(model_config).to(training_config.device)
checkpoint = torch.load('../mainRes/mimic3_checkpoints/best_model.pt',
                        map_location=training_config.device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_loader = create_dataloader(training_config.mimiciii_test_path, training_config.mimiciii_mapping_path,
                                batch_size=16, max_seq_length=model_config.max_position_embeddings, num_workers=0)
print(InVisitEvaluator(model, training_config.device).evaluate(test_loader))
```

## Hardware

The experiments in the paper were run on one NVIDIA RTX A6000 GPU.

## Citation

```bibtex
@inproceedings{zhong2025medditpro,
  title     = {MedDiTPro: A Prompt-Guided Diffusion Transformer for Multimodal Longitudinal Medical Data Synthesis},
  author    = {Zhong, Yuan and Wang, Xiaochen and Wang, Jiaqi and Zhang, Xiaokun and Ma, Fenglong},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages     = {4086--4097},
  year      = {2025},
  doi       = {10.1145/3711896.3737045}
}
```

Questions: please open a GitHub issue.
