from dataclasses import dataclass

@dataclass
class MIMIC3_ModelConfig:
    # Basic settings
    hidden_size: int = 768
    max_position_embeddings: int = 512  
    hidden_dropout_prob: float = 0.1
    
    # Feed-forward network multiplier for each transformer
    shared_ffn_multiplier: int = 2
    modality_ffn_multiplier: int = 2
    cross_modal_ffn_multiplier: int = 3
    denoise_ffn_multiplier: int = 4

    # Time prompt transformer settings
    time_num_attention_heads: int = 16
    time_num_hidden_layers: int = 8
    time_ffn_multiplier: int = 4
    num_time_tokens: int = 1
    
    @property
    def time_intermediate_size(self) -> int:
        return self.hidden_size * self.time_ffn_multiplier
    
    # Shared transformer settings
    shared_num_attention_heads: int = 8
    shared_num_hidden_layers: int = 4
    
    # Modality-specific prompt transformer settings
    modality_num_attention_heads: int = 8
    modality_num_hidden_layers: int = 6
    
    # Cross-modal transformer settings
    cross_modal_num_attention_heads: int = 12
    cross_modal_num_hidden_layers: int = 8
    
    # Denoising transformer settings
    denoise_num_attention_heads: int = 16
    denoise_num_hidden_layers: int = 12
    
    @property
    def shared_intermediate_size(self) -> int:
        return self.hidden_size * self.shared_ffn_multiplier
        
    @property
    def modality_intermediate_size(self) -> int:
        return self.hidden_size * self.modality_ffn_multiplier
        
    @property
    def cross_modal_intermediate_size(self) -> int:
        return self.hidden_size * self.cross_modal_ffn_multiplier
        
    @property
    def denoise_intermediate_size(self) -> int:
        return self.hidden_size * self.denoise_ffn_multiplier
    
    # Vocab sizes for each modality
    diag_vocab_size: int = 1071
    drug_vocab_size: int = 1476
    proc_vocab_size: int = 711
    lab_vocab_size: int = 710
    total_vocab_size: int = None
    # Prompt related
    num_modality_tokens: int = 20  # number of prompt tokens per modality
    num_modalities: int = 4  # diag, drug, proc, lab

    # Special tokens
    special_tokens: dict = None
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                # General special tokens
                'pad': 0,         # Padding token
                'no_record': 1,   # No record for a modality
                'bos': 2,         # Beginning of sequence
                'eos': 3,         # End of sequence
                'sep': 4,         # Separator between different parts
                
                # Modality-specific markers
                'diag_start': 5,  # Start of diagnosis codes
                'diag_end': 6,    # End of diagnosis codes
                'drug_start': 7,   # Start of medication codes
                'drug_end': 8,     # End of medication codes
                'proc_start': 9,  # Start of procedure codes
                'proc_end': 10,   # End of procedure codes
                'lab_start': 11,  # Start of lab codes
                'lab_end': 12,     # End of lab codes
                'mask': 13,       # Masking token for MLM
                'visit_start': 14,  # Start of visit marker
                'visit_end': 15,    # End of visit marker
                'no_next_visit': 16 # No next visit marker
            }
    # total vocab size for all modalities and special tokens
        if self.total_vocab_size is None:
            self.total_vocab_size = self.diag_vocab_size + self.drug_vocab_size + self.proc_vocab_size + self.lab_vocab_size + len(self.special_tokens)

@dataclass
class MIMIC3_TrainingConfig:
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    mask_prob = 0.0
    # Diffusion specific
    num_diffusion_steps: int = 1000
    min_beta: float = 1e-4
    max_beta: float = 0.02

    # Focal loss settings
    alpha: float = 0.75
    gamma: float = 5.0

    # Loss weights
    diff_weight: float = 1.0
    pred_weight: float = 1.0
    time_loss_weight: float = 1e-6
    
    # Optimizer settings
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    adam_betas: tuple = (0.9, 0.999)
    
    # Training regime
    gradient_accumulation_steps: int = 16
    use_amp: bool = True  # Automatic Mixed Precision
    seed: int = 42
    
    # Device
    device: str = 'cuda'
    num_workers: int = 4  # for dataloader
    
    # Logging & Checkpointing
    log_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 5000
    main_dir: str = './mainRes'
    experiment_name: str = 'mimic3'
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.0001
    
    # Data paths
    mimiciii_mapping_path: str = '/path_to_your/mimic_iii_code_to_index.pkl'
    mimiciii_train_path: str = '/path_to_your/mimic_iii_train_ehr.pkl'
    mimiciii_val_path: str = '/path_to_your/mimic_iii_val_ehr.pkl'
    mimiciii_test_path: str = '/path_to_your/mimic_iii_test_ehr.pkl'
    mimiciii_toy_path: str = '/path_to_your/mimic_iii_toy_ehr.pkl'
    
    # Debug mode
    debug: bool = False  # if True, use toy dataset and minimal settings
    
    def __post_init__(self):
        self.checkpoint_dir = f"{self.main_dir}/mimic3_checkpoints" 
        self.tensorboard_dir = f"{self.main_dir}/mimic3_runs"      
        self.log_dir = f"{self.main_dir}/mimic3_logs"    
        if self.debug:
            # Override settings for quick debugging
            self.batch_size = 4
            self.num_epochs = 2
            self.num_diffusion_steps = 10
            self.log_steps = 1
            self.eval_steps = 5
            self.save_steps = 10
            self.experiment_name = 'debug_run'
