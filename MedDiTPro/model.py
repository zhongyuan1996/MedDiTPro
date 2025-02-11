import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import repeat
import math
from typing import List

def sequentially_sample_unique_tokens(logits: torch.Tensor, num_needed: int, temperature: float = 5.0) -> List[int]:
    """
    Sample unique tokens from logits without replacement, using softmax with temperature.
    For sequential generation, we use a higher temperature (less random) and no additional noise.
    
    Args:
        logits: Tensor of shape [num_positions, vocab_size]
        num_needed: Number of unique tokens needed
        temperature: Temperature for softmax (higher = more deterministic)
            
    Returns:
        List of unique token indices
    """
    unique_tokens = set()
    all_tokens = []
    
    # For each position
    for pos_logits in logits:
        # Create a mask for already chosen tokens
        mask = torch.ones_like(pos_logits)
        for token in unique_tokens:
            mask[token] = 0
            
        # Apply temperature and masking
        modified_logits = (pos_logits / temperature) * mask
        modified_logits[~mask.bool()] = float('-inf')  # Mask out chosen tokens
        
        # Convert to probabilities
        probs = torch.softmax(modified_logits, dim=-1)
        
        # Sample from the distribution
        token = torch.multinomial(probs, num_samples=1).item()
        
        unique_tokens.add(token)
        all_tokens.append(token)
        
        if len(all_tokens) >= num_needed:
            break
                
    return all_tokens[:num_needed]

def sample_unique_tokens(logits: torch.Tensor, num_needed: int, temperature: float = 0.2, noise_scale: float = 0.5) -> List[int]:
    """
    Sample unique tokens from logits without replacement, using softmax with temperature and noise.
    
    Args:
        logits: Tensor of shape [num_positions, vocab_size]
        num_needed: Number of unique tokens needed
        temperature: Temperature for softmax (higher = more random, lower = more deterministic)
        noise_scale: Scale of Gumbel noise to add to logits
            
    Returns:
        List of unique token indices
    """
    unique_tokens = set()
    all_tokens = []
    
    # For each position
    for pos_logits in logits:
        # Create a mask for already chosen tokens
        mask = torch.ones_like(pos_logits)
        for token in unique_tokens:
            mask[token] = 0
            
        # Add Gumbel noise for increased randomness
        noise = torch.empty_like(pos_logits).uniform_(0, 1)
        noise = -torch.log(-torch.log(noise)) * noise_scale
        
        # Apply temperature, noise, and masking
        modified_logits = (pos_logits / temperature + noise) * mask
        modified_logits[~mask.bool()] = float('-inf')  # Mask out chosen tokens
        
        # Convert to probabilities
        probs = torch.softmax(modified_logits, dim=-1)
        
        # Sample from the distribution
        token = torch.multinomial(probs, num_samples=1).item()
        
        unique_tokens.add(token)
        all_tokens.append(token)
        
        if len(all_tokens) >= num_needed:
            break
                
    return all_tokens[:num_needed]

class DenoiseTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.denoise_num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.denoise_num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.denoise_intermediate_size),
            nn.GELU(),
            nn.Linear(config.denoise_intermediate_size, config.hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, prompts, prompt_attention_mask=None):

        # Self attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x
        
        # Cross attention with prompts
        residual = x
        x = self.norm2(x)
        
        # Reshape attention mask if provided
        if prompt_attention_mask is not None:
            mask = prompt_attention_mask[0].float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        else:
            mask = None
            print("No attention mask provided")
            
        x, _ = self.cross_attn(
            query=x,
            key=prompts,
            value=prompts,
            attn_mask=mask
        )
        x = residual + x
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x
        
        return x
    
class LatentDiffusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Denoising transformer layers with cross attention
        self.denoise_layers = nn.ModuleList([
            DenoiseTransformerLayer(config)
            for _ in range(config.denoise_num_hidden_layers)
        ])
    
    def get_noise_schedule(self, t, beta_start=1e-4, beta_end=0.02):
        """Linear noise schedule."""
        beta_t = beta_start + t * (beta_end - beta_start)
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        return alpha_bar_t.sqrt(), (1 - alpha_bar_t).sqrt()
    
    def add_noise(self, latents, t, modality_mask, special_tokens_mask):
        """Add noise to latent representations."""
        device = latents.device
        batch_size = latents.shape[0]
        noised_latents = latents.clone()
        
        for modality, time in t.items():
            current_mask = modality_mask[modality]
            if current_mask.any():
                sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_noise_schedule(time)
                noise = torch.randn_like(latents)
                noised_latents_current = sqrt_alpha_bar.view(-1, 1, 1) * latents + sqrt_one_minus_alpha_bar.view(-1, 1, 1) * noise
                noised_latents = torch.where(current_mask.unsqueeze(-1), noised_latents_current, noised_latents)
        
        return noised_latents
    
    def denoise(self, noised_latents, t, modality_mask, special_tokens_mask, modality_prompts, cross_prompt):
        """Denoise all modalities simultaneously with modality-specific and global prompt conditioning."""
        batch_size = noised_latents.shape[0]
        device = noised_latents.device
        
        
        # Time conditioning (keeping the same)
        time_cond = torch.zeros_like(noised_latents)
        for modality, time in t.items():
            time_embed = get_timestep_embedding(time, self.config.hidden_size)
            time_embed = self.time_proj(time_embed)
            modality_tokens = modality_mask[modality]
            time_cond = torch.where(
                modality_tokens.unsqueeze(-1),
                time_embed.view(-1, 1, self.config.hidden_size),
                time_cond
            )
        hidden_states = noised_latents + time_cond
        
        # Create attention mask with debugging prints
        total_prompt_length = sum(p.size(1) for p in modality_prompts.values()) + cross_prompt.size(1)
        
        combined_mask = torch.zeros((batch_size, noised_latents.size(1), total_prompt_length), device=device)
        current_prompt_idx = 0
        
        # Handle modality-specific prompts
        all_prompts = []
        for modality, prompt in modality_prompts.items():
            all_prompts.append(prompt)
            prompt_length = prompt.size(1)
            
            prompt_section = torch.zeros((batch_size, noised_latents.size(1), prompt_length), device=device)
            prompt_section[modality_mask[modality]] = 1
            
            combined_mask[:, :, current_prompt_idx:current_prompt_idx + prompt_length] = prompt_section
            current_prompt_idx += prompt_length
        
        # Add cross prompt section
        all_prompts.append(cross_prompt)
        combined_mask[:, :, -cross_prompt.size(1):] = 1
        
        # Concatenate prompts
        combined_prompts = torch.cat(all_prompts, dim=1)
        assert combined_prompts.size(1) == total_prompt_length, "Prompt length mismatch!"
        
        # Process through layers
        for layer in self.denoise_layers:
            hidden_states = layer(hidden_states, combined_prompts, combined_mask)
        
        return hidden_states


def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param embedding_dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x embedding_dim] Tensor of positional embeddings.
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class promptTransformerBlock(nn.Module):
    def __init__(self, config, is_cross_modal=False):
        super(promptTransformerBlock, self).__init__()
        self.config = config

        num_heads = config.cross_modal_num_attention_heads if is_cross_modal else config.modality_num_attention_heads
        num_layers = config.cross_modal_num_hidden_layers if is_cross_modal else config.modality_num_hidden_layers
        intermediate_size = config.cross_modal_intermediate_size if is_cross_modal else config.modality_intermediate_size
        
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.modality_prompt = nn.Parameter(torch.randn(1, config.num_modality_tokens, config.hidden_size))
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        prompts = repeat(self.modality_prompt, '1 n d -> b n d', b=batch_size)
        hidden_states = torch.cat([prompts, hidden_states], dim=1)

        # Usually attention_mask is None
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (self.modality_prompt.size(1), 0), value=1)
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))


        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)

        # Return both the prompt and the hidden states
        return hidden_states[:, self.modality_prompt.size(1):, :], hidden_states[:, :self.modality_prompt.size(1), :]


class regTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.shared_num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.shared_intermediate_size),
            nn.GELU(),
            nn.Linear(config.shared_intermediate_size, config.hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states, modality_mask=None):
        # Layer norm and residual connection for self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        if modality_mask is not None:
            batch_size, seq_len, _ = modality_mask.size()
            
            # Add diagonal entries to the modality mask
            diagonal_mask = torch.eye(seq_len, device=modality_mask.device).unsqueeze(0).repeat(batch_size, 1, 1)
            modality_mask = modality_mask.bool() | diagonal_mask.bool()
            
            # Expand mask for multi-head attention
            attention_mask = modality_mask.unsqueeze(1).repeat(1, self.config.shared_num_attention_heads, 1, 1)
            attention_mask = attention_mask.view(
                -1, attention_mask.size(-2), attention_mask.size(-1)
            )
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, -1e5
            )
        else:
            attention_mask = None
        
        # Self-attention
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        hidden_states = residual + hidden_states

        # Layer norm and residual connection for feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class EHRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.total_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size) #not used for now

        # Modality positional embeddings
        self.modality_embedding = nn.Embedding(config.num_modalities + 1, config.hidden_size)

        # Shared parameter layer
        self.shared_transformer = regTransformerBlock(config)

        # Modality-specific prompt transformers
        self.modaltiy_transformers = nn.ModuleDict({
            'diag': promptTransformerBlock(config, is_cross_modal=False),
            'drug': promptTransformerBlock(config, is_cross_modal=False),
            'proc': promptTransformerBlock(config, is_cross_modal=False),
            'lab': promptTransformerBlock(config, is_cross_modal=False),
        })

        # Cross-modality transformer
        self.cross_modal_transformer = promptTransformerBlock(config, is_cross_modal=True)

        self.time_prompt_generator = TimePromptGenerator(config)

        self.diffusion = LatentDiffusionModule(config)

        self.time_predictor = TimePredictor(config)

        self.prediction_heads = nn.ModuleDict({
            'diag': nn.Linear(config.hidden_size, config.diag_vocab_size),
            'drug': nn.Linear(config.hidden_size, config.drug_vocab_size),
            'proc': nn.Linear(config.hidden_size, config.proc_vocab_size),
            'lab': nn.Linear(config.hidden_size, config.lab_vocab_size),
        })

    def get_modality_segments(self, input_ids, special_tokens):
        """Split sequence into modality segments based on special tokens."""
        batch_size = input_ids.size(0)
        segments = {}
        
        for modality in ['diag', 'drug', 'proc', 'lab']:
            start_token = special_tokens[f'{modality}_start']
            end_token = special_tokens[f'{modality}_end']
            
            # Create mask for this modality
            modality_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            
            # Process each sequence in the batch
            for b in range(batch_size):
                seq = input_ids[b]
                start_positions = (seq == start_token).nonzero(as_tuple=True)[0]
                end_positions = (seq == end_token).nonzero(as_tuple=True)[0]
                
                # Ensure starts and ends are paired correctly
                if len(start_positions) != len(end_positions):
                    raise ValueError(f"Unmatched start and end tokens for modality {modality} in sequence {b}")
                
                # Mark the tokens between each start-end pair
                for start, end in zip(start_positions, end_positions):
                    if end > start:  # Ensure valid range
                        # Include tokens between start and end (excluding the tokens themselves)
                        modality_mask[b, start+1:end] = True
            
            segments[modality] = modality_mask
            
        return segments
    
    def generate(self, input_ids, seq_mask=None, modality_mask=None, modality_indices=None, timesteps=None):
        """Generate synthetic EHR data using sequential generation.
        Each modality section within a visit will have unique codes.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize output tensor
        output_ids = input_ids.clone()
        
        # Create base attention mask for causal attention
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        base_attention_mask = ~causal_mask
        
        if seq_mask is not None:
            base_attention_mask = base_attention_mask & seq_mask.unsqueeze(1)
        
        # Process each modality in sequence
        for modality in ['diag', 'drug', 'proc', 'lab']:
            start_token = self.config.special_tokens[f'{modality}_start']
            end_token = self.config.special_tokens[f'{modality}_end']
            
            # Find start and end positions
            start_pos = (output_ids == start_token).nonzero()[:, 1]  # [batch_size]
            end_pos = (output_ids == end_token).nonzero()[:, 1]      # [batch_size]
            
            # Skip if no positions found for this modality
            if len(start_pos) == 0 or len(end_pos) == 0:
                continue
                
            # Ensure we have matching pairs of start/end positions
            if len(start_pos) != len(end_pos):
                continue
                
            # For each sequence in the batch
            for b in range(batch_size):
                # Track used tokens for this modality section
                used_tokens = set()
                
                # Validate start and end positions
                current_start = start_pos[b] + 1
                current_end = end_pos[b]
                
                # Skip if invalid range
                if current_start >= current_end:
                    continue
                    
                # Get positions to fill for this sequence
                valid_positions = torch.arange(current_start, current_end, device=device)
                mask_positions = (output_ids[b, valid_positions] == self.config.special_tokens['mask'])
                positions_to_fill = valid_positions[mask_positions]
                
                # Skip if no positions to fill
                if len(positions_to_fill) == 0:
                    continue
                    
                # Process each position sequentially
                for pos in positions_to_fill:
                    # Get current attention mask up to this position
                    attention_mask = base_attention_mask[b, :pos+1, :pos+1].unsqueeze(0)
                    
                    # Get embeddings
                    hidden_states = self.token_embedding(output_ids)
                    
                    # Add modality embeddings if provided
                    if modality_indices is not None:
                        modality_embeds = self.modality_embedding(modality_indices)
                        hidden_states = hidden_states + modality_embeds
                    
                    # Process through transformers
                    # 1. Shared transformer
                    hidden_states = self.shared_transformer(hidden_states)
                    
                    # 2. Modality-specific transformer
                    current_states = hidden_states[:, :pos+1]
                    hidden_states_mod, _ = self.modaltiy_transformers[modality](
                        current_states,
                        attention_mask=attention_mask if self.training else None
                    )
                    
                    # 3. Cross-modal transformer
                    hidden_states_cross, _ = self.cross_modal_transformer(
                        hidden_states_mod,
                        attention_mask=attention_mask if self.training else None
                    )
                    
                    # Get predictions for current position
                    logits = self.prediction_heads[modality](hidden_states_cross[:, -1])
                    
                    # Create mask for already used tokens
                    vocab_mask = torch.ones_like(logits[0])
                    for token in used_tokens:
                        vocab_mask[token] = 0
                    
                    # Apply mask to logits
                    masked_logits = logits.clone()
                    masked_logits[0, ~vocab_mask.bool()] = float('-inf')
                    
                    # Sample unique token
                    sampled_token = sequentially_sample_unique_tokens(masked_logits, num_needed=1, temperature=50.0)[0]
                    
                    # Update used tokens set and output tensor
                    used_tokens.add(sampled_token)
                    output_ids[b, pos] = sampled_token
                    
                    # If we've used all possible tokens for this vocabulary, break
                    if len(used_tokens) >= self.prediction_heads[modality].out_features:
                        break
        
        # Prepare final predictions
        predictions = {}
        for modality in ['diag', 'drug', 'proc', 'lab']:
            start_token = self.config.special_tokens[f'{modality}_start']
            end_token = self.config.special_tokens[f'{modality}_end']
            
            # Create mask for valid positions
            modality_mask = torch.zeros_like(output_ids, dtype=torch.bool)
            
            starts = (output_ids == start_token).nonzero()
            ends = (output_ids == end_token).nonzero()
            
            # Mark valid positions
            for (batch_idx, start_idx), (_, end_idx) in zip(starts, ends):
                modality_mask[batch_idx, start_idx+1:end_idx] = True
            
            # Get generated tokens for this modality
            valid_positions = modality_mask & (output_ids != self.config.special_tokens['mask'])
            generated_tokens = output_ids[valid_positions].tolist()
            
            predictions[modality] = {
                'logits': self.prediction_heads[modality](hidden_states),
                'mask': valid_positions,
                'unique_tokens': generated_tokens
            }
        
        return predictions
        
    def generate_direct_diffusion(self, input_ids, seq_mask=None, modality_mask=None, modality_indices=None, timesteps=None):

        # expect input_ids to be a sequence that looks like training sequence but have special tokens and mask tokens only
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token in self.config.special_tokens.values():
            special_tokens_mask |= (input_ids == token)


        mask_positions = (input_ids == self.config.special_tokens['mask']).bool()
        
        # Get segments
        segments = {}
        for modality, mod_id in [('diag', 1), ('drug', 2), ('proc', 3), ('lab', 4)]:
            start_token = self.config.special_tokens[f'{modality}_start']
            end_token = self.config.special_tokens[f'{modality}_end']
            
            # Find positions between start and end tokens that contain mask tokens
            starts = (input_ids == start_token).nonzero()
            ends = (input_ids == end_token).nonzero()
            
            segment_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for start_pos, end_pos in zip(starts, ends):
                # Include positions between start and end tokens
                segment_mask[:, start_pos[1]+1:end_pos[1]] = True
            
            # Valid positions are those that are both mask tokens and within the segment
            segments[modality] = segment_mask & mask_positions

        hidden_states = self.token_embedding(input_ids)


        # Set maximum timestep for generation
        batch_size = input_ids.size(0)
        t = {
            'diag': torch.ones(batch_size, device=input_ids.device),
            'drug': torch.ones(batch_size, device=input_ids.device),
            'proc': torch.ones(batch_size, device=input_ids.device),
            'lab': torch.ones(batch_size, device=input_ids.device)
        }

        # Create dummy prompts
        num_prompt_tokens = self.config.num_modality_tokens
        hidden_size = self.config.hidden_size
        dummy_modality_prompts = {
            modality: torch.zeros(batch_size, num_prompt_tokens, hidden_size, device=input_ids.device)
            for modality in ['diag', 'drug', 'proc', 'lab']
        }
        dummy_cross_prompt = torch.zeros(batch_size, num_prompt_tokens, hidden_size, device=input_ids.device)


        denoised_latents = self.diffusion.denoise(
            hidden_states,
            t,
            segments,  # Using our correctly identified segments
            special_tokens_mask,
            dummy_modality_prompts,
            dummy_cross_prompt
        )

        predictions = {}
        for modality, valid_positions in segments.items():

            modality_preds = self.prediction_heads[modality](denoised_latents)
            valid_logits = modality_preds[valid_positions]
            
            # Generate unique tokens for this modality
            if valid_logits.size(0) > 0:
                tokens = sample_unique_tokens(valid_logits, valid_logits.size(0))
            else:
                tokens = []
            
            predictions[modality] = {
                'logits': modality_preds,
                'mask': valid_positions,
                'unique_tokens': tokens
            }

        return predictions

    def forward(self, input_ids, seq_mask, modality_mask, modality_indices, timesteps=None, ablation: str = None):
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        input_embeddings = self.token_embedding(input_ids)
        
        # 1. Ablate shared transformer - tests importance of shared feature learning
        if ablation == 'ab1_shared':
            shared_output = input_embeddings
        else:
            shared_output = self.shared_transformer(input_embeddings, modality_mask)
            
        # Get modality segments
        segments = self.get_modality_segments(input_ids, self.config.special_tokens)
        modality_outputs, modality_prompts = {}, {}
        
        # 2. Ablate modality-specific transformers - tests importance of modality-specific processing
        if ablation == 'ab2_modality':
            # Skip modality-specific processing
            modality_prompts = {
                modality: torch.zeros((batch_size, self.config.num_modality_tokens, 
                                    self.config.hidden_size), device=input_ids.device)
                for modality in segments.keys()
            }
        else:
            for modality, mask in segments.items():
                modality_hidden = shared_output.masked_fill(~mask.unsqueeze(-1), 0)
                modality_att_mask = seq_mask.bool() & mask.bool()
                
                modality_output, prompt_states = self.modaltiy_transformers[modality](
                    modality_hidden,
                    attention_mask=modality_att_mask if self.training else None
                )
                modality_outputs[modality] = modality_output
                modality_prompts[modality] = prompt_states
        
        # Add modality positional embeddings
        if modality_indices is not None:
            modality_positional_embeddings = self.modality_embedding(modality_indices)
            shared_output += modality_positional_embeddings
        
        # 3. Ablate cross-modal transformer - tests importance of cross-modality interactions
        if ablation == 'ab3_cross':
            cross_output = shared_output
            cross_prompt = torch.zeros((batch_size, self.config.num_modality_tokens, 
                                    self.config.hidden_size), device=input_ids.device)
        else:
            cross_output, cross_prompt = self.cross_modal_transformer(
                shared_output,
                attention_mask=seq_mask if self.training else None
            )
        
        # 4. Ablate diffusion module - tests impact of diffusion-based learning
        if ablation == 'ab4_diffusion':
            if self.training and timesteps is not None:
                special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for token in self.config.special_tokens.values():
                    special_tokens_mask |= (input_ids == token)
                
                # Skip noise addition and denoising
                denoised_latents = cross_output
                diffusion_loss = torch.tensor(0.0, device=input_ids.device)
        

        if self.training and timesteps is not None:
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token in self.config.special_tokens.values():
                special_tokens_mask |= (input_ids == token)

            noised_latents = self.diffusion.add_noise(
                cross_output, timesteps, segments, special_tokens_mask
            )
            denoised_latents = self.diffusion.denoise(
                noised_latents, timesteps, segments, special_tokens_mask,
                modality_prompts, cross_prompt
            )

            valid_tokens = ~special_tokens_mask
            diffusion_loss = F.mse_loss(
                denoised_latents[valid_tokens],
                cross_output[valid_tokens]
            )
        else:
            denoised_latents = cross_output
            diffusion_loss = None
        
        # Get predictions for each modality
        predictions = {}
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token in self.config.special_tokens.values():
            special_tokens_mask |= (input_ids == token)
            
        for modality, mask in segments.items():
            valid_tokens = mask & ~special_tokens_mask
            modality_preds = self.prediction_heads[modality](denoised_latents)
            predictions[modality] = {
                'logits': modality_preds,
                'mask': valid_tokens
            }
        
        return predictions, diffusion_loss


