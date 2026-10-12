"""3D adapters, a CPU baseline, and a MedGemma survival/evidence model.

The MedGemma path bypasses the original 2D tower and feeds continuous 3D tokens
to the language decoder. The supplied CNN is randomly initialized, not a
pretrained medical encoder. Cached pretrained spatial tokens are supported.
"""
import torch
from torch import nn
from torch.nn import functional as F


class VolumeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv3d(32, dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.position = nn.Parameter(torch.randn(1, 8, dim)*0.02)

    def forward(self, volume):
        return self.network(volume).flatten(2).transpose(1, 2) + self.position


class ImageAdapter(nn.Module):
    def __init__(self, cfg, output_dim):
        super().__init__()
        dim, count = cfg['feature_dim'], cfg['visual_tokens']
        if dim % 4:
            raise ValueError('feature_dim must be divisible by 4.')
        self.enabled = cfg.get('use_image', True)
        self.encoder = VolumeEncoder(dim) if cfg['image_mode'] == 'volume' else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, count, dim)*.02)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, image):
        tokens = self.encoder(image)
        q = self.queries.expand(image.shape[0], -1, -1)
        output, _ = self.attention(q, tokens, tokens, need_weights=False)
        output = self.project(output)
        return output if self.enabled else torch.zeros_like(output)


class TinySurvival(nn.Module):
    """Small CPU debugging baseline. No language model or clinical reasoning."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image = ImageAdapter(cfg, 64)
        self.head = nn.Sequential(nn.Linear(64+2*len(cfg['features']), 64), nn.GELU(),
                                  nn.Linear(64, len(cfg['time_edges'])-1))
        nn.init.constant_(self.head[-1].bias, -2.)

    def forward(self, batch):
        tokens = self.image(batch['image']).mean(1)
        return self.head(torch.cat([tokens, batch['tabular']], dim=1))

    def evidence_loss(self, batch):
        return self.head[-1].weight.sum()*0


class MedGemmaSurvival(nn.Module):
    def __init__(self, cfg, backbone=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            device = torch.device(cfg['device'])
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable. Use configs/smoke.json for CPU tests.')
            use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float32
            kwargs = dict(torch_dtype=dtype, attn_implementation='eager',
                          revision=cfg.get('model_revision', 'main'))
            if cfg.get('load_in_4bit', False):
                if device.type != 'cuda':
                    raise ValueError('4-bit loading requires CUDA in this implementation.')
                kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
                kwargs['device_map'] = {'': device.index or 0}
            backbone = Gemma3ForConditionalGeneration.from_pretrained(cfg['model_id'], **kwargs)
            if cfg.get('load_in_4bit', False):
                backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
            else:
                backbone.to(device)
            # Match language attention only: leave the unused original vision tower frozen.
            lora = LoraConfig(r=cfg.get('lora_rank', 8), lora_alpha=cfg.get('lora_alpha', 16),
                lora_dropout=0.05, bias='none',
                target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)')
            backbone = get_peft_model(backbone, lora)
            if cfg.get('gradient_checkpointing', True):
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg.get('model_revision', 'main'))
        self.backbone, self.tokenizer = backbone, tokenizer
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.base.config.use_cache = False
        dim = self.base.config.text_config.hidden_size
        self.image = ImageAdapter(cfg, dim)
        self.numeric = nn.Sequential(nn.Linear(2*len(cfg['features']), dim), nn.LayerNorm(dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(cfg['time_edges'])-1))
        nn.init.constant_(self.head[-1].bias, -2.)
        device = self.base.get_input_embeddings().weight.device
        self.image.to(device)
        self.numeric.to(device)
        self.head.to(device)

    @property
    def base(self):
        return self.backbone.get_base_model() if hasattr(self.backbone, 'get_base_model') else self.backbone

    def prompt(self, clinical, task):
        user = (f'Disease context: {self.cfg["disease"]}. Endpoint: all-cause survival from baseline.\n'
                f'Baseline clinical fields:\n{clinical}\n{task}\n'
                'Use only supplied evidence. Missing is not normal. Do not invent imaging findings.')
        return self.tokenizer.apply_chat_template([{'role': 'user', 'content': user}],
                                                  tokenize=False, add_generation_prompt=True)

    def inputs(self, batch, task, answers=None):
        """Put continuous tokens after the first chat token, keeping BOS first.

        No truncation is silent. Outcome fields never enter the text. Evidence
        answers are supplied only on a SEPARATE language-training forward pass.
        """
        prefixes = self.image(batch['image'])
        numeric = self.numeric(batch['tabular']).unsqueeze(1)
        if not self.cfg.get('use_tabular', True):
            numeric = torch.zeros_like(numeric)
        prefixes = torch.cat([prefixes, numeric], dim=1)
        embedding = self.base.get_input_embeddings()
        sequences, masks, labels = [], [], []
        for i, clinical in enumerate(batch['clinical_text']):
            prompt_ids = self.tokenizer.encode(self.prompt(clinical, task), add_special_tokens=False)
            if len(prompt_ids) > self.cfg.get('max_prompt_tokens', 512):
                raise ValueError('Prompt exceeds max_prompt_tokens. Increase it or reduce fields explicitly.')
            if not prompt_ids:
                raise ValueError('Empty chat prompt.')
            answer_ids = []
            if answers is not None:
                answer_ids = self.tokenizer.encode(answers[i], add_special_tokens=False)
                answer_ids = answer_ids[:self.cfg.get('evidence_max_tokens', 128)] + [self.tokenizer.eos_token_id]
            ids = torch.tensor(prompt_ids+answer_ids, device=prefixes.device)
            text = embedding(ids)
            prefix = prefixes[i].to(text.dtype)
            sequence = torch.cat([text[:1], prefix, text[1:]], dim=0)
            target = torch.full((len(sequence),), -100, device=ids.device, dtype=torch.long)
            if answer_ids:
                target[-len(answer_ids):] = ids[-len(answer_ids):]
            sequences.append(sequence)
            masks.append(torch.ones(len(sequence), device=ids.device, dtype=torch.long))
            labels.append(target)
        embeds = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        attention = nn.utils.rnn.pad_sequence(masks, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return embeds, attention, labels

    def forward(self, batch):
        embeds, mask, _ = self.inputs(batch, 'Summarize the baseline information for survival estimation.')
        # Calling the base model skips the expensive vocabulary head. Injected LoRA
        # modules still run inside the decoder; .get_base_model() does not remove them.
        result = self.base.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True)
        last = result.last_hidden_state[torch.arange(len(mask), device=mask.device), mask.sum(1)-1]
        return self.head(last.float())

    def evidence_loss(self, batch):
        """Self-supervised table transcription, NOT clinical rationale supervision."""
        if not self.cfg.get('use_tabular', True):
            return self.head[-1].weight.sum()*0
        answers = ['Recorded baseline fields:\n'+text for text in batch['clinical_text']]
        embeds, mask, labels = self.inputs(batch, 'Repeat the recorded baseline fields exactly. Preserve missing values.', answers)
        hidden = self.base.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        targets = labels[:, 1:]
        valid = targets != -100
        # Compute vocabulary logits only where answer tokens are supervised.
        logits = self.base.lm_head(hidden[:, :-1][valid]).float()
        cap = getattr(self.base.config.text_config, 'final_logit_softcapping', None)
        if cap is not None:
            logits = torch.tanh(logits/cap)*cap
        return F.cross_entropy(logits, targets[valid])

    @torch.no_grad()
    def verbalize(self, batch, evidence_record, max_new_tokens=180):
        if len(batch['patient_id']) != 1:
            raise ValueError('Verbalization currently supports one patient at a time.')
        task = ('Write a short research-only explanation of this numerical evidence record. '
                'Distinguish measured facts from model sensitivities; do not claim causation or name '
                'unverified CT abnormalities. If evidence is insufficient, say so.\n'+evidence_record)
        embeds, mask, _ = self.inputs(batch, task)
        # Returned output for inputs_embeds contains generated IDs, not input text IDs.
        ids = self.backbone.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False,
            max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


def build_model(cfg):
    if cfg['backend'] == 'tiny':
        return TinySurvival(cfg).to(cfg['device'])
    if cfg['backend'] == 'medgemma':
        return MedGemmaSurvival(cfg)
    raise ValueError('backend must be tiny or medgemma')
