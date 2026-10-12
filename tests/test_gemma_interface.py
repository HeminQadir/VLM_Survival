"""No gated weights: exercise the real Gemma3 API with a tiny random model."""
import importlib.util
import unittest
import torch

AVAILABLE = importlib.util.find_spec('transformers') is not None


class FakeTokenizer:
    pad_token_id, eos_token_id = 0, 2
    def apply_chat_template(self, messages, **kwargs):
        return 'BOS '+messages[0]['content']
    def encode(self, text, **kwargs):
        return [3+ord(c)%120 for c in text]
    def decode(self, ids, **kwargs):
        return ' '.join(str(int(i)) for i in ids)


@unittest.skipUnless(AVAILABLE, 'Install the medgemma extra for Gemma3 interface tests.')
class GemmaInterfaceTests(unittest.TestCase):
    def test_real_decoder_lora_gradient_and_auxiliary_task(self):
        from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig, Gemma3ForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        from vlm_survival.models import MedGemmaSurvival
        from vlm_survival.survival import survival_nll
        text = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=2048, sliding_window=512, pad_token_id=0,
            bos_token_id=1, eos_token_id=2)
        vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=16, patch_size=8)
        backbone = Gemma3ForConditionalGeneration(Gemma3Config(text_config=text.to_dict(),
            vision_config=vision.to_dict(), mm_tokens_per_image=4, image_token_index=127))
        lora = LoraConfig(r=2, lora_alpha=4, target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)')
        backbone = get_peft_model(backbone, lora)
        cfg = {'feature_dim': 16, 'visual_tokens': 2, 'image_mode': 'volume',
            'features': ['age'], 'time_edges': [0, 10, 20], 'disease': 'synthetic',
            'max_prompt_tokens': 1500, 'evidence_max_tokens': 8}
        model = MedGemmaSurvival(cfg, backbone=backbone, tokenizer=FakeTokenizer())
        batch = {'image': torch.randn(2, 1, 16, 16, 16), 'tabular': torch.randn(2, 2),
                 'clinical_text': ['age: 65', 'age: not measured'], 'patient_id': ['a', 'b']}
        logits = model(batch)
        self.assertEqual(logits.shape, (2, 2))
        loss = survival_nll(logits, torch.tensor([5., 12.]), torch.tensor([1., 0.]), cfg['time_edges'])
        loss.backward()
        self.assertGreater(model.image.encoder.network[0].weight.grad.abs().sum(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if 'lora_B' in n))
        model.zero_grad()
        evidence = model.evidence_loss(batch)
        self.assertTrue(torch.isfinite(evidence))
        evidence.backward()
        model.eval()
        one = {k: (v[:1] if torch.is_tensor(v) else v[:1]) for k, v in batch.items()}
        generated = model.verbalize(one, '{"survival": 0.5}', max_new_tokens=2)
        self.assertIsInstance(generated, str)


if __name__ == '__main__':
    unittest.main()
