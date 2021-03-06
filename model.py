import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'min-dalle'
sys.path.insert(0, submodule_dir.as_posix())

from min_dalle.generate_image import load_dalle_bart_metadata
from min_dalle.load_params import load_dalle_bart_flax_params
from min_dalle.min_dalle_torch import detokenize_torch
from min_dalle.models.dalle_bart_decoder_flax import DalleBartDecoderFlax
from min_dalle.models.dalle_bart_encoder_flax import DalleBartEncoderFlax
from min_dalle.text_tokenizer import TextTokenizer


class Model:
    def __init__(self, model_name: str = 'dalle-mini'):
        self.model_name = model_name
        config, tokenizer, encoder, decoder, params_dalle_bart = self.load_model(
            model_name=model_name)
        self.config = config
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.params_dalle_bart = params_dalle_bart

    def set_model(self, model_name: str) -> None:
        assert model_name in ['dalle-mini', 'dalle-mega']
        if model_name == self.model_name:
            return
        self.model_name = model_name
        config, tokenizer, encoder, decoder, params_dalle_bart = self.load_model(
            model_name=model_name)
        self.config = config
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.params_dalle_bart = params_dalle_bart

    def load_model(
        self, model_name: str
    ) -> tuple[dict, TextTokenizer, DalleBartEncoderFlax, DalleBartDecoderFlax,
               dict]:
        model_dir = os.getenv('DALLE_MODEL_DIR', 'pretrained')
        model_path = f'{model_dir}/{model_name}'
        config, vocab, merges = load_dalle_bart_metadata(model_path)
        params_dalle_bart = load_dalle_bart_flax_params(model_path)

        tokenizer = TextTokenizer(vocab, merges)

        encoder: DalleBartEncoderFlax = DalleBartEncoderFlax(
            attention_head_count=config['encoder_attention_heads'],
            embed_count=config['d_model'],
            glu_embed_count=config['encoder_ffn_dim'],
            text_token_count=config['max_text_length'],
            text_vocab_count=config['encoder_vocab_size'],
            layer_count=config['encoder_layers']).bind(
                {'params': params_dalle_bart.pop('encoder')})

        decoder = DalleBartDecoderFlax(
            image_token_count=config['image_length'],
            text_token_count=config['max_text_length'],
            image_vocab_count=config['image_vocab_size'],
            attention_head_count=config['decoder_attention_heads'],
            embed_count=config['d_model'],
            glu_embed_count=config['decoder_ffn_dim'],
            layer_count=config['decoder_layers'],
            start_token=config['decoder_start_token_id'])

        return config, tokenizer, encoder, decoder, params_dalle_bart

    def tokenize_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(text)
        text_tokens = np.ones((2, self.config['max_text_length']),
                              dtype=np.int32)
        text_tokens[0, :len(tokens)] = tokens
        text_tokens[1, :2] = [tokens[0], tokens[-1]]
        return text_tokens

    def encode_flax(self, text_tokens: np.ndarray) -> jnp.ndarray:
        return self.encoder(text_tokens)

    def decode_flax(self, text_tokens: np.ndarray, encoder_state: jnp.ndarray,
                    seed: int) -> np.ndarray:
        return self.decoder.sample_image_tokens(
            text_tokens, encoder_state, jax.random.PRNGKey(seed),
            self.params_dalle_bart['decoder'])

    def generate_image(self, text_tokens: np.ndarray, encoder_state,
                       seed: int) -> np.ndarray:
        image_tokens = np.zeros(self.config['image_length'])
        image_tokens[...] = self.decode_flax(text_tokens, encoder_state, seed)
        return detokenize_torch(image_tokens)

    def generate_images(self, text: str, seeds: list[int]) -> list[np.ndarray]:
        text_tokens = self.tokenize_text(text)
        encoder_state = self.encode_flax(text_tokens)
        return [
            self.generate_image(text_tokens, encoder_state, seed)
            for seed in seeds
        ]


def make_grid(images: list[np.ndarray], nrows: int, ncols: int) -> np.ndarray:
    h, w = images[0].shape[:2]
    grid = np.zeros((h * nrows, w * ncols, 3), dtype=np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            index = ncols * i + j
            if index >= len(images):
                break
            grid[h * i:h * (i + 1), w * j:w * (j + 1)] = images[index]
    return grid


class AppModel(Model):
    def run(self, text: str, model_name: str, seed: int, nrows: int,
            ncols: int) -> tuple[np.ndarray, list[np.ndarray]]:
        self.set_model(model_name)
        seeds = list(range(seed, seed + nrows * ncols))
        res = super().generate_images(text, seeds)
        grid = make_grid(res, nrows, ncols)
        return grid, res
