import os
import sys
from pathlib import Path
from typing import List

import torch

from src.utils.data_utils import mask_features

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
from src.vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline
from src.models.AutoencoderKL import AutoencoderKL
from src.models.emasc import EMASC
from src.models.inversion_adapter import InversionAdapter
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

import torchvision
from src.utils.encode_text_word_embedding import encode_text_word_embedding


@torch.no_grad()
def generate_images_from_tryon_pipe(pipe: StableDiffusionTryOnePipeline, inversion_adapter: InversionAdapter,
                                    test_dataloader: torch.utils.data.DataLoader, output_dir: str, order: str,
                                    save_name: str, text_usage: str, vision_encoder: CLIPVisionModelWithProjection,
                                    processor: CLIPProcessor, cloth_input_type: str, cloth_cond_rate: int = 1,
                                    num_vstar: int = 1, seed: int = 1234, num_inference_steps: int = 50,
                                    guidance_scale: int = 7.5,
                                    use_png: bool = False):
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch.get("image")
        mask_img = batch.get("inpaint_mask")
        if mask_img is not None:
            mask_img = mask_img.type(torch.float32)
        pose_map = batch.get("pose_map")
        warped_cloth = batch.get('warped_cloth')
        category = batch.get("category")
        cloth = batch.get("cloth")

        # Generate text prompts
        if text_usage == "noun_chunks":
            prompts = batch["captions"]
        elif text_usage == "none":
            prompts = [""] * len(batch["captions"])
        elif text_usage == 'inversion_adapter':
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',

            }
            text = [f'a photo of a model wearing {category_text[category]} {" $ " * num_vstar}' for
                    category in batch['category']]

            clip_cloth_features = batch.get('clip_cloth_features')
            if clip_cloth_features is None:
                with torch.no_grad():
                    # Get the visual features of the in-shop cloths
                    input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                           antialias=True).clamp(0, 1)
                    processed_images = processor(images=input_image, return_tensors="pt")
                    clip_cloth_features = vision_encoder(
                        processed_images.pixel_values.to(model_img.device)).last_hidden_state

            # Compute the predicted PTEs
            word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
            word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))

            # Tokenize text
            tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                            truncation=True, return_tensors="pt").input_ids
            tokenized_text = tokenized_text.to(word_embeddings.device)

            # Encode the text using the PTEs extracted from the in-shop cloths
            encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                               word_embeddings, num_vstar).last_hidden_state
        else:
            raise ValueError(f"Unknown text usage {text_usage}")

        # Generate images
        if text_usage == 'inversion_adapter':
            generated_images = pipe(
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                warped_cloth=warped_cloth,
                prompt_embeds=encoder_hidden_states,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                cloth_input_type=cloth_input_type,
                cloth_cond_rate=cloth_cond_rate,
                num_inference_steps=num_inference_steps
            ).images
        else:
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                warped_cloth=warped_cloth,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                cloth_input_type=cloth_input_type,
                cloth_cond_rate=cloth_cond_rate,
                num_inference_steps=num_inference_steps
            ).images

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))

            if use_png:
                name = name.replace(".jpg", ".png")
                gen_image.save(
                    os.path.join(save_path, cat, name))
            else:
                gen_image.save(
                    os.path.join(save_path, cat, name), quality=95)


def generate_images_inversion_adapter(pipe: StableDiffusionInpaintPipeline, inversion_adapter: InversionAdapter,
                                      vision_encoder: CLIPVisionModelWithProjection, processor: CLIPProcessor,
                                      test_dataloader: torch.utils.data.DataLoader, output_dir, order: str,
                                      save_name: str, num_vstar=1, seed=1234, num_inference_steps=50,
                                      guidance_scale=7.5, use_png=False) -> None:
    """
    Extract and save images using the SD inpainting pipeline using the PTEs from the inversion adapter.
    """
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch["image"]
        mask_img = batch["inpaint_mask"]
        mask_img = mask_img.type(torch.float32)
        category = batch["category"]
        # Generate images
        cloth = batch.get("cloth")
        clip_cloth_features = batch.get('clip_cloth_features')

        if clip_cloth_features is None:
            # Get the visual features of the in-shop cloths
            input_image = torchvision.transforms.functional.resize(
                (cloth + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
            processed_images = processor(images=input_image, return_tensors="pt")
            clip_cloth_features = vision_encoder(processed_images.pixel_values.to(model_img.device)).last_hidden_state

        word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))

        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * num_vstar}' for category in
                batch['category']]

        # Tokenize text
        tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(model_img.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                           word_embeddings,
                                                           num_vstar=num_vstar).last_hidden_state

        # Generate images
        generated_images = pipe(
            image=model_img,
            mask_image=mask_img,
            prompt_embeds=encoder_hidden_states,
            height=512,
            width=384,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            num_inference_steps=num_inference_steps
        ).images

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))

            if use_png:
                name = name.replace(".jpg", ".png")
                gen_image.save(
                    os.path.join(save_path, cat, name))
            else:
                gen_image.save(
                    os.path.join(save_path, cat, name), quality=95)


@torch.inference_mode()
def extract_save_vae_images(vae: AutoencoderKL, emasc: EMASC, test_dataloader: torch.utils.data.DataLoader,
                            int_layers: List[int], output_dir: str, order: str, save_name: str, emasc_type: str,
                            mask_feat: bool) -> None:
    """
    Extract and save image using only VAE or VAE + EMASC
    """
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    for idx, batch in enumerate(tqdm(test_dataloader)):
        category = batch["category"]

        if emasc_type != "none":
            # Extract intermediate features from 'im_mask' and encode image
            posterior_im, _ = vae.encode(batch["image"])
            _, intermediate_features = vae.encode(batch["im_mask"])
            intermediate_features = [intermediate_features[i] for i in int_layers]

            # Use EMASC
            processed_intermediate_features = emasc(intermediate_features)

            if mask_feat:
                processed_intermediate_features = mask_features(processed_intermediate_features, batch["inpaint_mask"])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents, processed_intermediate_features, int_layers).sample
        else:
            # Encode and decode image without EMASC
            posterior_im = vae.encode(batch["image"])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents).sample

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            gen_image = (gen_image + 1) / 2  # [-1, 1] -> [0, 1]
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            torchvision.utils.save_image(gen_image, os.path.join(save_path, cat, name), quality=95)
