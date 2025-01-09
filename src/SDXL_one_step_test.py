import os
import torch
from PIL import Image
from typing import List, Optional, Union

from diffusers import StableDiffusionXLPipeline
from diffusers.utils import BaseOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    rescale_noise_cfg,
    retrieve_timesteps,
)

# -----------------------
# (1) 커스텀 Output 클래스
# -----------------------
class X0PipelineOutput(BaseOutput):
    """
    x0_preds를 포함하여 반환하기 위한 커스텀 Output 클래스.
    """
    def __init__(
        self,
        images: Union[List[Image.Image], torch.FloatTensor],
        x0_preds: Optional[List[torch.FloatTensor]] = None,
    ):
        super().__init__()
        self.images = images
        self.x0_preds = x0_preds


# -----------------------
# (2) 커스텀 파이프라인
# -----------------------
class X0CaptureSDXLPipeline(StableDiffusionXLPipeline):
    """
    StableDiffusionXLPipeline을 상속받아,
    - 매 디노이징 스텝마다 x0(pred_original_sample)을 저장하고 (이미지로만 저장),
    - 최종 결과 이미지를 지정된 디렉토리에 자동으로 저장하도록 수정한 예시 파이프라인
    """

    @property
    def cross_attention_kwargs(self):
        # 원본 파이프라인에서 cross_attention_kwargs가 사용됨 -> 여기선 None으로 처리
        return None

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        eta: float = 0.0,
        generator: torch.Generator = None,
        latents: torch.FloatTensor = None,
        output_type: str = "pil",
        return_dict: bool = True,
        save_x0: bool = True,                 # x0를 리스트로 반환할지 여부
        image_out_dir: str = None,            # 최종 이미지 자동 저장 경로
        x0_out_dir: str = None,               # 스텝별 x0 이미지 자동 저장 경로
        original_size: tuple = None,          # SDXL micro-conditioning 용
        target_size: tuple = None,            # 보통 (height, width)와 동일
        **kwargs
    ):
        """
        prompt, height, width 등은 원본과 동일.
        save_x0=True -> 매 스텝별 x0를 리스트로 함께 반환.
        image_out_dir -> 최종 이미지 자동 저장 (None이면 미저장).
        x0_out_dir -> 스텝별 x0 이미지 자동 저장 (None이면 미저장).
        """

        # ------------------------------------------------------------------------------------
        # 1) 전처리
        # ------------------------------------------------------------------------------------
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.check_inputs(
            prompt=prompt,
            prompt_2=None,
            height=height,
            width=width,
            callback_steps=None,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=None,
        )

        # batch size 계산
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("prompt must be either `str` or `List[str]`")

        device = self._execution_device

        # ------------------------------------------------------------------------------------
        # 2) prompt -> text embeds
        # ------------------------------------------------------------------------------------
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
        )

        # ------------------------------------------------------------------------------------
        # 3) timesteps
        # ------------------------------------------------------------------------------------
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps=None
        )

        # ------------------------------------------------------------------------------------
        # 4) latents
        # ------------------------------------------------------------------------------------
        latents = self.prepare_latents(
            batch_size,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # ------------------------------------------------------------------------------------
        # 5) extra_step_kwargs
        # ------------------------------------------------------------------------------------
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ------------------------------------------------------------------------------------
        # 6) micro-conditioning
        # ------------------------------------------------------------------------------------
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size=original_size,
            crops_coords_top_left=(0, 0),
            target_size=target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(device)
        neg_add_time_ids = add_time_ids

        add_text_embeds = pooled_prompt_embeds.to(device)
        neg_text_embeds = neg_pooled_embeds.to(device)

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            # prompt_embeds와 neg_prompt_embeds를 합쳐서 classifier-free guidance
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([neg_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([neg_add_time_ids, add_time_ids], dim=0)

        # ------------------------------------------------------------------------------------
        # 7) 디노이징 루프
        # ------------------------------------------------------------------------------------
        self._num_timesteps = len(timesteps)
        x0_preds = [] if save_x0 else None

        for i, t in enumerate(timesteps):
            # (7.1) latent_model_input
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # (7.2) unet forward
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # (7.3) guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # (7.4) scheduler step
            step_output = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )
            if len(step_output) >= 2:
                latents = step_output[0]
                pred_x0 = step_output[1]
            else:
                latents = step_output[0]
                pred_x0 = None

            # (7.5) 스텝별 x0 저장
            if x0_preds is not None and pred_x0 is not None:
                x0_preds.append(pred_x0.detach().clone())

        # ------------------------------------------------------------------------------------
        # 8) 최종 이미지 디코딩
        # ------------------------------------------------------------------------------------
        if output_type == "latent":
            image = latents
        else:
            # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            
            # VAE 디코딩: float16 오버플로우 방지 및 디노말라이즈 추가
            needs_upcasting = self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False)

            if needs_upcasting:
                # VAE를 float32로 업캐스트
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    self.vae = self.vae.to(latents.dtype)

            # 디노말라이즈 (latents_mean, latents_std가 설정된 경우)
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            # 디코딩
            image = self.vae.decode(latents, return_dict=False)[0]

            # VAE를 다시 float16으로 복구
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)
            
        # ------------------------------------------------------------------------------------
        # 9) 이미지 저장
        # ------------------------------------------------------------------------------------
        # (9.1) 최종 이미지 저장
        if image_out_dir is not None:
            os.makedirs(image_out_dir, exist_ok=True)
            if not isinstance(image, list):
                image = [image]
            for idx, img in enumerate(image):
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                save_path = os.path.join(image_out_dir, f"final_image_{idx}.png")
                img.save(save_path)
                print(f"[X0CaptureSDXLPipeline] 최종 이미지 {idx} 저장 완료: {save_path}")

        # (9.2) x0 이미지를 저장(스텝별)
        if x0_out_dir is not None and x0_preds is not None:
            os.makedirs(x0_out_dir, exist_ok=True)
            for step_idx, x0_tensor in enumerate(x0_preds):
                # x0_tensor shape: (batch_size, C, H, W)
                batch_size_x0 = x0_tensor.shape[0]
                for b in range(batch_size_x0):
                    single_x0 = x0_tensor[b : b + 1]

                    # ** float16 / float32 타입 불일치 문제 해결 **
                    single_x0 = single_x0.to(self.vae.dtype)

                    # VAE 디코딩: float16 오버플로우 방지 및 디노말라이즈 추가
                    needs_upcasting = self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False)

                    if needs_upcasting:
                        # VAE를 float32로 업캐스트
                        self.upcast_vae()
                        single_x0 = single_x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                    elif single_x0.dtype != self.vae.dtype:
                        if torch.backends.mps.is_available():
                            self.vae = self.vae.to(single_x0.dtype)

                    # 디노말라이즈 (latents_mean, latents_std가 설정된 경우)
                    has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
                    has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

                    if has_latents_mean and has_latents_std:
                        latents_mean = (
                            torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(single_x0.device, single_x0.dtype)
                        )
                        latents_std = (
                            torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(single_x0.device, single_x0.dtype)
                        )
                        single_x0 = single_x0 * latents_std / self.vae.config.scaling_factor + latents_mean
                    else:
                        single_x0 = single_x0 / self.vae.config.scaling_factor

                    # 디코딩
                    x0_img = self.vae.decode(single_x0, return_dict=False)[0]

                    # VAE를 다시 float16으로 복구
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)

                    # post-process (RGB etc.)
                    x0_img = self.image_processor.postprocess(x0_img, output_type="pil")

                    # x0_img가 list인 경우 처리
                    if isinstance(x0_img, list):
                        x0_img = x0_img[0]

                    # 저장
                    x0_img_path = os.path.join(x0_out_dir, f"x0_step_{step_idx:03d}_batch_{b}.png")
                    x0_img.save(x0_img_path)
                    print(f"[X0CaptureSDXLPipeline] x0 스텝 {step_idx} (img) 저장 완료: {x0_img_path}")

        # ------------------------------------------------------------------------------------
        # 10) 반환
        # ------------------------------------------------------------------------------------
        if not return_dict:
            return (image, x0_preds) if save_x0 else (image,)

        return X0PipelineOutput(
            images=image,
            x0_preds=x0_preds if save_x0 else None,
        )

# if __name__ == "__main__":
#     pipe = X0CaptureSDXLPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         torch_dtype=torch.float16,
#     ).to("cuda")

#     # 예: batch_size = 1 -> x0_step_###_batch_0.png 파일이 만들어짐
#     images, x0_list = pipe(
#         prompt="A beautiful landscape with mountains, trees, and a lake",
#         height=1024,
#         width=1024,
#         num_inference_steps=50,
#         guidance_scale=7.5,
#         save_x0=True,
#         image_out_dir="./output_images",
#         x0_out_dir="./output_images/x0_steps",
#         original_size=(1024, 1024),
#         target_size=(1024, 1024),
#         return_dict = False,
#     )

#     print("생성된 최종 이미지 개수:", len(images))
#     print("스텝별 x0 수:", len(x0_list))
#     print("작업 완료!")

import os
import json
import argparse

import torch
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

# -------------------------------------------------------------------
# (1) 여기서 X0CaptureSDXLPipeline, X0PipelineOutput 클래스를
#     "이미 위에 정의된" 상태라고 가정합니다.
#     또는 별도 .py 파일로 만들어두고 import 해주셔도 됩니다.
# -------------------------------------------------------------------
# from my_x0_pipeline import X0CaptureSDXLPipeline

# -----------------------
# (2) Argument Parser
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt (e.g. each line has {'prompt': '...'})."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to write results to."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps (num_inference_steps)."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="Guidance scale."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the generated image."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the generated image."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling."
    )
    parser.add_argument(
        "--save_x0",
        action="store_true",
        help="If given, save x0 step images to [outdir]/[index]/x0_steps."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g. 'cpu' or 'cuda')."
    )
    return parser.parse_args()


# -----------------------
# (3) main 함수
# -----------------------
def main():
    opt = parse_args()
    seed_everything(opt.seed)

    # -------------------------------------------------------------------
    # (3.1) 파이프라인 로드
    # -------------------------------------------------------------------
    # 아래는 Stable Diffusion XL base 1.0 모델 사용 예시
    pipe = X0CaptureSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to(opt.device)

    # XFormers 메모리 최적화 사용
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()

    # outdir 생성
    os.makedirs(opt.outdir, exist_ok=True)

    # -------------------------------------------------------------------
    # (3.2) 메타데이터 파일 읽기
    # -------------------------------------------------------------------
    with open(opt.metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # -------------------------------------------------------------------
    # (3.3) JSONL 각 라인에 대해 이미지 생성
    # -------------------------------------------------------------------
    for idx, line in enumerate(tqdm(reversed(lines), desc="Prompt Loop")):
        data = json.loads(line)
        prompt = data.get("prompt", None)
        if prompt is None:
            print(f"[Warning] No 'prompt' found in line {idx}. Skipping.")
            continue

        # (3.3.1) index별 outdir
        outdir_index = os.path.join(opt.outdir, f"{idx:05d}")
        os.makedirs(outdir_index, exist_ok=True)

        # (3.3.2) x0_out_dir (옵션)
        x0_out_dir = os.path.join(outdir_index, "x0_steps") if opt.save_x0 else None

        # (3.3.3) 이미지 생성
        # return_dict=False 로 하면 (images, x0_preds) 형태 반환
        s = pipe(
            prompt=prompt,
            height=opt.height,
            width=opt.width,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            save_x0=opt.save_x0,  # x0 리스트 반환
            image_out_dir=outdir_index,  # 최종 이미지 저장
            x0_out_dir=x0_out_dir,       # x0 이미지 저장
            return_dict=False,
        )

        # (3.3.4) JSON 메타데이터도 함께 저장(원본 예시에 맞춰서)
        meta_file_path = os.path.join(outdir_index, "metadata.json")
        with open(meta_file_path, "w", encoding="utf-8") as mf:
            json.dump(data, mf, ensure_ascii=False, indent=2)

        # (3.3.5) 필요에 따라 추가 정보 출력
        # print(f"[Info] index={idx}, prompt='{prompt}' -> generated {len(images)} image(s).")
        # if x0_preds is not None:
        #     print(f"        x0 steps: {len(x0_preds)}")


if __name__ == "__main__":
    main()
