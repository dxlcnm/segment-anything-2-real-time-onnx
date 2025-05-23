from typing import Any
import argparse
import pathlib

import torch
from torch import nn
import onnx
from onnxsim import simplify
from sam2.build_sam import build_sam2, build_sam2_camera_predictor
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import MLP
import torch.nn.functional as F
import sys
import os
import hydra
import cv2
import numpy as np
import onnxruntime
sys.path.append(os.path.dirname(__file__))
# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

def perpare_data(
        img,
        image_size=1024,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
    ):
        if isinstance(img, np.ndarray):
            img_np = img
            img_np = cv2.resize(img_np, (image_size, image_size)) / 255.0
            height, width = img.shape[:2]
        else:
            img_np = (
                np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0
            )
            width, height = img.size
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()

        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img -= img_mean
        img /= img_std
        return img, width, height

class SAM2Parameters(nn.Module):
    def __init__(self, sam_model: SAM2Base) ->None:
        super().__init__()
        
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.no_mem_embed = sam_model.no_mem_embed
        self.no_mem_pos_enc = sam_model.no_mem_pos_enc
        self.no_obj_ptr = sam_model.no_obj_ptr
        self.no_obj_embed_spatial = sam_model.no_obj_embed_spatial
    
    def forward(self, dummpy_input: torch.Tensor) -> tuple[Any, Any, Any, Any, Any]:
        # dummpy_input = torch.empty(1): [1]
        # maskmem_tpos_enc: [7, 1, 1,64]
        # no_mem_embed: [1, 1, 256]
        # no_mem_pos_enc: [1, 1, 256]
        # no_obj_ptr: [1, 256]
        # no_obj_embed_spatial: [1, 64]
        return self.maskmem_tpos_enc, self.no_mem_embed, self.no_mem_pos_enc, self.no_obj_ptr, self.no_obj_embed_spatial

class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = self.model.image_encoder
        self.no_mem_embed = self.model.no_mem_embed
        self.no_mem_pos_enc = self.model.no_mem_pos_enc

    def forward(self, x: torch.Tensor) -> tuple[Any, Any, Any, Any, Any, Any]:
        # x: (1, 3, 1024, 1024)
        backbone_out = self.image_encoder(x)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][
            -self.model.num_feature_levels :
        ]
        vision_pos_embeds = backbone_out["vision_pos_enc"][
            -self.model.num_feature_levels :
        ]
        
        high_res_features_0 = feature_maps[0]
        high_res_features_1 = feature_maps[1]


        # original feat
        cur_vision_feat = feature_maps[-1]
        cur_vision_pos_embed = vision_pos_embeds[-1]
        # high_res_features_0   [1, 32, 256, 256]  
        # high_res_features_1   [1, 64, 128, 128]
        # cur_vision_feat   [1, 256, 64, 64]
        # cur_vision_pos_embed [1, 256, 64, 64]
        return high_res_features_0, high_res_features_1, cur_vision_feat, cur_vision_pos_embed


class SAM2ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base, multimask_output: bool) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.img_size = sam_model.image_size
        self.multimask_output = multimask_output
        
        self.obj_ptr_proj = sam_model.obj_ptr_proj
        self.pred_obj_scores = sam_model.pred_obj_scores
        self.soft_no_obj_ptr = sam_model.soft_no_obj_ptr
        self.fixed_no_obj_ptr = sam_model.fixed_no_obj_ptr
        if sam_model.pred_obj_scores and sam_model.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = sam_model.no_obj_ptr

    @torch.no_grad()
    def forward(
        self,
        cur_vison_feat: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        # cur_vis_feat: [1, 256, 64, 64]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
        # point_coords: [num_labels, num_points, 2]
        # point_labels: [num_labels, num_points, 1]
        # mask_input: [1, 1, 256, 256]
        # has_mask_input: [1]
        B = cur_vison_feat.shape[0]
        device = cur_vison_feat.device
        sparse_embedding = self.model.sam_prompt_encoder._embed_points(point_coords, point_labels, True)
        sparse_embedding0 = self._embed_points(point_coords, point_labels)
        # print("sparse embedding compare: ", torch.allclose(sparse_embedding, sparse_embedding0))
        dense_embedding = self._embed_masks(mask_input, has_mask_input)
        
            
        high_res_feats = [high_res_feats_0, high_res_feats_1]
        image_embed = cur_vison_feat # [1, 256, 64, 64]
        
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=self.multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_feats,
        )
        
        
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if self.multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token) # [1, 256]
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        # low_res_masks: [1, 1, 256, 256],  need to sigmoid and binarilization
        # high_res_masks: [1, 1, 1024, 1024], need to sigmoid and binarilization
        # obj_ptr: [1, 256]
        # object_score_logits: [1, 1]
        return low_res_masks, high_res_masks, obj_ptr, object_score_logits
        


    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:

        point_coords = point_coords + 0.5

        padding_point = torch.zeros(
            (point_coords.shape[0], 1, 2), device=point_coords.device
        )
        padding_label = -torch.ones(
            (point_labels.shape[0], 1), device=point_labels.device
        )
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)
        print("point_coords shape: ", point_coords.shape)
        print("point_labels shape: ", point_labels.shape)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(
            point_coords
        )
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.prompt_encoder.not_a_point_embed.weight
            * (point_labels == -1)
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding
    
class SAM2MemEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
    
    @torch.no_grad()
    def forward(self, 
                cur_vision_feat: torch.Tensor,
                high_res_masks: torch.Tensor,
                object_score_logits: torch.Tensor,
                )->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # cur_vision_feat: [1, 256, 64, 64] cur_vision_feat
        # high_res_masks: [1, 1, 1024, 1024] high_res_masks

        # object_score_logits [1, 1]
        # NxCxHxW -> HWxNxC = [1, 256, 64, 64] -> [4096, 1, 256]
        pix_feat_reshape = cur_vision_feat.flatten(2).permute(2, 0, 1).contiguous()
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=pix_feat_reshape,
            feat_sizes = self.feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False
        )
        # maskmem_features: [1, 64, 64, 64]
        # maskmem_pos_enc: [1, 64, 64, 64]
        
        return maskmem_features, maskmem_pos_enc[0]
        

class SAM2MemAttention(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.mem_dim = self.model.mem_dim
        self.add_tpos_enc_to_obj_ptrs = self.model.add_tpos_enc_to_obj_ptrs

        self.proj_tpos_enc_in_obj_ptrs = self.model.proj_tpos_enc_in_obj_ptrs
        self.max_obj_ptrs_in_encoder = self.model.max_obj_ptrs_in_encoder
        self.memory_attention = sam_model.memory_attention
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
        
        self.obj_ptr_tpos_proj = self.model.obj_ptr_tpos_proj
    
    def get_1d_sine_pe(self, pos_inds, dim, temperature=10000):
        """
        Get 1D sine positional embedding as in the original Transformer paper.
        """
        pe_dim = dim // 2
        dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
        dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

        pos_embed = pos_inds.unsqueeze(-1) / dim_t
        pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
        return pos_embed
    
    @torch.no_grad()
    def forward(self,
                current_vision_feat: torch.Tensor,      #[1, 256, 64, 64], cur frame vison feat
                current_vision_pos_embed: torch.Tensor, #[1, 256, 64, 64], cur_vision_pos_embed
                maskmem_features:torch.Tensor,          # [64*64*N, 1, 64] from [N, 64, 64, 64], cur frame mask memory encoder feature, 
                objmem_ptrs:torch.Tensor,                  # [N, 1, 256], N frames obj_ptr_tokens
                maskmem_pos_enc:torch.Tensor,           # [64*64*N, 1, 64] from [N, 64, 64, 64], cur frame mask memory position encoding, [64*64*N, 1, 64]
                objmem_pos_indice:torch.Tensor,          # [N] obj_ptr indices
                num_frames: torch.Tensor, # [1]
                ) -> torch.Tensor:

        cur_feats= current_vision_feat.flatten(2).permute(2, 0, 1).contiguous() # [4096, 1, 256]
        cur_pos = current_vision_pos_embed.flatten(2).permute(2, 0, 1).contiguous() #[4096, 1, 256]
        
        B = current_vision_feat.size(0)  # batch size on this frame
        C = self.model.hidden_dim
        H, W = self.feat_sizes[-1] # top-level (lowest-resolution) feature size
        device = current_vision_feat.device
        
        max_obj_ptrs_in_encoder = torch.minimum(
                                    num_frames, torch.tensor(self.max_obj_ptrs_in_encoder, device=num_frames.device)
                                )
        if self.add_tpos_enc_to_obj_ptrs:
            t_diff_max = max_obj_ptrs_in_encoder - 1
            tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
            obj_pos = objmem_pos_indice.detach().clone().to(
                            device=device, non_blocking=True
                        )
            obj_pos = self.get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
            obj_pos = self.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
            
        if self.mem_dim < C:
            # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
            obj_ptrs = objmem_ptrs.reshape(
                -1, B, C // self.mem_dim, self.mem_dim
            )
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1) # [4*N, 1, 64]
            obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0) # [4*N, 1, 64]

        num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory = torch.cat((maskmem_features, obj_ptrs),dim=0)
        memory_pos = torch.cat((maskmem_pos_enc, obj_pos),dim=0)
        pix_feat_with_mem = self.memory_attention(
            curr = cur_feats,
            curr_pos = cur_pos,
            memory = memory,
            memory_pos = memory_pos,
            num_obj_ptr_tokens= num_obj_ptr_tokens,
        )
        # reshape the output (HW)xBxC => BxCxHxW
        current_vision_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W) # [1,256,64,64]
        return current_vision_feat_with_mem # [1,256,64,64]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the SAM2 prompt encoder and mask decoder to an ONNX model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="The path to the SAM model checkpoint.",
    )
    
    parser.add_argument(
        "--output_parameters",
        type=str,
        required=False,
        help="The filename to save the parameters ONNX model to.",
    )

    parser.add_argument(
        "--output_encoder",
        type=str,
        required=False,
        help="The filename to save the encoder ONNX model to.",
    )

    parser.add_argument(
        "--output_decoder",
        type=str,
        required=False,
        help="The filename to save the decoder ONNX model to.",
    )
    
    parser.add_argument(
        "--output_memory_encoder",
        type=str,
        required=False,
        help="The filename to save the memory encoder ONNX model to.",
    )
        
    parser.add_argument(
        "--output_memory_attention",
        type=str,
        required=False,
        help="The filename to save the memory attention ONNX model to.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        help="In the form of sam2_hiera_{tiny, small, base_plus, large}.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="The ONNX opset version to use. Must be >=11",
    )

    args = parser.parse_args()
    
    args.checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    args.output_parameters = "output_models/sam2.1_hiera_tiny.parameters.onnx"
    args.output_encoder = "output_models/sam2.1_hiera_tiny.encoder.onnx"
    args.output_decoder = "output_models/sam2.1_hiera_tiny.decoder.onnx"
    args.output_memory_encoder = "output_models/sam2.1_hiera_tiny.memoryencoder.onnx"
    args.output_memory_attention = "output_models/sam2.1_hiera_tiny.memoryattention.onnx"
    args.model_type = "sam2.1_hiera_tiny"

    input_size = (1024, 1024)
    multimask_output = True
    model_type = args.model_type
    if model_type == "sam2_hiera_tiny":
        model_cfg = "sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small":
        model_cfg = "sam2_hiera_s.yaml"
    elif model_type == "sam2_hiera_base_plus":
        model_cfg = "sam2_hiera_b+.yaml"
    elif model_type == "sam2_hiera_large":
        model_cfg = "sam2_hiera_l.yaml"
    elif model_type == "sam2.1_hiera_tiny":
        model_cfg = "sam2.1_hiera_t.yaml"
    elif model_type == "sam2.1_hiera_small":
        model_cfg = "sam2.1_hiera_s.yaml"
    elif model_type == "sam2.1_hiera_base_plus":
        model_cfg = "sam2.1_hiera_b+.yaml"
    elif model_type == "sam2.1_hiera_large":
        model_cfg = "sam2.1_hiera_l.yaml"
    import os
    
    
    device = torch.device("cpu")
    model_cfg_new = os.path.join("configs/sam2.1/", model_cfg)
    sam2_model = build_sam2_camera_predictor(model_cfg_new, args.checkpoint, device=device)
    
    
    sam2_parameters = SAM2Parameters(
        sam2_model
    ).to(device)
    
    dummpy_input = torch.empty(1).to(device)
    maskmem_tpos_enc, no_mem_embed, no_mem_pos_enc, no_obj_ptr, no_obj_embed_spatial = sam2_parameters(
        dummpy_input
    )
    
    # maskmem_tpos_enc: [7, 1, 1,64]
    # no_mem_embed: [1, 1, 256]
    # no_mem_pos_enc: [1, 1, 256]
    # no_obj_ptr: [1, 256]
    # no_obj_embed_spatial: [1, 64]
    
    print("parameters maskmem_tpos_enc shape: ", maskmem_tpos_enc.shape)
    print("parameters no_mem_embed shape: ", no_mem_embed.shape)
    print("parameters no_mem_pos_enc shape: ", no_mem_pos_enc.shape)
    print("parameters no_obj_ptr shape: ", no_obj_ptr.shape)
    print("parameters no_obj_embed_spatial shape: ", no_obj_embed_spatial.shape)
    pathlib.Path(args.output_memory_encoder).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        sam2_parameters,
        (
            dummpy_input,
        ),
        args.output_parameters,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[
            "dummpy_input",
        ],
        output_names=["maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc", "no_obj_ptr", "no_obj_embed_spatial"],
    )
    print("Saved parameters to", args.output_parameters)
    print("Simplifying parameters...")
    onnx_model = onnx.load(args.output_parameters)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_parameters)
    print("Saved simplified parameters to", args.output_parameters)
    
    
    img = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    sam2_encoder = SAM2ImageEncoder(sam2_model).to(device)
    high_res_feats_0, high_res_feats_1, cur_vision_feat, cur_vision_pos_embed = sam2_encoder(img)
    print("high_res_feats_0 shape: ", high_res_feats_0.shape)
    print("high_res_feats_1 shape: ", high_res_feats_1.shape)
    print("cur_vision_feat shape: ", cur_vision_feat.shape)
    print("cur_vision_pos_embed shape: ", cur_vision_pos_embed.shape)

    pathlib.Path(args.output_encoder).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        sam2_encoder,
        img,
        args.output_encoder,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["high_res_feats_0", "high_res_feats_1", "cur_vision_feat", "cur_vision_pos_embed"],
    )
    print("Saved encoder to", args.output_encoder)
    print("Simplifying encoder...")
    onnx_model = onnx.load(args.output_encoder)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_encoder)
    print("Saved simplified encoder to", args.output_encoder)

    sam2_decoder = SAM2ImageDecoder(
        sam2_model, multimask_output=multimask_output
    ).to(device)

    embed_dim = sam2_model.sam_prompt_encoder.embed_dim
    embed_size = (
        sam2_model.image_size // sam2_model.backbone_stride,
        sam2_model.image_size // sam2_model.backbone_stride,
    )
    mask_input_size = [4 * x for x in embed_size]
    print(f"embed dim: {embed_dim}, embed size: {embed_size}, mask inut size: {mask_input_size}")

    point_coords = torch.randint(
        low=0, high=input_size[1], size=(1, 5, 2), dtype=torch.float
    ).to(device)
    point_labels = torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float).to(device)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float).to(device)
    has_mask_input = torch.tensor([0], dtype=torch.float).to(device)
    orig_im_size = torch.tensor(
        [input_size[0], input_size[1]], dtype=torch.float
    )

    
    low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam2_decoder(
        cur_vision_feat,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
        mask_input,
        has_mask_input,
    )
    
    print("decoder low_res_masks shape: ", low_res_masks.shape)
    print("decoder high_res_masks shape: ", high_res_masks.shape)
    print("decoder obj_ptr shape: ", obj_ptr.shape)
    print("decoder object_score_logits shape: ", object_score_logits.shape)

    pathlib.Path(args.output_decoder).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        sam2_decoder,
        (
            cur_vision_feat,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
        ),
        args.output_decoder,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[
            "cur_vision_feat",
            "high_res_feats_0",
            "high_res_feats_1",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        ],
        output_names=["low_res_masks", "high_res_masks", "obj_ptr", "object_score_logits"],
        dynamic_axes={
            "point_coords": {0: "num_labels", 1: "num_points"},
            "point_labels": {0: "num_labels", 1: "num_points"},
            "mask_input": {0: "num_labels"},
            "has_mask_input": {0: "num_labels"},
        },
    )
                
    print("Saved decoder to", args.output_decoder)
    print("Simplifying decoder...")
    onnx_model = onnx.load(args.output_decoder)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_decoder)
    print("Saved simplified decoder to", args.output_decoder)
    
    
    sam2_memory_encoder = SAM2MemEncoder(
        sam2_model
    ).to(device)
    
    maskmem_features, maskmem_pos_enc = sam2_memory_encoder(
        cur_vision_feat,
        high_res_masks,
        object_score_logits,
    )
    print("memory encoder maskmem_features shape: ", maskmem_features.shape)
    print("memory encoder maskmem_pos_enc shape: ", maskmem_pos_enc.shape)
    pathlib.Path(args.output_memory_encoder).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        sam2_memory_encoder,
        (
            cur_vision_feat,
            high_res_masks,
            object_score_logits,
        ),
        args.output_memory_encoder,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[
            "cur_vision_feat",
            "high_res_masks",
            "object_score_logits",
        ],
        output_names=["maskmem_features", "maskmem_pos_enc"],
    )
    print("Saved memory encoder to", args.output_memory_encoder)
    print("Simplifying memory encoder...")
    onnx_model = onnx.load(args.output_memory_encoder)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_memory_encoder)
    print("Saved simplified memory encoder to", args.output_memory_encoder)
    
    
    
    sam2_memory_attention = SAM2MemAttention(
        sam2_model
    ).to(device)
    
    
    N = 5
    maskmem_features = torch.rand(64*64*N, 1, 64, dtype=torch.float).to(device)
    objmem_ptrs = torch.rand(N, 1, 256, dtype=torch.float).to(device)
    maskmem_pos_enc = torch.rand(64*64*N, 1, 64, dtype=torch.float).to(device)
    objmem_pos_indice = torch.randint(low=1, high=10, size=(N,)).to(device)
    num_frames = torch.tensor([10], dtype=torch.float).to(device)
    
    cur_vision_feat_with_mem = sam2_memory_attention(
                cur_vision_feat,      #[1, 256, 64, 64], cur frame vison feat
                cur_vision_pos_embed, #[1, 256, 64, 64], cur_vision_pos_embed
                maskmem_features,          #[N, 64, 64, 64], cur frame mask memory encoder feature
                objmem_ptrs,                  # [N, 1, 256], N frames obj_ptr_tokens
                maskmem_pos_enc,           # [N, 64, 64, 64], cur frame mask memory position encoding
                objmem_pos_indice,          # [N] obj_ptr indices
                num_frames, # [1]
    )
    print("memory attention cur_vision_feat_with_mem shape: ", cur_vision_feat_with_mem.shape)
    
    pathlib.Path(args.output_memory_attention).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        sam2_memory_attention,
        (
            cur_vision_feat,      #[1, 256, 64, 64], cur frame vison feat
            cur_vision_pos_embed, #[1, 256, 64, 64], cur_vision_pos_embed
            maskmem_features,          #[N, 64, 64, 64], cur frame mask memory encoder feature
            objmem_ptrs,                  # [N, 1, 256], N frames obj_ptr_tokens
            maskmem_pos_enc,           # [N, 64, 64, 64], cur frame mask memory position encoding
            objmem_pos_indice,          # [N] obj_ptr indices
            num_frames, # [1]
        ),
        args.output_memory_attention,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[
            "cur_vision_feat",
            "cur_vision_pos_embed",
            "maskmem_features",
            "objmem_ptrs",
            "maskmem_pos_enc",
            "objmem_pos_indice",
            "num_frames"
        ],
        output_names=["cur_vision_feat_with_mem"],
        dynamic_axes={
            "maskmem_features": {0: "num_memory_frames"},
            "objmem_ptrs": {0: "num_memory_frames"},
            "maskmem_pos_enc": {0: "num_memory_frames"},
            "objmem_pos_indice": {0: "num_memory_frames"},
        }
    )
    print("Saved memory attention to", args.output_memory_attention)
    print("Simplifying memory attention...")
    onnx_model = onnx.load(args.output_memory_attention)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_memory_attention)
    print("Saved simplified memory attention to", args.output_memory_attention)

