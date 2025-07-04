from typing import List, Tuple, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.minicpmo import MiniCPMO
from sglang.srt.models.minicpmv import MiniCPMV
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


# Compatible with both 'O' and 'V'
class MiniCPMMultimodalProcessor(BaseMultimodalProcessor):
    models = [MiniCPMV, MiniCPMO]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.image_token = "(<image>./</image>)"
        self.audio_token = "(<audio>./</audio>)"
        
        # Set token IDs for process_and_combine_mm_data
        tokenizer = _processor.tokenizer
        self.IM_TOKEN_ID = tokenizer.unk_id
        self.AUDIO_TOKEN_ID = getattr(tokenizer, "audio_token_id", None)

    async def _process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.image_token,
                audio_token=self.audio_token,
            ),
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            audios=base_output.audios,
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        slice_start_id, slice_end_id, audio_start_id, audio_end_id = (
            None,
            None,
            None,
            None,
        )
        if tokenizer.slice_start_id:
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id
        if hasattr(tokenizer, "audio_start_id"):
            audio_start_id = tokenizer.audio_start_id
            audio_end_id = tokenizer.audio_end_id

        im_start_id = tokenizer.im_start_id
        im_end_id = tokenizer.im_end_id
        im_token_id = tokenizer.unk_id
        pixel_values = res["pixel_values"]
        tgt_sizes = res["tgt_sizes"]

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
            )

        if len(pixel_values) != len(tgt_sizes):
            raise ValueError(
                "Inconsistent batch lengths, found: "
                f"{len(pixel_values)} vs. {len(tgt_sizes)}"
            )

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
            # per image
            if len(pixel_b) != len(tgt_b):
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                pixel_values_flat += [pixel_n]
                tgt_sizes_flat += [tgt_n]

        pixel_values = pixel_values_flat

        items = []
        input_ids = res["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset_by_pair(
            input_ids=input_ids, mm_start_id=im_start_id, mm_end_id=im_end_id
        )
        slice_offsets = self.get_mm_items_offset_by_pair(
            input_ids=input_ids, mm_start_id=slice_start_id, mm_end_id=slice_end_id
        )
        image_offsets.extend(slice_offsets)
        image_offsets = sorted(image_offsets)

        if len(pixel_values) != 0:
            item = MultimodalDataItem(
                pixel_values=pixel_values,
                image_offsets=image_offsets,
                tgt_size=tgt_sizes_flat,
                modality=Modality.IMAGE,
            )
            items += [item]

        if (
            "audio_features" in res
            and res["audio_features"] is not None
            and len(res["audio_features"]) != 0
        ):
            if audio_start_id is not None and audio_end_id is not None:
                audio_offsets = self.get_mm_items_offset_by_pair(
                    input_ids=input_ids,
                    mm_start_id=audio_start_id,
                    mm_end_id=audio_end_id,
                )
            else:
                audio_offsets = None
            item = MultimodalDataItem(
                audio_features=[res["audio_features"]],
                audio_feature_lens=res["audio_feature_lens"],
                audio_offsets=audio_offsets,
                modality=Modality.AUDIO,
            )
            items += [item]

        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": audio_start_id,
            "audio_end_id": audio_end_id,
            "im_token_id": im_token_id,
            "im_start_id": im_start_id,
            "im_end_id": im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }

    async def old_process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.image_token,
                audio_token=self.audio_token,
            ),
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            audios=base_output.audios,
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        slice_start_id, slice_end_id, audio_start_id, audio_end_id = (
            None,
            None,
            None,
            None,
        )
        if tokenizer.slice_start_id:
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id
        if hasattr(tokenizer, "audio_start_id"):
            audio_start_id = tokenizer.audio_start_id
            audio_end_id = tokenizer.audio_end_id

        im_start_id = tokenizer.im_start_id
        im_end_id = tokenizer.im_end_id
        im_token_id = tokenizer.unk_id
        pixel_values = res["pixel_values"]
        tgt_sizes = res["tgt_sizes"]

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
            )

        if len(pixel_values) != len(tgt_sizes):
            raise ValueError(
                "Inconsistent batch lengths, found: "
                f"{len(pixel_values)} vs. {len(tgt_sizes)}"
            )

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
            # per image
            if len(pixel_b) != len(tgt_b):
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                pixel_values_flat += [pixel_n]
                tgt_sizes_flat += [tgt_n]

        pixel_values = pixel_values_flat

        items = []
        input_ids = res["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset_by_pair(
            input_ids=input_ids, mm_start_id=im_start_id, mm_end_id=im_end_id
        )
        slice_offsets = self.get_mm_items_offset_by_pair(
            input_ids=input_ids, mm_start_id=slice_start_id, mm_end_id=slice_end_id
        )
        image_offsets.extend(slice_offsets)
        image_offsets = sorted(image_offsets)

        if len(pixel_values) != 0:
            item = MultimodalDataItem(
                pixel_values=pixel_values,
                image_offsets=image_offsets,
                tgt_size=tgt_sizes_flat,
                modality=Modality.IMAGE,
            )
            items += [item]

        if (
            "audio_features" in res
            and res["audio_features"] is not None
            and len(res["audio_features"]) != 0
        ):
            if audio_start_id is not None and audio_end_id is not None:
                audio_offsets = self.get_mm_items_offset_by_pair(
                    input_ids=input_ids,
                    mm_start_id=audio_start_id,
                    mm_end_id=audio_end_id,
                )
            else:
                audio_offsets = None
            item = MultimodalDataItem(
                audio_features=[res["audio_features"]],
                audio_feature_lens=res["audio_feature_lens"],
                audio_offsets=audio_offsets,
                modality=Modality.AUDIO,
            )
            items += [item]

        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": audio_start_id,
            "audio_end_id": audio_end_id,
            "im_token_id": im_token_id,
            "im_start_id": im_start_id,
            "im_end_id": im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }


    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        # Check if all data is precomputed (dict format)
        all_precomputed = (
            all(isinstance(item, dict) or item is None for item in image_data or []) 
            and
            all(isinstance(item, dict) or item is None for item in audio_data or [])
        )
        
        if all_precomputed:
            # Handle precomputed features directly
            if isinstance(input_text, list):
                prompt = self._processor.tokenizer.decode(input_text)
            else:
                prompt = input_text
                
            base_output = BaseMultiModalProcessorOutput(
                input_text=prompt,
                images=image_data,
                audios=audio_data or [],
            )
            
            mm_items, input_ids = self._process_precomputed_features(base_output)
        else:
            # Normal processing with load_mm_data
            base_output = self.load_mm_data(
                prompt=input_text,
                max_req_input_len=max_req_input_len,
                audio_data=audio_data,
                image_data=image_data,
                multimodal_tokens=MultimodalSpecialTokens(
                    image_token=self.image_token,
                    audio_token=self.audio_token,
                ),
            )
            if base_output is None:
                return None
                
            # Use process_and_combine_mm_data for normal processing
            mm_items, input_ids = self.process_and_combine_mm_data(base_output)

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        slice_start_id, slice_end_id, audio_start_id, audio_end_id = (
            None,
            None,
            None,
            None,
        )
        if hasattr(tokenizer, "slice_start_id"):
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id
        if hasattr(tokenizer, "audio_start_id"):
            audio_start_id = tokenizer.audio_start_id
            audio_end_id = tokenizer.audio_end_id

        im_start_id = tokenizer.im_start_id
        im_end_id = tokenizer.im_end_id
        im_token_id = tokenizer.unk_id

        # Process mm_items to match the expected format
        processed_items = []
        for item in mm_items:
            if item.modality == Modality.IMAGE:
                # Handle pixel_values and tgt_sizes
                if hasattr(item, "pixel_values") and item.pixel_values is not None:
                    pixel_values = item.pixel_values
                    tgt_sizes = getattr(item, "tgt_size", [])
                    
                    # Update image offsets to include slice offsets
                    image_offsets = self.get_mm_items_offset_by_pair(
                        input_ids=input_ids, mm_start_id=im_start_id, mm_end_id=im_end_id
                    )
                    if slice_start_id is not None and slice_end_id is not None:
                        slice_offsets = self.get_mm_items_offset_by_pair(
                            input_ids=input_ids, mm_start_id=slice_start_id, mm_end_id=slice_end_id
                        )
                        image_offsets.extend(slice_offsets)
                    image_offsets = sorted(image_offsets)
                    
                    # Create new item with updated offsets
                    new_item = MultimodalDataItem(
                        pixel_values=pixel_values,
                        image_offsets=image_offsets,
                        tgt_size=tgt_sizes,
                        modality=Modality.IMAGE,
                    )
                    processed_items.append(new_item)
            elif item.modality == Modality.AUDIO:
                # Handle audio items
                if audio_start_id is not None and audio_end_id is not None:
                    audio_offsets = self.get_mm_items_offset_by_pair(
                        input_ids=input_ids,
                        mm_start_id=audio_start_id,
                        mm_end_id=audio_end_id,
                    )
                else:
                    audio_offsets = None
                
                # Update audio offsets
                item.audio_offsets = audio_offsets
                processed_items.append(item)

        return {
            "mm_items": processed_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": audio_start_id,
            "audio_end_id": audio_end_id,
            "im_token_id": im_token_id,
            "im_start_id": im_start_id,
            "im_end_id": im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }

    def _process_precomputed_features(
        self, base_output: BaseMultiModalProcessorOutput
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor]:
        """Process precomputed features for MiniCPM models."""
        # Tokenize input text
        input_ids = self._processor.tokenizer(
            base_output.input_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.flatten()
        
        # Process precomputed items
        mm_items = []
        for item in base_output.images or []:
            if isinstance(item, dict):
                modality_str = item.get("modality", "IMAGE")
                modality = Modality.from_str(modality_str)
                
                if modality == Modality.IMAGE:
                    # Extract pixel values and tgt_sizes
                    pixel_values = item.get("pixel_values")
                    tgt_sizes = item.get("tgt_sizes", item.get("tgt_size", []))
                    
                    if pixel_values is not None:
                        # Calculate offsets
                        tokenizer = self._processor.tokenizer
                        im_start_id = tokenizer.im_start_id
                        im_end_id = tokenizer.im_end_id
                        slice_start_id = getattr(tokenizer, "slice_start_id", None)
                        slice_end_id = getattr(tokenizer, "slice_end_id", None)
                        
                        image_offsets = self.get_mm_items_offset_by_pair(
                            input_ids=input_ids, mm_start_id=im_start_id, mm_end_id=im_end_id
                        )
                        
                        if slice_start_id is not None and slice_end_id is not None:
                            slice_offsets = self.get_mm_items_offset_by_pair(
                                input_ids=input_ids, mm_start_id=slice_start_id, mm_end_id=slice_end_id
                            )
                            image_offsets.extend(slice_offsets)
                        image_offsets = sorted(image_offsets)
                        
                        # Handle pixel_values format
                        if isinstance(pixel_values, list):
                            # If it's already a list, use it as is
                            pass
                        elif isinstance(pixel_values, torch.Tensor) and pixel_values.dim() >= 2:
                            # If it's a single image tensor, wrap it in a list
                            pixel_values = [pixel_values]
                        
                        # Handle tgt_sizes format
                        if isinstance(tgt_sizes, torch.Tensor):
                            # If it's a 2D tensor with multiple sizes, split it
                            if tgt_sizes.dim() == 2:
                                tgt_sizes = [tgt_sizes[i] for i in range(tgt_sizes.size(0))]
                            else:
                                tgt_sizes = [tgt_sizes]
                        elif not isinstance(tgt_sizes, list):
                            # Default tgt_size if not provided
                            tgt_sizes = []
                        
                        mm_item = MultimodalDataItem(
                            pixel_values=pixel_values,
                            image_offsets=image_offsets,
                            tgt_size=tgt_sizes,
                            modality=Modality.IMAGE,
                        )
                        mm_items.append(mm_item)
        
        # Process audio items
        for item in base_output.audios or []:
            if isinstance(item, dict):
                modality_str = item.get("modality", "AUDIO")
                modality = Modality.from_str(modality_str)
                
                if modality == Modality.AUDIO:
                    audio_features = item.get("audio_features")
                    audio_feature_lens = item.get("audio_feature_lens")
                    
                    if audio_features is not None:
                        tokenizer = self._processor.tokenizer
                        audio_start_id = getattr(tokenizer, "audio_start_id", None)
                        audio_end_id = getattr(tokenizer, "audio_end_id", None)
                        
                        if audio_start_id is not None and audio_end_id is not None:
                            audio_offsets = self.get_mm_items_offset_by_pair(
                                input_ids=input_ids,
                                mm_start_id=audio_start_id,
                                mm_end_id=audio_end_id,
                            )
                        else:
                            audio_offsets = None
                        
                        mm_item = MultimodalDataItem(
                            audio_features=[audio_features],
                            audio_feature_lens=audio_feature_lens,
                            audio_offsets=audio_offsets,
                            modality=Modality.AUDIO,
                        )
                        mm_items.append(mm_item)
        
        return mm_items, input_ids 