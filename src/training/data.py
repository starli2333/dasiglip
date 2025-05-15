# -*- coding: utf-8 -*-
import ast
import io
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from typing import Optional, Tuple, Any, Dict, Callable, Union
import argparse

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
# from webdataset.filters import _shuffle # Using a local copy or ensuring it's available
# from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample # Using local copies or ensuring available

from transformers import AutoProcessor, PreTrainedTokenizerBase

try:
    from open_clip.constants import NUM_DEGRADATION_TYPES, DEGRADATION_TYPES, DEGRADATION_TO_ID, DEGRADATION_TEXT_DESCRIPTIONS
except ImportError:
    DEGRADATION_TYPES = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed', 'snowy', 'uncompleted']
    NUM_DEGRADATION_TYPES = len(DEGRADATION_TYPES)
    DEGRADATION_TO_ID = {name: i for i, name in enumerate(DEGRADATION_TYPES)}
    DEGRADATION_TEXT_DESCRIPTIONS = [f"a {name} image" for name in DEGRADATION_TYPES]
    logging.warning("Could not import degradation constants from open_clip.constants. Using default values.")

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# --- Re-add webdataset helper functions if they were removed or ensure they are correctly imported ---
def _shuffle(src, bufsize, initial, rng):
    buf = []
    for x in src:
        if len(buf) < bufsize:
            buf.append(x)
        else:
            r = rng.randint(0, len(buf) -1)
            yield buf[r]
            buf[r] = x
    rng.shuffle(buf)
    for x in buf:
        yield x

def base_plus_ext(path):
    return os.path.splitext(path)

def url_opener(data, handler=wds.reraise_exception, **kw):
    return wds.tariterators.url_opener(data, handler=handler, **kw)

def tar_file_expander(data, handler=wds.reraise_exception, **kw):
    return wds.tariterators.tar_file_expander(data, handler=handler, **kw)

def valid_sample(sample):
    return sample is not None and "__key__" in sample and isinstance(sample, dict)
# --- End Webdataset helpers ---

def random_crop(pil_image, low_size=64):
    h, w = pil_image.size
    if h <= low_size or w <= low_size:
        return pil_image
    size = random.randint(low_size, min(h, w))
    rnd_h = random.randint(0, max(0, h - size))
    rnd_w = random.randint(0, max(0, w - size))
    return pil_image.crop((rnd_h, rnd_w, rnd_h + size, rnd_w + size))

class CsvDataset(Dataset):
    def __init__(
        self,
        input_filename: str,
        img_key: str,
        # caption_key now refers to the column with "caption:degradation_str"
        caption_key: str,
        # degradation_key is now effectively ignored for CSV loading,
        # but kept for consistency in function signature if args are passed directly.
        # The actual degradation string will be parsed from the caption_key column.
        degradation_key: Optional[str] = None, # Made optional
        sep: str = "\t",
        caption_degradation_separator: str = ":", # Separator used in the 'title' column
        crop: bool = False,
        args: Optional[argparse.Namespace] = None,
        processor: Optional[Any] = None,
        image_transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None
    ):
        logging.info(f'Loading csv data from {input_filename}.')
        try:
            df = pd.read_csv(input_filename, sep=sep)
        except Exception as e:
            logging.error(f"Error loading CSV {input_filename}: {e}")
            raise e

        self.images = df[img_key].tolist()
        # This column now contains "caption:degradation_str"
        self.raw_titles_with_degradation = df[caption_key].tolist()
        # self.degradation_labels_str is no longer directly loaded from a separate column

        self.processor = processor
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.caption_degradation_separator = caption_degradation_separator
        self.crop = crop
        self.args = args

        self.degradation_map = DEGRADATION_TO_ID
        self.num_degradation_types = NUM_DEGRADATION_TYPES
        self.degradation_text_descriptions = DEGRADATION_TEXT_DESCRIPTIONS

        if self.processor is None and (self.image_transform is None or self.tokenizer is None):
            raise ValueError("CsvDataset requires either a 'processor' or both 'image_transform' and 'tokenizer'.")

        logging.info(f'Found {len(self.images)} samples in {input_filename}.')

    def __len__(self):
        return len(self.images)

    def _parse_title_and_degradation(self, raw_title_str: str) -> Tuple[str, str]:
        """
        Parses the raw title string to extract caption and degradation.
        Assumes format "caption_text:degradation_type_string".
        """
        parts = raw_title_str.rsplit(self.caption_degradation_separator, 1) # Split from the right, once
        if len(parts) == 2:
            caption = parts[0].strip()
            degradation = parts[1].strip().lower()
            # Validate if the parsed degradation is known
            if degradation not in self.degradation_map:
                logging.warning(f"Parsed degradation '{degradation}' from '{raw_title_str}' is not in known DEGRADATION_TYPES. Treating as 'unknown' or using full title as caption.")
                # Fallback: use full title as caption, and a placeholder for degradation
                return raw_title_str.strip(), "unknown_degradation_placeholder" # Or handle as error
            return caption, degradation
        else:
            # If separator not found, assume the whole string is the caption
            # and degradation is unknown or needs a default.
            return raw_title_str.strip(), "unknown_degradation_placeholder" # Or a default known degradation if applicable

    def _tokenize_text(self, text: str, max_length: int =64) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Helper to tokenize text and return input_ids and attention_mask. """
        attention_mask = None
        if self.processor and hasattr(self.processor, 'tokenizer'):
            inputs = self.processor(
                text=[text], return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_length
            )
            tokens = inputs['input_ids'].squeeze(0)
            attention_mask = inputs.get('attention_mask', torch.ones_like(tokens)).squeeze(0)
        elif self.tokenizer:
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                 inputs = self.tokenizer(
                     text, return_tensors="pt", padding="max_length",
                     truncation=True, max_length=max_length
                 )
                 tokens = inputs['input_ids'].squeeze(0)
                 attention_mask = inputs.get('attention_mask', torch.ones_like(tokens)).squeeze(0)
            else: # open_clip.tokenize
                 tokens = self.tokenizer(text) # Returns (1, L) for open_clip.tokenize
                 if tokens.ndim == 2 and tokens.shape[0] == 1:
                     tokens = tokens.squeeze(0)
                 # SimpleTokenizer doesn't provide attention_mask, create one if needed or pass None
                 # For simplicity, let's assume models using simple tokenizer handle this.
                 attention_mask = torch.ones_like(tokens) if tokens is not None else None
        else:
            raise RuntimeError("No valid tokenizer found.")
        return tokens, attention_mask


    def __getitem__(self, idx):
        try:
            image_path = str(self.images[idx])
            raw_title = str(self.raw_titles_with_degradation[idx])

            # Parse caption and degradation from the raw title
            caption_str, degradation_label_str = self._parse_title_and_degradation(raw_title)

            try:
                pil_image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logging.warning(f"Could not load image {image_path}. Error: {e}. Returning dummy sample.")
                # Dummy data logic (same as before)
                dummy_image_size = (224, 224)
                if self.processor and hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'size'):
                    size_info = self.processor.image_processor.size
                    if isinstance(size_info, dict):
                        h = size_info.get('height', size_info.get('shortest_edge', 224))
                        w = size_info.get('width', size_info.get('shortest_edge', 224))
                        dummy_image_size = (h, w)
                    elif isinstance(size_info, int):
                        dummy_image_size = (size_info, size_info)
                elif self.image_transform and hasattr(self.image_transform, 'size'):
                     transform_size = self.image_transform.size
                     if isinstance(transform_size, int):
                         dummy_image_size = (transform_size, transform_size)
                     elif isinstance(transform_size, (list, tuple)) and len(transform_size) >= 2:
                         dummy_image_size = transform_size[-2:]

                dummy_image = torch.zeros((3, *dummy_image_size))
                dummy_caption_tokens = torch.zeros((64,), dtype=torch.long)
                dummy_attention_mask = torch.zeros((64,), dtype=torch.long)
                dummy_y_true = torch.zeros((self.num_degradation_types,))
                return dummy_image, {
                    "caption_tokens": dummy_caption_tokens,
                    "caption_attention_mask": dummy_attention_mask,
                    "degradation_target": dummy_y_true,
                    "true_degradation_text_tokens": None,
                    "true_degradation_text_attention_mask": None
                 }

            if self.crop and random.random() > 0.2:
                pil_image = random_crop(pil_image)

            if self.processor:
                image_inputs = self.processor(images=pil_image, return_tensors="pt")
                processed_image = image_inputs['pixel_values'].squeeze(0)
            elif self.image_transform:
                processed_image = self.image_transform(pil_image)
            else:
                raise RuntimeError("No valid image preprocessor found.")

            caption_tokens, caption_attention_mask = self._tokenize_text(caption_str)

            y_true = torch.zeros(self.num_degradation_types)
            if degradation_label_str in self.degradation_map:
                degradation_id = self.degradation_map[degradation_label_str]
                y_true[degradation_id] = 1.0
            else:
                # If 'unknown_degradation_placeholder' was returned by parser, log differently
                if degradation_label_str != "unknown_degradation_placeholder":
                    logging.warning(f"Unknown degradation label '{degradation_label_str}' (parsed from title) in sample {idx} (path: {image_path}). Target all zeros.")
                # else, already warned by parser

            degradation_text_tokens, degradation_text_attention_mask = None, None
            if self.args is not None and hasattr(self.args, 'degrad_contrastive_weight') and self.args.degrad_contrastive_weight > 0:
                 if degradation_label_str in self.degradation_map: # Only if degradation is known
                     degradation_id = self.degradation_map[degradation_label_str]
                     true_degradation_text = self.degradation_text_descriptions[degradation_id]
                     degradation_text_tokens, degradation_text_attention_mask = self._tokenize_text(true_degradation_text)

            text_dict = {
                "caption_tokens": caption_tokens,
                "caption_attention_mask": caption_attention_mask,
                "degradation_target": y_true,
                "true_degradation_text_tokens": degradation_text_tokens,
                "true_degradation_text_attention_mask": degradation_text_attention_mask,
            }
            return processed_image, text_dict

        except Exception as e:
             logging.error(f"CRITICAL Error processing sample index {idx}, title '{self.raw_titles_with_degradation[idx] if idx < len(self.raw_titles_with_degradation) else 'OOB'}': {e}", exc_info=True)
             dummy_image_size = (224,224)
             dummy_image = torch.zeros((3, *dummy_image_size))
             dummy_caption_tokens = torch.zeros((64,), dtype=torch.long)
             dummy_attention_mask = torch.zeros((64,), dtype=torch.long)
             dummy_y_true = torch.zeros((self.num_degradation_types,))
             return dummy_image, {
                 "caption_tokens": dummy_caption_tokens, "caption_attention_mask": dummy_attention_mask,
                 "degradation_target": dummy_y_true,
                 "true_degradation_text_tokens": None, "true_degradation_text_attention_mask": None
              }

# ... (Rest of the file: SharedEpoch, DataInfo, WebDataset functions, get_dataset_fn, get_data etc. remain the same as in data_py_corrected_for_tokenizer)
# Ensure to copy the full content from data_py_corrected_for_tokenizer for the parts not shown here.

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)
    def set_value(self, epoch):
        self.shared_epoch.value = epoch
    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: Optional[DistributedSampler] = None
    shared_epoch: Optional[SharedEpoch] = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else: # Assume urls is already a list
        all_urls = list(urls)
        return all_urls, weights if weights is not None else [1.0] * len(all_urls)


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    if not shards_list: return None, 0
    # Check if the first shard is a local path or URL
    if not (shards_list[0].startswith("http://") or shards_list[0].startswith("https://") or shards_list[0].startswith("s3://")):
        dir_path = os.path.dirname(shards_list[0])
        if not os.path.exists(dir_path): # Check if dir_path is valid
            logging.warning(f"Directory path for shards does not exist: {dir_path}. Cannot determine dataset size from local files.")
            return None, len(shards_list) # Return None for size, but count of shards

        sizes_filename = os.path.join(dir_path, 'sizes.json')
        len_filename = os.path.join(dir_path, '__len__')
        if os.path.exists(sizes_filename):
            try:
                sizes = json.load(open(sizes_filename, 'r'))
                total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list if os.path.basename(shard) in sizes])
            except Exception as e:
                logging.warning(f"Error reading sizes.json: {e}")
                total_size = None
        elif os.path.exists(len_filename):
            try:
                total_size = ast.literal_eval(open(len_filename, 'r').read())
            except Exception as e:
                logging.warning(f"Could not read or parse __len__ file: {e}")
                total_size = None
        else:
            total_size = None
    else: # For remote shards, we can't easily get size from files
        logging.warning("Cannot determine dataset size from remote shards without manifest. Size will be None.")
        total_size = None
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args: argparse.Namespace, image_processor: Callable, split: str):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset # Keep local import
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=image_processor)
    else:
        data_path = args.imagenet_train if is_train else args.imagenet_val
        assert data_path, f"ImageNet {split} path not provided."
        dataset = datasets.ImageFolder(data_path, transform=image_processor)

    if is_train:
        if not dataset.targets: 
            logging.warning(f"ImageNet {split} dataset is empty or has no targets.")
            sampler = None
        else:
            idxs = np.zeros(len(dataset.targets))
            target_array = np.array(dataset.targets)
            k = 50
            for c in range(1000): 
                m = target_array == c
                n = len(idxs[m])
                if n > 0:
                    arr = np.zeros(n)
                    arr[:min(k, n)] = 1
                    np.random.shuffle(arr)
                    idxs[m] = arr
            idxs = idxs.astype('int')
            valid_indices = np.where(idxs)[0]
            if len(valid_indices) == 0: 
                logging.warning(f"ImageNet {split} subsampling resulted in 0 samples. Using all samples.")
                sampler = None 
            else:
                sampler = SubsetRandomSampler(valid_indices)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True
    )
    return DataInfo(dataloader=dataloader, sampler=sampler)


def filter_no_caption_or_no_image_or_degradation(sample): 
    has_caption = ('txt' in sample or 'caption' in sample or 'json' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    has_degradation = False
    if 'json' in sample:
        try:
            metadata = json.loads(sample['json'].decode('utf-8')) 
            if isinstance(metadata, dict) and metadata.get("degradation"): 
                has_degradation = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass 
    elif 'degradation.txt' in sample:
        has_degradation = True
    return has_caption and has_image and has_degradation


def log_and_continue(exn):
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None: continue
        if lcase: suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample): yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample): yield current_sample

def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

def pytorch_worker_seed(increment=0):
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed
        if increment:
            seed += increment * max(1, worker_info.num_workers)
        return seed
    return wds.utils.pytorch_worker_seed()

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

class detshuffle2(wds.PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch
    def run(self, src):
        if isinstance(self.epoch, SharedEpoch): epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0: seed = pytorch_worker_seed(epoch)
        else: seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)

class ResampledShards2(IterableDataset):
    def __init__(self, urls, weights=None, nshards=sys.maxsize, worker_seed=None, deterministic=False, epoch=-1):
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        if not self.urls:
            logging.warning("ResampledShards2 initialized with empty URL list.")
        elif not isinstance(self.urls[0], str):
             raise ValueError("URLs must be strings.")
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch
    def __iter__(self):
        if not self.urls: return
        if isinstance(self.epoch, SharedEpoch): epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            if self.worker_seed is None: seed = pytorch_worker_seed(epoch)
            else: seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None: yield dict(url=self.rng.choice(self.urls))
            else: yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])

def _create_webdataset_pipeline(
    args: argparse.Namespace, is_train: bool, epoch: int, shared_epoch: SharedEpoch,
    processor: Optional[Any] = None,
    image_transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None
):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None, "Input shards (train_data or val_data) must be specified for WebDataset."
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    if resampled:
        pipeline = [ResampledShards2(input_shards, weights=args.train_data_upsampling_factors, deterministic=True, epoch=shared_epoch)]
    else:
        if args.train_data_upsampling_factors is not None:
             logging.warning("--train_data_upsampling_factors is ignored when --dataset-resampled is not set.")
        pipeline = [wds.SimpleShardList(input_shards)]

    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL, seed=args.seed, epoch=shared_epoch),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([tarfile_to_samples_nothrow, wds.shuffle(bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL)])
    else:
        pipeline.extend([wds.split_by_worker, wds.tarfile_to_samples(handler=log_and_continue)])

    def _tokenize_text_wds(text_data: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: # Renamed for clarity
        tokens, attention_mask = None, None
        if processor and hasattr(processor, 'tokenizer'):
            inputs = processor(text=[text_data], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
            tokens = inputs['input_ids'].squeeze(0)
            attention_mask = inputs.get('attention_mask', torch.ones_like(tokens)).squeeze(0) # Get mask or create one
        elif tokenizer:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                inputs = tokenizer(text_data, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
                tokens = inputs['input_ids'].squeeze(0)
                attention_mask = inputs.get('attention_mask', torch.ones_like(tokens)).squeeze(0)
            else: # open_clip.tokenize
                tokens = tokenizer(text_data).squeeze(0)
                # No attention mask from simple tokenizer
        if tokens is None:
            raise RuntimeError("No valid tokenizer for WebDataset text processing.")
        return tokens, attention_mask


    def process_sample_wds(sample: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        try:
            image_data = sample.get('png', sample.get('jpg', sample.get('jpeg', sample.get('webp'))))
            caption_data = sample.get('txt', sample.get('caption'))
            degradation_str = None
            if 'json' in sample:
                try:
                    metadata = json.loads(sample['json'].decode('utf-8'))
                    degradation_str = metadata.get(args.csv_degradation_key) # Use same key as CSV for consistency
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logging.warning(f"Error decoding JSON metadata in WDS sample {sample.get('__key__')}: {e}")
            elif 'degradation.txt' in sample: # Fallback for .degradation.txt
                degradation_str = sample['degradation.txt'].decode('utf-8').strip().lower()

            if image_data is None or caption_data is None or degradation_str is None:
                return None

            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            caption_str = caption_data.decode('utf-8').strip()
            degradation_label_str = degradation_str.strip().lower()

            if processor:
                image_inputs = processor(images=image, return_tensors="pt")
                processed_image = image_inputs['pixel_values'].squeeze(0)
            elif image_transform:
                processed_image = image_transform(image)
            else:
                raise RuntimeError("No image processor/transform for WebDataset.")

            caption_tokens, caption_attention_mask = _tokenize_text_wds(caption_str)

            y_true = torch.zeros(args.num_degradation_types)
            if degradation_label_str in DEGRADATION_TO_ID:
                degradation_id = DEGRADATION_TO_ID[degradation_label_str]
                y_true[degradation_id] = 1.0
            else:
                logging.warning(f"Unknown degradation label '{degradation_label_str}' in WDS sample {sample.get('__key__')}. Target all zeros.")

            degradation_text_tokens, degradation_text_attention_mask = None, None
            if hasattr(args, 'degrad_contrastive_weight') and args.degrad_contrastive_weight > 0:
                 if degradation_label_str in DEGRADATION_TO_ID:
                     degradation_id = DEGRADATION_TO_ID[degradation_label_str]
                     true_degradation_text = DEGRADATION_TEXT_DESCRIPTIONS[degradation_id]
                     degradation_text_tokens, degradation_text_attention_mask = _tokenize_text_wds(true_degradation_text)

            text_dict = {
                "caption_tokens": caption_tokens,
                "caption_attention_mask": caption_attention_mask,
                "degradation_target": y_true,
                "true_degradation_text_tokens": degradation_text_tokens,
                "true_degradation_text_attention_mask": degradation_text_attention_mask
            }
            return processed_image, text_dict
        except Exception as e:
            logging.warning(f"Error processing webdataset sample {sample.get('__key__', 'UnknownKey')}: {e}", exc_info=True)
            return None

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image_or_degradation), # Use updated filter
        wds.map(process_sample_wds),
        wds.select(lambda x: x is not None),
        wds.batched(args.batch_size, partial=not is_train)
    ])
    return wds.DataPipeline(*pipeline)


def get_wds_dataset(
    args: argparse.Namespace, is_train: bool, epoch: int = 0,
    processor: Optional[Any] = None,
    image_transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None
):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError('Number of train samples must be specified for webdataset if not in sizes.json or __len__')
        else:
            num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)
    dataset = _create_webdataset_pipeline(args, is_train, epoch, shared_epoch, processor, image_transform, tokenizer)

    if is_train:
        if not (getattr(args, 'dataset_resampled', False)):
            num_shards_actual = num_shards or (len(expand_urls(input_shards)[0]) if input_shards else 0)
            if num_shards_actual == 0: raise ValueError("No shards found for training WebDataset.")
            world_size = args.world_size if args.world_size > 0 else 1
            assert num_shards_actual >= args.workers * world_size, \
                f'Number of shards ({num_shards_actual}) must be >= total workers ({args.workers * world_size}).'
        round_fn = math.floor if getattr(args, 'commonmark_floor', False) else math.ceil
        global_batch_size = args.batch_size * args.world_size
        if num_samples == 0 and global_batch_size > 0 : num_batches = 0
        elif global_batch_size == 0: raise ValueError("Global batch size is zero.")
        else: num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        if num_workers == 0: num_worker_batches = num_batches
        else: num_worker_batches = round_fn(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches if num_worker_batches > 0 else 1)
    else:
        if args.batch_size == 0: raise ValueError("Batch size for validation cannot be zero.")
        num_batches = math.ceil(num_samples / args.batch_size) if num_samples > 0 else 0

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        images = torch.stack([item[0] for item in batch])
        text_dicts = [item[1] for item in batch]
        collated_text_dict = {}
        keys_to_collate = [
            "caption_tokens", "caption_attention_mask",
            "degradation_target",
            "true_degradation_text_tokens", "true_degradation_text_attention_mask"
        ]
        for key in keys_to_collate:
             if text_dicts and key in text_dicts[0] and text_dicts[0][key] is not None:
                 try:
                     collated_text_dict[key] = torch.stack([d[key] for d in text_dicts])
                 except RuntimeError as e:
                     logging.error(f"Error stacking key '{key}': {e}. Items: {[type(d.get(key)) for d in text_dicts]}")
                     collated_text_dict[key] = None
             else:
                 collated_text_dict[key] = None
        return images, collated_text_dict

    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, collate_fn=collate_fn
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(
    args: argparse.Namespace, is_train: bool, epoch: int = 0,
    processor: Optional[Any] = None,
    image_transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None
):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename, f"CSV input_filename not provided for {'train' if is_train else 'val'}."
    dataset = CsvDataset(
        input_filename,
        processor=processor,
        image_transform=image_transform,
        tokenizer=tokenizer,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key, # This will point to 'title'
        degradation_key=args.csv_degradation_key, # Effectively ignored by CsvDataset if parsing 'title'
        sep=args.csv_separator,
        # caption_degradation_separator is now an init arg for CsvDataset
        caption_degradation_separator=getattr(args, 'csv_caption_degradation_separator', ':'), # Get from args or default
        crop=args.crop,
        args=args
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=is_train) if args.distributed and is_train else None
    shuffle_flag = is_train and sampler is None

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        images = torch.stack([item[0] for item in batch])
        text_dicts = [item[1] for item in batch]
        collated_text_dict = {}
        keys_to_collate = [
            "caption_tokens", "caption_attention_mask",
            "degradation_target",
            "true_degradation_text_tokens", "true_degradation_text_attention_mask"
        ]
        for key in keys_to_collate:
             if text_dicts and key in text_dicts[0] and text_dicts[0][key] is not None:
                 try:
                     collated_text_dict[key] = torch.stack([d[key] for d in text_dicts])
                 except RuntimeError as e:
                     logging.error(f"Error stacking key '{key}': {e}. Items: {[type(d.get(key)) for d in text_dicts]}")
                     collated_text_dict[key] = None
             else:
                 collated_text_dict[key] = None
        return images, collated_text_dict

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last=is_train, collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_synthetic_dataset(
    args: argparse.Namespace, is_train: bool, epoch: int = 0,
    processor: Optional[Any] = None,
    image_transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None
):
    image_size_default = (224, 224)
    image_size = image_size_default
    active_preprocessor = processor if processor else image_transform
    if hasattr(active_preprocessor, 'image_processor') and hasattr(active_preprocessor.image_processor, 'size'):
         size_info = active_preprocessor.image_processor.size
         if isinstance(size_info, dict):
              h = size_info.get('height', size_info.get('shortest_edge', image_size_default[0]))
              w = size_info.get('width', size_info.get('shortest_edge', image_size_default[1]))
              image_size = (h, w)
         elif isinstance(size_info, int): image_size = (size_info, size_info)
    elif hasattr(active_preprocessor, 'size'):
         if isinstance(active_preprocessor.size, int): image_size = (active_preprocessor.size, active_preprocessor.size)
         elif isinstance(active_preprocessor.size, (list, tuple)) and len(active_preprocessor.size) >=2: image_size = active_preprocessor.size[-2:]

    dataset_size = args.train_num_samples if is_train and args.train_num_samples else (args.val_num_samples or 1000)
    dataset = SyntheticDataset( # SyntheticDataset needs to be adapted to use _tokenize_text like CsvDataset
        processor=processor,
        image_transform=image_transform, # For image part if processor not used
        tokenizer=tokenizer, # For text part if processor not used
        image_size=image_size,
        dataset_size=dataset_size,
        num_degradation_types=args.num_degradation_types,
        degradation_text_descriptions=DEGRADATION_TEXT_DESCRIPTIONS,
        add_degradation_text=(hasattr(args, 'degrad_contrastive_weight') and args.degrad_contrastive_weight > 0)
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=is_train) if args.distributed and is_train else None
    shuffle_flag = is_train and sampler is None

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        images = torch.stack([item[0] for item in batch])
        text_dicts = [item[1] for item in batch]
        collated_text_dict = {}
        keys_to_collate = [
            "caption_tokens", "caption_attention_mask",
            "degradation_target",
            "true_degradation_text_tokens", "true_degradation_text_attention_mask"
        ]
        for key in keys_to_collate:
             if text_dicts and key in text_dicts[0] and text_dicts[0][key] is not None:
                 try:
                     collated_text_dict[key] = torch.stack([d[key] for d in text_dicts])
                 except RuntimeError as e:
                     logging.error(f"Error stacking key '{key}': {e}. Items: {[type(d.get(key)) for d in text_dicts]}")
                     collated_text_dict[key] = None
             else:
                 collated_text_dict[key] = None
        return images, collated_text_dict

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last=is_train, collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset": return get_wds_dataset
    elif dataset_type == "csv": return get_csv_dataset
    elif dataset_type == "synthetic": return get_synthetic_dataset
    elif dataset_type == "auto":
        if data_path is None:
             logging.info("data_path is None for auto dataset type, assuming synthetic.")
             return get_synthetic_dataset
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']: return get_csv_dataset
        elif ext in ['tar']: return get_wds_dataset
        else: raise ValueError(f"Tried to figure out dataset type for {data_path}, but failed for extension {ext}.")
    else: raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(
    args: argparse.Namespace,
    processor_or_transforms: Union[Any, Tuple[Optional[Callable], Optional[Callable]]],
    epoch: int = 0,
    tokenizer: Optional[Any] = None # This is now an explicit argument
):
    data = {}
    is_hf_processor = hasattr(processor_or_transforms, 'tokenizer') and \
                      hasattr(processor_or_transforms, 'image_processor')

    current_processor = None
    current_image_transform_train = None
    current_image_transform_val = None
    current_tokenizer_for_datasets = None

    if is_hf_processor:
        current_processor = processor_or_transforms
        logging.info("Using Hugging Face Processor for data loading.")
    else:
        if isinstance(processor_or_transforms, tuple) and len(processor_or_transforms) == 2:
            current_image_transform_train, current_image_transform_val = processor_or_transforms
        else:
            current_image_transform_train = processor_or_transforms
            current_image_transform_val = processor_or_transforms
        current_tokenizer_for_datasets = tokenizer
        logging.info("Using separate image transforms and tokenizer for data loading.")

    if args.train_data or args.dataset_type == "synthetic":
        dataset_fn_train = get_dataset_fn(args.train_data, args.dataset_type)
        data["train"] = dataset_fn_train(
            args, is_train=True, epoch=epoch,
            processor=current_processor,
            image_transform=current_image_transform_train,
            tokenizer=current_tokenizer_for_datasets
        )

    if args.val_data:
        dataset_fn_val = get_dataset_fn(args.val_data, args.dataset_type)
        data["val"] = dataset_fn_val(
            args, is_train=False, epoch=epoch,
            processor=current_processor,
            image_transform=current_image_transform_val,
            tokenizer=current_tokenizer_for_datasets
        )

    if args.imagenet_val is not None or args.imagenet_v2 is not None:
        img_processor_for_imagenet = None
        if is_hf_processor:
            img_processor_for_imagenet = current_processor.image_processor
        elif current_image_transform_val is not None:
            img_processor_for_imagenet = current_image_transform_val
        elif current_image_transform_train is not None:
            img_processor_for_imagenet = current_image_transform_train
        else:
            raise ValueError("Cannot determine image processor/transform for ImageNet evaluation.")

        if args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, img_processor_for_imagenet, "val")
        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, img_processor_for_imagenet, "v2")
    return data

