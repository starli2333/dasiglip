# -*- coding: utf-8 -*-

# --- Standard OpenCLIP Constants (Keep if they were there) ---
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# --- DA-SigLIP Degradation Constants ---
# Define your list of degradation types in a fixed order
DEGRADATION_TYPES = [
    'motion-blurry',
    'hazy',
    'jpeg-compressed',
    'low-light',
    'noisy',
    'raindrop',
    'rainy',
    'shadowed',
    'snowy',
    'uncompleted' # Example for inpainting or other missing content
    # Add any other specific degradation types you have
]

NUM_DEGRADATION_TYPES = len(DEGRADATION_TYPES)

DEGRADATION_TO_ID = {name: i for i, name in enumerate(DEGRADATION_TYPES)}
ID_TO_DEGRADATION = {i: name for i, name in enumerate(DEGRADATION_TYPES)}

# Text descriptions for each degradation type (used for optional contrastive loss and for UIR part)
# You can make these more descriptive if needed
DEGRADATION_TEXT_DESCRIPTIONS = [f"a {name.replace('-', ' ')} image" for name in DEGRADATION_TYPES]

# Example:
# DEGRADATION_TEXT_DESCRIPTIONS = [
#     "an image with motion blur",
#     "a hazy image",
#     "a jpeg compressed image",
#     "a low-light image",
#     "a noisy image",
#     "an image with raindrops",
#     "a rainy image",
#     "an image with shadows",
#     "a snowy image",
#     "an incomplete image"
# ]

# --- Other constants if any ---

