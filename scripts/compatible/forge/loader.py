import os
import gc
import torch
import huggingface_guess
from diffusers import DiffusionPipeline
from modules import shared, script_callbacks
from modules.shared import opts, cmd_opts
from modules.timer import Timer
from safetensors import deserialize, serialize

from scripts.compatible.forge import sd_models
from backend.loader import *

def split_state_dict(sd, additional_state_dicts: list = None):
    sd = preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            asd = load_torch_file(asd)
            sd = replace_state_dict(sd, asd, guess)

    guess.clip_target = guess.clip_target(sd)

    state_dict = {
        guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix),
        guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)
    }

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + '.'])

    state_dict['ignore'] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f'StateDict Keys: {print_dict}')

    del state_dict['ignore']

    return state_dict, guess

@torch.inference_mode()
def forge_loader(sd, additional_state_dicts=None):
    try:
        state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts)
    except:
        raise ValueError('Failed to recognize model type!')
    
    repo_name = estimated_config.huggingface_repo

    dir_path = os.path.dirname(backend.loader.__file__)
    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)
    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.get(component_name, None)
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del state_dicts[component_name]
            if component is not None:
                huggingface_components[component_name] = component

    for M in possible_models:
        if any(isinstance(estimated_config, x) for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print('Failed to recognize model type!')
    return None
