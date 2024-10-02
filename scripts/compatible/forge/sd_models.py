import os
import safetensors.torch
import torch
from scripts.compatible.forge import loader
from modules import devices, shared
from modules.sd_models import *

checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_replacements_sd2_turbo = { # Converts SD 2.1 Turbo from SGM to LDM format.
    'conditioner.embedders.0.': 'cond_stage_model.',
}

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        if not shared.opts.disable_mmap_load_safetensors:
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            pl_sd = safetensors.torch.load(open(checkpoint_file, 'rb').read())
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    is_sd2_turbo = 'conditioner.embedders.0.model.ln_final.weight' in pl_sd and pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024

    sd = {}
    for k, v in pl_sd.items():
        if is_sd2_turbo:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd2_turbo)
        else:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

@torch.inference_mode()
def forge_model_reload(state_dict):
    current_hash = str(model_data.forge_loading_parameters)

    print('Loading Model: ' + str(model_data.forge_loading_parameters))

    timer = Timer()

    if model_data.sd_model:
        model_data.sd_model = None
        memory_management.unload_all_models()
        memory_management.soft_empty_cache()
        gc.collect()

    timer.record("unload existing model")

    checkpoint_info = model_data.forge_loading_parameters['checkpoint_info']

    if checkpoint_info is None:
        raise ValueError('You do not have any model! Please download at least one model in [models/Stable-diffusion].')

    additional_state_dicts = model_data.forge_loading_parameters.get('additional_modules', [])

    timer.record("cache state dict")

    dynamic_args['forge_unet_storage_dtype'] = model_data.forge_loading_parameters.get('unet_storage_dtype', None)
    dynamic_args['embedding_dir'] = cmd_opts.embeddings_dir
    dynamic_args['emphasis_name'] = opts.emphasis

    sd_model = loader.forge_loader(state_dict, additional_state_dicts=additional_state_dicts)
    timer.record("forge model load")

    sd_model.extra_generation_params = {}
    sd_model.comments = []
    sd_model.sd_checkpoint_info = checkpoint_info
    sd_model.filename = checkpoint_info.filename
    sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    model_data.set_sd_model(sd_model)

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    model_data.forge_hash = current_hash

    return sd_model, True
