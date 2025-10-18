from flask import Flask, request, jsonify, send_from_directory, render_template_string

app = Flask(__name__)


import os, sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#%pip install "torch>=2.1.0" einops datasets jaxtyping "sae_lens>=3.23.1" openai tabulate "nbformat>=4.2.0" umap-learn hdbscan eindex-callum git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python git+https://github.com/callummcdougall/sae_vis.git@callum/v3 "transformer-lens>=2.7.0"

from IPython.display import HTML, IFrame, clear_output, display
import random
from rich import print as rprint
from rich.table import Table
from tqdm.auto import tqdm
from functools import partial
import torch as t
import torch.nn.functional as F
from torch import Tensor, nn
from jaxtyping import Float, Int
import einops

import matplotlib.pyplot as plt
import numpy as np

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from tabulate import tabulate

#from google.colab import drive
import pickle, glob

def display_dashboard(
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    latent_idx=0,
    width=1200,
    height=800,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))

def steering_hook(
    activations,
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx] 

def generate_with_steering( 
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 50,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient=steering_coefficient,
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output

#utils.test_prompt(prompt, answer, gemma_2_2b,prepend_bos=True)



 

import inspect

def get_func_args(func):
    sig = inspect.signature(func)
    return {
        "args": [
            param.name
            for param in sig.parameters.values()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ],
        "defaults": {
            param.name: param.default
            for param in sig.parameters.values()
            if param.default is not param.empty
        },
    }




from huggingface_hub import login
TOKEN = ""
login(TOKEN)
prompt_emoji = "A random emoji:\n🎉\n😎\n🍕\n🚀\n🌈\n🚀\n😒\n"
prompt_word = "A random English word:\nsun\nriver\ncloud\nstone\nforest\ncold\nmirror\n"


gemma_2_2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)
layer = 24

release = "gemma-2-2b-res-matryoshka-dc"
sae_id = f"blocks.{layer}.hook_resid_post"

gemma_2_2b_sae = SAE.from_pretrained(release, sae_id, device=str(device))[0]

_, cache_emoji = gemma_2_2b.run_with_cache_with_saes(prompt_emoji, saes=[gemma_2_2b_sae])

_, cache_word = gemma_2_2b.run_with_cache_with_saes(prompt_word, saes=[gemma_2_2b_sae])


for name, param in cache_emoji.items():
    if "hook_sae" in name:
        print(f"{name:<43}: {tuple(param.shape)}")
        



saespace_acts_post_word = cache_word[f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"][0, -1, :].detach().cpu()
saespace_acts_post_emoji = cache_emoji[f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"][0, -1, :].detach().cpu()

normed_sparsedspace_post = saespace_acts_post_emoji - saespace_acts_post_word


vals, indices = torch.sort(normed_sparsedspace_post,descending=True) # llooking at high activs
indices_high = torch.where(vals> 0)

inspect_list = [] 
for i in zip(vals[indices_high],indices[indices_high]):
    inspect_list.append(i)

indices_inspect = indices[indices_high]
inspect_list 


import torch 
#latent_idx = int(indices_inspect[0].item())

#latent_idx = [indices_inspect[0],indices_inspect[5]] #layer 21

#latent_idx = [indices_inspect[0].item(), indices_inspect[2].item()] #layer 22: emojois + slang

#latent_idx = indices_inspect[0:3] #layer 23

#latent_idx = indices_inspect[0:3]#.item() #layer 24      # 10 is "face"/ emotion indices_inspect[10].item() , 2 is slang/emojis, 0 more emojis

latent_idx = [indices_inspect[0].item(),indices_inspect[2].item()] # layer 24

def steering_hook_sparsespace(
    activations,
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    # w_dec gets for a certain sparse space idx the activations pattern, and then we add it
    if isinstance(latent_idx,int):
        return activations + steering_coefficient *sae.W_dec[latent_idx] #Hmm wow how can we visualize this - interesting. Pretty simple in a way.
    else: #multi indx
        #print("enhancing multiple concepts")
        return activations + steering_coefficient *torch.mean(sae.W_dec[latent_idx],dim=0,keepdim=False) #Hmm wow how can we visualize this - interesting. Pretty simple in a way.




_steering_hook = partial(
    steering_hook_sparsespace,
    sae=gemma_2_2b_sae,
    latent_idx=latent_idx,
    steering_coefficient=600,
)






# Serve index.html directly
@app.route("/")
def index():
    with open("index.html") as f:
        return render_template_string(f.read())

# Serve logo image
@app.route("/logo.png")
def logo():
    return send_from_directory(".", "logo.png")

# Handle prompt generation
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    context = prompt

    sample_tokens = True
    preprend_bos_token = True
    samples =  3

    print(gemma_2_2b.generate(context,prepend_bos=preprend_bos_token,do_sample=sample_tokens,temperature=1.5, freq_penalty=2.0, verbose=False))


    # with tweak
    
    
    with_tweak = []
    without_tweak = []
    
    for i in range(samples):
        with gemma_2_2b.hooks(fwd_hooks=[(gemma_2_2b_sae.cfg.hook_name, _steering_hook)]):
            generated = gemma_2_2b.generate(context,prepend_bos=preprend_bos_token,do_sample=sample_tokens,temperature=1.5, freq_penalty=2.0, verbose=False)
            with_tweak.append(generated)

    for i in range(samples):
        generated = gemma_2_2b.generate(context,prepend_bos=preprend_bos_token,do_sample=sample_tokens,temperature=1.5, freq_penalty=2.0, verbose=False)
        without_tweak.append(generated)

    return jsonify({
        "with_tweak": with_tweak,
        "without_tweak": without_tweak
    })

if __name__ == "__main__":
    app.run(debug=True)
