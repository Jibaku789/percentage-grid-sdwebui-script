"""
    @author: Jibaku789
    @version: 1.1
    @date: November 2023
"""

import modules.scripts as scripts
import gradio as gr
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import random

from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):

    def title(self):
        return "Percentage Grid"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):

        model_name = gr.Textbox(label="Embedding/Lora/Hypernetwork Name", value="<hypernet:Example>", lines=1, elem_id=self.elem_id("model_name"))
        percentages = gr.Textbox(label="Percentage List", value="1.0, 0.8, 0.6, 0.4, 0.2, 0.0", lines=1, elem_id=self.elem_id("percentages"))
        seed_str = gr.Dropdown(label="Seed Strategy", choices=["Fixed", "Random"], value="Fixed", elem_id=self.elem_id("seed_strategy"))

        return [model_name, percentages, seed_str]


    def run(self, p, model_name, percentages, seed_str):

        images = []
        all_prompts = []
        infotexts = []
        seed = p.seed

        if seed == -1:
            if seed_str == "Fixed":
                seed = random.randint(0, 1000000)
                print(f"Forcing seed to: {seed}")

        for pct in list(map(lambda x: x.strip(), percentages.split(","))):

            model_with_mark = False
            model_name_adjusted = model_name.strip()
            
            if model_name.startswith("<"):
                model_with_mark = True
                model_name_adjusted = model_name_adjusted[1:]

            if model_name.endswith(">"):
                model_name_adjusted = model_name_adjusted[:-1]

            model_name_adjusted = f"{model_name_adjusted}:{pct}"

            if model_with_mark:
                model_name_adjusted = f"<{model_name_adjusted}>"

            prompt = f"{model_name_adjusted}, {p.prompt}"

            args = {
                "prompt": prompt,
                "seed": seed,
            }

            print(f"Processing: {args}")

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images
            
            all_prompts += proc.all_prompts
            infotexts.append(f"{float(pct)*100}%")

        # Create Grid
        f, axarr = plt.subplots(1, len(images))
        f.suptitle(f"Percentage Grid: {model_name}", fontsize=15)
        f.set_dpi(100)
        f.set_figwidth( (images[0].size[0] * len(images))  / 100)
        f.set_figheight( (images[0].size[1] * 1.5) / 100 )

        c = 0
        for img in images:
            axarr[c].imshow(img)
            axarr[c].set_title(infotexts[c])
            axarr[c].set_xticks([])
            axarr[c].set_yticks([])

            c += 1

        save_dir = os.path.abspath(f"{p.outpath_grids}/../percentage_grid")
        os.makedirs(save_dir, exist_ok=True)

        now = datetime.now()
        save_path = f"{save_dir}/{now.strftime('%Y %m %d %H %M %S')}.jpg"

        plt.subplots_adjust(left=0.,
                            bottom=0.20,
                            right=1.0,
                            top=0.9,
                            wspace=0.0,
                            hspace=0.0)

        plt.savefig(save_path, dpi=100)
        plt.plot()
        print(f"Saved grid into: {os.path.abspath(save_path)}")

        return Processed(p, images, seed, "", all_prompts=all_prompts, infotexts=infotexts)
    
# End of file