import modules.scripts as scripts
import gradio as gr
import os
import requests
import json

from modules import script_callbacks, shared
from modules.processing import Processed, process_images
from modules.shared import state

script_dir = scripts.basedir()
#script_name = os.path.splitext(os.path.basename(__file__))[0]


script_dir = scripts.basedir()

def on_ui_settings():
    section=("ifpromptmkr", "iFpromptMKR")
    shared.opts.add_option("character_path", shared.OptionInfo(
      "", "Select an iF or other SD prompt maker character you want to use inside the Oobabooga character Directory json only", section=section))
    shared.opts.add_option("stopping_string", shared.OptionInfo(
      "", 'Write a custom stopping string such as i.e for Alpaca use "### Assistant:" or your name i.e "ImpactFrames:"', section=section))
    shared.opts.add_option("xtokens", shared.OptionInfo(
      80, "Set the number of tokens to generate", gr.Number, section=section
      ))
    shared.opts.add_option("xtemperature", shared.OptionInfo(
      0.7, "Set the temperature of the generated text", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.05}, section=section
      ))
    shared.opts.add_option("xtop_k", shared.OptionInfo(
      30, "Set the top k of the generated text", gr.Number, section=section
      ))
    shared.opts.add_option("xtop_p", shared.OptionInfo(
      0.9, "Set the top p of the generated text", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.1}, section=section
      ))
    shared.opts.add_option("xtypical_p", shared.OptionInfo(
      1, "Set the typical p of the generated text", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.1}, section=section
      ))
    shared.opts.add_option("xrepetition_penalty", shared.OptionInfo(
      1.2, "Set the repetition penalty of the", gr.Slider, {"minimum": 0, "maximum": 2, "step": 0.1}, section=section
      ))
      

script_callbacks.on_ui_settings(on_ui_settings)

def get_character_list():
    # Get character path from options
    character_path = shared.opts.data.get("character_path", None)

    # Check if the character path is not None and exists
    if character_path and os.path.exists(character_path):
        # Get list of all json files
        return [os.path.splitext(f)[0] for f in os.listdir(character_path) if f.endswith('.json')]
    else:
        return []


class Script(scripts.Script):

    def title(self):
        return "iF_prompt_MKR"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):

        # Get list of character names from json files in the character directory
        character_list = get_character_list()

        params = {
            'selected_character': character_list if character_list else ['if_ai_SD', 'iF_Ai_SD_b', 'iF_Ai_SD_NSFW'],
            'prompt_prefix': 'Style-SylvaMagic, ',
            'input_prompt': '(Dark elf empress:1.2), enchanted Forrest',
            'negative_prompt': '(worst quality, low quality:1.3)',
            'prompt_subfix': '(rim lighting,:1.1) two tone lighting, <lora:epiNoiseoffset_v2:0.8>'

        }

        
        selected_character = gr.inputs.Dropdown(label="characters", choices=params['selected_character'])
        prompt_prefix = gr.inputs.Textbox(lines=1, placeholder=params['prompt_prefix'], label="Prompt Prefix")
        input_prompt = gr.inputs.Textbox(lines=1, placeholder=params['input_prompt'], label="Input Prompt")
        prompt_subfix = gr.inputs.Textbox(lines=1, placeholder=params['prompt_subfix'], label="Subfix for adding Loras (optional)")
        negative_prompt = gr.inputs.Textbox(lines=2, placeholder=params['negative_prompt'], label="Negative Prompt")
        

        selected_character.change(lambda x: params.update({'selected_character': x}), selected_character, None)
        prompt_prefix.change(lambda x: params.update({'prompt_prefix': x}), prompt_prefix, None)
        input_prompt.change(lambda x: params.update({'input_prompt': x}), input_prompt, None)
        prompt_subfix.change(lambda x: params.update({'prompt_subfix': x}), prompt_subfix, None)
        negative_prompt.change(lambda x: params.update({'negative_prompt': x}), negative_prompt, None)
     
        return [selected_character, prompt_prefix, input_prompt, negative_prompt, prompt_subfix]


    def run(self, p, selected_character ,prompt_prefix, input_prompt, negative_prompt, prompt_subfix, *args, **kwargs):
        generated_text = self.generate_text(selected_character, input_prompt)
        combined_prompt = prompt_prefix + ' ' + generated_text + ' ' + prompt_subfix
        p.prompt = combined_prompt
        p.negative_prompt = negative_prompt
        p.prompt_subfix = prompt_subfix
        p.selected_character = selected_character
        p.input_prompt = input_prompt
        p.generate_text = generated_text
        return process_images(p)
 

  
    def generate_text(self, character, prompt):

        print("Generating text...")
        stopping = shared.opts.data.get("stopping_string", None)
        if not stopping:
            stopping = "### Assistant:"
        xtokens = shared.opts.data.get("xtokens", 80)
        xtemperature = shared.opts.data.get("xtemperature", 0.7)
        xtop_k = shared.opts.data.get("xtop_k", 30)
        xtop_p = shared.opts.data.get("xtop_p", 0.9)
        xtypical_p = shared.opts.data.get("xtypical_p", 0.9)
        xrepetition_penalty = shared.opts.data.get("xrepetition_penalty", 1.2)
        

        data = {
            'user_input': prompt,
            'history': {'internal': [], 'visible': []},  # Add this line
            'mode': "chat",
            'character': character,
            'instruction_template': 'Wizard-Mega',
            'regenerate': False,
            '_continue': False,
            'stop_at_newline': False,
            'chat_prompt_size': 2048,
            'chat_generation_attempts': 1,
            'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
            'max_new_tokens': xtokens,
            'temperature': xtemperature,
            'top_k': xtop_k,
            'top_p': xtop_p,
            'do_sample': True,
            'typical_p': xtypical_p,
            'repetition_penalty': xrepetition_penalty,
            'encoder_repetition_penalty': 1.0,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': -1,
            'add_bos_token': True,
            'custom_stopping_strings': [stopping,],
            'truncation_length': 2048,
            'ban_eos_token': False,
        }  
        headers = {     
            "Content-Type": "application/json" 
        } 

        response = requests.post("http://127.0.0.1:5000/api/v1/chat",
                            data=json.dumps(data), headers=headers)


        if response.status_code == 200:
            print (response.content)
            results = json.loads(response.content)['results']
            if results:
                history = results[0]['history']
                if history:
                    visible = history['visible']
                    if visible:
                        generated_text = visible[-1][1]
            return generated_text
        
    def process_images(self, p):
        state.job_count = 0
        state.job_count += p.n_iter

        proc = process_images(p)

        return Processed(p, [proc.images[0]], p.seed, "", all_prompts=proc.all_prompts, infotexts=proc.infotexts)
        



