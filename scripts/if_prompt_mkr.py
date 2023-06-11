import modules.scripts as scripts
import gradio as gr
import os
import requests
import json
import re

from modules import script_callbacks, shared
from modules.processing import Processed, process_images
from modules.shared import state
from scripts.negfiles import get_negative
from scripts.modeltags import get_loras, get_embeddings

script_dir = scripts.basedir()
#script_name = os.path.splitext(os.path.basename(__file__))[0] # I might use this later


def on_ui_settings():
    section=("if_prompt_mkr", "iF_prompt_MKR")
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


    def ui(self, is_img2img):
        neg_prompts = get_negative(os.path.join(script_dir, "negfiles"))

        character_list = get_character_list()

        params = {
            'selected_character': character_list if character_list else ['if_ai_SD', 'iF_Ai_SD_b', 'iF_Ai_SD_NSFW'],
            'prompt_prefix': 'Style-SylvaMagic, ',
            'input_prompt': '(Dark elf empress:1.2), enchanted Forrest',
            'negative_prompt': '(Worst quality, Low quality:1.4), NSFW, ugly, ng_deepnegative_v1_75t, negative_hand-neg,',
            'prompt_subfix': '(rim lighting,:1.1) two tone lighting, <lora:epiNoiseoffset_v2:0.8>'

        }

        
        def on_neg_prompts_change(x):
            
            filename = neg_prompts[x][1]

            with open(filename, 'r') as file:
                new_neg_prompt = file.read()

            params.update({'negative_prompt': str(new_neg_prompt)})
            negative_prompt.value = str(new_neg_prompt)

            return new_neg_prompt
        
        with gr.Row():
            input_prompt = gr.inputs.Textbox(lines=1, placeholder=params['input_prompt'], label="Input Prompt")
            selected_character = gr.inputs.Dropdown(label="characters", choices=params['selected_character'])
        
        #The Idea is chosing and embeding and lora and will populate the keywords automatically 
        with gr.Accordion('Prefix & TIembeddings', open=True):
            ti_choices = ["None"]
            ti_choices.extend(get_embeddings())
            with gr.Row():
                prompt_prefix = gr.inputs.Textbox(lines=1, placeholder=params['prompt_prefix'], label="Prompt Prefix")
                embeddings_model = gr.inputs.Dropdown(label="Embeddings Model", choices=ti_choices)

        with gr.Accordion('Subfix & Loras', open=True):
            lora_choices = ["None"]
            lora_choices.extend(get_loras())
            with gr.Row(equal_height=True):
                prompt_subfix = gr.inputs.Textbox(lines=1, placeholder=params['prompt_subfix'], label="Subfix for adding Loras (optional)")
                lora_model = gr.inputs.Dropdown(label="Lora Model", choices=lora_choices)

        with gr.Row():  
            neg_prompts_dropdown = gr.Dropdown(
                label="neg_prompts", 
                choices=[n[0] for n in neg_prompts],
                type="index", 
                elem_id="iF_prompt_MKR_neg_prompts_dropdown")
            negative_prompt = gr.Textbox(lines=4, default=params['negative_prompt'], label="Negative Prompt") #, default_value=neg_prompts[0][1]
        with gr.Row():
            excluded_words = gr.inputs.Textbox(lines=1, placeholder="Enter case-sensitive words to exclude, separated by commas", label="Excluded Words")
        
            #prompt_count = gr.Number(value=1, label="How many prompts you need")
            #I plan to make a batch count option later for apending prompts and save them to a file, 
            #the idea is to make variations of the same prompt, will be useful for livestreams
        
        selected_character.change(lambda x: params.update({'selected_character': x}), selected_character, None)
        prompt_prefix.change(lambda x: params.update({'prompt_prefix': x}), prompt_prefix, None)
        input_prompt.change(lambda x: params.update({'input_prompt': x}), input_prompt, None)
        prompt_subfix.change(lambda x: params.update({'prompt_subfix': x}), prompt_subfix, None)
        
        excluded_words.change(lambda x: params.update({'excluded_words': [word.strip() for word in x.split(',')] if x else []}), excluded_words, None)
        #prompt_count.change(lambda x: params.update({'prompt_count': x}), prompt_count, None)
        
        neg_prompts_dropdown.change(on_neg_prompts_change, neg_prompts_dropdown, negative_prompt)
        negative_prompt.change(lambda x: params.update({'negative_prompt': x}), negative_prompt, None)
        #lora_model.change(lambda x: params.update({'lora_model': x}), lora_model, None)

        return [selected_character, prompt_prefix, input_prompt, prompt_subfix, excluded_words, negative_prompt]

    def run(self, p, selected_character ,prompt_prefix, input_prompt, prompt_subfix, excluded_words, negative_prompt, *args, **kwargs):
        generated_text = self.generate_text(selected_character, input_prompt, excluded_words)
        combined_prompt = prompt_prefix + ' ' + generated_text + ' ' + prompt_subfix
        p.prompt = combined_prompt
        p.negative_prompt = negative_prompt
        p.prompt_subfix = prompt_subfix
        p.selected_character = selected_character
        p.input_prompt = input_prompt
        p.generate_text = generated_text
        return process_images(p)
 

  
    def generate_text(self, character, prompt, words):

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
            'history': {'internal': [], 'visible': []}, #this is the history of the chat, it's not used here but it's needed for the chat to work
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
            print(response.content)
            results = json.loads(response.content)['results']
            if results:
                history = results[0]['history']
                if history:
                    visible = history['visible']
                    if visible:
                        generated_text = visible[-1][1]

                        # Remove words from excluded_words from generated_text
                        if words:
                            generated_text = ' '.join(word for word in generated_text.split() if word not in words)

                        # Remove audio tags from generated_text
                        if '<audio' in generated_text:
                            
                            print("Audio has been generated.")

                            generated_text = re.sub(r'<audio.*?>.*?</audio>', '', generated_text)

                        return generated_text
            else:
                print("No results found.")
        else:
            print("Error: Request failed with status code, probably Ooga isn't running with API flags check the readme", response.status_code)

        
    def process_images(self, p):
        state.job_count = 0
        state.job_count += p.n_iter

        proc = process_images(p)

        return Processed(p, [proc.images[0]], p.seed, "", all_prompts=proc.all_prompts, infotexts=proc.infotexts)
