import modules.scripts as scripts
import gradio as gr
import os
import requests
import json
import re
import modules
import unicodedata
from modules import images, script_callbacks, shared
from modules.processing import Processed, process_images
from modules.shared import state
from scripts.negfiles import get_negative
from scripts.modeltags import get_loras, get_embeddings, get_lora_trigger_words, get_embedding_trigger_words, get_trigger_files


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
    shared.opts.add_option("custom_embedding_path", shared.OptionInfo(
      "", "Select optional embedding Path", section=section))
    shared.opts.add_option("custom_lora_path", shared.OptionInfo(
      "", "Select optional lora Path", section=section))
    

script_callbacks.on_ui_settings(on_ui_settings)




class Script(scripts.Script):

    def title(self):
        return "iF_prompt_MKR"
    
    
    def ui(self, is_img2img):
        # Get character path from Oobabooga character Directory json defined on_ui_settings
        def get_character_list():
            # Get character path from options
            character_path = shared.opts.data.get("character_path", None)
            # Check if the character path is not None and exists
            if character_path and os.path.exists(character_path):
                # Get list of all json files
                return [os.path.splitext(f)[0] for f in os.listdir(character_path) if f.endswith('.json')]
            else:
                return []
            
        # Get Negative prompts form folder negfiles   
        neg_prompts = get_negative(os.path.join(script_dir, "negfiles"))

        # Assings The characters to the character list
        character_list = get_character_list()

        # set initial params values
        params = {
            'selected_character': character_list if character_list else ['if_ai_SD', 'iF_Ai_SD_b', 'iF_Ai_SD_NSFW'],
            'prompt_prefix': '((style-swirlmagic):0.35),',
            'input_prompt': '(CatGirl warrior:1.2), legendary sword,',
            'negative_prompt': '(nsfw), (worst quality, low quality:1.4), ((text, signature, captions):1.3),',
            'prompt_subfix': 'dark theme <lora:LowRA:0.6>, add_detail <lora:add_detail:0.6>,',
            'batch_size': 1,
            'batch_count': 1,

        }
        # lora and embedding dropdowns can access current user input
        prompt_prefix_value = params['prompt_prefix']
        prompt_subfix_value = params['prompt_subfix']

        # Updates the neg_prompts dropdown
        def on_neg_prompts_change(x):
            
            filename = neg_prompts[x][1]

            with open(filename, 'r') as file:
                new_neg_prompt = file.read()

            params.update({'negative_prompt': str(new_neg_prompt)})
            negative_prompt.value = str(new_neg_prompt)

            return new_neg_prompt
        
        # Adds loras to the subfix via dropdown
        def on_apply_lora(lora_model):
            if lora_model is None:
                print("No LORA model selected.")
                return

            print("Applying LORA model...")
            lora_name = lora_model
            print(f"Selected LORA model: {lora_name}")
            if lora_name:
                trigger_words = get_lora_trigger_words(lora_name)
                print(f"Success Trigger words for {lora_name}: {trigger_words}")

                current_text = params["prompt_subfix"]
                # Remove existing lora trigger words
                #current_text = re.sub(r"<lora:[^>]+>", "", current_text)
                current_text = re.sub(r"\{[^}]+\}", "", current_text)
                lora, ext = os.path.splitext(lora_name)
                new_prompt_subfix = f"{current_text} {trigger_words} <lora:{lora}:0.8>"
                params['prompt_subfix'] = new_prompt_subfix
                prompt_subfix.value = new_prompt_subfix


                print(f"Updated prompt_subfix: {prompt_subfix.value}")

            return new_prompt_subfix
        
        # Adds embeddings to the prefix via dropdown        
        def on_apply_embedding(embedding_model):
            if embedding_model is None:
                print("No embedding selected.")
                return

            print("Applying embedding model...")
            
            ti_name = embedding_model
            print(f"Selected embedding: {ti_name}")
            if ti_name:
                trigger_words = get_embedding_trigger_words(ti_name)
                print(f"Success Trigger words for {ti_name}: {trigger_words}")
                    
                current_text = params['prompt_prefix']
                #Remove existing embedding trigger words
                current_text = re.sub(r"\{[^}]+\}", "", current_text)
                new_prompt_prefix = f"{current_text} {trigger_words}"
                params['prompt_prefix'] = new_prompt_prefix
                prompt_prefix.value = new_prompt_prefix
                    
                print(f"Updated prompt_prefix: {prompt_prefix.value}")

            return new_prompt_prefix
     
        with gr.Row(scale=1, min_width=400):
            input_prompt = gr.Textbox(lines=1, placeholder=params['input_prompt'], label="Input Prompt", elem_id="iF_prompt_MKR_input_prompt")
            with gr.Column(scale=1, min_width=100):
                use_batch = gr.Checkbox(value=False, label='Use batch')
                with gr.Row():
                    batch_count = gr.Number(label="Batch count:", value=params['batch_count'])
                    batch_size = gr.Slider(1, 6, value=params['batch_size'], step=1, label='batch size')
                        
                selected_character = gr.inputs.Dropdown(label="characters", choices=params['selected_character'])       
        #The Idea is chosing and embeding and lora and will populate the keywords automatically 
        with gr.Accordion('Prefix & TIembeddings', open=True):
            ti_choices = ["None"]
            ti_choices.extend(get_embeddings())
            with gr.Row(scale=1, min_width=400):
                prompt_prefix = gr.Textbox(lines=1, default=prompt_prefix_value, label="Prompt Prefix", elem_id="iF_prompt_MKR_prompt_prefix")

                with gr.Column(scale=1, min_width=100):
                    embedding_model = gr.inputs.Dropdown(label="Embeddings Model", choices=ti_choices, default='')

        with gr.Accordion('Suffix & Loras', open=True):
            lora_choices = ["None"]
            lora_choices.extend(get_loras())
            with gr.Row(scale=1, min_width=400):
                prompt_subfix = gr.Textbox(lines=1, default=prompt_subfix_value, label="Subfix for adding Loras (optional)", elem_id="iF_prompt_MKR_prompt_subfix")

                with gr.Column(scale=1, min_width=100):
                    lora_model = gr.Dropdown(label="Lora Model", choices=lora_choices, default='None')
                    
        with gr.Row():  
            negative_prompt = gr.Textbox(lines=4, default=params['negative_prompt'], label="Negative Prompt", elem_id="iF_prompt_MKR_negative_prompt")
            neg_prompts_dropdown = gr.Dropdown(
                label="neg_prompts", 
                choices=[n[0] for n in neg_prompts],
                type="index", 
                elem_id="iF_prompt_MKR_neg_prompts_dropdown")
             #, default_value=neg_prompts[0][1]

        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                excluded_words = gr.inputs.Textbox(lines=1, placeholder="Enter case-sensitive words to exclude, separated by commas", label="Excluded Words")
                kofi_thx = gr.inputs.Textbox(lines=4, default="Img2Img Mode needs an image as Imput, it might fail without it | Make sure to finish with a coma (') your written inputs so it pravails when you update the dropdowns |", label="ImpactFrames Message")
            with gr.Column(scale=1, min_width=100):
                get_triger_files = gr.Button("Get All Trigger Files", elem_id="iF_prompt_MKR_get_triger_files")
                message = gr.inputs.Textbox(lines=2, default="Creates a file with all the trigger words for each model (takes 3-5 seconds for each model you have) if you already have .civitai.info in your model folder, then you don't need to run this", label="Trigger Message")
                

        use_batch.change(lambda x: params.update({'use_batch': x}), use_batch, None)
        selected_character.change(lambda x: params.update({'selected_character': x}), selected_character, None)
        prompt_prefix.change(lambda x: params.update({'prompt_prefix': x}), prompt_prefix, None)
        input_prompt.change(lambda x: params.update({'input_prompt': x}), input_prompt, None)
        prompt_subfix.change(lambda x: params.update({'prompt_subfix': x}), prompt_subfix, None)
        batch_count.change(lambda x: params.update({"batch_count": x}), batch_count, None)
        batch_size.change(lambda x: params.update({'batch_size': x}), batch_size, None)

        excluded_words.change(lambda x: params.update({'excluded_words': [word.strip() for word in x.split(',')] if x else []}), excluded_words, None)
        get_triger_files.click(get_trigger_files, inputs=[], outputs=[message])
        
        neg_prompts_dropdown.change(on_neg_prompts_change, neg_prompts_dropdown, negative_prompt)
        negative_prompt.change(lambda x: params.update({'negative_prompt': x}), negative_prompt, None)

        embedding_model.change(on_apply_embedding, inputs=[embedding_model], outputs=[prompt_prefix], )
        print("Embedding Model value:", embedding_model.value)
        lora_model.change(on_apply_lora, inputs=[lora_model], outputs=[prompt_subfix])
        print("LORA Model value:", lora_model.value)
        
        return [selected_character, prompt_prefix, input_prompt, prompt_subfix, excluded_words, negative_prompt, use_batch, batch_size, batch_count]
    

  
    def generate_text(self, character, prompt, words, use_batch, batch_size, batch_count):
        generated_texts = []

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
            'history': {'internal': [], 'visible': []},  # this is the history of the chat, it's not used here but it's needed for the chat to work
            'mode': "chat",
            'character': character,
            'instruction_template': 'Wizard-Mega',
            'regenerate': False,
            '_continue': False,
            'stop_at_newline': False,
            'chat_prompt_size': 2048,
            'chat_generation_attempts': 1,
            'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "".\n\n',
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

        if use_batch:
            for i in range(batch_count):
                for i in range(batch_size):
                    response = requests.post("http://127.0.0.1:5000/api/v1/chat",
                                            data=json.dumps(data), headers=headers)

                    if response.status_code == 200:
                        results = json.loads(response.content)['results']
                        if results:
                            history = results[0]['history']
                            if history:
                                visible = history['visible']
                                if visible:
                                    generated_text = visible[-1][1]

                                    if words:
                                        generated_text = ' '.join(word for word in generated_text.split() if word not in words)

                                    if '<audio' in generated_text:
                                        print("Audio has been generated.")
                                        generated_text = re.sub(r'<audio.*?>.*?</audio>', '', generated_text)

                                    generated_texts.append(generated_text)
                        else:
                            print("No results found.")
                    else:
                        print("Error: Request failed with status code, probably Ooga isn't running with API flags check the readme",
                            response.status_code)
        else:
            response = requests.post("http://127.0.0.1:5000/api/v1/chat",
                                    data=json.dumps(data), headers=headers)

            if response.status_code == 200:
                results = json.loads(response.content)['results']
                if results:
                    history = results[0]['history']
                    if history:
                        visible = history['visible']
                        if visible:
                            generated_text = visible[-1][1]

                            if words:
                                generated_text = ' '.join(word for word in generated_text.split() if word not in words)

                            if '<audio' in generated_text:
                                print("Audio has been generated.")
                                generated_text = re.sub(r'<audio.*?>.*?</audio>', '', generated_text)

                            generated_texts.append(generated_text)
                else:
                    print("No results found.")
            else:
                print("Error: Request failed with status code, probably Ooga isn't running with API flags check the readme",
                    response.status_code)

        return generated_texts


    
    def run(self, p, selected_character, prompt_prefix, input_prompt, prompt_subfix, excluded_words, negative_prompt, use_batch, batch_size, batch_count, *args, **kwargs):
        prompts = []
        batch_count = int(batch_count)
        batch_size = int(batch_size)

        generated_texts = self.generate_text(selected_character, input_prompt, excluded_words, use_batch, batch_size, batch_count)
        
        if not generated_texts:
            print("No text generated.")
            return
        for text in generated_texts:
            combined_prompt = prompt_prefix + ' ' + text + ' ' + prompt_subfix
            prompts.append(combined_prompt)

        p.negative_prompt = negative_prompt
        p.prompt_subfix = prompt_subfix
        p.selected_character = selected_character
        p.input_prompt = input_prompt
        
        modules.processing.fix_seed(p)
        
        p.do_not_save_grid = True
        state.job_count = 0
        permutations = 0
        
        state.job_count += len(prompts) * batch_count
        permutations += len(prompts)
            
        print(f"Creating {permutations} image permutations")
        image_results = []
        all_prompts = []
        infotexts = []
        current_seed = p.seed

        
        if len(prompts) == 1 and not use_batch:
            p.prompt = prompts[0]
            for i in range(batch_count * batch_size):
                p.seed = current_seed
                current_seed += 1

                proc = process_images(p)
                temp_grid = images.image_grid(proc.images, batch_size)
                image_results.append(temp_grid)

                all_prompts += proc.all_prompts
                infotexts += proc.infotexts
        else:
            for prompt in prompts:
                p.prompt = prompt
                p.seed = current_seed
                current_seed += 1

                proc = process_images(p)
                temp_grid = images.image_grid(proc.images, batch_size)
                image_results.append(temp_grid)

                all_prompts += proc.all_prompts
                infotexts += proc.infotexts

        if len(prompts) > 1:
            grid = images.image_grid(image_results, batch_size)
            infotexts.insert(0, infotexts[0])
            image_results.insert(0, grid)
            images.save_image(grid, p.outpath_grids, "grid", grid=True, p=p)

        return Processed(p, image_results, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
