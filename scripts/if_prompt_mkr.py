import modules.scripts as scripts
import gradio as gr
import os
import requests
import json
import copy
import re
import modules
import unicodedata
from modules import images, script_callbacks, shared
from modules.processing import Processed, process_images
from modules.shared import state
from scripts.extrafiles import get_negative, get_excluded_words
from scripts.modeltags import get_loras, get_embeddings, get_lora_trigger_words, get_embedding_trigger_words, get_trigger_files


script_dir = scripts.basedir()


def on_ui_settings():
    section=("if_prompt_mkr", "iF_prompt_MKR")
    shared.opts.add_option("HOST", shared.OptionInfo(
      "", "Host ip and port should be :5000 local machine or is different than http://127.0.0.1:5000 ", section=section))
    shared.opts.add_option("character_path", shared.OptionInfo(
      "", 'Select Ooga characters folder X:\oobabooga_windows\text-generation-webui\characters to list all the json characters you have installed', section=section))
    shared.opts.add_option("preset", shared.OptionInfo(
      "", 'Select a Yaml preset from oobabooga The default is "IF_promptMKR_preset" do not add the extension and use the double quotes', section=section))
    shared.opts.add_option("instruction_template", shared.OptionInfo(
      "", 'Set a instruction template of the Model default is "Wizard-Mega" i.e Vicuna-v1.1, WizardLM, Wizard-Mega, Alpaca, use the double quotes', section=section))
    shared.opts.add_option("stopping_string", shared.OptionInfo(
      "", 'Write a custom stopping string default is "### Assistant:" ', section=section))
    

script_callbacks.on_ui_settings(on_ui_settings)



class Script(scripts.Script):

    def title(self):
        return "iF_prompt_MKR"
    
    
    def ui(self, is_img2img):

        def get_character_list():
            character_path = shared.opts.data.get("character_path", None)

            print(f"Character Path: {character_path}")
            if character_path and os.path.exists(character_path):
                character_list = [os.path.splitext(f)[0] for f in os.listdir(character_path) if f.endswith('.json')]
                print(f"Character List: {character_list}")
                return character_list
            else:
                return []

        

        neg_prompts = get_negative(os.path.join(script_dir, "negfiles"))


        character_list = get_character_list()


        params = {
            'selected_character': character_list if character_list else ['if_ai_SD', 'iF_Ai_SD_b', 'iF_Ai_SD_NSFW', 'IF_prompt_MKR'],
            'prompt_prefix': '',
            'input_prompt': '(CatGirl warrior:1.2), legendary sword,',
            'negative_prompt': '(nsfw), (worst quality, low quality:1.4), ((text, signature, captions):1.3),',
            'prompt_subfix': 'dark theme <lora:LowRA:0.6>, add_detail <lora:add_detail:0.6>,',
            'prompt_per_batch': False,
            'prompt_per_image': False,
            'batch_size': 1,
            'batch_count': 1,
            'exclude_words': [],
            'remove_weights': False,
            'remove_author': False,
        }

        prompt_prefix_value = params['prompt_prefix']
        prompt_subfix_value = params['prompt_subfix']


        
        def on_neg_prompts_change(x):
            
            filename = neg_prompts[x][1]

            with open(filename, 'r') as file:
                new_neg_prompt = file.read()

            params.update({'negative_prompt': str(new_neg_prompt)})
            negative_prompt.value = str(new_neg_prompt)

            return new_neg_prompt
        

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
                current_text = re.sub(r"\{[^}]+\}", "", current_text)
                lora, ext = os.path.splitext(lora_name)
                new_prompt_subfix = f"{current_text} {trigger_words} <lora:{lora}:0.8>"
                params['prompt_subfix'] = new_prompt_subfix
                prompt_subfix.value = new_prompt_subfix


                print(f"Updated prompt_subfix: {prompt_subfix.value}")

            return new_prompt_subfix


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
                current_text = re.sub(r"\{[^}]+\}", "", current_text)
                new_prompt_prefix = f"{current_text} {trigger_words}"
                params['prompt_prefix'] = new_prompt_prefix
                prompt_prefix.value = new_prompt_prefix
                    
                print(f"Updated prompt_prefix: {prompt_prefix.value}")

            return new_prompt_prefix
        

     
        with gr.Row(scale=1, min_width=400):
            selected_character = gr.Dropdown(label="characters", choices=params['selected_character']) 
            with gr.Row():
                prompt_mode = gr.Radio(['Default', 'Per Image', 'Per Batch'], label='Prompt Mode')
        with gr.Row(scale=1, min_width=400):
            input_prompt = gr.Textbox(lines=1, placeholder=params['input_prompt'], label="Input Prompt", elem_id="iF_prompt_MKR_input_prompt")      
            with gr.Row():
                batch_count = gr.Number(label="Batch count:", value=params['batch_count'])
                batch_size = gr.Slider(1, 8, value=params['batch_size'], step=1, label='batch size')
         
        with gr.Accordion('Prefix & TIembeddings', open=True):
            ti_choices = ["None"]
            ti_choices.extend(get_embeddings())
            with gr.Row(scale=1, min_width=400):
                prompt_prefix = gr.Textbox(lines=1, default=prompt_prefix_value, label="Prefix or embeddigs (optonal)", elem_id="iF_prompt_MKR_prompt_prefix")

                with gr.Column(scale=1, min_width=100):
                    embedding_model = gr.Dropdown(label="Embeddings Model", choices=ti_choices, default='')

        with gr.Accordion('Suffix & Loras', open=True):
            lora_choices = ["None"]
            lora_choices.extend(get_loras())
            with gr.Row(scale=1, min_width=400):
                prompt_subfix = gr.Textbox(lines=1, default=prompt_subfix_value, label="Suffix or Loras (optional)", elem_id="iF_prompt_MKR_prompt_subfix")

                with gr.Column(scale=1, min_width=100):
                    lora_model = gr.Dropdown(label="Lora Model", choices=lora_choices, default='None')
                    
        with gr.Row():  
            negative_prompt = gr.Textbox(lines=4, default=params['negative_prompt'], label="Negative Prompt", elem_id="iF_prompt_MKR_negative_prompt")
            neg_prompts_dropdown = gr.Dropdown(
                label="neg_prompts", 
                choices=[n[0] for n in neg_prompts],
                type="index", 
                elem_id="iF_prompt_MKR_neg_prompts_dropdown")
             

        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                dynamic_excluded_words = gr.Textbox(lines=1, placeholder="Enter case-sensitive words to exclude, separated by commas", label="Excluded Words")
                remove_weights = gr.Checkbox(label="Remove weights from prompts", default=False)
                remove_author = gr.Checkbox(label="Remove Artists", default=False)
            with gr.Column(scale=1, min_width=100):
                get_triger_files = gr.Button("Get All Trigger Files", elem_id="iF_prompt_MKR_get_triger_files")
                message = gr.Textbox(lines=2, default="Creates a file with all the trigger words for each model (takes 3-5 seconds for each model you have) if you already have .civitai.info in your model folder, then you don't need to run this", label="Trigger Message")
                

        selected_character.change(lambda x: params.update({'selected_character': x}), selected_character, None)
        prompt_prefix.change(lambda x: params.update({'prompt_prefix': x}), prompt_prefix, None)
        input_prompt.change(lambda x: params.update({'input_prompt': x}), input_prompt, None)
        prompt_subfix.change(lambda x: params.update({'prompt_subfix': x}), prompt_subfix, None)
        batch_count.change(lambda x: params.update({"batch_count": x}), batch_count, None)
        batch_size.change(lambda x: params.update({'batch_size': x}), batch_size, None)
        remove_weights.change(lambda x: params.update({'remove_weights': x}), remove_weights, None)
        remove_author.change(lambda x: params.update({'remove_author': x}), remove_author, None)
        dynamic_excluded_words.change(lambda x: params.update({'dynamic_excluded_words': [word.strip() for word in x.split(',')] if x else []}), dynamic_excluded_words, None)
        get_triger_files.click(get_trigger_files, inputs=[], outputs=[message])
        
        neg_prompts_dropdown.change(on_neg_prompts_change, neg_prompts_dropdown, negative_prompt)
        negative_prompt.change(lambda x: params.update({'negative_prompt': x}), negative_prompt, None)

        embedding_model.change(on_apply_embedding, inputs=[embedding_model], outputs=[prompt_prefix], )
        print("Embedding Model value:", embedding_model.value)
        lora_model.change(on_apply_lora, inputs=[lora_model], outputs=[prompt_subfix])
        print("LORA Model value:", lora_model.value)
        
        return [selected_character, prompt_prefix, input_prompt, prompt_subfix, dynamic_excluded_words, negative_prompt, prompt_mode, batch_count, batch_size, remove_weights, remove_author]
    


    def send_request(self, data, headers):
        
        HOST = shared.opts.data.get('HOST', None)
        if not HOST:
            HOST = '127.0.0.1:5000'
        print(f"iF_prompt_MKR: Conecting to {HOST}")

        URI = f'http://{HOST}/api/v1/chat'
        response = requests.post(URI, data=json.dumps(data), headers=headers)
        if response.status_code != 200:
            print(f"iF_prompt_MKR: _Error_ Request failed with status code, probably Ooga isn't running with API flags check the readme", response.status_code)
            return None

        results = json.loads(response.content)['results']
        if not results:
            print("No results found.")
            return None

        history = results[0]['history']
        if not history:
            return None

        visible = history['visible']
        if not visible:
            return None

        return visible[-1][1]



    
    def process_text(self, generated_text, not_allowed_words, remove_weights, remove_author):

        
        if remove_author:
            generated_text = re.sub(r'\bby:.*', '', generated_text)

        if remove_weights:
            
            generated_text = re.sub(r'\(([^)]*):[\d\.]*\)', r'\1', generated_text)
            generated_text = re.sub(r'(\w+):[\d\.]*(?=[ ,]|$)', r'\1', generated_text)
        



        for word in not_allowed_words:
            word_regex = r'\b' + re.escape(word) + r'\b'
            generated_text = re.sub(word_regex, '', generated_text, flags=re.IGNORECASE)
        
        for phrase in re.findall(r'\(([^)]*)\)', generated_text): 
            original_phrase = phrase
            for word in not_allowed_words:
                word_regex = r'\b' + re.escape(word) + r'\b'
                phrase = re.sub(word_regex, '', phrase, flags=re.IGNORECASE) 
            generated_text = generated_text.replace('(' + original_phrase + ')', '(' + phrase + ')') 
            
        generated_text = re.sub(r'\(\s*,\s*,\s*\)', '(, )', generated_text)
        generated_text = re.sub(r'\s{2,}', ' ', generated_text)
        generated_text = re.sub(r'\.,', ',', generated_text)
        generated_text = re.sub(r',,', ',', generated_text)


        if '<audio' in generated_text:
            print(f"iF_prompt_MKR: Audio has been generated.")
            generated_text = re.sub(r'<audio.*?>.*?</audio>', '', generated_text)

        return generated_text


    def generate_text(self, p, character, prompt, not_allowed_words, prompt_per_image, prompt_per_batch, default_mode, batch_count, batch_size, remove_weights, remove_author):
        generated_texts = []

        print(f"iF_prompt_MKR: Generating a text prompt using: {character}")
        stopping = shared.opts.data.get("stopping_string", None)
        if not stopping:
            stopping = "### Assistant:"
        preset = shared.opts.data.get("preset", None)
        if not preset:
            preset = 'IF_promptMKR_preset'
        instruction_template = shared.opts.data.get("instruction_template", None)
        if not instruction_template: 
            instruction_template = 'Alpaca'
        if not character:
            character = "IFpromptMKR"

        data = {
            'user_input': prompt,
            'history': {'internal': [], 'visible': []},
            'mode': "chat",
            'your_name': "You",
            'character': character,
            'instruction_template': instruction_template,
            'preset': preset,
            'regenerate': False,
            '_continue': False,
            'stop_at_newline': False,
            'chat_prompt_size': 2048,
            'chat_generation_attempts': 1,
            'chat-instruct_command': 'Act like a prompt creator, brake keywords by comas, provide high quality, non-verboose, coherent, brief, concise, and not superfluous prompts, Only write the visuals elements of the picture, Never write art commentaries or intentions. Construct the prompt with the componet format, Always include all the keywords from the request verbatim as the main subject of the response: "".\n\n',
            'seed': -1,
            'add_bos_token': True,
            'custom_stopping_strings': [stopping,],
            'truncation_length': 2048,
            'ban_eos_token': False,
        }
        headers = {
            "Content-Type": "application/json"
        }

        if prompt_per_image:
            for i in range( batch_count * batch_size):
                generated_text = self.send_request(data, headers)
                if generated_text:
                    processed_text = self.process_text(generated_text, not_allowed_words, remove_weights, remove_author)
                    generated_texts.append(processed_text)
        elif prompt_per_batch:
            for i in range( batch_count):
                generated_text = self.send_request(data, headers)
                if generated_text:
                    processed_text = self.process_text(generated_text, not_allowed_words, remove_weights, remove_author)
                    generated_texts.append(processed_text)
        elif default_mode:
            generated_text = self.send_request(data, headers)
            if generated_text:
                processed_text = self.process_text(generated_text, not_allowed_words, remove_weights, remove_author)
                generated_texts.append(processed_text)

        return generated_texts


    def run(self, p, selected_character, prompt_prefix, input_prompt, prompt_subfix, dynamic_excluded_words, negative_prompt, prompt_mode, batch_count, batch_size, remove_weights, remove_author, *args, **kwargs):
        prompts = []
        prompt_per_image = (prompt_mode == 'Per Image')
        prompt_per_batch = (prompt_mode == 'Per Batch')
        default_mode = (prompt_mode == 'Default')
        batch_count = int(batch_count)
        batch_size = int(batch_size) 
        excluded_path = os.path.join(script_dir, "excluded/excluded_words.txt")   
        not_allowed_words = get_excluded_words(dynamic_excluded_words, excluded_path)
       
        generated_texts = self.generate_text(p, selected_character, input_prompt, not_allowed_words, prompt_per_image, prompt_per_batch, default_mode, batch_count, batch_size, remove_weights, remove_author)

        if not generated_texts:
            print(f"iF_prompt_MKR: No generated texts found for {selected_character}. Check if Oobabooga is running in API mode and the character is available in Oobabooga's character folder.")
            return

        for text in generated_texts:
            combined_prompt = prompt_prefix + ' ' + text + ' ' + prompt_subfix
            prompts.append(combined_prompt)

        p.prompts = prompts
        p.negative_prompt = negative_prompt
        p.prompt_subfix = prompt_subfix
        p.selected_character = selected_character
        p.input_prompt = input_prompt


        return self.process_images(p, prompt_per_image, prompt_per_batch, default_mode, batch_count, batch_size)


    def process_images(self, p, prompt_per_image, prompt_per_batch, default_mode, batch_count, batch_size):
        modules.processing.fix_seed(p)

        p.do_not_save_grid = True
        state.job_count = 0
        generations = 0
        generations += len(p.prompts)
            
        print(f"Creating {generations} image generations")

        imges_p = []
        all_prompts = []
        infotexts = [] 
        current_seed = p.seed


        if default_mode:
            p.n_iter = batch_count
            p.batch_size = batch_size
            state.job_count += len(p.prompts) * p.n_iter
            image_count = p.batch_size * p.n_iter
            generations = image_count
            print(f"iF_prompt_MKR: Processing {generations} image generations will have the same prompt")
            p.prompt = p.prompts[0]
            p.seed = current_seed
            current_seed += 1
                
            proc = process_images(p)    
            for img in proc.images:
                imges_p.append(img)

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts
                

        elif prompt_per_batch:
            state.job_count += len(p.prompts) * batch_count
            total_images = batch_count * batch_size
            generations = batch_size
            print(f"iF_prompt_MKR: Processing {generations} images will share {batch_count} prompts for a total of {total_images} image generations")
            for prompt in p.prompts:
                p.prompt = prompt
                for i in range(batch_size):
                    p.seed = current_seed
                    current_seed += 1
                    
                    proc = process_images(p)
                    tmp_grid = images.image_grid(proc.images, batch_size)
                    imges_p.append(tmp_grid)


                    all_prompts += proc.all_prompts
                    infotexts += proc.infotexts  
                           

        elif prompt_per_image:
            state.job_count += len(p.prompts) * batch_count
            generations = batch_count * batch_size
            print(f"iF_prompt_MKR: Processing {generations} image generations will different prompts")
            for prompt in p.prompts:
                p.prompt = prompt
                p.seed = current_seed
                current_seed += 1

                proc = process_images(p)
                tmp_grid = images.image_grid(proc.images, batch_size)
                imges_p.append(tmp_grid)
                
                all_prompts += proc.all_prompts
                infotexts += proc.infotexts

        
        if len(p.prompts) > 1 :
            
            grid = images.image_grid(imges_p, batch_size)
            infotexts.insert(0, infotexts[0])
            imges_p.insert(0, grid)
            images.save_image(grid, p.outpath_grids, "grid", grid=True, p=p)
        

        return Processed(p, imges_p, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)    
