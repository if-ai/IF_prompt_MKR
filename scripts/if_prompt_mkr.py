import modules.scripts as scripts
import gradio as gr
import os
import requests
import json
import copy
import re
import modules
import unicodedata
import base64
import io
import tempfile
import textwrap
import glob
import uuid
from modules import images, script_callbacks, shared
from modules.processing import Processed, process_images
from modules.shared import state
from scripts.extrafiles import get_negative, get_excluded_words
from scripts.modeltags import get_loras, get_embeddings, get_lora_trigger_words, get_embedding_trigger_words, get_trigger_files
from openai import OpenAI

from PIL import Image 
from pathlib import Path


script_dir = scripts.basedir()


def on_ui_settings():
    section=("if_prompt_mkr", "iF_prompt_MKR")
    shared.opts.add_option("api_choice", shared.OptionInfo(
      "", "Select the API for generating prompts: 'Oobabooga' or 'Ollama'.", section=section))
    shared.opts.add_option("base_ip", shared.OptionInfo(
      "127.0.0.1", "Base IP address for the APIs. Default is '127.0.0.1'.", section=section))
    shared.opts.add_option("oobabooga_port", shared.OptionInfo(
      "5000", "Port for Oobabooga API. Default is '5000'.", section=section))
    shared.opts.add_option("ollama_port", shared.OptionInfo(
      "11434", "Port for Ollama API. Default is '11434'.", section=section))
    shared.opts.add_option("character_path", shared.OptionInfo(
      "", 'Select TGWUI characters folder X:\Text-generation-webui\characters to list all the json characters you have installed', section=section))
    shared.opts.add_option("preset", shared.OptionInfo(
      "", 'Select a Yaml preset from oobabooga The default is "IF_promptMKR_preset" do not add the extension and use the double quotes', section=section))
    
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

        character_list = get_character_list()

        neg_prompts = get_negative(os.path.join(script_dir, "negfiles"))
        embellish_prompts = get_negative(os.path.join(script_dir, "embellishfiles"))
        style_prompts = get_negative(os.path.join(script_dir, "stylefiles"))



        params = {
            'selected_character': character_list if character_list else ['if_ai_SD', 'iF_Ai_SD_b', 'iF_Ai_SD_NSFW', 'IF_prompt_MKR'],
            'prompt_prefix': '',
            'input_prompt': '(CatGirl warrior:1.2), legendary sword,',
            'negative_prompt': '(nsfw), (worst quality, low quality:1.4), ((text, signature, captions):1.3),',
            'embellish_prompts': '',
            'style_prompts': '',
            'prompt_subfix': '',
            'prompt_per_batch': False,
            'prompt_per_image': False,
            'batch_size': 1,
            'batch_count': 1,
            'exclude_words': [],
            'remove_weights': False,
            'remove_author': False,
            'text_models': [],
            'vision_models': [],
            'selected_vision_model': 'default_model', 
            'selected_text_model': 'default_model'
        }

        prompt_prefix_value = params['prompt_prefix']
        prompt_subfix_value = params['prompt_subfix']

        current_vision_model = 'default_model'
        current_text_model = 'default_model'

        
        def on_neg_prompts_change(x):
            
            filename = neg_prompts[x][1]

            with open(filename, 'r') as file:
                new_neg_prompt = file.read()

            params.update({'negative_prompt': str(new_neg_prompt)})
            negative_prompt.value = str(new_neg_prompt)

            return new_neg_prompt
        
        def on_embellishments_change(x):
            
            filename = embellish_prompts[x][1]

            with open(filename, 'r') as file:
                new_embellish_prompt = file.read()

            params.update({'embellish_prompt': str(new_embellish_prompt)})
            embellish_prompt.value = str(new_embellish_prompt)

            return new_embellish_prompt
        
        def on_styles_change(x):
            filename = style_prompts[x][1]

            with open(filename, 'r') as file:
                new_style_prompt = file.read()

            params.update({'style_prompt': str(new_style_prompt)})
            style_prompt.value = str(new_style_prompt)  

            return new_style_prompt       

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
                embedding, ext = os.path.splitext(ti_name)
                new_prompt_prefix = f"{current_text} {trigger_words}"
                params['prompt_prefix'] = new_prompt_prefix
                prompt_prefix.value = new_prompt_prefix
                    
                print(f"Updated prompt_prefix: {prompt_prefix.value}")

            return new_prompt_prefix
        
        def update_vision_model_list(select_vision_model):
            api_choice = shared.opts.data.get('api_choice', 'Oobabooga').lower()
            base_ip = shared.opts.data.get('base_ip', '127.0.0.1')
            params['vision_models'] = []

            if api_choice == 'ollama':
                port = shared.opts.data.get('ollama_port', '11434')
                api_url = f'http://{base_ip}:{port}/api/tags'
                try:
                    response = requests.get(api_url)
                    response.raise_for_status()
                    params['vision_models'] = [model['name'] for model in response.json()['models']]
                except Exception as e:
                    print(f"Failed to fetch models from Ollama: {e}")

            elif api_choice == 'oobabooga':
                port = shared.opts.data.get('oobabooga_port', '5000')
                URI = f'http://{base_ip}:{port}/v1/internal/model/list'
                try:
                    response = requests.get(URI)
                    response.raise_for_status()
                    params['vision_models'] = response.json().get('model_names', [])
                except Exception as e:
                    print(f"Failed to fetch models from Oobabooga: {e}")
            select_vision_model.choices = params['vision_models']
            if params['selected_vision_model'] not in params['vision_models']:
                if params['vision_models']: 
                    params['selected_vision_model'] = params['vision_models'][0]
                else:
                    params['selected_vision_model'] = "Default"
            select_vision_model.value = params['selected_vision_model']

            return params['vision_models']
        
        def update_text_model_list(select_text_model):
            api_choice = shared.opts.data.get('api_choice', 'Oobabooga').lower()
            base_ip = shared.opts.data.get('base_ip', '127.0.0.1')
            params['text_models'] = []

            if api_choice == 'ollama':
                port = shared.opts.data.get('ollama_port', '11434')
                api_url = f'http://{base_ip}:{port}/api/tags'
                try:
                    response = requests.get(api_url)
                    response.raise_for_status()
                    params['text_models'] = [model['name'] for model in response.json()['models']]
                except Exception as e:
                    print(f"Failed to fetch models from Ollama: {e}")

            elif api_choice == 'oobabooga':
                port = shared.opts.data.get('oobabooga_port', '5000')
                URI = f'http://{base_ip}:{port}/v1/internal/model/list'
                try:
                    response = requests.get(URI)
                    response.raise_for_status()
                    params['text_models'] = response.json().get('model_names', [])
                except Exception as e:
                    print(f"Failed to fetch models from Oobabooga: {e}")
            select_text_model.choices = params['text_models']
            if params['selected_text_model'] not in params['text_models']:
                if params['text_models']: 
                    params['selected_text_model'] = params['text_models'][0]
                else:
                    params['selected_text_model'] = "Default"
            select_text_model.value = params['selected_text_model']

            return params['text_models']
        
        def on_text_model_selected(selected_text_model_name):
            global current_text_model
            current_text_model = selected_text_model_name
            print("Text Model Selected:", current_text_model)

        def on_vision_model_selected(selected_vision_model_name):
            global current_vision_model
            current_vision_model = selected_vision_model_name
            print("Vision Model Selected:", current_vision_model)
           
        def load_model_oobabooga(selected_vision_model_name, base_ip='127.0.0.1', port='5000'):
            global current_vision_model
            selected_vision_model_name = current_vision_model
            uri = f'http://{base_ip}:{port}/v1/internal/model/load'
            headers = {"Content-Type": "application/json"}
            data = {
                "model_name": selected_vision_model_name,
                "args": {"multimodal-pipeline": 'llava-v1.5-7b', "load_in_4bit": True },  
                "settings": {"instruction_template": 'LLaVA' }
            }
            try:
                response = requests.post(uri, headers=headers, json=data)
                if response.status_code == 200:
                    print("Model loaded successfully.")
                    return True
                else:
                    print(f"Failed to load model, status code: {response.status_code}")
                    return False
            except Exception as e:
                print(f"Error while loading model: {e}")
                return False
        def get_images(directory):
            image_paths = []
            for filename in os.listdir(directory):
                if not filename.endswith(':Zone.Identifier'):
                    full_path = os.path.join(directory, filename)
                    image_paths.append(full_path)
            return image_paths

        
        def batch_describe_pictures(common_image_prompt):
            image_directory = os.path.join(script_dir, "put_caption_images_here")
            image_paths = get_images(image_directory)
            dataset = []

            for image_path in image_paths:
                image = Image.open(image_path)  # Ensure this is an image object
                image_file_name = os.path.basename(image_path)
                caption = describe_picture(image, common_image_prompt)  # Correct order of parameters

                entry = {
                    "id": str(uuid.uuid4()),
                    "image": image_file_name,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
                        },
                        {
                            "from": "gpt",
                            "value": caption
                        },
                    ]
                }
                dataset.append(entry)

            json_filename = 'llava_dataset.json'
            with open(os.path.join(image_directory, json_filename), 'w') as f:
                json.dump(dataset, f, indent=4)

            return json_filename

        def process_single_image(image, prompt):
            caption = describe_picture(image, prompt)
            return caption
        def describe_picture(image, image_prompt=None):
            global current_vision_model
            selected_model = current_vision_model
            print(f"You are using the {selected_model} model, make sure it is a vision or multimodal model")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
                image.save(image_file, format='JPEG')
                image_path = image_file.name

            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            system_message = textwrap.dedent("""\
            Act as a visual prompt maker with the following guidelines:
            - Describe the image in vivid detail.
            - Break keywords by commas.
            - Provide high-quality, non-verbose, coherent, concise, and not superfluous descriptions.
            - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
            - Construct the prompt by describing framing, subjects, scene elements, background, aesthetics.
            - Limit yourself up to 7 keywords per component  
            - Be varied and creative.
            - Always reply on the same line, use around 100 words long. 
            - Do not enumerate or enunciate components.
            - Do not include any additional information in the response.                                                       
            The following is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship:
            'Epic, Cover Art, Full body shot, dynamic angle, A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy. Ciberpunk, dramatic lighthing, highly detailed. ' 
            Make a visual prompt for the following Image:
            """) if not image_prompt else "Please analyze the image and respond to the user's question."
            user_message = image_prompt if image_prompt else textwrap.dedent("""\
            Act as a visual prompt maker with the following guidelines:
            - Describe the image in vivid detail.
            - Break keywords by commas.
            - Provide high-quality, non-verbose, coherent, concise, and not superfluous descriptions.
            - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
            - Construct the prompt by describing framing, subjects, scene elements, background, aesthetics.
            - Limit yourself up to 7 keywords per component  
            - Be varied and creative.
            - Always reply on the same line, use around 100 words long. 
            - Do not enumerate or enunciate components.
            - Do not include any additional information in the response.                                                       
            The following is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship:
            'Epic, Cover Art, Full body shot, dynamic angle, A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy. Ciberpunk, dramatic lighthing, highly detailed. ' 
            Make a visual prompt for the following Image:
            """)
            
            api_choice = shared.opts.data.get('api_choice', 'Ollama').lower()
            base_ip = shared.opts.data.get('base_ip', '127.0.0.1')
            headers = {"Content-Type": "application/json"}

            if api_choice == 'ollama':
                port = shared.opts.data.get('ollama_port', '11434')
                api_url = f'http://{base_ip}:{port}/api/generate'

                data = {
                    "model": selected_model,
                    "system": system_message,
                    "prompt": user_message, 
                    "stream": False,
                    "images": [base64_image]
                }

                headers = {"Content-Type": "application/json"}
                try:
                    response = requests.post(api_url, headers=headers, json=data)
                    if response.status_code == 200:
                        response_data = response.json()
                        prompt_response = response_data.get('response', 'No response text found')
                        print(f"Caption for image {os.path.basename(image_path)}: {prompt_response}")
                        
                        # Clean up the temporary file
                        os.unlink(image_path)
                        return prompt_response
                    else:
                        print(f"Failed to generate prompt based on the image, status code: {response.status_code}")
                        return f"Failed to generate prompt for image {os.path.basename(image_path)}"
                except Exception as e:
                    print(f"Error while generating caption: {e}")
                    return f"Error while generating caption for image {os.path.basename(image_path)}"
          
            elif api_choice == 'oobabooga':
                port = shared.opts.data.get('oobabooga_port', '5000')
                HOST = f"{base_ip}:{port}"
                if load_model_oobabooga(selected_model, base_ip, port):
                    URI = f'http://{HOST}/v1/chat/completions'
                    data = {
                        'model': selected_model,
                        'messages': [{"role": "assistant", "content": system_message}, {"role": "user", "content": user_message, "images": [base64_image]}]
                    }

                    try:
                        response = requests.post(URI, headers=headers, json=data)
                        if response.status_code == 200:
                            prompt_response = response.json()['choices'][0]['message']['content']
                            return prompt_response
                        else:
                            return f"Failed to generate prompt based on the image, status code: {response.status_code}"
                    except Exception as e:
                        print(f"Error: {e}")
                        return "Failed to generate prompt based on the image"    
      

        with gr.Accordion('Text Model', open=True):
            with gr.Group():
                with gr.Row(scale=1):
                        with gr.Column(scale=1, min_width=100):
                            selected_character = gr.Dropdown(label="characters", choices=params['selected_character']) 
                        with gr.Column(scale=1, min_width=100):
                            select_text_model = gr.Dropdown(label="Text-Model", choices=[], value=params['selected_text_model'])
                        
                with gr.Row(scale=2, min_width=400):
                    input_prompt = gr.Textbox(lines=1, label="Input Prompt", value='(CatGirl warrior:1.2), legendary sword,', elem_id="iF_prompt_MKR_input_prompt")     
                    with gr.Row():
                        prompt_mode = gr.Radio(['Default', 'Per Image', 'Per Batch'], label='Prompt Mode', value='Default')
                        with gr.Column(scale=1, min_width=100):
                            batch_count = gr.Number(label="Batch count:", value=params['batch_count'])
                            batch_size = gr.Slider(1, 8, value=params['batch_size'], step=1, label='batch size')

        with gr.Accordion('Styling(Optional)', open=True):
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        embellish_prompt = gr.Textbox(lines=2, label="embellish Prompt", elem_id="iF_prompt_MKR_embellish_prompt")
                        embellish_prompts_dropdown = gr.Dropdown(
                            label="embellish Prompts",
                            choices=[e[0] for e in embellish_prompts],
                            type="index",
                            elem_id="iF_prompt_MKR_embellish_prompts_dropdown"
                        )
                    with gr.Column(scale=1, min_width=100):
                        style_prompt = gr.Textbox(lines=2, label="Style Prompt", elem_id="iF_prompt_MKR_style_prompt")
                        style_prompts_dropdown = gr.Dropdown(
                            label="Style Prompts",
                            choices=[s[0] for s in style_prompts],
                            type="index",
                            elem_id="iF_prompt_MKR_style_prompts_dropdown"
                        )     
         
        with gr.Accordion('Prefix & TIembeddings', open=True):
            ti_choices = ["None"]
            ti_choices.extend(get_embeddings())
            with gr.Row(scale=1, min_width=400):
                prompt_prefix = gr.Textbox(lines=1, default=prompt_prefix_value, label="Prefix or embeddigs (optonal)", elem_id="iF_prompt_MKR_prompt_prefix")

                with gr.Column(scale=1, min_width=100):
                    embedding_model = gr.Dropdown(label="Embeddings Model", choices=ti_choices, default='None')

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
        
        with gr.Accordion('Vision MultiModal', open=True):
            with gr.Group():
                with gr.Row():
                    image_prompt = gr.Textbox(
                        label="Analyse", placeholder="What would you call the style of this picture and who could be the artist?", scale=4
                    )
                    submit = gr.Button(
                        "Submit",
                        scale=1,
                    )
                with gr.Row():
                    img = gr.Image(type="pil", label="Upload or Drag an Image")
                    output = gr.TextArea(label="Response")  
                    with gr.Row():
                        select_vision_model = gr.Dropdown(label="Vision-Model", choices=[], value=params['selected_vision_model'])
        
        with gr.Accordion('Batch Image Captioner', open=False):
            with gr.Group():
                with gr.Row():
                    common_image_prompt = gr.Textbox(label="Ask a custom prompt", placeholder="Enter a common prompt for all images (optional)")
                    with gr.Column(scale=1, min_width=100):
                        batch_caption_button = gr.Button("Generate Captions for All Images in Folder")
                        batch_caption_result = gr.Textbox(label="When completed the Filename will appear here &on the image folder", interactive=False)
                

        batch_caption_button.click(batch_describe_pictures, inputs=[common_image_prompt], outputs=[batch_caption_result])
        select_text_model.change(lambda x: params.update({'selected_text_model': x}), select_text_model, None)
        update_text_model_list(select_text_model)
        select_text_model.change(on_text_model_selected, select_text_model, None)
        select_vision_model.change(lambda x: params.update({'selected_vision_model': x}), select_vision_model, None)
        update_vision_model_list(select_vision_model)
        select_vision_model.change(on_vision_model_selected, select_vision_model, None)
        submit.click(describe_picture, [img, image_prompt], output)
        image_prompt.submit(describe_picture, [img, image_prompt], output)
        batch_caption_button.click(batch_describe_pictures, inputs=[common_image_prompt], outputs=[batch_caption_result])
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
        style_prompts_dropdown.change(on_styles_change, style_prompts_dropdown, style_prompt)
        style_prompt.change(lambda x: params.update({'style_prompt': x}), style_prompt, None)
        embellish_prompts_dropdown.change(on_embellishments_change, embellish_prompts_dropdown, embellish_prompt)
        embellish_prompt.change(lambda x: params.update({'embellish_prompt': x}), embellish_prompt, None)
        neg_prompts_dropdown.change(on_neg_prompts_change, neg_prompts_dropdown, negative_prompt)
        negative_prompt.change(lambda x: params.update({'negative_prompt': x}), negative_prompt, None)
        embedding_model.change(on_apply_embedding, inputs=[embedding_model], outputs=[prompt_prefix], )
        print("Embedding Model value:", embedding_model.value)
        lora_model.change(on_apply_lora, inputs=[lora_model], outputs=[prompt_subfix])
        print("LORA Model value:", lora_model.value)
        
        return [selected_character, prompt_prefix, input_prompt, prompt_subfix, dynamic_excluded_words, negative_prompt, prompt_mode, batch_count, batch_size, remove_weights, remove_author, select_text_model, style_prompt, embellish_prompt ]
    
    def send_request(self, data, headers, **kwargs):
        api_choice = shared.opts.data.get('api_choice', 'oobabooga').lower()
        base_ip = shared.opts.data.get('base_ip', '127.0.0.1')
        headers = kwargs.get('headers', {"Content-Type": "application/json"})

        if api_choice == 'oobabooga':
            port = shared.opts.data.get('oobabooga_port', '5000')
            HOST = f"{base_ip}:{port}"
            URI = f'http://{HOST}/v1/chat/completions'
            response = requests.post(URI, headers=headers, json=data, verify=False)
        elif api_choice == 'ollama':
            port = shared.opts.data.get('ollama_port', '11434')
            base_url = f'http://{base_ip}:{port}/v1/chat/completions'
            response = requests.post(base_url, headers=headers, json=data)

        
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            print(f"Error: Request to {api_choice} failed with status code {response.status_code}")
            return None

 
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

    def generate_text(self, p, selected_character, input_prompt, batch_count, batch_size, remove_weights, remove_author, not_allowed_words, prompt_per_image, select_text_model, *args, **kwargs):
        generated_texts = []

        print(f"iF_prompt_MKR: Generating a text prompt using: {selected_character}")
        preset = shared.opts.data.get("preset", None)
        if not preset:
            preset = 'IF_promptMKR_preset'
        if not selected_character:
            selected_character = "IFpromptMKR"

        api_choice = shared.opts.data.get('api_choice', 'oobabooga').lower()

        prime_directive = textwrap.dedent("""\
            Act as a prompt maker with the following guidelines:
            - Break keywords by commas.
            - Provide high-quality, non-verbose, coherent, brief, concise, and not superfluous prompts.
            - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
            - Construct the prompt with the component format:
            1. Start with the subject and keyword description.
            2. Follow with scene keyword description.
            3. Finish with background and keyword description.
            - Limit yourself to no more than 7 keywords per component  
            - Include all the keywords from the user's request verbatim as the main subject of the response.
            - Be varied and creative.
            - Always reply on the same line and no more than 100 words long. 
            - Do not enumerate or enunciate components.
            - Do not include any additional information in the response.                                                       
            The followin is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and cosider the elemnts relationship.
            Example:
            Subject: Demon Hunter, Cyber City.
            prompt: A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy.                             
            Make a prompt for the following Subject:
            """)

        print(prime_directive)

        if api_choice == 'ollama':
             data = {
                'model': select_text_model,
                'messages': [
                    {"role": "system", "content": prime_directive},
                    {"role": "user", "content": input_prompt}
                ],  
            }
        else:  
            data = {
                'messages': [{"role": "user", "content": input_prompt}],
                'mode': "chat",
                'character': selected_character,
                'preset': preset,
            }

        if api_choice == 'oobabooga':
            for i in range(batch_count * batch_size if prompt_per_image else batch_count):
                generated_text = self.send_request(data, headers={"Content-Type": "application/json"})
                if generated_text:
                    processed_text = self.process_text(generated_text, not_allowed_words, remove_weights, remove_author)
                    generated_texts.append(processed_text)
        elif api_choice == 'ollama':
            for i in range(batch_count * batch_size if prompt_per_image else batch_count):
                generated_text = self.send_request(data, headers={"Content-Type": "application/json"})
                if generated_text:
                    processed_text = self.process_text(generated_text, not_allowed_words, remove_weights, remove_author)
                    generated_texts.append(processed_text)

        return generated_texts

    def run(self, p, selected_character, prompt_prefix, input_prompt, prompt_subfix, dynamic_excluded_words, negative_prompt, prompt_mode, batch_count, batch_size, remove_weights, remove_author, select_text_model, style_prompt, embellish_prompt, *args, **kwargs):
        prompts = []
        prompt_per_image = (prompt_mode == 'Per Image')
        prompt_per_batch = (prompt_mode == 'Per Batch')
        default_mode = (prompt_mode == 'Default')
        batch_count = int(batch_count)
        batch_size = int(batch_size) 
        excluded_path = os.path.join(script_dir, "excluded/excluded_words.txt")   
        not_allowed_words = get_excluded_words(dynamic_excluded_words, excluded_path)
        print(f"p: {p}")
        print(f"selected_character: {selected_character}")
        print(f"input_prompt: {input_prompt}")
        print(f"batch_count: {batch_count}")
        print(f"batch_size: {batch_size}")
        print(f"remove_weights: {remove_weights}")
        print(f"remove_author: {remove_author}")
        print(f"not_allowed_words: {not_allowed_words}")
        print(f"prompt_per_image: {prompt_per_image}")
        print(f"select_text_model: {select_text_model}")


        generated_texts = self.generate_text(p, selected_character, input_prompt, batch_count, batch_size, remove_weights, remove_author, not_allowed_words, prompt_per_image, select_text_model)


        if not generated_texts:
            print(f"iF_prompt_MKR: No generated texts found for {selected_character}. Check if Oobabooga is running in API mode and the character is available on tgwui character folder.")
            return

        for text in generated_texts:
            combined_prompt = prompt_prefix + ' ' + embellish_prompt + ' '+ text + ' ' + style_prompt + ' ' + prompt_subfix
            prompts.append(combined_prompt)

        p.prompts = prompts
        p.negative_prompt = negative_prompt
        p.prompt_subfix = prompt_subfix
        p.selected_character = selected_character
        p.input_prompt = input_prompt
        p.select_text_model = select_text_model


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
