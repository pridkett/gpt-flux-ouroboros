import websocket
import json
import uuid
import requests
import urllib.request
import urllib.parse
import argparse
import os
import logging
import datetime
import random

from rich.logging import RichHandler
from rich.traceback import install
from typing import Dict, Any

from image_quality import QualityEvaluator

install()

SERVER_ADDRESS = "localhost:7860"
DEFAULT_WORKFLOW = "workflow_api.json"
DEFAULT_IMAGE = "test_image.jpg"
DEFAULT_ITERATIONS = 3
DEFAULT_WEBSOCKET_TIMEOUT = 60
VARIANCE_STEPDOWN = 4
DEFAULT_VARIANCE = 60

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("run_workflow")
log.setLevel(logging.INFO)

client_id = str(uuid.uuid4())

def queue_prompt(server_address: str, prompt: str):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    log.info(f"Server Address: http://{server_address}/prompt")
    log.debug(f"Data: {data}")    
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(server_address: str, filename: str, subfolder: str, folder_type: str) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()
    
def get_history(server_address: str, prompt_id) -> Dict[Any, Any]:
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        resp = response.read()
        log.debug("Response: %s", resp)
        return json.loads(resp)

def score_image(image_path: str):
    pass

def save_result(ws: websocket._core.WebSocket, server_address:str, prompt:str, output_jsonl_fn: str, additional_json: Dict[str, Any]={}, quality_evaluator: QualityEvaluator=None, variance: int=None, stepdown: int=None):
    prompt_id = queue_prompt(server_address, prompt)['prompt_id']
    log.info("Prompt ID: %s", prompt_id)
    output_images = {}
    queue_remaining = 1
    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] != "crystools.monitor":
                    log.info(f"message: {message}")
                if message['type'] == 'status':
                    queue_remaining = message.get('data',[]).get('status', []).get('exec_info', []).get('queue_remaining', None)
                    # if queue_remaining == 0:
                    #     log.debug("Queue is empty")
                    #     break
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                continue #previews are binary data
    except websocket.WebSocketTimeoutException:
        if queue_remaining > 0:
            log.error(f"Timeout while waiting for response on prompt {prompt_id}. Queue remaining: {queue_remaining}")
            return None
        log.error(f"Timeout while waiting for response on prompt {prompt_id}. Queue is empty.")
    
    history = get_history(server_address, prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(server_address, image['filename'], image['subfolder'], image['type'])

                    images_output.append(image_data)
                output_images[node_id] = images_output

    #Commented out code to display the output images:
    output_jsonl = additional_json
    output_jsonl["sbs_images"] = []
    output_jsonl["images"] = []
    for node_id, image_data in output_images.items():
        log.debug(f"Node ID: {node_id}")
        from PIL import Image
        import io
        if not isinstance(image_data, list):
            image_data = [image_data]
        for ctr, img_data in enumerate(image_data):
            image = Image.open(io.BytesIO(img_data))
            output_file = f"{node_id}-{prompt_id}-{ctr}.png"
            image.save(output_file)
            if prompt[node_id]["inputs"]["filename_prefix"].endswith("SBS"):
                output_jsonl["sbs_images"].append(output_file)
            else:
                if quality_evaluator is not None:
                    quality = quality_evaluator.process_image(output_file, variance=variance)
                quality["filename"] = output_file
                output_jsonl["images"].append(quality)

    highest_score = 0
    best_image = None

    for image in output_jsonl["images"]:
        if not image["blurry"]["value"]:
            if image["NIMA_score"] > highest_score:
                highest_score = image["NIMA_score"]
                best_image = image

    if best_image is not None:
        # Do something with the best image
        log.debug(f"The image with the highest NIMA score is: {best_image}")
        output_jsonl["best_image"] = best_image
        output_jsonl["failed"] = False
    else:
        log.warning(f"No non-blurry images found. Checking with fallback variance of {variance-stepdown}.")

        best_image = max([image for image in output_jsonl["images"] if image["blurry"]["variance"] > variance - stepdown], key=lambda x: x["NIMA_score"], default=None)
        if best_image is not None:
            log.warning(f"Found non-blurry images with lower variance - returning the best of those")
            output_jsonl["best_image"] = best_image
            output_jsonl["failed"] = False
        else:
            log.warning(f"Still unable to find non-blurry images - returning `None` and handing off")
            output_jsonl["best_image"] = None
            output_jsonl["failed"] = True

    # Get the prompt that was used
    output_jsonl["llm_prompt"] = history['outputs']["38"]["text"]
    output_jsonl["time"] = datetime.datetime.now().isoformat()
    
    with open(output_jsonl_fn, "a") as f:
        json.dump(output_jsonl, f)
        f.write("\n")
    return output_jsonl

def upload_file(server_address:str, file, subfolder:str="", overwrite:bool=False):
    try:
        # Wrap file in formdata so it includes filename
        body = {"image": file}
        data = {}
        
        if overwrite:
            data["overwrite"] = "true"
  
        if subfolder:
            data["subfolder"] = subfolder

        resp = requests.post(f"http://{server_address}/upload/image", files=body,data=data)
        
        if resp.status_code == 200:
            data = resp.json()
            # Add the file to the dropdown list and update the widget value
            path = data["name"]
            if "subfolder" in data:
                if data["subfolder"] != "":
                    path = data["subfolder"] + "/" + path
            

        else:
            print(f"{resp.status_code} - {resp.reason}")
    except Exception as error:
        print(error)
    return path

def find_node_id_by_name(workflow, name):
    for node_id in workflow:
        if workflow[node_id]["_meta"]["title"] == name:
            return node_id
    return None

def main(server_address:str, workflow_path:str, image_path:str, api_key:str, output_jsonl_fn: str, iterations: int=3, variance: int=DEFAULT_VARIANCE, stepdown: int=VARIANCE_STEPDOWN):

    # create and load the QualityEvaluator
    quality_evaluator = QualityEvaluator(threshold=variance)

    #load workflow from file
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_data = f.read()

    workflow = json.loads(workflow_data)

    #set the text prompt for our positive CLIPTextEncode
    # workflow["6"]["inputs"]["text"]  = "masterpiece, best quality, a wide angle shot from the front of a girl posing on a bench in a beautiful meadow,:o face, short and rose hair,perfect legs, perfect arms, perfect eyes,perfect body, perfect feet,blue day sky,shorts, beautiful eyes,sharp focus, full body shot"
    # workflow["7"]["inputs"]["text"]  = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry , deformed,nsfw, deformed legs"
    ctr = 0
    
    run_id = str(uuid.uuid4())
    run_start_time = datetime.datetime.now().isoformat()
    run_source_image = image_path
    last_failed = False
    this_variance = variance

    all_results = []

    while ctr < iterations:
        #upload an image
        with open(image_path, "rb") as f:
            comfyui_path_image = upload_file(server_address, f,"",True)

        seed = random.randint(1, 1000000000000000)   
        log.info(f"ctr: {ctr} seed: {seed}")     
        noise_node = find_node_id_by_name(workflow, "RandomNoise")
        workflow[noise_node]["inputs"]["noise_seed"] = seed

        #set the image name for our LoadImage node
        image_node_id = find_node_id_by_name(workflow, "Load Image")
        workflow[image_node_id]["inputs"]["image"] = comfyui_path_image

        # Create a WebSocket object of type websocket._core.WebSocket
        ws: websocket._core.WebSocket = websocket.WebSocket() 
        ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
        ws.settimeout(DEFAULT_WEBSOCKET_TIMEOUT)

        additional_json = {"run_id": run_id, "run_start_time": run_start_time, "seed": seed, "source_image": run_source_image, "ctr": ctr, "previous_image": image_path, "last_failed": last_failed}
        results = save_result(ws, server_address, workflow, output_jsonl_fn=output_jsonl_fn, additional_json=additional_json, quality_evaluator=quality_evaluator, variance=this_variance, stepdown=stepdown)
        if results is not None:

            all_results.extend(results["images"])

            if results.get("failed", False):
                log.info("Failed to generate a non-blurry image - retrying with different seed")
                this_variance = this_variance - stepdown
                log.info(f"Stepping down acceptable variance to: {this_variance}")

                # check the previous generated images
                best_image = max((image for image in all_results if image["blurry"]["variance"] > this_variance), key=lambda x: x["NIMA_score"], default=None)
                last_failed = best_image is None
                
                if best_image:
                    log.info("Found an already generated image with acceptable variance - using that")
                    log.info(f"Best image: {best_image["filename"]} Variance: {best_image["blurry"]["variance"]} NIMA Score: {best_image["NIMA_score"]}")
                    additional_json["ctr"] = ctr + 0.5
                    
                    output_jsonl_d = additional_json
                    output_jsonl_d["best_image"] = best_image
                    output_jsonl_d["failed"] = False
                    output_jsonl_d["llm_prompt"] = results["llm_prompt"]
                    output_jsonl_d["time"] = datetime.datetime.now().isoformat()

                    with open(output_jsonl_fn, "a") as f:
                        json.dump(output_jsonl_d, f)
                        f.write("\n")
                    

            if not last_failed:
                ctr = ctr + 1
                image_path = results["best_image"]["filename"]
                last_failed = False
                this_variance = variance
                all_results = []
        else:
            log.info("Null results - retrying with different seed")
            last_failed = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run workflow script")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key", default=None)
    parser.add_argument("--image", type=str, help="Path to the image file", default=DEFAULT_IMAGE)
    parser.add_argument("--workflow", type=str, help="Path to the workflow JSON file", default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", type=str, help="Server address", default=SERVER_ADDRESS)
    parser.add_argument("--output", type=str, help="Output JSONL file", default="output.jsonl")
    parser.add_argument("--iterations", type=int, help="Number of iterations to run", default=DEFAULT_ITERATIONS)
    parser.add_argument("--variance", type=int, help="Default minimum variance to declare non-blurry", default=DEFAULT_VARIANCE)
    parser.add_argument("--variance_stepdown", type=int, help="Stepdown factor for variance", default=VARIANCE_STEPDOWN)
    args = parser.parse_args()

    server = args.server
    image_path = args.image
    workflow_path = args.workflow
    api_key = args.api_key
    output_jsonl = args.output
    iterations = args.iterations
    variance = args.variance
    stepdown = args.variance_stepdown

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    main(server, workflow_path, image_path, api_key, output_jsonl, iterations, variance=variance, stepdown=stepdown)
