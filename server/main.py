import json
import base64
import io
import os
import tempfile
import wave
import torch
import numpy as np
from typing import List
from pydantic import BaseModel
from scipy import signal
import audioop
from Logger import Log
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile
from fastapi.responses import StreamingResponse

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

import websockets as web

torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.") 

custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")

if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
    model_path = custom_model_path
    print("Loading custom model from", model_path, flush=True)
else:
    print("Loading default model", flush=True)
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Downloading XTTS Model:", model_name, flush=True)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS Model downloaded", flush=True)

print("Loading XTTS", flush=True)
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
model.to(device)
print("XTTS Loaded.", flush=True)

with open("./default_speaker.json", "r") as file:
    speaker = json.load(file)

global connections
connections = {}

print("Running XTTS Server ...", flush=True)

##### Run fastapi #####
app = FastAPI(
    title="XTTS Streaming server",
    description="""XTTS Streaming server""",
    version="0.0.1",
    docs_url="/",
)

#writing the logger code:-connections
logger = Log('handler.log')
logger = logger.initialize_logger_handler()




async def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav



async def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=8000, sample_width=1, channels=1, 
    original_sample_rate=24000
):
    """Return base64 encoded audio with proper resampling"""
    # Convert bytes to numpy array if needed
    if isinstance(frame_input, bytes):
        # Convert based on the width of each sample
        dtype = np.int16 if sample_width == 2 else (np.int8 if sample_width == 1 else np.float32)
        audio_data = np.frombuffer(frame_input, dtype=dtype)
    else:
        audio_data = frame_input


    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()

    # logger.info(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {np.min(audio_data)}, max: {np.max(audio_data)}")
    

    if audio_data.dtype == np.int16:
        audio_data_float = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.int8:
        audio_data_float = audio_data.astype(np.float32) / 127.0
    else:
        audio_data_float = audio_data

    if sample_rate != original_sample_rate:
        # Calculate resampling ratio
        ratio = sample_rate / original_sample_rate
        output_samples = int(len(audio_data_float) * ratio)
        resampled_audio = signal.resample(audio_data_float, output_samples)
        
        if sample_width == 2:
            audio_bytes = (np.clip(resampled_audio, -1, 1) * 32767).astype(np.int16).tobytes()
        elif sample_width == 1:
            audio_bytes = (np.clip(resampled_audio, -1, 1) * 127).astype(np.int8).tobytes()
        else:
            audio_bytes = resampled_audio.astype(np.float32).tobytes()
    
    audio_bytes_mu = audioop.lin2ulaw(audio_bytes, 2)
    wave_64 = base64.b64encode(audio_bytes_mu).decode("utf-8")
    return wave_64

class StreamingInputs(BaseModel):
    speaker_embedding: List[float] = speaker["speaker_embedding"]
    gpt_cond_latent: List[List[float]] = speaker["gpt_cond_latent"]
    text: str
    language: str = "en"
    add_wav_header: bool = False
    stream_chunk_size: str = "20"


async def predict_streaming_generator(stream_id, parsed_input):
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language

    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header

    # Create a queue for audio chunks
    chunk_queue = asyncio.Queue()

    def inference_task():
        try:
            for chunks in model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=stream_chunk_size,
                enable_text_splitting=True
            ):
                chunk_queue.put_nowait(chunks)
            chunk_queue.put_nowait(None)  # Signal end of stream
        except Exception as e:
            chunk_queue.put_nowait(e)  # Pass errors to the async side

    # Run inference in a separate thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, inference_task)

    
    await connections[stream_id].send_json({"chunk": processed_chunk})
        
    await connections[stream_id].send_json({"status": "completed"})
    logger.info(f"Completed stream for {stream_id}")


@app.websocket("/tts_stream")
async def predict_streaming_endpoint(parsed_input: WebSocket):
    await parsed_input.accept()
    logger.info("WebSocket connection established")

    while True:
        data_json = await parsed_input.receive_text()
        data_json = json.loads(data_json)

        if "stream_id" not in data_json:
            logger.info("Missing stream_id in request")
            continue


        connections[data_json["stream_id"]] = parsed_input
        
        try:


            if data_json["text"] is None:
                logger.info(f"Received empty text, closing connection for {data_json['stream_id']}")
                

            # Extract stream_id
            if "stream_id" not in data_json:
                logger.info("Missing stream_id in request")
                await parsed_input.send_json({"error": "Missing stream_id"})
                return
            

            logger.info(f"WebSocket connection ID: {data_json['stream_id']}, {parsed_input}, {data_json}")
            input_data = StreamingInputs(text=data_json["text"])


            await predict_streaming_generator(data_json["stream_id"], input_data)

        except WebSocketDisconnect:
            logger.info(f"WebSocket connection closed and deleted: {data_json['stream_id']}")
            connections[data_json["stream_id"]].close()
            del connections[data_json["stream_id"]]

        except Exception as e:
            logger.info(f"Error: {e}")
            try:
                await connections[data_json["stream_id"]].send_json({"error": str(e)})
            except Exception as e:
                logger.info(f"Error sending error message: {e}")

@app.post("/stop_stream/{stream_id}")
async def stop_stream(stream_id: str):
    if stream_id not in connections:
        return {"message": "Stream not found or already completed", "status": "error"}
    
    # Set the stop flag for this stream
    connections[stream_id]["voice_stop"] = True
    logger.info(f"Stop request received for stream_id: {stream_id}")
    
    return {"message": f"Stop request for stream {stream_id} received", "status": "success"}