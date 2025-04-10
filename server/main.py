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
import time
import traceback

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



def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav



def encode_audio_common(
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
    stream_id: str

# Add this at the top of your file with other imports
from contextlib import contextmanager

@contextmanager
def timing_context(name):
    """Context manager to measure execution time of a code block"""
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    logger.info(f"TIMING: {name} took {elapsed:.4f} seconds")



def predict_streaming_generator(parsed_input):
    total_inference_time = 0
    total_postprocess_time = 0
    total_encode_time = 0
    chunk_count = 0
    
    with timing_context("TTS Generation (total)"):
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
        text = parsed_input.text
        language = parsed_input.language

        stream_chunk_size = int(parsed_input.stream_chunk_size)
        add_wav_header = parsed_input.add_wav_header
        
        logger.info(f"Starting TTS generation for text: '{text}' (length: {len(text)})")

        # Use the generator from model.inference_stream
        stream_generator = model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=stream_chunk_size,
            enable_text_splitting=True
        )

        for i, chunks in enumerate(stream_generator):
            chunk_count += 1
            
            # Measure postprocessing time
            with timing_context(f"Postprocessing chunk {i}"):
                start_time = time.time()
                chunk = postprocess(chunks)
                elapsed = time.time() - start_time
                total_postprocess_time += elapsed
            
            # Measure encoding time
            with timing_context(f"Encoding chunk {i}"):
                start_time = time.time()
                processed_chunk = encode_audio_common(
                    chunk, 
                    encode_base64=False, 
                    sample_rate=8000, 
                    sample_width=2
                )
                elapsed = time.time() - start_time
                total_encode_time += elapsed
            
            logger.info(f"Processed chunk size:{i}// {len(processed_chunk)}")
            yield processed_chunk
        
        # Log summary statistics
        if chunk_count > 0:
            logger.info(f"PERFORMANCE SUMMARY:")
            logger.info(f"- Total chunks processed: {chunk_count}")
            logger.info(f"- Average postprocessing time: {total_postprocess_time/chunk_count:.4f}s per chunk")
            logger.info(f"- Average encoding time: {total_encode_time/chunk_count:.4f}s per chunk")
            logger.info(f"- Total postprocessing time: {total_postprocess_time:.4f}s")
            logger.info(f"- Total encoding time: {total_encode_time:.4f}s")
            # Inference time is the remainder
            total_time = time.time() - start_time
            inference_time = total_time - total_postprocess_time - total_encode_time
            logger.info(f"- Estimated inference time: {inference_time:.4f}s ({inference_time/total_time*100:.2f}%)")

        


@app.websocket("/tts_stream")
async def predict_streaming_endpoint(parsed_input: WebSocket):
    with timing_context("WebSocket connection lifetime"):
        await parsed_input.accept()
        logger.info("WebSocket connection established")
        
        try:
            with timing_context("WebSocket data receive"):
                data_json = await parsed_input.receive_text()
                data_json = json.loads(data_json)

            if data_json["text"] is None:
                parsed_input.close()
                logger.info(f"Received empty text, closing connection for {data_json['stream_id']}")
                return

            # Extract stream_id
            if "stream_id" not in data_json:
                logger.info("Missing stream_id in request")
                await parsed_input.send_json({"error": "Missing stream_id"})
                return
            
            connections[data_json["stream_id"]] = parsed_input

            logger.info(f"WebSocket connection ID: {data_json['stream_id']}, {parsed_input}, {data_json}")
            input_data = StreamingInputs(
                text=data_json["text"],
                stream_id=data_json["stream_id"]
            )

            chunk_count = 0
            total_send_time = 0.0
            
            with timing_context("Full TTS processing and sending"):
                for chunk in predict_streaming_generator(input_data):
                    chunk_count += 1
                    with timing_context(f"Sending chunk {chunk_count}"):
                        start_time = time.time()
                        await connections[data_json["stream_id"]].send_json({"chunk": chunk})
                        elapsed = time.time() - start_time
                        total_send_time += elapsed
                        logger.info(f"Sent chunk {chunk_count}, size: {len(chunk)}")
            
            if chunk_count > 0:
                logger.info(f"Avg send time: {total_send_time/chunk_count:.4f}s per chunk")
                
            await connections[data_json["stream_id"]].send_json({"status": "completed"})
            logger.info("Sending completion message")
        except WebSocketDisconnect:
            logger.info(f"WebSocket connection closed and deleted: {data_json['stream_id']}")
            connections[data_json["stream_id"]].close()
            del connections[data_json["stream_id"]]

        except Exception as e:
            logger.info(f"Error: {e}")
            logger.error("Traceback:\n" + traceback.format_exc())  
            try:
                await connections[data_json["stream_id"]].send_json({"error": str(e)})
            except Exception as e:
                logger.info(f"Error sending error message: {e}")
                logger.error("Traceback:\n" + traceback.format_exc())  

@app.post("/stop_stream/{stream_id}")
async def stop_stream(stream_id: str):
    if stream_id not in connections:
        return {"message": "Stream not found or already completed", "status": "error"}
    
    # Set the stop flag for this stream
    connections[stream_id]["voice_stop"] = True
    logger.info(f"Stop request received for stream_id: {stream_id}")
    
    return {"message": f"Stop request for stream {stream_id} received", "status": "success"}