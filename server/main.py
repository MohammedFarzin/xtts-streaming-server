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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile
from fastapi.responses import StreamingResponse

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

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
connections: dict[str, WebSocket] = {}

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
# @app.post("/clone_speaker")
# def predict_speaker(wav_file: UploadFile):
#     """Compute conditioning inputs from reference audio file."""
#     temp_audio_name = next(tempfile._get_candidate_names())
#     with open(temp_audio_name, "wb") as temp, torch.inference_mode():
#         temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
#         gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
#             temp_audio_name
#         )
#     return {
#         "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
#         "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
#     }


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

    logger.info(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {np.min(audio_data)}, max: {np.max(audio_data)}")
    

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


def predict_streaming_generator(parsed_input):
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language

    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header


    for i,chunks in enumerate(model.inference_stream(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=stream_chunk_size,
        enable_text_splitting=True
    )):
        chunk = postprocess(chunks)
        processed_chunk = encode_audio_common(
            chunk, 
            encode_base64=False, 
            sample_rate=8000, 
            sample_width=2
        )
        logger.infStreamSido(f"Processed chunk size:{i}// {len(processed_chunk)}")
        yield processed_chunk
class WebSocketConnectionManagement:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket.StreamSid] = websocket

    async def disconnect(self, websocket: WebSocket):
        del self.active_connections[websocket.StreamSid]

    async def send_message(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

@app.websocket("/tts_stream")
async def predict_streaming_endpoint(parsed_input: WebSocket):
    await parsed_input.accept()
    print("WebSocket connection established")
    connections[parsed_input.StreamSid] = parsed_input

    try:

        text = await parsed_input.receive_text()
        logger.info(f"Received text: {text}")
        input_data = StreamingInputs(text=text)

        async 
        for chunk in predict_streaming_generator(input_data):
            logger.info("Sending chunk")
            await parsed_input.send_json({"status": "processing", "chunk": chunk})
        
        # Send a completion message
        logger.info("Sending completion message")
        await parsed_input.send_json({"status": "completed"})
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.info(f"Error: {e}")
        try:
            await parsed_input.send_json({"error": str(e)})
        except Exception as e:
            logger.info(f"Error sending error message: {e}")


class TTSInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str

@app.post("/tts")
def predict_speech(parsed_input: TTSInputs):
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language

    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
    )

    wav = postprocess(torch.tensor(out["wav"]))

    return encode_audio_common(wav.tobytes())


@app.get("/studio_speakers")
def get_speakers():
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        return {
            speaker: {
                "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
            for speaker in model.speaker_manager.speakers.keys()
        }
    else:
        return {}
        
@app.get("/languages")
def get_languages():
    return config.languages