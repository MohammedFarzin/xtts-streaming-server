# --- Imports ---
import json
import base64
import numpy as np
import torch
import asyncio
import multiprocessing as mp # Use 'mp' alias for clarity
from queue import Empty as QueueEmpty # To catch queue timeout
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.utils.generic_utils import get_user_data_dir
import os
import threading
import logging
import time
import audioop

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Determine based on GPU memory and model size
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 12)) # Default to 2 workers
MAX_QUEUE_SIZE = NUM_WORKERS * 4 # Limit backlog
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# Use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_SAMPLE_RATE = 24000 # Default XTTSv2 sample rate
ENCODE_SAMPLE_RATE = 8000 # Sample rate for the output chunks
ENCODE_SAMPLE_WIDTH = 2 # Bytes per sample for output

logger.info(f"Using device: {DEVICE}")
logger.info(f"Number of workers: {NUM_WORKERS}")

# --- Queues and Shared Resources ---
# Queue for sending tasks (stream_id, inputs) TO workers
task_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
# Queue for receiving results (stream_id, chunk/None/error) FROM workers
result_queue = mp.Queue()
# Dictionary to hold active WebSocket connections, keyed by stream_id
connections: Dict[str, WebSocket] = {}
# Async queue to bridge results into the main FastAPI event loop
async_result_queue: asyncio.Queue = asyncio.Queue()
# Event to signal shutdown to workers and reader thread
stop_event = mp.Event()


# --- Model Loading ---
# Function to load the model - will be called ONCE per worker process
def load_model_worker():
    logger.info(f"[Worker {os.getpid()}] Loading model...")
    try:
        torch.set_num_threads(1) # Often helps performance in multiprocessing
        model_path = os.path.join(get_user_data_dir("tts"), MODEL_NAME.replace("/", "--"))
        if not os.path.exists(model_path):
            logger.info(f"[Worker {os.getpid()}] Model path not found, downloading...")
            mm = ModelManager()
            mm.download_model(MODEL_NAME)
            logger.info(f"[Worker {os.getpid()}] Model downloaded.")

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        model.to(DEVICE)
        logger.info(f"[Worker {os.getpid()}] Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"[Worker {os.getpid()}] Failed to load model: {e}", exc_info=True)
        return None # Indicate failure

# --- Audio Processing ---
# Synchronous post-processing, runs in the main process after getting results
def postprocess_chunk(chunk: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if chunk is None:
        return None
    if not isinstance(chunk, torch.Tensor):
         logger.error(f"Received non-Tensor chunk: {type(chunk)}")
         # Handle potential error objects passed as chunks
         return None # Or raise an error, depending on desired behavior

    """Post process the output waveform"""
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk[None, : int(chunk.shape[0])]
    chunk = np.clip(chunk, -1, 1)
    chunk = (chunk * 32767).astype(np.int16)
    return chunk

# Synchronous encoding, runs in the main process
def encode_audio_chunk(frame_input, encode_base64=True, sample_rate=8000, sample_width=1, channels=1, original_sample_rate=24000) -> Optional[str]:
    if frame_input is None:
        return None
    
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

    

# --- Worker Process Target Function ---
def worker_main(task_q: mp.Queue, result_q: mp.Queue, stop_evt: mp.Event):
    pid = os.getpid()
    logger.info(f"[Worker {pid}] Starting.")
    model = load_model_worker()
    if model is None:
        logger.error(f"[Worker {pid}] Model loading failed. Exiting.")
        return # Worker cannot function without a model

    while not stop_evt.is_set():
        try:
            # Wait for a task, timeout allows checking stop_event periodically
            task_data = task_q.get(timeout=1.0)
            if task_data is None: # Sentinel value for shutdown
                break

            with open("./default_speaker.json", "r") as file:
                speaker = json.load(file)

            stream_id, text= task_data
            lang, spkr_emb, gpt_lat, strm_chunk_sz = "en", speaker["speaker_embedding"], speaker["gpt_cond_latent"], "20"
            logger.info(f"[Worker {pid}] Received task for stream_id: {stream_id}")

            # Prepare inputs for the model
            speaker_embedding = torch.tensor(spkr_emb, device=DEVICE).unsqueeze(0).unsqueeze(-1)
            # Ensure gpt_cond_latent is correctly shaped [1, 1024, N]
            gpt_cond_latent_tensor = torch.tensor(gpt_lat, device=DEVICE).reshape((-1, 1024)).unsqueeze(0)


            # Run inference stream
            chunks_generator = model.inference_stream(
                text,
                lang,
                gpt_cond_latent_tensor,
                speaker_embedding,
                stream_chunk_size=strm_chunk_sz,
                enable_text_splitting=True,
            )

            # Send chunks back to the main process
            chunk_count = 0
            for chunk in chunks_generator:
                # Put the raw tensor chunk onto the queue
                # The main process will handle postprocessing/encoding
                result_q.put((stream_id, chunk))
                chunk_count += 1
                if stop_evt.is_set(): # Check if shutdown was requested during generation
                     logger.warning(f"[Worker {pid}] Stop event set during generation for {stream_id}. Breaking.")
                     break # Stop sending chunks for this task


            if not stop_evt.is_set(): # Only send completion if not stopped prematurely
                result_q.put((stream_id, None)) # Signal end of stream for this task
                logger.info(f"[Worker {pid}] Task completed for stream_id: {stream_id}, chunks: {chunk_count}")


        except QueueEmpty:
            # Timeout occurred, loop back to check stop_event
            continue
        except Exception as e:
            logger.error(f"[Worker {pid}] Error processing task for {stream_id}: {e}", exc_info=True)
            try:
                # Send error message back to main process
                result_q.put((stream_id, f"Worker Error: {e}"))
            except Exception as qe:
                 logger.error(f"[Worker {pid}] CRITICAL: Failed to put error message on result queue: {qe}")
        finally:
             # Clean up GPU memory if needed after each task
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()


    logger.info(f"[Worker {pid}] Exiting.")


# --- Queue Reader Thread (Main Process) ---
# Bridges the mp.Queue (results from workers) to asyncio.Queue (for FastAPI)
def queue_reader_thread(res_q: mp.Queue, async_q: asyncio.Queue, stop_evt: mp.Event):
    logger.info("[ReaderThread] Starting.")
    while not stop_evt.is_set():
        try:
            # Get result from worker processes, timeout allows checking stop_event
            result_data = res_q.get(timeout=1.0)
            if result_data is None: # Should not happen with current worker logic, but good practice
                 continue
            # Put the result onto the asyncio queue for the main event loop
            asyncio.run_coroutine_threadsafe(async_q.put(result_data), asyncio.get_event_loop())
        except QueueEmpty:
            continue # Just means no results arrived in the timeout period
        except Exception as e:
            logger.error(f"[ReaderThread] Error reading from result queue: {e}", exc_info=True)
            # Consider how to handle this - maybe signal an error state?

    logger.info("[ReaderThread] Exiting.")
    # Signal end to the asyncio consumer
    asyncio.run_coroutine_threadsafe(async_q.put(None), asyncio.get_event_loop())


# --- FastAPI Application ---
app = FastAPI(title="XTTS Multiprocessing Streaming Server")

# --- Pydantic Input Model ---
class StreamingInputs(BaseModel):
    stream_id: str = Field(..., description="Unique identifier for this streaming request.")
    text: str = Field(..., description="Text to be synthesized.")

# --- Background Task to Process Results ---
async def result_processor(async_q: asyncio.Queue):
    logger.info("[ResultProcessor] Starting.")
    active_streams = set() # Keep track of streams we expect data for

    while True:
        result_data = await async_q.get()
        if result_data is None:
            logger.info("[ResultProcessor] Received stop signal. Exiting.")
            break # End of processing

        stream_id, chunk_data = result_data

        if stream_id not in connections:
            logger.warning(f"[ResultProcessor] Received chunk for unknown or disconnected stream_id: {stream_id}")
            # If it was an error message, maybe log it specially
            if isinstance(chunk_data, str):
                 logger.error(f"[ResultProcessor] Orphaned error from worker for {stream_id}: {chunk_data}")
            continue

        websocket = connections[stream_id]

        try:
            if isinstance(chunk_data, str): # Error message from worker
                logger.error(f"[ResultProcessor] Sending error to client {stream_id}: {chunk_data}")
                await websocket.send_json({"status": "error", "message": chunk_data})
                # Remove connection after error? Depends on desired behavior.
                # del connections[stream_id]
                # active_streams.discard(stream_id)
            elif chunk_data is None: # End of stream signal from worker
                logger.info(f"[ResultProcessor] Sending completed signal for stream_id: {stream_id}")
                await websocket.send_json({"status": "completed"})
                # Clean up connection AFTER sending completion
                # del connections[stream_id] # Let disconnect handler do this
                active_streams.discard(stream_id)
            else: # Actual audio chunk (Tensor)
                 logger.debug(f"[ResultProcessor] Processing chunk for stream_id: {stream_id}")
                 # Run synchronous postprocessing/encoding in thread pool to avoid blocking event loop
                 processed_wav = await asyncio.to_thread(postprocess_chunk, chunk_data)
                 encoded_audio = await asyncio.to_thread(encode_audio_chunk, processed_wav, ENCODE_SAMPLE_RATE, ENCODE_SAMPLE_WIDTH)

                 if encoded_audio:
                     await websocket.send_json({"chunk": encoded_audio})
                 else:
                      logger.warning(f"[ResultProcessor] Skipping empty chunk for {stream_id} after processing.")

        except WebSocketDisconnect:
             logger.info(f"[ResultProcessor] Client {stream_id} disconnected while processing results.")
             # Connection will be removed by the main endpoint's finally block
             active_streams.discard(stream_id)
        except Exception as e:
            logger.error(f"[ResultProcessor] Error sending data to client {stream_id}: {e}", exc_info=True)
            # Attempt to send error to client if possible
            try:
                await websocket.send_json({"status": "error", "message": f"Server processing error: {e}"})
            except Exception:
                pass # Ignore if sending error also fails
            # Clean up?
            # del connections[stream_id]
            active_streams.discard(stream_id)

        finally:
             # Mark task done for the asyncio queue
             async_q.task_done()


# --- WebSocket Endpoint ---
@app.websocket("/tts_stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted: {websocket.client}")
    stream_id = None # Initialize stream_id

    try:
        # Expect initial message containing setup data
        initial_data = await websocket.receive_text()
        input_payload = StreamingInputs(**json.loads(initial_data))
        stream_id = input_payload.stream_id

        logger.info(f"Received initial request for stream_id: {stream_id}")

        if stream_id not in connections:
            connections[stream_id] = websocket


        # Prepare task data for the worker queue
        task_data = (
            stream_id,
            input_payload.text,
        )

        # Put the task onto the queue for a worker to pick up
        try:
             task_queue.put(task_data, block=True, timeout=5.0) # Block with timeout
             logger.info(f"Task queued for stream_id: {stream_id}")
        except Exception as e: # Specifically catch Full queue exception if not blocking
             logger.error(f"Failed to queue task for {stream_id}: {e}", exc_info=True)
             await websocket.send_json({"status": "error", "message": "Server busy, try again later."})
             # No need to close here, finally block handles cleanup
             return # Exit the handler for this connection


        # Keep the connection open to receive potential commands (e.g., stop)
        # Or just wait for completion/disconnection
        # This loop currently just waits for disconnect.
        while True:
             # You could await websocket.receive_text() here if you expect client commands
             # For now, just yielding control allows the result_processor to send chunks
             await asyncio.sleep(1) # Keep connection alive, check status implicitly

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {stream_id} ({websocket.client})")
    except json.JSONDecodeError:
         logger.error("Failed to parse initial JSON payload.")
         await websocket.close(code=1003, reason="Invalid JSON format")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {stream_id}: {e}", exc_info=True)
        # Try to send error before closing, but might fail if connection is already broken
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        # Cleanup: Remove connection reference
        if stream_id and stream_id in connections:
            del connections[stream_id]
            logger.info(f"Cleaned up connection for stream_id: {stream_id}")
        # Note: We don't stop workers or queues here, they are managed globally


# --- Global Worker Pool and Thread Management ---
worker_pool: List[mp.Process] = []
queue_reader: Optional[threading.Thread] = None
result_processor_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def startup_event():
    global worker_pool, queue_reader, result_processor_task, stop_event
    logger.info("Starting up...")

    # Reset stop event (important if server restarts)
    stop_event.clear()

    # Start worker processes
    logger.info(f"Starting {NUM_WORKERS} worker processes...")
    for i in range(NUM_WORKERS):
        process = mp.Process(
            target=worker_main,
            args=(task_queue, result_queue, stop_event),
            daemon=True # Make workers daemons so they exit if main process dies unexpectedly
        )
        worker_pool.append(process)
        process.start()
        logger.info(f"Worker process {process.pid} started.")

    # Start the queue reader thread
    logger.info("Starting queue reader thread...")
    queue_reader = threading.Thread(
        target=queue_reader_thread,
        args=(result_queue, async_result_queue, stop_event),
        daemon=True
    )
    queue_reader.start()

    # Start the background task in asyncio to process results
    logger.info("Starting asyncio result processor task...")
    result_processor_task = asyncio.create_task(result_processor(async_result_queue))

    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    global worker_pool, queue_reader, result_processor_task, stop_event
    logger.info("Shutting down...")

    # 1. Signal workers and reader thread to stop
    stop_event.set()
    logger.info("Stop event set.")

    # 2. Send sentinel values to task queue to potentially unblock workers waiting on get()
    # Do this carefully if the queue might be full
    logger.info("Attempting to unblock workers...")
    for _ in range(NUM_WORKERS):
         try:
             task_queue.put_nowait(None) # Use nowait to avoid blocking shutdown
         except Exception:
             logger.warning("Task queue full or closed while sending sentinel values.")
             break # Stop trying if queue is problematic


    # 3. Wait for worker processes to finish
    logger.info("Joining worker processes...")
    for i, process in enumerate(worker_pool):
        try:
             process.join(timeout=10.0) # Add timeout
             if process.is_alive():
                 logger.warning(f"Worker {process.pid} did not exit cleanly after 10s. Terminating.")
                 process.terminate() # Force terminate if needed
                 process.join() # Wait for termination
             else:
                  logger.info(f"Worker {process.pid} joined successfully.")
        except Exception as e:
            logger.error(f"Error joining worker {process.pid}: {e}")


    # 4. Wait for the queue reader thread to finish
    logger.info("Joining queue reader thread...")
    if queue_reader and queue_reader.is_alive():
        queue_reader.join(timeout=5.0)
        if queue_reader.is_alive():
             logger.warning("Queue reader thread did not exit cleanly after 5s.")


    # 5. Cancel and wait for the asyncio result processor task
    logger.info("Stopping result processor task...")
    if result_processor_task and not result_processor_task.done():
        result_processor_task.cancel()
        try:
            await result_processor_task
        except asyncio.CancelledError:
            logger.info("Result processor task cancelled successfully.")
        except Exception as e:
            logger.error(f"Error during result processor task shutdown: {e}")


    # 6. Close queues (optional, helps ensure resources are released)
    logger.info("Closing queues...")
    try:
        task_queue.close()
        result_queue.close()
        # Queues might already be joined by workers/reader exiting, ignore errors
        task_queue.join_thread()
        result_queue.join_thread()
    except Exception as e:
        logger.warning(f"Error closing queues: {e}")


    logger.info("Shutdown complete.")

# --- Main Execution ---
if __name__ == "__main__":
    # Required for multiprocessing to work correctly on some platforms (Windows, macOS)
    # Needs to be in the main block.
    mp.set_start_method("spawn", force=True) # 'spawn' is generally safer

    import uvicorn
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)