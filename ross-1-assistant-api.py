# pip install openai
# export OPENAI_API_KEY="sk-..."   # set your key before running

from openai import OpenAI
import os
from pathlib import Path

# ---------- CONFIG ----------
ASSISTANT_ID = "asst_yHWUBHquplDupejty9UBxt5f"      # <-- put your Assistant ID here
DOWNLOAD_DIR = Path("./assistant_outputs")
# ----------------------------

client = OpenAI()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def print_message_content(msg):
    """Print all parts of an assistant message and return any file IDs to download."""
    file_ids = []
    for part in msg.content:
        t = part.type
        if t == "text":
            print("\n--- TEXT ---\n")
            print(part.text.value)
        elif t == "image_file":
            print("\n--- IMAGE (file id) ---\n", part.image_file.file_id)
            file_ids.append(part.image_file.file_id)
        elif t == "output_file":
            print("\n--- OUTPUT FILE (file id) ---\n", part.output_file.file_id)
            file_ids.append(part.output_file.file_id)
        elif t == "file_path":
            print("\n--- FILE PATH (sandbox) ---\n", part.file_path.path)
        else:
            print(f"\n--- {t.upper()} (unhandled) ---\n", part)
    return file_ids

def download_file(file_id, out_dir: Path):
    """Download a file by file_id via Files API."""
    fmeta = client.files.retrieve(file_id)
    fname = fmeta.filename or f"{file_id}.bin"
    out_path = out_dir / fname
    content = client.files.content(file_id)
    with open(out_path, "wb") as f:
        f.write(content.read())
    print(f"Saved file: {out_path}")

def run_assistant_and_get_outputs(assistant_id: str, user_question: str):
    # 1) Create a new thread
    thread = client.beta.threads.create()

    # 2) Add the user's message
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_question
    )

    # 3) Create a run and wait until it's finished
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    if run.status != "completed":
        print("Run ended with status:", run.status)
        return

    # 4) Fetch the latest messages (newest first) and print assistant outputs
    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=3)

    all_file_ids = []
    for m in msgs.data:
        if m.role == "assistant":
            file_ids = print_message_content(m)
            all_file_ids.extend(file_ids)

    # 5) (Optional) Download any files produced by the assistant
    for fid in set(all_file_ids):
        try:
            download_file(fid, DOWNLOAD_DIR)
        except Exception as e:
            print(f"Could not download file {fid}: {e}")

if __name__ == "__main__":
    user_question = input("Enter your question for the assistant: ")
    run_assistant_and_get_outputs(ASSISTANT_ID, user_question)
