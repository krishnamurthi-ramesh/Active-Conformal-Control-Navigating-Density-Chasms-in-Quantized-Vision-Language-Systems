"""
ACC Streaming Agent — GPU Student with Nervous System Bridge

Communicates with the Oracle Campaign Manager via shared-memory ring buffer.
Sends hidden-state vectors at each generation step and responds to intervention
signals.

Multi-model support: Accepts --model flag for {llama3, mistral, phi3}.
"""
import argparse
import sys
import time
import json
import os
import torch
import numpy as np

# Flexible path resolution (container + local dev)
SRC_ROOT = "/app/src"
LOCAL_SRC = os.path.join(os.path.dirname(__file__), "..")
for p in [SRC_ROOT, LOCAL_SRC]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

from acc_core.system.ring_buffer import ACCOpsBridge
from wrappers.setup_student import setup_model


def _resolve_task_file():
    candidates = [
        os.environ.get("ACC_TASK_FILE"),
        "/app/data/benchmark_tasks.json",
        os.path.join(os.path.dirname(__file__), "../../01_DATA/benchmark_tasks.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return "/app/data/benchmark_tasks.json"


TASK_FILE = _resolve_task_file()


class ACCStreamingAgent:
    def __init__(self, model_name: str = "mistral"):
        print(" Connecting to Nervous System...")
        self.bridge = ACCOpsBridge(create=False)
        self.tokenizer, self.model = setup_model(model_name=model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_suite(self):
        if not os.path.exists(TASK_FILE):
            print(f" Task file not found: {TASK_FILE}")
            return
        with open(TASK_FILE, 'r') as f:
            tasks = json.load(f)

        print(f" STARTING BENCHMARK SUITE ({len(tasks)} Tasks)")

        for i, task in enumerate(tasks):
            print(f"\n========================================")
            print(f" TASK {i+1}/{len(tasks)}: {task['id']}")
            print(f"========================================")

            self.generate(task['prompt'], max_tokens=task['max_tokens'])

            # Cool-down to let Oracle reset
            time.sleep(3)

    def generate(self, prompt, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        current_ids = input_ids

        step_counter = 0

        for _ in range(max_tokens):
            step_counter += 1

            with torch.no_grad():
                outputs = self.model(current_ids, output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :]

                # Send State
                z_t = outputs.hidden_states[-1][:, -1, :].float().cpu().numpy().flatten()
                self.bridge.write_state(step_counter, z_t[:128])

                # Check Intervention
                if self.bridge.check_for_intervention():
                    print(f"    PAUSE at Step {step_counter} (Syncing KV Cache...)")
                    time.sleep(0.5)

            # Decode
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            word = self.tokenizer.decode(next_token[0])
            print(word, end="", flush=True)

            if next_token.item() == self.tokenizer.eos_token_id:
                break
        print("\n Task Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACC Streaming GPU Agent")
    parser.add_argument("--model", choices=["llama3", "mistral", "phi3"],
                        default="mistral", help="Student model architecture")
    args = parser.parse_args()
    agent = ACCStreamingAgent(model_name=args.model)
    agent.run_suite()
