#!/usr/bin/env python3
import logging
import os
import portalocker
import queue
import subprocess
import threading
import time
import pprint

from src.parallel import StateController

logfnh = "controller" + ".log"
logfn = os.path.join("logs", logfnh)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format=f"[Controller] %(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfn),
        logging.StreamHandler()])

state_controller = StateController()
lock = threading.Lock()
lock_fn = "lock.lock"

def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def run_process(command, stage_identifier):
    # dvc init file lock

    with lock:
        state_controller.acquire()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()

    stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))

    stdout_thread.start()
    stderr_thread.start()

    info_stage_unchanged = f"Stage \'{stage_identifier}\' didn\'t change, skipping"
    info_stage_cached = f"\'{stage_identifier}\' is cached - skipping run, checking out outputs"
    info_stage_out_empty = f"is empty."
    while stdout_thread.is_alive() or stderr_thread.is_alive():
        try:
            stdout_line = stdout_queue.get_nowait()
        except queue.Empty:
            stdout_line = None

        try:
            stderr_line = stderr_queue.get_nowait()
        except queue.Empty:
            stderr_line = None

        if stdout_line:
            res = stdout_line.replace("\n", "")
            if any([skip_info in res for skip_info in [info_stage_cached,
                                                       info_stage_unchanged,
                                                       info_stage_out_empty]]):
                state_controller.release()
            logging.debug(res)
        if stderr_line:
            res = stderr_line.replace("\n", "")
            if any([skip_info in res for skip_info in [info_stage_cached,
                                                       info_stage_unchanged,
                                                       info_stage_out_empty]]):
                state_controller.release()
            logging.debug(res)

    stdout_thread.join()
    stderr_thread.join()
    process.wait()
    print(f"WAITING DONE for command: {command}")

commands = []

import yaml
# read params and select all
with open("params.yaml") as fh:
    params = yaml.safe_load(fh)

# prods_strpart = ["switch_8_port"]
# prods_strpart = ["2700642"]
prods_strpart = [# "bus_coppler"
                 # ,
    "2700642",
                 # "switch_8_port", "switch_16_port"
]
includes_bool = lambda x: any([_str in x for _str in prods_strpart])

# commands.append([f"dvc repro data/images/"])
intermediate_barrier_stages = 16
cntr = 0
for i, dataset in enumerate(params['dataset'].keys()):
    if includes_bool(dataset):
        commands.append([f"dvc repro -s target-pose@{dataset}"])
        cntr += 1
    if cntr % intermediate_barrier_stages == 0 and cntr != 0:
        commands.append(["barrier"])

commands.append(["barrier"])
cntr = 0
for i, dataset in enumerate(params['dataset'].keys()):
    if includes_bool(dataset):
        commands.append([f"dvc repro -s render-synthetic@{dataset}"])
        cntr += 1
    if cntr % intermediate_barrier_stages == 0 and cntr != 0:
        commands.append(["barrier"])


commands.append(["barrier"])
cntr = 0

for i, dataset in enumerate(params['dataset'].keys()):
    if includes_bool(dataset):
        commands.append([f"dvc repro -s defect-mask@{dataset}"])
        commands.append([f"dvc repro -s defect-mask-texture@{dataset}"])
        cntr += 1
    if cntr % intermediate_barrier_stages == 0 and cntr != 0:
        commands.append(["barrier"])

commands.append(["barrier"])
cntr = 0
for dataset in params['dataset'].keys():
    if includes_bool(dataset):
        commands.append([f"dvc repro -s format_dataset_regular@{dataset} --force"])
        cntr += 1

logging.info(f"The following commands will be run:\n {pprint.pformat(commands)}")

threads = []
_commands = []
for i, command in enumerate(commands):
    logging.debug(f"Next command is: {command}")
    time.sleep(1)
    if command != ["barrier"]:
        thread = threading.Thread(target=run_process, args=(command, command[0].split(' ')[-1]))
        threads.append(thread)
        _commands.append(command)
        thread.start()

    if command == ["barrier"] or i == len(commands) - 1:
        for thread, _command in zip(threads, _commands):
            logging.debug(f"Join process of command \"{_command}\".")
            thread.join()
            logging.debug(f"Joining process done of command \"{_command}\".")
        threads = []
        _commands = []
