#!/usr/bin/env python3
import logging
import os
import portalocker
import time
logfnh = "controller" + ".log"
logfn = os.path.join("logs", logfnh)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format=f"[Controller] %(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfn),
        logging.StreamHandler()])

class StateController():
    def __init__(self, lock_fn="lock.lock", acquire_check_interval=0.1):
        self.lock_fn = lock_fn
        self.state = 'unlocked'
        self.acquire_check_interval = acquire_check_interval

        if os.path.exists(lock_fn):
            os.remove(lock_fn)
        os.mknod(lock_fn)
        with open(lock_fn, 'w') as fh:
            portalocker.lock(fh, portalocker.LOCK_EX)
            fh.writelines([self.state])
            fh.flush()
            portalocker.unlock(fh)

    def _state(self):
        _ok = False
        _count = 0
        _count_max = 5
        while not _ok and _count < _count_max:
            try:
                with open(self.lock_fn, 'r') as fh:
                    portalocker.lock(fh, portalocker.LOCK_EX)

                    out = fh.readlines()
                    out = out[0]
                    if out == 'locked':
                        state = 'locked'
                    else:
                        state = 'unlocked'
                    portalocker.unlock(fh)

            except IndexError as e:
                logging.debug(f"Index error due to some bug in file access. Try {_count+1}/{_count_max}.")
                _count += 1
                time.sleep(1)
                continue
            _ok = True

        return state

    def acquire(self):
        self.state = self._state()
        logging.debug(f"State={self.state}")
        while self.state == 'locked':
            self.state = self._state()
            time.sleep(self.acquire_check_interval)
        with open(self.lock_fn, 'w') as fh:
            portalocker.lock(fh, portalocker.LOCK_EX)
            fh.writelines(['locked'])
            fh.flush()
            self.state = 'locked'
            portalocker.unlock(fh)

    def release(self):
        with open(self.lock_fn, 'w') as fh:
            portalocker.lock(fh, portalocker.LOCK_EX)
            fh.writelines(['unlocked'])
            fh.flush()
            portalocker.unlock(fh)
