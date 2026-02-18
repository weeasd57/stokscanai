import threading
import time
from collections import deque
from typing import List

class MockBot:
    def __init__(self):
        self._logs = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def _log(self, msg: str):
        with self._lock:
            self._logs.append(msg)
    
    def get_safe_logs(self, limit: int = 1000) -> List[str]:
        with self._lock:
            all_logs = list(self._logs)
            return all_logs[-limit:] if len(all_logs) > limit else all_logs

def writer_thread(bot: MockBot, stop_event: threading.Event):
    i = 0
    while not stop_event.is_set():
        bot._log(f"Log entry {i}")
        i += 1
        # No sleep to stress test

def reader_thread(bot: MockBot, stop_event: threading.Event, results: list):
    try:
        while not stop_event.is_set():
            logs = bot.get_safe_logs(100)
            # Just iterating to simulate work
            for l in logs:
                pass
    except RuntimeError as e:
        results.append(str(e))
    except Exception as e:
        results.append(f"Unexpected error: {e}")

def run_test():
    bot = MockBot()
    stop_event = threading.Event()
    results = []
    
    writers = [threading.Thread(target=writer_thread, args=(bot, stop_event)) for _ in range(2)]
    readers = [threading.Thread(target=reader_thread, args=(bot, stop_event, results)) for _ in range(5)]
    
    for t in writers + readers:
        t.start()
    
    print("Testing thread safety for 5 seconds...")
    time.sleep(5)
    stop_event.set()
    
    for t in writers + readers:
        t.join()
    
    if results:
        print(f"FAILED: Found {len(results)} errors:")
        for r in results:
            print(f" - {r}")
    else:
        print("SUCCESS: No thread safety issues detected.")

if __name__ == "__main__":
    run_test()
