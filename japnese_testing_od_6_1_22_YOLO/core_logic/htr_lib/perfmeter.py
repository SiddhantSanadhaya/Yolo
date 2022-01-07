'''
API Performance Test Code :
  
Create file perfmeter.py write two function startOp and endOp . 
startOp will start timer and log it using logging service . 
endOp will stop that timer and put total time in log by calculating difference. 
So basically we need mechanism to calculate API performance. And log it properly.

'''
 

import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def startOp(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def endOp(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return (f"Elapsed time: {elapsed_time:0.4f} seconds")



