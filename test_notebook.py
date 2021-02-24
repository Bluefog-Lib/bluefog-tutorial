import subprocess
import time
import atexit

import papermill as pm

p_start = subprocess.Popen(
    'ibfrun start -np 4 --disable-heartbeat', shell=True)
time.sleep(10)


def cleanup_func():
    if not p_start.poll():
        print("terminate ibfrun start")
        p_start.terminate()
    p_stop = subprocess.Popen('ibfrun stop', shell=True)
    print("run ibfrun stop")
    p_stop.wait()


atexit.register(cleanup_func)


print("Start papermill")
pm.execute_notebook('hello.ipynb', "hello.out.ipynb")
print("End papermill")
p_start.terminate()