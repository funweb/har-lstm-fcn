import time
import urllib, json, os, ipykernel, ntpath
from notebook import notebookapp as app


def lab_or_notebook():
    length = len(list(app.list_running_servers()))
    if length:
        return "notebook"
    else:
        return "lab"


def ipy_nb_name(token_lists):
    """ Returns the short name of the notebook w/o .ipynb
        or get a FileNotFoundError exception if it cannot be determined
        NOTE: works only when the security is token-based or there is also no password
    """

    if lab_or_notebook() == "lab":
        from jupyter_server import serverapp as app
    else:
        from notebook import notebookapp as app
    #         from jupyter_server import serverapp as app

    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    #     from notebook import notebookapp as app
    for srv in app.list_running_servers():
        for token in token_lists:
            srv['token'] = token

            try:
                # print(token)
                if srv['token'] == '' and not srv['password']:  # No token and no password, ahem...
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions')
                    print('no token or password')
                else:
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions?token=' + srv['token'])
            except:
                pass
                # print("Token is error")

        sessions = json.load(req)

        for sess in sessions:
            if sess['kernel']['id'] == kernel_id:
                nb_path = sess['notebook']['path']
                return ntpath.basename(nb_path).replace('.ipynb', '')  # handles any OS

    raise FileNotFoundError("Can't identify the notebook name, Please check [token]")



import numpy as np

def getAvailableId(type="min"):
    """
    返回可用的 GPU ID
    Args:
        type: sequence, min,

    Returns:

    """
    import pynvml

    pynvml.nvmlInit()

    time.sleep(5)  # 等待 5s 使得其他程序就绪

    deviceCount = pynvml.nvmlDeviceGetCount()
    current_gpu_unit_use = []
    for id in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if use.memory < 80:  # 首先保证有可用内存, 然后选择运行着比较小计算量的GPU
            # if use.gpu < 90:
            current_gpu_unit_use.append(use.gpu)
        else:
            current_gpu_unit_use.append(100)

    pynvml.nvmlShutdown()

    if current_gpu_unit_use == []:
        GPU_NUM = str(-1)
    else:
        GPU_NUM = str(np.argmin(current_gpu_unit_use))
    print("GPU used: {}, final choose: {}".format(current_gpu_unit_use, GPU_NUM))
    return GPU_NUM