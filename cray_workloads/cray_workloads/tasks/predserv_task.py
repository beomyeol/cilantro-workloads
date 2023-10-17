"""
    Model serving task.
    -- kirthevasank
"""

import io
import pickle
import time
import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings(action='ignore', category=UserWarning)

def _serve_model(model, test_data):
    '''
    Task that serves a model on a data point.
    '''
    try:
        results = model.predict(test_data)
    except Exception as exc:
        raise exc
#     results = model.predict(test_data)
    return results

_MODEL = torchvision.models.resnet34(pretrained=False).eval()
_PREPROCESSOR = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # remove alpha channel
        transforms.Lambda(lambda t: t[:3, ...]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])


def prediction_serving_task(model, img_bytes, sleep_time):
    pil_img = Image.open(io.BytesIO(img_bytes))
    input_tensor = _PREPROCESSOR(pil_img).unsqueeze(0)
    with torch.no_grad():
        output_tensor = _MODEL(input_tensor)
    time.sleep(sleep_time)
    return int(torch.argmax(output_tensor[0]))

def prediction_serving_task_old(model, test_data, sleep_time):
    '''
    Task that serves a model on a data point.
    '''
    if isinstance(model, str):
        with open(model, 'rb') as model_file:
            new_model = pickle.load(model_file)
            model_file.close()
        model = new_model
#         print('Loaded model %s.'%(model))
#     print('Received %d data'%(len(test_data)))
#     start_time = time.time()
    all_results = []
    for elem in test_data:
        curr_result = _serve_model(model, [elem])
        all_results.append(curr_result)
    time.sleep(sleep_time)
#     end_time = time.time()
#     print('Time taken = %0.4f, sleep for %0.3f'%(end_time - start_time, sleep_time))
    return True

