from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    get_transforms,
    models,
)
import torch
from pathlib import Path
from io import BytesIO

import sys
import uvicorn
import aiohttp
import asyncio



from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    get_transforms,
    models,
)
import torch
from pathlib import Path
from io import BytesIO
import sys


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()

dl_pan_images_path = Path("/tmp")
dl_pan_fnames = [
    "/{}_1.jpg".format(c)
    for c in [   
        "Pan Card",     
        "Driving Licence"
        
    ]
]

dl_pan_data = ImageDataBunch.from_name_re(
    dl_pan_images_path,
    dl_pan_fnames,
    r"/([^/]+)_\d+.jpg$",
    ds_tfms=get_transforms(),
    size=224,
)
dl_pan_learner = create_cnn(dl_pan_data, models.resnet34)
dl_pan_learner.model.load_state_dict(
    torch.load("dl_or_pan.pth", map_location="cpu")
)



@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

def classifyImage():
    with open("pan_4.jpg",'rb') as f:
        bytes = f.read()
    results = predict_image_from_bytes(bytes)
    return results

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    losses = img.predict(dl_pan_learner)
    print(losses)
    return sorted(zip(dl_pan_learner.data.classes, map(float, losses)))

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
