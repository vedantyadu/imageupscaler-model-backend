from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi import status
import numpy as np
import cv2
from upscale import upscale


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/upscale/{model}")
async def upscale_route(model: str, file: UploadFile):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        upscaled_image = upscale(img_cv, model)
        _, img_encoded = cv2.imencode('.png', upscaled_image)
        return Response(content=img_encoded.tobytes(), media_type='image/png')
    
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'code': status.HTTP_500_INTERNAL_SERVER_ERROR, 'message': 'Internal Server Error'}
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000)
