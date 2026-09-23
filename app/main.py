from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.inference import CaptioningService

app = FastAPI(
    title="Image Captioning API",
    description="API for generating automatic image captions using CNN (ResNet-50) + LSTM.",
    version="1.0.0",
)

captioning_service = CaptioningService()


class CaptionResponse(BaseModel):
    """Pydantic model for the image caption response payload."""

    caption: str


@app.get("/")
def health_check():
    """Health check endpoint to verify that the service is up and running.

    Returns:
        dict: Operational status message.
    """
    return {"status": "ok"}


@app.post("/caption", response_model=CaptionResponse)
async def caption_endpoint(file: UploadFile = File(...)):
    """FastAPI route to accept an uploaded image file and return its generated caption.

    Args:
        file (UploadFile): The uploaded image file from the client.

    Returns:
        CaptionResponse: JSON response object containing the generated caption string.

    Raises:
        HTTPException: 400 Bad Request if the uploaded image file is invalid or corrupted.
    """
    image_bytes = await file.read()
    try:
        caption = captioning_service.caption_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CaptionResponse(caption=caption)