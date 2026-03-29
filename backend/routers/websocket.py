import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from core import decode_access_token, settings
from services import YOLOService

router = APIRouter()


@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket, token: str = Query(...)):
    payload = decode_access_token(token)
    if payload is None:
        await websocket.accept()
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()

    yolo = YOLOService.get_instance()
    conf = settings.default_conf
    iou = settings.default_iou
    frame_id = 0

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                break

            # Text message = config update
            text = message.get("text")
            if text is not None:
                try:
                    data = json.loads(text)
                    if data.get("type") == "config":
                        conf = float(data.get("conf", conf))
                        iou = float(data.get("iou", iou))
                        await websocket.send_json({"type": "config_ack", "conf": conf, "iou": iou})
                except (json.JSONDecodeError, ValueError):
                    pass
                continue

            # Binary message = JPEG frame
            jpeg_bytes = message.get("bytes")
            if jpeg_bytes:
                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                image = cv2.resize(image, (settings.image_size, settings.image_size))

                detections, inference_time = yolo.detect(image, conf=conf, iou=iou)

                frame_id += 1
                await websocket.send_json({
                    "detections": [d.model_dump() for d in detections],
                    "inference_time": round(inference_time, 4),
                    "frame_id": frame_id,
                })

    except (WebSocketDisconnect, RuntimeError):
        pass
