import dtlpy as dl
import logging
import os
import torch
from models.common import DetectMultiBackend
import numpy as np
from utils.general import (Profile, check_img_size, non_max_suppression, scale_coords)
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import urllib.request

logger = logging.getLogger('YOLOv7-Seg-Adapter')


@dl.Package.decorators.module(description='Model Adapter for Yolov7 Seg Model',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity=None):

        logger.info(f"TORCH VERSION - {torch.__version__}")
        logger.info(f"TORCH CUDA AVAILABLE - {torch.cuda.is_available()}")

        self.weights = None
        self.data = None
        self.imgsz = None
        self.conf_thres = None
        self.iou_thres = None
        self.max_det = 1000
        self.device = None
        super(ModelAdapter, self).__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename')
        checkpoint_url = self.configuration.get('checkpoint_url', None)

        if not os.path.isfile(os.path.join(local_path, model_filename)):
            if checkpoint_url is not None:
                os.makedirs(local_path, exist_ok=True)
                urllib.request.urlretrieve(checkpoint_url, os.path.join(local_path, 'yolov7-seg.pt'))
            else:
                raise Exception("checkpoints weights were not loaded! URL not found")

        self.weights = os.path.join(local_path, model_filename)
        self.data = self.configuration.get('data')
        device = self.configuration.get('device')
        if device == 'gpu':
            device = '0' if torch.cuda.is_available() else 'cpu'

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.conf_thres = self.configuration.get('conf_thres', 0.25)
        self.iou_thres = self.configuration.get('iou_thres', 0.45)
        imgsz_w = self.configuration.get('imgsz_w', 1280)
        imgsz_h = self.configuration.get('imgsz_h', 1280)
        self.imgsz = check_img_size((imgsz_w, imgsz_h), s=self.model.stride)

        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

        logger.info("Model loaded successfully")

    @smart_inference_mode()
    def run_inference(self, batch, upload_bbox=False):
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        annotation_collections = []

        for im0 in batch:
            im = im0[:, :, [2, 1, 0]]
            im = letterbox(im, self.imgsz, stride=self.model.stride, auto=self.model.pt)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            annotation_collection = dl.AnnotationCollection()
            annotation_collections.append(annotation_collection)
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False
                pred, out = self.model(im, augment=False, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred_ = non_max_suppression(prediction=pred,
                                            conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=None,
                                            agnostic=False,
                                            multi_label=False,
                                            labels=tuple(),
                                            max_det=self.max_det,
                                            nm=32)

            # Process predictions
            for i, det in enumerate(pred_):
                if len(det):
                    det_clone = det.clone()
                    det_clone_np = det_clone.cpu().numpy()
                    _masks = process_mask(proto[i], det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)  # HWC
                    _masks_np = _masks.cpu().numpy()
                    for current_mask_idx, current_mask in enumerate(_masks_np):
                        label = self.model.names[int(det_clone_np[current_mask_idx][5])]
                        confidence = det_clone_np[current_mask_idx][4]
                        if np.max(current_mask) == 0:
                            continue

                        annotation_collection.add(annotation_definition=dl.Polygon.from_segmentation(mask=current_mask,
                                                                                                     label=label),
                                                  model_info={'name': self.model_entity.name,
                                                              'model_id': self.model_entity.id,
                                                              'confidence': confidence})
                    if upload_bbox is True:
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        pred_np = det.cpu().numpy()

                        for current_pred in pred_np:
                            label = self.model.names[int(current_pred[5])]
                            confidence = current_pred[4]
                            annotation_collection.add(annotation_definition=dl.Box(top=current_pred[1],
                                                                                   left=current_pred[0],
                                                                                   bottom=current_pred[3],
                                                                                   right=current_pred[2],
                                                                                   label=label,
                                                                                   ),
                                                      model_info={'name': self.model_entity.name,
                                                                  'model_id': self.model_entity.id,
                                                                  'confidence': confidence}
                                                      )
        return annotation_collections

    def predict(self, batch, **kwargs):
        try:
            with torch.no_grad():
                return self.run_inference(batch=batch)

        except Exception as e:
            print(e)

