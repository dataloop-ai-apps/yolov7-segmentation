import dtlpy as dl
import logging
import os
import torch
from models.common import DetectMultiBackend
from PIL import Image
import shutil
import uuid

from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, non_max_suppression, scale_coords)
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode

logger = logging.getLogger('YOLOv7-Seg-Adapter')


@dl.Package.decorators.module(description='Model Adapter for Yolov7 Seg Model',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity=None):
        self.weights = None
        self.data = None
        self.imgsz = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = None
        super(ModelAdapter, self).__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename')
        self.weights = os.path.join(local_path, model_filename)
        self.data = self.configuration.get('data')
        # Load model
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)

        imgsz_w = self.configuration.get('imgsz_w', 1280)
        imgsz_h = self.configuration.get('imgsz_h', 1280)
        self.imgsz = check_img_size((imgsz_w, imgsz_h), s=self.model.stride)

        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

        logger.info("Model loaded successfully")

    @smart_inference_mode()
    def run_inference(self, source):
        source = str(source)
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt)
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        annotation_collections = []
        for path, im, im0s, vid_cap, s in dataset:
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
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if len(det):
                    det_clone = det.clone()
                    det_clone_np = det_clone.cpu().numpy()

                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    masks_np = masks.cpu().numpy()
                    for current_mask_idx, current_mask in enumerate(masks_np):
                        label = self.model.names[int(det_clone_np[current_mask_idx][5])]
                        confidence = det_clone_np[current_mask_idx][4]
                        annotation_collection.add(annotation_definition=dl.Polygon.from_segmentation(mask=current_mask,
                                                                                                     label=label),
                                                  model_info={'name': self.model_entity.name,
                                                              'model_id': self.model_entity.id,
                                                              'confidence': confidence})
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
        uid = str(uuid.uuid4())
        base_path = "{}".format(uid)
        os.makedirs(base_path, exist_ok=True)
        try:
            for i, img in enumerate(batch):
                img = Image.fromarray(img)
                img.save(os.path.join(base_path, str(i) + '.jpg'))

            with torch.no_grad():
                return self.run_inference(source=base_path)

        except Exception as e:
            print(e)
        finally:
            shutil.rmtree(base_path)
