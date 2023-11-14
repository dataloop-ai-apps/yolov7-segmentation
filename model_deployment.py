import dtlpy as dl
import os
from model_adapter import ModelAdapter

package_name = 'yolo-v7-segmentation'


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={
                                              'weights_filename': 'yolov7-seg.pt',
                                              'data': 'data/custom_data.yaml',
                                              'epochs': 5,
                                              'batch_size': 32,
                                              'imgsz_w': 1280,
                                              'imgsz_h': 1280,
                                              'conf_thres': 0.25,
                                              'iou_thres': 0.45,
                                              'max_det': 1000,
                                              'device': 'cpu'},
                                          output_type=[dl.AnnotationType.BOX, dl.AnnotationType.POLYGON]
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name=package_name,
                                    src_path=os.getcwd(),
                                    is_global=False,
                                    package_type='ml',
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.InstanceCatalog.REGULAR_M,
                                                                        runner_image='dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        preemptible=False,
                                                                        concurrency=1).to_json(),
                                        'executionTimeout': 10 * 3600,
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)
    return package


def model_creation(package: dl.Package, dataset: dl.Dataset):
    try:
        model = package.models.create(model_name=package_name,
                                      description='yolo v7 arch, pretrained on ms-coco',
                                      tags=[package_name, 'pretrained', 'ms-coco'],
                                      dataset_id=dataset.id,
                                      status='trained',
                                      scope='project',
                                      configuration={'weights_filename': 'yolov7-seg.pt',
                                                     'data': 'data/custom_data.yaml',
                                                     'epochs': 5,
                                                     'batch_size': 32,
                                                     'imgsz_w': 1280,
                                                     'imgsz_h': 1280,
                                                     'conf_thres': 0.25,
                                                     'iou_thres': 0.45,
                                                     'max_det': 1000,
                                                     'device': 'cpu'},
                                      project_id=package.project.id,
                                      input_type='image',
                                      output_type=dl.AnnotationType.POLYGON
                                      )
    except dl.exceptions.BadRequest:
        model = package.models.get(model_name='yolo-v7-segmentation')
        model.update()
    return model


def deploy():
    project_name = ''
    project = dl.projects.get(project_name)
    dataset = project.datasets.get(dataset_id='')
    package = package_creation(project=project)
    model = model_creation(package=package, dataset=dataset)
    model.artifacts.upload(r'')


deploy()
