from pathlib import Path
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2 import model_zoo
from detectron2.model_zoo.model_zoo import _ModelZooUrls
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

img_path = "input.png"
Path("imgs_output").mkdir(exist_ok=True)
img = read_image(img_path, format="BGR")

for i, config_path in enumerate(_ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()):
    # rpnとfast_rcnnは可視化対応していないので飛ばす
    if "rpn" in config_path or "fast_rcnn" in config_path:
        continue

    # config設定
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    score_thresh = 0.7
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_thresh
    cfg.freeze()

    # 検出＆可視化
    predictor = DefaultPredictor(cfg)
    visualizer = Visualizer(
        img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    predictions = predictor(img)
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to("cpu"), segments_info
        )
    else:
        if "sem_seg" in predictions:
            vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to("cpu")
            )
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

    # ファイル出力
    dataset_name, algorithm = config_path.split("/")
    algorithm = algorithm.split(".")[0]
    vis_output.save(f"imgs_output/{i:02d}-{dataset_name}-{algorithm}.jpg")
