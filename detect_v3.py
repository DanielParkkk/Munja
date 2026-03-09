import argparse
import os
import sys
from pathlib import Path
import torch
from torchvision import transforms
import pandas as pd
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
)
from utils.torch_utils import select_device

import classifier
import iou_cal


def cls_to_char(t_data, cls_idx):
    row = t_data[t_data[0] == int(cls_idx)]
    if len(row) > 0:
        return str(row.iloc[0, 1])
    return f"[{cls_idx}]"


def reorder_for_vertical_rtl(pred_bbox_list, pred_class_list, t_data, x_thresh=35):
    """
    옛한글 고문서용 읽기 순서 정렬
    - 열: 오른쪽 -> 왼쪽
    - 열 내부: 위 -> 아래

    x_thresh:
        같은 열로 묶을 x 중심값 허용 오차
        이미지에 따라 25~50 정도로 조절
    """
    items = []
    for bbox, cls_idx in zip(pred_bbox_list, pred_class_list):
        x1, y1, x2, y2 = map(int, bbox[:4])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ch = cls_to_char(t_data, cls_idx)

        items.append({
            "bbox": [x1, y1, x2, y2],
            "cx": cx,
            "cy": cy,
            "char": ch
        })

    # 1) 먼저 x 중심 기준으로 오른쪽 -> 왼쪽 정렬
    items.sort(key=lambda d: -d["cx"])

    # 2) 가까운 x 중심끼리 같은 열로 묶기
    columns = []
    for item in items:
        placed = False
        for col in columns:
            col_mean_x = sum(v["cx"] for v in col) / len(col)
            if abs(item["cx"] - col_mean_x) <= x_thresh:
                col.append(item)
                placed = True
                break
        if not placed:
            columns.append([item])

    # 3) 열 자체를 다시 오른쪽 -> 왼쪽으로 정렬
    columns.sort(key=lambda col: -(sum(v["cx"] for v in col) / len(col)))

    # 4) 각 열 내부는 위 -> 아래 정렬
    for col in columns:
        col.sort(key=lambda d: d["cy"])

    # 5) 최종 순서 펼치기
    ordered = []
    for col in columns:
        ordered.extend(col)

    return ordered


def print_only_ancient_text(pred_bbox_list, pred_class_list, t_data, x_thresh=35):
    ordered = reorder_for_vertical_rtl(
        pred_bbox_list,
        pred_class_list,
        t_data,
        x_thresh=x_thresh
    )

    chars = [item["char"] for item in ordered]
    text = "".join(chars)

    print("복원 문자열:")
    print(text)
    print()

    return text


@torch.no_grad()
def run(
    source=ROOT / 'sample_images',
):
    # 평가 모드 끄기: 라벨 없이 추론만 수행
    eval_flag = False

    weights = ROOT / 'weights_detector.pt'
    imgsz = (1280, 1280)
    conf_thres = 0.25
    iou_thres = 0.25
    max_det = 1000
    device = ''
    save_crop = False
    visualize = False
    project = ROOT / 'detect'
    name = 'exp'
    exist_ok = False

    # 결과 이미지 저장 안 함
    save_img = False

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load yolo model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # class.csv 로드
    t_data = pd.read_csv(ROOT / 'class.csv', header=None)
    t_data = t_data.iloc[:, [0, 1]]

    class_num = 0
    with open(ROOT / 'class.csv', 'r', encoding='UTF-8') as f:
        class_num = int(f.readlines()[-1].split(',')[0]) + 1
        print('분류기 class num : {}'.format(class_num))
        print('-------------------------------------------------')

    # Load classifier
    cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnistsimple_classifier_model = classifier.mnistsimple_Classifier_Model(class_num).to(cls_device)
    mnistsimple_classifier_model.load_state_dict(torch.load(ROOT / 'weights_classifier.pth', map_location=cls_device))
    mnistsimple_classifier_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1

    # Warmup
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

    try_cnt = 0

    for path, im, im0s, vid_cap, s, origin_img, x_pad, y_pad in dataset:
        try:
            start_time = time.time()

            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

            # Inference
            vis_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False, visualize=vis_path)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                s += '%gx%g ' % im.shape[2:]

                if len(det):
                    # 원본 크기로 bbox 복원
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    pred_bbox_list = iou_cal.tensor_to_list(det)

                    # 추론 모드
                    ocrdataset = classifier.mnistsimple_Dataset(
                        im0s,
                        pred_bbox_list,
                        [],
                        transforms=transform
                    )
                    ocrloader = torch.utils.data.DataLoader(
                        ocrdataset,
                        batch_size=1,
                        shuffle=False
                    )

                    pred_class_list = classifier.get_predictions(
                        mnistsimple_classifier_model,
                        device,
                        ocrloader,
                        []
                    )

                    print(path.split(sep='\\')[-1] + '\n')

                    # 옛한글 문자열만 출력
                    print_only_ancient_text(
                        pred_bbox_list,
                        pred_class_list,
                        t_data,
                        x_thresh=35
                    )
                else:
                    print(path.split(sep='\\')[-1] + '\n')
                    print("복원 문자열:")
                    print("")
                    print()

            print('소요 시간 : {} 초'.format(round(time.time() - start_time, 3)))
            print('에러 발생 횟수: {}'.format(try_cnt))
            print('-------------------------------------------------')

        except Exception as e:
            print(e)
            try_cnt += 1
            print('에러 발생 횟수: {}'.format(try_cnt))
            print('-------------------------------------------------')

    if save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='sample_images', help='dir')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)