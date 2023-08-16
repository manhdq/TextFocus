import argparse
import json
import os
import cv2
import os.path as osp
import sys
import numpy as np

import torch
import torch.nn.functional as F
from mmcv import Config

from dataset import build_dataset
from models import build_model
from models.utils import fuse_module
from prediction.grid_generator import GridGenerator
from prediction.focus_chip_generator import FocusChip
from utils import ResultFormat, visualize_focus_mask, visualize_detect

import time
import csv


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para / 1e6))
    print('-' * 90)


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def start_predict(model, dataload, img_name, img, cfg):
    start = time.time()

    scaled_img = dataload.scale_image_short(img)
    img_meta = dataload.get_img_meta(img, scaled_img)
    img_meta = dataload.convert_img_meta_to_tensor(img_meta)

    scaled_img = dataload.convert_img_to_tensor(scaled_img).unsqueeze(0)
    scaled_img = scaled_img.cuda()
    data = dict(imgs=scaled_img, img_metas=img_meta, cfg=cfg)

    outputs = model(**data)
    end = time.time()
    outputs.update(dict(time=end-start, num_tile=1))

    return outputs


pred_idx = 0
def start_recursive_predict(model, dataload, img_name, img, grid_gen, foc_chip_gen, cfg):
    global pred_idx
    pred_idx = 0

    start = time.time()
    ori_img_h, ori_img_w = img.shape[:2]

    scaled_down_tiles, base_scale_down = grid_gen.gen_from_ori_img(img)
    if scaled_down_tiles is None:
        print(f'Ignore {img_name} due to the image size ({ori_img_w}x{ori_img_h})!')
        return {
            'predictions': [],
            'prediction_time': 0
        }

    backbone_outs = []

    for tile in scaled_down_tiles:
        backbone_outs_res = recursive_predict(img_name,
                                            img,
                                            tile['image'],
                                            model=model,
                                            dataload=dataload,
                                            rank=0,
                                            base_scale_down=base_scale_down,
                                            grid_gen=grid_gen,
                                            foc_chip_gen=foc_chip_gen,
                                            prev_left_shift=tile['prev_left_shift'],
                                            prev_top_shift=tile['prev_top_shift'],
                                            prev_right_shift=tile['prev_bottom_shift'],
                                            prev_bottom_shift=tile['prev_bottom_shift'],
                                            cfg=cfg)
        backbone_outs.append(backbone_outs_res)

    end = time.time()
    focus_time = end - start

    start = time.time()

    backbone_outs, chip_scaled_size_list, rank_list, shift_infos_list = _flatten_and_stack_backbone_outs(backbone_outs)
    det_outs = model.get_det_map_after_backbone(backbone_outs)
    last_det_out = None

    for det_out, chip_scaled_size, rank, shift_infos \
                    in zip(det_outs, chip_scaled_size_list, rank_list, shift_infos_list):
        prev_left_shift, prev_top_shift, prev_right_shift, prev_bottom_shift = shift_infos
        det_out = det_out.unsqueeze(0)

        to_ori_scale = base_scale_down / \
                (pow(cfg.autofocus.zoom_in_scale, max(rank - 1, 0)) * \
                (pow(cfg.autofocus.first_row_zoom_in, min(rank, 1))))
        chip_org_size = (chip_scaled_size * to_ori_scale).astype(int)

        final_det_out = F.interpolate(det_out, (chip_org_size[0], chip_org_size[1]), mode="nearest")
        final_det_mask = torch.ones_like(final_det_out) * rank

        final_det_out = F.pad(final_det_out, (prev_left_shift, prev_right_shift, prev_top_shift, prev_bottom_shift), "constant", 0)
        final_det_mask = F.pad(final_det_mask, (prev_left_shift, prev_right_shift, prev_top_shift, prev_bottom_shift), "constant", 0)
        
        final_det_out = F.interpolate(final_det_out, (ori_img_h, ori_img_w), mode="nearest")
        if rank == 0:
            final_det_out = final_det_out * 0.8
        final_det_mask = F.interpolate(final_det_mask, (ori_img_h, ori_img_w), mode="nearest")

        if last_det_out is not None:
                last_det_out = (final_det_out * final_det_mask + last_det_out) / (final_det_mask + 1)
        else:
            last_det_out = final_det_out

    img_meta = dataload.get_img_meta(img, img)
    img_meta = dataload.convert_img_meta_to_tensor(img_meta)

    outputs = model.det_head.get_results(last_det_out, img_meta, cfg)
    _draw_detect_on_chip(img_name, img, outputs)

    end = time.time()
    detect_time = end - start

    outputs.update(dict(focus_time=focus_time))
    outputs.update(dict(detect_time=detect_time))
    outputs.update(dict(total_time=focus_time + detect_time))
    outputs.update(dict(num_tile=len(rank_list)))

    return outputs


def _flatten_and_stack_backbone_outs(backbone_outs):
    """
    Flatten the recursive backbone outs
    """
    backbone_outs_list = []
    chip_scaled_size_list = []
    rank_list = []
    shift_infos_list = []
    chip_backbone_queue = [*backbone_outs]

    while len(chip_backbone_queue) > 0:
        _backbone_out = chip_backbone_queue.pop(0)

        backbone_outs_list.append(_backbone_out['cur_backbone_out'])
        chip_scaled_size_list.append(_backbone_out['chip_scaled_size'])
        rank_list.append(_backbone_out['rank'])
        shift_infos_list.append(_backbone_out['shift_infos'])

        chip_backbone_queue.extend(_backbone_out['backbone_outs'])

    # # Sort based on rank
    # backbone_outs_list = [b for _,b in sorted(zip(rank_list, backbone_outs_list))]

    backbone_outs = stack_backbone_outs(backbone_outs_list)

    return backbone_outs, chip_scaled_size_list, rank_list, shift_infos_list


def stack_backbone_outs(backbone_outs_list):
    backbone_outs = []

    # Area of feats
    area_feats = []
    for f in backbone_outs_list:
        area_feats.append(f[0].shape[-2] * f[0].shape[-1])
    # Find best w and h of each feature map
    best_id = np.argmax(area_feats)

    for f_layer_id in range(4):
        f = []
        best_size = (backbone_outs_list[best_id][f_layer_id].shape[-2],
                    backbone_outs_list[best_id][f_layer_id].shape[-1])
        for f_id in range(len(backbone_outs_list)):
            f.append(F.interpolate(backbone_outs_list[f_id][f_layer_id], 
                                best_size, mode="nearest"))
        backbone_outs.append(torch.cat(f))
    
    return backbone_outs


def recursive_predict(img_name,
                    ori_image,
                    chip,
                    model,
                    dataload,
                    rank,
                    base_scale_down,
                    grid_gen,
                    foc_chip_gen,
                    prev_left_shift,
                    prev_top_shift,
                    prev_right_shift,
                    prev_bottom_shift,
                    cfg):

    scaled_chip = dataload.scale_image_short(chip)
    chip_meta = dataload.get_img_meta(chip, scaled_chip)
    chip_scaled_size = chip_meta['org_img_size']
    chip_meta = dataload.convert_img_meta_to_tensor(chip_meta)

    scaled_chip = dataload.convert_img_to_tensor(scaled_chip).unsqueeze(0)
    scaled_chip = scaled_chip.cuda()
    outputs = model.focus(imgs=scaled_chip)

    cur_backbone_out = outputs['backbone_out']
    autofocus_out = outputs['autofocus_out'][0].cpu().detach().numpy()

    if rank == 0:
        autofocus_out = np.ones_like(autofocus_out)

    to_ori_scale = base_scale_down / \
            (pow(cfg.autofocus.zoom_in_scale, max(rank - 1, 0)) * \
            (pow(cfg.autofocus.first_row_zoom_in, min(rank, 1))))
    chip_org_size = (chip_scaled_size * to_ori_scale).astype(int)

    if cfg.autofocus.draw_preds_chip:
        _draw_focus_mask_on_chip(img_name, chip, autofocus_out, rank)

    backbone_outs = []
    if rank < cfg.autofocus.max_focus_rank:
        chip_height, chip_width = chip.shape[:2]
        backbone_outs = _recursive_pred_on_focus(img_name,
                                                ori_image,
                                                model,
                                                dataload,
                                                autofocus_out,
                                                chip_width,
                                                chip_height,
                                                rank,
                                                base_scale_down,
                                                grid_gen,
                                                foc_chip_gen,
                                                to_ori_scale,
                                                prev_left_shift,
                                                prev_top_shift,
                                                prev_right_shift,
                                                prev_bottom_shift,
                                                cfg)

    result = {
        "cur_backbone_out": cur_backbone_out,
        "backbone_outs": backbone_outs,
        "chip_scaled_size": chip_scaled_size,
        "shift_infos": [prev_left_shift, prev_top_shift, prev_right_shift, prev_bottom_shift],
        "rank": rank
    }

    return result


def _recursive_pred_on_focus(img_name,
                            ori_image,
                            model,
                            dataload,
                            focus_mask,
                            chip_width,
                            chip_height,
                            rank,
                            base_scale_down,
                            grid_gen,
                            foc_chip_gen,
                            to_ori_scale,
                            prev_left_shift,
                            prev_top_shift,
                            prev_right_shift,
                            prev_bottom_shift,
                            cfg):
    """
    Cut focus chip and do prediction on this chip
    """
    ori_img_h, ori_img_w = ori_image.shape[:2]
    backbone_outs = []
    # Crop sub chips from the input chip by using the focus mask
    chip_coords = foc_chip_gen(focus_mask, chip_width, chip_height)
    for chip_coord in chip_coords:
        for tile_coord in grid_gen.gen_from_chip(chip_coord, rank):
            # Convert chip coordinates to original coordinates by
            # scaling chip coordinates
            # and shift chip coordinates to match the top-left of the original image
            resized_bbox = batch_scale_n_shift_dets(py_preds=[np.expand_dims(tile_coord, axis=0).astype(float)],
                                            scale=to_ori_scale,
                                            left_shift=prev_left_shift,
                                            top_shift=prev_top_shift,
                                            right_shift=prev_right_shift,
                                            bottom_shift=prev_bottom_shift)[0]
            ori_x1, ori_y1, ori_x2, ori_y2 = resized_bbox[0]

            # Crop interested region on original image
            ori_chip_crop = ori_image[round(ori_y1):round(ori_y2),
                                    round(ori_x1):round(ori_x2), :]

            zoom_scale = cfg.autofocus.zoom_in_scale if rank > 0 else cfg.autofocus.first_row_zoom_in
            zoom_in_x1, zoom_in_y1, zoom_in_x2, zoom_in_y2 = \
                list(map(lambda x: x * zoom_scale, tile_coord))
            zoom_in_chip_w = round(zoom_in_x2 - zoom_in_x1)
            zoom_in_chip_h = round(zoom_in_y2 - zoom_in_y1)

            zoom_in_chip_crop = cv2.resize(ori_chip_crop,
                                        (zoom_in_chip_w, zoom_in_chip_h),
                                        interpolation=grid_gen.interpolation)

            cur_backbone_outs = recursive_predict(img_name,
                                                ori_image,
                                                zoom_in_chip_crop,
                                                model,
                                                dataload,
                                                rank + 1,
                                                base_scale_down,
                                                grid_gen,
                                                foc_chip_gen,
                                                int(ori_x1),
                                                int(ori_y1),
                                                int(ori_img_w - ori_x2),
                                                int(ori_img_h - ori_y2),
                                                cfg)
            backbone_outs.append(cur_backbone_outs)

    return backbone_outs


def batch_scale_n_shift_dets(py_preds, scale, left_shift, top_shift, right_shift, bottom_shift):
    '''
    Scale and shift old coordinate to new coordinate
    `py_preds` contains list of lm preds according to `GDN` level
    '''
    _py_preds = []
    for cur_py_preds in py_preds:
        _cur_py_preds = cur_py_preds.copy()
        _cur_py_preds *= scale
        
        _cur_py_preds[..., 0::2] = _cur_py_preds[..., 0::2] + left_shift
        _cur_py_preds[..., 1::2] = _cur_py_preds[..., 1::2] + top_shift
        
        _py_preds.append(_cur_py_preds)
    return _py_preds


def _draw_focus_mask_on_chip(img_name, chip, focus_mask, rank):
    """
    Draw and save predictions to chips
    """
    global pred_idx
    chip_with_preds_save_path = None

    # Draw preds on img
    chip = visualize_focus_mask(image=chip,
                    mask=focus_mask,
                    mask_color=(255, 0, 255))
    chip_with_preds_save_path = os.path.abspath(
        os.path.join("chip_results", img_name.split('.')[0], f"{img_name.split('.')[0]}_with_preds_{rank}_{pred_idx}.jpg")
    )
    os.makedirs(os.path.join("chip_results", img_name.split('.')[0]), exist_ok=True)
    cv2.imwrite(chip_with_preds_save_path,
                chip[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    pred_idx += 1  # Increase pred_idx for the next saved prediction  ##TODO: For just 1 image
    return chip_with_preds_save_path


def _draw_detect_on_chip(img_name, chip, dets):
    """
    Draw and save predictions to chips
    """
    global pred_idx
    chip_with_preds_save_path = None

    py_preds, confidences = dets['bboxes'], dets['scores']
    for i in range(len(py_preds)):
        py_pred = py_preds[i]
        assert len(py_pred) % 2 == 0
        num_pts = len(py_pred) // 2
        py_preds[i] = py_pred.reshape((num_pts, 2))

    # Filter detections for visualize with vis_threshold
    selected_sample = np.array(confidences) >= 0.5
    vis_py_preds = [py_preds[i] for i in range(len(selected_sample)) if selected_sample[i]]
    vis_confidences = [confidences[i] for i in range(len(selected_sample)) if selected_sample[i]]

    # Draw preds on img
    chip = visualize_detect(image=chip,
                        points_group=vis_py_preds,
                        boundary_color=(255, 0, 255),
                        confidences=vis_confidences)
    chip_with_preds_save_path = os.path.abspath(
        os.path.join("chip_results", img_name.split('.')[0], f"{img_name.split('.')[0]}.jpg")
    )
    os.makedirs(os.path.join("chip_results", img_name.split('.')[0]), exist_ok=True)
    cv2.imwrite(chip_with_preds_save_path,
                chip[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    pred_idx += 1  # Increase pred_idx for the next saved prediction  ##TODO: For just 1 image
    return chip_with_preds_save_path


def test(testset, model, grid_gen, foc_chip_gen, cfg):
    model.eval()

    with_rec = hasattr(cfg.model, 'recognition_head')
#     if with_rec:
#         pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)

#     if cfg.vis:
#         vis = Visualizer(vis_path=osp.join('vis/', cfg.data.test.type))

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

#     if cfg.report_speed:
#         speed_meters = dict(
#             backbone_time=AverageMeter(500),
#             neck_time=AverageMeter(500),
#             det_head_time=AverageMeter(500),
#             det_post_time=AverageMeter(500),
#             rec_time=AverageMeter(500),
#             total_time=AverageMeter(500))

    print('Start testing %d images' % len(testset))
    cfg.debug = False
    cfg.report_speed = False

    num_tiles = []
    focus_times = []
    detect_times = []
    total_times = []
    
    for idx in range(len(testset)):
        print('Testing %d/%d\r' % (idx, len(testset)), end='', flush=True)

        # prepare input
        img = testset.get_image(idx)
        image_name = testset.img_paths[idx].split(os.sep)[-1].split('.')[0]

        start = time.time()
        # forward
        with torch.no_grad():
            # outputs = start_predict(model=model, dataload=testset, img=img, cfg=cfg)
            # print(outputs)
            # exit()
            if not model.using_autofocus:
                outputs = start_predict(model=model, dataload=testset, img_name=image_name, img=img, cfg=cfg)
            else:
                outputs = start_recursive_predict(model=model, dataload=testset, 
                                                img_name=image_name, img=img, 
                                                grid_gen=grid_gen, foc_chip_gen=foc_chip_gen, 
                                                cfg=cfg)

            # print('output', outputs)
        end = time.time()

        with open('./results/time/tmp.csv', 'a', encoding='utf8') as f:
            wr = csv.writer(f)
            wr.writerow([*outputs['scores'], end - start])
            
#         if cfg.report_speed:
#             report_speed(outputs, speed_meters)
        # post process of recognition
#         if with_rec:
#             outputs = pp.process(data['img_metas'], outputs)

        # save result
        image_name, _ = osp.splitext(osp.basename(testset.img_paths[idx]))
        image_path=testset.img_paths[idx]
        rf.write_result(image_name, image_path, outputs)
#         rf.write_result(data['img_metas'], outputs)

        # visualize
#         if cfg.vis:
#             vis.process(data['img_metas'], outputs)
        num_tiles.append(outputs['num_tile'])
        focus_times.append(outputs['focus_time'])
        detect_times.append(outputs['detect_time'])
        total_times.append(outputs['total_time'])
        print(f"Num tiles: {outputs['num_tile']} - Focus time: {outputs['focus_time']} - Detect time: {outputs['detect_time']} - Total time: {outputs['total_time']} s")

    print(f"Avg num tiles: {np.mean(num_tiles):.2f}")
    print(f"Avg Focus Time: {np.mean(focus_times):.2f}")
    print(f"Avg Detect Time: {np.mean(detect_times):.2f}")
    print(f"FPS: {len(testset) / np.sum(total_times):.2f}")
    print('Done!')


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
#     cfg.update(dict(vis=args.vis))
#     cfg.update(dict(debug=args.debug))
#     cfg.data.test.update(dict(debug=args.debug))
    cfg['resize_param'] = [args.resize_const, args.pos_const, args.len_const]
    print(json.dumps(cfg._cfg_dict, indent=4))

    # data loader
    testset = build_dataset(cfg.data, focus_gen=None, split="test")
    # test_loader = torch.utils.data.DataLoader(
    #     testset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=2,
    # )
    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(
            dict(
                voc=testset.voc,
                char2id=testset.char2id,
                id2char=testset.id2char,
            ))
    model = build_model(cfg)
    model = model.cuda()

    if os.path.isfile(cfg.checkpoint):
        print("Loading model and optimizer from checkpoint '{}'".format(
            cfg.checkpoint))

        checkpoint = torch.load(cfg.checkpoint)
        model_state_dict = checkpoint['state_dict']
        new_model_state_dict = model.state_dict()

        leftover_state_names = []
        for key, _ in new_model_state_dict.items():
            if "module." + key in model_state_dict:
                new_model_state_dict[key] = model_state_dict["module." + key]
            else:
                leftover_state_names.append(key)

        # d = dict()
        # for key, value in checkpoint['state_dict'].items():
        #     tmp = key[7:]  ## Eliminate "module."
        #     d[tmp] = value
        model.load_state_dict(new_model_state_dict)
        print("State names not exists in loaded checkpoint:")
        for state_name in leftover_state_names:
            print(f"- {state_name}")
    else:
        print("No checkpoint found at '{}'".format(args.resume))
        raise

    # fuse conv and bn
    model = fuse_module(model)
    model_structure(model)

    grid_gen = None
    foc_chip_gen = None
    if cfg.using_autofocus:
        grid_gen = GridGenerator(cfg.grid_generator)
        foc_chip_gen = FocusChip(cfg.focus_chip_generator)
    # test
    test(testset, model, grid_gen, foc_chip_gen, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', default='config/pan/pan_r18_ctw_finetune.py', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    parser.add_argument('--resize_const', default=1)
    parser.add_argument('--pos_const', default=0)
    parser.add_argument('--len_const', default=0)
#     parser.add_argument('--vis', action='store_true')
#     parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)