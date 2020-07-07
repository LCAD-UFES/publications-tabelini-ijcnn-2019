import os
import cv2
import json
import math
import time
import shutil
import pickle
import argparse
import numpy as np
import imgaug as ia
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt


def augment_target(target, multiply_value=None, add_value=None):
    if add_value is None:
        add_value = float(np.random.uniform(-120, 120))
    if multiply_value is None:
        multiply_value = float(np.random.uniform(0.75, 1.25))

    seq = iaa.Sequential([
        iaa.Add((add_value, add_value)),
        iaa.Multiply((multiply_value, multiply_value)),
    ])
    target = (target * 255.0).astype(np.uint8)
    return (seq.augment_image(target) / 255.0).astype(np.float32), add_value, multiply_value


def blur(img, mask, target_region, value=None):
    if value is None:
        value = float(np.random.uniform(0, 2.75))
    blur_effect = iaa.Sequential([iaa.GaussianBlur(value)])
    cpy = target_region.copy()
    cpy[mask[:, :, 0] > 0] = img[:, :, [0, 1, 2]][mask[:, :, 0] > 0]
    img = blur_effect.augment_image(cpy)
    return img


def histogram_noise(img, noise=(-15, 15), data=None):
    if data is not None:
        np.random.set_state(data['state'])
    noise = np.random.randint(noise[0], noise[1], size=img.shape)
    return np.clip(img + (noise / 255.0), 0, 1), {'state': np.random.get_state()}


def brightness_transform(template, template_mask, target_region):
    template_h, template_w, _ = template.shape

    target_grayscale = (cv2.cvtColor((target_region * 255.0).astype(np.uint8),
                                     cv2.COLOR_RGB2GRAY) / 255.0).astype(np.float32)

    target_average = target_grayscale.mean()
    indexes = template_mask[:, :, 0] > 0
    binary_template_mask = template_mask.copy()
    binary_template_mask[indexes] = 1

    difference = target_average - 170 / 255.0
    masked_difference = binary_template_mask[:, :, 0] * difference

    for channel in range(3):
        template[:, :, channel] += masked_difference

        template[:, :, channel] = np.clip(template[:, :, channel], 0, 1)

    return template, {}


def geometric_transform(template, template_mask, x, target_size=(1500, 1500), scale=None, prelodaded_data=None):
    data = {}
    rows, cols, _ = template.shape
    x -= int(round(target_size[1] / 2))
    relative_x = abs(2 * x / target_size[1])

    # perspective
    persp_max_min = (
        int(round(template.shape[1] * 0.07)), int(round(template.shape[1] * 0.14)))
    h, w, _ = template.shape
    persp_min, persp_max = persp_max_min
    if prelodaded_data is None:
        persp = np.random.randint(persp_min, persp_max + 1)
    else:
        persp = prelodaded_data['persp']
    data['persp'] = persp

    if x > 0:
        pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        pts2 = np.float32([[persp, persp], [w - 1, 0],
                           [persp, h - 1 - persp], [w - 1, h - 1]])
    else:
        pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        pts2 = np.float32([[0, 0], [w - 1 - persp, persp],
                           [0, h - 1], [w - 1 - persp, h - 1 - persp]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    template = cv2.warpPerspective(template, M, (cols, rows))
    template_mask = cv2.warpPerspective(
        template_mask, M, (cols, rows))

    # rotation
    if prelodaded_data is None:
        angle = np.random.uniform(-10, 10)
    else:
        angle = prelodaded_data['angle']
    data['angle'] = angle
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    template = cv2.warpAffine(template, M, (cols, rows), flags=cv2.INTER_LINEAR)
    template_mask = cv2.warpAffine(template_mask, M, (cols, rows), flags=cv2.INTER_LINEAR)

    # scale
    if prelodaded_data is None:
        if scale is None:
            scale_factor = (.175 + relative_x) * (min(template.shape[0], template.shape[1])) / 100.0
        else:
            scale_factor = scale
    else:
        scale_factor = prelodaded_data['scale']
    data['scale'] = scale_factor
    template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)
    template_mask = cv2.resize(
        template_mask, (0, 0), fx=scale_factor, fy=scale_factor)

    return template, template_mask, scale_factor, data


def blend(template, template_mask, target_image, target_bbox, steps=3):
    template = (template * 255).astype(np.uint8)
    target_image = (target_image * 255).astype(np.uint8)

    temp_template_mask = template_mask.copy()
    temp_template_mask = cv2.copyMakeBorder(temp_template_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    blend_mask = temp_template_mask.astype(np.float32) * (1.0 / steps)
    kernel = np.ones((3, 3), np.uint8)

    for step in range(steps - 1):
        temp_template_mask = cv2.erode(temp_template_mask, kernel)
        blend_mask += temp_template_mask * (1.0 / steps)

    x0 = target_bbox['x0']
    y0 = target_bbox['y0']
    x1 = target_bbox['x1']
    y1 = target_bbox['y1']
    blend_mask = blend_mask[1:-1, 1:-1]
    blended = (target_image[y0:y1, x0:x1] * (1 - blend_mask)) + (template[:, :, [0, 1, 2]] * blend_mask)

    return blended.astype(np.float32) / 255.0


def get_random_position(probabilities_vector, positions_list, img_size=(2048, 2048), sample_size=1):
    positions = np.random.choice(
        positions_list, size=sample_size, p=probabilities_vector)
    positions = [(position % img_size[1], math.ceil(
        position / img_size[0]) - 1) for position in positions]

    return positions


def multiply(img, multiply_value):
    aug = iaa.Multiply((multiply_value, multiply_value))
    img = (img * 255.0).astype(np.uint8)
    return (aug.augment_image(img) / 255.0).astype(np.float32)


def has_intersection(x0, y0, x1, y1, bboxes):
    for bbox in bboxes:
        if x1 > bbox['xmin'] and bbox['xmax'] > x0:
            if y1 > bbox['ymin'] and bbox['ymax'] > y0:
                return True
    return False


def remove_padding(template, template_mask):
    mask = template_mask[:, :, 0] > 0

    coords = np.argwhere(mask)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return template[y0:y1, x0:x1], template_mask[y0:y1, x0:x1], x0, y0, x1, y1


def process_img(target, template, template_mask, probabilities_vector, positions_list, multiply_value, position=None,
                bboxes=None, scale=None, prelodaded_data=None):
    template = cv2.copyMakeBorder(template, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    template_mask = cv2.copyMakeBorder(template_mask, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    data = {}
    if prelodaded_data is None:
        position_is_centered = position is None
        if position is None:
            position = get_random_position(probabilities_vector, positions_list, target.shape[:-1])[0]
    else:
        position_is_centered = prelodaded_data['position_is_centered']
        position = prelodaded_data['position']
    data['position'] = (int(position[0]), int(position[1]))
    data['position_is_centered'] = position_is_centered
    x, y = position
    template = multiply(template, multiply_value)
    template[:, :, 0][template_mask[:, :, 0] < 1] = 0
    template[:, :, 1][template_mask[:, :, 0] < 1] = 0
    template[:, :, 2][template_mask[:, :, 0] < 1] = 0
    template, template_mask, scale, geometric_transform_data = geometric_transform(template, template_mask, x,
                                                                                   scale=scale,
                                                                                   prelodaded_data=prelodaded_data[
                                                                                       'geometric_transform_data'] if prelodaded_data is not None else None)
    data['geometric_transform_data'] = geometric_transform_data
    template_h, template_w, _ = template.shape
    target_h, target_w, _ = target.shape
    if not position_is_centered:
        x0 = x
    else:
        x0 = int(round(x - template_w / 2))
    if x0 < 0:
        x0 = 0
    if not position_is_centered:
        y0 = y
    else:
        y0 = int(round(y - template_h / 2))
    if y0 < 0:
        y0 = 0

    x1 = x0 + template_w
    y1 = y0 + template_h

    if x1 >= target_w and position_is_centered or y1 >= target_h and position_is_centered:
        return target, None, scale, data
    if x1 >= target_w:
        diff = x1 - target_w + 1
        x0 -= diff
        x1 -= diff
    if y1 >= target_h:
        diff = y1 - target_h + 1
        y0 -= diff
        y1 -= diff
    template, brightness_transform_data = brightness_transform(template, template_mask, target[y0:y1, x0:x1])
    data['brightness_transform_data'] = brightness_transform_data

    template, histogram_noise_data = histogram_noise(template, data=prelodaded_data[
        'histogram_noise_data'] if prelodaded_data is not None else None)
    data['histogram_noise_data'] = histogram_noise_data

    # remove unused space from bbox
    w, h, _ = template.shape
    template, template_mask, dx0, dy0, dx1, dy1 = remove_padding(template, template_mask)
    x1 = x0 + template.shape[1]
    y1 = y0 + template.shape[0]

    if bboxes is not None and has_intersection(x0, y0, x1, y1, bboxes):
        return target, None, scale, data

    if x0 < 0 or y0 < 0 or x1 >= target_w or y1 >= target_h:
        return target, None, scale, data

    template = blend(template, template_mask, target, {
        'x0': x0,
        'y0': y0,
        'x1': x1,
        'y1': y1
    })
    # place template on target
    target[y0:y1, x0:x1] = template

    return target, {
        'xmin': x0,
        'ymin': y0,
        'xmax': x1,
        'ymax': y1
    }, scale, data


def get_mask_from_image(alpha_image):
    alpha_channel = alpha_image[:, :, -1]
    mask = np.zeros_like(alpha_image[:, :, :-1])
    mask[:, :, 0][alpha_channel > 0] = 1
    mask[:, :, 1][alpha_channel > 0] = 1
    mask[:, :, 2][alpha_channel > 0] = 1

    return mask


def hc_probabilties_vector(map_size):
    map = np.ones(map_size)
    return map.flatten() / map.sum()


def generate_sample(targets_path, img_name, templates, probabilities_vector, positions_list, images_out_path,
                    nb_imgs_generated, binary_annotation_lines_queue, multiclass_annotation_lines_queue, data_out_path,
                    load_path):
    np.random.seed(None)
    img_path = os.path.join(targets_path, img_name)
    try:
        target = cv2.imread(img_path).astype(np.float32) / 255.0
    except:
        print('Error with:', img_path)
        return
    if load_path is None:
        image_data = {
            'bbox_data': []
        }
        nb_signs_in_img = np.random.randint(1, 5 + 1)
        total = 0
        target, add_value, multiply_value = augment_target(target)
        image_data['add_value'] = add_value
        image_data['multiply_value'] = multiply_value
        bboxes = []
        template_ids_selected = np.random.choice(list(range(len(templates))), size=nb_signs_in_img, replace=False)
        image_data['template_ids'] = template_ids_selected
        scale = None
        while total < nb_signs_in_img:
            template = templates[template_ids_selected[total]][0]
            template_mask = templates[template_ids_selected[total]][1]
            template_category = templates[template_ids_selected[total]][2]

            target, bbox, scale, data = process_img(
                target.copy(), template.copy(), template_mask, probabilities_vector, positions_list, multiply_value,
                bboxes=bboxes, scale=scale)
            if bbox:
                bbox['category'] = template_category
                bboxes.append(bbox)
                image_data['bbox_data'].append({
                    'bbox': bbox,
                    'data': data
                })
                total += 1
            probs = [0.4, 0.5]
            while len(probs) > 0:
                do_place_below = np.random.choice([True, False], p=[probs[0], 1 - probs[0]]) and total < nb_signs_in_img
                if not (do_place_below and total < nb_signs_in_img and bbox):
                    break
                probs = probs[1:]
                position = (bbox['xmin'], bbox['ymax'])
                template = templates[template_ids_selected[total]][0]
                template_mask = templates[template_ids_selected[total]][1]
                template_category = templates[template_ids_selected[total]][2]
                target, bbox, scale, data = process_img(
                    target.copy(), template, template_mask, probabilities_vector, positions_list, multiply_value,
                    position, bboxes, scale=scale)
                if bbox:
                    bbox['category'] = template_category
                    bboxes.append(bbox)
                    image_data['bbox_data'].append({
                        'bbox': bbox,
                        'data': data
                    })
                    total += 1

        blur_value = float(np.random.uniform(0, 7)) * scale
        image_data['blur_value'] = blur_value
    else:
        with open(os.path.join(load_path, "{:05d}.pkl".format(nb_imgs_generated)), "rb") as data_in_f:
            image_data = pickle.load(data_in_f)

        target, _, _ = augment_target(target, multiply_value=image_data['multiply_value'],
                                      add_value=image_data['add_value'])
        bboxes = []
        scale = None
        template_ids_selected = image_data['template_ids']
        for total, bbox_data in enumerate(image_data['bbox_data']):
            template = templates[template_ids_selected[total]][0]
            template_mask = templates[template_ids_selected[total]][1]

            data = bbox_data['data']
            target, _, _, _ = process_img(target.copy(), template.copy(), template_mask, probabilities_vector,
                                          positions_list, image_data['multiply_value'], bboxes=bboxes, scale=scale,
                                          prelodaded_data=data)
            bboxes.append(bbox_data['bbox'])

        blur_value = image_data['blur_value']
    blur_effect = iaa.Sequential([iaa.GaussianBlur(blur_value, deterministic=True)])
    target = blur_effect.augment_image(target)

    img_out_path = os.path.join(images_out_path, "{:05d}_{}.jpg".format(nb_imgs_generated,
                                                                        os.path.splitext(img_name)[0]))
    cv2.imwrite(img_out_path, (target * 255).astype(np.uint8))
    binary_annotation_lines = []
    multiclass_annotation_lines = []
    for bbox in bboxes:
        binary_line = "{},{},{},{},{},{}".format(img_out_path, bbox['xmin'], bbox[
            'ymin'], bbox['xmax'], bbox['ymax'], "traffic_sign")
        multiclass_line = "{},{},{},{},{},{}".format(
            img_out_path, bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'], bbox['category'])

        binary_annotation_lines.append(binary_line)
        multiclass_annotation_lines.append(multiclass_line)

    multiclass_annotation_lines_queue.put(multiclass_annotation_lines)
    binary_annotation_lines_queue.put(binary_annotation_lines)

    with open(os.path.join(data_out_path, "{:05d}.pkl".format(nb_imgs_generated)), "wb") as data_out_f:
        pickle.dump(image_data, data_out_f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a data set with templates.')

    parser.add_argument('--bgs-path', dest='targets_path', type=str, required=True,
                        help='Path to the directory containing background images to be used.')
    parser.add_argument('--templates-path', dest='templates_path', type=str, required=True,
                        help='Path to the directory containing template images to be used.')
    parser.add_argument('--out-path', dest='out_path', type=str, required=True,
                        help='Path to the directory to save the images generated to.')
    parser.add_argument('--total-images', dest='total_images', type=int, required=True,
                        help='Number of images to be generated.')
    parser.add_argument('--data', dest='random_data', type=str, default=None,
                        help='Path to data directory (to reproduce a data set, not working perfectly at the moment, will be at future version)')
    parser.add_argument('--max_process', dest='max_process', type=int, default=1,
                        help='Maximum number of parallel processes')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    targets_path = args.targets_path
    templates_path = args.templates_path

    images_out_path = os.path.join(args.out_path, "imgs")
    data_out_path = os.path.join(args.out_path, "data")
    os.makedirs(args.out_path)
    os.makedirs(images_out_path)
    os.makedirs(data_out_path)

    shutil.copyfile(os.path.realpath(__file__), os.path.join(args.out_path, 'generate_dataset.py'))

    all_img_names = os.listdir(targets_path)
    probabilities_vector = hc_probabilties_vector((1500, 1500))
    positions_list = np.arange(0, probabilities_vector.size)

    template_names = os.listdir(templates_path)
    nb_classes = len(template_names)
    img_names = np.random.choice(all_img_names, size=args.total_images)

    templates = []
    t0 = time.time()
    for template_name in template_names:
        template_path = os.path.join(templates_path, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.0
        template_mask = get_mask_from_image(template)
        category = os.path.splitext(template_name)[0]
        templates.append((template, template_mask, category))

    MAX_PROCESS = args.max_process
    processes = []
    load_path = args.random_data
    binary_annotation_lines_queue = mp.Queue()
    multiclass_annotation_lines_queue = mp.Queue()
    binary_annotation_lines = []
    multiclass_annotation_lines = []
    nb_imgs_generated = 0
    t0_temp = time.time()
    pbar = tqdm(total=args.total_images)
    for idx, img_name in enumerate(img_names):
        p = mp.Process(target=generate_sample, args=((
            targets_path, img_name, templates, probabilities_vector, positions_list, images_out_path, idx,
            binary_annotation_lines_queue, multiclass_annotation_lines_queue, data_out_path, load_path)))
        p.daemon = True
        p.start()
        processes.append(p)
        if len(processes) == MAX_PROCESS:
            for p in processes:
                p.join()
                pbar.update(1)
                binary_annotation_lines += binary_annotation_lines_queue.get()
                multiclass_annotation_lines += multiclass_annotation_lines_queue.get()
                nb_imgs_generated += 1

                if nb_imgs_generated % 1000 == 0:
                    t1 = time.time()
                    print("Time spent in last 1000 images: {:.2f}s".format(t1 - t0_temp))
                    t0_temp = t1
            processes = []
    pbar.close()

    if len(processes) > 0:
        for p in processes:
            p.join()
            binary_annotation_lines += binary_annotation_lines_queue.get()
            multiclass_annotation_lines += multiclass_annotation_lines_queue.get()
            nb_imgs_generated += 1
            if nb_imgs_generated % MAX_PROCESS == 0:
                print("{}/{}".format(nb_imgs_generated, args.total_images))
            if nb_imgs_generated % 1000 == 0:
                t1 = time.time()
                print("Time spent in last 1000 images: {:.2f}s".format(t1 - t0_temp))
                t0_temp = t1
        processes = []

    print("Total time: {:.2f}s".format(time.time() - t0))
    print("Generating annotations...")

    with open(os.path.join(args.out_path, "multiclass.csv"), "w") as multi_f:
        multi_f.write("\n".join(multiclass_annotation_lines) + "\n")

    with open(os.path.join(args.out_path, "binary.csv"), "w") as binary_f:
        binary_f.write("\n".join(binary_annotation_lines) + "\n")
