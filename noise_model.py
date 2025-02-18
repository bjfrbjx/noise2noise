import argparse
import os

from freety_cn import put_chinese_text
import string
import random
import numpy as np
import cv2
from PIL import Image
from PIL.Image import Resampling

#fts = [put_chinese_text(f'font/{k}') for k in os.listdir("font")]
fts=[put_chinese_text(f'font/ablibabaPUHUI-M.ttf')]

def get_noise_model(noise_type="gaussian,0,50"):
    tokens = noise_type.split(sep=",")

    if tokens[0] == "gaussian":
        min_stddev = int(tokens[1])
        max_stddev = int(tokens[2])

        def gaussian_noise(img):
            noise_img = img.astype(np.float16)
            stddev = np.random.uniform(min_stddev, max_stddev)
            noise = np.random.randn(*img.shape) * stddev
            noise_img += noise
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
            return noise_img

        return gaussian_noise
    elif tokens[0] == "clean":
        return lambda img: img
    elif tokens[0] == "text2":
        zh_cn_list = []
        with open("3500常用汉字.txt", "r", encoding="utf-8") as f:
            for w in f.readlines():
                zh_cn_list.append(w.strip())

        def shuffled(list):
            random.shuffle(list)
            return list

        def zh_CN(num=1):
            return "".join(random.choice(zh_cn_list) for _ in range(num))

        def add_text(img):
            img = img.copy()
            h, w, _ = img.shape
            k=random.randint(10, 15)
            line_h=h//(k+1)
            for idx in range(0,k,3):
                random_str = ''.join([random.choice(string.printable[:62]) for _ in range(random.randint(3, 5))])
                random_str += zh_CN(random.randint(3, 10))
                random_str = "".join(shuffled(list(random_str)))
                font_scale = random.randint(50,80)
                x = random.randint(0, max(5,w-font_scale*len(random_str)-1))
                y = random.randint(idx*line_h,(idx+1)*line_h)
                color = (0xFF, 0xFF, 0xFF)
                img = random.choice(fts).draw_text(img, (x, y), random_str, font_scale, color)
            return img

        return add_text
    elif tokens[0] == "text":
        min_occupancy = int(tokens[1])
        max_occupancy = int(tokens[2])

        def add_text(img):
            img = img.copy()
            h, w, _ = img.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_for_cnt = np.zeros((h, w), np.uint8)
            occupancy = np.random.uniform(min_occupancy, max_occupancy)

            while True:
                n = random.randint(5, 10)
                random_str = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])
                font_scale = np.random.uniform(0.5, 1)
                thickness = random.randint(1, 3)
                (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
                x = random.randint(0, max(0, w - 1 - fw))
                y = random.randint(fh, h - 1 - baseline)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
                cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)

                if (img_for_cnt > 0).sum() > h * w * occupancy / 1000:
                    break
            return img

        return add_text
    elif tokens[0] == "impulse":
        min_occupancy = int(tokens[1])
        max_occupancy = int(tokens[2])

        def add_impulse_noise(img):
            occupancy = np.random.uniform(min_occupancy, max_occupancy)
            mask = np.random.binomial(size=img.shape, n=1, p=occupancy / 100)
            noise = np.random.randint(256, size=img.shape)
            img = img * (1 - mask) + noise * mask
            return img.astype(np.uint8)

        return add_impulse_noise
    elif tokens[0] == "mark":
        # 水印图所在文件夹
        mark_dir = tokens[1]
        mark_imgs = [Image.open(mark_dir + "/" + mark_file) for mark_file in os.listdir(mark_dir)]

        def paste_mark(img):
            # 复制背景图
            bg = Image.fromarray(img[:, :, ::-1])
            # 随机选一张水印(修订在范围内)
            layer = mark_imgs[random.randint(0, len(mark_imgs) - 1)]
            ori_size=max(layer.width,layer.height)*1.1
            target_size = min(bg.height,bg.width) * 0.9
            if ori_size>target_size:
                layer=layer.resize((int(bg.width*target_size/ori_size),int(bg.height*target_size/ori_size)))
            # 水印随机缩放
            layer_resize = (
                int(layer.size[0] * random.uniform(0.4, 1.5)), int(layer.size[1] * random.uniform(0.4, 1.5)))
            layer = layer.resize(layer_resize, Resampling.LANCZOS)
            layer_arr = np.copy(np.uint8(layer))
            # 水印随机透明度
            layer_arr[:, :, -1] = layer_arr[:, :, -1] * random.uniform(0.8, 1.0)
            # 复制水印图的数组
            layer = Image.fromarray(layer_arr, mode="RGBA")
            # 在背景中的随机位置
            x, y = random.randint(0, bg.size[0] - layer.size[0]), random.randint(0, bg.size[1] - layer.size[1])
            # 水印叠加到底图
            bg.paste(layer, (x, y), layer)
            # RGB->BGR
            return np.uint8(bg.convert(mode="RGB"))[:, :, ::-1]

        return paste_mark
    else:
        raise ValueError("noise_type should be 'gaussian', 'clean', 'mark', 'text', or 'impulse'")


def get_args():
    parser = argparse.ArgumentParser(description="test noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--noise_model", type=str, default="gaussian,0,50",
                        help="noise model to be tested")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_size = args.image_size
    noise_model = get_noise_model(args.noise_model)

    while True:
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128
        noisy_image = noise_model(image)
        cv2.imshow("noise image", noisy_image)
        key = cv2.waitKey(-1)

        # "q": quit
        if key == 113:
            return 0


if __name__ == '__main__':
    main()
