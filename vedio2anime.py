import argparse
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms

from Network import AnimeGenerator
from utils import preprocessing,custom_blur_demo
from adjust_brightness import adjust_brightness_from_src_to_dst

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, default='video/input/'+ '1.mp4',
                        help='video file or number for webcam')
    parser.add_argument('--checkpoint_dir', type=str, default='models/pytorch_generator_shinkai.pt',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--output', type=str, default='video/output/' + 'shinkai',
                        help='output path')
    parser.add_argument('--output_format', type=str, default='MP4V',
                        help='codec used in VideoWriter when saving video to file')
    """
    output_format: xxx.mp4('MP4V'), xxx.mkv('FMP4'), xxx.flv('FLV1'), xxx.avi('XIVD')
    ps. ffmpeg -i xxx.mkv -c:v libx264 -strict -2 xxxx.mp4, this command can convert mkv to mp4, which has small size.
    """

    return parser.parse_args()


def getfileloc(initialdir='/', method='open', title='Please select a file', filetypes=(("video files", ".mkv .avi .mp4"), ("all files","*.*"))):
    root = tk.Tk()
    if method == 'open':
        fileloc = filedialog.askopenfilename(parent=root, initialdir=initialdir, title=title, filetypes=filetypes)
    elif method == 'save':
        fileloc = filedialog.asksaveasfilename(parent=root, initialdir=initialdir, initialfile='out.avi', title=title, filetypes=filetypes)
    root.withdraw()
    return fileloc

def convert_image(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    img = np.asarray(img).transpose(2,0,1)
    img=torch.tensor(img,dtype=torch.float)
    #img = transforms.functional.normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    img=torch.unsqueeze(img,dim=0)
    return img

def inverse_image(img):
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8)
    return img

def cvt2anime_video(video, output, checkpoint_dir, output_format='MP4V', img_size=(256,256)):
    '''
    output_format: 4-letter code that specify codec to use for specific video type. e.g. for mp4 support use "H264", "MP4V", or "X264"
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #test_real = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='test')


    # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    codec = cv2.VideoWriter_fourcc(*output_format)

    with torch.no_grad():
        # tf.global_variables_initializer().run()
        # load model
        G=AnimeGenerator().cuda()  # checkpoint file information
        G.load_state_dict(torch.load(checkpoint_dir))


        # determine output width and height
        ret, img = vid.read()
        if img is None:
            print('Error! Failed to determine frame size: frame empty.')
            return
        img = preprocessing(img, img_size)
        height, width = img.shape[:2]
        # out = cv2.VideoWriter(os.path.join(output, vid_name.replace('mp4','mkv')), codec, fps, (width, height))
        out = cv2.VideoWriter(os.path.join(output, vid_name), codec, fps, (width, height))

        pbar = tqdm(total=total)
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while ret:
            ret, frame = vid.read()
            if frame is None:
                print('Warning: got empty frame.')
                continue

            img = convert_image(frame, img_size).cuda()
            G_out = G(img)
            G_out = (G_out[0] + 1) / 2
            G_out = G_out.cpu().numpy() * 255
            G_out = G_out.astype(np.uint8)
            G_out = G_out.transpose(1, 2, 0)

            # cv2.imshow('image', frame)
            # cv2.imshow('G',G_out[:,:,::-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            G_out = adjust_brightness_from_src_to_dst(G_out, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             
            G_out= custom_blur_demo(G_out)

            out.write(cv2.cvtColor(G_out, cv2.COLOR_BGR2RGB))

            # cv2.imshow('image', frame)
            # cv2.imshow('G',G_out[:,:,::-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            pbar.update(1)

        pbar.close()
        vid.release()
        # cv2.destroyAllWindows()
        return os.path.join(output, vid_name)


if __name__ == '__main__':
    arg = parse_args()
    # if not arg.video:
    #     arg.video = getfileloc(initialdir='input/')
    # else:
    #     arg.video = os.path.join(os.path.dirname(os.path.dirname(__file__)), arg.video)
    # if not arg.output:
    #     arg.output = getfileloc(initialdir='output/', method='save')
    # else:
    #     arg.output = os.path.join(os.path.dirname(os.path.dirname(__file__)), arg.output)
    if not os.path.exists(arg.output):
        os.makedirs(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.checkpoint_dir, output_format=arg.output_format)
    print(f'output video: {info}')