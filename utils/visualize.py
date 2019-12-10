import numpy as np
import cv2


class Debugger(object):
    def __init__(self, labels=None):
        self.imgs = {}
        colors = [(color_list[_]).astype(np.uint8) \
                  for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        self.names = labels

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        self.imgs[img_id] = img.copy()

    def add_mask(self, mask, bg, img_id='default', trans=0.8):
        self.imgs[img_id] = (mask.reshape(
            mask.shape[0], mask.shape[1], 1) * 255 * trans + \
                             bg * (1 - trans)).astype(np.uint8)

    def show_img(self, pause=None, img_id='default', win='default'):
        cv2.imshow('{}'.format(win), self.imgs[img_id])
        if pause:
            cv2.waitKey(pause)
        else:
            cv2.waitKey()

    def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]), interpolation=cv2.INTER_NEAREST)
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    '''
    # slow version
    def gen_colormap(self, img, output_res=None):
      # num_classes = len(self.colors)
      img[img < 0] = 0
      h, w = img.shape[1], img.shape[2]
      if output_res is None:
        output_res = (h * self.down_ratio, w * self.down_ratio)
      color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
      for i in range(img.shape[0]):
        resized = cv2.resize(img[i], (output_res[1], output_res[0]))
        resized = resized.reshape(output_res[0], output_res[1], 1)
        cl = self.colors[i] if not (self.theme == 'white') \
             else 255 - self.colors[i]
        color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
      return color_map
      '''

    def gen_colormap(self, img, output_res=None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
            self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if c == 1:
            colors = np.array([0, 0, 255], dtype=np.float32).reshape(1, 1, 1, 3)
            img = 1 - img
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        if output_res is not None:
            color_map = cv2.resize(color_map, (output_res[0], output_res[1]), interpolation=cv2.INTER_NEAREST)
        return color_map

    def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'):
        bbox = np.array(bbox, dtype=np.int32)
        # cat = (int(cat) + 1) % 80
        cat = int(cat)
        # print('cat', cat, self.names[cat])
        c = self.colors[cat][0][0].tolist()
        txt = '{}{:.1f}'.format(self.names[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(
            self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def add_gt_bbox(self, bbox, show_txt=True, img_id='default', color=(0, 0, 255)):
        cat = int(bbox[0])
        bbox = np.array(bbox[1:], dtype=np.int32)
        txt = '{}'.format(self.names[cat])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(
            self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        if show_txt:
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), color, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def show_all_imgs(self, pause=False, time=0):
        for i, v in self.imgs.items():
            cv2.imshow('{}'.format(i), v)
        if cv2.waitKey(0 if pause else 1) == 27:
            import sys
            sys.exit(0)

    def save_img(self, img_id='default', path='./cache/debug/'):
        cv2.imwrite(path + '{}.png'.format(img_id), self.imgs[img_id])

    def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
        if genID:
            try:
                idx = int(np.loadtxt(path + '/id.txt'))
            except:
                idx = 0
            prefix = idx
            np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

    def remove_side(self, img_id, img):
        if not (img_id in self.imgs):
            return
        ws = img.sum(axis=2).sum(axis=0)
        l = 0
        while ws[l] == 0 and l < len(ws):
            l += 1
        r = ws.shape[0] - 1
        while ws[r] == 0 and r > 0:
            r -= 1
        hs = img.sum(axis=2).sum(axis=1)
        t = 0
        while hs[t] == 0 and t < len(hs):
            t += 1
        b = hs.shape[0] - 1
        while hs[b] == 0 and b > 0:
            b -= 1
        self.imgs[img_id] = self.imgs[img_id][t:b + 1, l:r + 1].copy()

    def add_2d_detection(
            self, img, dets, gts=None, show_txt=True,
            thresh=0.5, img_id='det'):
        self.imgs[img_id] = img
        if gts is not None:
            for g in gts:
                self.add_gt_bbox(g, show_txt=show_txt, img_id=img_id)
        if dets is not None:
            for d in dets:
                if d[-2] > thresh:
                    bbox = d[:4]
                    self.add_coco_bbox(
                        bbox, d[-1], d[-2],
                        show_txt=show_txt, img_id=img_id)


color_list = np.array(
    [
        1.000, 1.000, 1.000,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
