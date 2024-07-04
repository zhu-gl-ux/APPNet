import os, cv2, numpy as np, torch
from torch.nn import functional as F
from PIL import Image
from .base import BaseDataset

class ADE20K(BaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        ignore_label = 255
        self.class_mapping = {0:ignore_label, 
         1:0, 
         2:1, 
         3:2, 
         4:3, 
         5:4, 
         6:5, 
         7:6, 
         8:7, 
         9:8, 
         10:9, 
         11:10, 
         12:11, 
         13:12, 
         14:13, 
         15:14, 
         16:15, 
         17:16, 
         18:17, 
         19:18, 
         20:19, 
         21:20, 
         22:21, 
         23:22, 
         24:23, 
         25:24, 
         26:25, 
         27:26, 
         28:27, 
         29:28, 
         30:29, 
         31:30, 
         32:31, 
         33:32, 
         34:33, 
         35:34, 
         36:35, 
         37:36, 
         38:37, 
         39:38, 
         40:39, 
         41:40, 
         42:41, 
         43:42, 
         44:43, 
         45:44, 
         46:45, 
         47:46, 
         48:47, 
         49:48, 
         50:49, 
         51:50, 
         52:51, 
         53:52, 
         54:53, 
         55:54, 
         56:55, 
         57:56, 
         58:57, 
         59:58, 
         60:59, 
         61:60, 
         62:61, 
         63:62, 
         64:63, 
         65:64, 
         66:65, 
         67:66, 
         68:67, 
         69:68, 
         70:69, 
         71:70, 
         72:71, 
         73:72, 
         74:73, 
         75:74, 
         76:75, 
         77:76, 
         78:77, 
         79:78, 
         80:79, 
         81:80, 
         82:81, 
         83:82, 
         84:83, 
         85:84, 
         86:85, 
         87:86, 
         88:87, 
         89:88, 
         90:89, 
         91:90, 
         92:91, 
         93:92, 
         94:93, 
         95:94, 
         96:95, 
         97:96, 
         98:97, 
         99:98, 
         100:99, 
         101:100, 
         102:101, 
         103:102, 
         104:103, 
         105:104, 
         106:105, 
         107:106, 
         108:107, 
         109:108, 
         110:109, 
         111:110, 
         112:111, 
         113:112, 
         114:113, 
         115:114, 
         116:115, 
         117:116, 
         118:117, 
         119:118, 
         120:119, 
         121:120, 
         122:121, 
         123:122, 
         124:123, 
         125:124, 
         126:125, 
         127:126, 
         128:127, 
         129:128, 
         130:129, 
         131:130, 
         132:131, 
         133:132, 
         134:133, 
         135:134, 
         136:135, 
         137:136, 
         138:137, 
         139:138, 
         140:139, 
         141:140, 
         142:141, 
         143:142, 
         144:143, 
         145:144, 
         146:145, 
         147:146, 
         148:147, 
         149:148, 
         150:149}
        self.ignore_label = ignore_label
        self.label2color = {0:(120, 120, 120), 
         1:(180, 120, 120), 
         2:(6, 230, 230), 
         3:(80, 50, 50), 
         4:(4, 200, 3), 
         5:(120, 120, 80), 
         6:(140, 140, 140), 
         7:(204, 5, 255), 
         8:(230, 230, 230), 
         9:(4, 250, 7), 
         10:(224, 5, 255), 
         11:(235, 255, 7), 
         12:(150, 5, 61), 
         13:(120, 120, 70), 
         14:(8, 255, 51), 
         15:(255, 6, 82), 
         16:(143, 255, 140), 
         17:(204, 255, 4), 
         18:(255, 51, 7), 
         19:(204, 70, 3), 
         20:(0, 102, 200), 
         21:(61, 230, 250), 
         22:(255, 6, 51), 
         23:(11, 102, 255), 
         24:(255, 7, 71), 
         25:(255, 9, 224), 
         26:(9, 7, 230), 
         27:(220, 220, 220), 
         28:(255, 9, 92), 
         29:(112, 9, 255), 
         30:(8, 255, 214), 
         31:(7, 255, 224), 
         32:(255, 184, 6), 
         33:(10, 255, 71), 
         34:(255, 41, 10), 
         35:(7, 255, 255), 
         36:(224, 255, 8), 
         37:(102, 8, 255), 
         38:(255, 61, 6), 
         39:(255, 194, 7), 
         40:(255, 122, 8), 
         41:(0, 255, 20), 
         42:(255, 8, 41), 
         43:(255, 5, 153), 
         44:(6, 51, 255), 
         45:(235, 12, 255), 
         46:(160, 150, 20), 
         47:(0, 163, 255), 
         48:(140, 140, 140), 
         49:(250, 10, 15), 
         50:(20, 255, 0), 
         51:(31, 255, 0), 
         52:(255, 31, 0), 
         53:(255, 224, 0), 
         54:(153, 255, 0), 
         55:(0, 0, 255), 
         56:(255, 71, 0), 
         57:(0, 235, 255), 
         58:(0, 173, 255), 
         59:(31, 0, 255), 
         60:(11, 200, 200), 
         61:(255, 82, 0), 
         62:(0, 255, 245), 
         63:(0, 61, 255), 
         64:(0, 255, 112), 
         65:(0, 255, 133), 
         66:(255, 0, 0), 
         67:(255, 163, 0), 
         68:(255, 102, 0), 
         69:(194, 255, 0), 
         70:(0, 143, 255), 
         71:(51, 255, 0), 
         72:(0, 82, 255), 
         73:(0, 255, 41), 
         74:(0, 255, 173), 
         75:(10, 0, 255), 
         76:(173, 255, 0), 
         77:(0, 255, 153), 
         78:(255, 92, 0), 
         79:(255, 0, 255), 
         80:(255, 0, 245), 
         81:(255, 0, 102), 
         82:(255, 173, 0), 
         83:(255, 0, 20), 
         84:(255, 184, 184), 
         85:(0, 31, 255), 
         86:(0, 255, 61), 
         87:(0, 71, 255), 
         88:(255, 0, 204), 
         89:(0, 255, 194), 
         90:(0, 255, 82), 
         91:(0, 10, 255), 
         92:(0, 112, 255), 
         93:(51, 0, 255), 
         94:(0, 194, 255), 
         95:(0, 122, 255), 
         96:(0, 255, 163), 
         97:(255, 153, 0), 
         98:(0, 255, 10), 
         99:(255, 112, 0), 
         100:(143, 255, 0), 
         101:(82, 0, 255), 
         102:(163, 255, 0), 
         103:(255, 235, 0), 
         104:(8, 184, 170), 
         105:(133, 0, 255), 
         106:(0, 255, 92), 
         107:(184, 0, 255), 
         108:(255, 0, 31), 
         109:(0, 184, 255), 
         110:(0, 214, 255), 
         111:(255, 0, 112), 
         112:(92, 255, 0), 
         113:(0, 224, 255), 
         114:(112, 224, 255), 
         115:(70, 184, 160), 
         116:(163, 0, 255), 
         117:(153, 0, 255), 
         118:(71, 255, 0), 
         119:(255, 0, 163), 
         120:(255, 204, 0), 
         121:(255, 0, 143), 
         122:(0, 255, 235), 
         123:(133, 255, 0), 
         124:(255, 0, 235), 
         125:(245, 0, 255), 
         126:(255, 0, 122), 
         127:(255, 245, 0), 
         128:(10, 190, 212), 
         129:(214, 255, 0), 
         130:(0, 204, 255), 
         131:(20, 0, 255), 
         132:(255, 255, 0), 
         133:(0, 153, 255), 
         134:(0, 41, 255), 
         135:(0, 255, 204), 
         136:(41, 0, 255), 
         137:(41, 255, 0), 
         138:(173, 0, 255), 
         139:(0, 245, 255), 
         140:(71, 0, 255), 
         141:(122, 0, 255), 
         142:(0, 255, 184), 
         143:(0, 92, 255), 
         144:(184, 255, 0), 
         145:(0, 133, 255), 
         146:(255, 214, 0), 
         147:(25, 194, 194), 
         148:(102, 255, 0), 
         149:(92, 0, 255), 
         255:(0, 0, 0)}
        self.label_reading_mode = cv2.IMREAD_GRAYSCALE
    def image2class(self, label):
        """Overwrite the parent class to convert grayscale label image"""
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype=(np.uint8))
        for k, v in self.class_mapping.items():
            classmap[label == k] = v

        return classmap