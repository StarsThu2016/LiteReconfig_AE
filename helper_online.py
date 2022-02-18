import pickle, torch, torchvision, socket, bisect, time, cv2, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import numpy as np
# sklearn hack since our weight is dumped with another sklearn version 0.23.2
import sklearn.linear_model.base
sys.modules['sklearn.linear_model._base'] = sklearn.linear_model.base
import sklearn.linear_model
import sklearn.tree
from sklearn.preprocessing import PolynomialFeatures
import torch.nn as nn
from torchvision import transforms
import tensorflow as tf

def get_config_list(FRCNN_only=True):

    # Get all 1036 FRCNN-based ABs, (si, shape, nprop, tracker, ds)
    list_sis = [1, 2, 4, 8, 20, 50, 100]
    list_fshapes = [224, 320, 448, 576]
    list_nprops = [1, 3, 5, 10, 20, 50, 100]
    list_trackers = [("medianflow", 4), ("medianflow", 2), ("medianflow", 1),
                     ("kcf", 4), ("csrt", 4), ("bboxmedianfixed", 4)]
    ker_sh_np = [(s, n) for s in list_fshapes for n in list_nprops]
    if not FRCNN_only:
        list_yshapes = list(range(224, 577, 32))
        ker_sh_np += [(s, -1) for s in list_yshapes]
    config_list = [(1, s, n, '', -1) for s, n in ker_sh_np]
    config_list += [(si, s, n, tracker, ds) for si in list_sis[1:] \
                    for s, n in ker_sh_np for tracker, ds in list_trackers]
    return config_list

def make_actv(actv):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'prelu':
        return nn.PReLU(init=0.2)
    elif actv == 'elu':
        return nn.ELU(inplace=True)
    elif actv is None:
        return nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid activation: {:s}'.format(actv)
        )

def make_norm(norm, dim):
    if norm == 'layer':
        return nn.LayerNorm(dim)
    elif norm is None:
        return nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid normalization: {:s}'.format(norm)
        )

class FiLMModel(nn.Module):

    def __init__(self, feats_dim, resrc_dim, hid_dim, film_dim, out_dim,
                             norm=None, actv='leaky_relu', dropout=0):
        super(FiLMModel, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(feats_dim, hid_dim), 
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim), 
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim), 
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
        )

        self.film = nn.Sequential(
            nn.Linear(resrc_dim, film_dim), 
            make_norm(norm, film_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(film_dim, film_dim), 
            make_norm(norm, film_dim), make_actv(actv), nn.Dropout(dropout),
            nn.Linear(film_dim, hid_dim * 4),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), 
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), 
            make_norm(norm, hid_dim), make_actv(actv), nn.Dropout(dropout),
        )
        self.out_fc = nn.Linear(hid_dim, out_dim)

    def forward(self, feats, resrc):
        x = self.stem(feats)
        h = self.film(resrc)
        g1, b1, g2, b2 = h.split(h.size(-1) // 4, -1)
        x = self.fc1(g1 * x + b1)
        x = self.fc2(g2 * x + b2)
        x = self.out_fc(x)
        return x

class FeatureToVecOneHeadOnline():

    def __init__(self, filename, mask=np.ones((1036,)).astype(bool), tv_version=None):

        torch.set_default_dtype(torch.float32)
        self.mask = mask
        self.model_acc = FiLMModel(1284, 22, 256, 128, 200, 'layer', 'leaky_relu', 0)
        if not tv_version:
            socket_name = socket.gethostname()
            version = {"xv3": "0.8.1", "tx2-1": "0.5.0", "tx2-2": "0.5.0"}[socket_name]
        else:
            version = tv_version
        self.model_fe = torch.hub.load('pytorch/vision:v{}'.format(version),
                                       'mobilenet_v2', pretrained=True)
        self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]
                            ),
                        ])
        mydict = torch.load(filename)
        self.model_acc.load_state_dict(mydict["net"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_acc.to(self.device)
        self.model_fe.to(self.device)
        self.model_fe.eval()
        self.model_acc.eval()

        # ordinal encoding for latency requirements
        self.lat_levels = {
            33.3: [0, 0, 0, 1],
            50:   [0, 0, 1, 1],
            100:  [0, 1, 1, 1],
            200:  [1, 1, 1, 1],
        }

        # ordinal encoding for CPU contention levels
        self.cpu_levels = {
            0: [0, 0, 0, 0, 0, 0, 1],
            1: [0, 0, 0, 0, 0, 1, 1],
            2: [0, 0, 0, 0, 1, 1, 1],
            3: [0, 0, 0, 1, 1, 1, 1],
            4: [0, 0, 1, 1, 1, 1, 1],
            5: [0, 1, 1, 1, 1, 1, 1],
            6: [1, 1, 1, 1, 1, 1, 1],
        }

        # ordinal encoding for GPU contention levels
        self.gpu_levels = {
            0:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            1:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            10: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            20: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            30: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            40: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            50: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            60: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            70: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            80: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            90: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        }

    def _encode_lat_req(self, lat_req):

        return np.array([self.lat_levels[l] for l in lat_req])  # [n, 4]

    def _encode_cpu_cont(self, cpu_cont):

        return np.array([self.cpu_levels[c] for c in cpu_cont]) # [n, 7]

    def _encode_gpu_cont(self, gpu_cont):

        return np.array([self.gpu_levels[g] for g in gpu_cont]) # [n, 11]

    def fe(self, img_pil):
        
        # (1, 3, 224, 224) torch array
        img_pre = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        # (1280,) torch array
        feature_heavy = torch.mean(self.model_fe.features(img_pre)[0], dim=(1, 2))
        # (1280,) numpy array
        feature_heavy = feature_heavy.cpu().detach().numpy()
        return feature_heavy

    def predict(self, feature, lat_req, cpu_cont, gpu_cont):

        # lat_req, cpu_cont, gpu_cont are (batch_size,) numpy arrays
        # output is a (1036,) numpy array
        feature = torch.from_numpy(np.array(feature)).float().to(self.device)  # [n, 4+1280]
        lat_req = torch.from_numpy(self._encode_lat_req(lat_req)).float().to(self.device)
        cpu_cont = torch.from_numpy(self._encode_cpu_cont(cpu_cont)).float().to(self.device)
        gpu_cont = torch.from_numpy(self._encode_gpu_cont(gpu_cont)).float().to(self.device)
        resrc = torch.cat([lat_req, cpu_cont, gpu_cont], dim=1) # [n, 22]
        output = self.model_acc.forward(feature, resrc)   # [n,4+1280] Tensor --> [n,200] Tensor
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = np.exp(output)
        return output_ret

class BaselineAccuracyPredictorOnline:

    def __init__(self, filename):
        
        self.accuracy = pickle.load(open(filename, "rb"))

    def predict(self):

        return self.accuracy

class NN_residual(torch.nn.Module): 

    # Architecture of the accuracy predictor
    #   first project (light, heavy) features to (256, 256) shape, 
    #   then concatenate and proceed with 5-layer fully connected layers
    #   1s in the mask determines the number of branches we predict on
    def __init__(self, input_dim, mask=np.ones((1036,)).astype(bool)):

        # dim = 4, #neurons = [4] --> [256, 256, 1036]
        # dim = X, #neurons = [4, X] --> [256+256=256, 256, 256, 256, 256, 1036], X = {768, 5400, 1280}
        super(NN_residual, self).__init__()
        self.input_dim = input_dim
        self.N_branch = sum(mask)
        if input_dim == 4:
            self.project0 = torch.nn.Linear(4, 256)          # [n,4]     -> [n,256]
            self.fc1 = torch.nn.Linear(256, 256)             # [n,256]   -> [n,256]
            self.fc2 = torch.nn.Linear(256, self.N_branch)   # [n,256]   -> [n,self.N_branch]
        else:
            self.project0 = torch.nn.Linear(4, 256)
            self.project1 = torch.nn.Linear(input_dim-4, 256)
            self.fc1 = torch.nn.Linear(256, 256)             # [n,256]   -> [n,256]
            self.fc2 = torch.nn.Linear(256, 256)             # [n,256]   -> [n,256]
            self.fc3 = torch.nn.Linear(256, 256)             # [n,256]   -> [n,256]
            self.fc4 = torch.nn.Linear(256, 256)             # [n,256]   -> [n,256]
            self.fc5 = torch.nn.Linear(256, self.N_branch)   # [n,256]   -> [n,self.N_branch]

    def forward(self, feature):

        # dim = 4, activation = relu, sigmoid
        # dim = X, activation = relu, relu, relu, relu, sigmoid
        x = feature.double()
        if self.input_dim == 4:
            x = torch.nn.functional.relu(self.project0(x))       # [n,4]             -> [n,256]
            x = x + torch.nn.functional.relu(self.fc1(x))        # [n,256] + [n,256] -> [n,256]
            x = torch.sigmoid(self.fc2(x))                       # [n,256]           -> [n,self.N_branch]
        else:
            xl = x[:, 0:4] 
            xl = torch.nn.functional.relu(self.project0(xl))     # [n,4]             -> [n,256]
            xh = x[:, 4:self.input_dim]
            xh = torch.nn.functional.relu(self.project1(xh))     # [n,X]             -> [n,256]
            x = xl + xh                                          # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc1(x))        # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc2(x))        # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc3(x))        # [n,256] + [n,256] -> [n,256]
            x = x + torch.nn.functional.relu(self.fc4(x))        # [n,256] + [n,256] -> [n,256]
            x = torch.sigmoid(self.fc5(x))                       # [n,256]           -> [n,self.N_branch]
        return x

class FeatureToVecOnline():

    def __init__(self, feature, filename, mask):

        torch.set_default_dtype(torch.float64)
        input_dim = {"light": 4, "HoC": 4+768, "HoG": 4+5400, # "MobileNetV2": 4+62720,
                     "MobileNetV2Pool": 4+1280, "RPN": 4+1024, "CPoP": 4+31}[feature]
        self.model = NN_residual(input_dim, mask=mask)
        mydict = torch.load(filename)
        self.model.load_state_dict(mydict["model"])
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.mask = mask

    def predict(self, feature):

        # input feature is a (batch_size, feature_dim) numpy array
        # output accuracy prediction is a (batch_size, 1036) numpy array
        feature = torch.from_numpy(feature).double().to(device=self.device)
        output = self.model.forward(feature)
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = output
        return output_ret

class FeatureToVecJointOnline():

    def __init__(self, filename, mask, trainable_fe, tv_version=None):

        torch.set_default_dtype(torch.float64)
        self.mask = mask
        input_dim = 4+1280  # Support "MobileNetV2Pool" only
        self.model_acc = NN_residual(input_dim, mask=mask)
        if not tv_version:
            socket_name = socket.gethostname()
            version = {"xv3": "0.8.1", "tx2-1": "0.5.0", "tx2-2": "0.5.0"}[socket_name]
        else:
            version = tv_version
        self.model_fe = torch.hub.load('pytorch/vision:v{}'.format(version),
                                       'mobilenet_v2', pretrained=True)

        mydict = torch.load(filename)
        if trainable_fe:
            self.model_acc.load_state_dict(mydict["model_acc"])
            self.model_fe.load_state_dict(mydict["model_fe"])
        else:
            self.model_acc.load_state_dict(mydict["model"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_acc.to(self.device)
        self.model_fe.to(self.device)
        self.preprocess = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),
                          ])

    def fe(self, img_pil):

        # (1, 3, 224, 224) torch array
        img_pre = self.preprocess(img_pil).double().unsqueeze(0).to(self.device)
        # (1280,) torch array
        feature_heavy = torch.mean(self.model_fe.features(img_pre)[0], dim=(1, 2))
        # (1280,) numpy array
        feature_heavy = feature_heavy.cpu().detach().numpy()
        return feature_heavy

    def predict(self, feature):

        # output_ret is a (1036,) numpy array
        feature = torch.from_numpy(np.array(feature)).double().to(self.device)  # [n, 4+1280]
        output = self.model_acc.forward(feature)   # [n,4+1280] Tensor --> [n,1036] Tensor
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = output
        return output_ret

class TrackerLatencyPredictor:

    def __init__(self, model_file):

        self.list_tracker = [("medianflow", 4), ("medianflow", 2),
                             ("medianflow", 1), ("kcf", 4),
                             ("csrt", 4), ("bboxmedianfixed", 4)]
        self.sis = [1, 2, 4, 8, 20, 50, 100]
        self.transform = PolynomialFeatures(2)
        self.init_models, self.tr_models = {}, {}
        self.init_coeff, self.init_intercept = [], []
        self.tracking_coeff, self.tracking_intercept = [], []

        all_models = pickle.load(open(model_file, 'rb'))
        for tracker, ds in self.list_tracker:
            key = "{}_ds{}_init".format(tracker, ds)
            if key in all_models:
                self.init_models[(tracker, ds)] = all_models[key]
                self.init_coeff.append(all_models[key].coef_)
                self.init_intercept.append(all_models[key].intercept_)
            else:
                print("Error in loading latency prediction model.")
                return

            key = "{}_ds{}_tracking".format(tracker, ds)
            if key in all_models:
                self.tr_models[(tracker, ds)] = all_models[key]
                self.tracking_coeff.append(all_models[key].coef_)
                self.tracking_intercept.append(all_models[key].intercept_)
            else:
                print("Error in loading latency prediction model.")
                return

        self.init_coeff = np.array(self.init_coeff)
        self.init_intercept = np.array(self.init_intercept)
        self.tracking_coeff = np.array(self.tracking_coeff)
        self.tracking_intercept = np.array(self.tracking_intercept)

    # faster implementation
    def batch_prediction(self, num_obj, avg_size, width, height, core=0):

        feature = [num_obj, avg_size, width, height, core]
        X = np.array(feature).reshape(1, -1)  # (1, 5) shape
        X = self.transform.fit_transform(X).squeeze()
        init_time = np.dot(self.init_coeff, X) + self.init_intercept
        tr_time = np.dot(self.tracking_coeff, X) + self.tracking_intercept
        return init_time, tr_time  # both in (6,) shape

    def predict(self, num_obj, avg_size, width, height, core=0):

        # latency_init0 = [self.predict_init(num_obj, avg_size, width, height,
        #                                  core, tracker, ds) \
        #                for (tracker, ds) in self.list_tracker]  # (6,) shape
        # latency_tr0 = [self.predict_tr(num_obj, avg_size, width, height,
        #                              core, tracker, ds) \
        #              for (tracker, ds) in self.list_tracker]  # (6,) shape
        latency_init, latency_tr = self.batch_prediction(num_obj, avg_size,
          width, height, core=core)

        # Construct the tracker latency array if si > 1, in (1440,) shape
        lat_init, lat_tr = np.zeros((6, 40, 6)), np.zeros((6, 40, 6))
        for idx in range(6):
            lat_init[:, :, idx] = latency_init[idx]
            lat_tr[:, :, idx] = latency_tr[idx]
        for idx, si in enumerate(self.sis[1:]):
            lat_init[idx, :, :] *= (1/si)
            lat_tr[idx, :, :] *= ((si-1)/si)
        detector_only = np.zeros((40,))
        treacker_latency = (lat_init + lat_tr).flatten()
        return np.concatenate((detector_only, treacker_latency))  # (1480,) shape

class DNNLatencyPredictor:

    def __init__(self, model_file, version="v2b"):

        if version == "v2b":  # baseline v2: LUT for each s, n, h, w, g
            def filter_det(lat):
                while len(lat) > 3:
                    mean, std = np.mean(lat), np.std(lat)
                    new_lat = [l for l in lat if abs(l-mean)<=1.5*std]
                    if len(new_lat) == len(lat):
                        lat = new_lat
                        break
                    else:
                        lat = new_lat
                return lat

            self.version = version
            self.fshapes = [224, 320, 448, 576]
            self.nprops = [1, 3, 5, 10, 20, 50, 100]
            self.yshapes = list(range(224, 577, 32))
            self.sis = [1, 2, 4, 8, 20, 50, 100]
            self.ker_sh_np = [('FRCNN', s, n) for s in self.fshapes for n in self.nprops]
            self.ker_sh_np += [('YOLO', s, -1) for s in self.yshapes]
            # map any (h, w) in val/test to 15 profiled (h, w) in val
            self.hwm = {(576, 1280): (720, 1280), (360, 480): (360, 480),
                        (360, 608): (360, 640), (358, 480): (360, 480),
                        (180, 320): (240, 320), (720, 1270): (720, 1280),
                        (720, 1280): (720, 1280), (144, 192): (240, 320),
                        (360, 636): (360, 640), (480, 872): (480, 640),
                        (480, 640): (480, 640), (352, 640): (358, 640),
                        (240, 426): (240, 426), (360, 376): (360, 450),
                        (720, 1278): (720, 1280), (424, 640): (424, 640),
                        (358, 640): (358, 640), (360, 490): (360, 480),
                        (304, 540): (320, 568), (270, 480): (270, 480),
                        (1080, 1920): (1080, 1920), (180, 240): (240, 320),
                        (360, 472): (360, 480), (360, 600): (360, 640),
                        (360, 524): (360, 540), (360, 534): (360, 540),
                        (320, 568): (320, 568), (720, 960): (720, 960),
                        (288, 512): (270, 480), (288, 384): (240, 426),
                        (360, 450): (360, 450), (360, 492): (360, 480),
                        (264, 396): (240, 426), (360, 640): (360, 640),
                        (264, 480): (270, 480), (360, 426): (360, 450),
                        (488, 640): (480, 640), (720, 406): (720, 406),
                        (240, 320): (240, 320), (816, 1920): (1080, 1920),
                        (160, 208): (240, 320), (360, 540): (360, 540)}

            data = pickle.load(open(model_file, "rb"))
            self.LUT, self.gl_seen = {}, set()
            for key, arr in data.items():
                dataset, kernel, shape, nprop, height, width, gl = key
                if dataset == 'test':
                    continue
                self.gl_seen.add(gl)
                arr = filter_det(arr)
                self.LUT[(kernel, shape, nprop, height, width, gl)] = np.mean(arr)

            # Construct cache to speed up
            self.gs = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 99 not allowed
            self.cache = {}
            for height, width in self.hwm:
                for g in self.gs:
                    self.cache[(height, width, g)] = self.predict(height, width, 0, g)

        elif version == 'v2':
            self.version = version
            self.fshapes = [224, 320, 448, 576]
            self.nprops = [1, 3, 5, 10, 20, 50, 100]
            self.yshapes = list(range(224, 577, 32))
            self.sis = [1, 2, 4, 8, 20, 50, 100]
            model = pickle.load(open(model_file, "rb"))
            frcnn_model = model['FRCNN']
            yolo_model = model['YOLO']
            self.frcnn_coeff_array = np.array([frcnn_model[(n, s)][0] for s in self.fshapes for n in self.nprops])
            self.frcnn_bias_array = np.array([frcnn_model[(n, s)][1] for s in self.fshapes for n in self.nprops])
            self.yolo_coeff_array = np.array([yolo_model[(n, s)][0] for s in self.yshapes for n in [-1]])
            self.yolo_bias_array = np.array([yolo_model[(n, s)][1] for s in self.yshapes for n in [-1]])
            self.coeff_array = np.concatenate((self.frcnn_coeff_array,self.yolo_coeff_array),axis=0)
            self.bias_array = np.concatenate((self.frcnn_bias_array,self.yolo_bias_array),axis=0)
            self.transform = PolynomialFeatures(2)        # (4,) -> (15,)

        else:  # v1: contains model for FRCNN only
            self.version = version
            model = pickle.load(open(model_file, "rb"))
            self.coeff_array = [model[(n, s)][0] for s in shapes for n in nprops]
            self.bias_array = [model[(n, s)][1] for s in shapes for n in nprops]
            self.coeff_array = np.array(self.coeff_array) # (28, 15)
            self.bias_array = np.array(self.bias_array)   # (28,)
            self.transform = PolynomialFeatures(2)        # (4,) -> (15,)
            self.sis = sis

    def predict(self, height, width, cpu_contention=0, gpu_contention=0):

        if self.version == 'v2b' and (height, width, gpu_contention) in self.cache:
            return self.cache[(height, width, gpu_contention)]
        if self.version == "v2b":  # baseline v2: LUT for each s, n, h, w, g
            if (height, width) in self.hwm:
                height, width = self.hwm[(height, width)]
            else:
                height, width = 720, 1280
            if not gpu_contention in self.gl_seen:
                print("gpu_contention not seen in offline data, set it to 90.")
                gpu_contention = 90
            ans_det = [self.LUT[(k, s, n, height, width, gpu_contention)] \
                       for k, s, n in self.ker_sh_np]  # (40,) list
            ans_det = np.array(ans_det)  # (40,) np
            ans = np.array([ans_det/si for si in self.sis[1:]])  # (6, 40) np
            ans = np.repeat(ans[:, :, np.newaxis], 6, axis=2)  # (6, 40, 6) np
            return np.concatenate((ans_det.flatten(), ans.flatten()))  # (1480,) shape
        elif self.version == 'v2':
            features = [height, width, gpu_contention]  # (3,)
            features = np.array(features).reshape(1, -1)
            features = self.transform.fit_transform(features).squeeze()  # (10,)
            ans_det = np.dot(self.coeff_array, features) + self.bias_array
            #now_time = now_time.reshape(4, 7)  # (shape, nprop), arranged (4, 7) shape
            # (si, shape, nprop) arranged , (7, 4, 7) shape
            ans = np.array([ans_det / si for si in self.sis[1:]])  # (6, 40) shape
            ans = np.repeat(ans[:, :, np.newaxis], 6, axis=2)  # (6, 40, 6) shape
            # (si, shape, nprop, tracker) arranged, (7, 4, 7, 6) shape
            return np.concatenate((ans_det.flatten(), ans.flatten()))  # (1480,) shape
        else:  # v1: contains model for FRCNN only
            features = [height, width, cpu_contention, gpu_contention] # (4,)
            features = np.array(features).reshape(1, -1)
            features = self.transform.fit_transform(features).squeeze() # (15,)
            now_time = np.dot(self.coeff_array, features) + self.bias_array
            now_time = now_time.reshape(4, 7) # (shape, nprop), arranged (4, 7) shape
            #(si, shape, nprop) arranged , (7, 4, 7) shape
            with_ds_array = np.stack([now_time/self.sis[i] for i in range(7)], axis=0)
            #(si, shape, nprop, tracker) arranged, (7, 4, 7, 6) shape
            final_array = np.repeat(with_ds_array[:, :, :, np.newaxis], 6, axis=3)
            return final_array.flatten()

class LatencyPredictor:

    def __init__(self, dlp_model="models/ApproxDet_LatDet_1228.pb",
                       tlp_model="models/ApproxDet_LatTr_1227.pb"):

        self.dlp = DNNLatencyPredictor(model_file=dlp_model, version='v2b')
        self.tlp = TrackerLatencyPredictor(model_file=tlp_model)

    def convert1480to1036(self, vec1480):

        # convert a (1480,) latency/accuracy np array to a (1036,) one
        vec40, vec1440 = vec1480[:40], vec1480[40:]
        vec6_40_6 = vec1440.reshape(6, 40, 6)
        vec6_28_6 = vec6_40_6[:, :28, :]
        vec28 = vec40[:28]
        vec1036 = np.concatenate((vec28, vec6_28_6.flatten()))
        return vec1036

    def predict(self, height=720, width=1280, nobj=1, objsize=220, cl=0, gl=0, FRCNN_only=True):

        per_branch_DNN_latency = self.dlp.predict(height=height, width=width, gpu_contention=gl)
        per_branch_tracker_latency = self.tlp.predict(nobj, objsize, width, height, core=cl)
        per_branch_latency = per_branch_DNN_latency + per_branch_tracker_latency
        if FRCNN_only:  # convert the (1480,) array to a (1036,) one
            per_branch_latency = self.convert1480to1036(per_branch_latency)
        return per_branch_latency

class FeatureExtractorOnline:

    def __init__(self, feature_name):

        self.feature_name = feature_name
        if feature_name == 'HoG':
            self.extractor = self.hog_extractor
        elif feature_name == 'HoC':
            self.extractor = self.hoc_extractor

    def hog_extractor(self, input_image):

        winSize = (320, 480)
        input_image = cv2.resize(input_image,winSize)
        blockSize = (80, 80)  # 105
        blockStride = (80, 80)
        cellSize = (16, 16)
        Bin = 9  # 3780
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, Bin)
        return hog.compute(input_image)[:, 0]

    def hoc_extractor(self, input_image):

        # input_image: (h, w, 3) numpy image in BGR order
        # output: 768 x 1 dimension vector
        h, w, _ = input_image.shape
        hist_b = cv2.calcHist([input_image], [0], None, [256], [0, 255])
        hist_b = hist_b / np.linalg.norm(hist_b, ord=2)
        hist_g = cv2.calcHist([input_image], [1], None, [256], [0, 255])
        hist_g = hist_g / np.linalg.norm(hist_g, ord=2)
        hist_r = cv2.calcHist([input_image], [2], None, [256], [0, 255])
        hist_r = hist_r / np.linalg.norm(hist_r, ord=2)
        return np.concatenate((hist_b, hist_g, hist_r), axis=0)[:, 0]

    def extract(self, input_image):

        return self.extractor(input_image)

class SchedulerOnline:

    # Supported protocols:
    # SmartAdapt_BL, SmartAdapt_MN2_1head_Top200
    # SmartAdapt_Lite, SmartAdapt_HoC, SmartAdapt_RPN, SmartAdapt_CPoP, SmartAdapt_MN2, SmartAdapt_MN2_joint
    # SmartAdapt_Lite_Top200, SmartAdapt_HoC_Top200, SmartAdapt_RPN_Top200
    # SmartAdapt_CPoP_Top200, SmartAdapt_MN2_Top200, SmartAdapt_MN2_joint_Top200
    def __init__(self, user_requirement=200, p95_requirement=True, protocol=None, mobile_device="tx2",
                 tv_version=None, dataset_prefix=None):

        prot2feat = {
            "SmartAdapt_BL": None,
            "SmartAdapt_Lite": "light", 
            "SmartAdapt_HoC": "HoC", 
            "SmartAdapt_HoG": "HoG", 
            "SmartAdapt_RPN": "RPN", 
            "SmartAdapt_CPoP": "CPoP", 
            "SmartAdapt_MN2": "MobileNetV2Pool",
            "SmartAdapt_MN2_joint": None,
            "SmartAdapt_Lite_Top200": "light", 
            "SmartAdapt_HoC_Top200": "HoC", 
            "SmartAdapt_HoG_Top200": "HoG", 
            "SmartAdapt_RPN_Top200": "RPN", 
            "SmartAdapt_CPoP_Top200": "CPoP", 
            "SmartAdapt_MN2_Top200": "MobileNetV2Pool",
            "SmartAdapt_MN2_joint_Top200": None,
            "SmartAdapt_MN2_1head_Top200": None,
            "SmartAdapt_MN2_1head_Top200_merge": None,
        }
        self.feat = prot2feat[protocol]
        self.protocol = protocol
        self.user_requirement = user_requirement
        self.latency_target = user_requirement/(1+0.3) if p95_requirement else user_requirement
        self.config_list = get_config_list(FRCNN_only=True)
        self.default_config = (100, 224, 1, 'medianflow', 4)  # (si, shape, nprop, tracker, ds)
        self.default_config_idx = self.config_list.index(self.default_config)
        if not dataset_prefix:
            socket_name = socket.gethostname()
            self.dataset_prefix = {
                "tx2-1": "/home/nvidia/sdcard/ILSVRC2015",
                "tx2-2": "/home/tx2/data/ILSVRC2015",
                "xv3": "/media/A833-362D/ILSVRC2015",
            }[socket_name]
        else:
            self.dataset_prefix = dataset_prefix

        # Load the masks      
        if protocol in ["SmartAdapt_Lite_Top200", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200", 
                        "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200", "SmartAdapt_MN2_Top200", 
                        "SmartAdapt_MN2_joint_Top200", "SmartAdapt_MN2_1head_Top200", "SmartAdapt_MN2_1head_Top200_merge"]:
            self.mask = pickle.load(open("models/mask_top200.pb", "rb"))  
        else:
            self.mask = np.ones((1036,)).astype(bool)

        # Initialize the accuracy predictor
        if protocol == "SmartAdapt_BL":
            self.accuracy_predictor = BaselineAccuracyPredictorOnline("models/SmartAdapt_BL.pb")
        elif protocol in ["SmartAdapt_Lite", "SmartAdapt_HoC", "SmartAdapt_HoG", 
                          "SmartAdapt_RPN", "SmartAdapt_CPoP",
                          "SmartAdapt_Lite_Top200", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200", 
                          "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200"]:
            filename = "models/{}.pb".format(protocol)
            self.accuracy_predictor = FeatureToVecOnline(feature=self.feat, filename=filename, mask=self.mask)
        elif protocol in ["SmartAdapt_MN2", "SmartAdapt_MN2_Top200", "SmartAdapt_MN2_joint", "SmartAdapt_MN2_joint_Top200"]:
            trainable_fe = (protocol in ["SmartAdapt_MN2_joint", "SmartAdapt_MN2_joint_Top200"])
            filename = "models/{}.pb".format(protocol)
            self.accuracy_predictor = FeatureToVecJointOnline(filename=filename, mask=self.mask, trainable_fe=trainable_fe, tv_version=tv_version)
        elif protocol in ["SmartAdapt_MN2_1head_Top200", "SmartAdapt_MN2_1head_Top200_merge"]:
            self.accuracy_predictor = FeatureToVecOneHeadOnline(filename="models/{}.pb".format(protocol), mask=self.mask, tv_version=tv_version)
        else:
            print("ERROR: protocol not supported in init()")
            return 

        # Initialize the HoC/HoG feature extractor
        if protocol in ["SmartAdapt_HoC", "SmartAdapt_HoG", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200"]:
            self.feature_extractor = FeatureExtractorOnline(feature_name=self.feat)

        # Initialize the latency predictor
        if not protocol == "SmartAdapt_MN2_1head_Top200":
            if mobile_device == "tx2":
                self.latency_predictor = LatencyPredictor()
            else:
                self.latency_predictor = LatencyPredictor(
                  dlp_model="models/ApproxDet_LatDet_0314_xv.pb",
                  tlp_model="models/ApproxDet_LatTr_0314_xv.pb",
                )
    
    def schedule(self, video_dir, video_idx, frame_idx, feature_cache, contention_levels):

        # Load a video frame
        time1 = time.time()
        video_frame_path = os.path.join(video_dir, "{:06d}.JPEG".format(frame_idx))
        full_path = os.path.join(self.dataset_prefix, video_frame_path)
        image_pil = Image.open(full_path)
        time2 = time.time()

        # Extract light feature
        width, height = image_pil.size
        nobj, objsize = feature_cache["nobj"], feature_cache["objsize"]
        feature = [height, width, nobj, objsize]

        # Extract heavy feature
        if self.protocol in ["SmartAdapt_BL", "SmartAdapt_Lite", "SmartAdapt_Lite_Top200"]:
            heavy_features = []
        elif self.protocol in ["SmartAdapt_HoC", "SmartAdapt_HoG", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200"]:
            heavy_features = list(self.feature_extractor.extract(np.array(image_pil)))
        elif self.protocol in ["SmartAdapt_RPN", "SmartAdapt_CPoP", "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200"]:
            heavy_features = list(feature_cache[self.feat])
        elif self.protocol in ["SmartAdapt_MN2", "SmartAdapt_MN2_joint", "SmartAdapt_MN2_Top200",
                               "SmartAdapt_MN2_joint_Top200", "SmartAdapt_MN2_1head_Top200",
                               "SmartAdapt_MN2_1head_Top200_merge"]:
            heavy_features = list(self.accuracy_predictor.fe(image_pil))
        feature += heavy_features
        feature = np.expand_dims(np.array(feature), axis=0)  # (1, 4+dim) numpy array
        time3 = time.time()

        if self.protocol == "SmartAdapt_BL":
            per_branch_accuracy = self.accuracy_predictor.predict()
        elif self.protocol in ["SmartAdapt_Lite", "SmartAdapt_Lite_Top200",
                          "SmartAdapt_HoC", "SmartAdapt_HoG", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200",
                          "SmartAdapt_RPN", "SmartAdapt_CPoP", "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200",
                          "SmartAdapt_MN2", "SmartAdapt_MN2_joint", "SmartAdapt_MN2_Top200", "SmartAdapt_MN2_joint_Top200"]:
            per_branch_accuracy = self.accuracy_predictor.predict(feature)
        elif self.protocol in ["SmartAdapt_MN2_1head_Top200", "SmartAdapt_MN2_1head_Top200_merge"]:
            lat_req = np.array([self.user_requirement])
            cpu_cont = np.array([contention_levels["cpu_level"]])
            gpu_cont = np.array([contention_levels["gpu_level"]])
            per_branch_accuracy = self.accuracy_predictor.predict(feature, lat_req, cpu_cont, gpu_cont)
        else:
            print("ERROR: protocol ({}) not supported in schedule()".format(self.protocol))
            return None, None, None
        assert per_branch_accuracy.shape[0] == len(self.config_list) == 1036
        time4 = time.time()

        if self.protocol == "SmartAdapt_MN2_1head_Top200":
            # One-head solution does not use the latency head
            # while SmartAdapt_MN2_1head_Top200_merge does use the latency head
            config = self.config_list[np.argmax(per_branch_accuracy)]
        else:
            # Two-head solution needs the Latency predictor
            per_branch_latency = self.latency_predictor.predict(\
                height=height, width=width, nobj=nobj, objsize=objsize,
                gl=contention_levels["gpu_level"], cl=contention_levels["cpu_level"], FRCNN_only=True)
            assert per_branch_latency.shape[0] == 1036

            acc_lat_config_tups = [(acc, lat, config) for acc, lat, config in \
                zip(per_branch_accuracy, per_branch_latency, self.config_list) if lat <= self.latency_target]
            if not acc_lat_config_tups:  # No positive branches (latency satisfy)
                config = self.default_config
            else:
                _, _, config = max(acc_lat_config_tups, key=lambda x: (x[0], -x[1]))
        time5 = time.time()

        run_log = {
            "scheduler_overhead": (time5-time2)*1e3,
            "FE_overhead": (time3-time2)*1e3,
            "pred_overhead": (time4-time3)*1e3,
            "latency_requirement": self.user_requirement,
            "cpu_level": contention_levels["cpu_level"],
            "mem_bw_level": contention_levels["mem_bw_level"],
            "gpu_level": contention_levels["gpu_level"]
        }

        return config, image_pil, run_log

def load_graph_from_file(filename):

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def output_dict_to_bboxes_single_img(output_dict):

    # Output translation, in (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
    # all outputs are float32 numpy arrays, so convert types as appropriate
    N = int(output_dict['num_detections'][0])
    boxes = [(cls-1, sc, box[0], box[1], box[2], box[3]) for cls, box, sc in \
      zip(output_dict['detection_classes'][0].astype(np.int64)[:N],
          output_dict['detection_boxes'][0][:N],
          output_dict['detection_scores'][0][:N])]
    return boxes

class OpenCVTracker:

    def __init__(self, ds = 1, name = 'kcf'):

        self.ds = ds
        self.prev_frame = None
        self.prev_bboxes = []  # (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
        self.internal_tracker = cv2.MultiTracker_create()
        self.original_info = []
        self.tracker_name = name

    def reset_self(self):

        self.prev_frame = None
        self.prev_bboxes = None # (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
        self.internal_tracker = None
        self.original_info = None
        self.__init__(ds=self.ds, name=self.tracker_name)

    def createTrackerByName(self, trackerType):

        # Create a tracker based on tracker name
        trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN',
                        'MOSSE', 'CSRT']
        trackerType = trackerType.upper()
        if trackerType == trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
            tracker = cv2.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)
        return tracker

    def resize(self, input_img):

        if self.ds == 4:
            return input_img[::4, ::4, :]
        elif self.ds == 2:
            return input_img[::2, ::2, :]
        elif self.ds == 1:
            return input_img
        else:
            height, width,_ = input_img.shape
            return cv2.resize(input_img, (int(width / self.ds),
                                         int(height / self.ds)))

    # transfer from (ymin,xmin,ymax,xmax) in [0,1]
    #            to (xmin,ymin,width,height) in original size
    def change_to_tracker_format(self,box,frame):

        height, width, _ = frame.shape
        new_input_boxes = (int(np.round(box[3] * width)), 
                           int(np.round(box[2] * height)), 
                           int(np.round((box[5] - box[3]) * width)),
                           int(np.round((box[4] - box[2]) * height)))
        return new_input_boxes

    # transfer from (xmin,ymin,width,height) in original size
    #            to (ymin,xmin,ymax,xmax) in [0,1]
    def recover_to_output_format(self,box,frame):

        height, width, _ = frame.shape
        new_output_boxes = (box[1] / height, box[0] / width,
                            (box[1] + box[3]) / height,
                            (box[0] + box[2]) / width)
        new_output_boxes = (max(new_output_boxes[0],0),
                            max(new_output_boxes[1],0),
                            min(new_output_boxes[2],1),
                            min(new_output_boxes[3],1))
        return new_output_boxes

    def set_prev_frame(self, frame = None, bboxes = []):

        self.reset_self()
        if self.prev_frame is not None:
            self.prev_frame = self.resize(frame)
        else:
            # do initial tracking
            self.prev_frame = self.resize(frame)
            for box in bboxes:
                new_input_boxes = self.change_to_tracker_format(box,self.prev_frame)
                if self.tracker_name == 'csrt' and \
                   new_input_boxes[2] * new_input_boxes[3] <= 10:
                    continue
                self.internal_tracker.add(self.createTrackerByName(self.tracker_name),
                                          self.prev_frame,new_input_boxes)
                self.original_info.append((box[0],box[1]))

    def track(self, curr_frame):

        curr_frame = self.resize(curr_frame)
        new_boxes = []
        success, boxes = self.internal_tracker.update(curr_frame)
        for origin_info, box in zip(self.original_info, boxes):
            if success:
                new_out_box = self.recover_to_output_format(box,curr_frame)
                final_box = origin_info + new_out_box
                new_boxes.append(final_box)
        return new_boxes

class FlowRawTracker:

    def __init__(self, ds = 1, anchor = "fixed", mode = "bbox_median"):

        self.ds = ds
        self.prev_frame = None
        self.prev_bboxes = [] # (cls, conf, ymin, xmin, ymax, xmax) in [0,1]
        self.anchor = anchor
        self.mode = mode

    def set_prev_frame(self, frame = None, bboxes = []):

        if not frame is None:
            prvs_img = self.resize(frame)
            self.prev_frame = cv2.cvtColor(prvs_img, cv2.COLOR_BGR2GRAY)
        self.prev_bboxes = bboxes

    def resize(self, input_img):

        if self.ds == 4:
            return input_img[::4, ::4, :]
        elif self.ds == 2:
            return input_img[::2, ::2, :]
        elif self.ds == 1:
            return input_img
        else:
            height, width,_ = input_img.shape
            return cv2.resize(input_img, (int(width / self.ds),
                                         int(height / self.ds)))

    def track(self, curr_frame):

        next_img = self.resize(curr_frame)
        next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        height, width = next_img.shape
        
        img_flow = cv2.calcOpticalFlowFarneback(prev = self.prev_frame, next = next_img,
                                       flow = None, pyr_scale = 0.5,
                                       levels = 3, winsize = 15, iterations = 3,
                                       poly_n = 5, poly_sigma = 1.1,
                                       flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # Extract flow output
        new_bboxes = []
        for cls, conf, ymin, xmin, ymax, xmax in self.prev_bboxes:
            sig_pts0, sig_pts = [], []

            # First, find out the "significant" flows
            for x in range(int(xmin*width), int(xmax*width), 4):
                for y in range(int(ymin*height), int(ymax*height), 4):
                    dx, dy = int(img_flow[y, x, 0]), int(img_flow[y, x, 1])
                    st_pt = (x, y)        # width(y-dim) first
                    en_pt = (x+dx, y+dy)  # width(y-dim) first
                    if (abs(dx)+abs(dy) >= 5/self.ds):
                        sig_pts0.append((x, y, x+dx, y+dy))

            # Reject anapolies flows
            abs_xs = [abs(sig_pt[2] - sig_pt[0]) for sig_pt in sig_pts0]
            abs_ys = [abs(sig_pt[3] - sig_pt[1]) for sig_pt in sig_pts0]
            if len(abs_xs) >= 1 and len(abs_ys) >= 1:
                mean_x, std_x = np.mean(abs_xs), np.std(abs_xs)
                mean_y, std_y = np.mean(abs_ys), np.std(abs_ys)
                
                for x, y, x_pl_dx, y_pl_dy in sig_pts0:
                    if abs(x_pl_dx - x) - mean_x <= 2 * std_x and \
                       abs(y_pl_dy - y) - mean_y <= 2 * std_y:
                        sig_pts.append((x, y, x_pl_dx, y_pl_dy))
            else:
                sig_pts = sig_pts0

            # If the box moves
            if len(sig_pts) > 0:
                if self.mode == "pixel":
                    # new box is based on per-pixel mins, maxs
                    sig_xs = [sig_pt[2]/width for sig_pt in sig_pts]
                    sig_ys = [sig_pt[3]/height for sig_pt in sig_pts]
                    _ymin, _xmin = min(sig_ys), min(sig_xs), 
                    _ymax, _xmax = max(sig_ys), max(sig_xs)
                elif self.mode == "bbox":
                    # new box is based on mean dx, dy
                    dxs = [sig_pt[2] - sig_pt[0] for sig_pt in sig_pts]
                    dys = [sig_pt[3] - sig_pt[1] for sig_pt in sig_pts]
                    mean_x, mean_y = np.mean(dxs)/width, np.mean(dys)/height
                    _ymin, _xmin = ymin + mean_y, xmin + mean_x
                    _ymax, _xmax = ymax + mean_y, xmax  + mean_x
                elif self.mode == "bbox_median":
                    # new box is based on median dx, dy
                    dxs = [sig_pt[2] - sig_pt[0] for sig_pt in sig_pts]
                    dys = [sig_pt[3] - sig_pt[1] for sig_pt in sig_pts]
                    median_x, median_y = np.median(dxs)/width, np.median(dys)/height
                    _ymin, _xmin = ymin + median_y, xmin + median_x
                    _ymax, _xmax = ymax + median_y, xmax  + median_x
                else:
                    print("Error in mode of the object tracker.")
                    return

                # Make sure the new box is trancated within the frame
                #  -- [0, width-1] x [0, height-1]
                _xmin, _ymin = max(0, _xmin), max(0, _ymin)
                _xmax, _ymax  = min(1, _xmax), min(1, _ymax)
                if _ymin < _ymax and _xmin < _xmax:  
                    new_bboxes.append((cls, conf, _ymin, _xmin, _ymax, _xmax))
            else:
                new_bboxes.append((cls, conf, ymin, xmin, ymax, xmax))

        if self.anchor != "fixed": # then the reference frame is "moving"
            self.prev_bboxes = new_bboxes
            self.prev_frame = next_img
        return new_bboxes

class MBODF:
    
    # A Multi-branch Object Detection Framework
    def __init__(self, feat, kernel, frcnn_weight, fout_det, fout_lat, tv_version=None, dataset_prefix=None):
        
        self.fout_det = fout_det
        self.fout_lat = fout_lat
        if not dataset_prefix:
            socket_name = socket.gethostname()
            self.dataset_prefix = {
                "tx2-1": "/home/nvidia/sdcard/ILSVRC2015",
                "tx2-2": "/home/tx2/data/ILSVRC2015",
                "xv3": "/media/A833-362D/ILSVRC2015",
            }[socket_name]
        else:
            self.dataset_prefix = dataset_prefix
        self.feat = feat
        self.kernel = kernel
        if not self.kernel in ["FRCNN", "FRCNN+"]:
            print("Error, kernel not found")
        self.last_config = (-1, -1, -1, "", -1)

        self.detection_graph = load_graph_from_file(frcnn_weight)
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with self.detection_graph.as_default():
            graph = tf.compat.v1.get_default_graph()
            self.tensor_frame = graph.get_tensor_by_name('image_tensor:0')
            self.tensor_nprop = graph.get_tensor_by_name('ApproxDet_num_proposals:0')
            self.tensor_shape = graph.get_tensor_by_name('ApproxDet_min_dim:0')
            self.tname = {"RPN": "FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu",
                          "CPoP": "all_class_predictions_with_background"}
            keys = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']
            if feat in ["RPN", "CPoP"]:
                keys.append(self.tname[feat])
            self.output_tensor_dict = {key:graph.get_tensor_by_name(key + ':0') for key in keys}
            self.sess = tf.compat.v1.Session(config=tf_config)
        self.preheat()

    def preheat(self):

        preheat_img_dir1 = os.path.join(self.dataset_prefix,
          "Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00703000")
        preheat_img_dir2 = os.path.join(self.dataset_prefix,
          "Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00012009")
        preheat_img_dir3 = os.path.join(self.dataset_prefix,
          "Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00016004")
        preheat_img_dirs = [preheat_img_dir1, preheat_img_dir2, preheat_img_dir3]

        shapes, nprops = [224, 320, 448, 576], [1, 3, 5, 10, 20, 50, 100]
        configs = [(shape, nprop) for shape in shapes for nprop in nprops for run in range(3)]
        configs_w_path = [(*config, "{}/{:06d}.JPEG".format(preheat_img_dir, idx)) \
                          for preheat_img_dir in preheat_img_dirs \
                          for idx, config in enumerate(configs)]
        for shape, nprop, full_path in tqdm(configs_w_path, desc="Preheating the FRCNN kernel"):
            image_raw = np.array(Image.open(full_path))
            image_np = image_raw.astype(np.uint8)
            image_4D = np.expand_dims(image_np, axis=0)
            feed_dict = {self.tensor_frame: image_4D, self.tensor_nprop: nprop, self.tensor_shape: shape}
            output_dict = self.sess.run(self.output_tensor_dict, feed_dict=feed_dict)

    def run(self, config, frame_cnt_GoF, video_dir, frame_idx, img_pil, features, run_log):

        si, shape, nprop, tracker_name, ds = config
        if si != 1 and (tracker_name, ds) != (self.last_config[3], self.last_config[4]):
            if tracker_name == "bboxmedianfixed":
                self.tracker = FlowRawTracker(ds=ds, anchor="fixed", mode="bbox_median")
            else:
                self.tracker = OpenCVTracker(ds=ds, name=tracker_name)
        for frame_idx_delta in range(frame_cnt_GoF):
            # 1. Load a frame from the storage
            time1 = time.time()
            filename = "{:06d}.JPEG".format(frame_idx+frame_idx_delta)
            video_frame_path = os.path.join(video_dir, filename)
            if frame_idx_delta == 0:
                image_raw = np.array(img_pil)
            else:
                full_path = os.path.join(self.dataset_prefix, video_frame_path)
                image_raw = np.array(Image.open(full_path))
            time2 = time.time()

            if frame_idx_delta == 0:  # Do "detection"
                with self.detection_graph.as_default():
                    image_np = image_raw.astype(np.uint8)
                    image_4D = np.expand_dims(image_np, axis=0)
                    feed_dict = {self.tensor_frame: image_4D, self.tensor_nprop: nprop, self.tensor_shape: shape}
                    output_dict = self.sess.run(self.output_tensor_dict, feed_dict=feed_dict)
                    bboxes = output_dict_to_bboxes_single_img(output_dict)
                time3 = time.time()

                if si > 1:
                    # tracker requires 3D numpy array in BGR
                    image_np = image_raw[:, :, ::-1]
                    self.tracker.set_prev_frame(frame=image_np, bboxes=bboxes)
                time4 = time.time()
            else:  # Do "tracking"
                time3 = time.time()

                # Format change for the tracker: an BGR numpy array
                image_np = image_raw[:, :, ::-1]
                bboxes = self.tracker.track(image_np)
                time4 = time.time()

            # 5. Keep track of the cached features: nobj, objsize, (optional) RPN, CPoP
            (height, width), nobj = image_raw.shape[:2], len(bboxes)
            sizes = [(ymax-ymin)*(xmax-xmin)*height*width \
               for _, _, ymin, xmin, ymax, xmax in bboxes]
            avgsize = np.sqrt(np.sum(sizes)) if (sizes and np.sum(sizes)>0) else 0
            if nobj > 0:
                features["nobj"], features["objsize"] = nobj, avgsize
            if self.feat == "RPN" and frame_idx_delta == 0:
                features["RPN"] = np.mean(output_dict[self.tname[self.feat]], axis=(0, 1, 2))
            if self.feat == "CPoP" and frame_idx_delta == 0:
                features["CPoP"] = np.mean(output_dict[self.tname[self.feat]], axis=0)
            time5 = time.time()

            # 6a. per-obj detection log
            for cls, conf, ymin, xmin, ymax, xmax in bboxes:
                print("{} {} {} {} {} {} {}".format(video_frame_path, cls,
                  conf, ymin, xmin, ymax, xmax), file=self.fout_det)
            # 6b. per-frame log
            loading_lat = (time2-time1)*1e3
            detection_lat = (time3-time2)*1e3
            tracker_lat = (time4-time3)*1e3
            is_det_frame = (frame_idx_delta == 0)
            if is_det_frame:
                overhead_lat = (time5-time4)*1e3 + run_log["scheduler_overhead"]
                overhead_lat_FE, overhead_lat_pred = (time5-time4)*1e3+run_log["FE_overhead"], run_log["pred_overhead"]
            else:
                overhead_lat, overhead_lat_FE, overhead_lat_pred = 0, 0, 0
            line = "{} {} {} ".format(video_frame_path, height, width)
            line += "{} ".format(is_det_frame)
            line += "{} {} {} {} {} ".format(si, shape, nprop, tracker_name, ds)
            line += "{:.3f} {:.3f} {:.3f} ".format(loading_lat, detection_lat, tracker_lat)
            line += "{:.3f} {:.3f} {:.3f} ".format(overhead_lat, overhead_lat_FE, overhead_lat_pred)
            line += "latency {} ".format(run_log["latency_requirement"])
            line += "{} ".format(run_log["cpu_level"])
            line += "{} ".format(run_log["mem_bw_level"])
            line += "{} ".format(run_log["gpu_level"])
            if "protocol" in run_log:  # particular for scheduler with CB
                line += "{} ".format(run_log["protocol"])
                line += "{:.3f} ".format(run_log["lat_pred_overhead"])
                line += "{:.3f} ".format(run_log["feat_sel_overhead"])
                line += "{:.3f} ".format(run_log["pareto_overhead"])
            line += "{} {} ".format(features["nobj"], features["objsize"])
            line += "{}".format(nobj)
            for _, _, ymin, xmin, ymax, xmax in bboxes:
                size = (ymax-ymin)*(xmax-xmin)
                line += " {:.6f}".format(size)
            print(line, file=self.fout_lat)
