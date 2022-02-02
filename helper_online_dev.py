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
from helper_online import get_config_list, LatencyPredictor, FeatureExtractorOnline
from helper_online import FeatureToVecOneHeadOnline, BaselineAccuracyPredictorOnline

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
        x = feature.float()
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

        torch.set_default_dtype(torch.float32)
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
        feature = torch.from_numpy(feature).float().to(device=self.device)
        output = self.model.forward(feature)
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = output
        return output_ret

class FeatureToVecJointOnline():

    def __init__(self, filename, mask, trainable_fe, tv_version=None):

        torch.set_default_dtype(torch.float32)
        self.mask = mask
        input_dim = 4+1280  # Support "MobileNetV2Pool" only
        self.model_acc = NN_residual(input_dim, mask=mask)
        socket_name = socket.gethostname()
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
        img_pre = self.preprocess(img_pil).float().unsqueeze(0).to(self.device)
        # (1280,) torch array
        feature_heavy = torch.mean(self.model_fe.features(img_pre)[0], dim=(1, 2))
        # (1280,) numpy array
        feature_heavy = feature_heavy.cpu().detach().numpy()
        return feature_heavy

    def predict(self, feature):

        # output_ret is a (1036,) numpy array
        feature = torch.from_numpy(np.array(feature)).float().to(self.device)  # [n, 4+1280]
        output = self.model_acc.forward(feature)   # [n,4+1280] Tensor --> [n,1036] Tensor
        output = output.cpu().detach().numpy()[0, :]
        output_ret = np.zeros((1036,))
        output_ret[self.mask] = output
        return output_ret

class SchedulerCBOnline:

    # Supported protocols:
    # SmartAdapt_BL
    # SmartAdapt_Lite, SmartAdapt_HoC, SmartAdapt_RPN, SmartAdapt_CPoP, SmartAdapt_MN2, SmartAdapt_MN2_joint
    # SmartAdapt_Lite_Top200, SmartAdapt_HoC_Top200, SmartAdapt_RPN_Top200
    # SmartAdapt_CPoP_Top200, SmartAdapt_MN2_Top200, SmartAdapt_MN2_joint_Top200
    # 
    # Special protocol:
    # SmartAdapt_MN2_1head_Top200: use two-head solution, i.e. SmartAdapt_MN2_1head_Top200_merge
    def __init__(self, contention_levels,
                 user_requirement=33.3, p95_requirement=True, mobile_device="tx2",
                 cost_filename="models/SmartAdapt_cost_20211009.pb", 
                 benefit_filename="models/SmartAdapt_benefit_20211009.pb",
                 protocol_options_avail=["SmartAdapt_BL", "SmartAdapt_RPN", "SmartAdapt_MN2", 
                                         "SmartAdapt_MN2_joint_Top200", "SmartAdapt_MN2_1head_Top200"],
                 tv_version=None, dataset_prefix=None):

        self.prot2feat = {
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
        }
        self.cost_LUT = pickle.load(open(cost_filename, "rb"))
        self.benefit_LUT = pickle.load(open(benefit_filename, "rb"))
        # A protocol will be consider if it is at least better than the baseline
        protocol_options = []
        for protocol in protocol_options_avail:
            key = (protocol, mobile_device, contention_levels["gpu_level"], user_requirement)
            if (key in self.benefit_LUT) and (self.benefit_LUT[key] >= 0):
                protocol_options.append(protocol)
        # print("scheduler.protocol_options = {}".format(protocol_options))
        self.protocol_options = protocol_options
        self.mobile_device = mobile_device
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

        self.mask, self.accuracy_predictor = {}, {}
        for protocol in protocol_options:
            feat = self.prot2feat[protocol]

            # Load the masks
            if protocol in ["SmartAdapt_Lite_Top200", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200",
                            "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200", "SmartAdapt_MN2_Top200",
                            "SmartAdapt_MN2_joint_Top200", "SmartAdapt_MN2_1head_Top200"]:
                self.mask[protocol] = pickle.load(open("models/mask_top200.pb", "rb"))
            else:
                self.mask[protocol] = np.ones((1036,)).astype(bool)

            # Initialize the accuracy predictor
            if protocol == "SmartAdapt_BL":
                self.accuracy_predictor[protocol] = BaselineAccuracyPredictorOnline("models/SmartAdapt_BL.pb")
            elif protocol in ["SmartAdapt_Lite", "SmartAdapt_HoC", "SmartAdapt_HoG",
                              "SmartAdapt_RPN", "SmartAdapt_CPoP",
                              "SmartAdapt_Lite_Top200", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200",
                              "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200"]:
                filename = "models/{}.pb".format(protocol)
                self.accuracy_predictor[protocol] = \
                  FeatureToVecOnline(feature=feat, filename=filename, mask=self.mask[protocol])
            elif protocol in ["SmartAdapt_MN2", "SmartAdapt_MN2_Top200",
                              "SmartAdapt_MN2_joint", "SmartAdapt_MN2_joint_Top200"]:
                trainable_fe = (protocol in ["SmartAdapt_MN2_joint", "SmartAdapt_MN2_joint_Top200"])
                filename = "models/{}.pb".format(protocol)
                self.accuracy_predictor[protocol] = \
                  FeatureToVecJointOnline(filename=filename, mask=self.mask[protocol], trainable_fe=trainable_fe, tv_version=tv_version)
            elif protocol in ["SmartAdapt_MN2_1head_Top200"]:
                self.accuracy_predictor[protocol] = \
                    FeatureToVecOneHeadOnline(filename="models/{}.pb".format(protocol), mask=self.mask[protocol])
            else:
                print("ERROR: protocol not supported in init()")
                return

        # Initialize the HoC/HoG feature extractor
        self.feature_extractor = None
        for protocol in protocol_options:
            if protocol in ["SmartAdapt_HoC", "SmartAdapt_HoG", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200"]:
                if not self.feature_extractor:
                    self.feature_extractor = FeatureExtractorOnline(feature_name=self.prot2feat[protocol])
                else:
                    print("Error! Only one HoC/HOG model is allowed!")


        # Initialize the latency predictor
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

        # Two-head solution needs the Latency predictor
        per_branch_latency = self.latency_predictor.predict(\
            height=height, width=width, nobj=nobj, objsize=objsize,
            gl=contention_levels["gpu_level"], cl=contention_levels["cpu_level"], FRCNN_only=True)
        assert per_branch_latency.shape[0] == 1036

        # Content agnostic accuracy
        per_branch_accuracy_cag = self.accuracy_predictor["SmartAdapt_BL"].predict()
        time2a = time.time()

        # Feature selection
        results = []
        for idx, protocol in enumerate(self.protocol_options):
            per_branch_cost = self.cost_LUT[(protocol, self.mobile_device, contention_levels["gpu_level"])]
            benfit = self.benefit_LUT[(protocol, self.mobile_device, contention_levels["gpu_level"], self.user_requirement)]
            acc_lat_config_tups = [(acc+benfit, lat+cost, config) for acc, lat, cost, config in \
                zip(per_branch_accuracy_cag, per_branch_latency, per_branch_cost, self.config_list) \
                if lat+cost <= self.latency_target]
            if not acc_lat_config_tups:  # No positive branches (latency satisfy)
                config = self.default_config
                acc_est = per_branch_accuracy[self.default_config_idx]
                lat_est = per_branch_latency[self.default_config_idx] + per_branch_cost[self.default_config_idx]
            else:
                acc_est, lat_est, config = max(acc_lat_config_tups, key=lambda x: (x[0], -x[1]))
            results.append((protocol, acc_est, lat_est, config))
        # For debug purpose
        # print("Feature selection results: {}".format(results))
        self.protocol, acc_est, lat_est, config = max(results, key=lambda x: (x[1], -x[2]))
        time3 = time.time()

        if self.protocol == "SmartAdapt_BL":
            time4 = time.time()
            time5 = time.time()
            time6 = time.time()
            pass
        else:
            # Extract heavy feature
            if self.protocol in ["SmartAdapt_Lite", "SmartAdapt_Lite_Top200"]:
                heavy_features = []
            elif self.protocol in ["SmartAdapt_HoC", "SmartAdapt_HoG", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200"]:
                heavy_features = list(self.feature_extractor.extract(np.array(image_pil)))
            elif self.protocol in ["SmartAdapt_RPN", "SmartAdapt_CPoP", "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200"]:
                heavy_features = list(feature_cache[self.prot2feat[self.protocol]])
            elif self.protocol in ["SmartAdapt_MN2", "SmartAdapt_MN2_joint", "SmartAdapt_MN2_Top200",
                                   "SmartAdapt_MN2_joint_Top200", "SmartAdapt_MN2_1head_Top200"]:
                heavy_features = list(self.accuracy_predictor[self.protocol].fe(image_pil))
            feature += heavy_features
            feature = np.expand_dims(np.array(feature), axis=0)  # (1, 4+dim) numpy array
            time4 = time.time()

            if self.protocol in ["SmartAdapt_Lite", "SmartAdapt_Lite_Top200",
                              "SmartAdapt_HoC", "SmartAdapt_HoG", "SmartAdapt_HoC_Top200", "SmartAdapt_HoG_Top200",
                              "SmartAdapt_RPN", "SmartAdapt_CPoP", "SmartAdapt_RPN_Top200", "SmartAdapt_CPoP_Top200",
                              "SmartAdapt_MN2", "SmartAdapt_MN2_joint", "SmartAdapt_MN2_Top200", "SmartAdapt_MN2_joint_Top200"]:
                per_branch_accuracy = self.accuracy_predictor[self.protocol].predict(feature)
            elif self.protocol in ["SmartAdapt_MN2_1head_Top200"]:
                lat_req = np.array([self.user_requirement])
                cpu_cont = np.array([contention_levels["cpu_level"]])
                gpu_cont = np.array([contention_levels["gpu_level"]])
                per_branch_accuracy = self.accuracy_predictor[self.protocol].predict(feature, lat_req, cpu_cont, gpu_cont)
            else:
                print("ERROR: protocol ({}) not supported in schedule()".format(self.protocol))
                return None, None, None
            assert per_branch_accuracy.shape[0] == len(self.config_list) == 1036
            time5 = time.time()

            acc_lat_config_tups = [(acc, lat, config) for acc, lat, config in \
                zip(per_branch_accuracy, per_branch_latency, self.config_list) if lat <= self.latency_target]
            if not acc_lat_config_tups:  # No positive branches (latency satisfy)
                config = self.default_config
            else:
                _, _, config = max(acc_lat_config_tups, key=lambda x: (x[0], -x[1]))
            time6 = time.time()

        run_log = {
            "scheduler_overhead": (time6-time2)*1e3,
            "FE_overhead": (time4-time3)*1e3,  # in phase 2, optional
            "pred_overhead": (time5-time4)*1e3,  # in phase 2, optional
            "pareto_overhead": (time6-time5)*1e3,  # in phase 2, optional
            "lat_pred_overhead": (time2a-time2)*1e3,  # in phase 1, must
            "feat_sel_overhead": (time3-time2a)*1e3,  # in phase 1, must
            "protocol": self.protocol,
            "latency_requirement": self.user_requirement,
            "cpu_level": contention_levels["cpu_level"],
            "mem_bw_level": contention_levels["mem_bw_level"],
            "gpu_level": contention_levels["gpu_level"]
        }
        return config, image_pil, run_log
