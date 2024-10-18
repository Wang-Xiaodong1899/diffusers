
from torch.utils.data import Dataset
import json
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode, resize
import os
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def image2pil(filename):
    return Image.open(filename)

def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)

def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr

class NuscenesDatasetForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/home/wuzhirong/PKU_new/Nuscenes-v1.0-trainval-CAM_FRONT",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 9, # must be (4k+1)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = True,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            prompts = torch.load(os.path.join(preprocessed_data_path,"prompts.npy"))
            videos = torch.load(os.path.join(preprocessed_data_path,"videos.npy"))
            images = torch.load(os.path.join(preprocessed_data_path,"images.npy"))
            progress_dataset_bar = tqdm(
                range(0, prompts.shape[0]),
                desc="Loading preprocessed prompts and videos",
            )
            self.instance_prompts = []
            self.instance_videos =[]
            for i in range(prompts.shape[0]):
                progress_dataset_bar.update(1)
                self.instance_prompts.append(prompts[i:i+1])
                self.instance_videos.append((videos[i:i+1],images[i:i+1]))
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        if self.preload_all_data:
            return len(self.instance_prompts)
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        my_scene = self.scenes[index]
        all_frames = self.frames_group[my_scene]
        
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames]            

        scene_annotation = self.annotations[my_scene]

        seek_id = seek_start//4*4
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        driving_prompt = annotation.get("Driving action", "").strip()
        # print(driving_prompt)

        frames = torch.Tensor(np.stack([image2arr(path) for path in seek_path]))

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        if self.preload_all_data:
            return {
                "instance_prompt": self.instance_prompts[index],
                "instance_video": self.instance_videos[index],
            }
        else:
            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }


class NuscenesDatasetAllframesForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/root/autodl-tmp/nuscenes/all",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 49, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples = self.get_samples(split)

            print('Total samples: %d' % len(self.samples))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.samples)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        selected_scenes = self.splits[split] # all scenes
        all_scenes = self.nusc.scene
        selected_scenes_meta = []
        for sce in all_scenes:
            if sce['name'] in selected_scenes:
                selected_scenes_meta.append(sce)
        
        samples_group_by_scene = []
        for scene in selected_scenes_meta:
            samples_group_by_scene.append(
                {
                'scene': scene['name'],
                'video': self.get_all_frames_from_scene(scene)
                }
            )
        
        return samples_group_by_scene

    def get_all_frames_from_scene(self, scene):
        # get all frames (keyframes, sweeps)
        first_sample_token = scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor]) # first frame sensor token
        # frames = 0
        all_frames_dict = [] # len() -> frame number
        while True:
            all_frames_dict.append(cam_front_data)
            # filename = cam_front_data['filename']  # current-frame filename
            next_sample_data_token = cam_front_data['next']  # next-frame sensor token
            if not next_sample_data_token: # ''
                break
            cam_front_data = self.nusc.get('sample_data', next_sample_data_token)
        
        return all_frames_dict
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        my_scene = self.samples[index]
        my_scene_name = my_scene['scene']
        my_sample_list = my_scene["video"]
        inner_frame_len = len(my_sample_list)
        
        key_frame_idx = []
        for seek_idx in range(inner_frame_len):
            if my_sample_list[seek_idx]['is_key_frame']:
                key_frame_idx.append(seek_idx)
        
        seek_sample = None
        while seek_sample is None or not seek_sample['is_key_frame']:
            seek_start = random.randint(0, inner_frame_len - 51)
            seek_sample = my_sample_list[seek_start]

        seek_samples = my_sample_list[seek_start: seek_start + self.max_num_frames]
        
        # all frames
        frame_paths = []
        for my_sample in seek_samples:
            file_path = my_sample["filename"]
            image_path = os.path.join(self.data_root, file_path)
            frame_paths.append(image_path)

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = key_frame_idx.index(seek_start)

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx)
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                import pdb; pdb.set_trace()
            # driving_prompt = driving_prompt + ". " + caption
            # print(driving_prompt)

        frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        if self.preload_all_data:
            return {
                "instance_prompt": self.instance_prompts[index],
                "instance_video": self.instance_videos[index],
            }
        else:
            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }


# if __name__ == "__main__":
#     train_dataset = NuscenesDatasetAllframesForCogvidx(split="train")
#     for i in tqdm(range(len(train_dataset))):
#         item = train_dataset[i]
#         import pdb; pdb.set_trace()