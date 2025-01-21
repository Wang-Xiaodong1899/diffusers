import json

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from openai import OpenAI
from tqdm import tqdm


# client = OpenAI(
#     base_url="https://api.ai-gaochao.cn/v1/",
#     api_key='sk-j39KGi7DXHUkHipN18D34a12E3Aa4d868b9c6dF8371c0e67'
# )
client = OpenAI(
    base_url="https://api.chatanywhere.tech/v1/",
    api_key='sk-88uyXsAdEyDN5ESbWVWTtG6Do6vj9y2biMqtMsIsf6pqDvvY'
)

# 0-31
split = "train"
command_path = f'/data/wangxd/nuscenes/nusc_action_{split}.json'
with open(command_path, 'r') as f:
    command_dict = json.load(f)

# nusc = NuScenes(version='v1.0-trainval', dataroot="/data/wangxd/nuscenes/", verbose=True)

splits = create_splits_scenes()

scenes = splits[split]

print(len(scenes))

output_path = f'/data/wangxd/nuscenes/nusc_action_26keyframe_all_{split}_500_700.jsonl'
with open(output_path, 'w') as f:

    scene_item = {}
    for scene in tqdm(scenes[500:]):
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
                "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
                "scene-0171", "scene-0172", "scene-0173", "scene-0174",
                "scene-0175", "scene-0176", "scene-0419"
        ]
        if scene in command_miss_scenes:
            continue
        drive_list = command_dict[scene] # dict
        item = {}
        for idx in range(len(drive_list)-18): # 26-8
            candidates = []
            for cidx in range(idx, idx+18):
                command = drive_list[f"{cidx}"]
                if not candidates or candidates[-1] != command:
                    candidates.append(command)
            final_command = '. '.join(candidates)

            content = f"""
    To give you continuous driving actions in chronological order, please summarize and refine into a simpler and easier to read continuous driving instructions, which must be simple and continuous one sentence, keeping only the most important actions, and can only have a maximum of 4 actions.

    {final_command}

    Response:\n
            """
            # print(content)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot with knowledge of the field of autonomous driving. "
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            # print(response_message)
            # print(completion.model)
            # import pdb; pdb.set_trace()
            item[f"{idx}"] = response_message
        scene_item[scene] = item
        json_line = {
            scene: item
        }
        f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        f.flush()
