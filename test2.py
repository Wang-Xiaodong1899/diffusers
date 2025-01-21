import json


def jsonl_to_dict():
    result_dict = {}
    with open("/data/wangxd/nuscenes/nusc_action_26keyframe_all_train.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            result_dict.update(data)
    
    with open("/data/wangxd/nuscenes/nusc_action_26keyframe_all_train_500_700.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            result_dict.update(data)
    
    print(len(result_dict))
    return result_dict

def save_dict_to_json(output_file, dictionary):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(dictionary, file, indent=4, ensure_ascii=False)

output_file = '/data/wangxd/nuscenes/nusc_action_26keyframe_all_train.json'

scene_dict = jsonl_to_dict()
save_dict_to_json(output_file, scene_dict)

print(len(scene_dict))
