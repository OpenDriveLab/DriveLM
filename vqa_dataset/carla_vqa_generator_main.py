import argparse
from vqa_dataset.carla_vqa_generator import QAsGenerator
import string
import random
import pathlib
import json
import os

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description="QA Generator for DriveLM Carla")

    # Dataset and path settings
    path_group = parser.add_argument_group('Dataset and Path Settings')
    # path_group.add_argument('--base-folder', type=str, default='database',
    #                         help='Base folder for dataset')
    path_group.add_argument('--path-keyframes', type=str, required=True, default='path/to/keyframes.txt',
                            help='Path to the keyframes.txt')
    path_group.add_argument('--data-directory', type=str, required=True, default='path/to/dataset/data',
                            help='Data directory containing the dataset')
    path_group.add_argument('--output-graph-directory', type=str, required=True, default='output/path/of/graph/vqas',
                            help='Output directory for the vqa-graph')
    path_group.add_argument('--output-graph-examples-directory', type=str, default='output/path/of/graph/vqas',
                            help='Output directory for examples of the vqa-graph')

    # Image and camera parameters
    img_group = parser.add_argument_group('Image and Camera Parameters')
    img_group.add_argument('--target-image-size', nargs=2, type=int, default=[1024, 384],
                           help='Target image size [width, height]')
    img_group.add_argument('--original-image-size', nargs=2, type=int, default=[1024, 512],
                           help='Original image size [width, height]')
    img_group.add_argument('--original-fov', type=float, default=110,
                           help='Original field of view')

    # Region of interest (ROI) for image projection
    roi_group = parser.add_argument_group('Region of Interest (ROI) Parameters')
    roi_group.add_argument('--min-y', type=int, default=0,
                           help='Minimum Y coordinate for ROI (to cut part of the bottom)')
    roi_group.add_argument('--max-y', type=int, default=None,
                           help='Maximum Y coordinate for ROI (to cut part of the bottom)')

    # Sampling parameters
    sampling_group = parser.add_argument_group('Sampling Parameters')
    sampling_group.add_argument('--random-subset-count', type=int, default=-1,
                                help='Number of random samples to use (-1 for all samples)')
    sampling_group.add_argument('--sample-frame-mode', choices=['all', 'keyframes', 'uniform'], default='keyframes',
                                help='Frame sampling mode')
    sampling_group.add_argument('--sample-uniform-interval', type=int, default=5,
                                help='Interval for uniform sampling (if sample-frame-mode is "uniform")')

    # Visualization and saving options
    viz_group = parser.add_argument_group('Visualization and Saving Options')
    viz_group.add_argument('--save-examples', action='store_true', default=False,
                           help='Save example images')
    viz_group.add_argument('--visualize-projection', action='store_true', default=False,
                           help='Visualize object centers & bounding boxes in the image')
    viz_group.add_argument('--filter-routes-by-result', action='store_true', default=False,
                           help='Skip routes based on expert driving results')
    viz_group.add_argument('--remove-pedestrian-scenarios', action='store_true', default=False,
                           help='Skip scenarios with pedestrians')

    args = parser.parse_args()

    # Compute derived parameters
    args.min_x = args.original_image_size[0] // 2 - args.target_image_size[0] // 2
    args.max_x = args.original_image_size[0] // 2 + args.target_image_size[0] // 2
    if args.max_y is None:
        args.max_y = args.target_image_size[1]

    return args

def generate_random_string(length=32):
    """
    Generate a random alphanumeric string of a given length.
    """
    # Define the character set from which to generate the random string
    characters = string.ascii_letters + string.digits

    # Generate a random string by joining random characters from the character set
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def convert_carla_to_nuscenes_and_save(args, carla_file_content):
    """
    Convert a CARLA format file to a Nuscenes format file.
    """
    # Generate a random scene ID
    scene_id = generate_random_string()

    # Create a dictionary to store the scene data
    scene_data = {scene_id: {}}
    scene_data[scene_id]['scene_description'] = None
    scene_data[scene_id]['key_frames'] = {}

    for path, tick in zip(carla_file_content['image_paths'], carla_file_content['llava_format']):
        # Generate a random tick ID
        tick_id = generate_random_string()
        tick_data = {}
        scene_data[scene_id]['key_frames'][tick_id] = tick_data

        # Store key object information
        tick_data['key_object_infos'] = tick['conversations']['key_object_infos']

        # Create a dictionary to store QA data
        qa_data = {}
        tick_data['QA'] = qa_data
        qa_data['perception'] = []
        qa_data['prediction'] = []
        qa_data['planning'] = []
        qa_data['behavior'] = []

        # Store image path
        image_path = tick['image']
        tick_data['image_paths'] = {key: None for key in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']}
        tick_data['image_paths']['CAM_FRONT'] = image_path

        for key, items in tick['conversations'].items():
            # Available keys: important_objects, dynamic_vehicles, roadlayout, stopsign, trafficlight, ego
            if key == 'key_object_infos':
                continue

            for i in range(len(items) // 2):
                q_dic = items[2*i]
                a_dic = items[2*i + 1]
                qa_item = {
                    'Q': q_dic['value'],
                    'A': a_dic['value'],
                    'C': None,
                    'con_up': q_dic['connection_up'],
                    'con_down': q_dic['connection_down'],
                    'cluster': q_dic['chain'],
                    'layer': q_dic['layer']
                }
                if 'object_tags' in q_dic:
                    qa_item['object_tags'] = q_dic['object_tags']

                qa_data[q_dic['type']].append(qa_item)

        scenario_name, route_number, _, image_number = path.split('/')[-4:]
        route_number = route_number.split('_')[0] + '_' + route_number.split('_')[1]
        image_number = image_number.replace('.jpg', '')
        
        save_dir = os.path.join(args.output_graph_directory, scenario_name, route_number)
        pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
        with open(save_dir + '/' + image_number + '.json', 'w', encoding='utf-8') as f:
            json.dump(tick_data, f, indent=4)

if __name__ == '__main__':
    args = parse_arguments()
    qas_generator = QAsGenerator(args)
    vqa_llava_format = qas_generator.create_qa_pairs()

    # Convert and save VQA LLAVA format data
    convert_carla_to_nuscenes_and_save(args, vqa_llava_format)
