{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "import torchvision.transforms as transforms\n",
    "from tqdm.auto import tqdm\n",
    "from PIL import Image\n",
    "from collections import defaultdict\n",
    "import math\n",
    "from pathlib import Path"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "image_folder = \"./\"\n",
    "csvfile = \"./dev.csv\"\n",
    "\n",
    "df = pd.read_csv(csvfile)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "id_to_labels = {str(row['fname']): row['labels'].split(',') for _, row in df.iterrows()}\n",
    "unique_labels = sorted(set(label for labels in id_to_labels.values() for label in labels))\n",
    "label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}\n",
    "num_classes = len(label_to_idx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pad_num(n, width):\n",
    "  return str(n).zfill(width)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "file_paths = list(Path(image_folder).rglob(\"*.png\"))\n",
    "max_images = 1000\n",
    "shard_size = 100\n",
    "image_tensor = []\n",
    "label_tensor = []\n",
    "processed_examples = 0\n",
    "processed_shards = 0\n",
    "rel_idxs = []\n",
    "max_images = min(max_images, len(file_paths))\n",
    "num_shards = (max_images + shard_size - 1) // shard_size\n",
    "pad_size = math.log10(num_shards) + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_chunk(imgs, lbls, idx):\n",
    "  shard_data = {\"imgs\": imgs, \"lbls\": lbls}\n",
    "  padded_num = pad_num(idx, int(pad_size))\n",
    "  torch.save(shard_data, os.path.join(\".\", \"processed\", f\"fsd_{padded_num}.pth\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "transform = transforms.Compose([\n",
    "  transforms.Resize((224, 224)),\n",
    "  transforms.ToTensor(),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Given a shard, it returns 2 tensors, one for image and another for the label vector which it also creates \n",
    "def process_files(shard_paths):\n",
    "  img_tensor = []\n",
    "  lbls_tensor = []\n",
    "  for filename in shard_paths:\n",
    "    img_id = filename.stem # get file name\n",
    "    img_path = os.path.join(image_folder, filename)\n",
    "\n",
    "    if img_id in id_to_labels:\n",
    "      img = Image.open(img_path).convert(\"RGB\")\n",
    "      img = transform(img)\n",
    "      img_tensor.append(img)\n",
    "\n",
    "      label_vector = torch.zeros(num_classes, dtype=torch.float32)\n",
    "      for label in id_to_labels[img_id]:\n",
    "        if label in label_to_idx:\n",
    "          label_vector[label_to_idx[label]] = 1\n",
    "      lbls_tensor.append(label_vector)\n",
    "  img_tensor = torch.stack(img_tensor)\n",
    "  lbls_tensor = torch.stack(lbls_tensor)\n",
    "\n",
    "  class_idxs = [torch.where(lbls_tensor[:, c] == 1)[0] for c in range(num_classes)]\n",
    "  \n",
    "  return img_tensor, lbls_tensor, class_idxs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing 1000 files\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dce0ab8cedf04ac5a9446807064322ac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Processing chunks:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "acc_shard_class_idxs = []\n",
    "class_idxs = []\n",
    "\n",
    "print(f\"Processing {max_images} files\")\n",
    "for shard_idx in tqdm(range(0, max_images, shard_size), desc=\"Processing chunks\"):\n",
    "  shard_paths = file_paths[shard_idx: min(shard_idx + shard_size, max_images)]\n",
    "  imgs, lbls, shard_class_idxs = process_files(shard_paths)\n",
    "  save_chunk(imgs, lbls, processed_shards)\n",
    "  acc_shard_class_idxs.append(shard_class_idxs) # store a list of precalculated indices which are part of each class\n",
    "  \n",
    "  processed_shards += 1 # for shard naming scheme\n",
    "  \n",
    "  processed_examples += len(imgs) # update total examples for tracking in rel_idxs\n",
    "  rel_idxs.append(processed_examples - 1)  # store index of the latest example of this chunk so we can identify which chunk an index is from\n",
    "  \n",
    "class_idxs = [torch.cat(shard_class_idxs) for shard_class_idxs in zip(*acc_shard_class_idxs)]\n",
    "torch.save({\n",
    "  'rel_idxs': rel_idxs,\n",
    "  'class_idxs': class_idxs\n",
    "}, os.path.join(\"./processed\", f\"fsd_meta.pth\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Aircraft', 'Alarm', 'Animal', 'Applause', 'Bark', 'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bell', 'Bicycle', 'Bicycle_bell', 'Bird', 'Bird_vocalization_and_bird_call_and_bird_song', 'Boat_and_Water_vehicle', 'Boiling', 'Boom', 'Bowed_string_instrument', 'Brass_instrument', 'Breathing', 'Burping_and_eructation', 'Bus', 'Buzz', 'Camera', 'Car', 'Car_passing_by', 'Cat', 'Chatter', 'Cheering', 'Chewing_and_mastication', 'Chicken_and_rooster', 'Child_speech_and_kid_speaking', 'Chime', 'Chink_and_clink', 'Chirp_and_tweet', 'Chuckle_and_chortle', 'Church_bell', 'Clapping', 'Clock', 'Coin_(dropping)', 'Computer_keyboard', 'Conversation', 'Cough', 'Cowbell', 'Crack', 'Crackle', 'Crash_cymbal', 'Cricket', 'Crow', 'Crowd', 'Crumpling_and_crinkling', 'Crushing', 'Crying_and_sobbing', 'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Cymbal', 'Dishes_and_pots_and_pans', 'Dog', 'Domestic_animals_and_pets', 'Domestic_sounds_and_home_sounds', 'Door', 'Doorbell', 'Drawer_open_or_close', 'Drill', 'Drip', 'Drum', 'Drum_kit', 'Electric_guitar', 'Engine', 'Engine_starting', 'Explosion', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)', 'Finger_snapping', 'Fire', 'Fireworks', 'Fixed-wing_aircraft_and_airplane', 'Fowl', 'Frog', 'Frying_(food)', 'Gasp', 'Giggle', 'Glass', 'Glockenspiel', 'Gong', 'Growling', 'Guitar', 'Gull_and_seagull', 'Gunshot_and_gunfire', 'Gurgling', 'Hammer', 'Hands', 'Harmonica', 'Harp', 'Hi-hat', 'Hiss', 'Human_group_actions', 'Human_voice', 'Idling', 'Insect', 'Keyboard_(musical)', 'Keys_jangling', 'Knock', 'Laughter', 'Liquid', 'Livestock_and_farm_animals_and_working_animals', 'Male_singing', 'Male_speech_and_man_speaking', 'Mallet_percussion', 'Marimba_and_xylophone', 'Mechanical_fan', 'Mechanisms', 'Meow', 'Microwave_oven', 'Motor_vehicle_(road)', 'Motorcycle', 'Music', 'Musical_instrument', 'Ocean', 'Organ', 'Packing_tape_and_duct_tape', 'Percussion', 'Piano', 'Plucked_string_instrument', 'Pour', 'Power_tool', 'Printer', 'Purr', 'Race_car_and_auto_racing', 'Rail_transport', 'Rain', 'Raindrop', 'Ratchet_and_pawl', 'Rattle', 'Rattle_(instrument)', 'Respiratory_sounds', 'Ringtone', 'Run', 'Sawing', 'Scissors', 'Scratching_(performance_technique)', 'Screaming', 'Screech', 'Shatter', 'Shout', 'Sigh', 'Singing', 'Sink_(filling_or_washing)', 'Siren', 'Skateboard', 'Slam', 'Sliding_door', 'Snare_drum', 'Sneeze', 'Speech', 'Speech_synthesizer', 'Splash_and_splatter', 'Squeak', 'Stream', 'Strum', 'Subway_and_metro_and_underground', 'Tabla', 'Tambourine', 'Tap', 'Tearing', 'Telephone', 'Thump_and_thud', 'Thunder', 'Thunderstorm', 'Tick', 'Tick-tock', 'Toilet_flush', 'Tools', 'Traffic_noise_and_roadway_noise', 'Train', 'Trickle_and_dribble', 'Truck', 'Trumpet', 'Typewriter', 'Typing', 'Vehicle', 'Vehicle_horn_and_car_horn_and_honking', 'Walk_and_footsteps', 'Water', 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Whoosh_and_swoosh_and_swish', 'Wild_animals', 'Wind', 'Wind_chime', 'Wind_instrument_and_woodwind_instrument', 'Wood', 'Writing', 'Yell', 'Zipper_(clothing)']\n"
     ]
    }
   ],
   "source": [
    "print(unique_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\aniru\\AppData\\Local\\Temp\\ipykernel_6288\\699929314.py:1: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.\n",
      "  meta_file_test = torch.load(\"./processed/fsd_meta.pth\")\n"
     ]
    }
   ],
   "source": [
    "meta_file_test = torch.load(\"./processed/fsd_meta.pth\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
