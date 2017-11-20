from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import jieba
import argparse

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.

  Returns:
    A list of strings; the tokenized caption.
  """
  tokenized_caption = ["<S>"]
  
  ########## English words 
  # tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  # print(len(caption))
  
  ##########cut into words using jieba
  words = jieba.cut(caption)
  for token in words:
      tokenized_caption.append(token)  
  
  ##########cut into character directly    
#   for token in caption:
#       tokenized_caption.append(token)  
      
  tokenized_caption.append("</S>")
  return tokenized_caption


def _load_and_process_metadata(captions_file, image_dir):
  """Loads image metadata from a JSON file and processes the captions.

  Args:
  3  captions_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.

  Returns:
    A list of ImageMetadata.
  """
  with open(captions_file, "r") as f:
    caption_data = json.load(f)

  # Extract the filenames.
  # id_to_filename = [(x["image_id"], x["image_id"]) for x in caption_data]
  id_to_filename = [(100000 + i, x["image_id"]) for (i, x) in enumerate(caption_data)]
  filename_to_id = dict([(x, y) for (y, x) in id_to_filename])

  # Extract the captions. Each image_id is associated with multiple captions.
  id_to_captions = {}
  for annotation in caption_data:
    image_id = filename_to_id[ annotation["image_id"]]
    caption = annotation["caption"]
    # take place for this id
    id_to_captions.setdefault(image_id, caption)
    # add one caption for this id,so this id may have multiple captions
    # id_to_captions[image_id].append(caption)

  assert len(id_to_filename) == len(id_to_captions)
  assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
  print("Loaded caption metadata for %d images from %s" % 
        (len(id_to_filename), captions_file))

  # Process the captions and combine the data into a list of ImageMetadata.
  print("Processing captions.")
  image_metadata = []
  num_captions = 0
  for image_id, base_filename in id_to_filename:
    jimg = {}  
    jimg['file_path'] = os.path.join(image_dir, base_filename)
    jimg['id'] = image_id

    jimg['captions'] = [_process_caption(c) for c in id_to_captions[image_id]]
    image_metadata.append(jimg)
    num_captions += len(jimg['captions'])
  print("Finished processing %d captions for %d images in %s" % 
        (num_captions, len(id_to_filename), captions_file))

  return image_metadata


def main(params):
  
  # Load image metadata from caption files.
  dataset = _load_and_process_metadata(params['input_captions'], params['input_image_dir'])
  json.dump(dataset, open(params['output_json'], 'w'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_captions', default='./ai-data/caption.txt', help='input caption file')
  parser.add_argument('--input_image_dir', default='./ai-data/images', help='input image directory')
  parser.add_argument('--output_json', default='./ai-data/output/coco_raw.json', help='raw json file')
  args = parser.parse_args()
  params = vars(args)  # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent=2))
  main(params)
