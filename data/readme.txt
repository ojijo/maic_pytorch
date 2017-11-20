To build RTF
	build_TFR.py
	--train_image_dir="ai-data/images"   --val_image_dir="ai-data/v_images"  --train_captions_file="ai-data/caption.txt"  --val_captions_file="ai-data/v_caption.txt"  --output_dir="ai-data/output"  --word_counts_output_file="ai-data/output/word_counts.txt"

To build HDF5
	1. build_HDF_json
	  		dataset = _load_and_process_metadata("ai-data/caption.txt", "ai-data/images")
  			json.dump(dataset, open('ai-data/output/coco_raw.json', 'w'))
	2. build_HDF
			