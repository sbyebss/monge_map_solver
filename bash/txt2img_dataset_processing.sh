set -e

# Processing Datasets with img2dataset
# Reference: https://github.com/rom1504/img2dataset

# Set the starting step (default to 1 if not specified)
starting_step=${1:-1}

dataset_path=$(pwd)/datasets
echo "Starting Dataset Processing"

# --------------------- LAION DATASET PROCESSING ---------------------
echo "Processing LAION Dataset..."

# Define the dataset location
dataset_location=$dataset_path/laion-art_test
mkdir -p $dataset_location

# We need 71GB of space for the cc3m dataset
# Convert the required size to kilobytes (1 GB = 1024 * 1024 KB)
required_size=$(( (71) * 1024 * 1024 ))

# Get the available space in the file system where dataset_location resides, in kilobytes
available_space=$(df -k --output=avail "$dataset_location" | tail -n1)

# Convert the available space to gigabytes for display
available_space_gb=$(echo "scale=2; $available_space / 1024 / 1024" | bc)

# Check if the available space is greater than or equal to the required size
if [ "$available_space" -ge "$required_size" ]; then
    echo "Storage check passed: The drive has sufficient available space (${available_space_gb} GB)."
else
    echo "Storage check failed: The drive does not have enough available space. Required: 179 GB, Available: ${available_space_gb} GB"
    # Exit the script or handle the error as needed
    exit 1
fi

mkdir -p $dataset_location/laion-art

if [ "$starting_step" -le 1 ]; then
    # Step 1: Download the laion-art dataset
    echo "Downloading laion-art dataset..."
    wget https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet -O "${dataset_location}/laion-art/laion-art.parquet"
fi

if [ "$starting_step" -le 2 ]; then
    # Step 2: Preprocess the dataset
    echo "Preprocessing the dataset..."
    python src/scripts/data_preprocessing.py "${dataset_location}/laion-art" "${dataset_location}/laion-art-en"
fi


if [ "$starting_step" -le 3 ]; then
    # Step 3: Use img2dataset to process the dataset
    cd $dataset_location
    echo "Running img2dataset for LAION..."
    img2dataset \
        --url_list "${dataset_location}/laion-art-en" \
        --input_format "parquet" \
        --url_col "URL" \
        --caption_col "TEXT" \
        --output_format webdataset \
        --output_folder "${dataset_location}/laion-high-resolution-en" \
        --processes_count 16 \
        --thread_count 64 \
        --image_size 224 \
        --min_image_size 128 \
        --max_aspect_ratio 1.5 \
        --resize_only_if_bigger=True \
        --resize_mode="keep_ratio" \
        --skip_reencode=True \
        --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
        --enable_wandb False
fi

# Get the CLIP embeddings of the images
if [ "$starting_step" -le 4 ]; then
    echo "Extracting CLIP embeddings..."
    cd $dataset_location/laion-high-resolution-en
    clip-retrieval inference \
        --input_dataset "{00000..328}.tar" \
        --output_folder clip_emb \
        --input_format "webdataset" \
        --enable_metadata True \
        --clip_model ViT-L/14 \
        --enable_wandb False
fi

if [ "$starting_step" -le 5 ]; then
    # Step 3: Reorder the image embeddings
    echo "Reordering image embeddings..."
    cd $dataset_location/laion-high-resolution-en/clip_emb/metadata
    reorder-embeddings example-key --metadata-folder .
fi

cd $dataset_location/laion-high-resolution-en/clip_emb
mkdir -p img_emb_reorder text_emb_reorder

echo "Reordering image embeddings - Images..."
reorder-embeddings reorder \
    --embeddings-folder img_emb \
    --metadata-folder metadata \
    --output-shard-width 5 \
    --index-width 4 \
    --output-folder img_emb_reorder \
    --verbose True \
    --limit 330

echo "Reordering image embeddings - Texts..."
reorder-embeddings reorder \
    --embeddings-folder text_emb \
    --metadata-folder metadata \
    --output-shard-width 5 \
    --index-width 4 \
    --output-folder text_emb_reorder \
    --verbose True \
    --limit 330

echo "LAION dataset embedding reordering completed."

# --------------------- CC3M_NO_WATERMARK DATASET PROCESSING ---------------------
echo "Processing CC3M_NO_WATERMARK Dataset..."

dataset_location=$dataset_path/cc3m_test
mkdir -p $dataset_location

# We need 108GB of space for the cc3m dataset
# Convert the required size to kilobytes (1 GB = 1024 * 1024 KB)
required_size=$(( (108) * 1024 * 1024 ))

# Get the available space in the file system where dataset_location resides, in kilobytes
available_space=$(df -k --output=avail "$dataset_location" | tail -n1)

# Convert the available space to gigabytes for display
available_space_gb=$(echo "scale=2; $available_space / 1024 / 1024" | bc)

# Check if the available space is greater than or equal to the required size
if [ "$available_space" -ge "$required_size" ]; then
    echo "Storage check passed: The drive has sufficient available space (${available_space_gb} GB)."
else
    echo "Storage check failed: The drive does not have enough available space. Required: 179 GB, Available: ${available_space_gb} GB"
    # Exit the script or handle the error as needed
    exit 1
fi


cd $dataset_location

# Download the cc3m_no_watermark.tsv metadata
gdown https://drive.google.com/uc?id=1Y2FWTqJDfT2CTWgwkkAf1H1zAeD1FXSS

# If you encountered "Access denied with the following error:
#         Cannot retrieve the public link of the file. You may need to change
#         the permission to 'Anyone with the link', or have had many accesses."
# You can download manually https://drive.google.com/file/d/1Y2FWTqJDfT2CTWgwkkAf1H1zAeD1FXSS/view?usp=sharing
# to local folder $dataset_location

# Step 1: Download and process the cc3m dataset
echo "Running img2dataset for CC3M_NO_WATERMARK..."
img2dataset \
    --url_list cc3m_no_watermark.tsv \
    --input_format "tsv" \
    --url_col "url" \
    --caption_col "caption" \
    --output_format webdataset \
    --output_folder cc3m_no_watermark \
    --processes_count 16 \
    --thread_count 64 \
    --image_size 224 \
    --min_image_size 128 \
    --max_aspect_ratio 1.5 \
    --resize_only_if_bigger=True \
    --resize_mode="keep_ratio" \
    --skip_reencode=True \
    --enable_wandb False

# Step 2: Get the CLIP embeddings of the images
echo "Extracting CLIP embeddings for CC3M_NO_WATERMARK..."
cd $dataset_location/cc3m_no_watermark
clip-retrieval inference \
    --input_dataset "{00000..167}.tar" \
    --output_folder clip_emb \
    --input_format "webdataset" \
    --enable_metadata True \
    --clip_model ViT-L/14 \
    --enable_wandb False

# Step 3: Reorder the embeddings
echo "Reordering embeddings for CC3M_NO_WATERMARK..."
cd $dataset_location/cc3m_no_watermark/clip_emb
mkdir -p img_emb_reorder text_emb_reorder

echo "Reordering embeddings - Images..."
reorder-embeddings reorder \
    --embeddings-folder img_emb \
    --metadata-folder metadata \
    --output-shard-width 5 \
    --index-width 4 \
    --output-folder img_emb_reorder \
    --verbose True \
    --limit 167

echo "Reordering embeddings - Texts..."
reorder-embeddings reorder \
    --embeddings-folder text_emb \
    --metadata-folder metadata \
    --output-shard-width 5 \
    --index-width 4 \
    --output-folder text_emb_reorder \
    --verbose True \
    --limit 167

echo "CC3M_NO_WATERMARK dataset processing completed."

echo "Dataset processing script completed."
