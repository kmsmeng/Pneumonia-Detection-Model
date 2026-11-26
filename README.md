## Pneumonia Detection Model

Deep learning model for detecting pneumonia from chest X-rays images.
Fine Tuned YOLO-26 on RSNA Pneumonia Dataset.
An Object Detection Model that outputs bounding box for pneumonia spots on X-Ray Scans Input.

### Setup

1. Clone the repository
git clone https://github.com/kmsmeng/Pneumonia-Detection-Model.git
cd Pneumonia-Detection-Model

2. Create virtual environment (Python 3.13)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

3. Set up PyTorch (install this BEFORE requirements.txt)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

This is the CUDA 13.0 build. If your GPU/driver supports a different CUDA version:
- run 'nvidia-smi' in a terminal
- it will show something like 'CUDA Version: 13.0'. That is your Compute Platform.
- pick the matching command at https://pytorch.org/get-started/locally/ (keep torch==2.10.0)

4. Install dependencies (versions are pinned)
pip install -r requirements.txt

5. To Download the Kaggle Dataset(RSNA Pneumonia Dataset)
- create a .env file
- get your 'username' and 'key' from the kaggle
- example .env file:

KAGGLE_USERNAME=yourusername
KAGGLE_KEY=stringkey

- after that the first code block will download the dataset locally
