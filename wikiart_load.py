import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# 로컬 저장 경로 설정
local_dir = "/data/DNN_data/wikiart_imagefolder"

# ImageFolder 형식으로 데이터 저장 함수
def save_as_imagefolder(dataset, base_dir, label_key="artist"):
    """
    Hugging Face Dataset을 ImageFolder 형식으로 저장.
    :param dataset: Hugging Face Dataset
    :param base_dir: 데이터를 저장할 기본 디렉토리
    :param label_key: 레이블 키 (예: 'artist', 'style', 'genre')
    """
    os.makedirs(base_dir, exist_ok=True)

    for idx, data in enumerate(dataset):
        image = data["image"]
        label = data[label_key]

        # 레이블별 디렉토리 생성
        label_dir = os.path.join(base_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # 이미지 저장 경로
        image_path = os.path.join(label_dir, f"{idx}.jpg")

        # 이미지 저장
        image.save(image_path)

print("Downloading dataset...")
os.makedirs(local_dir, exist_ok=True)
dataset = load_dataset("huggan/wikiart")

# Train 데이터만 사용하므로 이를 8:2로 분할
train_data, valid_data = train_test_split(
    dataset["train"].to_pandas(),
    test_size=0.2,
    stratify=dataset["train"]["artist"],  # artist 레이블 기준 분할
    random_state=42
)

# 다시 Dataset 형식으로 변환
train_data = dataset["train"].select(train_data.index)
valid_data = dataset["train"].select(valid_data.index)

# Train 및 Validation 데이터를 ImageFolder 형식으로 저장
train_dir = os.path.join(local_dir, "train")
valid_dir = os.path.join(local_dir, "valid")

save_as_imagefolder(train_data, train_dir, label_key="artist")
save_as_imagefolder(valid_data, valid_dir, label_key="artist")

print(f"Dataset saved to {local_dir} in ImageFolder format.")

# 저장된 데이터 확인
print(f"Train data saved in: {os.path.join(local_dir, 'train')}")
print(f"Validation data saved in: {os.path.join(local_dir, 'valid')}")
