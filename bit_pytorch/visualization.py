import numpy as np
import matplotlib.pyplot as plt

# .npz 파일 불러오기
npz_file_path = "./log_0821/dist_test/gradient_block1.npz"
data = np.load(npz_file_path, allow_pickle=True)

# 각 블록의 gradient summations 추출
grad_block1 = data['block1']
# grad_block2 = data['block2']
# grad_block3 = data['block3']
# grad_block4 = data['block4']

# 시각화 함수 정의
def plot_gradient_summations(grad_block, block_name):
    plt.figure(figsize=(10, 6))
    
    for step_idx, summation in enumerate(grad_block):
        plt.plot(summation, label=f"Step {step_idx * 100}")

    plt.title(f"{block_name}의 Gradient Summation")
    plt.xlabel("채널 인덱스")
    plt.ylabel("Gradient Summation")
    plt.legend()
    plt.show()

# # 각 블록에 대해 시각화 수행
# plot_gradient_summations(grad_block1, "블록 1")
# plot_gradient_summations(grad_block2, "블록 2")
# plot_gradient_summations(grad_block3, "블록 3")
# plot_gradient_summations(grad_block4, "블록 4")

print(grad_block1[1].shape)

for i in grad_block1:
    print(i.shape)
