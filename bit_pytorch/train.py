# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8


from os.path import join as pjoin  # pylint: disable=g-importing-member
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torchvision as tv

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule

sys.path.append('/data/DNN_data/pytorch-fgvc-dataset')
print(sys.path)
import cub2011

import deeplake

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i


def mktrainval(args, logger):
  """Returns train and validation datasets."""
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  elif args.dataset == "oxford_flowers102":
    train_set = tv.datasets.Flowers102(args.datadir, transform=train_tx, split='train', download=True)
    valid_set = tv.datasets.Flowers102(args.datadir, transform=val_tx, split='val', download=True)
  elif args.dataset == 'cubs':
    train_set = cub2011.Cub2011(args.datadir, transform=train_tx, train=True, download=False)
    valid_set = cub2011.Cub2011(args.datadir, transform=val_tx, train=False, download=False)
  elif args.dataset == 'cars':
    train_set = tv.datasets.StanfordCars(args.datadir, transform=train_tx, split='train', download=False)
    valid_set = tv.datasets.StanfordCars(args.datadir, transform=val_tx, split='test', download=False)
  elif args.dataset == 'wikiart':
    ds = deeplake.load('hub://activeloop/wiki-art')

    # save_directory = os.path.expanduser('/home/data/DNN_data/wikiart_ds/')

    # deeplake.export(save_directory)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")


  # 요것은 어디에 쓰이려나 ..
  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, step):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)
        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())

    # measure elapsed time
    end = time.time()

  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  logger.flush()
  return all_c, all_top1, all_top5


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):

  
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def moving_average(data):
    # 결과를 저장할 리스트
    smoothed = []
    window_size = 50

    # window size는 data 길이로 설정
    for i in range(1, len(data)+1):
        start_idx = max(0, i - window_size)
        smoothed.append(data[start_idx:i].mean().item())

    return torch.tensor(smoothed) # numpy 변수로 리턴


def get_T(max_values_tensor, warm_up):
    # # scale value
    # k = torch.tensor(0.5, dtype=max_values_tensor.dtype, device=max_values_tensor.device)

    # moving avg 적용
    smoothed_max_values = moving_average(max_values_tensor)

    initial = smoothed_max_values[warm_up - 51]
    final = smoothed_max_values[-1]
    diff = (final - initial)
    epsilon = diff / initial 
    mean = (initial + final) / 2

    # 수렴 판단 기준(10% 증가)
    if (epsilon < 0.1):
      return epsilon, mean
    else:
      T_delta = final * epsilon * 1.2
      return epsilon, final + T_delta


def main(args):
  logger = bit_common.setup_logger(args)

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device_id = 3
  device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
  logger.info(f"Loading model from {args.model}.npz")

  # classifier 설정하는 것임. zero_head는 true일 때 head의 가중치를 0으로 보냄.
  if (args.dataset == 'oxford_flowers102'):
    class_counts = len(set(valid_set._labels))
  elif (args.dataset == 'cubs'):
    class_counts = len(valid_set.class_names)
  else:
    class_counts = len(valid_set.classes)

  model = models.KNOWN_MODELS[args.model](head_size=class_counts, zero_head=True)
  model.load_from(np.load(f"{args.model}.npz"))

  logger.info("Moving model onto all GPUs")
  # model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  step = 0

  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")

  model = model.to(device)
  optim.zero_grad()

  model.train()
  mixup = bit_hyperrule.get_mixup(len(train_set))
  cri = torch.nn.CrossEntropyLoss().to(device)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  #AMP(자동 혼합 정밀도)를 위한 GradScaler 초기화
  scaler = torch.cuda.amp.GradScaler()
  
  # Conv layer의 총 parameter 개수 확인
  total_elements1, total_elements2, total_elements3, total_elements4 = 0, 0, 0, 0
  
  for name, param in model.named_parameters():
        if name.startswith("body.block1") and 'conv' in name:
            total_elements1 += param.numel()
        elif name.startswith("body.block2") and 'conv' in name:
            total_elements2 += param.numel()
        elif name.startswith("body.block3") and 'conv' in name:
            total_elements3 += param.numel()
        elif name.startswith("body.block4") and 'conv' in name:
            total_elements4 += param.numel()

  with lb.Uninterrupt() as u:

    # output gradient의 zero ratio 모으기
    all_zero_ratio = {}

    # zero_ratio 모으기
    all_zero_ratio1 = []
    all_zero_ratio2 = []
    all_zero_ratio3 = []
    all_zero_ratio4 = []

    # zero_elements 개수를 저장할 리스트
    all_zero_elements1 = []
    all_zero_elements2 = []
    all_zero_elements3 = []
    all_zero_elements4 = []

    # 학습 루프 예시
    previous_scale = scaler.get_scale()  # 초기 scale 값을 저장

    # warm-up stage 변수 설정
    warmup_step = bit_hyperrule.get_schedule(len(train_set)) # dataset별로 warm-up step 가져오기
    warmup_step = warmup_step[0]

    # customconv2d의 step 1 증가
    for module in model.modules():
      if isinstance(module, models.CustomConv2d):
        module.warmup_step = warmup_step
    logger.info(f"warm_up step : {warmup_step}")

    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break

      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      if mixup > 0.0:
        x, y_a, y_b = mixup_data(x, y, mixup_l)

      # compute output, measure은 시간 측정을 위해 사용
      with chrono.measure("fprop"):
        logits = model(x)
        if mixup > 0.0:
          c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
        else:
          c = cri(logits, y)
        c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # Accumulate grads
      with chrono.measure("grads"):
        scaler.scale(c / args.batch_split).backward()
        accum_steps += 1

      # # Apply thresholds and calculate zero ratio
      # zero_elements1 = torch.tensor(0, device=device)
      # zero_elements2 = torch.tensor(0, device=device)
      # zero_elements3 = torch.tensor(0, device=device)
      # zero_elements4 = torch.tensor(0, device=device)

      # # scaler.unscale_(optim)  # Unscale the gradients before modifying them

      # ################### Conv Gradient에 threshold 적용 ############################
      # for name, param in model.named_parameters():
      #   if name.startswith("body.block1") and 'conv' in name:
      #       zero_elements1 += (param.grad.data == 0).sum()
      #   elif name.startswith("body.block2") and 'conv' in name:
      #       zero_elements2 += (param.grad.data == 0).sum()
      #   elif name.startswith("body.block3") and 'conv' in name:
      #       zero_elements3 += (param.grad.data == 0).sum()
      #   elif name.startswith("body.block4") and 'conv' in name:
      #       zero_elements4 += (param.grad.data == 0).sum()
      # #####################################################################################
      
      # # Transfer the results to CPU
      # zero_elements1 = zero_elements1.item()
      # zero_elements2 = zero_elements2.item()
      # zero_elements3 = zero_elements3.item()
      # zero_elements4 = zero_elements4.item()

      # #6-10. zero ratio 계산 및 저장
      # zero_ratio1 = 100.0 * zero_elements1 / total_elements1 if total_elements1 > 0 else 0
      # zero_ratio2 = 100.0 * zero_elements2 / total_elements2 if total_elements2 > 0 else 0
      # zero_ratio3 = 100.0 * zero_elements3 / total_elements3 if total_elements3 > 0 else 0
      # zero_ratio4 = 100.0 * zero_elements4 / total_elements4 if total_elements4 > 0 else 0

      # # step마다 zero_ratio 모으는 부분
      # all_zero_ratio1.extend([zero_ratio1])  # block1
      # all_zero_ratio2.extend([zero_ratio2])  # block2
      # all_zero_ratio3.extend([zero_ratio3])  # block3
      # all_zero_ratio4.extend([zero_ratio4])  # block4

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      # logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e}) zero_ratio1: {zero_ratio1:.5f}% zero_ratio2: {zero_ratio2:.5f}% zero_ratio3: {zero_ratio3:.5f}% zero_ratio4: {zero_ratio4:.5f}%")  # pylint: disable=logging-format-interpolation  
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")

      output_gradients = models.get_output_gradient()
      
      # 각 layer의 output gradient의 zero ratio 확인
      for module, grad_output in output_gradients.items():
          if module not in all_zero_ratio:
            all_zero_ratio[module] = torch.tensor([], device='cpu')

          out_zero_elements = (grad_output == 0).sum().item()
          out_total_elements = grad_output.numel()
          zero_ratio = 100.0 * out_zero_elements / out_total_elements if out_total_elements > 0 else 0
          zero_ratio = torch.tensor([zero_ratio], device='cpu')  # CPU 텐서로 저장

          if step >= warmup_step:
            # if (step == warmup_step):
            #   logger.info(f"{step}부터 all_zero_ratio에 저장.")
            #   logger.info(f"warmup_step: {warmup_step}")
            all_zero_ratio[module] = torch.cat([zero_ratio, all_zero_ratio[module]], dim=0)
            logger.info(f"Layer {module} - Output Gradient Zero Ratio: {zero_ratio.item():.5f}%, out_zero_elements : {out_zero_elements}, out_total_elements : {out_total_elements}")
      logger.flush()

      # if (step == 2000) and (accum_steps == 1):
      #   grad_out_dist = models.get_output_gradient() # 4000~6000 step에서 max value 가져옴

      #   for layer_id, grad_out in grad_out_dist.items(): # 딕셔너리에서 하나씩 꺼내와서 npz 파일로 저장
      #     save_path = pjoin(args.logdir, args.name, 'grad_out_2000', f"grad_out_{layer_id}_2000.npz")
      #     np.savez(save_path, data=grad_out.cpu().numpy())
      #   logger.info("npz file 저장 완료.")

      # if (step == 9999) and (accum_steps == 1):
      #   grad_out_dist = models.get_output_gradient() # 4000~6000 step에서 max value 가져옴

      #   for layer_id, grad_out in grad_out_dist.items(): # 딕셔너리에서 하나씩 꺼내와서 npz 파일로 저장
      #     save_path = pjoin(args.logdir, args.name, 'grad_out_10000', f"grad_out_{layer_id}_10000.npz")
      #     np.savez(save_path, data=grad_out.cpu().numpy())
      #   logger.info("npz file 저장 완료.")

      # if (step == 3999) and (accum_steps == args.batch_split):
      #   avg_max_values = models.get_avg_max_values() # 4000~6000 step에서 max value 가져옴

      #   for layer_id, avg_value in avg_max_values.items(): # 딕셔너리에서 하나씩 꺼내와서 npz 파일로 저장
      #     save_path = pjoin(args.logdir, args.name, 'avg_max_values', f"avg_value_{layer_id}_4000.npz")
      #     np.savez(save_path, data=avg_value.cpu().numpy())
      #   logger.info("평균값 저장 완료.")

      # if (step == 9999) and (accum_steps == args.batch_split):
      #   avg_max_values = models.get_avg_max_values() # 4000~6000 step에서 max value 가져옴

      #   for layer_id, avg_value in avg_max_values.items(): # 딕셔너리에서 하나씩 꺼내와서 npz 파일로 저장
      #     save_path = pjoin(args.logdir, args.name, 'avg_max_values_10000', f"avg_value_{layer_id}_10000.npz")
      #     np.savez(save_path, data=avg_value.cpu().numpy())
      #   logger.info("평균값 저장 완료.")

      # if (step == 499) and (accum_steps == args.batch_split):
      #   avg_max_values = models.get_avg_max_values() # 4000~6000 step에서 max value 가져옴

      #   for layer_id, avg_value in avg_max_values.items(): # 딕셔너리에서 하나씩 꺼내와서 npz 파일로 저장
      #     save_path = pjoin(args.logdir, args.name, 'avg_max_values_500', f"avg_value_{layer_id}_500.npz")
      #     np.savez(save_path, data=avg_value.cpu().numpy())
      #   logger.info("평균값 저장 완료.")

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          scaler.step(optim)
          scaler.update()
          optim.zero_grad()
        step += 1

        # 현재 scale 값을 가져와 이전 scale 값과 비교
        current_scale = scaler.get_scale()
        scale_factor = current_scale / previous_scale  # 이전 scale에 대한 증가 또는 감소 비율 계산

        # custom layer의 T 값을 조정 (예: T 값을 scale 변화에 맞춰 증가 또는 감소)
        if scale_factor != 1:
          for module in model.modules():
            if isinstance(module, models.CustomConv2d) :
              module.T *= scale_factor
              logger.info(f"Layer {module.layer_id} - Scale_facor : {scale_factor} Updated T: {module.T}")

        # 다음 스텝에서 사용할 수 있도록 이전 scale 값 업데이트
        previous_scale = current_scale
        
        # customconv2d의 step 1 증가
        for module in model.modules():
          if isinstance(module, models.CustomConv2d):
            module.steps += 1
            
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(model, valid_loader, device, chrono, logger, step)
          if args.save:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim" : optim.state_dict(),
            }, savename)

      end = time.time()

      # warm-up단계에서 max value를 담고 있는 리스트 파일로 저장
      if (step == warmup_step) and (accum_steps == 0):
          accum_c_max_values = models.get_max_tensor() # 리스트 받아오기

          for layer_id, max_values_tensor in accum_c_max_values.items():
              # # max_values NPZ 파일로 압축하여 저장
              # save_path = pjoin(args.logdir, args.name, 'max_values', f"max_values_{layer_id}.npz")
              # np.savez(save_path, data=accum_c_max_values[layer_id].cpu().numpy())
              print(f"max_values_tensor: {max_values_tensor.shape}")
              epsilon, threshold = get_T(max_values_tensor, warmup_step)

              # 각 레이어의 T 값을 설정
              for module in model.modules():
                  if isinstance(module, models.CustomConv2d) and module.layer_id == layer_id:
                      module.T = threshold # avg value로 thresholding 적용
                      # module.T = 0 # Thresholding 생략
                      logger.info(f"Set T for layer {layer_id} to {module.T} and epsilon: {epsilon}") # 모니터링


    # # Final eval at end of training.
    # #6-13. Final eval at end of training.
    # avg_zero_ratio1 = np.mean(all_zero_ratio1) if all_zero_ratio1 else 0
    # avg_zero_ratio2 = np.mean(all_zero_ratio2) if all_zero_ratio2 else 0
    # avg_zero_ratio3 = np.mean(all_zero_ratio3) if all_zero_ratio3 else 0
    # avg_zero_ratio4 = np.mean(all_zero_ratio4) if all_zero_ratio4 else 0

    for layer_id, zero_ratio in all_zero_ratio.items():
      avg_zero_ratio = torch.mean(zero_ratio).item()
      logger.info(f"Final average zero_ratio_{layer_id} : {avg_zero_ratio:.5f}%")

    run_eval(model, valid_loader, device, chrono, logger, step='end')

  # logger.info(f"Final Average zero_ratio1: {avg_zero_ratio1:.5f}% zero_ratio2: {avg_zero_ratio2:.5f}% zero_ratio3: {avg_zero_ratio3:.5f}% zero_ratio4: {avg_zero_ratio4:.5f}% ")
  logger.info(f"Timings:\n{chrono}")



if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  
  main(parser.parse_args())
