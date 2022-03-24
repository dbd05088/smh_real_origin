"""
_ainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
# -*- coding: utf-8 -*-
import logging.config
import os
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch import nn
from pseudo_main import MetaPseudo
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from augmentation import RandAugmentCIFAR
from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
from pseudo_main import MetaPseudo

cifar100_mean = (0.507075, 0.486549, 0.440918)
cifar100_std = (0.267334, 0.256438, 0.276151)
cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
cifar_mean = 0
cifar_std = 0

class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugmentCIFAR(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


def make_blurry(major_ratio, num_labeled, num_classes):
    train = pd.read_json('./dataset/train_json.json')
    test = pd.read_json('./dataset/test_json.json')
    rnd_seed = 3  # random seed
    num_tasks = 5  # the number of tasks.
    np.random.seed(rnd_seed)

    klass = train.klass.unique()
    num_cls_per_task = len(klass) // num_tasks
    # print(num_cls_per_task) # cifar10이므로 10/5 = 2
    np.random.shuffle(klass)
    # enumerate : index 번호와 class를 tuple의 형태로 반환
    # house : 0, cat : 1 이런식으로 순서대로 배정해주는 것
    class2label = {cls_: idx for idx, cls_ in enumerate(klass)}
    '''
    apply와 lambda
    lambda를 하면 대입을 해야하는 것을 대신해주게 된다. 여기서는 x에 class2label[x]를 넣어주는 역할을 한줄의 코드로 가능하게 만들어준다.
    원래 x에 0~9까지의 숫자가 들어 있었기 때문에, 각 class에 해당하는 애를 matching 시켜줌
    이를 바탕으로 label이라는 column을 새롭게 matching 시켜줌
    '''
    train["label"] = train.klass.apply(lambda x: class2label[x])
    test["label"] = test.klass.apply(lambda x: class2label[x])

    task_class = np.split(klass, num_tasks)  # task별로 class를 나눠줌 (2개씩!!)

    # list comprehension
    '''
    [(변수를 활용할 방법) for (사용할 변수 이름) in (순회할 수 있는 값)]
    '''
    # task별로 쪼갠 것 (isin을 통해서 tc에 있는 애들은 true, 없으면 false를 반환하게 하고, train[true]인 애들끼리 묶어줌)
    # 이를 통해서 task별로 train data와 test data들을 묶음 단위로 쪼개줄 수 있다.
    task_train = [train[train.klass.isin(tc)] for tc in task_class]
    task_test = [test[test.klass.isin(tc)] for tc in task_class]
    # major_ratio = 0.9 # 0.9 for blurry-10, 0.7 for blurry-30.
    # num_labeled = 4000 # cifar10 train total 50000
    # num_classes = 10 # cifar10
    major_classes = []
    label_per_class = num_labeled // num_classes

    task_trainM = []
    task_trainN = []
    for t in task_train:
        major_classes.append(list(t.klass.unique()))
        sub_task_trainN = []

        # sample 함수를 통해서 M%에 해당하는 data 추출
        taskM = t.sample(n=int(len(t) * major_ratio), replace=False)
        taskN = pd.concat([taskM, t]).drop_duplicates(keep=False)
        taskN_size = len(taskN)

        task_trainM.append(taskM)

        # 각각의 task에서 M/5%씩 추출
        for _ in range(len(task_train)-1):
            sub_task_trainN.append(taskN.sample(
                n=taskN_size//(len(task_train)-1)))

        task_trainN.append(sub_task_trainN)

    task_mixed_train = []
    for idx, task in enumerate(task_trainM):
        other_task_samples = pd.DataFrame()
        for j in range(len(task_trainM)):
            if idx != j:
                other_task_samples = pd.concat(
                    [other_task_samples, task_trainN[j].pop(0)])
        mixed_task = pd.concat([task, other_task_samples])
        task_mixed_train.append(mixed_task)

    labeled_idx = []
    total_data = []
    for idx, data_per_task in enumerate(task_mixed_train):
        major_class = major_classes[idx]
        labeled_data = []
        
        # unlabeled data는 전체 data
        unlabeled_data = data_per_task
        
        for mc in major_class:
            index = list(np.where(data_per_task.klass == mc)[0])
            np.random.seed(rnd_seed)
            index = np.random.choice(index, label_per_class, False) # choice 함수 : index에서 label_per_class만큼 choose하고 replace = False
            
            # labeled data로 일부 추출
            for i in index:
                labeled_data.append(task_train[idx].iloc[i, :])
                
            
        # test data 생성 (previous task ~ current task까지)
        current_task_test = task_test[0:idx+1]
        test_df = pd.DataFrame()
        for test_idx, task in enumerate(current_task_test):
            if test_idx==0:
                test_df = pd.DataFrame(task)
            else:
                test_df = pd.concat([test_df, pd.DataFrame(task)])
        total_data.append((pd.DataFrame(labeled_data), pd.DataFrame(unlabeled_data), pd.DataFrame(task_test[idx]), test_df))
    return total_data

def main():
    args = config.base_parser()

    # blurry 10을 구현하고 싶으면 첫 parameter에 0.9 / disjoint 원하면 1.0 대입
    if args.dataset == "cifar10":
        num_class = 10
        cifar_mean = cifar10_mean
        cifar_std = cifar10_std
        
    elif args.dataset == "cifar100":
        num_class = 100
        cifar_mean = cifar100_mean
        cifar_std = cifar100_std

    else:
        num_class = 1000
    total_data = make_blurry(0.9, args.num_labeled, num_class)

    # make_blurry 작동 확인
    # print("------------make_blurry------------")
    # print(len(total_data))

    # Save file name
    tr_names = ""
    for trans in args.transforms:
        tr_names += "_" + trans
    save_path = f"{args.dataset}/{args.mode}_{args.mem_manage}_{args.stream_env}_msz{args.memory_size}_rnd{args.rnd_seed}{tr_names}"

    logging.config.fileConfig("./configuration/logging.conf")

    # logger = logging.getLogger("name") 꼴로 logging instance 출력
    logger = logging.getLogger()

    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    # handler 객체 생성
    fileHandler = logging.FileHandler(
        "logs/{}.log".format(save_path), mode="w")

    # formatter 객체 생성
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )

    # handler에 format 설정
    fileHandler.setFormatter(formatter)

    # logger에 생성한 handler 추가
    logger.addHandler(fileHandler)

    # pytorch로 tensor board를 사용하기 위해서는 summary writer instance를 생성해야 한다.
    # writer = SummaryWriter("tensorboard")

    # gpu 사용 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    # 각 dataset에 대해서 평균, 표준편자 등등의 정보를 담고 있는 우리가 정의한 함수
    # mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)

    n, m = 2, 10  # default

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    transform_unlabeled = TransformMPL(
        args, mean=cifar_mean, std=cifar_std)

    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std)
    ])

    logger.info(f"Using train-transforms {transform_labeled}")


    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
    if "autoaug" in args.transforms:
        train_transform.append(select_autoaugment(args.dataset))

    rm_train_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    logger.info(f"Using train-transforms {train_transform}")

    rm_test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, transform_labeled, transform_unlabeled, transform_test, 10, rm_train_transform, rm_test_transform
    )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)

    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        task_acc = 0.0
        eval_dict = dict()

        # get datalist
        '''
        cur_labeled_train_datalist, cur_unlabeled_train_datalist = get_train_datalist(
            args, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)
        '''
        cur_labeled_train_datalist = total_data[cur_iter][0]
        cur_unlabeled_train_datalist = total_data[cur_iter][1]
        cur_pseudo_test_datalist = total_data[cur_iter][2]
        cur_test_datalist = total_data[cur_iter][3]
        metapseudo = MetaPseudo(method,
                                args, cur_labeled_train_datalist, cur_unlabeled_train_datalist, cur_pseudo_test_datalist, transform_labeled, transform_unlabeled, transform_test, rm_train_transform)

        pseudo_images, pseudo_targets = metapseudo.train_loop()
        # print("**********pseudo*********")
        # print("input_len", len(pseudo_images))
        # print("target_len", len(pseudo_targets))

        # Reduce datalist in Debug mode
        if args.debug:
            random.shuffle(cur_labeled_train_datalist)
            random.shuffle(cur_unlabeled_train_datalist)
            random.shuffle(cur_test_datalist)
            # debug mode에서는 일부만 사용
            # labeled data는 어차피 적으니깐 전부 다 사용
            cur_unlabeled_train_datalist = cur_unlabeled_train_datalist[:2560]
            cur_test_datalist = cur_test_datalist[:2560]

        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(
            cur_labeled_train_datalist, cur_test_datalist, pseudo_images, pseudo_targets)
        # Increment known class for current task iteration.
        method.before_task(cur_labeled_train_datalist, cur_unlabeled_train_datalist,
                           cur_iter, args.init_model, args.init_opt)

        # The way to handle streamed samles
        logger.info(f"[2-3] Start to train under {args.stream_env}")

        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            # Offline Train
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=16,
            )
            if args.mode == "joint":
                logger.info(f"joint accuracy: {task_acc}")

        elif args.stream_env == "online":
            # Online Train
            logger.info("Train over streamed data once")

            # meta pseudo label까지 포함해서 RM training
            # train
            method.train(
                cur_iter=cur_iter,
                n_epoch=1,
                batch_size=args.batchsize,
                n_worker=16,
            )

            method.update_memory(cur_iter)

            # No stremed training data, train with only memory_list
            method.set_current_dataset([], cur_test_datalist, [], [])

            logger.info("Train over memory")
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=64,
                n_worker=16,
            )

            method.after_task(cur_iter)

        logger.info("[2-4] Update the information for the current task")
        method.after_task(cur_iter)
        task_records["task_acc"].append(task_acc)
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        # Notify to NSML
        logger.info("[2-5] Report task result")
        print("Metrics/TaskAcc", task_acc, cur_iter)

    # np.save(f"results/{save_path}.npy", task_records["task_acc"])

    # Accuracy (A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    acc_arr = np.array(task_records["cls_acc"])
    # cls_acc = (k, j), acc for j at k
    cls_acc = acc_arr.reshape(-1,
                              args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    for k in range(args.n_tasks):
        forget_k = []
        for j in range(args.n_tasks):
            if j < k:
                forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
            else:
                forget_k.append(None)
        task_records["forget"].append(forget_k)
    F_last = np.mean(task_records["forget"][-1][:-1])

    # Intrasigence (I)
    I_last = args.joint_acc - A_last

    logger.info(f"======== Summary =======")
    logger.info(
        f"A_last {A_last} | A_avg {A_avg} | F_last {F_last} | I_last {I_last}")


if __name__ == "__main__":
    main()

