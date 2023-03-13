import os
import sys
from collections import defaultdict
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

from ml.models import YakuPredictor
from ml.datasets import YakuDataset

# A logger for this file
logger = getLogger(__name__)


class Trainer(object):
    def __init__(
        self,
        config,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device=torch.device("cpu"),
    ):
        self.config = config
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.finish_train = False
        self.writer = SummaryWriter(config.out_dir)
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        torch.autograd.set_detect_anomaly(True)
        self.tqdm = tqdm(
            initial=self.steps, total=self.config.train.train_max_steps, desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "yaku_predictor": self.optimizer["yaku_predictor"].state_dict(),
            },
            "scheduler": {
                "yaku_predictor": self.scheduler["yaku_predictor"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "yaku_predictor": self.model["yaku_predictor"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["yaku_predictor"].load_state_dict(state_dict["model"]["yaku_predictor"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["yaku_predictor"].load_state_dict(
                state_dict["optimizer"]["yaku_predictor"]
            )
            self.scheduler["yaku_predictor"].load_state_dict(
                state_dict["scheduler"]["yaku_predictor"]
            )

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_ = self.model["yaku_predictor"](x)

        # bce loss
        bce_loss = self.criterion["bce"](y_, y)
        self.total_train_loss["train/bce_loss"] += bce_loss.item()

        # backward
        self.optimizer["yaku_predictor"].zero_grad()
        bce_loss.backward()
        if self.config.train.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["yaku_predictor"].parameters(),
                self.config.train.grad_norm,
            )
        self.optimizer["yaku_predictor"].step()
        self.scheduler["yaku_predictor"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logger.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_ = self.model["yaku_predictor"](x)

        # bce loss
        bce_loss = self.criterion["bce"](y_, y)
        self.total_train_loss["eval/bce_loss"] += bce_loss.item()

        # total accuracy (all classification predicts are accurate)
        total_acc = self.criterion["total_acc"](y, y_)
        self.total_train_loss["eval/total_acc"] += total_acc.item()

        # binary accuracy
        binary_acc = self.criterion["binary_acc"](y, y_)
        self.total_train_loss["eval/binary_acc"] += binary_acc.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["valid"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config.train.save_interval_steps == 0:
            self.save_checkpoint(
                os.path.join(
                    self.config.out_dir,
                    "checkpoints",
                    f"checkpoint-{self.steps}steps.pkl",
                )
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config.train.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.train.log_interval_steps
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.train.train_max_steps:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        ignore
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            ignore (list): list iof ignored yaku id
        """
        # 55 is all yaku count
        index = np.ones(55, dtype=bool)
        index[ignore] = False
        self.use_yaku_id = index

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of obs and yaku.

        Returns:
            Tensor: feature batch (B, 144).
            Tensor: achieved yaku batch (B, 55).
        """
        obs_batch = []
        y_batch = []
        for idx in range(len(batch)):
            filename, obs, y = batch[idx]
            seat = int(os.path.basename(os.path.dirname(filename)))  # 0: 東 in 東一局
            current_hand = np.sum(obs[0:7, :], axis=0)
            tsumogiri = np.sum(obs[13:16, :], axis=0)
            discarded_from_hand = np.sum(obs[25:28, :], axis=0)
            opened = np.sum(obs[37:44, :], axis=0)  # furo
            round_wind = np.zeros(4)
            round_ = int(np.sum(obs[103: 110, 0]))
            round_wind[round_ // 4] = 1
            own_wind = np.zeros(4)
            own_wind[(round_ - seat) // 4] = 1

            obs = np.concatenate(
                (
                    round_wind,
                    own_wind,
                    current_hand,
                    opened,
                    discarded_from_hand,
                    tsumogiri,
                    ),
                axis=0)

            obs_batch += [obs.astype(np.float32)]
            y_batch += [y[self.use_yaku_id].astype(np.float32)]

        obs_batch = torch.FloatTensor(np.array(obs_batch))
        y_batch = torch.FloatTensor(np.array(y_batch))

        return obs_batch, y_batch


@hydra.main(version_base=None, config_path="config", config_name="train_yaku_predictor")
def main(config: DictConfig) -> None:
    """Run training process."""

    if not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    train_dataset = YakuDataset(
        data_list=to_absolute_path(config.data.train_list),
        return_filename=True,
        allow_cache=config.data.allow_cache,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = YakuDataset(
        data_list=to_absolute_path(config.data.valid_list),
        return_filename=True,
        allow_cache=config.data.allow_cache,
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {"train": train_dataset, "valid": valid_dataset}

    # get data loader
    collater = Collater(
        config.data.ignore
    )
    train_sampler, valid_sampler = None, None
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            pin_memory=config.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=valid_sampler,
            pin_memory=config.data.pin_memory,
        ),
    }

    # define models and optimizers
    model = {
        "yaku_predictor": hydra.utils.instantiate(config.model).to(device),
    }

    # define training criteria
    criterion = {
        "bce": hydra.utils.instantiate(config.train.bce_loss).to(device),
        "total_acc": hydra.utils.instantiate(config.train.total_acc).to(device),
        "binary_acc": hydra.utils.instantiate(config.train.binary_acc).to(device),
    }

    # define optimizers and schedulers
    optimizer = {
        "yaku_predictor": hydra.utils.instantiate(
            config.train.yaku_predictor_optimizer, params=model["yaku_predictor"].parameters()
        ),
    }
    scheduler = {
        "yaku_predictor": hydra.utils.instantiate(
            config.train.yaku_predictor_scheduler, optimizer=optimizer["yaku_predictor"]
        ),
    }

    # define trainer
    trainer = Trainer(
        config=config,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # load trained parameters from checkpoint
    if config.train.resume:
        resume = os.path.join(
            config.out_dir, "checkpoints", f"checkpoint-{config.train.resume}steps.pkl"
        )
        if os.path.exists(resume):
            trainer.load_checkpoint(resume)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(
                config.out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"
            )
        )
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
