import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional

import deepspeed
import torch
import torch.distributed
import torch.optim
import torch.utils.data
import tqdm
import transformers
from utils.ds_utils import get_train_ds_config
from deepspeed import DeepSpeedConfig, get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from dotenv import load_dotenv
from dataset.Coqdataset import CoqDataCollator, CoqDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from pretrain_args import PretrainArguments
from model.utils import print_model_parameters, save_model, init_from_pretrained
from utils.file import dump_json_file, load_json_file
from utils.train import get_all_reduce_mean, print_rank_0, set_random_seed, to_device, is_rank_0, parse_remaining_args_to_dict
from utils import *
from utils.train import clean_dict, clear_memory
from utils.optimizers import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
set_random_seed(42)
set_random_seed(42)


class Trainer:
    def __init__(self):
        self.training_args = None

        self.ds_config = None
        self.device = None

        self.model = None
        self.tokenizer = None
        self.config = None
        self.ds_engine = None

        self.data_collator = None
        self.train_dataloader = None
        self.total_steps = None
        self.eval_dataloader = None
        self.lr_scheduler = None
        self.optimizer = None
        self.monitor = None

        self.init()


    def init(self):
        # 初始化参数
        self.parse_arguments()
        # 初始化分布式
        self.init_distributed()
        # 初始化deepspeed配置
        self.init_ds_config()

        torch.distributed.barrier()
        print_rank_0("torch distributed barrier", rank=self.training_args.global_rank, wrap=True)

        # 初始化模型和tokenizer
        self.init_model_and_tokenizer()

        # 初始化数据集,dataloader
        self.build_dataloader()

        self.total_steps = len(self.train_dataloader) * self.training_args.num_train_epochs
        
        self.init_optimizer()
        self.build_scheduler()

        self.init_deepspeed()

        self.init_monitor()

    def parse_arguments(self):
        parser = transformers.HfArgumentParser(PretrainArguments)
        (
            self.training_args,
            remaining_args,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        remaining_args_dict = parse_remaining_args_to_dict(remaining_args)

        print_rank_0(self.training_args)
        print_rank_0(remaining_args_dict)
    
    def init_monitor(self):
        self.monitor = MonitorMaster(DeepSpeedConfig(self.training_args.deepspeed).monitor_config)

    def init_distributed(self):
        """Initialize distributed training setup."""
        accelerator = get_accelerator()
        if self.training_args.local_rank == -1:
            self.device = torch.device(accelerator.device_name())
        else:
            accelerator.set_device(self.training_args.local_rank)
            self.device = torch.device(accelerator.device_name(), self.training_args.local_rank)
            deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=10))
        self.training_args.global_rank = torch.distributed.get_rank()

    def build_ds_config(self):
        
        ds_config = get_train_ds_config(offload=self.training_args.offload_params,
                                        adam_offload=self.training_args.offload_adam,
                                        stage=self.training_args.zero_stage)
        
        ds_config["train_micro_batch_size_per_gpu"] = self.training_args.per_device_train_batch_size
        ds_config["train_batch_size"] = (
                self.training_args.per_device_train_batch_size
                * torch.distributed.get_world_size()
                * self.training_args.gradient_accumulation_steps
        )
        
        ds_config['wandb'] = {
            "enabled": self.training_args.wandb_enabled,
            "project": self.training_args.wandb_project_name
        }
        
        return ds_config
    
    def init_ds_config(self):
        if self.training_args.deepspeed:
            ds_config = load_json_file(self.training_args.deepspeed)
        
            ds_config["train_micro_batch_size_per_gpu"] = self.training_args.per_device_train_batch_size
            ds_config["train_batch_size"] = (
                self.training_args.per_device_train_batch_size
                * torch.distributed.get_world_size()
                * self.training_args.gradient_accumulation_steps
            )
        else:
            ds_config = self.build_ds_config()

        self.ds_config = ds_config
    
    def init_optimizer(self):
        no_decay = ["bias", "gamma", "beta", "layer_norm.weight", "layer_norm_1.weight", "layer_norm_2.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.training_args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0}
        ]

        if self.training_args.deepspeed:
            zero_stage = self.ds_config["zero_optimization"]["stage"]
            if zero_stage in (1, 2):
                optimizer_class = deepspeed.ops.adam.FusedAdam
            elif zero_stage == 3:
                optimizer_class = deepspeed.ops.adam.DeepSpeedCPUAdam
        else:
            if self.ds_config['zero_optimization']['offload_optimizer']['device']:
                offload_device = self.ds_config['zero_optimization']['offload_optimizer']['device']
                if offload_device in ('none', None):
                    optimizer_class = torch.optim.AdamW
                else:
                    optimizer_class = deepspeed.ops.adam.DeepSpeedCPUAdam
            else:
                raise ValueError(f"No optimizer class specified")
        self.optimizer = optimizer_class(optimizer_grouped_parameters, 
                                    lr=self.training_args.learning_rate, 
                                    betas=(0.9, 0.999),
                                    # scale_parameter=False, 
                                    # relative_step=False
                                    )
    
    def build_scheduler(self):
        if self.training_args.scheduler not in str2scheduler:
            raise ValueError(f"Unknown scheduler: {self.training_args.scheduler}")
        
        if self.training_args.scheduler in ['constant']:
            self.scheduler = str2scheduler[self.training_args.scheduler](self.optimizer)
        if self.training_args.scheduler in ['constant_with_warmup']:
            self.scheduler = str2scheduler[self.training_args.scheduler](self.optimizer)
        elif self.training_args.scheduler in ["tri_stage"]:
            self.scheduler = str2scheduler[self.training_args.scheduler](self.optimizer, 
                                                                    self.total_steps * self.training_args.warmup, 
                                                                    self.total_steps * self.training_args.decay, 
                                                                    self.total_steps)
        else:
            self.scheduler = str2scheduler[self.training_args.scheduler](self.optimizer, 
                                                                    self.total_steps * self.training_args.warmup, 
                                                                    self.total_steps)
        

    def init_model_and_tokenizer(self):
        print_rank_0("start load model", rank=self.training_args.global_rank, wrap=True)

        model, tokenizer, config = init_from_pretrained(
            pretrained_dir=self.training_args.pretrained_dir,
            model_max_length=self.training_args.model_max_length
        )

        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        print_model_parameters(model)

        if self.training_args.gradient_checkpointing:
            print_rank_0("gradient checkpointing", rank=self.training_args.global_rank, wrap=True)
            model.gradient_checkpointing_enable()
        print_rank_0("end load model", rank=self.training_args.global_rank, wrap=True)
    
    def build_dataloader(self):
        train_dataset = CoqDataset(self.training_args.data_path)
        
        eval_dataset = CoqDataset(self.training_args.eval_path)
        print('===========')
        print_rank_0(len(train_dataset), rank=self.training_args.global_rank)
        print('===========')
        print_rank_0(len(eval_dataset), rank=self.training_args.global_rank)
        print_rank_0("end load dataset", rank=self.training_args.global_rank, wrap=True)

        if self.training_args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = RandomSampler(eval_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=False)
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        
        self.data_collator = CoqDataCollator(
            tokenizer=self.tokenizer,
            max_length = self.training_args.model_max_length
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=4,
            prefetch_factor=4,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            batch_size=self.training_args.per_device_train_batch_size,
            # pin_memory=True
        )

        self.eval_dataloader = DataLoader(
            eval_dataset,
            num_workers=4,
            prefetch_factor=4,
            collate_fn=self.data_collator,
            sampler=eval_sampler,
            batch_size=self.training_args.per_device_eval_batch_size,
            # pin_memory=True
        )

    def init_deepspeed(self):
        print_rank_0("start deepspeed init", rank=self.training_args.global_rank, wrap=True)
        
        ds_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            collate_fn=self.data_collator,
            args=self.training_args,
            config=self.ds_config,
            dist_init_required=True,
        )
        self.ds_engine = ds_engine

        print_rank_0("end deepspeed init", rank=self.training_args.global_rank, wrap=True)

    def train(self):
        print_rank_0("start training", rank=self.training_args.global_rank, wrap=True)
        start_epoch = 0
        start_step = 0

        for epoch in range(start_epoch, int(self.training_args.num_train_epochs)):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{self.training_args.num_train_epochs}, "
                f"Total Micro Batches {len(self.train_dataloader)}",
                rank=self.training_args.global_rank,
            )
            self.train_epoch(epoch, start_step)
            start_step = 0
        
        self.save()
        if self.training_args.output_dir is not None:
            print_rank_0(
                "Saving the final model ...",
                rank=self.training_args.global_rank,
            )
            output_dir = self.training_args.output_dir
            self.ds_engine.save_checkpoint(output_dir)
            self.ds_engine.save_fp16_model(f"{output_dir}/fp16", "pytorch_model.bin") 
    
    @torch.no_grad()
    def evaluate(self, mode='train') -> Tuple[float, float]:
        assert mode in ('train', 'test')
        start_time = time.time()

        print_rank_0(f'Start evaluation on {mode} data.')
        self.ds_engine.eval()

        losses = 0
        step = 0

        for batch in tqdm.tqdm(self.eval_dataloader, desc="evaluating...", total=len(self.eval_dataloader), disable=not is_rank_0()):
            batch = to_device(batch, self.device)
            outputs = self.ds_engine(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
            step += 1
            
            clean_dict(batch)
            del outputs
            del loss

        losses = losses / (step + 1)

        losses = get_all_reduce_mean(losses.clone().detach())

        perplexity = torch.exp(losses).item()

        print_rank_0(
            f'Evaluation completed in {(time.time() - start_time):.2f} seconds, loss = {losses.item()}, perplexity= {perplexity}')

        return losses.item(), perplexity

    def train_step(self, batch):
        batch = to_device(batch, self.device)
        outputs = self.ds_engine(**batch, use_cache=False)
        # print(outputs)
        loss = outputs.loss
        self.ds_engine.backward(loss)
        self.ds_engine.step()
        
        clean_dict(batch)
        del outputs
        return loss

    def train_epoch(self, epoch, start_step=0):
        self.ds_engine.train()

        for step, batch in tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="Train",
        ):

            try:
                start_time = time.time()
                loss = self.train_step(batch)
                cost_time = time.time() - start_time
            except torch.cuda.OutOfMemoryError as e:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: {v.shape}")
                print(f"由于内存溢出错误，跳过此批次: {e}")
                torch.cuda.empty_cache()
                continue
            
            self.log_step(epoch,step,loss)
            del loss
            
            self.ds_engine.monitor.write_events([("ups", 1. / cost_time, self.ds_engine.global_samples)])

            if (step + 1) % self.training_args.save_steps == 0 and step != start_step:
                self.evaluate_and_save(epoch, step)
                self.ds_engine.train()
                clear_memory()

        self.evaluate_and_save(epoch, step)

    def log_step(self, epoch: int, step: int, loss):
        logging.info(
            f"\nEpoch: {epoch}, Step: {step}, Global Rank: {self.training_args.global_rank}, Loss: {loss} \t"
            + "\t"
            + str(self.ds_engine.optimizer.param_groups[0]["lr"])
        )

        self.ds_engine.monitor.write_events(
            [('epoch', epoch, self.ds_engine.global_samples),
             ("step", step, self.ds_engine.global_samples),
             ("loss", loss, self.ds_engine.global_samples),
             ("lr", self.ds_engine.optimizer.param_groups[0]["lr"], self.ds_engine.global_samples)])
        
    def evaluate_and_save(self, epoch, step):
        loss,perplexity  = self.evaluate()
        self.ds_engine.monitor.write_events(
            [("eval_loss", loss, self.ds_engine.global_samples),
             ("eval_perplexity", perplexity, self.ds_engine.global_samples)])
        print_rank_0(
            f"Epoch: {epoch}, Step: {step}, Loss: {loss}, Perplexity: {perplexity}",
            rank=self.training_args.global_rank,
        )
        self.save(epoch,step=step)

    def save(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        output_dir = self.training_args.output_dir

        if epoch is not None:
            model_save_dir = f"{output_dir}/epoch_{epoch}"
            if step:
                model_save_dir += f"_step_{step}"
        else:
            model_save_dir = f"{output_dir}/final"

        print_rank_0(
            f"Saving the model, epoch {epoch}, step {step}...",
            rank=self.training_args.global_rank,
        )

        client_state = {
            "epoch": epoch,
            "step": step,
            "global_samples": self.ds_engine.global_samples,
            "scheduler_state": self.scheduler.state_dict()
        }
        self.ds_engine.save_checkpoint(output_dir, client_state=client_state)

        try:
            save_model(self.ds_engine, self.config, self.tokenizer, model_save_dir)
        except:
            print_rank_0(
                f"Failed to save the model, epoch {epoch}, step {step}",
                rank=self.training_args.global_rank,
            )
            self._save_checkpoint(step)
    
    def _save_checkpoint(self, total_steps: int):
        steps_model_path = os.path.join(self.training_args.output_dir, '{}_steps'.format(total_steps))

        if self.ds_config["zero_optimization"]["stage"] == 3:
            state_dict = self.ds_engine._zero3_consolidated_16bit_state_dict()
        else:
            state_dict = self.ds_engine.state_dict()
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        if is_rank_0():
            self.ds_engine.save_pretrained(steps_model_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(steps_model_path)

            logging.info(f'Saved model of {total_steps} steps to {steps_model_path}')

        del state_dict
        torch.distributed.barrier()


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
