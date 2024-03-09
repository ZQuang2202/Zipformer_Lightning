from typing import Any, Optional, Sequence, Union, List, Tuple
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torchaudio
from torchmetrics import MeanMetric
from torchmetrics.functional import word_error_rate
from torchmetrics.text import WordErrorRate
import lightning as ltn
try:
    import k2
except:
    pass
from ..layers.scaling import ScaledLinear
from ..scorer.greedy import greedy_search_batch
from ..optim.ede_optim import Eve
from ..optim.noam import NoamLR
from ..optim.sophia import SophiaG
from ..optim.optim import Eden, ScaledAdam
try:
    import fast_rnnt
except:
    pass
from warprnnt_pytorch import RNNTLoss
from collections import defaultdict
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name

class RNNT(ltn.LightningModule):
    def __init__(self,
                 encoder_embed: torch.nn.Module,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 jointer: torch.nn.Module,
                 text_extractor=None,
                 k=10,
                 lm_scale: float = 0.25,
                 am_scale: float = 0.0,
                 prune_range: float = 5,
                 simple_loss_scale: float=0.5,
                 delay_penalty: float=0.0,
                 model_warm_step = 2000,
                 freeze_encoder: bool = False,
                 freeze_decoder: bool = False,
                 pretrained_path: str = '',
                 base_lr=0.0001,
                 lr_epochs = 6,
                 lr_batches = 7500,
                 optim_warmup_steps=500,
                 decode_chunk_size: int = 16,
                 left_context: int = 64,
                 max_duration = 300,
                 world_size=1,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.decoder = decoder
        self.jointer = jointer
        self.log_hyperthesis_num = k
        self.blank_id = self.decoder.blank_id
        self.lm_scale = lm_scale
        self.am_scale = am_scale
        self.base_lr = base_lr
        self.optim_warmup_steps = optim_warmup_steps
        self.lr_epochs = lr_epochs
        self.lr_batches = lr_batches
        self.delay_penalty = delay_penalty
        self.prune_range = prune_range
        self.simple_loss_scale = simple_loss_scale
        self.model_warm_step = model_warm_step
        self.decode_chunk_size = decode_chunk_size
        self.left_context = left_context
        self.ref_duration = 600.0
        self.max_duration = max_duration
        self.avg_train_loss = MeanMetric()
        self.avg_valid_loss = MeanMetric()
        self.avg_valid_wer = MeanMetric()
        self.avg_test_wer = MeanMetric()
        self.avg_test_loss = MeanMetric()
        self.text_extractor = text_extractor
        # self.automatic_optimization = False
        self.world_size = world_size
        if pretrained_path and pretrained_path.strip():
            checkpoint = torch.load(
                pretrained_path, map_location=next(self.parameters()).device)
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        if freeze_encoder:
            self.freeze_encoder()
        if freeze_decoder:
            self.freeze_decoder()
            
        self.refs = []
        self.hyps = []
        self.ids = []
        self.rnnt_loss = RNNTLoss(blank=self.blank_id, reduction="mean")
        self.wer_func = WordErrorRate()
    def freeze_encoder(self):
        for layer in self.encoder.parameters():
            layer.requires_grad_(False)

    def freeze_decoder(self):
        for layer in self.decoder.parameters():
            layer.requires_grad_(False)
    def make_pad_mask(self, lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
        """
        Args:
        lengths:
            A 1-D tensor containing sentence lengths.
        max_len:
            The length of masks.
        Returns:
        Return a 2-D bool tensor, where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.

        >>> lengths = torch.tensor([1, 3, 2, 5])
        >>> make_pad_mask(lengths)
        tensor([[False,  True,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False, False, False]])
        """
        assert lengths.ndim == 1, lengths.ndim
        max_len = max(max_len, lengths.max())
        n = lengths.size(0)
        seq_range = torch.arange(0, max_len, device=lengths.device)
        expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

        return expaned_lengths >= lengths.unsqueeze(-1)
    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = self.make_pad_mask(lengths=x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens
    
    def forward(self, x, x_lens, y, return_enc=False, return_dec=False, compute_logits=True):
        encoder_out, x_lens = self.encoder.forward_encoder(x, x_lens)
        pad_values = torch.full(
            size=(y.size(0), 1),
            fill_value=self.blank_id,
            device=y.device,
            dtype=y.dtype,
        )
        sos_y_padded = torch.cat((pad_values, y), dim=1)
        sos_y_padded = sos_y_padded.to(torch.int64)
        decoder_out = self.decoder(sos_y_padded)
        if compute_logits:
            logits = self.jointer(encoder_out, decoder_out, project_input=True)
        else:
            logits = None
        outputs = [logits,]
        if return_enc:
            outputs.append(encoder_out)
        if return_dec:
            outputs.append(decoder_out)
        outputs.append(x_lens)
        return tuple(outputs)

    def compute_fast_rnnt_loss(self, x, x_lens, y, y_lens, warmup=1.0, return_ans=False):
        y = y.to(torch.int64)
        logits, encoder_out, x_lens = self.forward(
            x, x_lens, y, warmup=warmup, return_enc=True)
        termination_symbol = 0

        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64).to(x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        loss = fast_rnnt.rnnt_loss(
            logits=logits,
            symbols=y,
            termination_symbol=termination_symbol,
            boundary=boundary,
            reduction=self.reduction,
        )
        if return_ans:
            ans = greedy_search_batch(
                self.decoder, self.jointer, encoder_out, x_lens)
            return loss, ans
        return loss

    def compute_warp_transducer_loss(self, x, x_lens, y, y_lens,return_ans=False):
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 2, y.shape

        assert x.size(0) == x_lens.size(0) == y.shape[0], (x.shape, x_lens.shape, y.shape[0])
        
        # compute encoder outputs
        encoder_out, x_lens = self.forward_encoder(x, x_lens)
        
        # compute transducer loss
        pad_values = torch.full(
                size=(y.size(0), 1),
                fill_value=self.blank_id,
                device=y.device,
                dtype=y.dtype,
            )
        sos_y_padded = torch.cat((pad_values, y), dim=1)
        sos_y_padded = sos_y_padded.to(torch.int64)
        decoder_out = self.decoder(sos_y_padded)
        logits = self.jointer(encoder_out, decoder_out, project_input=True)

        # print(logits.shape, encoder_out.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.rnnt_loss(logits.float(), y, x_lens, y_lens)
        if return_ans:
            ans = greedy_search_batch(
                self.decoder, self.jointer, encoder_out, x_lens)
            return loss, ans
        return loss

    def compute_k2_loss(
        self,
        x, x_lens, y, y_lens, warmup=1.0, return_ans=False
    ):

        _, encoder_outs, decoder_outs, x_lens = self.forward(
            x, x_lens, y, warmup=warmup, return_dec=True, return_enc=True, compute_logits=False
        )
        x_lens = x_lens.contiguous().to(torch.int64)
        pad_values = torch.full(
            size=(y.size(0), 1),
            fill_value=self.blank_id,
            device=y.device,
            dtype=y.dtype,
        )
        sos_y_padded = torch.cat((pad_values, y), dim=1)
        sos_y_padded = sos_y_padded.to(torch.int64)
        boundary = torch.zeros(
            (x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens
        lm = self.simple_lm_proj(decoder_outs)
        am = self.simple_am_proj(encoder_outs)

        y = y.to(torch.int64)
        # print(y)
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y,
                termination_symbol=self.blank_id,
                lm_only_scale=self.lm_scale,
                am_only_scale=self.am_scale,
                boundary=boundary,
                reduction='sum',
                delay_penalty=self.delay_penalty,
                return_grad=True,
            )
        # print(simple_loss)
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=int(self.prune_range),
        )
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.jointer.encoder_proj(encoder_outs),
            lm=self.jointer.decoder_proj(decoder_outs),
            ranges=ranges,
        )
        logits = self.jointer(am_pruned, lm_pruned, project_input=False)
        logits = logits.contiguous()
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y,
                ranges=ranges,
                termination_symbol=self.blank_id,
                boundary=boundary,
                delay_penalty=self.delay_penalty,
                reduction='sum',
            )
        pruned_loss_scale = (
            0.0 if warmup < 1.0 else (
                0.1 if warmup > 1.0 and warmup < 2.0 else 1.0)
        )
        loss = (
            self.simple_loss_scale * simple_loss
            + pruned_loss_scale * pruned_loss)
        # print(simple_loss, pruned_loss)
        if return_ans:
            ans = greedy_search_batch(
                self.decoder, self.jointer, encoder_outs, x_lens)
            return loss, ans
        return loss

    def on_train_epoch_end(self) -> None:
        self.avg_train_loss.reset()
        self.avg_valid_loss.reset()
        self.avg_valid_wer.reset()

    def on_validation_epoch_end(self) -> None:
        for ref, hyp in zip(self.refs[:self.log_hyperthesis_num], self.hyps[:self.log_hyperthesis_num]):
            self.print('ref:', ref)
            self.print('hyp:', hyp)
            self.print('-' * 10)
        self.refs = []
        self.hyps = []
        self.ids = []

    def on_test_epoch_end(self) -> None:
        self.avg_test_loss.reset()
        self.avg_test_wer.reset()
        for ref, hyp in zip(self.refs[:self.log_hyperthesis_num], self.hyps[:self.log_hyperthesis_num]):
            self.print('ref:', ref)
            self.print('hyp:', hyp)
            self.print('-' * 10)
        
    def set_batch_count(self,batch_idx):
        if batch_idx % 10 == 0:
            batch_count = self.trainer.global_step * (self.max_duration*self.world_size) / self.ref_duration
            set_batch_count(self.encoder_embed, batch_count)
            set_batch_count(self.decoder, batch_count)
            set_batch_count(self.jointer, batch_count)
            set_batch_count(self.encoder, batch_count)
            
    def training_step(self, batch, batch_idx, logging=True):

        ids, x, x_lens, y, y_lens = batch
        batch_size = x.size(0)
        batch_sizes = self.all_gather(batch_size)
        self.set_batch_count(batch_idx)
        
        with torch.set_grad_enabled(True):
            loss = self.compute_warp_transducer_loss(x, x_lens, y, y_lens)
            loss *= batch_size * batch_sizes.size(0) / batch_sizes.sum()
        self.avg_train_loss.update(loss)

        if logging:
            self.log('step_train_loss', loss, prog_bar=True, sync_dist=True)
            self.log('avg_train_loss', self.avg_train_loss.compute(),prog_bar=True, sync_dist=True)
            scheduler = self.lr_schedulers()
            cur_lr = max(scheduler.get_last_lr())
            self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
            self.log('Step', self.trainer.global_step,prog_bar=True, sync_dist=True)
            if len(self.trainer.optimizers[0].param_groups) > 1:
                optim_names = ['encoder', 'decoder', 'jointer']
                assert len(self.trainer.optimizers[0].param_groups) == len(
                    optim_names)
                for optim_name, group in zip(optim_names, self.trainer.optimizers[0].param_groups):
                    cur_lr = group['lr']
                    self.log(f"{optim_name}_lr", cur_lr,
                             prog_bar=True, sync_dist=True)
            # else:
            #     cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            #     self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        ids, x, x_lens, y, y_lens = batch
        batch_size = x.size(0)
        batch_sizes = self.all_gather(batch_size)
        with torch.set_grad_enabled(False):
            loss, ans = self.compute_warp_transducer_loss(
                x, x_lens, y, y_lens, return_ans=True)
        # Normalize by world size / batch size
        loss *= batch_size * batch_sizes.size(0) / batch_sizes.sum()
        self.avg_valid_loss.update(loss)
        # self.log('step_valid_loss', loss, prog_bar=True, sync_dist=True)
        self.log('avg_valid_loss', self.avg_valid_loss.compute(),
                 prog_bar=True, sync_dist=True)

        refs = self.text_extractor.decode(y.cpu().numpy().tolist())
        hyps = self.text_extractor.decode(ans)
        self.refs.extend(refs)
        self.hyps.extend(hyps)
        wer = self.wer_func(hyps, refs)
        self.avg_valid_wer.update(wer)
        avg_wer = self.avg_valid_wer.compute()
        avg_wers = self.all_gather(avg_wer)
        self.log('avg_valid_wer', sum(avg_wers) /
                 len(avg_wers), prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        ids, x, x_lens, y, y_lens = batch
        batch_size = x.size(0)
        batch_sizes = self.all_gather(batch_size)
        with torch.set_grad_enabled(False):
            loss, ans = self.compute_warp_transducer_loss(
                x, x_lens, y, y_lens, return_ans=True)
        # Normalize by world size / batch size
        loss *= batch_size * batch_sizes.size(0) / batch_sizes.sum()
        self.avg_test_loss.update(loss)
        # self.log('step_valid_loss', loss, prog_bar=True, sync_dist=True)
        self.log('avg_test_loss', self.avg_test_loss.compute(),
                 prog_bar=True, sync_dist=True)

        refs = self.text_extractor.decode(y.cpu().numpy().tolist())
        hyps = self.text_extractor.decode(ans)
        self.refs.extend(refs)
        self.hyps.extend(hyps)
        wer = self.wer_func(hyps, refs)
        self.avg_test_wer.update(wer)
        avg_wer = self.avg_test_wer.compute()
        avg_wers = self.all_gather(avg_wer)
        self.log('avg_test_wer', sum(avg_wers) /
                 len(avg_wers), prog_bar=True, sync_dist=True)
        
    
    def get_parameter_groups_with_lrs(
        self,
        model: nn.Module,
        include_names: bool = False,
        freeze_modules: List[str] = [],
    ) -> List[dict]:
        """
        This is for use with the ScaledAdam optimizers (more recent versions that accept lists of
        named-parameters; we can, if needed, create a version without the names).

        It provides a way to specify learning-rate scales inside the module, so that if
        any nn.Module in the hierarchy has a floating-point parameter 'lr_scale', it will
        scale the LR of any parameters inside that module or its submodules.  Note: you
        can set module parameters outside the __init__ function, e.g.:
        >>> a = nn.Linear(10, 10)
        >>> a.lr_scale = 0.5

        Returns: a list of dicts, of the following form:
        if include_names == False:
            [  { 'params': [ tensor1, tensor2, ... ], 'lr': 0.01 },
            { 'params': [ tensor3, tensor4, ... ], 'lr': 0.005 },
            ...   ]
        if include_names == true:
            [  { 'named_params': [ (name1, tensor1), (name2, tensor2), ... ], 'lr': 0.01 },
            { 'named_params': [ (name3, tensor3), (name4, tensor4), ... ], 'lr': 0.005 },
            ...   ]

        """
        named_modules = list(model.named_modules())
        # flat_lr_scale just contains the lr_scale explicitly specified
        # for each prefix of the name, e.g. 'encoder.layers.3', these need
        # to be multiplied for all prefix of the name of any given parameter.
        flat_lr_scale = defaultdict(lambda: 1.0)
        names = []
        for name, m in model.named_modules():
            names.append(name)
            if hasattr(m, "lr_scale"):
                flat_lr_scale[name] = m.lr_scale

        # lr_to_parames is a dict from learning rate (floating point) to: if
        # include_names == true, a list of (name, parameter) for that learning rate;
        # otherwise a list of parameters for that learning rate.
        lr_to_params = defaultdict(list)

        for name, parameter in model.named_parameters():
            split_name = name.split(".")
            # caution: as a special case, if the name is '', split_name will be [ '' ].
            prefix = split_name[0]
            if prefix == "module":  # DDP
                module_name = split_name[1]
                if module_name in freeze_modules:
                    self.log(f"Remove {name} from parameters")
                    continue
            else:
                if prefix in freeze_modules:
                    self.log(f"Remove {name} from parameters")
                    continue
            cur_lr = self.base_lr* flat_lr_scale[prefix]
            if prefix != "":
                cur_lr *= flat_lr_scale[""]
            for part in split_name[1:]:
                prefix = ".".join([prefix, part])
                cur_lr *= flat_lr_scale[prefix]
            lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

        if include_names:
            return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
        else:
            return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]
        
    def configure_optimizers(self) -> Any:
        # if isinstance(self.base_lr, list):
        #     modules = [self.encoder, self.decoder, self.jointer]
        #     assert len(self.base_lr) == len(modules)
        #     # optimizer = Eve(
        #         # params=[{'params': module.parameters(), 'lr': lr} for module, lr in zip(modules, self.lr)])
        #     optimizer = SophiaG(
        #         params=[{'params': module.parameters(), 'lr': lr} for module, lr in zip(modules, self.base_lr)])
            
        # else:
        #     # optimizer = Eve(self.trainer.model.parameters(), lr=self.base_lr)
        #     optimizer = SophiaG(self.trainer.model.parameters(), lr=self.base_lr)
        # if self.optim_warmup_steps > 0:
        #     # lr_scheduler = Eden(
        #     #     optimizer, self.optim_warmup_steps, self.optim_warmup_epochs)
        #     lr_scheduler = NoamLR(optimizer, self.optim_warmup_steps)
        #     return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        # return optimizer
            
        optimizer = ScaledAdam(
            self.get_parameter_groups_with_lrs(self.trainer.model, include_names=True),
            self.trainer.model.parameters(),
            lr=self.base_lr
        )

        scheduler = Eden(optimizer, 
                         self.lr_batches, 
                         self.lr_epochs, 
                         self.optim_warmup_steps, 
                    )

        return [optimizer],  [{"scheduler": scheduler, "interval": "step"}]
    
    def on_train_epoch_start(self) -> None:
        sch = self.lr_schedulers()
        sch.step_epoch(self.current_epoch)  

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_batch(self.trainer.global_step + 1)