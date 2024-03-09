import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import shutil
import logging
import argparse
from hyperpyyaml import load_hyperpyyaml


parser = argparse.ArgumentParser(
        description="Train a zipformer experiment with pytorch lightning",
    )

parser.add_argument(
    "--config-file",
    type=str,
    help="A yaml-formatted file using the extended YAML syntax.",
)

parser.add_argument( 
    "--flash",
    action='store_true',
    help="Enables or disables flash scaled dot product attention.",
)

parser.add_argument(
    "--compile",
    action='store_true',
    help="Enables or disables compile torch 2.0.",
)

args = parser.parse_args()

if args.flash:
    torch.backends.cuda.enable_flash_sdp(True)

with open(args.config_file) as fin:
    modules = load_hyperpyyaml(fin)
    trainer = modules['trainer']
    model = modules['model']
    if args.compile:
        model = torch.compile(model)
    os.makedirs(trainer.log_dir, exist_ok=True)
    shutil.copy(args.config, trainer.log_dir)
    datamodule = modules['datamodule']
    ## Training
    trainer.fit(
        model,
        datamodule=datamodule,
    )

    ## Test 
    test_dataloader = modules['datamodule'].test_dataloader()
    trainer.test(
        dataloaders=test_dataloader,
        verbose=True,
    )
