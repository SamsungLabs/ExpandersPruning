"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
James Stewart (j2.stewart@samsung.com; james1995stewart@gmail.com)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import os

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from arg_parser import arg_parser
from model_factory import model_factory
from data_loader_factory import data_loader_factory
from model_trainer.model_trainer import ModelTrainer
from model_validator.model_validator import ModelValidator
from model_tester.model_tester import ModelTester

from utils import parameter_counter
from utils.config import TENSORBOARD_DIR
from utils.config import MODEL_SAVE_DIR

def main(args):
    """
    Constructs a model and runs training loop for specified number of epochs.
    Implements logging (console + tensorboard) and saving of model checkpoints.

    """

    if torch.cuda.is_available() and args.cuda:
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    print(args)

    print("Generating model")
    model = model_factory.generate_model(args)
    model.to(device)
    print(model)
    num_params = parameter_counter.count_parameters(args.model, model, print_summary=True)
    print(f"Generated model with {num_params} parameters")

    if args.tensorboard:
        depth = str(args.depth) if args.depth != -1 else ""
        degree = f"deg_{args.degree}" if args.degree != -1 else ""
        writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, args.model, \
                                depth, args.dataset, degree))
    else:
        writer = None

    print("Fetching data")
    loader = data_loader_factory.generate_data_loader(args)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              nesterov = args.nesterov,
                              weight_decay = args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    if args.lr_scheduler == "steplr":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, \
                                                 gamma=args.lr_decay_rate)
    else:
        raise NotImplementedError(f"Learning rate scheduler {args.lr_scheduler} not implemented")

    print("Training model")
    trainer = ModelTrainer(args,
                            model,
                            optimizer,
                            loader.get_train_loader(),
                            device)

    validator = ModelValidator(args,
                                model,
                                loader.get_val_loader(),
                                device)

    best_val_accuracy = 0

    for epoch in range(args.epochs):
        print(f"[Epoch {epoch+1}/{args.epochs}, lr={lr_scheduler.get_last_lr()[0]}]")

        train_result = trainer.train()
        val_result = validator.validate()

        if val_result["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_result["accuracy"]
            save_path = os.path.join(MODEL_SAVE_DIR, args.model, str(args.depth) \
                                     if args.depth != -1 else "", args.dataset)
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt" if args.degree == -1 \
                                                        else f"model_deg_{args.degree}.pt"))

        print(f"\t[Training time] {train_result['time']:.2f}s")
        print(f"\t[Training set] Average loss: {train_result['loss']:.2f}, Accuracy: {train_result['accuracy']:.2f}%")
        print(f"\t[Validation time] {val_result['time']:.2f}s")
        print(f"\t[Validation set] Average loss: {val_result['loss']:.2f}, Accuracy: {val_result['accuracy']:.2f}%")
        print(f"\t[Validation set] Best accuracy: {best_val_accuracy:.2f}%")

        if writer is not None:
            writer.add_scalar("Training loss", train_result["loss"], epoch)
            writer.add_scalar("Training accuracy", train_result["accuracy"], epoch)
            writer.add_scalar("Validation loss", val_result["loss"], epoch)
            writer.add_scalar("Validation accuracy", val_result["accuracy"], epoch)
            writer.add_scalar("Best validation accuracy", best_val_accuracy, epoch)

        if (args.lr_scheduler == "steplr" and
            args.min_learning_rate < args.lr_decay_rate*lr_scheduler.get_last_lr()[0] and
            args.max_learning_rate > args.lr_decay_rate*lr_scheduler.get_last_lr()[0]):
            lr_scheduler.step()

    print("Completed training")

    print("Testing model")

    test_result = ModelTester(args,
                               model,
                               loader.get_test_loader(),
                               device).test()

    print(f"\t[Test time] {test_result['time']:.2f}s")
    print(f"\t[Test set] Average loss: {test_result['loss']:.2f}, Accuracy: {test_result['accuracy']:.2f}%")

    if writer is not None:
        writer.add_scalar("Test loss", test_result["loss"])
        writer.add_scalar("Test accuracy", test_result["accuracy"])

if __name__ == "__main__":
    args = arg_parser.create_parser().parse_args()
    main(args)
