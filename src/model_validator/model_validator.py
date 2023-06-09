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

import time
import torch
from torch import nn

class ModelValidator():
    """ Model validator class. """

    def __init__(self, args, model, val_loader, device):
        self.device = device

        if args.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Criterion {args.criterion} not implemented")

        self.model = model
        self.val_loader = val_loader

    def validate(self):
        """ Runs validation loop for model and data. """ 

        self.model.eval()

        time_start = time.time()

        running_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for _, data in enumerate(self.val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                output_classes = torch.argmax(outputs, 1)

                loss = self.criterion(outputs, labels)

                total += labels.size(0)
                correct += (output_classes == labels).float().sum()
                running_loss += loss.item()

        return {"time": time.time()-time_start,
                "loss": running_loss/len(self.val_loader),
                "accuracy": 100*correct/total}
    