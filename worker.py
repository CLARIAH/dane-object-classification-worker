from dane.base_classes import base_worker
import json
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder
import torch.utils.data as data
import types
import glob
from dane.config import cfg
from dane import Result
from dane import errors


def get_idx(x: str) -> int:
    return int(os.path.basename(x).split(".")[0].split("_")[-1])


class KeyframeFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, loader=folder.default_loader):
        self.transform = transform
        self.data_dir = data_dir
        self.loader = loader
        self.samples = glob.glob(os.path.join(data_dir, "*.jpg"))
        self.samples.sort(key=get_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        sample = self.loader(path)
        fidx = int(os.path.basename(path).split(".")[0].split("_")[-1])

        if self.transform:
            sample = self.transform(sample)

        return sample, idx, fidx


class obj_class_worker(base_worker):
    # we specify a queue name because every worker of this type should
    # listen to the same queue
    __queue_name = "OBJECTCLASSIFICATION"
    __binding_key = "*.OBJECTCLASSIFICATION"
    __depends_on = ["SHOTDETECTION"]

    def __init__(self, config):
        super().__init__(
            queue=self.__queue_name,
            binding_key=self.__binding_key,
            # depends_on=self.__depends_on,
            config=config,
        )

        if config.CUDA.VISIBLE_DEVICES is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA.VISIBLE_DEVICES)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # TODO report if using CPU, or potentially restrict to GPU
        self.model = torchvision.models.resnet.resnet50(pretrained=True)
        self.model.sm = nn.Softmax(dim=1)

        self.store_embeddings = config.CLASSIFICATION.STORE_EMBEDDINGS

        # TODO Maybe this is too hacky
        def new_forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            feat = torch.flatten(x, 1)
            x = self.fc(feat)
            return feat, self.sm(x)

        self.model.forward = types.MethodType(new_forward, self.model)
        # it does work (tm)

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.batch_size = config.CLASSIFICATION.BATCH_SIZE
        self.load_workers = config.CLASSIFICATION.LOAD_WORKERS
        self.threshold = self.config.CLASSIFICATION.THRESHOLD

        self.INDEX = "dane-" + self.__queue_name.lower()

        if self.store_embeddings:
            if not self.handler.es.indices.exists(index=self.INDEX):
                self.handler.es.indices.create(
                    index=self.INDEX,
                    body={
                        "mappings": {
                            "properties": {
                                "doc_id": {"type": "keyword"},
                                "task_id": {"type": "keyword"},
                                "class_probs": {"type": "dense_vector", "dims": 1000},
                                "features": {"type": "dense_vector", "dims": 2048},
                                "frame_pos": {"type": "integer"},
                            }
                        }
                    },
                )

        if os.path.exists("imagenet1000_clsidx_to_labels.json"):
            with open("imagenet1000_clsidx_to_labels.json") as f:
                self.class2label = json.load(f)
        else:
            # json has str keys, so do the same
            self.class2label = {str(i): str(i) for i in range(1000)}

    def classify(self, key_dir, task, doc):
        dataset = KeyframeFolder(key_dir, transform=self.transform)

        if len(dataset) < 1:
            raise IOError("No keyframe found")

        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.load_workers,
        )

        out_data = []

        for img, idx, fidx in dataloader:
            idx = idx.numpy()
            fidx = fidx.numpy()

            with torch.no_grad():
                feat_out, softmax_out = self.model.forward(img.to(self.device))
            del img

            logits = softmax_out.cpu().numpy()
            feat_out = feat_out.cpu().numpy()

            for j, frame_idx in enumerate(fidx):
                if self.store_embeddings:
                    es_data = {
                        "doc_id": doc._id,
                        "task_id": task._id,
                        "features": feat_out[j].tolist(),
                        "class_probs": logits[j].tolist(),
                        "frame_pos": int(frame_idx),
                    }

                    self.handler.es.index(index=self.INDEX, body=json.dumps(es_data))

                # threshold score and gather readable labels
                top_n = {
                    self.class2label[str(i)]: float(score)
                    for i, score in enumerate(logits[j])
                    if score >= self.threshold
                }

                # sort decreasing based on score
                top_n = {
                    k: v
                    for k, v in sorted(
                        top_n.items(), key=lambda item: item[1], reverse=True
                    )
                }

                if len(top_n) > 0:
                    out_data.append(
                        {
                            "frame": int(frame_idx),
                            "classes": list(top_n.keys()),
                            "scores": list(top_n.values()),
                        }
                    )

            del feat_out, softmax_out

        return out_data

    def callback(self, task, doc):
        try:
            possibles = self.handler.searchResult(doc._id, "SHOTDETECTION")
            key_folder = possibles[0].payload["keyframe_folder"]
        except errors.ResultExistsError:
            # SHOTDETECTION is assigned, but no result (yet)
            return {"state": 412, "message": "No SHOTDETECTION result (yet)"}
        except errors.TaskAssignedError:
            # This shouldnt happen if SHOTDETECTION is dependency
            return {
                "state": 500,
                "message": "No SHOTDETECTION task assigned to document",
            }

        if not os.path.exists(key_folder):
            # TODO find better error no.
            return {
                "state": 500,
                "message": "keyframe folder does not exist, cannot handle request",
            }

        try:
            out_data = self.classify(key_folder, task, doc)
        except IOError as e:
            return {"state": 500, "message": "IOError:" + str(e)}
        except Exception as e:
            return {
                "state": 500,
                "message": "Unhandled error during classification: " + str(e),
            }
        else:
            r = Result(
                self.generator,
                payload={"classifications": out_data, "raw_features_index": self.INDEX},
                api=self.handler,
            )
            r.save(task._id)

            return {"state": 200, "message": "Success"}


if __name__ == "__main__":
    worker = obj_class_worker(cfg)

    print(" # Initialising worker. Ctrl+C to exit")
    try:
        worker.run()
    except (KeyboardInterrupt, SystemExit):
        worker.stop()
