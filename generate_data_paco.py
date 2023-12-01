from pathlib import Path
import json
import re
from collections import defaultdict

import torch


def main():
    paco_dir = Path("data_dir/paco")

    metadata_attrs = []
    split_dir = paco_dir / "compositional-split-natural"
    split_dir.mkdir(exist_ok=True)

    for split_str in ["train", "val", "test"]:
        with open(paco_dir / f"paco_lvis_v1_{split_str}.json") as f:
            data = json.load(f)

        id_to_attr = {
            attr["id"]: "_".join(filter(bool, re.split("[()]", attr["name"])))
            for attr in data["attributes"]
        }
        id_to_category = {
            category["id"]: "_".join(category["name"].split(":"))
            for category in data["categories"]
        }  # might need to remove parentheses
        id_to_img_filename = {
            img["id"]: img["file_name"].split("/")[-1] for img in data["images"]
        }

        data_attrs = []

        img_id_to_annotations = defaultdict(list)
        for annotation in data["annotations"]:
            img_id_to_annotations[annotation["image_id"]].append(annotation)

        for img_id in img_id_to_annotations:
            annotations = img_id_to_annotations[img_id]
            img_attrs = []
            img_attr_tuples = []
            top1 = max(annotations, key=lambda x: x["area"])

            obj = id_to_category[top1["category_id"]]
            for attr_id in top1["attribute_ids"]:
                attr = id_to_attr[attr_id]
                img_attrs.append(f"{attr} {obj}")
                img_attr_tuples.append({"attr": attr, "obj": obj})

            data_attrs.extend(img_attrs)
            for img_attr in img_attr_tuples:
                metadata_dict = {"image": id_to_img_filename[img_id], "set": split_str}
                metadata_dict.update(img_attr)
                metadata_attrs.append(metadata_dict)

        with open(split_dir / f"{split_str}_pairs.txt", "w") as f:
            f.write("\n".join(list(set(data_attrs))))

    torch.save(metadata_attrs, paco_dir / "metadata_compositional-split-natural.t7")


if __name__ == "__main__":
    main()
