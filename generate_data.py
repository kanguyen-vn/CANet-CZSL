from pathlib import Path
import json

import torch


def main():
    vg_dir = Path("data_dir/vg")
    with open(vg_dir / "attributes.json") as f:
        js = json.load(f)

    metadata_attrs = []
    split_dir = vg_dir / "compositional-split-natural"
    split_dir.mkdir(exist_ok=True)

    for split_str in ["train", "val", "test"]:
        with open(vg_dir / f"vg_naturaldisasters/{split_str}.json") as f:
            data = json.load(f)
            data_ids = [x["id"] for x in data["images"]]
        
        data_attrs = []

        # Swap test and val splits
        alt_set = split_str if split_str == "train" else ("val" if split_str == "test" else "test")
        
        for img in js:
            img_id = img["image_id"]
            if img_id not in data_ids:
                continue
            if "attributes" not in img or len(img["attributes"]) == 0:
                continue
            img_attrs = []
            img_attr_tuples = []
            top1 = max(img["attributes"], key=lambda x: int("attributes" in x) * x["w"] * x["h"])
            if "attributes" not in top1:
                continue
            for obj in top1["names"]:
                obj = obj.strip()
                for attr in top1["attributes"]:
                    attr = attr.strip()
                    split_attr = attr.split()
                    if len(split_attr) > 1:
                        attr = ".".join(split_attr)
                    split_obj = obj.split()
                    if len(split_obj) > 1:
                        obj = ".".join(split_obj)
                    img_attrs.append(f"{attr} {obj}")
                    img_attr_tuples.append({"attr": attr, "obj": obj})
            data_attrs.extend(img_attrs)
            for img_attr in img_attr_tuples:
                metadata_dict = {"image": f"{img_id}.jpg", "set": alt_set}
                metadata_dict.update(img_attr)
                metadata_attrs.append(metadata_dict)

        with open(split_dir / f"{alt_set}_pairs.txt", "w") as f:
            f.write("\n".join(list(set(data_attrs))))

    torch.save(metadata_attrs, vg_dir / "metadata_compositional-split-natural.t7")


if __name__ == "__main__":
    main()