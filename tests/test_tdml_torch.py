import os
import pytest
import numpy as np
from PIL import Image
from types import SimpleNamespace
import pytdml
import pytdml.io
import pytdml.ml
from torchvision import transforms


def test_torch_eo_image_object_td():
    # Load the training dataset
    training_dataset = pytdml.io.read_from_json(
        "tests/data/object-detection/COWC_partial.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Set parameters
    train_size = [128, 128]

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    train_dataset = pytdml.ml.TorchEOImageObjectTD(  # create Torch train dataset
        training_dataset.data, class_map, transform=pytdml.ml.BaseTransform(train_size)
    )

    img, label, img_height, img_width = train_dataset[0]


def test_torch_eo_image_scene_td():
    training_dataset = pytdml.io.read_from_json(
        "tests/data/scene-classification/WHU-RS19.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    trans_size = [64, 64]
    train_dataset = pytdml.ml.TorchEOImageSceneTD(  # create Torch train dataset
        training_dataset.data,
        class_map,
        transform=transforms.Compose(  # transform for the training set
            [
                transforms.RandomResizedCrop(
                    size=156, scale=(0.8, 1.0)
                ),  # random resize
                transforms.RandomRotation(degrees=15),  # random rotate
                transforms.RandomHorizontalFlip(),  # random flip
                transforms.CenterCrop(size=124),  # center crop
                transforms.ToTensor(),  # transform to tensor
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # normalize
            ]
        ),
    )

    img, label = train_dataset[0]


def test_torch_eo_image_segmentation_td():
    # Load the training dataset
    training_dataset = pytdml.io.read_from_json(
        "tests/data/semantic_segmentation/GID-5C.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    train_set, val_set, test_set = pytdml.ml.split_train_valid_test(
        training_dataset, 0.7, 0.2, 0.1
    )  # split dataset
    train_dataset = pytdml.ml.TorchEOImageSegmentationTD(  # create Torch train dataset
        train_set,
        class_map,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    val_dataset = pytdml.ml.TorchEOImageSegmentationTD(
        val_set,
        class_map,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )


def test_torch_scene_classification_td():
    training_dataset = pytdml.io.read_from_json(
        "tests/data/scene-classification/WHU-RS19.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    train_dataset = pytdml.ml.TorchSceneClassificationTD(  # create Torch train dataset
        training_dataset.data,
        root=".",
        class_map=class_map,
        transform=transforms.Compose(  # transform for the training set
            [
                transforms.ToTensor(),  # transform to tensor
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # normalize
            ]
        ),
    )

    img, label = train_dataset[0]


def test_torch_object_detection_td():
    # Load the training dataset
    training_dataset = pytdml.io.read_from_json(
        "tests/data/object-detection/COWC_partial.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    train_dataset = pytdml.ml.TorchObjectDetectionTD(  # create Torch train dataset
        training_dataset.data, root=".", class_map=class_map, transform=None
    )

    img, targets = train_dataset[0]


def test_torch_semantic_segmentation_td(tmp_path):
    training_dataset = pytdml.io.read_from_json(
        "tests/data/semantic_segmentation/GID-5C.json"
    )  # read from TDML json file

    train_set, val_set, test_set = pytdml.ml.split_train_valid_test(
        training_dataset, 0.7, 0.2, 0.1
    )  # split dataset

    # Prepare dummy png files and override the first sample paths
    img_path = tmp_path / "img.png"
    lbl_path = tmp_path / "lbl.png"

    dummy_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    dummy_lbl = (np.random.randint(0, 5, (64, 64))).astype(np.uint8)

    Image.fromarray(dummy_img).save(img_path)  # save dummy image
    Image.fromarray(dummy_lbl).save(lbl_path)  # save dummy label

    first = train_set[0]
    first.data_url[0] = str(img_path)
    first.labels[0].image_url = str(lbl_path)

    train_dataset = pytdml.ml.TorchSemanticSegmentationTD(
        train_set,
        root=".",
        classes=None,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    image, label = train_dataset[0]



def test_torch_change_detection_td(tmp_path):
    # Prepare dummy change detection samples
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    label_path = tmp_path / "label.png"

    before = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # dummy before image
    after = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # dummy after image
    label = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # dummy label image
    Image.fromarray(before).save(before_path)  # save before image
    Image.fromarray(after).save(after_path)  # save after image
    Image.fromarray(label).save(label_path)  # save label image

    td_list = [  # build a minimal td_list for TorchChangeDetectionTD
        SimpleNamespace(
            data_url=[str(before_path), str(after_path)],
            labels=[SimpleNamespace(image_url=str(label_path))],
        )
    ]

    train_dataset = pytdml.ml.TorchChangeDetectionTD(  # create Torch train dataset
        td_list, root=str(tmp_path), transform=transforms.ToTensor()
    )

    before_img, after_img, label_img = train_dataset[0]


def test_torch_stereo_td(tmp_path):
    # Prepare dummy stereo samples
    target_path = tmp_path / "target.png"
    ref_path = tmp_path / "ref.png"
    disp_path = tmp_path / "disp.png"

    target = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # dummy target image
    ref = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # dummy reference image
    disp = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # dummy disparity image
    Image.fromarray(target).save(target_path)  # save target image
    Image.fromarray(ref).save(ref_path)  # save reference image
    Image.fromarray(disp).save(disp_path)  # save disparity image

    td_list = [  # build a minimal td_list for TorchStereoTD
        SimpleNamespace(
            data_url=[str(target_path), str(ref_path)],
            labels=[SimpleNamespace(image_url=str(disp_path))],
        )
    ]

    train_dataset = pytdml.ml.TorchStereoTD(  # create Torch train dataset
        td_list, root=str(tmp_path), transform=transforms.ToTensor()
    )

    target_img, ref_img, disp_img = train_dataset[0]


def test_torch_3d_model_reconstruction_td(tmp_path):
    # Prepare a minimal directory structure to exercise Torch3DModelReconstructionTD
    tdml = SimpleNamespace(name="demo_mvs", data=[0])  # minimal tdml-like object

    base_dir = tmp_path / tdml.name
    cams_dir = base_dir / "Cams"
    depths_dir = base_dir / "Depths"
    imgs_dir = base_dir / "Images"
    view1_dir = imgs_dir / "view1"
    view2_dir = imgs_dir / "view2"

    cams_dir.mkdir(parents=True)  # create Cams dir
    depths_dir.mkdir(parents=True)  # create Depths dir
    view1_dir.mkdir(parents=True)  # create Images/view1 dir
    view2_dir.mkdir(parents=True)  # create Images/view2 dir

    # Cams/Depths entries are treated as strings and iterated character-by-character
    (cams_dir / "a").write_text("cam\n")  # camera file name: "a"
    (depths_dir / "d").write_bytes(b"")  # depth file name: "d"

    img1 = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)  # dummy image 1
    img2 = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)  # dummy image 2
    depth_img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)  # dummy depth image

    # These are used when listing Images/*
    Image.fromarray(img1).save(view1_dir / "img1", format="PNG")  # view1 contains img1
    Image.fromarray(img2).save(view2_dir / "img2", format="PNG")  # view2 contains img2

    # These are used when opening with relative paths in __getitem__
    Image.fromarray(img1).save(base_dir / "img1", format="PNG")  # save img1 at base
    Image.fromarray(img2).save(base_dir / "img2", format="PNG")  # save img2 at base
    Image.fromarray(depth_img).save(base_dir / "d", format="PNG")  # save depth at base
    (base_dir / "a").write_text("cam\n")  # save cam text at base

    cwd = os.getcwd()
    try:
        # _load_data uses os.listdir(item) where item is a subdir name under Images.
        os.chdir(str(imgs_dir))
        train_dataset = pytdml.ml.Torch3DModelReconstructionTD(  # create Torch train dataset
            tdml, str(tmp_path)
        )

        # __getitem__ opens files by relative name.
        os.chdir(str(base_dir))
        images, depth_imgs, cam_txt = train_dataset[0]
    finally:
        os.chdir(cwd)