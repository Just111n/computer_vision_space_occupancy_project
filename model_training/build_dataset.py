import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
import argparse

def unzip(ls: list[tuple]):
    """
    Unzips a list of constant-length tuples into its individual lists
    
    Basically the same as array transposition

    Example:
    ```python
    tup_ls = [(1,2), (3,4)]
    unzip(tup_ls)
    >>> ([1,3], [2,4])
    ```
    
    :param ls: A list of constant-length tuples
    :type ls: list[tuple]
    :return: A tuple of lists
    :rtype: tuple[list[Any], ...]
    """

    # check that all tuples in list have the same length
    assert all(len(t) == max([len(t) for t in ls]) for t in ls), 'Varying tuple length'

    # unpacks each tuple in ls as arguments for zip()
    # each item in zip object is converted to list
    # return lists as a tuple
    return tuple(list(e) for e in zip(*ls))

def get_train_val_test_filepaths(src_dir: str, img_subdir: str, ann_subdir: str, dset_name: str) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    Retrieves file paths for training, validation, and testing images and annotations from the mit_indoor dataset, and splits the
    training set into train and validation subsets.
    
    :param src_dir: The directory path where your dataset is stored.
    :type src_dir: str
    :param img_subdir: The sub-directory path that stores the images in `src_dir`.
    :type img_subdir: str
    :param ann_subdir: The sub-directory path that contains the annotations in `src_dir`.
    :type ann_subdir: str
    :param dset_name: The name of the data subset of mit_indoor for which you want to retrieve the file paths.
    :type dset_name: str
    :return: The following lists in order:

        1. `train_img_fp_ls`: List of file paths for training images
        2. `train_ann_fp_ls`: List of file paths for training annotations
        3. `val_img_fp_ls`: List of file paths for validation images
        4. `val_ann_fp_ls`: List of file paths for validation annotations
        5. `test_img_fp_ls`: List of file paths for testing images
    
    :rtype: tuple[list[str]]
    """

    # get image and annotation file names from source directory
    img_fn_ls = [path for path in os.listdir(os.path.join(src_dir, img_subdir, dset_name)) if path.split('.')[-1].lower() == 'jpg']
    ann_fn_ls = [path for path in os.listdir(os.path.join(src_dir, ann_subdir, dset_name)) if path.split('.')[-1].lower() == 'xml']

    # number of images != number of annotations
    # corresponding imag-annotation pairs share same name
    # get names of each image and annotation
    img_names_set = set(['.'.join(fn.split('.')[:-1]) for fn in img_fn_ls])
    ann_names_set = set(['.'.join(fn.split('.')[:-1]) for fn in ann_fn_ls])

    # use images with corresponding annotations for training & validation
    # other images reserved for testing
    train_val_img_names = img_names_set.intersection(ann_names_set)
    test_img_names = img_names_set - train_val_img_names
    
    # get relative paths of images and annotations
    train_val_img_fp_ls = sorted([os.path.join(src_dir, img_dir, dset_name, n+'.jpg') for n in train_val_img_names])
    train_val_ann_fp_ls = sorted([os.path.join(src_dir, ann_dir, dset_name, n+'.xml') for n in train_val_img_names])
    test_img_fp_ls = sorted([os.path.join(src_dir, img_dir, dset_name, n+'.jpg') for n in test_img_names])

    # split training and validation datasets
    # can play with the random state
    train_img_fp_ls, val_img_fp_ls, train_ann_fp_ls, val_ann_fp_ls = train_test_split(train_val_img_fp_ls, train_val_ann_fp_ls,
                                                                                      test_size=0.2, random_state=42)

    return train_img_fp_ls, train_ann_fp_ls, val_img_fp_ls, val_ann_fp_ls, test_img_fp_ls

def batch_get_train_val_test_filepaths(src_dir: str, img_subdir: str, ann_subdir: str, dset_names: list[str]):
    train_img_fp_ls = []
    train_ann_fp_ls = []
    val_img_fp_ls = []
    val_ann_fp_ls = []
    test_img_fp_ls = []

    for dset in dset_names:
        a,b,c,d,e = get_train_val_test_filepaths(src_dir, img_subdir, ann_subdir, dset)
        train_img_fp_ls += a
        train_ann_fp_ls += b
        val_img_fp_ls += c
        val_ann_fp_ls += d
        test_img_fp_ls += e

    return train_img_fp_ls, train_ann_fp_ls, val_img_fp_ls, val_ann_fp_ls, test_img_fp_ls

def batch_read_images(img_fp_ls: list[str]) -> tuple[list[str], list[np.ndarray]]:
    """
    Read images from list of file paths using `cv2.imread`, filter out any invalid images (NoneType), and return a
    list of valid image file paths and a list of corresponding images.
    
    :param img_fp_ls: List of file paths to image files
    :type img_fp_ls: list[str]
    :return: A tuple containing a list of valid image filepaths and a list of corresponding image data.
    :rtype: (list[str], list[np.ndarray])
    """
    # read images from list of filepaths
    img_ls = [cv2.imread(fp) for fp in img_fp_ls]

    return img_ls

def remove_broken_images(img_ls: list[np.ndarray]):
    # identify invalid images (NoneType)
    num_invalid = sum(1 for img in img_ls if img is None)

    # filter out invalid images and the corresponding XML filepaths
    out_img_ls = [img for img in img_ls if img is not None]
    if num_invalid > 0:
        print(f'Removed {num_invalid} invalid image' + ('s' if num_invalid > 1 else ''))

    return out_img_ls

def remove_broken_images_with_xml_fp(img_ls: list[np.ndarray], xml_fp_ls: list[str]):
    # identify invalid images (NoneType)
    num_invalid = sum(1 for img in img_ls if img is None)

    # filter out invalid images and the corresponding XML filepaths
    img_xml_ls = [(img, xml_fp) for img, xml_fp in zip(img_ls, xml_fp_ls) if img is not None]
    if num_invalid > 0:
        print(f'Removed {num_invalid} invalid image' + ('s' if num_invalid > 1 else ''))
    
    out_img_ls, out_xml_fp_ls = unzip(img_xml_ls)

    return out_img_ls, out_xml_fp_ls

def xml_to_mask_map(img: np.ndarray, ann_xml: str) -> tuple[np.ndarray, dict[int, str]]:
    """
    Converts annotation XML data into a mask image and a mapping of polygon IDs to object
    names.

    The mask would have the same size as the corresponding image. 
    Each mask contain pixel values corresponding to polygon IDs in the mapping. 
    Pixel value of 0 represents background. 
    Smaller polygons would have larger polygon ID values.
    
    :param img: A NumPy array representing an image
    :type img: np.ndarray
    :param ann_xml: Path to XML file containing annotations for objects in the corresponding image
    :type ann_xml: str
    :return: The converted mask for the input image, and the corresponding polygon class mapping
    :rtype: (np.ndarray, dict[int, str])
    """

    # get the root of the XML parse tree
    ann_tree = ET.parse(ann_xml)
    ann_root = ann_tree.getroot()

    # get list of objects from XML
    objects = []
    for obj in ann_root.findall('object'):
        # get object name
        obj_name = obj.find('name').text.strip()

        # check if object is related to chair or person (singular)
        is_chair = ('chair' in obj_name and 'chairs' not in obj_name and 'person' not in obj_name) or obj_name == 'chiar'
        is_person = 'person' in obj_name

        if is_chair or is_person:
            # get the corresponding polygon as a list of points
            polygon = []
            for pt in obj.find('polygon').findall('pt'):
                # get point coordinates
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)

                # add point to polygon
                polygon.append((x,y))

            # add object to list of objects as a dict
            objects.append({'name': obj_name, 'polygon': polygon})

    # initialise mask with shape of original image
    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    # sort polygons by decreasing area
    # prioritise smaller objects
    objects.sort(key=lambda obj: cv2.contourArea(np.array(obj['polygon'])), reverse=True)

    polygon_map = dict()
    for i, obj in enumerate(objects):
        polygon = np.array(obj['polygon'], dtype=np.int32)

        # fill polygon with polygon ID value
        cv2.fillPoly(mask, [polygon], color=i+1)

        # assign each polygon with a positive integer ID
        # map polygon ID to corresponding class name
        polygon_map[i+1] = obj['name']

    return mask, polygon_map

def batch_xml_to_mask_map(img_ls: list[np.ndarray], xml_fp_ls: list[str]) -> tuple[list[np.ndarray], list[dict[int, str]]]:
    """
    Apply `xml_to_mask_map` on lists of images and corresponding xml filepaths.

    Returns list of masks and list of corresponding class mappings

    :param img_ls: A list of NumPy arrays representing images
    :type img_ls: list[np.ndarray]
    :param xml_fp_ls: A list of paths to XML files containing annotations for objects in the corresponding images
    :type xml_fp_ls: list[str]
    :return: A list of converted masks for the input images, and the corresponding polygon class mappings
    :rtype: (list[np.ndarray], list[dict[int, str]])
    """

    # apply xml_to_mask_map for every image-xml path pair
    mask_map_ls = [xml_to_mask_map(img, xml) for img, xml in zip(img_ls, xml_fp_ls)]

    out_mask_ls, out_map_ls = unzip(mask_map_ls)
    return out_mask_ls, out_map_ls

def clean_mappings(map_ls: list[dict[int, str]]):
    """
    Clean the polygon mappings such that all individual parts of 'chair' and 'person' 
    (not multiple chairs or persons) are considered 'chair' and 'person' respectively.

    :param map_ls: A list of polygon mappings to class names
    :type map_ls: list[dict[int, str]]
    :return: A list of cleaned polygon mappings to class names
    :rtype: list[dict[int, str]]
    """

    # make a copy of the mapping list
    # prevent overwriting the original list
    out_map_ls = [o.copy() for o in map_ls]

    for i in range(len(out_map_ls)):
        obj = out_map_ls[i]
        for polygon_id, class_name in obj.items():
            # if polygon contains a single chair (not multiple)
            # change class to 'chair'
            if 'chair' in class_name and 'chairs' not in class_name and class_name != 'chair':
                obj[polygon_id] = 'chair'
            if class_name == 'chiar':
                obj[polygon_id] = 'chair'

            # if polygon contains a single person (not multiple)
            # change class to 'person'
            if 'person' in class_name  and class_name != 'person':
                obj[polygon_id] = 'person'

        # update changes to mapping
        out_map_ls[i] = obj

    return out_map_ls

def remove_unlabeled_data(img_ls: list[np.ndarray], mask_ls: list[np.ndarray], map_ls: list[dict[int, str]]):
    new_zip = [(img, mask, mapping) for img, mask, mapping in zip(img_ls, mask_ls, map_ls)if len(mapping) > 0]
    return unzip(new_zip)

def append_file_ext(filename_ls: list[str], ext_ls: list[str]):
    """
    Append a list of file extensions to a list of filenames.
    Output dimension = (number of file_extensions, number of filenames)

    :param filename_ls: A list of filenames
    :type filename_ls: list[str]
    :param ext_ls: A list of file extensions
    :type ext_ls: list[str]
    :return: A list of cleaned polygon mappings to class names
    :rtype: list[dict[int, str]]
    """

    # create a tuple of lists of filenames with extensions
    out = tuple(['.'.join([filename, ext]) for filename in filename_ls] for ext in ext_ls)
    return out if len(ext_ls) > 1 else out[0]

def batch_save_images(img_ls: list[np.ndarray], fn_ls: list[str], dst_dir: str):
    """
    Save a list of images given the corresponding list of filenames 
    and the destination directory using `cv2.imwrite`. This function assumes that 
    the filenames have valid image extensions.

    :param img_ls: A list of images as NumPy arrays
    :type img_ls: list[np.ndarray]
    :param fn_ls: A list of filenames
    :type fn_ls: list[str]
    :param dst_dir: Destination directory path
    :type dst_dir: str
    :return: None
    """

    # prepare destination directory
    prepare_dir(dst_dir)
    
    for img, fn in zip(img_ls, fn_ls):
        # save image
        cv2.imwrite(os.path.join(dst_dir, fn), img)

def batch_save_mappings(map_ls: list[dict[int, str]], fn_ls: list[str], dst_dir: str):
    
    """
    Save a list of mappings given the corresponding list of filenames 
    and the destination directory using `json.dump`. This function assumes that 
    each filename in the input has the `.json` extension.

    :param map_ls: A list of mappings as dicts
    :type img_ls: list[dict[int, str]]
    :param fn_ls: A list of filenames
    :type fn_ls: list[str]
    :param dst_dir: Destination directory path
    :type dst_dir: str
    :return: None
    :rtype: None
    """

    # prepare destination directory
    prepare_dir(dst_dir)

    for m, fn in zip(map_ls, fn_ls):
        with open(os.path.join(dst_dir, fn), 'w') as json_out:
            # save mapping as json
            # add indent for human readability
            json.dump(m, json_out, indent=4)
            json_out.close()

def prepare_dir(dir: str):
    """
    Prepare directory for rewriting.

    Create directories if they do not exist already. Otherwise, clear the contents 
    of the directory.
    
    :param dir: Directory path
    :type dir: str
    :return: None
    :rtype: None
    """
    if os.path.isdir(dir):
        try:
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))
        except Exception as e:
            print(f'Error removing {dir}: {e}')
    os.makedirs(dir, exist_ok=True)

if __name__ == '__main__':
    # define variables
    src_dir = '../mit_indoor'
    dst_dir = './data'
    img_dir = 'Images'
    ann_dir = 'Annotations'
    dset_names = [
        'meeting_room',
        'classroom',
        'office',
        'auditorium',
        'inside_bus',
        'library',
        'tv_studio',
    ]


    # split the dataset to train-val-test
    print('Splitting dataset...')
    train_img_fp_ls, train_ann_fp_ls, val_img_fp_ls, val_ann_fp_ls, test_img_fp_ls = batch_get_train_val_test_filepaths(src_dir, img_dir, ann_dir, dset_names)

    # read images from dataset
    print('Reading images...')
    train_img_ls = batch_read_images(train_img_fp_ls)
    val_img_ls = batch_read_images(val_img_fp_ls)
    test_img_ls = batch_read_images(test_img_fp_ls)

    # remove images that were not read properly
    train_img_ls, train_ann_fp_ls = remove_broken_images_with_xml_fp(train_img_ls, train_ann_fp_ls)
    val_img_ls, val_ann_fp_ls = remove_broken_images_with_xml_fp(val_img_ls, val_ann_fp_ls)
    test_img_ls = remove_broken_images(test_img_ls)

    # convert xml to masks and mappings
    print('Generating masks and mappings from XML...')
    train_mask_ls, train_map_ls = batch_xml_to_mask_map(train_img_ls, train_ann_fp_ls)
    val_mask_ls, val_map_ls = batch_xml_to_mask_map(val_img_ls, val_ann_fp_ls)

    # remove data points where mapping is empty
    train_img_ls, train_mask_ls, train_map_ls = remove_unlabeled_data(train_img_ls, train_mask_ls, train_map_ls)
    val_img_ls, val_mask_ls, val_map_ls = remove_unlabeled_data(val_img_ls, val_mask_ls, val_map_ls)

    # clean polygon mappings
    print('Cleaning mappings...')
    train_map_ls = clean_mappings(train_map_ls)
    val_map_ls = clean_mappings(val_map_ls)

    # generate file names
    print('Generating file names...')
    train_out_names = [f'train_{str(i).zfill(3)}' for i in range(len(train_img_ls))]
    val_out_names = [f'val_{str(i).zfill(3)}' for i in range(len(val_img_ls))]
    test_out_names = [f'test_{str(i).zfill(3)}' for i in range(len(test_img_ls))]

    train_out_img_fn_ls, train_out_mask_fn_ls, train_out_map_fn_ls = append_file_ext(train_out_names, ['jpg', 'png', 'json'])
    val_out_img_fn_ls, val_out_mask_fn_ls, val_out_map_fn_ls = append_file_ext(val_out_names, ['jpg', 'png', 'json'])
    test_out_img_fn_ls = append_file_ext(test_out_names, ['jpg'])

    # ensure that the destination directory exists
    os.makedirs('data', exist_ok=True)

    # save images, masks, and mappings
    print('Saving images and mappings...')
    batch_save_images(train_img_ls, train_out_img_fn_ls, os.path.join(dst_dir, 'images/train'))
    batch_save_images(train_mask_ls, train_out_mask_fn_ls, os.path.join(dst_dir, 'masks/train'))
    batch_save_mappings(train_map_ls, train_out_map_fn_ls, os.path.join(dst_dir, 'mappings/train'))
    batch_save_images(val_img_ls, val_out_img_fn_ls, os.path.join(dst_dir, 'images/val'))
    batch_save_images(val_mask_ls, val_out_mask_fn_ls, os.path.join(dst_dir, 'masks/val'))
    batch_save_mappings(val_map_ls, val_out_map_fn_ls, os.path.join(dst_dir, 'mappings/val'))
    batch_save_images(test_img_ls, test_out_img_fn_ls, os.path.join(dst_dir, 'images/test'))

    print(f'Successfully created dataset in {dst_dir}')



    


