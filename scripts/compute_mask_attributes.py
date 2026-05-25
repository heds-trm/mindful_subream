import SimpleITK as sitk
import os
import sys
from pathlib import Path
import pandas as pd
import csv
import argparse
import numpy as np

def compute_attributes(filename_mask, filter_odd_results = False, classes = {'M':1, 'B':0}, maxval=None, overwrite=False):
  img = sitk.ReadImage(filename_mask)
  
  # Optional binarization
  if maxval is not None:
      # Step 1: BinaryThreshold to 255
      minmax = sitk.MinimumMaximumImageFilter()
      minmax.Execute(img)
      img_min = minmax.GetMinimum()
      img_max = minmax.GetMaximum()
      
      # print(f"min: {img_min}, max: {img_max}, maxval: {maxval}")

      img_bin = sitk.BinaryThreshold(
          img,
          lowerThreshold=img_min+1,
          upperThreshold=img_max,
          insideValue=255,
          outsideValue=0
      )

      # Step 2: Convert to NumPy and determine dtype
      img_np = sitk.GetArrayFromImage(img_bin)
      current_dtype = img_np.dtype

      # Step 3: Check if maxval fits into current dtype
      #maxval_safe = maxval
      if np.can_cast(maxval, current_dtype, casting='safe'):
          maxval_casted = np.array(maxval).astype(current_dtype)
      else:
          # Promote to larger dtype (e.g., from uint8 to uint16)
          new_dtype = np.promote_types(current_dtype, np.array(maxval).dtype)
          img_np = img_np.astype(new_dtype)
          maxval_casted = maxval  # now fits safely
          # print(f"Promoting dtype from {current_dtype} to {new_dtype} for maxval {maxval}")

      # Step 4: Replace 255 → maxval
      img_np[img_np == 255] = maxval_casted

      # Step 5: Recreate image with correct type
      img = sitk.GetImageFromArray(img_np)
      img.CopyInformation(img_bin)

      if overwrite:
          sitk.WriteImage(img, filename_mask, useCompression=True)  


  stats = sitk.LabelIntensityStatisticsImageFilter()
  stats.Execute(img,img)
  labels = stats.GetLabels()
  label = max(labels)

  if(len(labels)>1):
    print("<!>WARNING: More than one label in mask image, taking the largest one. Check statistics<!>")

  #print(f"{labels}")
  
  size_w = stats.GetBoundingBox(label)[3] * img.GetSpacing()[0]
  size_h = stats.GetBoundingBox(label)[4] * img.GetSpacing()[1]
  size_d = stats.GetBoundingBox(label)[5] * img.GetSpacing()[2]
  
  max_allowed_size = 100.0
  
  # if filter_odd_results:
  #   if size_w > max_allowed_size or size_h > max_allowed_size or size_d > max_allowed_size:
  #     print(f"filtering strange mask {filename_mask}")
      
  #     #threshold image
  #     img = sitk.BinaryThreshold( img, label-1, label, label, 0 )
      
  #     # remove small elements
  #     img = sitk.BinaryMorphologicalOpening( img, (2,2,2), sitk.sitkBall, 0, label )

  #     sitk.WriteImage(img, "tamama.mha")

  #     stats.Execute(img,img)
  #     labels = stats.GetLabels()
  #     label = max(labels)
  
  stats_dict = {}
  stats_dict['mask_filename'] = filename_mask
  for c in classes:
    if f"_{c}" in filename_mask:
      stats_dict['type'] = classes[c]
      break
      
  if 'type' not in stats_dict:
    return None
    
  # bbox: [xstart, ystart, start, xsize, ysize, zsize].
  start = (stats.GetBoundingBox(label)[0],stats.GetBoundingBox(label)[1],stats.GetBoundingBox(label)[2])
  
  index_center = [
    start[0] + stats.GetBoundingBox(label)[3] * 0.5, 
    start[1] + stats.GetBoundingBox(label)[4] * 0.5, 
    start[2] + stats.GetBoundingBox(label)[5] * 0.5 
    ]
  center = img.TransformContinuousIndexToPhysicalPoint(index_center)
  
  stats_dict['bounding_box_center_x'] = center[0]
  stats_dict['bounding_box_center_y'] = center[1]
  stats_dict['bounding_box_center_z'] = center[2]
  stats_dict['bounding_box_size_x'] = stats.GetBoundingBox(label)[3] * img.GetSpacing()[0]
  stats_dict['bounding_box_size_y'] = stats.GetBoundingBox(label)[4] * img.GetSpacing()[1]
  stats_dict['bounding_box_size_z'] = stats.GetBoundingBox(label)[5] * img.GetSpacing()[2]  
  
  stats_dict['volume'] = stats.GetPhysicalSize(label)
  stats_dict['center_of_gravity_x'] = stats.GetCenterOfGravity(label)[0]
  stats_dict['center_of_gravity_y'] = stats.GetCenterOfGravity(label)[1]
  stats_dict['center_of_gravity_z'] = stats.GetCenterOfGravity(label)[2] 
  stats_dict['elongation'] = stats.GetElongation(label)
  stats_dict['equivalent_ellipsoid_diameter_1'] = stats. GetEquivalentEllipsoidDiameter(label)[0]
  stats_dict['equivalent_ellipsoid_diameter_2'] = stats. GetEquivalentEllipsoidDiameter(label)[1]
  stats_dict['equivalent_ellipsoid_diameter_3'] = stats. GetEquivalentEllipsoidDiameter(label)[2]  
  stats_dict['flatness'] = stats.GetFlatness(label)
  return stats_dict

if __name__ == '__main__':
  # parser
  parser = argparse.ArgumentParser()
  parser.add_argument("mask_mha_directory", type=str, help="directory with mask mha files")
  parser.add_argument("--include_file", type=str, help="include only masks in provided file")
  parser.add_argument("--output_csv_file", type=str, help="output csv filename")
  parser.add_argument("--output_xlsx_file", type=str, help="output xlsx filename")
  parser.add_argument("--classes", type=str, help="classes as L1:int1,L2:int2;...", default='B:0,M:1')  
  parser.add_argument("--binarize", type=np.uint16, default=None, help="Binarize image to max value (e.g., 255).")
  parser.add_argument("--overwrite", action="store_true", help="Overwrite the original image with the binarized version.")

  args = parser.parse_args()
    
  directory = args.mask_mha_directory # directory containing masks

  dump_to_csv = True if args.output_csv_file else False
  dump_to_xlsx = True if args.output_xlsx_file else False
  
  if not dump_to_csv and not dump_to_xlsx:
    print("Please specify at least one output file format.")
    sys.exit()

  if dump_to_csv:
    output_attributes_file_csv = args.output_csv_file
  
  if dump_to_xlsx:
    output_attributes_file_xlsx = args.output_xlsx_file
  
  include_fnames = []
  if args.include_file:
    if not os.path.exists(args.include_file):
      print(f"include file {args.include_file} does not exist")
    with open(args.include_file) as file:
      lines = file.readlines()
      include_fnames = [line.rstrip() for line in lines]

  classes = {}

  for kv in args.classes.split(','):
    if ':' in kv:
      k = kv.split(':')[0]
      v = kv.split(':')[1]
      classes[k] = int(v)
  
  if not classes:
    print(f"Provided classes config {classes} is incorrect")
    sys.exit()
  
  print(f"classes with corresponding code: {classes}")

  mha_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and '.mha' == (Path(directory) / Path(f)).suffix] 
  mha_files.sort()

  mask_attributes = []

  for mask_filename in mha_files:
    if len(include_fnames) > 0:
      found = False
      for f in include_fnames:
        if f in mask_filename:
          found = True
          break
      if not found: continue
      
    print(f"processing {Path(mask_filename).name}")
    mask_atts = compute_attributes(mask_filename, filter_odd_results=True, classes = classes, maxval=args.binarize, overwrite=args.overwrite)
    if mask_atts is None:
      print(f"{mask_filename} is incorrect with respect to classes {classes}")
    mask_attributes.append(mask_atts)

  if len(mask_attributes) > 0:
    keys = mask_attributes[0].keys()

    if dump_to_csv:
      with open(output_attributes_file_csv, 'w', newline='') as output_file:
          dict_writer = csv.DictWriter(output_file, keys)
          dict_writer.writeheader()
          dict_writer.writerows(mask_attributes)

    if dump_to_xlsx:
      df = pd.DataFrame.from_dict(mask_attributes)
      df.to_excel(output_attributes_file_xlsx, index=False)
  else:
    print("cannot find valid mask mha file, check directory or content of include_file.")




    
