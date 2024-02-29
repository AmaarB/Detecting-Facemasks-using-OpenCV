import numpy as np  
import os
from PIL import Image
import sys
import xmltodict

data_path = "facemask_data"

img_names=[] 

# get names of all image files
#os.walk generates the file names in a directory tree
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
     #os.path.join is a function which joins one or more path components in this aspect (dirname, filename)
        if os.path.join(dirname, filename)[-3:]!="xml":
          # if the file extension is not "xml" then add the file name to img_names list
          img_names.append(filename)

# paths of the folders of the downloaded data
image_path = "facemask_data/images/"
label_path = "facemask_data/annotations/"

# create two folders "with_mask" and "without_mask" in your system
# specify the path to the two folders below 
with_mask_path = "with_mask/"
without_mask_path = "without_mask/"

## function save faces from images to disk
def save_faces(img_name):
  # load image
  img = Image.open(image_path+img_name+".png")
  
  # load annotations file (xml)
  #fdopen() returns an open file object ocnnedted to the file descriptor (fd). Now defined functions can be performed on the file object
  with open(label_path+img_name+".xml") as fd:
    # convert xml to dictionary
    doc = xmltodict.parse(fd.read())
      
  if type(doc["annotation"]["object"]) == list:
    for i, obj in enumerate(doc["annotation"]["object"]):
      # find x and y coordinates of a face in the image
      xmin,ymin = int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']) 
      xmax,ymax = int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])

      # crop face from the image as an array
      face = np.array(img)[ymin:ymax,xmin:xmax,:]
      
      # convert face array to PIL image format
      face_image = Image.fromarray(face)

      # check the label of the extracted face and save it in the right folder
      if obj['name'] == 'with_mask':
          face_image.save(with_mask_path + img_name+"_"+str(i)+".png")

      else:
          face_image.save(without_mask_path + img_name+"_"+str(i)+".png")

# start saving images of faces
for i in img_names:
  save_faces(i.split(".")[0])