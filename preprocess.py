import os, glob, sys, shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
from .models.s3fd_facedet import S3FD
import cv2

# Load the face detector (you can ignore this part)
DET = S3FD(device='cuda')
    
class your_dataset(torch.utils.data.Dataset):
    def __init__(self, files):

        self.data   = files

        print('{:d} files in the dataset'.format(len(self.data)))

    def __getitem__(self, index):

      fname = self.data[index]
    
      try:
        # return image if read is successful
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_np, fname
      except:
        # return empty if not successful
        return np.array([]), np.array([]), fname

    def __len__(self):
      return len(self.data)

def Preprocess(data_dir, orig_path, temp_path):
    files = glob.glob(orig_path+'/*/*.jpg') + glob.glob(orig_path+'/*/*.png')
    print(len(files),'original images found.')
    DET = S3FD(device='cuda')
    

    dataset = your_dataset(files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    pbar = tqdm(loader)

    for data in pbar:

        # skip if there is read error
        if len(data[0].shape) != 4:
            print('Skipping {} - read error'.format(data[2]))
            continue

        image     = data[0][0].numpy()
        image_np  = data[1][0].numpy()
        fname     = data[2][0]

        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[0.5])

        pbar.set_description("{:d} faces detected in {}".format(len(bboxes),fname))

        ## this removes all images with no face detection or two or more face detections
        if len(bboxes) == 1:

            # padding value
            bsi = 300

            # find center and square size
            sx = int((bboxes[0][0]+bboxes[0][2])/2) + bsi
            sy = int((bboxes[0][1]+bboxes[0][3])/2) + bsi
            ss = int(max((bboxes[0][3]-bboxes[0][1]),(bboxes[0][2]-bboxes[0][0]))/1.5)

            # pad the image
            image = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

            # crop the face
            face = image[int(sy-ss):int(sy+ss),int(sx-ss):int(sx+ss)]

            # check that it is square and RGB
            if face.shape[0] == face.shape[1] and face.shape[0] > 10 and face.shape[2] == 3:

                face = cv2.resize(face,(256,256))
                outname = fname.replace(orig_path,temp_path).replace('.png','.jpg')
                os.makedirs(os.path.dirname(outname),exist_ok=True)
                cv2.imwrite(outname,face)

            else:

                print('[INFO] Non square image {}'.format(fname))
                
    # output_files = glob.glob(temp_path+'/*/*.jpg')
    shutil.make_archive(data_dir+'/vggface', 'zip', root_dir=temp_path)
    
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/vggface')
    parser.add_argument('--orig_path', type=str, default='/train')
    parser.add_argument('--temp_path', type=str, default='/preprocessed')
    
    arg = parser.parse_args()

    Preprocess(arg.data_dir, arg.orig_path, arg.temp_path)
