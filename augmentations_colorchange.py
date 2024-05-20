import albumentations as A
import cv2
import os


p=1
file_path = 'C:/Users/heenatigalakanat3478/Saskatchewan Polytechnic/Peter Lucas Gravel Pit AI_GRP - General/ARD Grant/Data/Hillshade images/Manitoba verification images/25x25/nonpit/'
output_path='C:/Users/heenatigalakanat3478/Saskatchewan Polytechnic/Peter Lucas Gravel Pit AI_GRP - General/ARD Grant/Data/Hillshade images/Manitoba verification images/25x25/augment/nonpit/'

def get_filenames(file_path):
    return os.listdir(file_path)

'''
returns a list of transformations
Only safe augmentationg are used.
safe: no color changes, no pixel level changes
'''
def Get_transform_list():
    transform= [
        A.RGBShift(p=p),
        A.HueSaturationValue(p=p),
        A.ChannelShuffle(p=p),
        A.CLAHE(p=p),
        A.RandomBrightness(p=p),
        A.RandomGamma(p=p),
        A.ToGray(p=p),
        A.JpegCompression(p=p),
        A.ChannelDropout(p=p),
        A.ColorJitter(p=p),
        A.Equalize(p=p),

    ]
    name_list= [
        'RGBShift','HueSaturationValue','ChannelShuffle','CLAHE','RandomBrightness','RandomGamma','ToGray','JpegCompression','ChannelDropout','ColorJitter','Equalize'
    ]
    return transform,name_list

def apply_transform(image,transformation):
    return transformation(image=image)["image"]

def check_save_path(path):
    if os.path.isdir(path):
        return path
    else:
        os.mkdir(path)
        return path
def createFolderifNotExists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    files=get_filenames(file_path)
    transform_list,name_list=Get_transform_list()
    save_path=check_save_path(output_path)
    
    for file in files:
        i=0
        filename=file.rsplit('.', 1)[0]
        
        if file.endswith('.jpg'):
            try:
                image= cv2.imread(file_path+file)
                for a in transform_list:
                    createFolderifNotExists(save_path+'/'+str(i))
                    aug_img=apply_transform(image,a)                    
                    cv2.imwrite(save_path+'/'+str(i)+'/'+filename+str(i)+name_list[i]+'.jpg',aug_img)
                    print('Saved: '+filename+str(i)+name_list[i]+'.jpg')
                    i+=1
            
            except Exception as e:
                print(e)
                continue
            
    

if __name__ == "__main__":
    main()





