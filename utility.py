import matplotlib.pyplot as plt
import cv2 as cv
import os
import colorDetect_20250205 as colorDetect
import pandas as pd
import seaborn as sns

def plot_IMG(img_nameList_sub, path):
    for img_num, img_name in enumerate(img_nameList_sub):
        print(f'No. {img_num}, {img_name}')

        fig, ax = plt.subplots(3, 8, figsize=(3.5, 2), dpi=300)
        for tube_num in range(8):
            img_split = cv.imread(f"{path}/Split/{img_name}_Tube{tube_num+1}.jpg")
            img_courseg = cv.imread(f"{path}/courseSeg_tube/{img_name}_Tube{tube_num+1}_courseg.png", -1)
            img_fineseg = cv.imread(f"{path}/fineSeg_tube/{img_name}_Tube{tube_num+1}_fineseg.png", -1)

            ax[0,tube_num].imshow(cv.cvtColor(img_split, cv.COLOR_BGR2RGB))
            ax[1,tube_num].imshow(cv.cvtColor(img_courseg, cv.COLOR_BGRA2RGBA))
            ax[2,tube_num].imshow(cv.cvtColor(img_fineseg, cv.COLOR_BGRA2RGBA))
            ax[0,tube_num].axis('off')
            ax[1,tube_num].axis('off')
            ax[2,tube_num].axis('off')

        plt.tight_layout()

        fig.savefig(f"{path}/fineSeg_pic/{img_name}_fineseg.png")

        plt.close()
    
    return True


def plot_resultIMG(img_nameList_sub, colorResult_df, path):

    for img_num, img_name in enumerate(img_nameList_sub):
        
        result_df_sub = colorResult_df.query(f'Image == "{img_name}"')

        print(f'No. {img_num}, {img_name}')

        img_cleaned_HSV_list = []

        fig, ax = plt.subplots(4, 8, figsize=(3, 3), dpi=300)
        for tube_num in range(8):
            img_split = cv.imread(f"{path}/Split/{img_name}_Tube{tube_num+1}.jpg")
            img_courseg = cv.imread(f"{path}/courseSeg_tube/{img_name}_Tube{tube_num+1}_courseg.png", -1)
            img_fineseg = cv.imread(f"{path}/fineSeg_tube/{img_name}_Tube{tube_num+1}_fineseg.png", -1)
            img_cleaned = cv.imread(f"{path}/cleaned_tube/{img_name}_Tube{tube_num+1}_cleaned.png", -1)

            ax[0,tube_num].imshow(cv.cvtColor(img_split, cv.COLOR_BGR2RGB))
            ax[1,tube_num].imshow(cv.cvtColor(img_courseg, cv.COLOR_BGRA2RGBA))
            ax[2,tube_num].imshow(cv.cvtColor(img_fineseg, cv.COLOR_BGRA2RGBA))
            ax[3,tube_num].imshow(cv.cvtColor(img_cleaned, cv.COLOR_BGRA2RGBA))
            ax[0,tube_num].axis('off')
            ax[1,tube_num].axis('off')
            ax[2,tube_num].axis('off')
            ax[3,tube_num].axis('off')

            ax[0,tube_num].set_title(result_df_sub.iloc[0, tube_num+1], fontsize=5)

            img_cleaned_HSV_list.append(colorDetect.img2df(img_name, tube_num, img_cleaned)[1])

        plt.suptitle(img_name, fontsize=5)

        plt.tight_layout()

        fig.savefig(f"{path}/result_pic/{img_name}_result.png")

        plt.close()

        img_cleaned_HSV_df = pd.concat(img_cleaned_HSV_list)
        # img_cleaned_HSV_df.to_csv(f"{path}/result_boxplot/{img_name}.csv")

        sns.boxplot(img_cleaned_HSV_df.query('Mask == True'), x='Tube', y='H')
        plt.savefig(f"{path}/result_boxplot/{img_name}_boxplot.png")
        plt.close()
    
    return True


def plot_resultIMG_4tube(img_nameList_sub, colorResult_df, path):

    for img_num, img_name in enumerate(img_nameList_sub):
        
        result_df_sub = colorResult_df.query(f'Image == "{img_name}"')

        print(f'No. {img_num}, {img_name}')

        fig, ax = plt.subplots(4, 4, figsize=(3, 4), dpi=600)

        img_cleaned_HSV_list = []
        for tube_num in range(4):
            img_split = cv.imread(f"{path}/Split/{img_name}_Tube{tube_num+1}.jpg")
            img_courseg = cv.imread(f"{path}/courseSeg_tube/{img_name}_Tube{tube_num+1}_courseg.png", -1)
            img_fineseg = cv.imread(f"{path}/fineSeg_tube/{img_name}_Tube{tube_num+1}_fineseg.png", -1)
            img_cleaned = cv.imread(f"{path}/cleaned_tube/{img_name}_Tube{tube_num+1}_cleaned.png", -1)

            ax[0,tube_num].imshow(cv.cvtColor(img_split, cv.COLOR_BGR2RGB))
            ax[1,tube_num].imshow(cv.cvtColor(img_courseg, cv.COLOR_BGRA2RGBA))
            ax[2,tube_num].imshow(cv.cvtColor(img_fineseg, cv.COLOR_BGRA2RGBA))
            ax[3,tube_num].imshow(cv.cvtColor(img_cleaned, cv.COLOR_BGRA2RGBA))
            ax[0,tube_num].axis('off')
            ax[1,tube_num].axis('off')
            ax[2,tube_num].axis('off')
            ax[3,tube_num].axis('off')

            ax[0,tube_num].set_title(result_df_sub.iloc[0, tube_num+1], fontsize=5)

            img_cleaned_HSV_list.append(colorDetect.img2df(img_name, tube_num, img_cleaned)[1])

        plt.suptitle(img_name, fontsize=5)

        # plt.tight_layout()
        # Adjust layout to reduce space
        plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Fine-tune these values for the best appearance


        fig.savefig(f"{path}/result_pic/{img_name}_result.png")

        plt.close()

        img_cleaned_HSV_df = pd.concat(img_cleaned_HSV_list)
        # img_cleaned_HSV_df.to_csv(f"{path}/result_boxplot/{img_name}.csv")

        sns.boxplot(img_cleaned_HSV_df.query('Mask == True'), x='Tube', y='H')
        plt.savefig(f"{path}/result_boxplot/{img_name}_boxplot.png")
        plt.close()
    
    return True


def find_img_name(file_dir):
    File_Name=[]
    for files in os.listdir(file_dir):
        if (os.path.splitext(files)[1] == '.jpg') | (os.path.splitext(files)[1] == '.jpeg'):
            File_Name.append(os.path.splitext(files)[0])
    return File_Name