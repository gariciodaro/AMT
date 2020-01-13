import cv2
from auxiliar import helper


class BowAux():
    """docstring for BowAux"""
    def __init__(self, box=50, n_cluster=45):
        
        self.box=box
        self.n_cluster=n_cluster
        self.detect = cv2.xfeatures2d.SIFT_create()
        self.extract = cv2.xfeatures2d.SIFT_create()
        self.bow_kmeans_trainer=cv2.BOWKMeansTrainer(self.n_cluster)
        self.flann_params = dict(algorithm = 1, trees = 10)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})

    def extract_sift(self,image):
        extract=self.extract
        detect=self.detect
        return extract.compute(image, detect.detect(image))[1]

    def new_bow_vocabulary(self,i_train,path_images):
        extract_bow = cv2.BOWImgDescriptorExtractor(self.extract, self.matcher)

        for each in i_train.name:
            name_image=each
            image = cv2.imread(path_images+name_image)
            masked_image=helper.image_prepare(image,guasian_kernel_dim=(11,11))
            masked_image=helper.gray_equalized(masked_image,greay_step=False)
            #pre_image=helper.image_prepare(image,guasian_kernel_dim=(11,11))
            #contours_mask=helper.get_contour_mask(pre_image)
            #masked_image=helper.multiple_image_contour(image,contours_mask)
            #masked_image=helper.gray_equalized(masked_image)
            #masked_image=helper.image_prepare(masked_image,guasian_kernel_dim=(11,11))
            
            
            grid_images,map_grid_images=helper.image_splitter(masked_image,self.box)

            for rectan_image in grid_images:
                #plt.imshow(rectan_image)
                try:
                    self.bow_kmeans_trainer.add(self.extract_sift(rectan_image))
                except:
                    pass
        self.voc = self.bow_kmeans_trainer.cluster()
        extract_bow.setVocabulary(self.voc)
        self.extract_bow=extract_bow
        return self.voc


    def input_bow_vocabulary(self,voc):
        self.voc = voc
        extract_bow = cv2.BOWImgDescriptorExtractor(self.extract, self.matcher)
        extract_bow.setVocabulary(self.voc)
        self.extract_bow=extract_bow






        