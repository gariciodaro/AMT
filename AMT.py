from auxiliar import helper
from auxiliar.classes.Predictor import Predictor
from auxiliar.classes.BowAux import BowAux

import sys
import cv2
import numpy as np

# load code vectors templates
tail_code_vector=helper.load_object('./pickle_objects/tail_code_vector.file')
longest_superior_fin=helper.load_object('./pickle_objects/longest_superior_fin.file')
lower_fin=helper.load_object('./pickle_objects/lower_fin.file')
lower_longer_fin=helper.load_object('./pickle_objects/lower_longer_fin.file')

co=0
#Bag of words parameters for mouth
box_mouth=50
n_cluster_mouth=45
bow_aux_mouth=BowAux(box=box_mouth,n_cluster=n_cluster_mouth)
voc_mouth=helper.load_object('./pickle_objects/voc_mouth.file')
bow_aux_mouth.input_bow_vocabulary(voc=voc_mouth)
clf_mouth=helper.load_object("./pickle_objects/F_mouth_clf.file")
scaler_mouth=helper.load_object("./pickle_objects/F_mouth_scaler.file")
predictor_mouth=Predictor(best_clf=clf_mouth,min_max_scaler=scaler_mouth)

#Bag of words parameters for fins
box_fins=50
n_cluster_fins=200
bow_aux_fins=BowAux(box=box_fins,n_cluster=n_cluster_fins)
voc_fins=helper.load_object('./pickle_objects/voc_fins.file')
bow_aux_fins.input_bow_vocabulary(voc=voc_fins)
clf_fins=helper.load_object("./pickle_objects/F_fins_clf.file")
scaler_fins=helper.load_object("./pickle_objects/F_fins_scaler.file")
predictor_fins=Predictor(best_clf=clf_fins,min_max_scaler=scaler_fins)




if __name__== "__main__":
    image_1=sys.argv[1]
    search_fin=sys.argv[2]
    confusion_string=sys.argv[3]
    if confusion_string=="True":
        confusion=True
    else:
        confusion=False



    path='./images/'
    image = cv2.imread(path+image_1)
    #helper.display(image)
    pre_image=helper.image_prepare(image,guasian_kernel_dim=(11,11))
    #helper.display(pre_image)
    contours_mask=helper.get_contour_mask(pre_image)
    #helper.display(contours_mask)
    masked_image=helper.multiple_image_contour(image,contours_mask)
    #helper.display(masked_image)
    masked_image=helper.gray_equalized(masked_image)
    Mo_dim=None

    ################### get dimension
    CPd_coordinates=helper.tail_detection(masked_image,"CPd")
    CFd_coordinates=helper.tail_detection(masked_image,"CFd",CPd_coordinates[1])
    TL_coordinates =helper.TL(masked_image)
    Eh_coordinates=helper.eye_detection(image,contours_mask,masked_image,dim_to_calculate="Eh",confusion=confusion,
                                        show_image=False)
    Hd_coordinates=helper.eye_detection(image,contours_mask,masked_image,dim_to_calculate="Hd",confusion=confusion)
    Ed_coordinates=helper.eye_detection(image,contours_mask,masked_image,dim_to_calculate="Ed",confusion=confusion,
                                        Hd_coor=Hd_coordinates)
    Bd2_prepare_image=helper.Bd2_prepare(masked_image)
    #helper.display(Bd2_prepare_image)
    Bd_coordinates=helper.Bd2(masked_image,Bd2_prepare_image,
                              min_col_searh=Hd_coordinates[1],max_col_searh=CPd_coordinates[1])

    if search_fin=="yes":
        try:
            row_fins_coordinate,col_fins_coordinate,image=helper.image_prediction(name_image=image_1,
                                                                                  model=predictor_fins,
                                                                          box=bow_aux_fins.box,
                                                                          extract_bow=bow_aux_fins.extract_bow,
                                                                          detect=bow_aux_fins.detect,
                                                                          path=path,
                                                                          n_cluster=bow_aux_fins.n_cluster,
                                                                          part="fins",
                                                                          show_image=False)
        except:
            row_fins_coordinate,col_fins_coordinate=None,None
            #PFi_coordinates=helper.PFi(row_fins_coordinate,col_fins_coordinate,Bd_coordinates[0],masked_image)
    


    row_mouth_coordinate=helper.image_prediction(name_image=image_1,model=predictor_mouth,
                                          box=bow_aux_mouth.box,
                                          extract_bow=bow_aux_mouth.extract_bow,
                                          detect=bow_aux_mouth.detect,
                                          path=path,
                                          n_cluster=bow_aux_mouth.n_cluster,
                                          part="mouth",
                                          show_image=False)
    if row_mouth_coordinate is not None:
        if(row_mouth_coordinate<TL_coordinates[0]):
            row_mouth_coordinate=TL_coordinates[0]
        Mo_coordinates=helper.Mo(row_mouth_coordinate,TL_coordinates[1],Eh_coordinates[2])
        
        

    ##################### plot dimensions
    final=helper.dim_plotter(image,TL_coordinates,(0,255,0),False)
    final=helper.dim_plotter(final,CPd_coordinates,(0,0,255),False)
    final=helper.dim_plotter(final,CFd_coordinates,(252, 148, 18),False)
    final=helper.dim_plotter(final,Eh_coordinates,(255,0,0),False)
    final=helper.dim_plotter(final,Hd_coordinates,(255,255,0),False)
    final=helper.dim_plotter(final,Ed_coordinates,(255, 105, 145),False)
    final=helper.dim_plotter(final,Bd_coordinates,(255,255,255),False)
    if row_mouth_coordinate is not None:
        final=helper.dim_plotter(final,Mo_coordinates,(255,100,255),False)
    
    #final=helper.dim_plotter(final,PFi_coordinates,(150,100,255),False)

    ##################### rate of pixels
    rate_pixel=helper.rate_pixel(image_1,TL_coordinates)
    CPd_dim=helper.pixes_to_lenght([CPd_coordinates[2],CPd_coordinates[0]],rate_pixel)
    CFd_dim=helper.pixes_to_lenght([CFd_coordinates[2],CFd_coordinates[0]],rate_pixel)
    TL_dim=helper.pixes_to_lenght([TL_coordinates[-1],TL_coordinates[1]],rate_pixel)
    Eh_dim=helper.pixes_to_lenght([Eh_coordinates[2],Eh_coordinates[0]],rate_pixel)
    Hd_dim=helper.pixes_to_lenght([Hd_coordinates[2],Hd_coordinates[0]],rate_pixel)
    Ed_dim=helper.pixes_to_lenght([Ed_coordinates[2],Ed_coordinates[0]],rate_pixel)
    Bd_dim=helper.pixes_to_lenght([Bd_coordinates[2],Bd_coordinates[0]],rate_pixel)
    if row_mouth_coordinate is not None:
        Mo_dim=helper.pixes_to_lenght([Mo_coordinates[2],Mo_coordinates[0]],rate_pixel)

    final,CFs_dim=helper.template_matching(image=final,gray_image=masked_image,template=tail_code_vector)
    #final,CFs_dim=helper.template_matching(image=final,gray_image=masked_image,template=longest_superior_fin)
    #final,CFs_dim=helper.template_matching(image=final,gray_image=masked_image,template=lower_fin)
    #final,CFs_dim=helper.template_matching(image=final,gray_image=masked_image,template=lower_longer_fin)
    CFs_dim=CFs_dim*rate_pixel
    banner=helper.build_banner(image,CPd_dim,TL_dim,Eh_dim,Hd_dim,Ed_dim,Bd_dim,CFs_dim,CFd_dim,Mo_dim)

    #banner=helper.build_banner(image,CPd_dim,TL_dim,Eh_dim,Hd_dim,Bd_dim,CFs_dim)
    combi=np.concatenate((final,banner), axis=1)
    helper.display((combi).astype(np.uint8))
    #plt.imshow(combi)
    co=co+1

        



