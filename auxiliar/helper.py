import pandas as pd
import numpy as np
#Matplotlib is a useful plotting library for python 
import matplotlib.pyplot as plt
#conda install opencv
import cv2
import re
import pickle


NUMBER_ONLY = re.compile('[^0-9]')

def image_splitter(grey_image,box):
    """
    Give a grey image, that is, one channel image.
    box: will split the image into BOX*BOX non overlapping
    smaller images.
    Return:
    images_cropped: A list containing each split
    map_cropped: a dictionary, keys: index of the split, values: the origin of the box in the original
    image coordinates.
    """
    map_cropped={}
    images_cropped=[]
    H, W = grey_image.shape
    index_image=0
    for i in np.arange(0,H-box,box):
        for j in np.arange(0,W-box,box):
            crop=grey_image[i:i+box,j:j+box]
            images_cropped.append(crop)
            map_cropped[index_image]=[i,j]
            index_image+=1
    return images_cropped,map_cropped





def save_object(path,object):
    with open(path,"wb") as f:
        pickle.dump(object,f,pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path,"rb") as f:
        file = pickle.load(f) 
    return file


def get_low_value_thresh(img,gray=False,box=5):
    m,n=int(img.shape[0]/2),int(img.shape[1]/2)
    avg_center_picture=[]
    #box=5
    i,j=m-box,n-box
    while i<=m+box:
        while j<=n+box:
            avg_center_picture.append(img[i][j])
            #print(i,j)
            j+=1
        i+=1
    # multipled values found in the cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if(gray):
        minimun_intesity=int(np.mean(avg_center_picture))
    else:
        minimun_intesity=int(np.mean(np.mean(avg_center_picture,axis=0)*(0.299,0.587,0.114)))
    return minimun_intesity



#def give_cluster(image_pro):
    '''
    takes a image in gray scale with dim (640,360)
    '''
#    image=image_pro.reshape(1,image_pro.shape[0]*image_pro.shape[1])
#    return k_means_object.predict(image)


def display(img):
    # Show image
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    plt.axis('on')
    plt.show()
    
plt.rcParams['image.cmap'] = 'gray'   

def auto_canny(image, sigma=0.20):
    image = image.astype(np.uint8)
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def image_prepare(image,guasian_kernel_dim=(3,3),gray_step=True,blur_tec="gausian_smothing",resize=(640,360),resize_bol=False):
    gray_image=image
    if(gray_step):
        minimun_intesity=get_low_value_thresh(image)
        ret,thresh1 = cv2.threshold(image,minimun_intesity,255,cv2.THRESH_BINARY)
        gray_image = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)

    if(blur_tec=="gausian_smothing"):
        #print('gausian_smothing')
        blur = cv2.GaussianBlur(gray_image, guasian_kernel_dim, 0)

    elif(blur_tec=="bilateral_smothing"):
        #print('bilateral_smothing')
        blur = cv2.bilateralFilter(src=gray_image,d=100,sigmaColor=75,sigmaSpace=75)

    if(resize_bol):
        resized = cv2.resize(blur, resize, interpolation = cv2.INTER_AREA)
    else: 
        resized = blur
    return resized



def position_refineir(grey_image,mov,initial_row,initial_col,max_steps=10):
    move=True
    step=0
    if(mov=="down"):
        val=grey_image[initial_row,initial_col]
        while move:
            val_down=grey_image[initial_row+step,initial_col]
            if(val_down<=val):
                step+=1
            else:
                initial_row=initial_row+step
                move=False

            if(step>max_steps):
                move=False

    elif(mov=="up"):
        val=grey_image[initial_row,initial_col]
        while move:
            val_up=grey_image[initial_row-step,initial_col]
            #print(val,val_up,step,max_steps)
            if(val_up<=val):
                step+=1
            else:
                initial_row=initial_row-step
                move=False

            if(step>max_steps):
                move=False
    #my right            
    elif(mov=="right"):
        val=grey_image[initial_row,initial_col]
        while move:
            val_right=grey_image[initial_row,initial_col+step]
            #print(val,val_right,step,max_steps)
            if(val_right<=val):
                step+=1
            else:
                initial_col=initial_col+step
                #print('fin',initial_col)
                move=False
            if(step>max_steps):
                move=False
    elif(mov=="left"):
        val=grey_image[initial_row,initial_col]
        while move:
            val_left=grey_image[initial_row,initial_col-step]
            if(val_left<=val):
                step+=1
            else:
                initial_col=initial_col-step
                move=False
            if(step>max_steps):
                move=False

    return initial_row,initial_col


def TL(grey_image):
    grey_image=image_prepare(grey_image,guasian_kernel_dim=(11,11),gray_step=False)
    grey_image=auto_canny(grey_image)
    #grey_image=grey_image/255
    #display(grey_image)
    sum_per_row=np.sum(grey_image,axis=1)
    max_sum_row=max(sum_per_row)
    row_sum_index=[index for index,row in enumerate(sum_per_row) if max_sum_row==row]
    row_pos=row_sum_index[0]

    sum_per_col=np.sum(grey_image,axis=0)
    col_sum_index=[index for index,col in enumerate(sum_per_col) if col>3]
    #print(len(col_sum_index))
    col_pos=col_sum_index[0]
    rear_col_pos=col_sum_index[-1]
    #display(grey_image)
    row_pos,col_pos=position_refineir(grey_image,"up",row_pos,col_pos)
    row_pos,col_pos=position_refineir(grey_image,"down",row_pos,col_pos)
    row_pos,col_pos=position_refineir(grey_image,"right",row_pos,col_pos,max_steps=100)

    return [row_pos,col_pos,row_pos,rear_col_pos]

def key_with_maxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def key_with_minval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(min(v))]


def tail_detection(contours_mask,dim_to_calculate,col_position_CPd=None):
    try:
        tail_code_vector=load_object('pickle_objects/tail_code_vector.file')

        detected_tail=template_matching(contours_mask,tail_code_vector,nearest_col=False,mode="remove",return_difference=True,tail=True)
        #display(detected_tail)

        if(dim_to_calculate=="CPd"):
            detected_tail=detected_tail[:,0:detected_tail.shape[1]-int((detected_tail.shape[1]*20)/100)]
        # detect last columns where the last no summation is
        sum_per_col=np.sum(detected_tail,axis=0)
        col_sum_index=[index for index,col in enumerate(sum_per_col) if col!=0]

        distancia={}
        for each_col in col_sum_index:
            row_values=detected_tail[:,each_col]
            non_zero_row=[index for index,row in enumerate(row_values) if row!=0]
            d=non_zero_row[-1]-non_zero_row[0]
            if(d>20):
                distancia[each_col]=non_zero_row[-1]-non_zero_row[0]

        if(dim_to_calculate=="CPd"):
            distance_op=key_with_minval(distancia)


        elif(dim_to_calculate=="CFd"):
            distancia_ajust={k:v for k,v in distancia.items() if k>col_position_CPd}
            distance_op=key_with_maxval(distancia_ajust)

        row_values=detected_tail[:,distance_op]
        index_per_row_no_zero=[index for index,row in enumerate(row_values) if row!=0]
        #print(index_per_row_no_zero[0],index_per_row_no_zero[-1])

        row_pos=index_per_row_no_zero[0]
        col_pos=distance_op

        rear_row_pos=index_per_row_no_zero[-1]
        rear_col_pos=distance_op
    except:
        row_pos,col_pos,rear_row_pos,rear_col_pos=0,0,0,0

    
    return [row_pos,col_pos,rear_row_pos,rear_col_pos]



def dim_plotter(image,two_pairs_coordinates,color,show_imaga=True):
    
    image_copy=image.copy()
    p1_row, p1_col =two_pairs_coordinates[0],two_pairs_coordinates[1]
    p2_row, p2_col =two_pairs_coordinates[2],two_pairs_coordinates[3]
    lineThickness = 1
    cv2.line(image_copy, (p1_col,p1_row), (p2_col,p2_row), color, lineThickness)

    cv2.circle(img=image_copy, center=(p1_col,p1_row),
                  radius=6, color=color,
                  thickness=1, lineType=10, shift=0)
    cv2.circle(img=image_copy, center=(p2_col,p2_row),
                  radius=6, color=color,
                  thickness=1, lineType=10, shift=0)

    #cv2.line(image_copy, (p1_col,p1_row), (p2_col,p2_row-10), (255,0,0), lineThickness)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #leng_line=int(col_pos_tail/2)
    #cv2.circle(img=image_copy, center=(leng_line+10,horital_pos_line),
    #              radius=12, color=(255,255,255),
    #              thickness=-1, lineType=10, shift=0)
    #cv2.putText(image_copy,dim_text,(leng_line,horital_pos_line+6), font, 0.4,(255,0,0),1,cv2.LINE_AA)
    
    
    #cv2.line(image_copy, (col_pos, 0), (col_pos, row_pos), (255,0,0), lineThickness)
    #cv2.line(image_copy, (col_pos_tail, 0), (col_pos_tail, row_pos_tail), (255,0,0), lineThickness)

    if(show_imaga):
        plt.figure(figsize = (12,12))

        plt.imshow(image_copy)
        plt.axis('on')
        plt.show()
    else:
        return image_copy

    
def check_name_for_SL_TL(string):
    string=string.lower()
    for_sl=string.split('sl')
    for_tl=string.split('tl')
    try:
        if len(for_sl)>1 and len(for_tl)>1:
            checker=[True,True]
        elif len(for_sl)>1 and len(for_tl)==1:
            checker=[True,False]
        elif len(for_sl)==1 and len(for_tl)>1:
            checker=[False,True]
        else:
            checker=[False,False]
    except:
        checker=[False,False]

    return checker

def get_dim_from_name(string):
    string=str.strip(string.lower())
    string=string.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    checker=check_name_for_SL_TL(string)
    NUMBER_ONLY = re.compile('[^0-9.]')
    numbers_only = str.strip(NUMBER_ONLY.sub(" ",string)).split()
    SL_dim,TL_dim=0,0
    dic_dim={}
    if(checker[1]):
        dic_dim['TL_dim']=np.max(np.array(numbers_only).astype(float))

    elif(checker[0]==True and checker[1]==False):
        dic_dim['SL_dim']=np.max(np.array(numbers_only).astype(float))
    else:
        dic_dim['TL_dim']=0
    return dic_dim

def get_contour_mask(image_prepared):
    image_copy=image_prepared.copy()
    result=cv2.findContours(image_copy, 1, 2)
    contours= result[1]
    #print(contours[0])
    contours_mask = np.zeros( (image_copy.shape[0],image_copy.shape[1]) ) 
    cv2.fillPoly(contours_mask, pts =contours, color=(255,255,255))
    contours_mask=contours_mask/255
    #helper.display(contours_mask)
    return contours_mask

def multiple_image_contour(image,contours_mask):
    out = image.copy()
    out[:,:,0] =contours_mask*image[:,:,0]
    out[:,:,1] =contours_mask*image[:,:,1]
    out[:,:,2] =contours_mask*image[:,:,2]
    return out

def gray_equalized(masked_image,greay_step=True):
    gray_image=masked_image
    if(greay_step):
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    eq_image=cv2.equalizeHist(gray_image)
    return eq_image



def template_matching(gray_image,template,nearest_col=True,mode="draw",image=None,lower_fin=False,return_difference=False,tail=False):
    """
    image in gray scale. Masked image
    template, code vector of part.
    """
    img=gray_image.copy()
    if(lower_fin):
        gw,gh=gray_image.shape
        gray_image=gray_image[0:gw,0:int(gh/2)]
    
    gray_image = gray_image.astype(np.uint8)

    #display(gray_image)
    template=template.astype(np.uint8)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_image,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.99
    loc = np.where( res >= threshold)
    #print(loc) 
    #print('len(loc)',len(loc[0]))
    while len(loc[0])==0:
        #print(threshold)
        threshold=threshold-0.01
        loc = np.where( res >= threshold)

    #loc_list_pairs=[]
    #for i in np.arange(0,len(loc[0]),1):
    #    loc_list_pairs.append((loc[1][i],loc[0][i])) 
    #print(loc)
    min_row=min(loc[0])
    max_row=min(loc[0])+template.shape[0]

    min_col=min(loc[1])
    max_col=max(loc[1])+template.shape[1]
    #print(min_row,max_row)
    #print(min_col,max_col)
    #start_point = loc_list_pairs[0] 
    #end_point = loc_list_pairs[-1]
    #print(start_point,end_point,loc_list_pairs) 
    #thickness = -1
    #contours_mask = np.zeros( (gray_image.shape[0],gray_image.shape[1]),dtype=np.uint8) 
    #cv2.fillPoly(contours_mask, a3, 255 )

    #cv2.rectangle(gray_image, (min_col,min_row), (max_col,max_row), 255, thickness)
    #ontours_mask=contours_mask
    #a3 = np.array( [[[10,10],[100,10],[100,100],[10,100]]], dtype=np.int32 )
    #im = np.zeros([240,320],dtype=np.uint8)
    #cv2.fillPoly( im, a3, 255 )

    #print(loc)
    
    #contours_mask=contours_mask/255
    #isolated=contours_mask*gray_image
    focus=gray_image[min_row:max_row,min_col:max_col]

    if(mode=="draw"):
        dic_rec_col={}
        dic_rec_coor={}
        for index,pt in enumerate(zip(*loc[::-1])):
            dic_rec_col[index]=pt[0]
            dic_rec_coor[index]=pt
            #cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,100,100), thickness=1,lineType=0)

        min_key=key_with_minval(dic_rec_col)
        pt=dic_rec_coor[min_key]
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,100,100), thickness=1,lineType=0)

        sum_focus=np.sum(focus)
        return image,sum_focus
    else:
        temp_dic={}
        temp_col=[]

        for index,pt in enumerate(zip(*loc[::-1])):
            temp_dic[index]=pt
            temp_col.append(pt[0])
            #cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,100,100), 1)
        if(nearest_col):
            min_col_coor_rec=min(temp_col)
            #print(min_col_coor_rec)
            filtred_dic=[k for k, v in temp_dic.items() if v[0]==min_col_coor_rec][0]
        else:
            max_col_coor_rec=max(temp_col)
            #print(min_col_coor_rec)
            filtred_dic=[k for k, v in temp_dic.items() if v[0]==max_col_coor_rec][0]
        pair_recta=temp_dic.get(filtred_dic)
        #print(pair_recta)

        min_row=pair_recta[1]
        max_row=pair_recta[1]+template.shape[0]

        min_col=pair_recta[0]
        max_col=pair_recta[0]+template.shape[1]

        mask=np.zeros((img.shape[0],img.shape[1]))
        #print(mask.shape[0],mask.shape[1])
        #display(min_row,max_row,template.shape[0])
        if(tail==True):
            mask[min_row:max_row,min_col:]=img[min_row:max_row,min_col:]
        else:
            mask[min_row:max_row,min_col:max_col]=img[min_row:max_row,min_col:max_col]
        if(return_difference):
            return mask
        else:
            diff_image=img-mask
            return diff_image


        #return gray_image[min_row:max_row,min_col:max_col]

def eye_detection(image,contours_mask,masked_image,dim_to_calculate,confusion=True,minRadius=6,Hd_coor=None,show_image=False):

    ksize=5
    sigma=0.5
    theta=1*np.pi/5
    lamda=1*np.pi/5
    gamma=1.0
    phi=0
    kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)

    eq=gray_equalized(masked_image,greay_step=False)

    #plt.imshow(kernel)
    fimg=cv2.filter2D(eq,cv2.CV_8UC3,kernel)


    masked_image=gray_equalized(masked_image,greay_step=False)

    #masked_image=image_prepare(guasian_kernel_dim=(3,3),image=masked_image,gray_step=False)
    #display(masked_image)
    #print(masked_image)
    H, W = masked_image.shape
    cimg=masked_image.copy()

    image_cir=image.copy()
    try:
        #display(fimg)
        circles_1 = cv2.HoughCircles(fimg,cv2.HOUGH_GRADIENT,1,20,param1=45,param2=30,minRadius=minRadius,maxRadius=23)
        circles_1 = np.uint16(np.around(circles_1))
        num_cir_1=len(circles_1[0,:])
        #print(circles_1)
        if(np.linalg.norm(circles_1)==0):
            num_cir_1=500
        
    except:
        num_cir_1=500

    try:
        circles_2 = cv2.HoughCircles(masked_image,cv2.HOUGH_GRADIENT,1,20,param1=45,param2=30,minRadius=minRadius,maxRadius=23)
        circles_2 = np.uint16(np.around(circles_2))
        num_cir_2=len(circles_2[0,:])
        if(np.linalg.norm(circles_2)==0):
            num_cir_2=500
    except:
        num_cir_2=500

    
    #print(len(circles[0,:]))
    #print(num_cir_1,num_cir_2)
    if confusion:
        if(num_cir_1>=num_cir_2):
            circles=circles_1
        else:
            circles=circles_2
    else:
        if(num_cir_1<=num_cir_2):
            circles=circles_1
        else:
            circles=circles_2



    #circles=circles_2
    #print(circles)
    dic_circules={}

    for index,i in enumerate(circles[0,:]):
        # draw the outer circle
        dic_circules[index]=i[0]
        cv2.circle(image_cir,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(image_cir,(i[0],i[1]),2,(0,0,255),3)
    #center_row=circles[0][0][1]
    if(show_image):
        display(image_cir)
    minimal_col_circle=key_with_minval(dic_circules)
    center_col=circles[0][minimal_col_circle][0]
    radius=circles[0][minimal_col_circle][2]
    center_row=circles[0][minimal_col_circle][1]

    dist={}
    for index,i in enumerate(circles[0,:]):
        # draw the outer circle
        x,y=i[0],i[1]
        #print(x,center_col)
        #print(x-center_col)
        dx=int(x)-int(center_col)
        dy=int(y)-int(center_row)
        d=(dx**2+dy**2)**(0.5)
        #print(dx**2,dy**2)
        #print(d,W*0.2)
        if(d!=0 and d<W*0.1/1.5):
            #print(index)
            dist[index]=d
    try:
        minimal_d_circle=key_with_minval(dist)
        center_col_2=circles[0][minimal_d_circle][0]
        radius_2=circles[0][minimal_d_circle][2]
        center_row_2=circles[0][minimal_d_circle][1]

        if(center_row_2<center_row):
            center_col=circles[0][minimal_d_circle][0]
            radius=circles[0][minimal_d_circle][2]
            center_row=circles[0][minimal_d_circle][1]
    except:
        pass

    #print(center_row,center_col)

    focus_eye=masked_image[:,center_col]
    row_row=[index for index,each_row in enumerate(focus_eye) if each_row!=0]
    col_pos=center_col
    rear_col_pos=center_col


    try:
        if(dim_to_calculate=="Hd"):    
            row_pos=row_row[0]
            rear_row_pos=center_row
            #print(masked_image[row_pos,col_pos])
            #display(contours_mask)
            #if(masked_image[row_pos,col_pos]=!0)
            #row_pos,col_pos=position_refineir(contours_mask,"down",row_pos,col_pos,max_steps=100)
            #print("Hd center_row",center_row)
        elif(dim_to_calculate=="Eh"):
            row_pos=center_row
            rear_row_pos=row_row[-1]
            #display(contours_mask)
            rear_row_pos,rear_col_pos=position_refineir(contours_mask,"up",rear_row_pos,rear_col_pos,max_steps=100)
            #print("Eh center_row",center_row)
        elif(dim_to_calculate=="Ed"):
            row_pos=Hd_coor[2]-radius
            rear_row_pos=Hd_coor[2]+radius
            col_pos=center_col+radius*2
            rear_col_pos=center_col+radius*2
    except:
        row_pos,col_pos,rear_row_pos,rear_col_pos=0,0,0,0
    
    #masked_image=auto_canny(masked_image)
    #print(rear_row_pos,rear_col_pos)
    
    #print(center_row,center_col,radius)
    return [row_pos,col_pos,rear_row_pos,rear_col_pos]

def Bd2_prepare(masked_image):
    longest_superior_fin=load_object('pickle_objects/longest_superior_fin.file')
    lower_fin=load_object('pickle_objects/lower_fin.file')

    #lower_fin=image_prepare(guasian_kernel_dim=(11,11),image=lower_fin,gray_step=False)
    #lower_fin=auto_canny(lower_fin)

    lower_longer_fin=load_object('pickle_objects/lower_longer_fin.file')

    no_fin_superio=template_matching(masked_image,longest_superior_fin,mode="remove",image=None)
    
    
    no_fin_superio=image_prepare(guasian_kernel_dim=(11,11),image=no_fin_superio,gray_step=False)

    edges_no_fin_superio=auto_canny(no_fin_superio)

    no_fin_inf=template_matching(edges_no_fin_superio,lower_fin,mode="remove",image=None,lower_fin=True)
    
    
    no_fin_inf_long=template_matching(no_fin_inf,lower_longer_fin,nearest_col=False,mode="remove",image=None)

    #display(no_fin_inf_long)

    return no_fin_inf_long

def fin_prepare(image):
    longest_superior_fin=load_object('pickle_objects/longest_superior_fin.file')
    lower_fin=load_object('pickle_objects/lower_fin.file')
    lower_longer_fin=load_object('pickle_objects/lower_longer_fin.file')

    no_fin_superio=template_matching(image,longest_superior_fin,mode="remove",image=None)


    no_fin_inf=template_matching(no_fin_superio,lower_fin,mode="remove",image=None,lower_fin=True)
    #display(no_fin_inf)
    
    no_fin_inf_long=template_matching(no_fin_inf,lower_longer_fin,nearest_col=False,mode="remove",image=None)
    return no_fin_inf_long

    


def Bd2(masked_image,prepared_image,min_col_searh,max_col_searh):
    try:
        mask=np.zeros((prepared_image.shape[0],prepared_image.shape[1]))
        mask[:,min_col_searh:max_col_searh]=prepared_image[:,min_col_searh:max_col_searh]
        #display(mask)
        prepared_image=mask
        sum_per_col=np.sum(prepared_image,axis=0)
        max_sum_col=max(sum_per_col)
        col_max=[index for index,each_col in enumerate(sum_per_col) if each_col==max_sum_col]
        col_pos=col_max[0]
        rows=[index for index,each_row in enumerate(prepared_image[:,col_pos]) if each_row!=0]

        row_pos=rows[0]
        rear_row_pos=rows[-1]
        rear_col_pos=col_pos
        #row_pos,col_pos=position_refineir(grey_image,"up",row_pos,col_pos)
        #row_pos,col_pos=position_refineir(grey_image,"down",row_pos,col_pos)
        #row_pos,col_pos=position_refineir(grey_image,"right",row_pos,col_pos)

        
        masked_image=image_prepare(guasian_kernel_dim=(3,3),image=masked_image,gray_step=False)

        minimun_intesity=get_low_value_thresh(masked_image,gray=True,box=100)
        ret,masked_image = cv2.threshold(masked_image,minimun_intesity,255,cv2.THRESH_BINARY)

        #masked_image=auto_canny(masked_image)
        #print(rear_row_pos,rear_col_pos)
        rear_row_pos,rear_col_pos=position_refineir(masked_image,"up",rear_row_pos,rear_col_pos,max_steps=200)
        #print(rear_row_pos,rear_col_pos)
        #masked_image=auto_canny(masked_image)
        #display(masked_image)
    except:
        row_pos,col_pos,rear_row_pos,rear_col_pos=0,0,0,0

    return [row_pos,col_pos,rear_row_pos,rear_col_pos]


def rate_pixel(image_name,TL_coordinates):
    try:
        dim=get_dim_from_name(image_name).get('TL_dim')
        ref_distance_pix=int(TL_coordinates[-1])-int(TL_coordinates[1])
        rate_pixel=dim/ref_distance_pix
    except:
        dim=1
        rate_pixel=1
    
    return rate_pixel

def pixes_to_lenght(coordinates,rate_pixel):
    ref_distance_pix=abs(int(coordinates[1])-int(coordinates[0]))
    return rate_pixel*ref_distance_pix


def build_banner(image,CPd_dim,TL_dim,Eh_dim,Hd_dim,Ed_dim,Bd_dim,CFs_dim,CFd_dim,Mo_dim):
    banner = np.zeros((image.shape[0],80,image.shape[2]))
    #helper.display(banner)
    font = cv2.FONT_HERSHEY_SIMPLEX
    w=50
    h=20
    inital_col=20
    inital_row=21
    padding_left=3
    padding_up=15
    box_separation=4
    inital_row_copy=inital_row

    inital_row=1*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h), (0,0,255), -1)
    cv2.putText(banner,"CPd "+str(round(CPd_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(255,255,255),1,cv2.LINE_AA)

    inital_row=2*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h), (0,255,0), -1)
    cv2.putText(banner,"Tl "+str(round(TL_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(0,0,0),1,cv2.LINE_AA)

    inital_row=3*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h), (255,0,0), -1)
    cv2.putText(banner,"Eh "+str(round(Eh_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(255,255,255),1,cv2.LINE_AA)

    inital_row=4*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h), (255,255,0), -1)
    cv2.putText(banner,"Hd "+str(round(Hd_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(0,0,0),1,cv2.LINE_AA)

    inital_row=5*(inital_row_copy+box_separation)
    cv2.rectangle(banner,(inital_col,inital_row), (inital_col+w,inital_row+h), (255, 105, 145), -1)
    cv2.putText(banner,"Ed "+str(round(Ed_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(0,0,0),1,cv2.LINE_AA)

    inital_row=6*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h), (255,255,255), -1)
    cv2.putText(banner,"Bd "+str(round(Bd_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(0,0,0),1,cv2.LINE_AA)

    inital_row=7*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h), (0,100,100), -1)
    cv2.putText(banner,"CFs "+str(round(CFs_dim/1000, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(255,255,255),1,cv2.LINE_AA)

    inital_row=8*(inital_row_copy+box_separation)
    cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h),(252, 148, 18), -1)
    cv2.putText(banner,"CFd "+str(round(CFd_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(255,255,255),1,cv2.LINE_AA)

    if Mo_dim is not None:
        inital_row=9*(inital_row_copy+box_separation)
        cv2.rectangle(banner, (inital_col,inital_row), (inital_col+w,inital_row+h),(255,100,255), -1)
        cv2.putText(banner,"Mo "+str(round(Mo_dim, 1)),(inital_col+padding_left,inital_row+padding_up), font, 0.32,(255,255,255),1,cv2.LINE_AA)

    #display(banner)
    return banner




#box=50
def data_set_creator(df_X,labes,box,extract_bow,detect,path,n_cluster,show_image=False):
    def bow_features(im):
        return extract_bow.compute(im, detect.detect(im))
    df_X_holder=[]
    global_indexes=[]
    font = cv2.FONT_HERSHEY_SIMPLEX
    i=0
    y_temp=[]

    #longest_superior_fin=load_object('pickle_objects/longest_superior_fin.file')
    #lower_fin=load_object('pickle_objects/lower_fin.file')
    #lower_longer_fin=load_object('pickle_objects/lower_longer_fin.file')

    for each in df_X.name:
        name_image=each
        global_index=df_X.loc[df_X.name==name_image].index[0]
        
        image = cv2.imread(path+name_image)

        masked_image=image_prepare(image,guasian_kernel_dim=(11,11))
        masked_image=gray_equalized(masked_image,greay_step=False)
        #pre_image=image_prepare(image,guasian_kernel_dim=(3,3))
        #contours_mask=get_contour_mask(pre_image)
        #masked_image=multiple_image_contour(image,contours_mask)
        #masked_image=gray_equalized(masked_image)

        

        #no_fin_superio=template_matching(masked_image,longest_superior_fin,mode="remove",image=None)
        
        #no_fin_inf=template_matching(no_fin_superio,lower_fin,mode="remove",image=None,lower_fin=True)
        
        #no_fin_inf_long=template_matching(no_fin_inf,lower_longer_fin,nearest_col=False,mode="remove",image=None)
        
        
        grid_images,map_grid_images=image_splitter(masked_image,box)
        splited_image=image.copy()
        X_temp=[]
        
        list_labels_boca=labes.get(global_index)[0]
        list_labels_ojos=labes.get(global_index)[1]
        list_labels_aletas=labes.get(global_index)[2]
        for rectan in np.arange(0,len(map_grid_images),1):
            pt=map_grid_images.get(rectan)
            
            cv2.putText(splited_image,str(rectan),(pt[1]+int(box/2),pt[0]+int(box/2)),
                        font, 0.32,(255,0,0),1,cv2.LINE_AA)
            
            X_temp.append(bow_features(grid_images[rectan]))
            global_indexes.append(global_index)
            if(rectan in list_labels_boca):
                cv2.rectangle(splited_image, (pt[1],pt[0]), (pt[1] + box, pt[0] + box),
                              (0,100,100), thickness=5,lineType=0)
                y_temp.append('mouth')
            elif(rectan in list_labels_ojos):
                cv2.rectangle(splited_image, (pt[1],pt[0]), (pt[1] + box, pt[0] + box),
                              (100,0,100), thickness=5,lineType=0)
                y_temp.append('eye')
            elif(rectan in list_labels_aletas):
                cv2.rectangle(splited_image, (pt[1],pt[0]), (pt[1] + box, pt[0] + box),
                              (100,100,0), thickness=5,lineType=0)
                y_temp.append('fins')
            else:
                cv2.rectangle(splited_image, (pt[1],pt[0]), (pt[1] + box, pt[0] + box),
                              (255,255,255), thickness=1,lineType=0)
                y_temp.append('a_noise')
            
        X_norm=[]
        for each in X_temp:
            if each is None:
                each=np.zeros((1,n_cluster))
            X_norm.append(each)

        extracted_descriptors=pd.DataFrame(np.concatenate(X_norm))
        extracted_descriptors['global_index']=global_index
        extracted_descriptors['local_index']=extracted_descriptors.index
        if(i==0):
            df_X_holder=extracted_descriptors
        else:
            df_X_holder=pd.concat([df_X_holder, extracted_descriptors], ignore_index=True, sort=False)
        if(show_image):
            print('super',global_index)
            print('global_index',global_index)
            print('name_image',name_image)
            display(splited_image)
            print('contador',i)
        i=i+1
        
    #X_train=pd.DataFrame(X_train,index=i_train.index)
    df_y=pd.DataFrame(y_temp,columns=['classes'])
    return df_X_holder,df_y


def Mo(row_mouth_coordinate,TL_coordinate_front_col,Eh_coordinates_rear_row):
    row_pos=row_mouth_coordinate
    col_pos=TL_coordinate_front_col
    rear_row_pos=Eh_coordinates_rear_row
    rear_col_pos=TL_coordinate_front_col
    return [row_pos,col_pos,rear_row_pos,rear_col_pos]

def PFi(row_fins_coordinate,col_fins_coordinate,Bd_coordinates_front_row,masked_image):

    row_pos=Bd_coordinates_front_row
    col_pos=col_fins_coordinate+25
    rear_row_pos=row_fins_coordinate+25
    rear_col_pos=col_fins_coordinate+25

    row_pos,col_pos=position_refineir(masked_image,"down",row_pos,col_pos,max_steps=200)

    return [row_pos,col_pos,rear_row_pos,rear_col_pos]


def image_prediction(name_image,model,box,extract_bow,detect,path,n_cluster,part,show_image=True):
    def bow_features(im):
        return extract_bow.compute(im, detect.detect(im))
    font = cv2.FONT_HERSHEY_SIMPLEX
    i=0
    
    image = cv2.imread(path+name_image)



    masked_image=image_prepare(image,guasian_kernel_dim=(11,11))
    masked_image=gray_equalized(masked_image,greay_step=False)
    #display(masked_image)
    if(part=="fins"):
        masked_image=fin_prepare(masked_image).astype(np.uint8)

    #masked_image=fin_prepare(masked_image).astype(np.uint8)

    #display(masked_image)
    #pre_image=image_prepare(image,guasian_kernel_dim=(3,3))
    #contours_mask=get_contour_mask(pre_image)
    #masked_image=multiple_image_contour(image,contours_mask)
    #masked_image=gray_equalized(masked_image)
    
    #display(masked_image)
    grid_images,map_grid_images=image_splitter(masked_image,box)
    splited_image=image.copy()
    rectan_score={}
    for rectan in np.arange(0,len(map_grid_images),1):
        
        pt=map_grid_images.get(rectan)
        if(part=="mouth"):
            only_f_s_col=[keys for keys,values in map_grid_images.items() if values[1]==0 or values[1]==50]
        else:
            only_f_s_col=[keys for keys,values in map_grid_images.items()]

        if(rectan in only_f_s_col):
            try:
                X=bow_features(grid_images[rectan])
                X=pd.DataFrame(X,columns=[ i for i in np.arange(0,n_cluster,1)])
                y=model.make_predictions(X).Predicted_Target[0]
                #predicted_prob=predicted_prob.append(y)
                if(y>0.7):
                    rectan_score[rectan]=y
                    cv2.rectangle(splited_image, (pt[1],pt[0]), (pt[1] + box, pt[0] + box),(200,0,200), thickness=2,lineType=0)
                    cv2.putText(splited_image,str(round(y,2)),(pt[1]+int(box/2),pt[0]+int(box/2)),font, 0.32,(255,255,255),1,cv2.LINE_AA)
            except:
                y=0.0

            #cv2.putText(splited_image,str(round(y,2)),(pt[1]+int(box/2),pt[0]+int(box/2)),
            #            font, 0.32,(255,255,255),1,cv2.LINE_AA)

            #cv2.rectangle(splited_image, (pt[1],pt[0]), (pt[1] + box, pt[0] + box),(255,255,255), thickness=1,lineType=0)
        

    if(show_image):
        print('name_image',name_image)
        display(splited_image)
        print('contador',i)
    i=i+1
    

    try:
        max_rectan_by_score=key_with_maxval(rectan_score)
        if(part=="mouth"):
            return map_grid_images.get(max_rectan_by_score)[0]
        else:
            coor=map_grid_images.get(max_rectan_by_score)
            return coor[0],coor[1],splited_image

    except:
        return None
    