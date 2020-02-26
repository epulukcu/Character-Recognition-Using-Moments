import geek as geek
from PIL import Image, ImageDraw, ImageOps
import numpy as np

def main():

    img = Image.open('Image.jpeg')                                             #array formatta değil
    img_gray = img.convert('L')                            # converts the image to grayscale image (array hale getirmek)
    #img_bin = img.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    img_gray.show()
    ONE = 150
    a = np.asarray(img_gray)                                    # from PIL to np array (her pikselin numarası var 0-256)
    a_bin = threshold(a,100,ONE,0)                            # a'yı her biriyle karşılaştır 100 den büyükse 0 küçükse 1
    im = Image.fromarray(a_bin)                                                            # from np array to PIL format
    im.show()

    #a_bin = binary_image(100,100, ONE)                                                           #creates a binary image
    a_bin = np.asarray(im)
    label = blob_coloring_8_connected(a_bin, ONE)
    new_img2 = np2PIL_color(label)
    new_img2.show()



def binary_image(nrow,ncol,Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow,ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10


    for i in range (50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i-20][90-i+1] = 1
        mask_lines[i-20][90-i+2] = 1
        mask_lines[i-20][90-i+3] = 1


    #mask_circle1 = np.abs((x - x0) * 2 + (y - y0) * 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
    #mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    #mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    #mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    #imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    #imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge

def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def threshold(im,T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    #print("nrow, ncol", nrow, ncol)
    im = np.zeros(shape=(nrow,ncol), dtype = int)
    a = np.zeros(shape=max_label, dtype = int)
    a = np.arange(0,max_label, dtype = int)
    color_map = np.zeros(shape = (max_label,3), dtype= np.uint8)
    color_im = np.zeros(shape = (nrow, ncol,3), dtype= np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][1] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][2] = np.random.randint(0,255,1,dtype = np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c   = bim[i][j]
                l   = bim[i][j - 1]
                u   = bim[i - 1 ][j]
                label_u  = im[i -1][j]
                label_l  = im[i][j - 1]
                label_ld = im[i-1][j - 1 ]
                label_rd = im[i+1][j - 1 ]

                im[i][j] = max_label
                if c == ONE:
                    min_label = min( label_u, label_l, label_ld, label_rd)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label  :
                         update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label  :
                            update_array(a, min_label, label_l)

                        if min_label != label_ld and label_ld != max_label  :
                         update_array(a, min_label, label_ld)

                        if min_label != label_rd and label_rd != max_label  :
                            update_array(a, min_label, label_rd)

                else :
                    im[i][j] = max_label
    # final reduction in label array
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    #second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j],0]
                color_im[i][j][1] = color_map[im[i][j],1]
                color_im[i][j][2] = color_map[im[i][j],2]
    #return color_im
    list = []
    counter = -1
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] in list:
                print(im[i][j])
            else:
                list.append(im[i][j])
                counter = counter + 1
    list.remove(10000)                                                                          #list uzunluğu kısalsın

    width, height = 4, counter
    rectangle = [[0 for x in range(width)] for y in range(height)]
    my_k_values=[] #labelların min max değerlerini tuttuğumuz liste (1,2 ve 3ler)
    for label in list:
        min_i = nrow
        min_j = ncol
        max_i = 0
        max_j = 0
        for i in range(nrow):
            for j in range(ncol):
                if label== im[i][j]:
                    if min_i>i:
                        min_i=i
                    if min_j>j:
                        min_j=j
                    if max_i<i:
                        max_i=i
                    if max_j<j:
                        max_j=j

        #Square çizdirme kodu:
        #boy = max_i - min_i
        #en = max_j - min_j
        #max_j = max_j + ( (boy - en)/2 )
        #min_j = min_j - ( (boy - en)/2 )

        #i row, j col
        my_k_values.append([label,min_i,min_j,max_i,max_j])
    #print("my_k_values:")
    print(my_k_values)


    source_img = Image.open('Image.jpeg').convert("RGBA")
    draw = ImageDraw.Draw(source_img)

    for b in my_k_values:
        save_unique_images(source_img, b[2], b[1], b[4], b[3])

        draw.rectangle(((b[2], b[1]),(b[4], b[3])), fill=None, outline='red', width=2)
        source_img.save("output.png", "PNG")

    source_img.show()
    return color_im

#imageleri croplamak
def save_unique_images(img, b2,b1,b4,b3):
    im1 = img.crop((b2, b1, b4, b3))
    im1.show()
    binaryImage(im1)

#croplanan imagei arraye dönüştürmek
#convert the array to binary array. if the rgb value is below a cer
def binaryImage(image):
    img_bin = image.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    img_bin.show()

def update_array(a, label1, label2) :
    index = lab_small = lab_large = 0
    if label1 < label2 :
        lab_small = label1
        lab_large = label2
    else :
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else: #a[index] == lab_small
            break

    return

if __name__=='__main__':
    main()