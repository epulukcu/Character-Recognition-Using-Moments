"""
Name - Surname: Elif PULUKÇU
Date: 02.03.2020
Program Explanation: This program recognizes character in a given image. To do that, takes the file as an input and convert it to a grey image,
highlights the figures with the help of a red rectangle that located character. Crops the rectangle and resize it
to make a square. Then, program converts the image to a binary image with zeros and ones. Calculates moments for the image. Finally, compares
test images values with the training database.
"""


import tkinter
import tk as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import math # for moment calculations
import tkinter.filedialog
import tkinter.ttk as ttk
import tkinter.filedialog
from tkinter import messagebox
from tkinter.filedialog import *
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import csv

root = Tk()
my_k_values = []

xscrollbar = Scrollbar(root, orient=HORIZONTAL)
xscrollbar.pack(side=BOTTOM, fill=X)
yscrollbar = Scrollbar(root)
yscrollbar.pack(side=RIGHT, fill=Y)
text = Text(root, wrap=NONE,
                xscrollcommand=xscrollbar.set,
                yscrollcommand=yscrollbar.set)

# Creates a list containing 10 lists, each of 7 items, all set to 0
#for HU
#w, h = 7, 30; # for 3 x 10 = 30 numbers, in other word 3 sets(0-9) of number's HU moments
#trainingDataHU = [[0 for x in range(w)] for y in range(h)]
#testDataHU = [[0 for x in range(w)] for y in range(h)]

# Creates a list containing 10 lists, each of 10 items, all set to 0
#for R
#r, c = 10, 30; # for 3 x 10 = 30 numbers, in other word 3 sets(0-9) of number's R moments
#trainingDataR = [[0 for x in range(r)] for y in range(c)]
#testDataR = [[0 for x in range(r)] for y in range(c)]

def main():

    #GRAPHICAL USER INTERFACE"
    global root
    global source_img
    global im1
    global im
    #global numberCounterofAnImage
    #numberCounterofAnImage = 0
    global imageCounter
    imageCounter = 0
    global testFlag
    testFlag = 0

    root.geometry("800x600")
    root.title("CHARACTER RECOGNITION USING MOMENTS - ELİF PULUKÇU")

    root.menubar = tk.Menu(root, font="TkMenuFont")
    root.configure(menu=root.menubar)

    # GUI TAB DEFINITIONS #
    # TAB 1:
    tabControl = ttk.Notebook(root)
    root.tab1 = ttk.Frame(tabControl)
    tabControl.add(root.tab1, text=" TRAINING ")
    # TAB 2:
    root.tab2 = ttk.Frame(tabControl)
    tabControl.add(root.tab2, text=" TEST & RESULTS ")
    tabControl.pack(expand=1, fill="both")

    #TAB1: TRAINING PART GUI ELEMENTS #

    #labelFrame = LabelFrame(root.tab1, text="Select Your Training Image:")
    #labelFrame.grid(column=0, row=0, padx=8, pady=4)

    #label = Label(labelFrame, text="")
    #label.grid(column=0, row=0, sticky='W')

    #Browse & Open Button
    button1 = Button(root.tab1, text="Open Image")
    button1.grid(column=0, row=0, sticky='W')
    button1.place(relx=0.007, rely=0.1, height=54, width=92)
    button1.configure(background="#d9d9d9",text="Browse & Open", width=87, command = lambda: openImage())

    #ALL IN ONE
    button5 = Button(root.tab1, text="START")
    button5.grid(column=0, row=0, sticky='W')
    button5.place(relx=0.143, rely=0.1, height=54, width=92)
    button5.configure(background="#FF0000",text="START", width=87, command= lambda: imageProcess())

    #HU Moment Button
    button6 = Button(root.tab1, text="HU Moment")
    button6.grid(column=0, row=0, sticky='W')
    button6.place(relx=0.286, rely=0.1, height=54, width=92)
    button6.configure(background="#d9d9d9",text="HU Moment", width=87, command= lambda: huMoment(a_bin))

    #R Moment Button
    button7 = Button(root.tab1, text="R Moment")
    button7.grid(column=0, row=0, sticky='W')
    button7.place(relx=0.429, rely=0.1, height=54, width=92)
    button7.configure(background="#d9d9d9",text="R Moment", width=87, command= lambda: rMoment(h_1, h_2, h_3, h_4, h_5, h_6, h_7))

    #Zernike Moment Button
    button8 = Button(root.tab1, text="Zernike Moment")
    button8.grid(column=0, row=0, sticky='W')
    button8.place(relx=0.572, rely=0.1, height=54, width=100)
    button8.configure(background="#d9d9d9",text="Zernike Moment", width=100)

    #E2E Training Button
    button9 = Button(root.tab1, text="Save to \nTraining Database")
    button9.grid(column=0, row=0, sticky='W')
    button9.place(relx=0.72, rely=0.1, height=54, width=120)
    button9.configure(background="#d9d9d9",text="Save to \nTraining Database", width=120, command= lambda: SaveToTrainingDatabase())

    #Text box components such as vertical and horizontal scroll bars
    text.pack()
    text.insert(tk.END, "REPORT OF CHARACTER RECOGNITION USING MOMENTS - ELİF PULUKÇU")
    xscrollbar.config(command=text.xview)
    yscrollbar.config(command=text.yview)

    #TAB2: TESTING PART & RESULT PART BUTTONS#
    #Browse & Open Button
    button20 = Button(root.tab2, text="Open Image")
    button20.grid(column=0, row=0, sticky='W')
    button20.place(relx=0.007, rely=0.1, height=54, width=92)
    button20.configure(background="#d9d9d9",text="Browse & Open", width=87, command=openImage)

    #Test an Image
    button21 = Button(root.tab2, text="Test The Image")
    button21.grid(column=0, row=0, sticky='W')
    button21.place(relx=0.133, rely=0.1, height=54, width=92)
    button21.configure(background="#FF0000",text="Test an Image", width=87, command= lambda: testTheImage())

    #Compare the test Image results with Training database
    button23 = Button(root.tab2, text="Compare the Test Image Results with Training Database")
    button23.grid(column=0, row=0, sticky='W')
    button23.place(relx=0.256, rely=0.1, height=54, width=322)
    button23.configure(background="#d9d9d9",text="Compare the Test Image Results with Training Database", width=87, command= lambda: compare())

    root.mainloop()

def openImage():

    global filePath
    filePath = tkinter.filedialog.askopenfilename()
    fp = open(filePath,'rb')
    global img_gray
    img_gray = Image.open(fp).convert("L",dither=Image.NONE)
    img_gray.show()

    text.insert(tk.END, "\n File Path:" + filePath)


def imageProcess():
    imageCounter =+1
    ONE = 150
    a = np.asarray(img_gray)                               # from PIL to np array (her pikselin numarası var 0-256)
    a_bin = threshold(a,100,ONE,0)                         # a'yı her biriyle karşılaştır 100 den büyükse 0 küçükse 1
    im = Image.fromarray(a_bin)                            # from np array to PIL format
    im.show()

    a_bin = binary_image(100,100, ONE)                     #creates a binary image
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

    mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
    imge = np.logical_or(mask_lines, mask_square1) * Value

    return imge

def np2PIL(im):
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    #print("size of arr: ",im.shape)
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

def blob_coloring_8_connected(bim, ONE):
    max_label = int(100000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
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
            if im[i][j] in list: #Pyhon does not accept empty if, so that I added this part
                #print(im[i][j])
                a=1
            else:
                list.append(im[i][j])
                counter = counter + 1
    list.remove(100000)  #To make list shorter

    width, height = 4, counter
    rectangle = [[0 for x in range(width)] for y in range(height)]
    my_k_values=[] #The list that we kept min and max values of labels (1, 2, 3)
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

        #i row, j col
        my_k_values.append([label,min_i,min_j,max_i,max_j])
    #print("my_k_values:")

    text.insert(tk.END, "\n KNN Values:")
    text.insert(tk.END, my_k_values)
    #print(my_k_values)

    source_img = Image.open(filePath).convert("RGBA")
    draw = ImageDraw.Draw(source_img)

    for b in my_k_values:
        save_unique_images(source_img, b[2], b[1], b[4], b[3])
        #numberCounterofAnImage += 1
        draw.rectangle(((b[2], b[1]),(b[4], b[3])), fill=None, outline='red', width=2)
        source_img.save("output.png", "PNG")

    source_img.show()
    return color_im

#Crop and resize the image
def save_unique_images(img, b2,b1,b4,b3):
    im1 = img.crop((b2, b1, b4, b3))
    im1 = im1.resize((21,21))
    im1.show()
    binaryImage(im1)

#Make an array of cropped images
#convert the array to binary array. if the rgb value is below a cer
def binaryImage(image):
    global a_bin
    img_bin = image.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    img_bin.show()
    #ONE = 150
    a = np.asarray(img_bin)  # from PIL to np array
    np.set_printoptions(threshold=100000)
    #print (a) #printing true false array
    a_bin = threshold(a, True, 1, 0)

# Hu moment start
def huMoment(f):
    #print("HU Moment :")
    #This function computes Hu's seven invariant moments
    global h_1, h_2, h_3, h_4, h_5, h_6, h_7
    u_00 = u_pq(f, 0, 0)
    # Scale invariance is obtained by normalization.
    # The normalized central moment is given below
    eta = lambda f, p, q: u_pq(f, p, q)/(u_00**((p+q+2)/2))
    # normalized central moments used to compute Hu's seven moments invariat
    e_20 = eta(f, 2, 0) #the seventh letter of the Greek alphabe
    e_02 = eta(f, 0, 2)
    e_11 = eta(f, 1, 1)
    e_12 = eta(f, 1, 2)
    e_21 = eta(f, 2, 1)
    e_30 = eta(f, 3, 0)
    e_03 = eta(f, 0, 3)

    # Formulation of Hu Moment
    h_1 = e_20 + e_02
    h_2 = 4*((e_11)*e_11) + (e_20-e_02)**2
    h_3 = (e_30 - 3*e_12)**2 + (3*e_21 - e_03)**2
    h_4 = (e_30 + e_12)**2 + (e_21 + e_03)**2
    h_5 = (e_30 - 3*e_12)*(e_30 + e_12)*((e_30+e_12)**2 - 3*(e_21+e_03)**2) + (3*e_21 - e_03)*(e_21 + e_03)*(3*(e_30 + e_12)**2 - (e_21 + e_03)**2)
    h_6 = (e_20 - e_02)*((e_30 + e_12)**2 - (e_21 + e_03)**2) + 4*e_11*(e_30 + e_12)*(e_21 + e_03)
    h_7 = (3*e_21 - e_03)*(e_30 + e_12)*((e_30 + e_12)**2 - 3*(e_21 + e_03)**2) - (3*e_12-e_30)*(e_21 + e_03)*(3*(e_30 + e_12)**2 - (e_21 + e_03)**2)

    text.insert(tk.END, "\n") #insert() function is inserts a given element at a given index in a list
    text.insert(tk.END, "\n Hu Moments: ")
    text.insert(tk.END, h_1)
    text.insert(tk.END, ", ")
    text.insert(tk.END, h_2)
    text.insert(tk.END, ", ")
    text.insert(tk.END, h_3)
    text.insert(tk.END, ", ")
    text.insert(tk.END, h_4)
    text.insert(tk.END, ", ")
    text.insert(tk.END, h_5)
    text.insert(tk.END, ", ")
    text.insert(tk.END, h_6)
    text.insert(tk.END, ", ")
    text.insert(tk.END, h_7)
    text.insert(tk.END, "\n")

    #print(trainingDataHU)

    #print(h_1, h_2, h_3, h_4, h_5, h_6, h_7)
    return [h_1, h_2, h_3, h_4, h_5, h_6, h_7]

def u_pq(f, p, q):
    # Centroid moment invariant to rotation. This function is equivalent to the m_pq but translating the centre of image f(x,y) to the centroid.

    u = 0
    c = normalizedMoment(f)
    for x in range(0, len(f)):
        for y in range(0, len(f[0])):
            u += ((x-c[0]+1)**p)*((y-c[1]+1)**q)*f[x][y]

    text.insert(tk.END, "\n The Central (p,q)th Moments : ")
    text.insert(tk.END, "\n")
    text.insert(tk.END, m_pq)
    return u

def normalizedMoment(f):
    #Normalized moment
    m_00 = m_pq(f,0,0)
    text.insert(tk.END, "\n Normalized Moment : ")
    text.insert(tk.END, "\n")
    text.insert(tk.END, m_pq(f, 1, 0) / m_00)
    text.insert(tk.END, "\n")
    text.insert(tk.END, m_pq(f, 0 ,1) / m_00)

    return [m_pq(f,1,0) / m_00, m_pq(f,0,1) / m_00]

def m_pq(f, p, q):
    #Two-dimensional (p+q)th order moment of image f(x,y) where p,q = 0, 1, 2, ...

    mpq = 0
    for k in range(0, len(f)):
        for l in range(0, len(f[0])):
            mpq += ( (k + 1)**p ) * ( (l + 1)**q ) * f[k][l]

    text.insert(tk.END, "\n 2D (p+q)th Order Moment of Image : ")
    text.insert(tk.END, "\n")
    text.insert(tk.END, mpq)
    return mpq
#Hu moment end

#R moment begin
def rMoment(h_1, h_2, h_3, h_4, h_5, h_6, h_7):
    #print("R Moment :")
    global r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10
    #  improve the scale invariability of Hu moments:
    r_1 = math.sqrt(h_2) / h_1
    r_2 = ( h_1 + math.sqrt(h_2) ) / ( h_1 - math.sqrt(h_2) )
    r_3 = ( math.sqrt(h_3) ) / ( math.sqrt(h_4) )
    r_4 = math.sqrt(h_3) / math.sqrt(abs(h_5))
    r_5 = math.sqrt(h_4) / math.sqrt(abs(h_5))
    r_6 = abs(h_6) / (h_1 * h_3)
    r_7 = abs(h_6) / (h_1 * math.sqrt(abs(h_5)))
    r_8 = abs(h_6) / (h_3 * math.sqrt(h_2))
    r_9 = abs(h_6) / (math.sqrt(h_2 * abs(h_5)))
    r_10 = abs(h_5) / (h_3 * h_4)

    text.insert(tk.END, "\n")
    text.insert(tk.END, "\n R Moments: ")
    text.insert(tk.END, r_1)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_2)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_3)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_4)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_5)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_6)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_7)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_8)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_9)
    text.insert(tk.END, ", ")
    text.insert(tk.END, r_10)
    text.insert(tk.END, "\n")
    text.insert(tk.END, "-----------------------------------")

    #print (r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10)
    return [r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10]
#R moment end

# Zernike moment start
# print("Zernike Moment :")
# Zernike moment end

#Save moments on a Text File
def SaveToTrainingDatabase():

    if testFlag != 1:
       text_file = open("TrainingDatabase.txt", "a") #"a" means "append mode"
       singleStr = str(h_1) + ";" + str(h_2) + ";" + str(h_3) + ";" + str(h_4) + ";" + str(h_5) + ";" + str(h_6) + ";" + str(h_7)+ ";" + str(r_1) + ";" + str(r_2) + ";" + str(r_3) + ";" + str(r_4) + ";" + str(r_5) + ";" + str(r_6) + ";" + str(r_7) + ";" + str(r_8) + ";" + str(r_9) + ";" + str(r_10)
       text_file.write(singleStr)
       text_file.close()

def testTheImage():
    testFlag = 1
    imageProcess()
    huMoment(a_bin)
    rMoment(h_1, h_2, h_3, h_4, h_5, h_6, h_7)

    test_text_file = open("TestResults.txt", "a")
    testSingleStr = str(h_1) + ";" + str(h_2) + ";" + str(h_3) + ";" + str(h_4) + ";" + str(h_5) + ";" + str(h_6) + ";" + str(h_7)+ ";" + str(r_1) + ";" + str(r_2) + ";" + str(r_3) + ";" + str(r_4) + ";" + str(r_5) + ";" + str(r_6) + ";" + str(r_7) + ";" + str(r_8) + ";" + str(r_9) + ";" + str(r_10)
    test_text_file.write(testSingleStr)
    test_text_file.close()

def compare():
    training = open("TrainingDatabase.txt", "r+")
    datatr = training.read()
    tr=re.split(";",datatr)

    print(tr[0]) # 1HU1
    print(tr[7]) # 1R1
    print(tr[17]) # 2HU1
    print(tr[24]) # 2R1
    print(tr[34]) # 3HU1
    print(tr[41]) # 3R1
    print(tr[51])  # 4HU1
    print(tr[58])  # 4R1
    print(tr[68])  # 5HU1
    print(tr[75])  # 5R1
    print(tr[85])  # 6HU1
    print(tr[92])  # 6R1
    print(tr[102])  # 7HU1
    print(tr[109])  # 7R1
    print(tr[119])  # 8R1
    print(tr[129])  # 9R1
    print(tr[139])  # 10R1

    test = open("TestResults.txt", "r+")
    datate = test.read()
    te=re.split(";",datate)
    print(te[0])

if __name__=='__main__':
    main()
