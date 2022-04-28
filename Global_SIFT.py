import matplotlib.pyplot as plt
from  scipy.ndimage.filters import convolve as corr
import skimage.io
import skimage.transform
from scipy.ndimage.filters import gaussian_filter
import numpy as np

def octave(img,sigma_for_every_octave):
    g1 = gaussian_filter(img, sigma_for_every_octave[0])
    g2 = gaussian_filter(img, sigma_for_every_octave[1])
    g3 = gaussian_filter(img, sigma_for_every_octave[2])
    g4 = gaussian_filter(img, sigma_for_every_octave[3])
    g5 = gaussian_filter(img, sigma_for_every_octave[4])
    dog1=g2-g1
    dog2=g3-g2
    dog3=g4-g3
    dog4=g5-g4
    oct=[dog1,dog2,dog3,dog4]
    return oct
sigma1=[0.707107, 1.000000 ,1.414214 ,2.000000 ,2.828427]
sigma2=[1.414214 ,2.000000 ,2.828427 ,4.000000 ,5.656854]
sigma3=[2.828427 ,4.000000 ,5.656854 ,8.000000 ,11.313708]
sigma4=[5.656854 ,8.000000 ,11.313708 ,16.000000 ,22.627417]
def neighbour(octave,l,x,y):
    img=octave[l]
    neighbour=[img[ x - 1, y],
               img[ x + 1, y],
               img[ x, y + 1],
               img[ x, y - 1],
               img[ x + 1, y + 1],
               img[ x + 1, y - 1],
               img[ x - 1, y + 1],
               img[ x - 1, y - 1]]
    if l != 0:
        prev=octave[l-1]
        neighbour+=[
        prev[ x, y],
        prev[ x + 1, y],
        prev[ x - 1, y],
        prev[ x, y + 1],
        prev[ x, y - 1],
        prev[ x + 1, y + 1],
        prev[ x + 1, y - 1],
        prev[ x - 1, y + 1],
        prev[ x - 1, y - 1]  ]
    if l !=3:
        next=octave[l+1]
        neighbour+=[
        next[ x, y],
        next[ x + 1, y],
        next[ x - 1, y],
        next[ x, y + 1],
        next[ x, y - 1],
        next[ x + 1, y + 1],
        next[ x + 1, y - 1],
        next[ x - 1, y + 1],
        next[ x - 1, y - 1]]
    return neighbour
def detect(oct1,oct_num):
    interestedpoint=[]
    for i in range (4):
        img = oct1[i]
        for x in range(20,img.shape[0]-20):
            for y in range(20,img.shape[1]-20):
                pixel=img[x,y]
                neigh=neighbour(oct1,i,x,y)
                Min=True
                Max=True
                for n in neigh:
                    if n>=pixel:
                        Max=False
                    if n<=pixel:
                        Min=False
                if Max or Min:
                    interestedpoint.append([oct_num,i,x,y])
    return interestedpoint
def gaussian(mm,nn,sigma):
    #p is power of k
    m=mm//2
    n=nn//2
    g = np.zeros((2*m+1, 2*n+1))
    for x in range(-m,m):
        for y in range(-n,n):
            g[x+m,y+n]=np.exp((-x**2-y**2)/(2*sigma**2))/(2*np.pi*sigma)
    return g[:mm,:nn]
img=skimage.io.imread('img.png')
img=skimage.color.rgb2gray(img)
# =============
pyramid=[img]
# =============
g=img[::2,::2]
pyramid.append(g)
# =============
g=g[::2,::2]
pyramid.append(g)
# ============
g=g[::2,::2]
pyramid.append(g)

# Building octave for every level

oct1=octave(pyramid[0],sigma1)
oct2 = octave(pyramid[1], sigma2)
oct3 = octave(pyramid[2], sigma3)
oct4 = octave(pyramid[3], sigma4)
points1=detect(oct1,0)
points2=detect(oct2,1)
points3=detect(oct3,2)
points4=detect(oct4,3)
interestedpoint=[]
interestedpoint+=points1
interestedpoint+=points2
interestedpoint+=points3
interestedpoint+=points4
#dervative x
#dervative y

ix=np.array([[-1],[0],[1]])
iy=ix.T
# =======================
OR=[]
GM=[]
# print(np.isnan(ix))
for img in pyramid:
    # =============
    IX=corr(img,ix)
    IY=corr(img,iy)
    # =============
    g=(IX**2+IY**2)**.5
    GM.append(g)
    # =============
    oriant=np.arctan(IY/IX)
    r=np.degrees(oriant)
    # =============
    nn=np.isnan(r)
    r[nn==True]=90
    # ============
    r[r<0]+=360
    OR.append(r)

# OR,GM
# intersetpoin=[  p1,p2,p3.......   ]
# p1=[oct_num,scale,x,y,...]
# sigma[]
# OR,GM


new_interestedpoint=[]
sigma=[sigma1,sigma2,sigma3,sigma4]
# g=gaussian(8,8,4.000000)
for p in interestedpoint:
    oct_num=p[0]
    scale=p[1]
    x=p[2]
    y=p[3]
    r=OR[oct_num]
    m=GM[oct_num]

    r_w=r[x-4:x+4,y-4:y+4]
    #.astype('uint')
    m_w=m[x-4:x+4,y-4:y+4]
    g=gaussian(8,8,1.5*sigma[oct_num][scale])
    gm=g*m_w

    r_w=r_w/10
    r_w=r_w.astype('int')

    hist=[0]*36
    for i in range(8):
        for j in range(8):
            hist[r_w[i,j]]+=gm[i,j]
    max_or=max(hist)
    # new=[oct_num,scale,x,y,hist.index(max_or)]
    # new_interestedpoint.append(new)

    for h in hist:
        if h>.8*max_or:
            new=[oct_num,scale,x,y,hist.index(h)]
            new_interestedpoint.append(new)
# print(new_interestedpoint)
# [p1,p2..]
# [oct_n,sca,x,y,angle]
final_inter=[]
for p in new_interestedpoint:
    oct_num=p[0]
    scale=p[1]
    x=p[2]
    y=p[3]
    angle=p[4]
    r=OR[oct_num]
    m=GM[oct_num]

    r_w=r[x-13:x+13,y-13:y+13]
    m_w=m[x-13:x+13,y-13:y+13]

#     print(angle)
    if angle!=0:
        r_w=skimage.transform.rotate(r_w,angle*10)
        m_w=skimage.transform.rotate(m_w,angle*10)
    r_w=r_w[5:21,5:21]
    m_w=m_w[5:21,5:21]
    desc=[]
    g=gaussian(16,16,.5*sigma[oct_num][scale])
    for i in range(0,16,4):
        for j in range(0,16,4):
            r_sub_w=r_w[i:i+4,j:j+4]
            m_sub_w=m_w[i:i+4,j:j+4]
            g_sub_w = g[i:i+4,j:j+4]
            mg_sub_w=m_sub_w*g_sub_w
    #       histogram for r_window and gm_sub window

            h=[0]*8
            r_sub_w=r_sub_w/45
            r_sub_w=r_sub_w.astype('int')
            for ii in range(4):
                for jj in range(4):
                    h[r_sub_w[ii,jj]]+=mg_sub_w[ii,jj]

            desc+=h
    # normalize desc
    # renoemaliz
    s=sum(desc)
    for d in range(128):
        desc[d]/=s
        if desc[d]>.2:
            desc[d]=0
#     print(desc)
    s=sum(desc)
    for d in range(128):
        desc[d]/=s

    p+=desc
    final_inter.append(p)
    print(p)

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(oct1[i],cmap='gray')
plt.show()
