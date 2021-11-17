import math
def gradient(size_x, size_y, col1, col2, angle):
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    cols = []

    dr = (col2[0]-col1[0])
    dg = (col2[1]-col1[1])
    db = (col2[2]-col1[2])

    drx = 2*dr/size_x*sin
    dgx = 2*dg/size_x*sin
    dbx = 2*db/size_x*sin

    dry = 2*dr/size_y*cos
    dgy = 2*dg/size_y*cos
    dby = 2*db/size_y*cos
    for i in range(size_y):
        ty = i/(size_y-1)
        c = col1.copy()
        for o in range(size_x):
            tx = o/(size_x-1)
            c[0] += tx*drx
            c[1] += tx*dgx
            c[2] += tx*dbx
            cols.extend(c)
        
        col1[0] += ty*dry
        col1[1] += ty*dgy
        col1[2] += ty*dby
    for c in range(len(cols)):
        cols[c] = int(cols[c]/2)
    return cols

def normalize(gradient):
    for i in range(len(gradient)):
        gradient[i] = gradient[i]/255.0
    return gradient


        
