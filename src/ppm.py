import array

def create_image(size, data, file):
    width = size
    height = size
    PPMheader = 'P6\n' +str(width) + ' ' +str(height) + '\n255\n'
    # Create and fill a red PPM image
    image = array.array('B', data)
    # Save as PPM image
    with open(file, 'wb') as f:
        f.write(bytearray(PPMheader, 'ascii'))
        image.tofile(f)

def read_image(filename):
    with open(filename, 'rb') as f:
        # Read first line - expecting "P6"
        line = f.readline().decode('latin-1')
        if not line.startswith('P6'):
           print("ERROR: Expected PPM file to start with P6")
           return False

        # Read second line - expecting "width height"
        line = f.readline().decode('latin-1')
        dims = line.split()
        width,height=int(dims[0]),int(dims[1])

        # Read third line - expecting "255"
        line = f.readline().decode('latin-1')
        if not line.startswith('255'):
           print("ERROR: Expected 8-bit PPM with MAXVAL=255")
           return False

        image = f.read(width*height*3)
        return image